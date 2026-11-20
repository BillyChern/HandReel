# Copyright 2026 The alongray authors. SPDX-License-Identifier: Apache-2.0
"""Per-method median along-ray share with a sequence-cluster bootstrap 95% interval (the share of squared centre error along the camera ray).

    python reproduce/alongray_shares.py MANIFEST.jsonl [--boot 10000] [--json out.json]
    python reproduce/alongray_shares.py --demo DIR          # synthetic windows with a known answer, then pooled

Needs alongray 0.4.1 (`uv pip install ./packages/alongray`). The submitted main paper reports such a share on
32 DexYCB development units (Sec. 5.2). The paper's values need DexYCB or HO3D-v3 ground truth and per-window outputs
that this release does not include; `reproduce/README.md`.

MANIFEST.jsonl holds one row per scored window and method:
    {"group": "H-AG|ours", "window": "...", "cluster": "<sequence>", "pred": "pred.npz", "gt": "gt.npz", "cell": 46}
`pred` and `gt` are alongray's input files (key `object`: (T, 3) camera-frame object centres in metres, or (T, 4, 4)
poses), holding the window's scored frames (relative paths are read from the manifest's directory). `cell` (optional,
per group) seeds the bootstrap as the registered analysis did; without it the seed is `--seed`.

Input contract (refused with exit status 2 otherwise):
- each (group, window) pair occurs once in the manifest; a duplicate would silently reweight the median and the
  bootstrap;
- frame correspondence: when either file holds `frame_idx` ((T,) integer frame ids), both must, and the two must be
  identical in the same order (no reordering, intersection or matching is done); without `frame_idx`, row t of
  `pred` is taken to be the same frame as row t of `gt`, so both must have the same number of rows, and their order
  is the caller's responsibility (this docstring is the input contract; no other document adds to it).

`--json` writes {"groups": {...}, "provenance": {...}}: per group the result and the window ids used and left out;
the provenance holds the sha256 of the manifest and of every input file, the numpy, scipy and alongray versions and
the bootstrap settings (B, seed, the cell of each group).

Per window: the frames finite in both files; fewer than alongray.MIN_ROWS (3) leave the window out. The window's share
is A = (e_par / e_3d)^2 = <a_t^2> / <|c_t - g_t|^2>, with a_t = (c_t - g_t) . g_t / |g_t| the along-ray component of
the error at the true ray: the mean squared along-ray error over the mean squared 3-D error (A = 1: all error along
the camera ray; an isotropic error gives a mean of 1/3). It is computed directly from these definitions, which are
alongray's e_par and e_3d (equal to alongray.score's to rounding; a test checks it), so a window on which alongray's
gauge fit would refuse (fitted lam <= 0) still gets its share. A window with zero 3-D error has no share and is left
out, as before. Per-window refusals, each with its reason, are reported and never enter the median:
- a true point at the camera centre (|g_t| = 0): the ray is undefined;
- a predicted point at the camera centre (|c_t| <= 1e-6 m, exactly (0, 0, 0) included): almost always zero-filled
  missing frames, which push the share towards 1 (a few such rows can turn an isotropic window "along-ray
  dominated"). Such a window is always refused, with the count of those rows; write missing frames as NaN, which
  are dropped;
- an input that alongray's reader refuses (shape, a pose whose bottom row is not 0 0 0 1).
Warnings raised while reading a window are recorded per window and also re-emitted, never silenced.

Per group: the median of A over the scored windows, and a percentile 95% interval from B bootstrap draws that each
resample the group's clusters with replacement (numpy Generator `default_rng([628, cell])`, one `integers(0, K, K)`
call per draw over the K sorted cluster labels, median of the drawn windows, `numpy.quantile` at 0.025 and 0.975).
With fewer than 2 distinct clusters the interval is NOT ESTIMABLE (every draw would be the whole group): the median
is kept, the interval is reported as null with the reason, and no reading is made. The reading "along-ray dominated"
needs median >= 0.5 and a lower bound > 1/3; "not determined" when the interval holds 1/3 or 0.5.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any

import alongray
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import strict_json as SJ

SEED = 628
REQUIRED = ("group", "window", "cluster", "pred", "gt")


class ManifestError(ValueError):
    """A manifest row or input pair that breaks the input contract (module docstring)."""
SHARE_ISO, SHARE_MIN = 1.0 / 3.0, 0.5


ORIGIN_TOL_M = 1e-6      # a predicted point this close to the camera centre is a zero-filled frame, not a pose


def window_share(pred: np.ndarray, gt: np.ndarray) -> dict[str, Any]:
    """One window's share and its accounting (module docstring): {"status": "scored" | "left_out" | "refused",
    "share", "reason", "n_rows", "n_nonfinite", "n_zero_pred", "n_used", "warnings"}."""
    out: dict[str, Any] = {"status": "scored", "share": None, "reason": None, "n_rows": int(np.shape(pred)[0]),
                           "n_nonfinite": 0, "n_zero_pred": 0, "n_used": 0, "warnings": []}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            p = alongray.reference_points(np.asarray(pred, float), kind="object")
            g = alongray.reference_points(np.asarray(gt, float), kind="object")
        except ValueError as e:
            out.update(status="refused", reason=f"input refused by alongray's reader: {e}")
            p = g = None
    out["warnings"] = sorted({str(w.message) for w in caught})
    for w in caught:                                   # recorded above and re-emitted: never silenced
        warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
    if p is None:
        return out
    ok = np.isfinite(p).all(1) & np.isfinite(g).all(1)
    out["n_nonfinite"] = int((~ok).sum())
    zero = ok & (np.linalg.norm(np.where(ok[:, None], p, 0.0), axis=1) <= ORIGIN_TOL_M)
    out["n_zero_pred"] = int(zero.sum())
    z = np.linalg.norm(g, axis=1)
    if np.any(ok & (z == 0.0)):
        out.update(status="refused", reason=f"true point at the camera centre on {int((ok & (z == 0.0)).sum())} "
                                            "row(s): the ray is undefined")
        return out
    if zero.any():
        out.update(status="refused", reason=f"{int(zero.sum())} of {out['n_rows']} predicted row(s) at the camera "
                                            f"centre (|c| <= {ORIGIN_TOL_M:g} m): zero-filled missing frames? write "
                                            "missing frames as NaN")
        return out
    out["n_used"] = int(ok.sum())
    if out["n_used"] < alongray.MIN_ROWS:
        out.update(status="left_out", reason=f"fewer than {alongray.MIN_ROWS} frames finite in both")
        return out
    d = p[ok] - g[ok]
    a = np.einsum("ij,ij->i", d, g[ok]) / z[ok]
    den = float(np.mean(np.einsum("ij,ij->i", d, d)))
    if den == 0.0:
        out.update(status="left_out", reason="zero 3-D error: the share is undefined")
        return out
    out["share"] = float(np.mean(a * a)) / den
    return out


MIN_CLUSTERS = 2


def median_ci(values: list[float], clusters: list[str], cell: int | None, n_boot: int, seed: int) -> tuple:
    """Median over windows and its cluster-bootstrap percentile 95% interval (module docstring); the interval is
    (nan, nan) when fewer than MIN_CLUSTERS distinct clusters exist (see interval_status)."""
    v, cl = np.asarray(values, np.float64), np.asarray(clusters)
    if v.size == 0:
        return math.nan, math.nan, math.nan
    keys = sorted(set(cl.tolist()))
    if len(keys) < MIN_CLUSTERS:
        return float(np.median(v)), math.nan, math.nan
    groups = [v[cl == k] for k in keys]
    rng = np.random.default_rng([SEED, int(cell)]) if cell is not None else np.random.default_rng(seed)
    meds = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        meds[b] = np.median(np.concatenate([groups[i] for i in pick]))
    return float(np.median(v)), float(np.quantile(meds, 0.025)), float(np.quantile(meds, 0.975))


def interval_status(clusters: list[str]) -> str:
    k = len(set(clusters))
    if k < MIN_CLUSTERS:
        return (f"NOT ESTIMABLE: {k} distinct cluster(s); a cluster-bootstrap interval needs at least {MIN_CLUSTERS}")
    return "estimated"


def reading(med: float, lo: float, hi: float) -> str:
    if not all(np.isfinite([med, lo, hi])):
        return "not determined (no interval)"
    if med >= SHARE_MIN and lo > SHARE_ISO:
        return "along-ray dominated"
    if any(lo <= x <= hi for x in (SHARE_ISO, SHARE_MIN)):
        return "not determined"
    return "not along-ray dominated"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """The `object` array and the optional `frame_idx` of one alongray input file."""
    with np.load(path, allow_pickle=False) as z:
        if "object" not in z.files:
            raise ManifestError(f"{path.name}: no 'object' array")
        fid = np.asarray(z["frame_idx"]) if "frame_idx" in z.files else None
        return np.asarray(z["object"]), fid


def paired_frames(pred: Path, gt: Path, where: str) -> tuple[np.ndarray, np.ndarray]:
    """The prediction and ground truth of one window after the frame-correspondence check (module docstring)."""
    (p, fp), (g, fg) = _load(pred), _load(gt)
    if (fp is None) != (fg is None):
        raise ManifestError(f"{where}: frame_idx is in {'pred' if fp is not None else 'gt'} only; give it in both "
                            "files or in neither")
    if fp is not None:
        if fp.ndim != 1 or fp.dtype.kind not in "iu" or fp.shape[0] != p.shape[0] or fg.shape[0] != g.shape[0]:
            raise ManifestError(f"{where}: frame_idx must be (T,) integers, one per row of 'object'")
        if fp.shape != fg.shape or not np.array_equal(fp, fg):
            raise ManifestError(f"{where}: pred and gt frame_idx differ (identical ids in the same order are "
                                "required)")
    elif p.shape[0] != g.shape[0]:
        raise ManifestError(f"{where}: pred has {p.shape[0]} rows and gt {g.shape[0]}; without frame_idx row t of "
                            "each must be the same frame")
    return p, g


def pool(manifest: Path, n_boot: int, seed: int) -> dict[str, Any]:
    """Pool the manifest's windows per group; {"groups": ..., "provenance": ...} (module docstring)."""
    rows = []
    for ln, x in enumerate(manifest.read_text().splitlines(), 1):
        if not x.strip():
            continue
        try:
            rows.append(SJ.loads(x, f"manifest line {ln}"))
        except ValueError as e:                        # DuplicateKeyError and malformed JSON
            raise ManifestError(str(e) if isinstance(e, SJ.DuplicateKeyError) else f"manifest line {ln}: {e}") from e
    base = manifest.parent
    seen: set[tuple[str, str]] = set()
    for i, r in enumerate(rows, 1):
        missing = [k for k in REQUIRED if k not in r]
        if missing:
            raise ManifestError(f"manifest row {i}: missing {', '.join(missing)}")
        key = (str(r["group"]), str(r["window"]))
        if key in seen:
            raise ManifestError(f"manifest row {i}: duplicate (group, window) = {key}")
        seen.add(key)
    groups: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    for r in rows:
        g = groups.setdefault(r["group"], {"values": [], "clusters": [], "used": [], "left_out": [], "refused": [],
                                           "flags": [], "cell": r.get("cell")})
        if r.get("cell") != g["cell"]:
            raise ManifestError(f"group {r['group']}: rows disagree on 'cell'")
        for k in ("pred", "gt"):
            hashes.setdefault(str(r[k]), _sha256(base / r[k]))
        p, gt = paired_frames(base / r["pred"], base / r["gt"], f"{r['group']} window {r['window']}")
        w = window_share(p, gt)
        wid = str(r["window"])
        if w["n_zero_pred"] or w["warnings"] or w["n_nonfinite"]:
            g["flags"].append({"window": wid, **{k: w[k] for k in ("n_rows", "n_nonfinite", "n_zero_pred",
                                                                    "warnings")}})
        if w["status"] == "refused":
            g["refused"].append({"window": wid, "reason": w["reason"]})
            continue
        if w["status"] == "left_out":
            g["left_out"].append({"window": wid, "reason": w["reason"]})
            continue
        g["values"].append(w["share"])
        g["clusters"].append(str(r["cluster"]))
        g["used"].append(str(r["window"]))
    out = {}
    for name, g in groups.items():
        med, lo, hi = median_ci(g["values"], g["clusters"], g["cell"], n_boot, seed)
        if g["values"]:
            median_status, status = "estimated", interval_status(g["clusters"])
        else:                                          # no share to take a median of: null, never NaN
            extra = f", {len(g['left_out'])} left out" if g["left_out"] else ""
            median_status = status = f"NOT ESTIMABLE: 0 scored windows ({len(g['refused'])} refused{extra})"
            med = None
        est = status == "estimated"
        out[name] = {"median_share": med, "median": median_status, "ci95": [lo, hi] if est else None,
                     "interval": status,
                     "reading": reading(med, lo, hi) if est else f"not determined ({status})",
                     "n_assigned": len(g["values"]) + len(g["left_out"]) + len(g["refused"]),
                     "n_windows": len(g["values"]), "n_clusters": len(set(g["clusters"])),
                     "left_out": len(g["left_out"]), "refused": len(g["refused"]), "cell": g["cell"], "boot": n_boot,
                     "windows_used": g["used"], "windows_left_out": g["left_out"], "windows_refused": g["refused"],
                     "window_flags": g["flags"]}
    import scipy

    prov = {"manifest": manifest.name, "manifest_sha256": _sha256(manifest), "input_sha256": hashes,
            "numpy": np.__version__, "scipy": scipy.__version__, "alongray": alongray.__version__,
            "origin_tolerance_m": ORIGIN_TOL_M, "min_clusters_for_an_interval": MIN_CLUSTERS,
            "bootstrap": {"B": n_boot, "seed": seed, "seed_rule": "default_rng([628, cell]) for a group with a "
                          "cell, else default_rng(seed)", "cells": {k: g["cell"] for k, g in groups.items()}}}
    return {"groups": out, "provenance": prov}


def demo(out: Path) -> Path:
    """Three synthetic groups with known shares: a pure depth (gauge) error, a pure lateral error, and a mix."""
    rng = np.random.default_rng(0)
    out.mkdir(parents=True, exist_ok=True)
    lines = []
    for w in range(6):
        gt = np.column_stack([rng.normal(0, 0.05, 20), rng.normal(0, 0.05, 20), rng.uniform(0.5, 0.8, 20)])
        u = gt / np.linalg.norm(gt, axis=1, keepdims=True)
        lat = np.cross(u, [0.0, 1.0, 0.0])
        lat /= np.linalg.norm(lat, axis=1, keepdims=True)                  # perpendicular to each true ray
        preds = {"along": 1.1 * gt, "lateral": gt + 0.03 * lat, "mixed": gt + 0.03 * u + 0.03 * lat}
        np.savez(out / f"gt_{w}.npz", object=gt)
        for name, p in preds.items():
            np.savez(out / f"{name}_{w}.npz", object=p)
            lines.append(json.dumps({"group": name, "window": f"w{w}", "cluster": f"seq{w // 2}",
                                     "pred": f"{name}_{w}.npz", "gt": f"gt_{w}.npz"}))
    (out / "manifest.jsonl").write_text("\n".join(lines) + "\n")
    return out / "manifest.jsonl"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("manifest", nargs="?", type=Path)
    ap.add_argument("--demo", type=Path, default=None, help="write synthetic windows into DIR and pool them")
    ap.add_argument("--boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0, help="bootstrap seed for groups without a 'cell'")
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args(argv)
    if (a.manifest is None) == (a.demo is None):
        ap.error("give MANIFEST.jsonl or --demo DIR")
    try:
        res = pool(demo(a.demo) if a.demo else a.manifest, a.boot, a.seed)
    except ManifestError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    for name, r in res["groups"].items():
        iv = f"[{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]" if r["ci95"] else "interval NOT ESTIMABLE"
        med = "NOT ESTIMABLE" if r["median_share"] is None else f"{r['median_share']:.3f}"
        print(f"{name}: median share {med} {iv} ({r['n_windows']} windows, {r['n_clusters']} "
              f"clusters; {r['reading']})" + (f"; {r['left_out']} left out, {r['refused']} refused of "
                                               f"{r['n_assigned']}" if r["left_out"] or r["refused"] else ""))
        for x in r["windows_refused"]:
            print(f"  refused {x['window']}: {x['reason']}")
    if a.json:
        a.json.write_text(json.dumps(res, indent=1, allow_nan=False) + "\n")    # strict JSON: a NaN raises
    return 0


if __name__ == "__main__":
    sys.exit(main())
