# Copyright 2026 The HandReel authors. SPDX-License-Identifier: Apache-2.0
"""CLI: audit certified runs for the motionless-object null, the contact-removed null and the grip census.

    python -m fastdo.audit run_a.npz run_b.npz ... [--certificate CD|C289|cfg.json|module:factory]
                                                 [--json report.json] [--removal displace|pairs|xml]

Each run `.npz` holds:
- `reference` (n, 7) object rows on the executor grid, warm-up included; or `frames` (h, 7) with scalars `ref_dt`,
  `sim_dt`, `n_rows` (placed by `nullaudit.reference_on_executor_grid`);
- scalars `warmup_rows` (W) and `required_rows` (T);
- for the simulated checks: `scene` (the scene `.xml` or `.mjb`, relative to the `.npz`), `controls` (>= T, nu),
  `q0` and `weld_eq_ids` (required; an empty array for none), and `ctrl0`, `gravity`, `ctrl_steps`, `terminate_z`
  (each one missing is a default and is reported); optional `executor_model` (an `.mjb` of the model the run was
  executed on, when the loader edits the compiled scene: it is then the simulated model, and `--removal xml` copies
  its settings onto the pair-deleted XML), `sim_dt` (checked against the model's timestep), the executed rows `qpos`,
  `qvel`, `ctrl` (the positive control must match them), `hand_away` (column, offset) for `--removal pairs`, and
  `hand_geom` / `object_geom` (ngeom,) bool masks.
Give the reference in one encoding: `reference` together with a frame-grid field (`frames`, `ref_dt`, `n_rows`) is
refused. Without a scene only the motionless-object null runs; the contact-removed null is then counted as not run, and a
simulation field in such a file is refused because nothing would read it. One or two of `qpos`, `qvel`, `ctrl` are
refused: the stored trace is all three. A run is named by its file stem, and two files with one stem are refused.

Exit status: 0 when the report has no problems; 1 when the certificate rejects its own reference on an informative
clip, a run is invalid, a positive control fails or is certificate-only (no executed rows and no
`executor_model`; a matching `sim_dt` alone does not tie the replay to the executor), a default was used, a
non-default `--replay-tol` was given, `ctrl0` differs from the first control row (it has no effect), the grip census
stopped at a non-finite state or sampled no post-release row, or geoms outside the hand mask touch the object in the
certified replay
(`--allow-warnings` returns 0 and still prints them); 2 on an input the audit refuses (`InputError`: a missing or
ill-shaped npz field, an unknown npz field, an unreadable file or scene, a bad certificate config, a short reference
or control, a non-unit quaternion, a `hand_away` or `--hand-away` with a removal other than pairs, a simulation
field without a scene, a partial stored trace, two run files with one stem); 3 on any other
exception, an internal error (printed with its traceback). The JSON report is strict JSON: NaN is written as null,
and an infinite value as "inf" or "-inf".
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import re
import sys
import traceback
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from nullaudit import reference_on_executor_grid

from . import __version__
from .certificates import PRESETS, load_certificate, make_certificate, resolve_thresholds
from .errors import InputError
from .runner import QUAT_TOL, REPLAY_TOL, Executed, RunAudit, audit_run, report_dict
from .scene import compile_without_pairs, free_object_geom, geom_mask, load_model

EXEC_KEYS = ("q0", "ctrl0", "gravity")
NPZ_KEYS = frozenset({"reference", "frames", "ref_dt", "sim_dt", "n_rows", "warmup_rows", "required_rows", "scene",
                      "controls", *EXEC_KEYS, "weld_eq_ids", "ctrl_steps", "terminate_z", "executor_model", "qpos",
                      "qvel", "ctrl", "hand_away", "hand_geom", "object_geom"})
SIM_KEYS = ("controls", *EXEC_KEYS, "weld_eq_ids", "ctrl_steps", "terminate_z", "sim_dt", "qpos", "qvel", "ctrl",
            "hand_geom", "object_geom")          # read only when a scene or an executor model is named
TRACE_KEYS = ("qpos", "qvel", "ctrl")
HAND_DEFAULT = "collision_hand_"


def _load_npz(path: Path) -> dict[str, Any]:
    """The run file's arrays; a missing, unreadable or non-npz file, or one that needs pickles, is refused."""
    try:
        z = np.load(path, allow_pickle=False)
    except (OSError, EOFError, ValueError, zipfile.BadZipFile) as e:
        raise InputError(f"{path}: cannot read it as an .npz ({e})") from e
    if not isinstance(z, np.lib.npyio.NpzFile):
        raise InputError(f"{path}: not an .npz archive")
    with z:
        try:
            return {k: z[k] for k in z.files}
        except (OSError, ValueError, zipfile.BadZipFile) as e:          # a corrupt member, a pickled array
            raise InputError(f"{path}: cannot read its arrays without pickles ({e})") from e


def _get(z: dict[str, Any], key: str, path: Path) -> Any:
    if key not in z:
        raise InputError(f"{path}: missing `{key}`")
    return z[key]


def _scalar(z: dict[str, Any], key: str, path: Path, kind: type) -> Any:
    """npz field `key` as one int, float or str; refused if it is not a single value of that kind."""
    v = np.asarray(_get(z, key, path))
    if v.size != 1:
        raise InputError(f"{path}: `{key}` must be a single value, got shape {v.shape}")
    x = v.item()
    if kind is int and not (isinstance(x, (int, float)) and not isinstance(x, bool) and float(x).is_integer()):
        raise InputError(f"{path}: `{key}` must be an integer, got {x!r}")
    if kind is float and not (isinstance(x, (int, float)) and not isinstance(x, bool)):
        raise InputError(f"{path}: `{key}` must be a number, got {x!r}")
    if kind is str and not isinstance(x, str):
        raise InputError(f"{path}: `{key}` must be a string, got {x!r}")
    return kind(x)


def _floats(z: dict[str, Any], key: str, path: Path) -> np.ndarray:
    try:
        return np.asarray(z[key], np.float64)
    except (TypeError, ValueError) as e:
        raise InputError(f"{path}: `{key}` must be numeric ({e})") from e


def _ids(z: dict[str, Any], key: str, path: Path) -> tuple[int, ...]:
    v = np.atleast_1d(_floats(z, key, path))
    if v.ndim != 1 or not (np.isfinite(v) & (v == np.round(v))).all():
        raise InputError(f"{path}: `{key}` must be a 1-d array of integer ids, got {np.asarray(z[key]).tolist()!r}")
    return tuple(int(i) for i in v)


def _mask_field(z: dict[str, Any], key: str, model: Any, path: Path) -> np.ndarray:
    m = np.asarray(z[key])
    if m.dtype != bool or m.shape != (model.ngeom,):
        raise InputError(f"{path}: `{key}` must be a ({model.ngeom},) bool mask, got {m.dtype} {m.shape}")
    return m


def _regex_mask(model: Any, pattern: str, what: str) -> np.ndarray:
    try:
        return geom_mask(model, pattern)
    except re.error as e:
        raise InputError(f"--{what}: bad regular expression {pattern!r} ({e})") from e


def _reference(z: dict[str, Any], path: Path) -> np.ndarray:
    if "reference" in z:
        return _floats(z, "reference", path)
    frames = np.asarray(_get(z, "frames", path))
    if frames.ndim != 2 or frames.shape[0] == 0:
        raise InputError(f"{path}: `frames` must be (h, 7) rows, got shape {frames.shape}")
    # Checked BEFORE the grid is built: a zero, negative or non-finite step still yields some grid (the repeat count
    # is max(1, round(ref_dt / sim_dt))) or an internal error, and a negative row count slices instead of failing.
    # Without a scene the runner's own time-step check never sees `sim_dt`.
    ref_dt, sim_dt = (_scalar(z, k, path, float) for k in ("ref_dt", "sim_dt"))
    for k, v in (("ref_dt", ref_dt), ("sim_dt", sim_dt)):
        if not (math.isfinite(v) and v > 0):
            raise InputError(f"{path}: `{k}` must be a positive finite number of seconds, got {v!r}")
    warm, rows = _scalar(z, "warmup_rows", path, int), _scalar(z, "n_rows", path, int)
    if warm < 0 or rows < 1:
        raise InputError(f"{path}: `warmup_rows` must be >= 0 and `n_rows` >= 1, got {warm} and {rows}")
    return reference_on_executor_grid(frames, ref_dt, sim_dt, warm, rows).astype(np.float64)


def _scene_kw(path: Path, z: dict[str, Any], args: argparse.Namespace, info: dict[str, Any],
              defaulted: list[str]) -> dict[str, Any]:
    scene = (path.parent / _scalar(z, "scene", path, str)).resolve() if "scene" in z else None
    exe = (path.parent / _scalar(z, "executor_model", path, str)).resolve() if "executor_model" in z else None
    if exe is not None and exe.suffix != ".mjb":
        raise InputError(f"{path}: executor_model must be an .mjb file")
    src = exe if exe is not None else scene
    assert src is not None
    model = load_model(src)
    for key, opt, flag in (("hand_geom", args.hand_geom, "--hand-geom"), ("object_geom", args.object_geom,
                                                                       "--object-geom")):
        if key in z and opt:
            raise InputError(f"{path}: the npz `{key}` mask and {flag} {opt!r} both name the geoms; give one")
    if "hand_geom" in z:
        hand = _mask_field(z, "hand_geom", model, path)
    else:
        if args.hand_geom is None:
            defaulted.append(f"hand_geom (regex {HAND_DEFAULT!r})")
        hand = _regex_mask(model, args.hand_geom or HAND_DEFAULT, "hand-geom")
    obj = (_mask_field(z, "object_geom", model, path) if "object_geom" in z
           else _regex_mask(model, args.object_geom, "object-geom") if args.object_geom else free_object_geom(model))
    kw: dict[str, Any] = {"model": model, "hand_geom": hand, "object_geom": obj,
                          **{k: _floats(z, k, path) for k in EXEC_KEYS if k in z},
                          "weld_eq_ids": _ids(z, "weld_eq_ids", path) if "weld_eq_ids" in z else None,
                          "ctrl_steps": _scalar(z, "ctrl_steps", path, int) if "ctrl_steps" in z else None,
                          "terminate_z": _scalar(z, "terminate_z", path, float) if "terminate_z" in z else None,
                          "controls": z.get("controls"), "census_stride": args.stride,
                          "sim_dt": _scalar(z, "sim_dt", path, float) if "sim_dt" in z else None,
                          "replay_tol": args.replay_tol, "allow_replay_mismatch": args.allow_replay_mismatch,
                          "allow_uncertified_run": args.allow_uncertified_run, "model_is_executor": exe is not None}
    if all(k in z for k in TRACE_KEYS):
        kw["executed"] = Executed(z["qpos"], z["qvel"], z["ctrl"])
    removal = args.removal
    if removal == "xml":
        if scene is None or scene.suffix != ".xml":
            raise InputError(f"{path}: --removal xml needs an .xml scene")
        kw["removed_model"], rec = compile_without_pairs(scene, hand, obj, executor_model=model if exe else None)
        info.update(pairs_removed=rec["pairs_removed"], ka_h=rec["ka_h"]["pass"], after_removal=rec["after"],
                    matched_from_executor=rec["matched_from_executor"])
        removal = "model"
    kw["removal"] = removal
    if removal == "pairs":
        hand_away = z.get("hand_away", args.hand_away)
        if ("hand_away" in z and args.hand_away is not None
                and np.asarray(z["hand_away"]).ravel().tolist() != [float(x) for x in args.hand_away]):
            raise InputError(f"{path}: the npz `hand_away` {np.asarray(z['hand_away']).tolist()!r} conflicts "
                             f"with --hand-away {list(args.hand_away)!r}")
        if hand_away is None:
            raise InputError(f"{path}: --removal pairs needs --hand-away COLUMN OFFSET or an npz `hand_away`")
        kw["hand_away"] = hand_away                  # checked by audit_run: two numbers, a column in [0, nu)
    info.update(scene=None if scene is None else str(scene), executor_model=None if exe is None else str(exe),
                hand_geoms=int(hand.sum()), object_geoms=int(obj.sum()))
    return kw


def _spec_thresholds(spec: str) -> Any:
    """The thresholds of a preset or JSON spec, None for a module:factory certificate."""
    return resolve_thresholds(spec) if spec in PRESETS or spec.endswith(".json") else None


def _check_keys(z: dict[str, Any], path: Path, removal: str) -> None:
    """Refuse a field the audit would ignore: an unknown key (a misspelt `sim_dtt` would silently drop its check),
    a `hand_away` that the chosen removal does not use, a simulation field in a file that names neither a scene
    nor an executor model, or one or two of the three stored trace arrays."""
    unknown = sorted(set(z) - NPZ_KEYS)
    if unknown:
        raise InputError(f"{path}: unknown npz field(s) {unknown} (the module docstring lists the fields)")
    if "hand_away" in z and removal != "pairs":
        raise InputError(f"{path}: `hand_away` is used only by --removal pairs, not --removal {removal}")
    grid = sorted(k for k in ("frames", "ref_dt", "n_rows") if k in z)
    if "reference" in z and grid:
        # two encodings of one reference: `_reference` would use `reference` and never read the frame grid
        raise InputError(f"{path}: `reference` and the frame-grid fields {grid} are two encodings of the reference; "
                         "give one of them")
    if "scene" not in z and "executor_model" not in z:
        # `sim_dt` places a frame-grid reference (`frames`, `ref_dt`, `sim_dt`, `n_rows`) and is read in that mode
        read_here = {"sim_dt"} if "reference" not in z else set()
        unused = sorted(k for k in (*SIM_KEYS, "hand_away") if k in z and k not in read_here)
        if unused:
            raise InputError(f"{path}: {unused} are simulation fields, and the file names neither `scene` nor "
                             "`executor_model`; nothing would read them")
    trace = [k for k in TRACE_KEYS if k in z]
    if 0 < len(trace) < len(TRACE_KEYS):
        missing = [k for k in TRACE_KEYS if k not in z]
        raise InputError(f"{path}: the stored trace needs `qpos`, `qvel` and `ctrl` together; {missing} missing")


def audit_file(path: Path, args: argparse.Namespace) -> tuple[RunAudit, dict[str, Any]]:
    z = _load_npz(path)
    _check_keys(z, path, args.removal)
    ref = _reference(z, path)
    W, T = _scalar(z, "warmup_rows", path, int), _scalar(z, "required_rows", path, int)
    th = _spec_thresholds(args.certificate)
    if th is not None:
        cert = make_certificate(th, ref, warmup_rows=W, required_rows=T)
    else:
        cert = load_certificate(args.certificate)(ref, warmup_rows=W, required_rows=T)
    info: dict[str, Any] = {"run": str(path)}
    defaulted: list[str] = []
    kw = _scene_kw(path, z, args, info, defaulted) if ("scene" in z or "executor_model" in z) else {}
    run = audit_run(ref, cert, clip_id=path.stem, certificate_name=args.certificate, warmup_rows=W,
                    required_rows=T, quat_tol=None if args.allow_nonunit_quaternions else QUAT_TOL, **kw)
    if defaulted:
        run = dataclasses.replace(run, defaulted=tuple(defaulted) + run.defaulted)
    return run, info


def _json_safe(x: Any) -> Any:
    """The report with numpy scalars as Python numbers, NaN as null and +-inf as "inf" / "-inf" (strict JSON)."""
    if isinstance(x, dict):
        return {k: _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    if isinstance(x, np.ndarray):
        return _json_safe(x.tolist())
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and not np.isfinite(x):
        return None if np.isnan(x) else ("inf" if x > 0 else "-inf")
    return x


def main(argv: list[str] | None = None) -> int:
    """Run the CLI; returns the exit status (module docstring). An exception that is not an input error is printed
    with its traceback and gives exit status 3."""
    try:
        return _main(argv)
    except Exception:  # noqa: BLE001 (the CLI's last line: an internal error gets exit status 3)
        traceback.print_exc()
        print("error: internal error (the traceback above); exit status 3", file=sys.stderr)
        return 3


def _main(argv: list[str] | None) -> int:
    ap = argparse.ArgumentParser(prog="python -m fastdo.audit", description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", type=Path, help="one .npz per certified run (see the module docstring)")
    ap.add_argument("--certificate", default="CD",
                    help="preset (CD, C289), a JSON thresholds file, or module:factory (default CD)")
    ap.add_argument("--hand-geom", default=None,
                    help=f"regex on geom names (default {HAND_DEFAULT!r}, reported as a default)")
    ap.add_argument("--object-geom", default="", help="regex on geom names (default: the free joint's body)")
    ap.add_argument("--removal", default="displace", choices=("displace", "pairs", "xml"))
    ap.add_argument("--hand-away", nargs=2, type=float, default=None, metavar=("COLUMN", "OFFSET"),
                    help="with --removal pairs: control column and offset that move the hand away (required)")
    ap.add_argument("--stride", type=int, default=5, help="census row stride (default 5)")
    ap.add_argument("--replay-tol", type=float, default=REPLAY_TOL,
                    help="positive control: max |replay - executed qpos| (default %(default)g)")
    ap.add_argument("--allow-replay-mismatch", action="store_true",
                    help="positive control: waive an invalid replay, a departure from the executed rows and a "
                         "timestep mismatch (reported)")
    ap.add_argument("--allow-uncertified-run", action="store_true",
                    help="positive control: waive only the certificate's rejection of the replay, to apply the "
                         "nulls to uncertified runs (reported)")
    ap.add_argument("--allow-nonunit-quaternions", action="store_true",
                    help="audit references whose quaternions are off unit norm (counted and reported)")
    ap.add_argument("--allow-warnings", action="store_true", help="exit 0 even when the report lists problems")
    ap.add_argument("--json", type=Path, default=None, help="write the JSON report here")
    a = ap.parse_args(argv)
    if a.hand_away is not None:
        a.hand_away = tuple(a.hand_away)
    runs, infos = [], []
    try:
        if a.hand_away is not None and a.removal != "pairs":
            raise InputError(f"--hand-away is used only by --removal pairs, not --removal {a.removal}")
        if not (np.isfinite(a.replay_tol) and a.replay_tol >= 0):
            raise InputError(f"--replay-tol {a.replay_tol!r} must be finite and >= 0")
        for flag, rx in (("hand-geom", a.hand_geom), ("object-geom", a.object_geom)):
            try:
                re.compile(rx or "")
            except re.error as e:
                raise InputError(f"--{flag}: bad regular expression {rx!r} ({e})") from e
        th = _spec_thresholds(a.certificate)
        stems = [Path(p).stem for p in a.runs]
        repeated = sorted({x for x in stems if stems.count(x) > 1})
        if repeated:
            raise InputError(f"run files share a name: {', '.join(repeated)}; a run is identified by its file "
                             "name without the extension, so rename one of each pair")
        for p in a.runs:
            r, info = audit_file(p, a)
            runs.append(r)
            infos.append(info)
    except InputError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    doc = report_dict(runs, tool_version=__version__,
                      inputs={"certificate": a.certificate, "removal": a.removal,
                              "thresholds": None if th is None else dataclasses.asdict(th), "runs": infos})
    s = doc["summary"]
    print(s["headline"])
    g = s["grip_census"]
    if g["runs"]:
        print(f"Grip census: median ledger normal force / weight, per-run medians {g['min']:.3g}-{g['max']:.3g} "
              f"(median {g['median_of_run_medians']:.3g}) over {g['runs']} runs; {g['excluded']} runs excluded.")
    if a.json:
        a.json.write_text(json.dumps(_json_safe(doc), indent=1, allow_nan=False))
    for p in s["problems"]:
        print(f"PROBLEM: {p}", file=sys.stderr)
    return 1 if s["problems"] and not a.allow_warnings else 0


if __name__ == "__main__":
    sys.exit(main())
