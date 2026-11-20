# Copyright 2026 The alongray authors
# SPDX-License-Identifier: Apache-2.0
"""``alongray PRED.npz GT.npz``: score a prediction file against a ground-truth file.

Both files hold camera-frame arrays in metres under the same keys:

- ``object`` (required): (T, 3) object centres or (T, 4, 4) object-to-camera poses;
- ``hand`` (optional, in both or neither): (T, 3) wrists or (T, J, 3) joints (root = ``--hand-root``);
- ``object_points`` (optional, in both or neither): (T, V, 3) per-frame object clouds, for the aligned Chamfer;
- ``K`` (optional, in either file): (3, 3) intrinsics, or pass ``--intrinsics fx,fy,cx,cy``;
- ``frame_ids`` (optional, in both or neither): (T,) integer frame ids, one per row; the rows are joined on them by
  the rules of :mod:`alongray.change` (identical id sets, no duplicates, increasing id order);
  ``--require-frame-ids`` refuses files without them;
- ``frame_idx`` (optional, in both or neither; not with ``frame_ids``): (T,) integer frame ids; the files are matched
  on the shared ids (the intersection; duplicates are refused);
- ``valid`` (optional, in either file): (T,) boolean frame mask; masked frames are dropped from every body.
- ``image_size`` (optional, in either file): ``(W, H)`` in pixels, used only by the camera-frame warnings.

Without ``frame_ids``, ``frame_idx`` and ``valid`` every body is scored on its own, as in 0.1, and row t of each
array is taken to be frame t in both files (synchronized row by row; nothing checks it).
:func:`check_camera_frame` refuses or warns on common world-frame and mixed-frame mistakes (heuristic, see its docstring). ``--scene`` adds a ``scene``
entry (per-body and shared gauges, the relative gauge; :func:`alongray.scene.scene_gauge`); it needs a hand.

Prints one JSON object ``{body: {e_par, gauge_part, off_orbit_part, lam, ...}}`` (metres; ``reproj_rms_px`` in
pixels; Chamfer in cm^2). Exit status: 0 on success, 2 on a usage error or on degenerate or malformed input
(wrong shapes, mixed conventions, a non-positive scale, an unreadable file).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from alongray import __version__
from alongray.scene import scene_gauge
from alongray.score import _frame_join, intrinsics_matrix, reference_points, score

__all__ = ["check_camera_frame", "main"]


def _load(path: Path) -> dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=False)
    if not isinstance(z, np.lib.npyio.NpzFile):
        raise ValueError(f"{path} is not an .npz archive of named arrays")  # noqa: TRY004 (input error, exit 2)
    with z:
        return {k: np.asarray(z[k]) for k in z.files}


def _intrinsics(pred: dict[str, np.ndarray], gt: dict[str, np.ndarray], flag: str | None) -> np.ndarray | None:
    if flag is not None:
        return np.array([float(v) for v in flag.split(",")])
    kp, kg = pred.get("K"), gt.get("K")
    if kp is not None and kg is not None and not np.array_equal(kp, kg):
        raise ValueError("pred and gt files hold different K; pass --intrinsics to choose")
    return kg if kg is not None else kp


def _jsonable(obj: object) -> object:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _frame_ids(d: dict[str, np.ndarray], name: str, n: int) -> np.ndarray:
    """The file's `frame_idx`: one integer id per row of its `object`, representable as int64. A shorter list would
    leave rows without an id and drop them silently; a longer one would address a row that does not exist."""
    f = np.asarray(d["frame_idx"])
    if f.ndim != 1 or not np.issubdtype(f.dtype, np.integer):
        raise ValueError(f"'frame_idx' in the {name} file must be a 1-D integer array, got {f.dtype} {f.shape}")
    if f.shape[0] != n:
        raise ValueError(f"'frame_idx' in the {name} file must give one id per row ({n}), got {f.shape[0]}")
    if f.dtype.kind == "u" and f.size and int(f.max()) > np.iinfo(np.int64).max:
        raise ValueError(f"'frame_idx' in the {name} file must fit in int64, got {int(f.max())}")
    return f.astype(np.int64)


def _match_frames(pred: dict[str, np.ndarray], gt: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray] | None:
    """Row indices into pred and gt: joined on ``frame_ids`` (identical sets) or matched on the shared ``frame_idx``
    values; None when neither file has either key."""
    if ("frame_ids" in pred) != ("frame_ids" in gt):
        raise ValueError("'frame_ids' must be in both files or in neither")
    if "frame_ids" in pred:
        if "frame_idx" in pred or "frame_idx" in gt:
            raise ValueError("pass frame_ids (identical id sets) or frame_idx (shared ids), not both")
        return _frame_join(pred["frame_ids"], gt["frame_ids"], len(pred["object"]), len(gt["object"]))
    if ("frame_idx" in pred) != ("frame_idx" in gt):
        raise ValueError("'frame_idx' must be in both files or in neither")
    if "frame_idx" not in pred:
        return None
    fp = _frame_ids(pred, "prediction", len(pred["object"]))
    fg = _frame_ids(gt, "ground-truth", len(gt["object"]))
    if len(np.unique(fp)) != len(fp) or len(np.unique(fg)) != len(fg):
        raise ValueError("frame_idx has duplicates")
    common = np.intersect1d(fp, fg)
    if common.size == 0:
        raise ValueError("pred and gt share no frame_idx")
    op, og = np.argsort(fp), np.argsort(fg)
    return op[np.searchsorted(fp, common, sorter=op)], og[np.searchsorted(fg, common, sorter=og)]


def _frame_mask(d: dict[str, np.ndarray], n: int, name: str) -> np.ndarray:
    if "valid" not in d:
        return np.ones(n, bool)
    v = np.asarray(d["valid"])
    if v.shape != (n,):
        raise ValueError(f"'valid' in the {name} file must be a ({n},) frame mask, got shape {v.shape}")
    if v.dtype != np.bool_ and not (np.issubdtype(v.dtype, np.integer) and np.isin(v, (0, 1)).all()):
        raise ValueError(f"'valid' in the {name} file must be boolean or 0/1 integers, got {v.dtype}")
    return v.astype(bool)


def _prepare(pred: dict[str, np.ndarray], gt: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray],
                                                                                dict[str, np.ndarray]]:
    """Frame matching and frame masks (masked frames become NaN), only when ``frame_ids``, ``frame_idx`` or ``valid``
    is present.

    Without either key every body is passed through as given and scored on its own, as in 0.1 (bodies may then have
    different frame counts).
    """
    if not any(k in d for k in ("frame_ids", "frame_idx", "valid") for d in (pred, gt)):
        return pred, gt
    keys: list[str] = [k for k in ("object", "hand", "object_points") if k in pred and k in gt]
    tp, tg = len(pred["object"]), len(gt["object"])
    for d, t, name in ((pred, tp, "prediction"), (gt, tg, "ground-truth")):
        bad = [k for k in keys if len(d[k]) != t]
        if bad:
            raise ValueError(f"with frame_ids, frame_idx or valid, every body needs the object's frame count: {bad} "
                             f"in the {name} file do not have {t} frames")
    vp, vg = _frame_mask(pred, tp, "prediction"), _frame_mask(gt, tg, "ground-truth")
    idx = _match_frames(pred, gt)
    if idx is None and tp != tg:
        raise ValueError(f"pred has {tp} frames and gt {tg}, and there are no frame_ids or frame_idx to match "
                         "them")
    ip, ig = (np.arange(tp), np.arange(tg)) if idx is None else idx
    p2 = {k: np.asarray(pred[k], np.float64)[ip] for k in keys}
    g2 = {k: np.asarray(gt[k], np.float64)[ig] for k in keys}
    keep = vp[ip] & vg[ig]
    for d in (p2, g2):
        for k in keys:
            d[k][~keep] = np.nan
    return {**pred, **p2}, {**gt, **g2}


def _body_refs(d: dict[str, np.ndarray], hand_root: int) -> dict[str, np.ndarray]:
    """Finite reference points per body in one file (object centre, hand root)."""
    out = {"object": reference_points(d["object"], kind="object")}
    if "hand" in d:
        out["hand"] = reference_points(d["hand"], hand_root, kind="hand")
    return {b: r[np.isfinite(r).all(axis=1)] for b, r in out.items()}


def _majority_sign(z: np.ndarray) -> float:
    return 1.0 if np.sum(z > 0) >= np.sum(z < 0) else -1.0


def check_camera_frame(pred: dict[str, np.ndarray], gt: dict[str, np.ndarray], k: np.ndarray | None,
                       hand_root: int = 0, image_size: tuple[float, float] | None = None) -> None:
    """Heuristic camera-frame checks (README "Command line"). They WARN by default and refuse only on unambiguous
    evidence, so that no genuine prediction is refused.

    Refused (``ValueError``):
      - ground-truth reference points (object centre and hand root together) of the minority sign of z beyond
        ``tol = max(0.05 m, 0.1 median|z|)``: a camera-frame truth keeps one sign (+ OpenCV, - OpenGL);
      - a MIRRORED prediction: at least 90% of its reference points on the opposite side of the camera from the
        truth's majority, beyond the same ``tol`` (the two files use different axis conventions).
    Warned (``UserWarning``):
      - ground-truth points of the minority sign within ``tol`` (an origin just behind the camera);
      - any prediction points on the opposite side that do not make it mirrored (a bad prediction is scored);
      - per body: with intrinsics, +z forward and an explicit ``image_size = (W, H)``, fewer than half of the
        ground-truth points projecting inside ``[-W/2, 1.5 W] x [-H/2, 1.5 H]``; otherwise a median
        ``max(|x|, |y|) / |z|`` above 3 (wider than any ordinary lens; independent of the principal point).

    A world frame that happens to look like a camera frame passes: these checks catch common mistakes, they cannot
    prove the frame.
    """
    gb, pb = _body_refs(gt, hand_root), _body_refs(pred, hand_root)
    g = np.concatenate(list(gb.values()))
    if not len(g):
        return
    z = g[:, 2]
    sign = _majority_sign(z)
    tol = max(0.05, 0.1 * float(np.median(np.abs(z))))
    minority = z[np.sign(z) == -sign]
    if minority.size and float(np.max(np.abs(minority))) > tol:
        raise ValueError(f"the ground truth's z changes sign by more than {tol:.3g} m: world-frame, mixed-frame, or "
                         "an object origin behind the camera")
    if minority.size:
        warnings.warn(f"{minority.size} ground-truth reference point(s) sit just behind the camera (|z| <= {tol:.3g} "
                      "m); scored as given", UserWarning, stacklevel=2)
    p = np.concatenate(list(pb.values()))
    if len(p):
        pz = p[:, 2]
        far_opposite = (np.sign(pz) == -sign) & (np.abs(pz) > tol)
        if far_opposite.mean() >= 0.9:
            raise ValueError(f"{far_opposite.mean():.0%} of the prediction's reference points lie on the opposite side "
                             f"of the camera from the ground truth's (beyond {tol:.3g} m): the prediction looks "
                             "mirrored, i.e. the two files use different camera-axis conventions")
        n_opp = int(np.sum(np.sign(pz) == -sign))
        if n_opp:
            warnings.warn(f"{n_opp} predicted reference point(s) lie on the other side of the camera from the ground "
                          "truth; scored as given", UserWarning, stacklevel=2)
    for body, gr in gb.items():
        front = gr[np.sign(gr[:, 2]) == sign]
        if not front.size:
            continue
        if k is not None and image_size is not None and sign > 0:
            kk = intrinsics_matrix(k)
            uv = (front @ kk.T)[:, :2] / front[:, 2:3]
            w, h = float(image_size[0]), float(image_size[1])
            inside = (uv[:, 0] >= -w / 2) & (uv[:, 0] <= 1.5 * w) & (uv[:, 1] >= -h / 2) & (uv[:, 1] <= 1.5 * h)
            if inside.mean() < 0.5:
                warnings.warn(f"only {inside.mean():.0%} of the ground-truth {body} points project near the "
                              f"{w:g} x {h:g} image: check the frame and the intrinsics", UserWarning, stacklevel=2)
        else:
            ratio = float(np.median(np.max(np.abs(front[:, :2]), axis=1) / np.abs(front[:, 2])))
            if ratio > 3.0:
                warnings.warn(f"ground-truth {body}: median max(|x|, |y|) / |z| = {ratio:.3g} > 3 is wider than any "
                              "ordinary lens: check that the inputs are in the camera frame", UserWarning,
                              stacklevel=2)


def _image_size(pred: dict[str, np.ndarray], gt: dict[str, np.ndarray]) -> tuple[float, float] | None:
    vals = [np.asarray(d["image_size"], np.float64).reshape(-1) for d in (gt, pred) if "image_size" in d]
    for v in vals:
        if v.shape != (2,) or not (np.isfinite(v).all() and (v > 0).all()):
            raise ValueError(f"'image_size' must be (W, H) in pixels, both > 0, got {v.tolist()}")
    if len(vals) == 2 and not np.array_equal(vals[0], vals[1]):
        raise ValueError("pred and gt files hold different image_size")
    return (float(vals[0][0]), float(vals[0][1])) if vals else None


def _pair(pred: dict[str, np.ndarray], gt: dict[str, np.ndarray],
          key: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    if (key in pred) != (key in gt):
        raise ValueError(f"'{key}' must be in both files or in neither")
    return pred.get(key), gt.get(key)


def run(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """Parse ``argv``, score, and return the result dict (what :func:`main` prints)."""
    ap = argparse.ArgumentParser(prog="alongray", description=__doc__.split("\n\n")[0])
    ap.add_argument("pred", type=Path, help="prediction .npz")
    ap.add_argument("gt", type=Path, help="ground-truth .npz")
    ap.add_argument("--intrinsics", default=None, help="fx,fy,cx,cy (overrides K in the files)")
    ap.add_argument("--hand-root", type=int, default=0, help="root joint index for (T,J,3) hands (default 0)")
    ap.add_argument("--out", type=Path, default=None, help="also write the JSON here")
    ap.add_argument("--scene", action="store_true", help="add the scene-level (relative) gauge when a hand is given")
    ap.add_argument("--require-frame-ids", action="store_true",
                    help="refuse files without a 'frame_ids' array (rows are then never assumed synchronized)")
    ap.add_argument("--version", action="version", version=f"alongray {__version__}")
    args = ap.parse_args(argv)
    pred, gt = _load(args.pred), _load(args.gt)
    if "object" not in pred or "object" not in gt:
        raise ValueError("both files need an 'object' array")
    if args.require_frame_ids and not ("frame_ids" in pred and "frame_ids" in gt):
        raise ValueError("--require-frame-ids: both files need a 'frame_ids' array (one integer id per row)")
    pred, gt = _prepare(pred, gt)
    ph, gh = _pair(pred, gt, "hand")
    pp, gp = _pair(pred, gt, "object_points")
    if args.scene and ph is None:
        raise ValueError("--scene needs a 'hand' array in both files (the relative gauge is between two bodies)")
    kk = _intrinsics(pred, gt, args.intrinsics)
    check_camera_frame(pred, gt, kk, int(args.hand_root), _image_size(pred, gt))
    res = score(pred["object"], gt["object"], ph, gh, kk,
                hand_root=args.hand_root, pred_object_points=pp, gt_object_points=gp)
    raw: dict[str, Any] = res.as_dict()
    if args.scene and ph is not None and gh is not None:
        raw["scene"] = scene_gauge({"object": (pred["object"], gt["object"]), "hand": (ph, gh)},
                                   hand_root=int(args.hand_root)).as_dict()
    out: dict[str, Any] = _jsonable(raw)  # type: ignore[assignment]
    if args.out is not None:
        args.out.write_text(json.dumps(out, indent=2) + "\n")
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """Console entry point."""
    try:
        out = run(argv)
    except (ValueError, KeyError, OSError, zipfile.BadZipFile) as e:
        print(f"alongray: error: {e}", file=sys.stderr)
        return 2
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
