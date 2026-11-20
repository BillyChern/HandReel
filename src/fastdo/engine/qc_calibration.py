"""QC calibration data from engine artifacts: per-clip features, pose-error labels and the rule fit.

Protocol: DexYCB calibration clips run through the engine with their GT YCB mesh as the
shape (`bench.dexycb_shape: template`) and the deployed batched HaWoR hands. A clip's features
are exactly the `qc` stage's (`read_qc_inputs` + `compute_qc_features`). Its label is `pose_error_m` (the mean
per-frame translation error, m) between the solved poses expressed in the template's own frame (`template_frame_poses`)
and DexYCB's GT object poses at the solve rows, over the rows whose GT pose is valid.

`fit_rule_from_clips` fits `fit_acceptance_rule` and refuses a threshold without its finite-sample coverage guarantee
(`conformal_rank(n, alpha) > n` for the `n` good calibration clips) unless `allow_unguaranteed`.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
from torch import Tensor

from fastdo.engine.workers_qc import SOLVE_FILE, FeaturesFn, compute_qc_features, read_qc_inputs
from fastdo.engine.workers_shape import TEMPLATE_BACKEND, read_shape_meta
from fastdo.qc import (
    AcceptanceRule,
    ConformalCalibration,
    QcFeatures,
    conformal_rank,
    fit_acceptance_rule,
    pose_error_m,
)
from fastdo.qc.features import qc_features

__all__ = [
    "CalibrationClip",
    "GtPosesFn",
    "calibration_clip",
    "collect_calibration_clips",
    "coverage_guaranteed",
    "fit_rule_from_clips",
    "template_frame_poses",
    "write_calibration_features",
]

logger = logging.getLogger(__name__)

GtPosesFn = Callable[[str], tuple[np.ndarray, np.ndarray]]   # sequence -> (object_pose (N,4,4), valid (N,) bool)


@dataclass(frozen=True)
class CalibrationClip:
    """One labelled calibration clip."""

    clip_id: str
    sequence: str
    features: QcFeatures
    pose_error_m: float
    n_frames: int            # solve rows
    n_gt_frames: int         # solve rows with a valid GT pose (the label's support)
    metric_scale: float      # the solved scale in metres per template unit (1.0 for a perfect metric solve)


def template_frame_poses(poses: np.ndarray | Tensor, scale: float,
                         shape_meta: Mapping[str, Any]) -> tuple[np.ndarray, float]:
    """Canonical-frame poses of a template shape, re-expressed in the template's own (raw, metric) frame.

    The shape stage maps the template as `canonical = (raw - c) * ns`, so `s R canonical + t = (s ns) R raw +
    (t - s ns R c)`: the rotation is unchanged, the translation becomes `t - s * ns * R @ c`, the metric scale `s * ns`.

    Returns:
        `(poses (T,4,4) float64, metric_scale)`.

    Raises:
        ValueError: Unless `shape_meta["backend"]` is `"template"`: a generated mesh's frame is arbitrary, so a pose
            error in it is meaningless.
    """
    backend = shape_meta.get("backend")
    if backend != TEMPLATE_BACKEND:
        raise ValueError(f"template_frame_poses needs a {TEMPLATE_BACKEND!r} shape, got backend {backend!r}: a pose "
                         "error in a generated mesh's arbitrary frame is meaningless")
    ns = float(shape_meta["normalization_scale"])
    center = np.asarray(shape_meta["normalization_center"], dtype=np.float64).reshape(3)
    array = poses.detach().cpu().double().numpy() if isinstance(poses, Tensor) else np.asarray(poses, np.float64)
    out = np.array(array, dtype=np.float64, copy=True)
    metric_scale = float(scale) * ns
    out[:, :3, 3] = out[:, :3, 3] - metric_scale * (out[:, :3, :3] @ center)
    return out, metric_scale


def calibration_clip(clip_id: str, sequence: str, artifact_dir: Path, gt_poses: tuple[np.ndarray, np.ndarray],
                     features_fn: FeaturesFn = qc_features, laplace_std_threshold: float = 0.01) -> CalibrationClip:
    """The features and `pose_error_m` label of one clip's engine artifacts.

    Args:
        gt_poses: `(object_pose (N_source,4,4), valid (N_source,) bool)`, indexed by source frame.

    Raises:
        ValueError: If no solve row has a valid GT pose, the GT does not cover the solve rows, or the shape is not a
            template (and on any `read_qc_inputs` row check).
        FileNotFoundError: If an artifact is missing.
    """
    artifact_dir = Path(artifact_dir)
    inputs = read_qc_inputs(artifact_dir)
    features = compute_qc_features(inputs, features_fn, laplace_std_threshold)
    solution = inputs.solve.solution
    poses, metric_scale = template_frame_poses(solution.poses, solution.scale, read_shape_meta(artifact_dir))
    object_pose, valid = np.asarray(gt_poses[0], dtype=np.float64), np.asarray(gt_poses[1], dtype=bool).reshape(-1)
    rows = inputs.frame_idx
    if object_pose.shape[0] != valid.shape[0] or rows.min() < 0 or rows.max() >= valid.shape[0]:
        raise ValueError(f"{clip_id} ({sequence}): the GT covers {valid.shape[0]} source frames ({object_pose.shape[0]} "
                         f"poses), but the solve rows span frames {int(rows.min())}..{int(rows.max())}")
    ok = valid[rows]
    if not ok.any():
        raise ValueError(f"{clip_id} ({sequence}): no solve row has a valid GT pose (confidence > 0)")
    error = pose_error_m(poses[ok], object_pose[rows][ok])
    return CalibrationClip(clip_id=clip_id, sequence=sequence, features=features, pose_error_m=float(error),
                           n_frames=int(rows.size), n_gt_frames=int(ok.sum()), metric_scale=metric_scale)


def collect_calibration_clips(artifact_root: Path, clips: Sequence[tuple[str, str]], gt_poses_fn: GtPosesFn,
                              features_fn: FeaturesFn = qc_features, laplace_std_threshold: float = 0.01
                              ) -> tuple[list[CalibrationClip], list[dict[str, str]]]:
    """`calibration_clip` of every `(clip_id, sequence)` under `artifact_root`, and the clips skipped.

    A clip without `solve.npz` (its solve stage did not finish, so it never reaches qc), or whose inputs or GT raise
    `FileNotFoundError` or `ValueError`, is returned in `skipped` as `{clip_id, sequence, reason}` and logged at
    WARNING; none is dropped silently.
    """
    labelled: list[CalibrationClip] = []
    skipped: list[dict[str, str]] = []
    for clip_id, sequence in clips:
        artifact_dir = Path(artifact_root) / clip_id
        if not (artifact_dir / SOLVE_FILE).is_file():
            reason = f"no {SOLVE_FILE}: the clip's solve stage did not finish, so it never reaches qc"
        else:
            try:
                labelled.append(calibration_clip(clip_id, sequence, artifact_dir, gt_poses_fn(sequence), features_fn,
                                                 laplace_std_threshold))
                continue
            except (FileNotFoundError, ValueError) as exc:
                reason = f"{type(exc).__name__}: {exc}"
        skipped.append({"clip_id": clip_id, "sequence": sequence, "reason": reason})
        logger.warning("calibration clip %s (%s) skipped: %s", clip_id, sequence, reason)
    return labelled, skipped


def coverage_guaranteed(calibration: ConformalCalibration) -> bool:
    """Whether split conformal's finite-sample guarantee holds for the rule: `conformal_rank(n, alpha) <= n`."""
    n = int(calibration.n_calibration)
    return n >= 1 and conformal_rank(n, calibration.alpha) <= n


def _min_calibration(alpha: float) -> int:
    a = Fraction(repr(float(alpha)))
    return math.ceil((1 - a) / a)


def fit_rule_from_clips(clips: Sequence[CalibrationClip], alpha: float = 0.1, tau: float = 0.02,
                        fit_fraction: float = 0.5, seed: int = 0, allow_unguaranteed: bool = False) -> AcceptanceRule:
    """`fit_acceptance_rule` on the clips' features and pose errors, with the coverage check.

    Raises:
        ValueError: If `fit_acceptance_rule` refuses the clips (below 2 good or 1 bad; the message gives both counts),
            or, unless `allow_unguaranteed` (then a WARNING), when the good calibration clips are too few for the
            coverage guarantee at `alpha` (the message names "coverage", their count and the minimum).
    """
    errors = np.array([clip.pose_error_m for clip in clips], dtype=np.float64)
    n_good = int((errors < tau).sum())
    try:
        rule = fit_acceptance_rule([clip.features for clip in clips], errors, alpha=alpha, tau=tau,
                                   fit_fraction=fit_fraction, seed=seed)
    except ValueError as exc:
        raise ValueError(f"cannot fit the acceptance rule on {len(clips)} calibration clips ({n_good} good, "
                         f"{len(clips) - n_good} bad at tau={tau} m): {exc}") from exc
    if not coverage_guaranteed(rule.calibration):
        message = (f"the conformal coverage guarantee is unattainable: {rule.calibration.n_calibration} good calibration "
                   f"clips at alpha={alpha} (of {n_good} good clips; fit_fraction {fit_fraction} keeps the others for "
                   f"the fit), and split conformal needs at least {_min_calibration(alpha)}")
        if not allow_unguaranteed:
            raise ValueError(f"{message}. Add calibration sequences from the allowed pool; alpha and tau are not "
                             "loosened.")
        logger.warning("%s; allow_unguaranteed: the rule is kept without the guarantee", message)
    return rule


def write_calibration_features(path: Path, clips: Sequence[CalibrationClip], skipped: Sequence[Mapping[str, str]],
                               tau: float) -> None:
    """Writes every clip's features and label (`good` = `pose_error_m < tau`) and the skipped clips as standard JSON
    (tmp sibling, then `os.replace`)."""
    rows = [{"clip_id": c.clip_id, "sequence": c.sequence, "features": dataclasses.asdict(c.features),
             "pose_error_m": c.pose_error_m, "good": bool(c.pose_error_m < tau), "n_frames": c.n_frames,
             "n_gt_frames": c.n_gt_frames, "metric_scale": c.metric_scale} for c in clips]
    payload = {"tau": float(tau), "n_clips": len(rows), "n_good": sum(bool(c.pose_error_m < tau) for c in clips), "clips": rows,
               "skipped": [dict(entry) for entry in skipped]}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
