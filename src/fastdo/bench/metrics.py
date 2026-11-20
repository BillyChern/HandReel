"""Reconstruction (Chamfer, F-score) and retargeting (E_pos, E_rot, success) metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


def _nn_dist(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    tree = cKDTree(dst)
    d, _ = tree.query(src, k=1)
    return np.asarray(d, dtype=np.float64)


def chamfer_distance(pred: np.ndarray, gt: np.ndarray, squared: bool = True) -> float:
    """Symmetric Chamfer distance: mean_pred min d + mean_gt min d (squared by default)."""
    d_pg = _nn_dist(pred, gt)
    d_gp = _nn_dist(gt, pred)
    if squared:
        d_pg, d_gp = d_pg**2, d_gp**2
    return float(d_pg.mean() + d_gp.mean())


def f_score(pred: np.ndarray, gt: np.ndarray, threshold: float) -> float:
    """F-score at a distance threshold (precision/recall of points within threshold)."""
    precision = float((_nn_dist(pred, gt) <= threshold).mean())
    recall = float((_nn_dist(gt, pred) <= threshold).mean())
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


@dataclass(frozen=True)
class RetargetErrors:
    """Mean position error (m), mean geodesic rotation error (rad), and success flag."""

    e_pos: float
    e_rot: float
    success: bool


def _geodesic(r_a: np.ndarray, r_b: np.ndarray) -> np.ndarray:
    """Geodesic angle (rad) per frame, as `2 asin(|R_a - R_b|_F / 2 sqrt 2)` (exact for rotations).

    `arccos((tr(R_a R_b^T) - 1) / 2)` has a floor: float32 rounding moves the trace by ~1e-7, which arccos turns into
    ~5e-4 rad (0.028 deg) for a matrix against ITSELF. The chord form is accurate near zero.
    """
    chord = np.linalg.norm(np.asarray(r_a, np.float64) - np.asarray(r_b, np.float64), axis=(1, 2))
    return 2.0 * np.arcsin(np.clip(chord / (2.0 * np.sqrt(2.0)), 0.0, 1.0))


def retarget_errors(pred_pose: np.ndarray, ref_pose: np.ndarray, pos_thresh: float = 0.1,
                    rot_thresh: float = 0.5) -> RetargetErrors:
    """Errors between predicted and reference object trajectories, (T,4,4) each."""
    if pred_pose.shape != ref_pose.shape or pred_pose.ndim != 3:
        raise ValueError(f"pose shapes must match and be (T,4,4): {pred_pose.shape} vs {ref_pose.shape}")
    e_pos = float(np.linalg.norm(pred_pose[:, :3, 3] - ref_pose[:, :3, 3], axis=1).mean())
    e_rot = float(_geodesic(pred_pose[:, :3, :3], ref_pose[:, :3, :3]).mean())
    return RetargetErrors(e_pos=e_pos, e_rot=e_rot, success=bool(e_pos < pos_thresh and e_rot < rot_thresh))
