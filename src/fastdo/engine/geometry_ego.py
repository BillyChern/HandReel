"""The geometry stage's opt-in camera ego-motion (`geometry_stage.ego_motion`, default off).

The model is `fastdo.geometry.ego_motion`; the correspondences are `fastdo.geometry.ego_tracks`. This module is the
stage glue: `plan_for_clip` builds a clip's tracker plan from its decode rows, `mask.npz` (when present) and BOTH hands
of `hands.npz`, BEFORE MoGe-2 runs, so the gathered geometry path can copy just the planned pointmap samples off
the device; `finish_clip` solves and writes `ego_motion.npz` from those samples after the clip's metric scale is applied.

`geometry.npz` is never touched: off or on, its arrays and bytes are what the stage wrote without ego-motion. The
ego-motion reads the pointmaps AFTER the same float32 rescale the stored arrays get, so the dense and the gathered path
hand `finish_clip` identical samples.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from fastdo.engine.worker_support import load_npz, rows_by_frame_index
from fastdo.engine.workers_hands import HANDS_FILE, SIDES
from fastdo.geometry.ego_motion import EGO_MOTION_FILE, EgoMotionConfig, save_ego_motion
from fastdo.geometry.ego_tracks import EgoPlan, exclusion_masks, plan_correspondences, solve_from_samples

__all__ = ["finish_clip", "plan_for_clip"]

logger = logging.getLogger(__name__)
MASK_FILE = "mask.npz"


def plan_for_clip(artifact_dir: Path, frames: np.ndarray, frame_idx: np.ndarray, K: np.ndarray,
                  cfg: EgoMotionConfig, masks: np.ndarray | None = None) -> tuple[EgoPlan, float]:
    """`(plan, seconds)`: the clip's background correspondences, excluding its object mask and both hands.

    `masks` is the clip's (N,H,W) object masks when the caller already read `mask.npz` (pinned to the decode rows);
    otherwise `mask.npz` is read here when it exists.

    Raises:
        ValueError: If `mask.npz` or `hands.npz` rows are not the decode rows.
    """
    t0 = time.perf_counter()
    artifact_dir = Path(artifact_dir)
    n, height, width = int(frames.shape[0]), int(frames.shape[1]), int(frames.shape[2])
    mask_path = artifact_dir / MASK_FILE
    if masks is None and mask_path.is_file():
        mask = load_npz(mask_path)
        if not np.array_equal(np.asarray(mask["frame_idx"]), np.asarray(frame_idx)):
            raise ValueError(f"{mask_path}: rows are not the decode rows; refusing to exclude another clip's object")
        masks = np.asarray(mask["masks"], dtype=bool)
    joints = valid = None
    hands_path = artifact_dir / HANDS_FILE
    if hands_path.is_file():
        with np.load(hands_path, allow_pickle=False) as hands:
            rows = rows_by_frame_index(hands["frame_idx"], frame_idx, str(hands_path))
            joints = {s: np.asarray(hands[f"{s}_joints"], dtype=np.float64)[rows] for s in SIDES}
            valid = {s: np.asarray(hands[f"{s}_valid"], dtype=bool)[rows] for s in SIDES}
    exclude = exclusion_masks(n, height, width, cfg, masks, joints, valid, np.asarray(K, dtype=np.float64))
    plan = plan_correspondences(frames, exclude, cfg)
    return plan, time.perf_counter() - t0


def finish_clip(artifact_dir: Path, plan: EgoPlan, samples: np.ndarray, frame_idx: np.ndarray,
                cfg: EgoMotionConfig, plan_seconds: float) -> None:
    """Solves the ego-motion from the plan's metric samples and writes `ego_motion.npz`."""
    t0 = time.perf_counter()
    ego = solve_from_samples(plan, samples, cfg)
    seconds = time.perf_counter() - t0
    save_ego_motion(Path(artifact_dir) / EGO_MOTION_FILE, ego, frame_idx, cfg,
                    extra={"plan_seconds": round(plan_seconds, 4), "solve_seconds": round(seconds, 4),
                           "n_samples": len(plan.frame), "median_corners": float(np.median(plan.n_corners))})
    s = ego.summary
    logger.info("ego-motion for %s: ok %s, rows %.2f, loop p90 %.2f deg, inliers %.2f, path %.3f m, rot %.1f deg "
                "(plan %.2f s, solve %.2f s)", artifact_dir, s["ok"], s["row_frac"], s["loop_rot_deg_p90"],
                s["median_inlier_frac"], s["cam_path_m"], s["cam_rot_extent_deg"], plan_seconds, seconds)
