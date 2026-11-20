"""Support-plane snapping: INTERFACE ONLY, not implemented. It needs a gravity / support-plane estimate from scene
pointmaps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["SnapResult", "SupportPlane", "snap_to_support"]


@dataclass(frozen=True)
class SupportPlane:
    normal_world: np.ndarray   # (3,) unit, pointing up (away from the support)
    offset_m: float            # plane n.x = offset in the conditioning world
    inlier_frac: float         # RANSAC inlier fraction on the support region
    n_points: int
    source: str                # "pointmap_ransac" | "imu" | "declared"


@dataclass(frozen=True)
class SnapResult:
    pose_world: np.ndarray     # (4,4) snapped pose
    rotation_deg: float        # applied rotation about the hull's centre of mass
    translation_m: float       # applied translation along the normal
    refused: str = ""          # "" or the reason (> max_angle_deg / > max_dist_m)


def snap_to_support(segment_pose: np.ndarray, hull_vertices: np.ndarray, plane: SupportPlane,
                    max_angle_deg: float = 15.0, max_dist_m: float = 0.02) -> SnapResult:
    """Nearest stable hull face onto `plane` with the minimal rotation about the hull's centre of mass, then a
    translation along the normal; refused beyond `max_angle_deg` / `max_dist_m`. Not implemented: raises
    NotImplementedError."""
    raise NotImplementedError("support-plane snapping needs the 10-22 pointmaps (design section 7)")
