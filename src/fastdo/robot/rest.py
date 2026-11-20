"""Resting poses of the reconstructed object on the robot's table (`fastdo.robot.reach` uses them).

**Support** (`support_inside`): at a pose, the support points are the convex-hull vertices within `tol` of the lowest
one (MuJoCo collides the object as its convex hull); the object rests when its centre of mass (the hull's volume
centroid) projects inside their convex polygon. The signed horizontal distance to the polygon edge is returned
(> 0 inside); with fewer than three non-collinear support points the polygon is that point or segment.

**Stable "up" candidates** (`stable_ups`), for reconstructions whose gravity was never estimated: the object rests
on the table at its start pose, so one physical estimate of "up" is the outward normal of the hull face it rests on.
`trimesh.poses.compute_stable_poses` lists the hull's stable faces with the probability of landing on each when
dropped (solid angle seen from the centre of mass); each face gives a world up (the face's inward direction at the
object's anchor pose). A candidate is kept when its probability is at least `min_probability` and it lies within
`max_prior_deg` of a level-to-looking-down camera (`placement.pitched_up` at a pitch in `pitch_range_deg`): the
wearer looks at the table, so up cannot point anywhere. Every candidate is an ASSUMPTION about gravity; the package
names it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh
from scipy.spatial import ConvexHull, QhullError

from fastdo.robot.placement import pitched_up

__all__ = ["StableUp", "stable_ups", "support_inside"]


@dataclass(frozen=True)
class StableUp:
    up_world: np.ndarray          # (3,) video-world up if the object rests on this face at the anchor pose
    up_object: np.ndarray         # (3,) the same direction in the metric object frame
    probability: float            # trimesh's landing probability of the face
    prior_angle_deg: float        # angle to the nearest level-to-looking-down camera up

    def to_dict(self) -> dict:
        return {"up_object": np.round(self.up_object, 4).tolist(), "probability": round(self.probability, 4),
                "prior_angle_deg": round(self.prior_angle_deg, 1)}


def support_inside(hull_vertices: np.ndarray, hull_com: np.ndarray, pose: np.ndarray, tol: float) -> float:
    """Signed horizontal distance (m, > 0 inside) of the centre of mass from the support polygon at `pose` (4,4)
    (module docstring)."""
    rot, t = pose[:3, :3], pose[:3, 3]
    pts = np.asarray(hull_vertices, dtype=np.float64) @ rot.T + t
    com = (rot @ np.asarray(hull_com, dtype=np.float64) + t)[:2]
    sup = pts[pts[:, 2] <= pts[:, 2].min() + tol, :2]
    try:
        hull = ConvexHull(sup)
    except (QhullError, ValueError):
        if len(sup) == 1 or np.ptp(sup, axis=0).max() < 1e-9:
            return -float(np.linalg.norm(com - sup[0]))
        d = np.linalg.svd(sup - sup.mean(axis=0))[2][0]
        a, b = sup[np.argmin(sup @ d)], sup[np.argmax(sup @ d)]
        u = np.clip((com - a) @ (b - a) / max(float((b - a) @ (b - a)), 1e-18), 0.0, 1.0)
        return -float(np.linalg.norm(com - (a + u * (b - a))))
    return float(-np.max(hull.equations[:, :2] @ com + hull.equations[:, 2]))


def stable_ups(hull: trimesh.Trimesh, object_rot_world: np.ndarray, cam_to_world0: np.ndarray,
               min_probability: float = 0.02, max_prior_deg: float = 30.0,
               pitch_range_deg: tuple[float, float] = (0.0, 75.0)) -> tuple[StableUp, ...]:
    """The stable-face up candidates of the module docstring, most probable first (ties: nearest the prior)."""
    transforms, probs = trimesh.poses.compute_stable_poses(hull)
    priors = np.stack([pitched_up(cam_to_world0, p) for p in np.arange(pitch_range_deg[0], pitch_range_deg[1] + 1e-9,
                                                                       1.0)])
    priors /= np.linalg.norm(priors, axis=1, keepdims=True)
    out = []
    for tf, p in zip(transforms, probs, strict=True):
        if p < min_probability:
            continue
        up_obj = np.asarray(tf[:3, :3]).T @ np.array([0.0, 0.0, 1.0])
        up_w = np.asarray(object_rot_world, dtype=np.float64) @ up_obj
        ang = float(np.degrees(np.arccos(np.clip(priors @ up_w, -1.0, 1.0)).min()))
        if ang <= max_prior_deg:
            out.append(StableUp(up_world=up_w, up_object=up_obj, probability=float(p), prior_angle_deg=ang))
    return tuple(sorted(out, key=lambda s: (-round(s.probability, 6), s.prior_angle_deg)))
