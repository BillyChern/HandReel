"""Kinematic-retargeting acceptance statistics: clearance and penetration measured on the hand's collision primitives.

The same function the penetration term optimizes (`fastdo.retarget.primitives.primitive_distances`:
capsule-cover minimum signed distance to the object SDF grid, in metres), summarized per trajectory:
the fraction of frames where any primitive penetrates (`d_min < 0`), the fraction where every
primitive clears 0.5 mm, and per-finger median clearance on contact frames (a finger's clearance in a
frame is the minimum over its covers). Contact frames follow the C3/C4 convention of
`fastdo.retarget.isolation.kinematic_reference_stats`: frames where the robot contact measure on
`n_per_link` sampled surface points per link carries any band weight.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import Tensor

from fastdo.retarget.contact import robot_contact_measure
from fastdo.retarget.labels import INDEX, MIDDLE, PALM, PINKY, RING, THUMB, robot_link_labels
from fastdo.retarget.mj_model import RobotScene
from fastdo.retarget.primitives import hand_capsule_covers, primitive_distances
from fastdo.solver.sdf import SdfGrid

__all__ = ["PrimitiveClearanceStats", "primitive_clearance_stats", "robot_contact_frames"]

_FINGERS = (("thumb", THUMB), ("index", INDEX), ("middle", MIDDLE), ("ring", RING), ("pinky", PINKY),
            ("palm", PALM))


def robot_contact_frames(qpos: np.ndarray, scene: RobotScene, sdf: SdfGrid, surface: Tensor, mesh_scale: float,
                         n_per_link: int, seed: int) -> Tensor:
    """`(T,)` bool: frames where the robot contact measure (sampled link surfaces) has any band weight."""
    local_pts, body_ids, _ = scene.surface_points(n_per_link, seed=seed)
    labels = robot_link_labels(scene.model, body_ids)
    measure = robot_contact_measure(torch.as_tensor(qpos, dtype=torch.float64), scene, body_ids, local_pts,
                                    labels, sdf, surface, mesh_scale, jacobian=False)
    assert not isinstance(measure, tuple)
    return measure.weights.sum(dim=1) > 0


@dataclass(frozen=True)
class PrimitiveClearanceStats:
    """Per-trajectory primitive clearance summary (millimetres); finger medians are over contact frames."""

    n_frames: int
    n_contact_frames: int
    frac_frames_penetrating: float
    frac_frames_clear_0p5mm: float
    frame_min_mm_min: float
    frame_min_mm_p1: float
    frame_min_mm_median: float
    thumb_median_mm: float
    index_median_mm: float
    middle_median_mm: float
    ring_median_mm: float
    pinky_median_mm: float
    palm_median_mm: float

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def primitive_clearance_stats(qpos: np.ndarray, scene: RobotScene, sdf: SdfGrid, mesh_scale: float,
                              contact_frames: Tensor, box_face_tol: float = 5e-4) -> PrimitiveClearanceStats:
    """Summarize `primitive_distances` for `qpos (T, nq)`; `contact_frames (T,)` from `robot_contact_frames`."""
    covers = hand_capsule_covers(scene, box_face_tol)
    d_mm = primitive_distances(qpos, scene, covers, sdf, mesh_scale) * 1000.0               # (T,P)
    frame_min = d_mm.min(dim=1).values
    labels = robot_link_labels(scene.model, covers.body_ids)
    contact = contact_frames.to(torch.bool)

    def _finger_median(label: int) -> float:
        sel = labels == label
        if not bool(sel.any()) or not bool(contact.any()):
            return float("nan")
        return float(torch.median(d_mm[:, sel].min(dim=1).values[contact]))

    medians = {f"{name}_median_mm": _finger_median(label) for name, label in _FINGERS}
    return PrimitiveClearanceStats(
        n_frames=int(d_mm.shape[0]), n_contact_frames=int(contact.sum()),
        frac_frames_penetrating=float((frame_min < 0.0).double().mean()),
        frac_frames_clear_0p5mm=float((frame_min >= 0.5).double().mean()),
        frame_min_mm_min=float(frame_min.min()), frame_min_mm_p1=float(torch.quantile(frame_min, 0.01)),
        frame_min_mm_median=float(frame_min.median()), **medians,
    )
