"""Synthetic windows for the reference-conditioner tests."""

from __future__ import annotations

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from fastdo.reference import HandInput, WindowInputs, declared_static
from fastdo.reference.symmetry_group import trivial_group

BOX = trimesh.creation.box(extents=[0.12, 0.08, 0.05])


def asym_mesh() -> trimesh.Trimesh:
    """A box with an off-centre block: no rotational symmetry."""
    a = trimesh.creation.box(extents=[0.12, 0.08, 0.05])
    b = trimesh.creation.box(extents=[0.04, 0.03, 0.03])
    b.apply_translation([0.05, 0.03, 0.035])
    return trimesh.util.concatenate([a, b])


def pose(rotvec, t) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    out[:3, 3] = t
    return out


def noisy(base: np.ndarray, n: int, rng, pos_sd: float = 0.002, rot_sd_deg: float = 0.5) -> np.ndarray:
    """`n` copies of `base` with a LEFT perturbation (camera frame) of the given sizes."""
    out = np.empty((n, 4, 4))
    for i in range(n):
        d = pose(rng.normal(0, np.radians(rot_sd_deg), 3), rng.normal(0, pos_sd, 3))
        out[i] = d @ base
    return out


def window(poses: np.ndarray, dist: np.ndarray, mesh=None, other_dist: np.ndarray | None = None, group=None,
           cov=None, conf=None, declared_absent_other: bool = True, world=None) -> WindowInputs:
    """A declared-static window at 10 Hz whose right-hand distance is `dist` (precomputed; NaN = invalid)."""
    mesh = mesh if mesh is not None else asym_mesh()
    n = len(poses)
    hands = {"right": HandInput(None, np.isfinite(dist), False, dist)}
    od = np.full(n, np.nan) if other_dist is None else other_dist
    hands["left"] = HandInput(None, np.isfinite(od), declared_absent_other and other_dist is None, od)
    g = group if group is not None else trivial_group(_c_star(mesh))
    return WindowInputs(poses=poses, times=np.arange(n) / 10.0, scale=1.0, mesh_vertices=np.asarray(mesh.vertices),
                        mesh_faces=np.asarray(mesh.faces), hands=hands, world=world or declared_static(n),
                        confidence=conf, covariance=cov, group=g)


def _c_star(mesh) -> np.ndarray:
    from fastdo.robot.symmetry import rotational_symmetries
    return rotational_symmetries(np.asarray(mesh.vertices), np.asarray(mesh.faces), 1.0, step_deg=90)[0].centre
