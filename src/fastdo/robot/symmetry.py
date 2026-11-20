"""Near-rotational symmetry of a reconstructed object, and the one task adaptation it allows.

**The adaptation.** When the robot cannot orient its wrist the way the human did, the only change of the task allowed
here is a rotation of the GRASP about an axis the object is symmetric about: the wrist target keeps its pose relative
to the object, rotated by a fixed angle theta about the object's axis through its centre,
`wrist'_t = O_t S O_t^-1 wrist_t` with `S` that rotation in the object's metric frame and `O_t` the object pose. If
the object maps onto itself under `S`, the rotated hand touches the same surface the human touched, so the object
trajectory and the hand-object contact geometry are unchanged: the human could have grasped it that way. A uniform
rescale of the hand path is NOT allowed (it changes the object's motion), nor is any rotation the object is not
symmetric under. Every applied rotation is reported in the package.

**The test** (`rotational_symmetries`). `n_samples` points are sampled on the mesh surface (vertices x scale, fixed
seed); the axes are the principal axes of the samples about their centroid. For each axis and each angle in
`step_deg` steps, the samples are rotated about the axis through the centroid and the two-way nearest-neighbour
distance between the rotated and the original sample set is measured; its 95th percentile is the surface mismatch.
An angle is ALLOWED when the mismatch is at most `tol_m` (3 mm by default: the mismatch of two independent 16,000-point
samplings of the same reconstructed meshes is 0.7-1.1 mm, so 3 mm leaves about 2 mm of real asymmetry, below the 5 mm
table-penetration gate and a fraction of a Dex3 fingertip). On those meshes: the hook screwdriver (C3) passes
at every angle about its long axis (1.7-3.2 mm at 30 deg steps); the squeeze bottle (C1) passes at none (7-14 mm).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

__all__ = ["SymmetryAxis", "grasp_rotation", "rotate_grasp", "rotational_symmetries"]


@dataclass(frozen=True)
class SymmetryAxis:
    index: int                        # principal-axis index (0 = smallest extent)
    axis: np.ndarray                  # (3,) unit axis in the object's metric frame (mesh vertices x scale)
    centre: np.ndarray                # (3,) a point on the axis (the surface samples' centroid)
    extent_m: float                   # extent of the samples along the axis
    mismatch_m: dict[int, float]      # angle (deg) -> 95th-percentile two-way surface distance (m)
    allowed_deg: tuple[int, ...]      # angles with mismatch <= tol (0 always)

    def to_dict(self) -> dict:
        return {"index": self.index, "axis": self.axis.tolist(), "centre_m": self.centre.tolist(),
                "extent_m": self.extent_m, "mismatch_mm": {str(k): round(v * 1000, 2) for k, v in self.mismatch_m.items()},
                "allowed_deg": list(self.allowed_deg)}


def rotational_symmetries(mesh_vertices: np.ndarray, mesh_faces: np.ndarray, scale: float, tol_m: float = 0.003,
                          step_deg: int = 15, n_samples: int = 16000, seed: int = 0) -> list[SymmetryAxis]:
    """The three principal axes with their surface mismatch per angle (module docstring)."""
    mesh = trimesh.Trimesh(scale * np.asarray(mesh_vertices, dtype=np.float64), np.asarray(mesh_faces), process=False)
    pts = np.asarray(trimesh.sample.sample_surface(mesh, n_samples, seed=seed)[0], dtype=np.float64)
    centre = pts.mean(axis=0)
    x = pts - centre
    _, vecs = np.linalg.eigh(x.T @ x / len(x))
    tree = cKDTree(pts)
    out = []
    for k in range(3):
        axis = vecs[:, k] / np.linalg.norm(vecs[:, k])
        mism: dict[int, float] = {}
        for deg in range(step_deg, 360, step_deg):
            rot = Rotation.from_rotvec(np.radians(deg) * axis).as_matrix()
            y = x @ rot.T + centre
            d = np.concatenate([tree.query(y)[0], cKDTree(y).query(pts)[0]])
            mism[deg] = float(np.percentile(d, 95))
        allowed = (0,) + tuple(deg for deg, v in mism.items() if v <= tol_m)
        out.append(SymmetryAxis(index=k, axis=axis, centre=centre, extent_m=float(np.ptp(x @ axis)),
                                mismatch_m=mism, allowed_deg=allowed))
    return out


def grasp_rotation(axis: np.ndarray, centre: np.ndarray, angle_rad: float) -> np.ndarray:
    """`(4,4)` rotation by `angle_rad` about `axis` through `centre`, in the object's metric frame."""
    rot = Rotation.from_rotvec(float(angle_rad) * np.asarray(axis, dtype=np.float64)).as_matrix()
    out = np.eye(4)
    out[:3, :3] = rot
    out[:3, 3] = np.asarray(centre, dtype=np.float64) - rot @ np.asarray(centre, dtype=np.float64)
    return out


def rotate_grasp(wrist: np.ndarray, objects: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """`(T,4,4)` wrist poses `objects_t @ rotation @ inv(objects_t) @ wrist_t` (module docstring); `objects` are the
    object poses (metric object frame -> same frame as `wrist`)."""
    rel = np.linalg.solve(objects, wrist)
    return np.einsum("tij,jk,tkl->til", objects, rotation, rel)
