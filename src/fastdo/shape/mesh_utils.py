"""Mesh preparation shared by the shape stage and the DexYCB evaluation (D4).

- `normalize_mesh` recentres a mesh on its bounding-box centre and rescales it to a unit bounding-box diagonal. Its
  output is the "canonical unit" mesh that the M1 solver's `scale` and the retargeting `mesh_scale` (metres per
  canonical unit) refer to.
- `check_watertight` merges duplicate vertices and applies `trimesh.repair.fill_holes` once when a mesh is not
  watertight. If the result is still not watertight, it logs a warning and returns the input unchanged. It never
  raises, since a non-watertight mesh still yields a usable, if less reliable, pseudonormal SDF.
- `canonical_samples` returns seeded surface samples with outward-oriented face normals. It is the single
  implementation behind the solver's `MeshSamples` here and in `run/pipeline/eval_solver_dexycb.py`.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import trimesh
from torch import Tensor

__all__ = ["canonical_samples", "check_watertight", "normalize_mesh"]

logger = logging.getLogger(__name__)


def normalize_mesh(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Recentres `vertices` on their bounding-box centre and scales them to a unit bounding-box diagonal.

    Args:
        vertices: `(V,3)` vertex positions; not modified.
        faces: `(F,3)` vertex indices, returned as an int64 copy.

    Returns:
        `(vertices (V,3) float64, faces (F,3) int64, applied_scale)`, where
        `normalized = (vertices - centre) * applied_scale`. `1 / applied_scale` is the input's bounding-box diagonal.

    Raises:
        ValueError: If the vertices are not an `(V,3)` array of finite values or their bounding box is degenerate.
    """
    v = np.asarray(vertices, dtype=np.float64)
    if v.ndim != 2 or v.shape[1] != 3 or v.shape[0] == 0:
        raise ValueError(f"vertices must be (V,3) with V >= 1, got shape {v.shape}")
    if not np.isfinite(v).all():
        raise ValueError("vertices contain non-finite values")
    lo, hi = v.min(axis=0), v.max(axis=0)
    diagonal = float(np.linalg.norm(hi - lo))
    if not diagonal > 0.0:
        raise ValueError("mesh bounding box is degenerate (zero diagonal); cannot normalize")
    applied_scale = 1.0 / diagonal
    normalized = (v - (lo + hi) / 2.0) * applied_scale
    return normalized, np.asarray(faces, dtype=np.int64).copy(), applied_scale


def check_watertight(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    """`(vertices, faces, True)` for a watertight mesh, repairing it once if needed; never raises.

    A mesh that is already watertight is returned unchanged. Otherwise a copy has its duplicate vertices merged
    (marching-cubes and decimated meshes often repeat seam vertices, which alone breaks edge manifoldness) and
    `trimesh.repair.fill_holes` applied once. The repaired arrays are returned if that closes the mesh; if it does
    not, or trimesh fails on the input, a warning is logged and the input comes back unchanged with `False`.
    """
    try:
        mesh = trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=False)
        if mesh.is_watertight:
            return vertices, faces, True
        repaired = mesh.copy()
        repaired.merge_vertices()
        trimesh.repair.fill_holes(repaired)
        if repaired.is_watertight:
            logger.info("mesh repaired to watertight (%d -> %d faces)", len(mesh.faces), len(repaired.faces))
            return (np.asarray(repaired.vertices, dtype=np.float64), np.asarray(repaired.faces, dtype=np.int64),
                    True)
        logger.warning("mesh is not watertight after one fill_holes pass (%d faces); using it unrepaired",
                       len(mesh.faces))
    except Exception as exc:  # noqa: BLE001 - the contract is "never raises": trimesh can fail in many ways
        logger.warning("watertight check failed on this mesh (%s: %s); using it unrepaired", type(exc).__name__, exc)
    return vertices, faces, False


def canonical_samples(vertices: np.ndarray, faces: np.ndarray, n: int, seed: int) -> tuple[Tensor, Tensor]:
    """`n` seeded area-uniform surface samples and their outward-oriented unit face normals.

    Args:
        vertices: `(V,3)` canonical-frame vertices.
        faces: `(F,3)` vertex indices.
        n: Number of samples.
        seed: Seed of `trimesh.sample.sample_surface`; equal inputs give bit-identical outputs.

    Returns:
        `(points (n,3) float32, normals (n,3) float32)`.

    Mesh files do not guarantee a winding order, while `visible_samples`'s front-facing test assumes outward normals.
    Every normal is therefore flipped when `mean n_j . (y_j - centroid)` over the samples is
    negative.
    """
    mesh = trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=False)
    points, face_idx = trimesh.sample.sample_surface(mesh, n, seed=seed)
    points_t = torch.as_tensor(np.asarray(points), dtype=torch.float32)
    normals_t = torch.as_tensor(np.asarray(mesh.face_normals[face_idx], dtype=np.float32))
    centroid = points_t.mean(dim=0)
    outward = float((normals_t * (points_t - centroid)).sum(-1).mean())
    if outward < 0:
        logger.info("mesh normals appear inward-oriented (mean n.(y-centroid)=%.4f); flipping", outward)
        return points_t, -normals_t
    logger.info("mesh normals confirmed outward-oriented (mean n.(y-centroid)=%.4f)", outward)
    return points_t, normals_t
