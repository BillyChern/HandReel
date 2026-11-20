"""Small exact geometry helpers for the temporal object bundle adjustment (TBA).

Conventions (the production solver's): a camera-from-object pose ``P`` (4x4) maps a canonical template point ``p``
to the camera frame as ``X = s R p + t`` with one window scale ``s``. Pixels are ``(x, y)``. A keyframe ray of pixel
``q`` is ``r = K^-1 [q, 1]`` (its z component is 1), so a point on it is ``X = d r`` with ``d`` its camera z depth.

Everything is float64 numpy; nothing here allocates on a GPU.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

__all__ = [
    "kabsch",
    "pixel_rays",
    "pose_inverse",
    "posed_points",
    "project",
    "ray_mesh_depth",
    "relative_poses",
    "rotvec",
    "rotvec_to_matrix",
]


def pixel_rays(uv: np.ndarray, K: np.ndarray) -> np.ndarray:
    """``(N,3)`` rays ``K^-1 [u, v, 1]`` (z = 1, so ``d * r`` has camera depth ``d``)."""
    uv = np.asarray(uv, np.float64).reshape(-1, 2)
    Kinv = np.linalg.inv(np.asarray(K, np.float64))
    hom = np.concatenate([uv, np.ones((len(uv), 1))], axis=1)
    r = hom @ Kinv.T
    return r / r[:, 2:3]


def project(X: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Pixels ``(..., 2)`` of camera-frame points ``(..., 3)`` (no clamping: the caller checks z > 0)."""
    X = np.asarray(X, np.float64)
    K = np.asarray(K, np.float64)
    z = X[..., 2]
    return np.stack([K[0, 0] * X[..., 0] / z + K[0, 2], K[1, 1] * X[..., 1] / z + K[1, 2]], axis=-1)


def posed_points(points: np.ndarray, pose: np.ndarray, scale: float) -> np.ndarray:
    """``s R p + t`` of canonical points ``(M,3)`` under one pose ``(4,4)``."""
    pose = np.asarray(pose, np.float64)
    return float(scale) * np.asarray(points, np.float64) @ pose[:3, :3].T + pose[:3, 3]


def pose_inverse(pose: np.ndarray) -> np.ndarray:
    """Inverse of a rigid 4x4 (batched over leading axes)."""
    pose = np.asarray(pose, np.float64)
    out = np.zeros_like(pose)
    rt = np.swapaxes(pose[..., :3, :3], -1, -2)
    out[..., :3, :3] = rt
    out[..., :3, 3] = -np.einsum("...ij,...j->...i", rt, pose[..., :3, 3])
    out[..., 3, 3] = 1.0
    return out


def relative_poses(poses: np.ndarray, key: int) -> np.ndarray:
    """``M_k`` with ``X_k = M_k X_key`` for the object's camera-frame points: ``R_k R_key^T``, ``t_k - R_k R_key^T t_key``.

    The template scale cancels: every posed template point satisfies it whatever ``s`` is.
    """
    poses = np.asarray(poses, np.float64)
    return poses @ pose_inverse(poses[key])[None]


def rotvec(R: np.ndarray) -> np.ndarray:
    """Axis-angle vectors ``(...,3)`` of rotation matrices ``(...,3,3)``."""
    R = np.asarray(R, np.float64)
    return Rotation.from_matrix(R.reshape(-1, 3, 3)).as_rotvec().reshape(*R.shape[:-2], 3)


def rotvec_to_matrix(w: np.ndarray) -> np.ndarray:
    """Rotation matrices ``(...,3,3)`` of axis-angle vectors ``(...,3)``."""
    w = np.asarray(w, np.float64)
    return Rotation.from_rotvec(w.reshape(-1, 3)).as_matrix().reshape(*w.shape[:-1], 3, 3)


def kabsch(src: np.ndarray, dst: np.ndarray, weights: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Rigid ``(R, t)`` minimising ``sum_i w_i |R src_i + t - dst_i|^2`` (no scale; proper rotation).

    Raises:
        ValueError: With fewer than 3 points of positive weight.
    """
    a = np.asarray(src, np.float64).reshape(-1, 3)
    b = np.asarray(dst, np.float64).reshape(-1, 3)
    w = np.ones(len(a)) if weights is None else np.asarray(weights, np.float64).reshape(-1)
    if int((w > 0).sum()) < 3:
        raise ValueError("kabsch needs at least 3 points of positive weight")
    w = w / w.sum()
    ca, cb = w @ a, w @ b
    H = (a - ca).T @ ((b - cb) * w[:, None])
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cb - R @ ca


def ray_mesh_depth(rays: np.ndarray, vertices: np.ndarray, faces: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """First hit of rays from the camera centre against a camera-frame mesh (Moller-Trumbore, both face sides).

    Args:
        rays: ``(N,3)`` directions with z = 1 (``pixel_rays``), so the hit parameter IS the camera depth.
        vertices: ``(V,3)`` camera-frame vertices.
        faces: ``(F,3)`` triangle indices.
        chunk: Faces per vectorised block.

    Returns:
        ``(N,)`` depth of the nearest hit with depth > 0, NaN where the ray misses.
    """
    d = np.asarray(rays, np.float64).reshape(-1, 3)
    V = np.asarray(vertices, np.float64)
    F = np.asarray(faces, np.int64)
    best = np.full(len(d), np.inf)
    eps = 1e-12
    for lo in range(0, len(F), chunk):
        f = F[lo:lo + chunk]
        v0, e1, e2 = V[f[:, 0]], V[f[:, 1]] - V[f[:, 0]], V[f[:, 2]] - V[f[:, 0]]
        p = np.cross(d[:, None, :], e2[None, :, :])                        # (N,F,3)
        det = np.einsum("fj,nfj->nf", e1, p)
        ok = np.abs(det) > eps
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        s = -v0                                                              # origin (0,0,0) minus v0
        u = np.einsum("fj,nfj->nf", s, p) * inv
        q = np.cross(s, e1)                                                  # (F,3)
        v = np.einsum("nj,fj->nf", d, q) * inv
        t = np.einsum("fj,fj->f", e2, q)[None, :] * inv
        hit = ok & (u >= -1e-12) & (v >= -1e-12) & (u + v <= 1.0 + 1e-12) & (t > 1e-9)
        t = np.where(hit, t, np.inf)
        best = np.minimum(best, t.min(axis=1))
    return np.where(np.isfinite(best), best, np.nan)
