"""NumPy SO(3) helpers for the reference conditioner (float64, batched over leading axes).

Conventions match `fastdo.geometry.lie` (axis-angle `theta`, `Exp(theta) = I + A K + B K^2`); the Jacobians are the
left Jacobian `J_l(theta) = I + B K + C K^2` and the right Jacobian `J_r(theta) = J_l(-theta)`. `log` uses SciPy's
quaternion route, which is stable near 0 and pi.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

__all__ = ["angle_deg", "exp", "hat", "jl", "jl_inv", "jr_inv", "log", "project"]


def hat(w: np.ndarray) -> np.ndarray:
    """Skew matrix of `w (...,3)` -> `(...,3,3)`."""
    w = np.asarray(w, dtype=np.float64)
    z = np.zeros(w.shape[:-1])
    return np.stack([np.stack([z, -w[..., 2], w[..., 1]], -1), np.stack([w[..., 2], z, -w[..., 0]], -1),
                     np.stack([-w[..., 1], w[..., 0], z], -1)], -2)


def exp(theta: np.ndarray) -> np.ndarray:
    """Axis-angle `(...,3)` -> rotation `(...,3,3)`."""
    th = np.asarray(theta, dtype=np.float64)
    flat = th.reshape(-1, 3)
    return Rotation.from_rotvec(flat).as_matrix().reshape(*th.shape[:-1], 3, 3)


def log(R: np.ndarray) -> np.ndarray:
    """Rotation `(...,3,3)` -> axis-angle `(...,3)`, angle in [0, pi]."""
    r = np.asarray(R, dtype=np.float64)
    flat = r.reshape(-1, 3, 3)
    return Rotation.from_matrix(flat).as_rotvec().reshape(*r.shape[:-2], 3)


def _coeffs(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`(B, C)` with B = (1 - cos t)/t^2 and C = (t - sin t)/t^3 (Taylor near 0)."""
    t = np.linalg.norm(theta, axis=-1)
    small = t < 1e-4
    ts = np.where(small, 1.0, t)
    b = np.where(small, 0.5 - t * t / 24.0, (1.0 - np.cos(ts)) / ts ** 2)
    c = np.where(small, 1.0 / 6.0 - t * t / 120.0, (ts - np.sin(ts)) / ts ** 3)
    return b, c


def jl(theta: np.ndarray) -> np.ndarray:
    """Left Jacobian of SO(3)."""
    th = np.asarray(theta, dtype=np.float64)
    b, c = _coeffs(th)
    k = hat(th)
    return np.eye(3) + b[..., None, None] * k + c[..., None, None] * (k @ k)


def jl_inv(theta: np.ndarray) -> np.ndarray:
    """Inverse left Jacobian of SO(3): I - K/2 + (1/t^2)(1 - (t/2) cot(t/2)) K^2."""
    th = np.asarray(theta, dtype=np.float64)
    t = np.linalg.norm(th, axis=-1)
    small = t < 1e-4
    ts = np.where(small, 1.0, t)
    coef = np.where(small, 1.0 / 12.0 + t * t / 720.0, (1.0 - (ts / 2.0) / np.tan(ts / 2.0)) / ts ** 2)
    k = hat(th)
    return np.eye(3) - 0.5 * k + coef[..., None, None] * (k @ k)


def jr_inv(theta: np.ndarray) -> np.ndarray:
    """Inverse right Jacobian: `J_r^{-1}(theta) = J_l^{-1}(-theta)`."""
    return jl_inv(-np.asarray(theta, dtype=np.float64))


def angle_deg(ra: np.ndarray, rb: np.ndarray) -> np.ndarray:
    """Geodesic angle (deg) between rotations `ra` and `rb` (broadcast)."""
    rel = np.einsum("...ji,...jk->...ik", ra, rb)
    c = (np.trace(rel, axis1=-2, axis2=-1) - 1.0) / 2.0
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def project(m: np.ndarray) -> np.ndarray:
    """Nearest rotation (Frobenius) to a 3x3 matrix (SVD, det +1)."""
    u, _, vt = np.linalg.svd(np.asarray(m, dtype=np.float64))
    d = np.sign(np.linalg.det(u @ vt))
    return u @ np.diag([1.0, 1.0, d]) @ vt
