"""SE(3) object error for the amortised tracker.

    E_t = (T^ref_t)^-1 T_t,  xi_t = log(E_t)^v = (rho_t, omega_t),  theta_t = |omega_t|,
    rho_t = V(omega_t)^-1 R^ref_t^T (p_t - p^ref_t),
    V(w) = I + (1 - cos th)/th^2 [w]x + (th - sin th)/th^3 [w]x^2.

E is a LEFT-invariant error: replacing both poses by G T and G T^ref (a change of world frame) leaves xi unchanged.
Quaternions are wxyz. Everything is written element by element (no BLAS, no reductions whose order could depend on the
batch), so a batched call returns, for each row, exactly the bits of a one-row call (the replay rebuild relies
on it). Small angles use series expansions; the rotation log goes through the quaternion and atan2, which is accurate
up to theta = pi.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "dot3",
    "object_error",
    "quat_conj",
    "quat_from_mat",
    "quat_mul",
    "quat_rotate",
    "quat_to_mat",
    "se3_exp",
    "se3_log",
    "so3_exp_quat",
    "so3_log_quat",
    "v_apply",
    "v_inv_apply",
]

_SMALL = 1e-4


def dot3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1] + a[..., 2] * b[..., 2]


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.stack([a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1], a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
                     a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]], -1)


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.concatenate([q[..., :1], -q[..., 1:]], -1)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, av, bw, bv = a[..., 0], a[..., 1:], b[..., 0], b[..., 1:]
    w = aw * bw - dot3(av, bv)
    v = aw[..., None] * bv + bw[..., None] * av + _cross(av, bv)
    return np.concatenate([w[..., None], v], -1)


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """R(q) v for unit q (v + 2 w (u x v) + 2 u x (u x v))."""
    w, u = q[..., :1], q[..., 1:]
    t = 2.0 * _cross(u, v)
    return v + w * t + _cross(u, t)


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                     2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                     2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(*q.shape[:-1], 3, 3)


def quat_from_mat(R: np.ndarray) -> np.ndarray:
    """Shepperd's method (largest diagonal pivot), w >= 0."""
    R = np.asarray(R, np.float64)
    m00, m11, m22 = R[..., 0, 0], R[..., 1, 1], R[..., 2, 2]
    tr = m00 + m11 + m22
    cands = np.stack([
        np.stack([1 + tr, R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1),
        np.stack([R[..., 2, 1] - R[..., 1, 2], 1 + m00 - m11 - m22, R[..., 0, 1] + R[..., 1, 0], R[..., 0, 2] + R[..., 2, 0]], -1),
        np.stack([R[..., 0, 2] - R[..., 2, 0], R[..., 0, 1] + R[..., 1, 0], 1 - m00 + m11 - m22, R[..., 1, 2] + R[..., 2, 1]], -1),
        np.stack([R[..., 1, 0] - R[..., 0, 1], R[..., 0, 2] + R[..., 2, 0], R[..., 1, 2] + R[..., 2, 1], 1 - m00 - m11 + m22], -1),
    ], -2)
    piv = np.argmax(np.stack([tr, m00, m11, m22], -1), -1)
    q = np.take_along_axis(cands, piv[..., None, None], -2)[..., 0, :]
    q = q / np.sqrt(q[..., 0] ** 2 + q[..., 1] ** 2 + q[..., 2] ** 2 + q[..., 3] ** 2)[..., None]
    return np.where(q[..., :1] < 0, -q, q)


def so3_log_quat(q: np.ndarray) -> np.ndarray:
    """omega with exp(omega) = R(q) and |omega| <= pi."""
    q = np.where(q[..., :1] < 0, -q, q)
    w, v = q[..., 0], q[..., 1:]
    s = np.sqrt(dot3(v, v))
    th = 2.0 * np.arctan2(s, w)
    small = s < _SMALL
    # th / s = 2 atan2(s, w) / s -> 2/w (1 - s^2/(3 w^2)) for small s (w ~ 1)
    k = np.where(small, 2.0 / np.where(small, w, 1.0) * (1.0 - s * s / (3.0 * w * w)), th / np.where(small, 1.0, s))
    return k[..., None] * v


def so3_exp_quat(om: np.ndarray) -> np.ndarray:
    th = np.sqrt(dot3(om, om))
    small = th < _SMALL
    h = 0.5 * th
    sinc = np.where(small, 0.5 - th * th / 48.0, np.sin(h) / np.where(small, 1.0, th))
    return np.concatenate([np.cos(h)[..., None], sinc[..., None] * om], -1)


def _coeffs(th: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A = (1 - cos th)/th^2, B = (th - sin th)/th^3, with series near 0."""
    small = th < 1e-3
    t = np.where(small, 1.0, th)
    t2 = th * th
    A = np.where(small, 0.5 - t2 / 24.0 + t2 * t2 / 720.0, (1.0 - np.cos(t)) / (t * t))
    B = np.where(small, 1.0 / 6.0 - t2 / 120.0 + t2 * t2 / 5040.0, (t - np.sin(t)) / (t * t * t))
    return A, B


def v_apply(om: np.ndarray, x: np.ndarray) -> np.ndarray:
    th = np.sqrt(dot3(om, om))
    A, B = _coeffs(th)
    wx = _cross(om, x)
    return x + A[..., None] * wx + B[..., None] * _cross(om, wx)


def v_inv_apply(om: np.ndarray, x: np.ndarray) -> np.ndarray:
    """V(w)^-1 x = x - 1/2 w x x + C w x (w x x), C = (1 - th sin th / (2 (1 - cos th))) / th^2."""
    th = np.sqrt(dot3(om, om))
    small = th < 1e-3
    t = np.where(small, 1.0, th)
    t2 = th * th
    C = np.where(small, 1.0 / 12.0 + t2 / 720.0 + t2 * t2 / 30240.0,
                 (1.0 - t * np.sin(t) / (2.0 * (1.0 - np.cos(t)))) / (t * t))
    wx = _cross(om, x)
    return x - 0.5 * wx + C[..., None] * _cross(om, wx)


def se3_log(p: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    om = so3_log_quat(q)
    return v_inv_apply(om, p), om


def se3_exp(rho: np.ndarray, om: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return v_apply(om, rho), so3_exp_quat(om)


def object_error(p: np.ndarray, q: np.ndarray, p_ref: np.ndarray, q_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(rho, omega) of E = (T^ref)^-1 T for poses (p, q) and (p_ref, q_ref) (module docstring)."""
    qi = quat_conj(q_ref)
    qe = quat_mul(qi, q)
    pe = quat_rotate(qi, p - p_ref)
    return se3_log(pe, qe)
