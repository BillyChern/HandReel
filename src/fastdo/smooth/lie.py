"""SO(3) / SE(3) in float64 NumPy for the pose smoother (`fastdo.smooth`).

Conventions (stated once, used everywhere):
- a twist is xi = (phi, rho) in R^6, ROTATION FIRST; ``hat(xi) = [[phi^, rho], [0, 0]]``;
- ``se3_exp(xi) = [[Exp(phi), J_l(phi) rho], [0, 1]]`` and ``se3_log`` is its inverse on |phi| < pi;
- body (right) perturbation ``T = Tbar Exp(delta)``; ``Ad_T = [[R, 0], [t^ R, R]]`` so that
  ``T Exp(xi) T^-1 = Exp(Ad_T xi)``;
- SE(3) Jacobians: ``Exp(xi + d) ~ Exp(xi) Exp(J_r(xi) d)`` and ``Log(Exp(xi) Exp(d)) ~ xi + J_r(xi)^-1 d``;
  ``J_l(xi) = J_r(-xi)``; in (phi, rho) order ``J_l = [[J, 0], [Q(phi, rho), J]]`` with J the SO(3) left
  Jacobian and Q Barfoot's (State Estimation for Robotics, eq. 7.86, with his (rho, phi) order swapped).

Small angles use Taylor series below ``_EPS`` so every function is smooth through 0 (tests/smooth/test_lie.py checks
each against central differences).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "se3_adjoint",
    "se3_exp",
    "se3_jl",
    "se3_jr",
    "se3_jr_inv",
    "se3_log",
    "so3_exp",
    "so3_hat",
    "so3_jl",
    "so3_jl_inv",
    "so3_jr",
    "so3_jr_inv",
    "so3_log",
]

_EPS = 1e-2


def _coefs(th: float) -> tuple[float, float, float]:
    """sin(th)/th, (1 - cos th)/th^2, (th - sin th)/th^3, by series below _EPS (error < 1e-16 there)."""
    if th < _EPS:
        t2 = th * th
        return 1.0 - t2 / 6.0 + t2 * t2 / 120.0, 0.5 - t2 / 24.0 + t2 * t2 / 720.0, 1.0 / 6.0 - t2 / 120.0 + t2 * t2 / 5040.0
    return np.sin(th) / th, (1.0 - np.cos(th)) / th ** 2, (th - np.sin(th)) / th ** 3


def so3_hat(w: np.ndarray) -> np.ndarray:
    """Skew matrix of a 3-vector: ``hat(w) @ v == cross(w, v)``."""
    x, y, z = (float(v) for v in w)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def so3_exp(phi: np.ndarray) -> np.ndarray:
    """Rodrigues' formula."""
    phi = np.asarray(phi, np.float64)
    a, b, _ = _coefs(float(np.linalg.norm(phi)))
    W = so3_hat(phi)
    return np.eye(3) + a * W + b * W @ W


def so3_log(R: np.ndarray) -> np.ndarray:
    """Rotation vector of R (angle in [0, pi]); stable near 0 and near pi."""
    R = np.asarray(R, np.float64)
    c = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    th = float(np.arccos(c))
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if th < _EPS:
        return 0.5 * v * (1.0 + th ** 2 / 6.0 + 7.0 * th ** 4 / 360.0)
    if np.pi - th < 1e-6:
        # axis from the symmetric part: R + I = 2 a a^T at th = pi
        B = 0.5 * (R + np.eye(3))
        i = int(np.argmax(np.diag(B)))
        a = B[:, i] / np.sqrt(max(B[i, i], 1e-300))
        if np.dot(a, v) < 0.0:
            a = -a
        return th * a / np.linalg.norm(a)
    return th / (2.0 * np.sin(th)) * v


def so3_jl(phi: np.ndarray) -> np.ndarray:
    """SO(3) left Jacobian J_l(phi) (= the V of se3_exp)."""
    phi = np.asarray(phi, np.float64)
    _, b, c = _coefs(float(np.linalg.norm(phi)))
    W = so3_hat(phi)
    return np.eye(3) + b * W + c * W @ W


def so3_jr(phi: np.ndarray) -> np.ndarray:
    return so3_jl(-np.asarray(phi, np.float64))


def so3_jl_inv(phi: np.ndarray) -> np.ndarray:
    phi = np.asarray(phi, np.float64)
    th = float(np.linalg.norm(phi))
    W = so3_hat(phi)
    if th < _EPS:
        coef = 1.0 / 12.0 + th * th / 720.0
    else:
        coef = 1.0 / th ** 2 - (1.0 + np.cos(th)) / (2.0 * th * np.sin(th))
    return np.eye(3) - 0.5 * W + coef * W @ W


def so3_jr_inv(phi: np.ndarray) -> np.ndarray:
    return so3_jl_inv(-np.asarray(phi, np.float64))


def se3_exp(xi: np.ndarray) -> np.ndarray:
    xi = np.asarray(xi, np.float64)
    T = np.eye(4)
    T[:3, :3] = so3_exp(xi[:3])
    T[:3, 3] = so3_jl(xi[:3]) @ xi[3:]
    return T


def se3_log(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, np.float64)
    phi = so3_log(T[:3, :3])
    rho = np.linalg.solve(so3_jl(phi), T[:3, 3])
    return np.concatenate([phi, rho])


def se3_adjoint(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, np.float64)
    R, t = T[:3, :3], T[:3, 3]
    A = np.zeros((6, 6))
    A[:3, :3] = R
    A[3:, 3:] = R
    A[3:, :3] = so3_hat(t) @ R
    return A


def _q(phi: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Barfoot's Q(rho, phi) (eq. 7.86)."""
    th = float(np.linalg.norm(phi))
    P, Rh = so3_hat(phi), so3_hat(rho)
    if th < 5e-2:
        # series (sympy): a = 1/6 - th^2/120, b = 1/24 - th^2/720, c = 1/120 - th^2/2520, next terms O(th^4) < 1e-10 here
        t2 = th * th
        a, b, c = 1.0 / 6.0 - t2 / 120.0, 1.0 / 24.0 - t2 / 720.0, 1.0 / 120.0 - t2 / 2520.0
    else:
        s, co = np.sin(th), np.cos(th)
        a = (th - s) / th ** 3
        b = (th ** 2 + 2.0 * co - 2.0) / (2.0 * th ** 4)
        c = (2.0 * th - 3.0 * s + th * co) / (2.0 * th ** 5)
    return (0.5 * Rh + a * (P @ Rh + Rh @ P + P @ Rh @ P)
            + b * (P @ P @ Rh + Rh @ P @ P - 3.0 * P @ Rh @ P)
            + c * (P @ Rh @ P @ P + P @ P @ Rh @ P))


def se3_jl(xi: np.ndarray) -> np.ndarray:
    xi = np.asarray(xi, np.float64)
    J = so3_jl(xi[:3])
    out = np.zeros((6, 6))
    out[:3, :3] = J
    out[3:, 3:] = J
    out[3:, :3] = _q(xi[:3], xi[3:])
    return out


def se3_jr(xi: np.ndarray) -> np.ndarray:
    return se3_jl(-np.asarray(xi, np.float64))


def se3_jr_inv(xi: np.ndarray) -> np.ndarray:
    J = se3_jr(xi)
    Ji = np.linalg.inv(J[:3, :3])
    out = np.zeros((6, 6))
    out[:3, :3] = Ji
    out[3:, 3:] = Ji
    out[3:, :3] = -Ji @ J[3:, :3] @ Ji
    return out
