"""Batched SO(3)/SE(3) exponential, logarithm, adjoint and point Jacobians (PyTorch)."""

from __future__ import annotations

import torch
from torch import Tensor

_EPS = 1e-8


def hat(w: Tensor) -> Tensor:
    """Skew-symmetric matrix of w (...,3) -> (...,3,3)."""
    zeros = torch.zeros_like(w[..., 0])
    return torch.stack([
        torch.stack([zeros, -w[..., 2], w[..., 1]], dim=-1),
        torch.stack([w[..., 2], zeros, -w[..., 0]], dim=-1),
        torch.stack([-w[..., 1], w[..., 0], zeros], dim=-1),
    ], dim=-2)


def vee(W: Tensor) -> Tensor:
    """Inverse of hat: (...,3,3) -> (...,3)."""
    return torch.stack([W[..., 2, 1], W[..., 0, 2], W[..., 1, 0]], dim=-1)


def _sinc_coeffs(angle: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Return (A, B, C) with A = sin t / t, B = (1 - cos t) / t^2, C = (t - sin t) / t^3, Taylor-safe."""
    t2 = angle * angle
    small = angle < 1e-4
    A = torch.where(small, 1.0 - t2 / 6.0, torch.sin(angle) / angle.clamp_min(_EPS))
    B = torch.where(small, 0.5 - t2 / 24.0, (1.0 - torch.cos(angle)) / t2.clamp_min(_EPS))
    C = torch.where(small, 1.0 / 6.0 - t2 / 120.0, (angle - torch.sin(angle)) / (t2 * angle).clamp_min(_EPS))
    return A, B, C


def so3_exp(theta: Tensor) -> Tensor:
    """Rodrigues: axis-angle (...,3) -> rotation (...,3,3)."""
    angle = torch.linalg.norm(theta, dim=-1)
    A, B, _ = _sinc_coeffs(angle)
    K = hat(theta)
    eye = torch.eye(3, dtype=theta.dtype, device=theta.device).expand(K.shape)
    return eye + A[..., None, None] * K + B[..., None, None] * (K @ K)


def so3_log(R: Tensor) -> Tensor:
    """Rotation (...,3,3) -> axis-angle (...,3), angle in [0, pi]; stable near 0 and pi in float32/64."""
    cos = ((R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] - 1.0) * 0.5).clamp(-1.0, 1.0)
    w = vee(R - R.transpose(-1, -2)) * 0.5            # = sin(angle) * axis, accurate in float32
    sin_norm = torch.linalg.norm(w, dim=-1)           # = sin(angle) >= 0
    angle = torch.atan2(sin_norm, cos)                # accurate where acos(cos) is not
    small = angle < 1e-4
    scale = torch.where(small, 1.0 + angle * angle / 6.0, angle / sin_norm.clamp_min(_EPS))
    regular = w * scale[..., None]
    # Near pi the antisymmetric part vanishes; recover the axis from the symmetric part:
    # (R + R^T)/2 - cos I = (1 - cos) a a^T, whose largest-diagonal column is (1 - cos) a_i a.
    eye = torch.eye(3, dtype=R.dtype, device=R.device)
    S = (R + R.transpose(-1, -2)) * 0.5 - cos[..., None, None] * eye
    idx = torch.argmax(torch.diagonal(S, dim1=-2, dim2=-1), dim=-1)
    col = torch.gather(S, -1, idx[..., None, None].expand(*S.shape[:-1], 1)).squeeze(-1)
    axis = col / torch.linalg.norm(col, dim=-1, keepdim=True).clamp_min(_EPS)
    sign = torch.where((axis * w).sum(-1, keepdim=True) < 0.0, -1.0, 1.0)
    near_pi_result = axis * sign * angle[..., None]
    near_pi = (sin_norm < 1e-4) & (cos < 0.0)
    return torch.where(near_pi[..., None], near_pi_result, regular)


def _left_jacobian_so3(theta: Tensor) -> Tensor:
    angle = torch.linalg.norm(theta, dim=-1)
    _, B, C = _sinc_coeffs(angle)
    K = hat(theta)
    eye = torch.eye(3, dtype=theta.dtype, device=theta.device).expand(K.shape)
    return eye + B[..., None, None] * K + C[..., None, None] * (K @ K)


def _left_jacobian_so3_inv(theta: Tensor) -> Tensor:
    angle = torch.linalg.norm(theta, dim=-1)
    K = hat(theta)
    eye = torch.eye(3, dtype=theta.dtype, device=theta.device).expand(K.shape)
    t2 = angle * angle
    small = angle < 1e-4
    half = angle * 0.5
    coef = torch.where(small, 1.0 / 12.0 + t2 / 720.0,
                       (1.0 - half * torch.cos(half) / torch.sin(half).clamp_min(_EPS)) / t2.clamp_min(_EPS))
    return eye - 0.5 * K + coef[..., None, None] * (K @ K)


def se3_exp(xi: Tensor) -> Tensor:
    """se(3) vector [rho, theta] (...,6) -> homogeneous transform (...,4,4)."""
    rho, theta = xi[..., :3], xi[..., 3:]
    R = so3_exp(theta)
    t = (_left_jacobian_so3(theta) @ rho.unsqueeze(-1)).squeeze(-1)
    T = torch.zeros(*xi.shape[:-1], 4, 4, dtype=xi.dtype, device=xi.device)
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    T[..., 3, 3] = 1.0
    return T


def se3_log(T: Tensor) -> Tensor:
    """Homogeneous transform (...,4,4) -> se(3) vector [rho, theta] (...,6)."""
    theta = so3_log(T[..., :3, :3])
    rho = (_left_jacobian_so3_inv(theta) @ T[..., :3, 3].unsqueeze(-1)).squeeze(-1)
    return torch.cat([rho, theta], dim=-1)


def se3_inv(T: Tensor) -> Tensor:
    Rt = T[..., :3, :3].transpose(-1, -2)
    out = torch.zeros_like(T)
    out[..., :3, :3] = Rt
    out[..., :3, 3] = -(Rt @ T[..., :3, 3].unsqueeze(-1)).squeeze(-1)
    out[..., 3, 3] = 1.0
    return out


def se3_adjoint(T: Tensor) -> Tensor:
    """Adjoint (...,6,6) such that Exp(Ad_T xi) = T Exp(xi) T^-1 with xi = [rho, theta]."""
    R = T[..., :3, :3]
    tx = hat(T[..., :3, 3])
    top = torch.cat([R, tx @ R], dim=-1)
    bot = torch.cat([torch.zeros_like(R), R], dim=-1)
    return torch.cat([top, bot], dim=-2)


def apply_left(T: Tensor, xi: Tensor) -> Tensor:
    """Left perturbation Exp(xi) @ T."""
    return se3_exp(xi) @ T


def point_jacobian(p: Tensor) -> Tensor:
    """d(Exp(xi) p)/d xi at xi = 0 for camera-frame points p (...,3): [I | -hat(p)] -> (...,3,6)."""
    eye = torch.eye(3, dtype=p.dtype, device=p.device).expand(*p.shape[:-1], 3, 3)
    return torch.cat([eye, -hat(p)], dim=-1)


def quat_wxyz_to_matrix(q: Tensor) -> Tensor:
    """MuJoCo/wxyz-convention quaternion (...,4) -> rotation matrix (...,3,3); normalizes `q` first."""
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    r = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1)
    return r.reshape(*q.shape[:-1], 3, 3)
