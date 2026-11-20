import math

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from fastdo.geometry.lie import (
    apply_left,
    hat,
    point_jacobian,
    se3_adjoint,
    se3_exp,
    se3_inv,
    se3_log,
    so3_exp,
    so3_log,
    vee,
)

torch.manual_seed(0)


def _rand_xi(n: int, scale: float = 1.0, seed: int = 0) -> torch.Tensor:
    """Random twists from a local generator (independent of global RNG state / test order).

    Rotation parts are scaled to keep angles well below pi so Log(Exp(xi)) == xi holds exactly.
    """
    gen = torch.Generator().manual_seed(seed)
    xi = torch.randn(n, 6, dtype=torch.float64, generator=gen) * scale
    xi[:, 3:] *= 0.6
    return xi


def test_hat_vee_roundtrip() -> None:
    w = torch.randn(5, 3, dtype=torch.float64)
    assert torch.allclose(vee(hat(w)), w)
    assert torch.allclose(hat(w), -hat(w).transpose(-1, -2))


def test_so3_exp_matches_scipy() -> None:
    theta = torch.randn(8, 3, dtype=torch.float64)
    ours = so3_exp(theta).numpy()
    ref = R.from_rotvec(theta.numpy()).as_matrix()
    assert np.allclose(ours, ref, atol=1e-10)


def test_so3_log_roundtrip_and_small_angle() -> None:
    # log is the inverse of exp only on the principal branch (|theta| < pi); draw from a seeded generator so the test no
    # longer depends on the global RNG state left by earlier tests, and keep every angle inside the branch.
    gen = torch.Generator().manual_seed(0)
    theta = torch.randn(8, 3, dtype=torch.float64, generator=gen)
    theta = theta * (3.0 * torch.rand(8, 1, dtype=torch.float64, generator=gen) / theta.norm(dim=1, keepdim=True))
    assert torch.allclose(so3_log(so3_exp(theta)), theta, atol=1e-9)
    tiny = torch.tensor([[1e-9, -2e-9, 3e-9]], dtype=torch.float64)
    assert torch.allclose(so3_log(so3_exp(tiny)), tiny, atol=1e-12)


def test_se3_exp_log_roundtrip() -> None:
    xi = _rand_xi(8)
    T = se3_exp(xi)
    assert torch.allclose(T[:, 3, :], torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64).expand(8, 4))
    assert torch.allclose(se3_log(T), xi, atol=1e-9)


def test_se3_inv() -> None:
    T = se3_exp(_rand_xi(4))
    eye = torch.eye(4, dtype=torch.float64).expand(4, 4, 4)
    assert torch.allclose(se3_inv(T) @ T, eye, atol=1e-12)


def test_adjoint_identity() -> None:
    T = se3_exp(_rand_xi(4))
    xi = _rand_xi(4, 0.3)
    lhs = se3_exp((se3_adjoint(T) @ xi.unsqueeze(-1)).squeeze(-1))
    rhs = T @ se3_exp(xi) @ se3_inv(T)
    assert torch.allclose(lhs, rhs, atol=1e-9)


def test_point_jacobian_matches_finite_difference() -> None:
    T = se3_exp(_rand_xi(1))[0]
    y = torch.randn(3, dtype=torch.float64)
    p = T[:3, :3] @ y + T[:3, 3]
    J = point_jacobian(p)                      # (3,6)
    eps = 1e-6
    num = torch.zeros(3, 6, dtype=torch.float64)
    for i in range(6):
        d = torch.zeros(6, dtype=torch.float64)
        d[i] = eps
        Tp = apply_left(T, d)
        num[:, i] = ((Tp[:3, :3] @ y + Tp[:3, 3]) - p) / eps
    assert torch.allclose(J, num, atol=1e-5)


def test_so3_log_near_pi_roundtrip() -> None:
    g = torch.Generator().manual_seed(3)
    axes = torch.randn(6, 3, dtype=torch.float64, generator=g)
    axes = axes / axes.norm(dim=1, keepdim=True)
    for angle in (3.10, 3.14, math.pi - 1e-5, math.pi - 1e-7, math.pi):
        R = so3_exp(axes * angle)
        R_rec = so3_exp(so3_log(R))
        assert torch.allclose(R_rec, R, atol=1e-8), f"angle {angle}: {(R_rec - R).abs().max()}"


def test_so3_log_continuity_sweep() -> None:
    axis = torch.tensor([0.3, -0.5, 0.8], dtype=torch.float64)
    axis = axis / axis.norm()
    angles = torch.linspace(2.5, math.pi, 400, dtype=torch.float64)
    R = so3_exp(axis[None] * angles[:, None])
    err = (so3_exp(so3_log(R)) - R).abs().flatten(1).max(dim=1).values
    assert float(err.max()) < 1e-8


def test_so3_log_float32_near_pi() -> None:
    axis = torch.tensor([[0.3, -0.5, 0.8]], dtype=torch.float32)
    axis = axis / axis.norm()
    for angle in (0.5, 2.9, 3.13, 3.14, 3.141, 3.1414, math.pi - 1e-5, math.pi):
        R = so3_exp(axis * angle)
        R_rec = so3_exp(so3_log(R))
        assert float((R_rec - R).abs().max()) < 1e-4, f"float32 angle {angle}: {(R_rec - R).abs().max()}"
