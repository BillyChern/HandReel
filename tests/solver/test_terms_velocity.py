"""Constant-velocity (second-difference) SE(3) prior, `fastdo.solver.terms.velocity_term`.

Kept in its own file rather than appended to `test_terms.py` only because that file is already at the
repository's 400-line cap. Residual convention: the same
body-frame relative motion `Log(T_k^-1 T_{k+1})` as `smoothness_term`, differenced over consecutive
pairs, `r_k = sqrt(lam_v / dt^3) [Log(T_k^-1 T_{k+1}) - Log(T_{k-1}^-1 T_k)]` for k = 1..T-2.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from fastdo.geometry.lie import apply_left, se3_exp
from fastdo.solver.terms import velocity_term

DT = torch.float64


def _random_poses(T: int, seed: int, spread: float = 0.3) -> Tensor:
    g = torch.Generator().manual_seed(seed)
    poses = se3_exp(torch.randn(T, 6, dtype=DT, generator=g) * spread)
    poses[:, 2, 3] += 0.6
    return poses


@pytest.mark.parametrize("side", ["left", "right"])
def test_velocity_residual_zero_on_constant_velocity_trajectory(side: str) -> None:
    """T_k = Exp(k xi) T_0 - and its right-multiplied twin T_0 Exp(k xi), which has the same
    relative motions up to a fixed conjugation - is annihilated to 1e-12."""
    g = torch.Generator().manual_seed(3)
    xi = torch.randn(6, dtype=DT, generator=g) * 0.1
    T0 = se3_exp(torch.randn(6, dtype=DT, generator=g) * 0.2)
    T0[2, 3] += 0.6
    steps = torch.stack([se3_exp(k * xi) for k in range(10)])
    poses = steps @ T0 if side == "left" else T0 @ steps
    r, J_prev, J_mid, J_next = velocity_term(poses, lam_v=2.5)
    assert r.shape == (8, 6)
    assert J_prev.shape == J_mid.shape == J_next.shape == (8, 6, 6)
    assert float(r.abs().max()) < 1e-12, float(r.abs().max())


def test_velocity_residual_nonzero_on_random_walk() -> None:
    g = torch.Generator().manual_seed(4)
    increments = se3_exp(torch.randn(10, 6, dtype=DT, generator=g) * 0.05)
    poses = [torch.eye(4, dtype=DT)]
    for k in range(9):
        poses.append(increments[k] @ poses[-1])
    r, _, _, _ = velocity_term(torch.stack(poses), lam_v=1.0)
    assert float(r.norm(dim=1).min()) > 1e-3, r.norm(dim=1)


@pytest.mark.parametrize("dt", [1.0, 0.5])
def test_velocity_jacobian_matches_central_differences(dt: float) -> None:
    """All three 6x6 blocks of every residual against central finite differences of left perturbations
    on random (not small-motion) poses, max abs error < 1e-6 in float64; frames outside a residual's
    3-frame window must not affect it at all."""
    T = 6
    poses = _random_poses(T, seed=5)
    lam_v = 1.7
    _r, J_prev, J_mid, J_next = velocity_term(poses, lam_v, dt=dt)
    blocks = (J_prev, J_mid, J_next)
    h = 1e-6
    max_err = 0.0
    for frame in range(T):
        for col in range(6):
            d = torch.zeros(6, dtype=DT)
            d[col] = h
            pp, pm = poses.clone(), poses.clone()
            pp[frame] = apply_left(poses[frame], d)
            pm[frame] = apply_left(poses[frame], -d)
            num = (velocity_term(pp, lam_v, dt=dt)[0] - velocity_term(pm, lam_v, dt=dt)[0]) / (2 * h)
            for i in range(T - 2):
                offset = frame - i
                ana = blocks[offset][i, :, col] if 0 <= offset <= 2 else torch.zeros(6, dtype=DT)
                max_err = max(max_err, float((num[i] - ana).abs().max()))
    assert max_err < 1e-6, max_err


def test_velocity_dt_scaling_matches_design_units() -> None:
    """`lam_v` is the inverse acceleration-noise density: the per-frame weight is lam_v / dt^3 (design
    11.7: lambda_v corresponds to 1/(dt^3 q_c)), so dt=2 with lam_v equals dt=1 with lam_v / 8."""
    poses = _random_poses(7, seed=6)
    a = velocity_term(poses, lam_v=3.0, dt=2.0)
    b = velocity_term(poses, lam_v=3.0 / 8.0, dt=1.0)
    for x, y in zip(a, b, strict=True):
        assert torch.allclose(x, y, atol=1e-14)


@pytest.mark.parametrize("T", [1, 2])
def test_velocity_term_is_empty_below_three_frames(T: int) -> None:
    r, J_prev, J_mid, J_next = velocity_term(_random_poses(T, seed=T), lam_v=1.0)
    assert r.shape == (0, 6)
    assert J_prev.shape == J_mid.shape == J_next.shape == (0, 6, 6)
    assert r.dtype == DT
