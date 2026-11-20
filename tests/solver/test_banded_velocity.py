"""Normal equations with the constant-velocity prior.

The second-difference residual couples frames (k-1, k, k+1), so the pose block of H gains (k, k+2)
blocks (`BlockTridiagonal.U2`) and is block-pentadiagonal in 6x6 blocks. `solve_block_tridiagonal` and
`marginal_covariances` regroup consecutive frame pairs into 12x12 super-blocks (odd T padded with an
identity dummy frame), which is exactly block-tridiagonal again. Every check here is against an
independent brute-force oracle that stacks the global Jacobian row by row (no import of the assembly
code under test), on odd and even T, with a live scale column.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor

from fastdo.solver.banded import (
    BlockTridiagonal,
    Priors,
    assemble_block_tridiagonal,
    densify,
    marginal_covariances,
    solve_block_tridiagonal,
)
from fastdo.solver.terms import TermOutput

DT = torch.float64
Smooth3 = tuple[Tensor, Tensor, Tensor]
Smooth4 = tuple[Tensor, Tensor, Tensor, Tensor]
T_VALUES = [1, 2, 3, 4, 7, 8, 41]


def _random_problem(T: int, seed: int, with_smooth: bool = True
                    ) -> tuple[list[TermOutput], Smooth3 | None, Smooth4]:
    g = torch.Generator().manual_seed(seed)
    n, d = 5, 2
    terms = [TermOutput(torch.randn(T, n, d, dtype=DT, generator=g), torch.randn(T, n, d, 7, dtype=DT, generator=g),
                        torch.rand(T, n, dtype=DT, generator=g))]
    smooth = None
    if with_smooth and T > 1:
        smooth = (torch.randn(T - 1, 6, dtype=DT, generator=g), torch.randn(T - 1, 6, 6, dtype=DT, generator=g),
                  torch.randn(T - 1, 6, 6, dtype=DT, generator=g))
    m = max(T - 2, 0)
    velocity = (torch.randn(m, 6, dtype=DT, generator=g), torch.randn(m, 6, 6, dtype=DT, generator=g),
                torch.randn(m, 6, 6, dtype=DT, generator=g), torch.randn(m, 6, 6, dtype=DT, generator=g))
    return terms, smooth, velocity


def _dense_oracle(terms: list[TermOutput], smooth: Smooth3 | None, velocity: Smooth4 | None, T: int
                  ) -> tuple[Tensor, Tensor]:
    """Stack every weighted residual row into a global (rows, 6T+1) Jacobian; H = J^T J, g = J^T r."""
    n = 6 * T + 1
    rows_J: list[Tensor] = []
    rows_r: list[Tensor] = []
    for term in terms:
        sw = term.weight.clamp_min(0.0).sqrt()
        for k in range(T):
            Jk = (term.jacobian[k] * sw[k, :, None, None]).reshape(-1, 7)
            Jg = torch.zeros(Jk.shape[0], n, dtype=DT)
            Jg[:, 6 * k:6 * k + 6] = Jk[:, :6]
            Jg[:, n - 1] = Jk[:, 6]
            rows_J.append(Jg)
            rows_r.append((term.residual[k] * sw[k, :, None]).reshape(-1))
    if smooth is not None:
        r, Jk, Jk1 = smooth
        for k in range(T - 1):
            Jg = torch.zeros(6, n, dtype=DT)
            Jg[:, 6 * k:6 * k + 6] = Jk[k]
            Jg[:, 6 * k + 6:6 * k + 12] = Jk1[k]
            rows_J.append(Jg)
            rows_r.append(r[k])
    if velocity is not None:
        rv, A, B, C = velocity
        for i in range(rv.shape[0]):
            Jg = torch.zeros(6, n, dtype=DT)
            for off, blk in enumerate((A, B, C)):
                Jg[:, 6 * (i + off):6 * (i + off) + 6] = blk[i]
            rows_J.append(Jg)
            rows_r.append(rv[i])
    J = torch.cat(rows_J, dim=0)
    rr = torch.cat(rows_r, dim=0)
    return J.T @ J, J.T @ rr


@pytest.mark.parametrize("T", T_VALUES)
@pytest.mark.parametrize("with_smooth", [True, False])
def test_densify_with_velocity_matches_brute_force(T: int, with_smooth: bool) -> None:
    terms, smooth, velocity = _random_problem(T, seed=10 + T, with_smooth=with_smooth)
    sys = assemble_block_tridiagonal(terms, Priors(smooth=smooth, velocity=velocity), T)
    assert sys.U2 is not None and sys.U2.shape == (max(T - 2, 0), 6, 6)
    H_ref, g_ref = _dense_oracle(terms, smooth, velocity, T)
    H, g = densify(sys)
    assert torch.allclose(H, H_ref, atol=1e-12), (H - H_ref).abs().max()
    assert torch.allclose(g, g_ref, atol=1e-12), (g - g_ref).abs().max()


@pytest.mark.parametrize("T", T_VALUES)
@pytest.mark.parametrize("lam", [0.0, 1e-3, 1.0])
def test_paired_solve_matches_dense_solve(T: int, lam: float) -> None:
    terms, smooth, velocity = _random_problem(T, seed=100 + T)
    sys = assemble_block_tridiagonal(terms, Priors(smooth=smooth, velocity=velocity), T)
    H, g = _dense_oracle(terms, smooth, velocity, T)
    n = H.shape[0]
    damped = H + lam * torch.diag(torch.diagonal(H)) + 1e-9 * torch.eye(n, dtype=DT)
    delta_dense = torch.linalg.solve(damped, -g)
    delta_pose, delta_sigma = solve_block_tridiagonal(sys, lam)
    assert delta_pose.shape == (T, 6) and delta_sigma.shape == ()
    delta = torch.cat([delta_pose.reshape(-1), delta_sigma.reshape(1)])
    assert torch.allclose(delta, delta_dense, rtol=1e-9, atol=1e-9), (delta - delta_dense).abs().max()


@pytest.mark.parametrize("T", T_VALUES)
def test_paired_marginal_covariances_match_dense_inverse(T: int) -> None:
    terms, smooth, velocity = _random_problem(T, seed=200 + T)
    sys = assemble_block_tridiagonal(terms, Priors(smooth=smooth, velocity=velocity), T)
    H, _ = _dense_oracle(terms, smooth, velocity, T)
    lam = 1e-2
    n = H.shape[0]
    Hinv = torch.linalg.inv(H + lam * torch.diag(torch.diagonal(H)) + 1e-9 * torch.eye(n, dtype=DT))
    cov = marginal_covariances(sys, lam=lam, floor=1e-9)
    assert cov.shape == (T, 6, 6)
    for k in range(T):
        sl = slice(6 * k, 6 * k + 6)
        assert torch.allclose(cov[k], Hinv[sl, sl], rtol=1e-9, atol=1e-9), (k, (cov[k] - Hinv[sl, sl]).abs().max())


def test_priors_without_velocity_is_the_tridiagonal_system() -> None:
    """`Priors(smooth=s)` and a bare `Smooth` triple assemble the identical 6x6-block system (U2 None),
    so every caller without a velocity prior keeps the unchanged block-Thomas path."""
    terms, smooth, _ = _random_problem(9, seed=3)
    a = assemble_block_tridiagonal(terms, smooth, 9)
    b = assemble_block_tridiagonal(terms, Priors(smooth=smooth), 9)
    assert a.U2 is None and b.U2 is None
    for field in ("D", "U", "c", "g", "h_ss", "g_s"):
        assert torch.equal(getattr(a, field), getattr(b, field)), field


def test_zero_velocity_blocks_reproduce_the_tridiagonal_solution() -> None:
    """An all-zero U2 routes through the paired 12x12 recursion and must reproduce the 6x6 path."""
    terms, smooth, _ = _random_problem(11, seed=4)
    tri = assemble_block_tridiagonal(terms, smooth, 11)
    penta = BlockTridiagonal(D=tri.D, U=tri.U, c=tri.c, h_ss=tri.h_ss, g=tri.g, g_s=tri.g_s,
                             U2=torch.zeros(9, 6, 6, dtype=DT))
    dp_a, ds_a = solve_block_tridiagonal(tri, 1e-2)
    dp_b, ds_b = solve_block_tridiagonal(penta, 1e-2)
    assert torch.allclose(dp_a, dp_b, rtol=1e-10, atol=1e-10)
    assert torch.allclose(ds_a, ds_b, rtol=1e-10, atol=1e-10)
    assert torch.allclose(marginal_covariances(tri), marginal_covariances(penta), rtol=1e-9, atol=1e-9)


def test_paired_solve_is_differentiable() -> None:
    """The regrouping (padding + block concatenation) must not break the banded solve's autograd guarantee."""
    T = 3
    g = torch.Generator().manual_seed(7)
    J = torch.randn(T, 8, 6, dtype=DT, generator=g)
    D = (J.transpose(-1, -2) @ J + 0.5 * torch.eye(6, dtype=DT)).requires_grad_(True)
    U = (0.05 * torch.randn(T - 1, 6, 6, dtype=DT, generator=g)).requires_grad_(True)
    U2 = (0.05 * torch.randn(T - 2, 6, 6, dtype=DT, generator=g)).requires_grad_(True)
    c = torch.randn(T, 6, dtype=DT, generator=g).requires_grad_(True)
    gv = torch.randn(T, 6, dtype=DT, generator=g).requires_grad_(True)
    h_ss = torch.tensor(4.0, dtype=DT, requires_grad=True)
    g_s = torch.tensor(0.3, dtype=DT, requires_grad=True)

    def fn(D: Tensor, U: Tensor, U2: Tensor, c: Tensor, gv: Tensor, h_ss: Tensor, g_s: Tensor
           ) -> tuple[Tensor, Tensor]:
        sys = BlockTridiagonal(D=0.5 * (D + D.transpose(-1, -2)), U=U, c=c, h_ss=h_ss, g=gv, g_s=g_s, U2=U2)
        return solve_block_tridiagonal(sys, lam=1e-2)

    assert torch.autograd.gradcheck(fn, (D, U, U2, c, gv, h_ss, g_s))


def test_paired_marginal_covariances_are_differentiable() -> None:
    """Autograd through `marginal_covariances` on the paired path (odd T, so the identity
    dummy frame, the 12x12 slicing back to per-frame 6x6 blocks and the scale's rank-one term are all on
    the graph), matching the banded solve's guarantee for the 6x6 path."""
    T = 3
    g = torch.Generator().manual_seed(9)
    J = torch.randn(T, 8, 6, dtype=DT, generator=g)
    D = (J.transpose(-1, -2) @ J + 0.5 * torch.eye(6, dtype=DT)).requires_grad_(True)
    U = (0.05 * torch.randn(T - 1, 6, 6, dtype=DT, generator=g)).requires_grad_(True)
    U2 = (0.05 * torch.randn(T - 2, 6, 6, dtype=DT, generator=g)).requires_grad_(True)
    c = (0.3 * torch.randn(T, 6, dtype=DT, generator=g)).requires_grad_(True)
    h_ss = torch.tensor(4.0, dtype=DT, requires_grad=True)

    def fn(D: Tensor, U: Tensor, U2: Tensor, c: Tensor, h_ss: Tensor) -> Tensor:
        sys = BlockTridiagonal(D=0.5 * (D + D.transpose(-1, -2)), U=U, c=c, h_ss=h_ss, g=torch.zeros_like(c),
                               g_s=torch.zeros((), dtype=DT), U2=U2)
        return marginal_covariances(sys, lam=1e-2)

    assert torch.autograd.gradcheck(fn, (D, U, U2, c, h_ss))


def test_scipy_banded_reference_rejects_velocity_system() -> None:
    """`lm_step(solver="banded")` assumes bandwidth 11; a velocity-prior system reaches
    offset 17, so it must raise instead of silently dropping the (k, k+2) blocks. The dense reference still
    agrees with the paired block solve."""
    from fastdo.solver.lm import lm_step

    T = 7
    terms, smooth, velocity = _random_problem(T, seed=21)
    sys = assemble_block_tridiagonal(terms, Priors(smooth=smooth, velocity=velocity), T)
    H, g = densify(sys)
    with pytest.raises(ValueError, match="bandwidth"):
        lm_step(H, g, 0.1, solver="banded")
    delta_pose, delta_sigma = solve_block_tridiagonal(sys, 0.1)
    delta = torch.cat([delta_pose.reshape(-1), delta_sigma.reshape(1)])
    assert torch.allclose(lm_step(H, g, 0.1, solver="dense"), delta, rtol=1e-9, atol=1e-9)


def test_lm_optimize_with_velocity_prior_keeps_system_in_sync() -> None:
    """`lm_optimize` accepts an evaluate callback returning `Priors`; `total_cost` counts the velocity
    residual and `LMResult.system` is the paired system re-assembled at the returned state."""
    from fastdo.geometry.lie import se3_exp
    from fastdo.solver.lm import LMConfig, assemble_normal_equations, lm_optimize, total_cost
    from fastdo.solver.terms import (
        project,
        smoothness_term,
        track_term,
        transform_points,
        velocity_term,
    )

    gen = torch.Generator().manual_seed(8)
    T, J = 7, 16
    anchors = torch.randn(J, 3, dtype=DT, generator=gen) * 0.3
    xi = torch.tensor([0.004, -0.003, 0.002, 0.02, 0.01, -0.015], dtype=DT)
    poses_gt = torch.stack([se3_exp(k * xi) for k in range(T)])
    poses_gt[:, 2, 3] += 0.8
    ls = torch.tensor(math.log(0.5), dtype=DT)
    K = torch.tensor([[300.0, 0, 160.0], [0, 300.0, 120.0], [0, 0, 1.0]], dtype=DT)
    tracks = project(transform_points(anchors, poses_gt, ls.exp()), K)
    vis = torch.ones(T, J, dtype=torch.bool)
    vis[2:5] = False                                  # a 3-frame unobserved run, bridged by the priors

    def evaluate(poses: Tensor, log_scale: Tensor, refresh: bool) -> tuple[list[TermOutput], Priors]:
        trk = track_term(anchors, tracks, vis, poses, log_scale, K, cauchy_c=4.0)
        return [trk], Priors(smooth=smoothness_term(poses, 1e-3), velocity=velocity_term(poses, 10.0))

    poses0 = torch.stack([se3_exp(0.01 * torch.randn(6, dtype=DT, generator=gen)) @ p for p in poses_gt])
    res = lm_optimize(poses0, ls, evaluate, LMConfig(max_iter=30, fix_scale=True))
    terms, priors = evaluate(res.poses, res.log_scale, True)
    assert priors.velocity is not None
    vel_cost = float((priors.velocity[0] ** 2).sum())
    assert res.cost == pytest.approx(float(total_cost(terms, priors)))
    assert float(total_cost(terms, priors)) == pytest.approx(float(total_cost(terms, priors.smooth)) + vel_cost)
    assert res.system.U2 is not None
    H_fresh, _ = assemble_normal_equations(terms, priors, T)
    # fix_scale zeroes the stored system's scale row/column, so compare the pose block
    assert torch.allclose(res.H[:-1, :-1], H_fresh[:-1, :-1], rtol=1e-12, atol=1e-12)
    err = (res.poses[:, :3, 3] - poses_gt[:, :3, 3]).norm(dim=1)
    assert float(err.max()) < 1e-3, err
