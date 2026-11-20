"""Block-tridiagonal normal equations (design Section 11.6).

`assemble_block_tridiagonal` replaces `assemble_normal_equations`'s Python loop over frames with a
handful of batched contractions; `solve_block_tridiagonal` replaces the dense O(n^3) factorization with
a block-Thomas (block-LDL^T) recursion plus a rank-one Schur complement for the shared scale parameter;
`marginal_covariances` replaces the dense pseudo-inverse with the O(T) backward (Rauch-Tung-Striebel)
covariance recursion. `densify` turns a `BlockTridiagonal` back into the dense `(H, g)` pair so these can
be checked against `torch.linalg.solve`/`torch.linalg.inv` on the equivalent dense system.
"""

from __future__ import annotations

import dataclasses
import math
import time

import pytest
import torch
from torch import Tensor

from fastdo.solver.banded import (
    BlockTridiagonal,
    assemble_block_tridiagonal,
    densify,
    marginal_covariances,
    solve_block_tridiagonal,
)
from fastdo.solver.terms import TermOutput

torch.manual_seed(0)
DT = torch.float64


def _random_problem(T: int, n: int = 5, d: int = 2, seed: int = 0
                    ) -> tuple[list[TermOutput], tuple[Tensor, Tensor, Tensor] | None]:
    g = torch.Generator().manual_seed(seed)
    J = torch.randn(T, n, d, 7, dtype=DT, generator=g)
    r = torch.randn(T, n, d, dtype=DT, generator=g)
    w = torch.rand(T, n, dtype=DT, generator=g)
    terms = [TermOutput(r, J, w)]
    smooth = None
    if T > 1:
        smooth = (torch.randn(T - 1, 6, dtype=DT, generator=g), torch.randn(T - 1, 6, 6, dtype=DT, generator=g),
                  torch.randn(T - 1, 6, 6, dtype=DT, generator=g))
    return terms, smooth


def _dense_assemble_reference(terms: list[TermOutput], smooth: tuple[Tensor, Tensor, Tensor] | None,
                              n_frames: int) -> tuple[Tensor, Tensor]:
    """Independent oracle: the earlier dense assembler, restored verbatim from
    `git show dedcf0b:src/fastdo/solver/lm.py` (the commit `assemble_normal_equations` became a thin
    `densify(assemble_block_tridiagonal(...))` wrapper on top of). Deliberately self-contained -- does
    not import `_term_blocks`/`assemble_block_tridiagonal`/`densify` from `fastdo.solver.banded` at all
    -- so `test_densify_matches_dense_assembly` checks the new code against genuinely independent code,
    not against itself (the previous version of this test compared
    `densify(assemble_block_tridiagonal(...))` against `assemble_normal_equations`, which is now just
    that same call wrapped -- a tautology that would pass even if both were wrong)."""
    def term_blocks(term: TermOutput) -> tuple[Tensor, Tensor]:
        sw = term.weight.clamp_min(0.0).sqrt()
        Jw = (term.jacobian * sw[..., None, None]).flatten(1, 2)
        rw = (term.residual * sw[..., None]).flatten(1, 2)
        Jw = Jw.to(torch.float64)
        rw = rw.to(torch.float64)
        return Jw.transpose(1, 2) @ Jw, (Jw.transpose(1, 2) @ rw.unsqueeze(-1)).squeeze(-1)

    T = n_frames
    n = 6 * T + 1
    device = terms[0].residual.device if terms else (smooth[0].device if smooth else torch.device("cpu"))
    H = torch.zeros(n, n, dtype=torch.float64, device=device)
    g = torch.zeros(n, dtype=torch.float64, device=device)
    for term in terms:
        Hk, gk = term_blocks(term)
        for k in range(T):
            sl = slice(6 * k, 6 * k + 6)
            H[sl, sl] += Hk[k, :6, :6]
            H[sl, n - 1] += Hk[k, :6, 6]
            H[n - 1, sl] += Hk[k, 6, :6]
            H[n - 1, n - 1] += Hk[k, 6, 6]
            g[sl] += gk[k, :6]
            g[n - 1] += gk[k, 6]
    if smooth is not None:
        r, Jk, Jk1 = (x.to(torch.float64) for x in smooth)
        for k in range(T - 1):
            a, b = slice(6 * k, 6 * k + 6), slice(6 * k + 6, 6 * k + 12)
            H[a, a] += Jk[k].T @ Jk[k]
            H[a, b] += Jk[k].T @ Jk1[k]
            H[b, a] += Jk1[k].T @ Jk[k]
            H[b, b] += Jk1[k].T @ Jk1[k]
            g[a] += Jk[k].T @ r[k]
            g[b] += Jk1[k].T @ r[k]
    return H, g


@pytest.mark.parametrize("T", [1, 2, 7, 40])
def test_densify_matches_dense_assembly(T: int) -> None:
    """`densify(assemble_block_tridiagonal(...))` against `_dense_assemble_reference` (the restored
    earlier loop assembler, not `assemble_normal_equations` -- see that function's docstring for why
    comparing against the latter would be circular)."""
    terms, smooth = _random_problem(T, seed=T)
    H_dense, g_dense = _dense_assemble_reference(terms, smooth, T)
    sys = assemble_block_tridiagonal(terms, smooth, T)
    H_block, g_block = densify(sys)
    assert torch.allclose(H_block, H_dense, atol=1e-12)
    assert torch.allclose(g_block, g_dense, atol=1e-12)


@pytest.mark.parametrize("T", [1, 2, 7, 40])
@pytest.mark.parametrize("lam", [0.0, 1e-3, 1.0])
def test_solve_matches_dense_solve(T: int, lam: float) -> None:
    terms, smooth = _random_problem(T, seed=100 + T)
    sys = assemble_block_tridiagonal(terms, smooth, T)
    H, g = densify(sys)
    n = H.shape[0]
    damped = H + lam * torch.diag(torch.diagonal(H)) + 1e-9 * torch.eye(n, dtype=DT)
    delta_dense = torch.linalg.solve(damped, -g)
    delta_pose, delta_sigma = solve_block_tridiagonal(sys, lam)
    delta_block = torch.cat([delta_pose.reshape(-1), delta_sigma.reshape(1)])
    assert torch.allclose(delta_block, delta_dense, rtol=1e-9, atol=1e-9), (delta_block - delta_dense).abs().max()


@pytest.mark.parametrize("T", [1, 2, 7, 40])
def test_marginal_covariances_match_dense_inverse_blocks(T: int) -> None:
    terms, smooth = _random_problem(T, seed=200 + T)
    sys = assemble_block_tridiagonal(terms, smooth, T)
    H, _g = densify(sys)
    n = H.shape[0]
    Hinv = torch.linalg.inv(H + 1e-9 * torch.eye(n, dtype=DT))
    cov = marginal_covariances(sys, lam=0.0, floor=1e-9)
    assert cov.shape == (T, 6, 6)
    for k in range(T):
        sl = slice(6 * k, 6 * k + 6)
        assert torch.allclose(cov[k], Hinv[sl, sl], rtol=1e-9, atol=1e-9), (
            k, (cov[k] - Hinv[sl, sl]).abs().max())


def test_degenerate_unobserved_frame_with_smoothness_solves_without_nan() -> None:
    """A frame with all-zero data blocks (fully unobserved) but smoothness on both sides: the solve and
    covariance must stay finite. Unlike the no-smoothness case below, this frame's covariance block has
    no closed form to check against here: the RTS backward recursion couples every frame to its
    neighbours through `M_k Sigma_{k+1} M_k^T`, so `Sigma[unobserved]` depends on the (arbitrary, random)
    blocks at every later frame too, not just its own local diagonal block -- so this test only checks
    finiteness for the solve/covariance, plus (in closed form) that the pre-solve *input* block
    `sys.D[unobserved]` equals exactly the two smoothness-only contributions, with zero leakage from the
    zeroed data term. (`test_degenerate_unobserved_frame_without_smoothness_floor_dominates` turns
    smoothness off too, which decouples the system completely, so *that* covariance block does have the
    closed form `I/floor` -- checked there.)"""
    T = 5
    unobserved = 2
    terms, smooth = _random_problem(T, seed=7)
    _r, Jk, Jk1 = smooth
    zeroed_terms = [TermOutput(t.residual.clone(), t.jacobian.clone(), t.weight.clone()) for t in terms]
    for t in zeroed_terms:
        t.weight[unobserved] = 0.0
    sys = assemble_block_tridiagonal(zeroed_terms, smooth, T)
    delta_pose, delta_sigma = solve_block_tridiagonal(sys, lam=1e-2)
    assert torch.isfinite(delta_pose).all() and torch.isfinite(delta_sigma).all()
    cov = marginal_covariances(sys, lam=0.0, floor=1e-9)
    assert torch.isfinite(cov).all()
    # Reference: assemble the same problem but with the unobserved frame's data-term rows dropped
    # entirely (weight=0 already makes their contribution exactly zero, so this is just documentation
    # that the block at `unobserved` is driven purely by the two smoothness blocks touching it).
    # Frame `unobserved` is the "own" frame of pair k=unobserved (Jk[unobserved]) and the "next" frame
    # of pair k=unobserved-1 (Jk1[unobserved-1]) -- see assemble_block_tridiagonal's D[:-1]/D[1:] split.
    only_smooth_D = Jk[unobserved].T @ Jk[unobserved] + Jk1[unobserved - 1].T @ Jk1[unobserved - 1]
    assert torch.allclose(sys.D[unobserved], only_smooth_D, atol=1e-10)


def test_degenerate_unobserved_frame_without_smoothness_floor_dominates() -> None:
    """With smoothness off entirely and a frame carrying no data (all weights zero), the block-diagonal
    system decouples per frame; the floor keeps that frame's block solvable and its covariance blows up
    to ~1/floor, exactly like the dense pseudo-inverse would (both are governed by the same ridge)."""
    T = 3
    floor = 1e-9
    terms, _ = _random_problem(T, seed=9)
    for t in terms:
        t.weight[:] = 0.0
    sys = assemble_block_tridiagonal(terms, None, T)
    assert torch.allclose(sys.D, torch.zeros_like(sys.D))
    assert sys.U.shape == (T - 1, 6, 6) and torch.allclose(sys.U, torch.zeros_like(sys.U))
    cov = marginal_covariances(sys, lam=0.0, floor=floor)
    assert torch.isfinite(cov).all()
    expected = torch.eye(6, dtype=DT) / floor
    for k in range(T):
        assert torch.allclose(cov[k], expected, rtol=1e-6), (k, cov[k])


def test_fix_scale_semantics_via_lm_optimize() -> None:
    """`lm_optimize(cfg.fix_scale=True)` must zero the scale row/column and set the diagonal entry to 1
    on the stored system (so delta_sigma solves to 0 and the covariance's scale contribution vanishes),
    exactly like the old dense-path zeroing of H's last row/column."""
    from fastdo.geometry.lie import se3_exp
    from fastdo.solver.lm import LMConfig, lm_optimize
    from fastdo.solver.terms import project, track_term, transform_points

    T, M, J = 3, 60, 12
    Y = torch.randn(M, 3, dtype=DT) * 0.3
    poses_gt = se3_exp(torch.randn(T, 6, dtype=DT) * 0.05)
    poses_gt[:, 2, 3] += 0.8
    ls_gt = torch.tensor(math.log(0.5), dtype=DT)
    K = torch.tensor([[300.0, 0, 160.0], [0, 300.0, 120.0], [0, 0, 1.0]], dtype=DT)
    anchors = Y[:J]
    tracks = project(transform_points(anchors, poses_gt, ls_gt.exp()), K)
    vis = torch.ones(T, J, dtype=torch.bool)

    def evaluate(poses, ls, refresh):
        trk = track_term(anchors, tracks, vis, poses, ls, K, cauchy_c=4.0)
        from fastdo.solver.terms import smoothness_term
        return [trk], smoothness_term(poses, lam_v=1e-4)

    res = lm_optimize(poses_gt, ls_gt, evaluate, LMConfig(max_iter=2, fix_scale=True))
    assert torch.allclose(res.system.c, torch.zeros_like(res.system.c))
    assert float(res.system.h_ss) == 1.0
    assert float(res.system.g_s) == 0.0
    assert float(res.log_scale) == float(ls_gt)  # scale never moved


@pytest.mark.parametrize("T", [1, 40])
def test_solve_and_covariance_handle_zero_frame_batch(T: int) -> None:
    """T=1 (no smoothness pair at all) must not crash the sequential recursion (empty inner loops)."""
    terms, smooth = _random_problem(T, seed=T + 1000)
    sys = assemble_block_tridiagonal(terms, smooth, T)
    delta_pose, delta_sigma = solve_block_tridiagonal(sys, lam=1e-2)
    assert delta_pose.shape == (T, 6)
    assert delta_sigma.shape == ()
    cov = marginal_covariances(sys)
    assert cov.shape == (T, 6, 6)


def _spd_leaf_system(T: int, seed: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """`requires_grad=True` float64 leaf tensors for a well-conditioned SPD block-tridiagonal system:
    `D` built from `J^T J + 0.5*I` so every diagonal block is strictly PD by construction, `U` scaled
    down so the whole (block-tridiagonal) matrix stays PD too (banded.py's module docstring: a Schur
    complement of a PD matrix is itself PD, so a well-conditioned `D`/`U` keeps every Cholesky in
    `_forward_schur` well posed for `gradcheck`'s finite differences)."""
    gen = torch.Generator().manual_seed(seed)
    J = torch.randn(T, 8, 6, dtype=DT, generator=gen)
    D = (J.transpose(-1, -2) @ J + 0.5 * torch.eye(6, dtype=DT)).requires_grad_(True)
    U = (0.05 * torch.randn(max(T - 1, 0), 6, 6, dtype=DT, generator=gen)).requires_grad_(True)
    c = torch.randn(T, 6, dtype=DT, generator=gen).requires_grad_(True)
    g = torch.randn(T, 6, dtype=DT, generator=gen).requires_grad_(True)
    h_ss = torch.tensor(4.0, dtype=DT).requires_grad_(True)
    g_s = torch.tensor(0.3, dtype=DT).requires_grad_(True)
    return D, U, c, g, h_ss, g_s


def test_solve_block_tridiagonal_is_differentiable() -> None:
    """`solve_block_tridiagonal` must actually be differentiable end-to-end --
    design 11.6 explicitly requires the block-Thomas recursion to be plain torch (not scipy) so later
    code can differentiate through it, and the pre-fix code silently broke that promise with a
    `.detach()` on every input. Verify with `gradcheck` now instead of assuming it.

    `D` is symmetrized inside the differentiated closure (`0.5*(D+D^T)`) rather than passed to
    `gradcheck` raw: `torch.linalg.cholesky` (used throughout `_forward_schur`) only reads one triangle
    of its input and its backward implicitly assumes a symmetric input, so `gradcheck`'s independent
    per-element finite-difference perturbations of a raw (non-reparametrized) `D` would probe direction
    that break that assumption and spuriously fail even though the code is correct -- a standard gotcha
    for gradchecking any symmetric-matrix-consuming function (cholesky/eig/etc.), not a bug in
    `solve_block_tridiagonal` itself. At the actual (unperturbed) point `D` is already exactly symmetric
    by construction (`_spd_leaf_system`), so `0.5*(D+D^T) == D` there and nothing about the tested
    function's behavior changes."""
    D, U, c, g, h_ss, g_s = _spd_leaf_system(T=3, seed=123)

    def solve_fn(D: Tensor, U: Tensor, c: Tensor, g: Tensor, h_ss: Tensor, g_s: Tensor
                ) -> tuple[Tensor, Tensor]:
        D_sym = 0.5 * (D + D.transpose(-1, -2))
        sys = BlockTridiagonal(D=D_sym, U=U, c=c, h_ss=h_ss, g=g, g_s=g_s)
        return solve_block_tridiagonal(sys, lam=1e-2)

    assert torch.autograd.gradcheck(solve_fn, (D, U, c, g, h_ss, g_s))


def test_marginal_covariances_is_differentiable() -> None:
    """Same as above for `marginal_covariances`; checks the trace of the returned (T,6,6) covariance
    (a scalar, as `gradcheck` needs) w.r.t. the inputs that actually affect it (`D`, `U`, `c`, `h_ss`;
    `g`/`g_s` don't feed the covariance at all, so they are left as fixed context, not differentiated).
    `D` is symmetrized inside the closure for the same reason as the solve test above."""
    D, U, c, g, h_ss, g_s = _spd_leaf_system(T=3, seed=321)

    def cov_trace_fn(D: Tensor, U: Tensor, c: Tensor, h_ss: Tensor) -> Tensor:
        D_sym = 0.5 * (D + D.transpose(-1, -2))
        sys = BlockTridiagonal(D=D_sym, U=U, c=c, h_ss=h_ss, g=g, g_s=g_s)
        cov = marginal_covariances(sys)
        return cov.diagonal(dim1=-2, dim2=-1).sum()

    assert torch.autograd.gradcheck(cov_trace_fn, (D, U, c, h_ss))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_solve_and_covariance_match_cpu_when_system_is_on_cuda() -> None:
    """`solve_block_tridiagonal`/`marginal_covariances` move blocks to CPU internally (design 11.6 item
    2: the sequential recursion is faster there) and move results back to the caller's device --
    verify the returned tensors actually land back on CUDA and match a CPU-resident run of the identical
    problem to 1e-9."""
    T = 7
    terms, smooth = _random_problem(T, seed=42)
    cuda_terms = [TermOutput(t.residual.cuda(), t.jacobian.cuda(), t.weight.cuda()) for t in terms]
    cuda_smooth = (smooth[0].cuda(), smooth[1].cuda(), smooth[2].cuda()) if smooth is not None else None
    sys_cpu = assemble_block_tridiagonal(terms, smooth, T)
    sys_cuda = assemble_block_tridiagonal(cuda_terms, cuda_smooth, T)
    assert sys_cuda.D.device.type == "cuda"
    dp_cpu, ds_cpu = solve_block_tridiagonal(sys_cpu, lam=1e-2)
    dp_cuda, ds_cuda = solve_block_tridiagonal(sys_cuda, lam=1e-2)
    assert dp_cuda.device.type == "cuda" and ds_cuda.device.type == "cuda"
    assert torch.allclose(dp_cuda.cpu(), dp_cpu, rtol=1e-9, atol=1e-9)
    assert torch.allclose(ds_cuda.cpu(), ds_cpu, rtol=1e-9, atol=1e-9)
    cov_cpu = marginal_covariances(sys_cpu)
    cov_cuda = marginal_covariances(sys_cuda)
    assert cov_cuda.device.type == "cuda"
    assert torch.allclose(cov_cuda.cpu(), cov_cpu, rtol=1e-9, atol=1e-9)


@pytest.mark.timing
@pytest.mark.skipif(not torch.cuda.is_available(), reason="timing gate needs the GPU")
def test_assembly_and_solve_timing_at_t100() -> None:
    """Design Section 11.6's headline claim: assembly + block-Thomas solve at T=100 with a realistic
    N=1024 pointmap-sized residual block should be a small, roughly constant number of milliseconds
    after warm-up (vs. the old ~30ms Python-loop assembly + ~30ms dense solve). The block-Thomas
    recursion is an inherently sequential Python loop over T on CPU (see module docstring / design
    11.6 item 2), so its floor is dispatch overhead, not FLOPs; the host may also run other
    workloads concurrently, so the threshold below is deliberately generous (an order of magnitude
    above the design's own 3ms target) rather than a tight contention-sensitive gate."""
    device = "cuda"
    T, N = 100, 1024
    gen = torch.Generator(device=device).manual_seed(0)
    J = torch.randn(T, N, 3, 7, dtype=DT, device=device, generator=gen)
    r = torch.randn(T, N, 3, dtype=DT, device=device, generator=gen)
    w = torch.rand(T, N, dtype=DT, device=device, generator=gen)
    terms = [TermOutput(r, J, w)]
    smooth = (torch.randn(T - 1, 6, dtype=DT, device=device, generator=gen),
             torch.randn(T - 1, 6, 6, dtype=DT, device=device, generator=gen),
             torch.randn(T - 1, 6, 6, dtype=DT, device=device, generator=gen))

    def _once() -> tuple[float, float]:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        sys = assemble_block_tridiagonal(terms, smooth, T)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        solve_block_tridiagonal(sys, lam=1e-2)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        return t1 - t0, t2 - t1

    for _ in range(3):
        _once()  # warm-up
    assembly_times = []
    solve_times = []
    for _ in range(10):
        a, s = _once()
        assembly_times.append(a)
        solve_times.append(s)
    assembly_ms = 1000 * min(assembly_times)
    solve_ms = 1000 * min(solve_times)
    print(f"\nassembly={assembly_ms:.3f}ms solve={solve_ms:.3f}ms (min of 10, T={T}, N={N})")
    assert assembly_ms < 30.0, assembly_ms
    assert solve_ms < 30.0, solve_ms


@pytest.mark.timing
@pytest.mark.skipif(not torch.cuda.is_available(), reason="timing gate needs the GPU")
def test_marginal_covariances_timing_at_t100() -> None:
    device = "cuda"
    T, N = 100, 1024
    gen = torch.Generator(device=device).manual_seed(1)
    J = torch.randn(T, N, 3, 7, dtype=DT, device=device, generator=gen)
    r = torch.randn(T, N, 3, dtype=DT, device=device, generator=gen)
    w = torch.rand(T, N, dtype=DT, device=device, generator=gen)
    terms = [TermOutput(r, J, w)]
    smooth = (torch.randn(T - 1, 6, dtype=DT, device=device, generator=gen),
             torch.randn(T - 1, 6, 6, dtype=DT, device=device, generator=gen),
             torch.randn(T - 1, 6, 6, dtype=DT, device=device, generator=gen))
    sys = assemble_block_tridiagonal(terms, smooth, T)
    for _ in range(3):
        marginal_covariances(sys)
        torch.cuda.synchronize()
    times = []
    for _ in range(10):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        marginal_covariances(sys)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    cov_ms = 1000 * min(times)
    print(f"\nmarginal_covariances={cov_ms:.3f}ms (min of 10, T={T}, N={N})")
    assert cov_ms < 20.0, cov_ms


def test_block_tridiagonal_is_frozen_dataclass() -> None:
    terms, smooth = _random_problem(3)
    sys = assemble_block_tridiagonal(terms, smooth, 3)
    assert isinstance(sys, BlockTridiagonal)
    with pytest.raises(dataclasses.FrozenInstanceError):
        sys.D = sys.D  # type: ignore[misc]
