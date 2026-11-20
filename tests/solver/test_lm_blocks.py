"""Tests for `fastdo.solver.lm_blocks`: generic block-tridiagonal LM."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fastdo.solver.lm_blocks import (
    BlockLMConfig,
    BlockTerm,
    SmoothTerm,
    assemble_block_system,
    lm_optimize_blocks,
    solve_block_tridiagonal,
    total_block_cost,
)

torch.manual_seed(0)
DT = torch.float64


def _dense_from_blocks(diag: np.ndarray, offdiag: np.ndarray | None, n: int) -> np.ndarray:
    """Brute-force assembly of the same block-tridiagonal matrix, for a ground-truth comparison."""
    T = diag.shape[0]
    A = np.zeros((T * n, T * n))
    for k in range(T):
        A[k * n:(k + 1) * n, k * n:(k + 1) * n] = diag[k]
    if offdiag is not None:
        for k in range(T - 1):
            A[k * n:(k + 1) * n, (k + 1) * n:(k + 2) * n] = offdiag[k]
            A[(k + 1) * n:(k + 2) * n, k * n:(k + 1) * n] = offdiag[k].T
    return A


def _random_problem(T: int, n: int, R: int) -> tuple[list[BlockTerm], SmoothTerm]:
    terms = [
        BlockTerm(
            residual=torch.randn(T, R, dtype=DT),
            jacobian=torch.randn(T, R, n, dtype=DT),
            weight=torch.rand(T, R, dtype=DT),
        ),
        BlockTerm(
            residual=torch.randn(T, R, dtype=DT),
            jacobian=torch.randn(T, R, n, dtype=DT),
            weight=torch.ones(T, R, dtype=DT),
        ),
    ]
    smooth = SmoothTerm(
        residual=torch.randn(T - 1, n, dtype=DT),
        jac_k=torch.randn(T - 1, n, n, dtype=DT),
        jac_k1=torch.randn(T - 1, n, n, dtype=DT),
        weight=torch.rand(T - 1, n, dtype=DT),
    )
    return terms, smooth


def test_assemble_matches_brute_force_dense_blocks() -> None:
    T, n, R = 6, 5, 8
    terms, smooth = _random_problem(T, n, R)
    diag, offdiag, g = assemble_block_system(terms, smooth, T, n)
    assert diag.shape == (T, n, n)
    assert offdiag is not None and offdiag.shape == (T - 1, n, n)
    assert g.shape == (T, n)

    A = _dense_from_blocks(diag.numpy(), offdiag.numpy(), n)
    # brute-force J^T W J / J^T W r over the *global* flattened parameter vector
    M = T * n
    Jg = torch.zeros(0, M, dtype=DT)
    rg = torch.zeros(0, dtype=DT)
    for term in terms:
        for k in range(T):
            sw = term.weight[k].sqrt()
            block = torch.zeros(R, M, dtype=DT)
            block[:, k * n:(k + 1) * n] = term.jacobian[k] * sw[:, None]
            Jg = torch.cat([Jg, block], dim=0)
            rg = torch.cat([rg, term.residual[k] * sw], dim=0)
    for k in range(T - 1):
        sw = smooth.weight[k].sqrt()
        block = torch.zeros(n, M, dtype=DT)
        block[:, k * n:(k + 1) * n] = smooth.jac_k[k] * sw[:, None]
        block[:, (k + 1) * n:(k + 2) * n] = smooth.jac_k1[k] * sw[:, None]
        Jg = torch.cat([Jg, block], dim=0)
        rg = torch.cat([rg, smooth.residual[k] * sw], dim=0)
    A_brute = (Jg.T @ Jg).numpy()
    g_brute = (Jg.T @ rg).numpy()
    assert np.allclose(A, A_brute, atol=1e-9)
    assert np.allclose(g.numpy().reshape(-1), g_brute, atol=1e-9)


def test_banded_solve_matches_dense_solve() -> None:
    T, n, R = 7, 4, 6
    terms, smooth = _random_problem(T, n, R)
    diag, offdiag, g = assemble_block_system(terms, smooth, T, n)
    lam = 0.37

    delta_banded = solve_block_tridiagonal(diag, offdiag, g, lam).numpy()

    assert offdiag is not None
    A = _dense_from_blocks(diag.numpy(), offdiag.numpy(), n)
    idx = np.arange(A.shape[0])
    A_damped = A.copy()
    A_damped[idx, idx] += lam * np.diag(A) + 1e-9
    delta_dense = np.linalg.solve(A_damped, -g.numpy().reshape(-1)).reshape(T, n)

    assert np.allclose(delta_banded, delta_dense, atol=1e-9)


def test_banded_solve_with_no_smoothness_term_matches_dense() -> None:
    T, n, R = 5, 3, 4
    terms, _ = _random_problem(T, n, R)
    diag, offdiag, g = assemble_block_system(terms, None, T, n)
    assert offdiag is None

    delta_banded = solve_block_tridiagonal(diag, offdiag, g, lam=1.0).numpy()
    A = _dense_from_blocks(diag.numpy(), None, n)
    idx = np.arange(A.shape[0])
    A_damped = A.copy()
    A_damped[idx, idx] += 1.0 * np.diag(A) + 1e-9
    delta_dense = np.linalg.solve(A_damped, -g.numpy().reshape(-1)).reshape(T, n)
    assert np.allclose(delta_banded, delta_dense, atol=1e-9)


def test_lm_optimize_blocks_converges_on_quadratic_target() -> None:
    """r_k(x) = x_k - target_k (identity Jacobian): should converge in one damped Newton step."""
    T, n = 5, 3
    target = torch.randn(T, n, dtype=DT)
    eye = torch.eye(n, dtype=DT).expand(T, n, n)

    def evaluate(x: torch.Tensor, refresh: bool) -> tuple[list[BlockTerm], None]:
        r = x - target
        term = BlockTerm(residual=r, jacobian=eye, weight=torch.ones(T, n, dtype=DT))
        return [term], None

    x0 = torch.zeros(T, n, dtype=DT)
    res = lm_optimize_blocks(x0, evaluate, BlockLMConfig(max_iter=20))
    assert torch.allclose(res.x, target, atol=1e-6)
    assert res.cost < 1e-10


def test_lm_optimize_blocks_reduces_cost_with_smoothness_and_refresh_flag() -> None:
    """A term whose Jacobian depends on a 'plan' cached only on refresh=True (mirrors the OT term)."""
    T, n = 6, 2
    target = torch.randn(T, n, dtype=DT) * 0.5
    state: dict[str, torch.Tensor] = {}

    def evaluate(x: torch.Tensor, refresh: bool) -> tuple[list[BlockTerm], SmoothTerm]:
        if refresh or "scale" not in state:
            state["scale"] = 1.0 + 0.0 * x.sum()  # a fixed-point "plan" analogue, refreshed on accept
        scale = state["scale"]
        r = scale * (x - target)
        jac = (scale * torch.eye(n, dtype=DT)).expand(T, n, n)
        term = BlockTerm(residual=r, jacobian=jac, weight=torch.ones(T, n, dtype=DT))
        sr = 0.1 * (x[1:] - x[:-1])
        jk = (-0.1 * torch.eye(n, dtype=DT)).expand(T - 1, n, n)
        jk1 = (0.1 * torch.eye(n, dtype=DT)).expand(T - 1, n, n)
        smooth = SmoothTerm(residual=sr, jac_k=jk, jac_k1=jk1, weight=torch.ones(T - 1, n, dtype=DT))
        return [term], smooth

    x0 = torch.zeros(T, n, dtype=DT)
    cost0 = total_block_cost(*evaluate(x0, True))
    # This problem is linear in x (constant Jacobians), so a single *undamped* Newton step from
    # x0 lands exactly at the joint least-squares optimum -- the reference this test checks
    # `lm_optimize_blocks` converges to (with enough iterations and a tight rel_tol so LM's usual
    # early-stop-on-small-relative-improvement doesn't cut it off first).
    diag0, offdiag0, g0 = assemble_block_system(*evaluate(x0, True), T, n)
    x_star = x0 + solve_block_tridiagonal(diag0, offdiag0, g0, lam=0.0)
    true_cost = total_block_cost(*evaluate(x_star, True))

    res = lm_optimize_blocks(x0, evaluate, BlockLMConfig(max_iter=50, rel_tol=1e-12))
    assert res.cost < cost0
    assert res.cost == pytest.approx(true_cost, abs=1e-8)


def test_lm_optimize_blocks_is_deterministic_given_a_fixed_seed() -> None:
    torch.manual_seed(123)
    T, n = 4, 3
    target = torch.randn(T, n, dtype=DT)
    eye = torch.eye(n, dtype=DT).expand(T, n, n)

    def evaluate(x: torch.Tensor, refresh: bool) -> tuple[list[BlockTerm], None]:
        r = x - target
        return [BlockTerm(residual=r, jacobian=eye, weight=torch.ones(T, n, dtype=DT))], None

    x0 = torch.randn(T, n, dtype=DT) * 0.1
    res1 = lm_optimize_blocks(x0.clone(), evaluate, BlockLMConfig(max_iter=10))
    res2 = lm_optimize_blocks(x0.clone(), evaluate, BlockLMConfig(max_iter=10))
    assert torch.equal(res1.x, res2.x)
    assert res1.iterations == res2.iterations


def _banded_reference(diag: np.ndarray, offdiag: np.ndarray | None, n: int, lam: float) -> np.ndarray:
    """The pre-C3c `_banded_from_blocks` construction, kept verbatim as the reference for the index cache."""
    T = diag.shape[0]
    M = T * n
    u = 2 * n - 1
    ab = np.zeros((u + 1, M), dtype=np.float64)
    iu, ju = np.triu_indices(n)
    frame = np.arange(T)[:, None]
    rows = frame * n + iu[None, :]
    cols = frame * n + ju[None, :]
    damped = diag.copy()
    diag_idx = np.arange(n)
    damped[:, diag_idx, diag_idx] += lam * diag[:, diag_idx, diag_idx] + 1e-9
    ab[u + rows - cols, cols] = damped[:, iu, ju]
    if offdiag is not None:
        ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        ii_flat, jj_flat = ii.ravel(), jj.ravel()
        kframe = np.arange(T - 1)[:, None]
        off_rows = kframe * n + ii_flat[None, :]
        off_cols = kframe * n + n + jj_flat[None, :]
        ab[u + off_rows - off_cols, off_cols] = offdiag.reshape(T - 1, n * n)
    return ab


def test_banded_index_cache_reproduces_banded_storage() -> None:
    from fastdo.solver.lm_blocks import _banded_from_blocks, _banded_indices

    rng = np.random.default_rng(0)
    for T, n in ((17, 5), (138, 28)):
        diag, off = rng.normal(size=(T, n, n)), rng.normal(size=(T - 1, n, n))
        for _ in range(2):                                            # second call reads the cached indices
            assert np.array_equal(_banded_from_blocks(diag, off, n, 0.37), _banded_reference(diag, off, n, 0.37))
        assert np.array_equal(_banded_from_blocks(diag, None, n, 0.37), _banded_reference(diag, None, n, 0.37))
        assert _banded_indices(T, n) is _banded_indices(T, n)


def test_single_thread_blas_solve_matches_dense() -> None:
    T, n, R = 17, 5, 8
    terms, smooth = _random_problem(T, n, R)
    diag, offdiag, g = assemble_block_system(terms, smooth, T, n)
    lam = 0.37
    delta = solve_block_tridiagonal(diag, offdiag, g, lam, blas_threads=1).numpy()
    assert offdiag is not None
    A = _dense_from_blocks(diag.numpy(), offdiag.numpy(), n)
    idx = np.arange(A.shape[0])
    A[idx, idx] += lam * np.diag(A) + 1e-9
    dense = np.linalg.solve(A, -g.numpy().reshape(-1)).reshape(T, n)
    assert np.allclose(delta, dense, atol=1e-9)


def test_final_refresh_false_returns_the_same_state() -> None:
    """Skipping the refresh after the loop-ending accepted step changes neither `x` nor the
    iteration count; `on_final_accept` receives the returned state instead."""
    torch.manual_seed(0)
    T, R, n = 6, 5, 3
    A = torch.randn(T, R, n, dtype=DT)
    b = torch.randn(T, R, dtype=DT)
    eye = torch.eye(n, dtype=DT)
    refreshes = [0]

    def evaluate(x: torch.Tensor, refresh: bool) -> tuple[list[BlockTerm], SmoothTerm]:
        refreshes[0] += int(refresh)
        term = BlockTerm(residual=torch.einsum("trn,tn->tr", A, x) - b, jacobian=A, weight=torch.ones(T, R, dtype=DT))
        smooth = SmoothTerm(residual=0.3 * (x[1:] - x[:-1]), jac_k=(-0.3 * eye).expand(T - 1, n, n),
                            jac_k1=(0.3 * eye).expand(T - 1, n, n), weight=torch.ones(T - 1, n, dtype=DT))
        return [term], smooth

    x0 = torch.zeros(T, n, dtype=DT)
    for max_iter, rel_tol, expected_iterations in ((4, 1e-4, 3), (30, 1.0, 1)):
        refreshes[0] = 0
        full = lm_optimize_blocks(x0, evaluate, BlockLMConfig(max_iter=max_iter, rel_tol=rel_tol))
        full_refreshes, refreshes[0] = refreshes[0], 0
        accepted: list[torch.Tensor] = []
        lean = lm_optimize_blocks(x0, evaluate, BlockLMConfig(max_iter=max_iter, rel_tol=rel_tol, final_refresh=False),
                                  on_final_accept=accepted.append)
        assert torch.equal(full.x, lean.x)
        assert full.iterations == lean.iterations == expected_iterations
        assert full_refreshes - refreshes[0] == 1
        assert len(accepted) == 1 and accepted[0] is lean.x


def test_final_refresh_false_ending_on_the_last_iteration_returns_the_same_state() -> None:
    """The `it == max_iter` exit: `rel_tol=0` lets no accepted step end the loop early (every
    accepted step has `rel > 0`) and `lam0=10` keeps this linear problem far from convergence, so both paths accept
    all `max_iter` trials, the last one included. As `lm_optimize_blocks` specifies, `x` and `iterations` are
    unchanged; the lean `cost` is the last trial's, which equals the refreshed cost because this `evaluate` does not
    depend on `refresh`; the lean path skips exactly that one refresh and hands the returned tensor (the last trial)
    to `on_final_accept`."""
    torch.manual_seed(1)
    T, R, n, max_iter = 6, 5, 3, 4
    A = torch.randn(T, R, n, dtype=DT)
    b = torch.randn(T, R, dtype=DT)
    eye = torch.eye(n, dtype=DT)
    calls: list[tuple[bool, torch.Tensor]] = []

    def evaluate(x: torch.Tensor, refresh: bool) -> tuple[list[BlockTerm], SmoothTerm]:
        calls.append((refresh, x))
        term = BlockTerm(residual=torch.einsum("trn,tn->tr", A, x) - b, jacobian=A, weight=torch.ones(T, R, dtype=DT))
        smooth = SmoothTerm(residual=0.3 * (x[1:] - x[:-1]), jac_k=(-0.3 * eye).expand(T - 1, n, n),
                            jac_k1=(0.3 * eye).expand(T - 1, n, n), weight=torch.ones(T - 1, n, dtype=DT))
        return [term], smooth

    x0 = torch.zeros(T, n, dtype=DT)
    full = lm_optimize_blocks(x0, evaluate, BlockLMConfig(max_iter=max_iter, lam0=10.0, rel_tol=0.0))
    full_calls = [refresh for refresh, _ in calls]
    calls.clear()
    accepted: list[torch.Tensor] = []
    lean = lm_optimize_blocks(x0, evaluate,
                              BlockLMConfig(max_iter=max_iter, lam0=10.0, rel_tol=0.0, final_refresh=False),
                              on_final_accept=accepted.append)
    lean_calls = [refresh for refresh, _ in calls]
    assert full_calls == [True] + [False, True] * max_iter                   # every trial accepted and refreshed
    assert lean_calls == [True] + [False, True] * (max_iter - 1) + [False]   # the last accepted trial is not
    assert full.iterations == lean.iterations == max_iter
    assert torch.equal(full.x, lean.x)
    assert lean.cost == full.cost
    assert len(accepted) == 1 and accepted[0] is lean.x and calls[-1][1] is lean.x
