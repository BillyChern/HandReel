"""Tests for the kNN-truncated Sinkhorn (design 11.5)."""

import time

import numpy as np
import pytest
import torch
import trimesh

from fastdo.solver.ot import plan_barycenters, sinkhorn_unbalanced_potentials
from fastdo.solver.ot_sparse import (
    knn_tail_mass,
    plan_barycenters_sparse,
    sinkhorn_unbalanced_sparse,
)

torch.manual_seed(0)
DT = torch.float64


def _clouds(t: int, n: int, m: int) -> tuple[torch.Tensor, torch.Tensor]:
    y = torch.randn(t, m, 3, dtype=DT)
    x = torch.randn(t, n, 3, dtype=DT)
    return x, y


def _full_idx(t: int, n: int, m: int) -> torch.Tensor:
    """Candidate set = every column (k = M), the exactness configuration."""
    return torch.arange(m, dtype=torch.int64).view(1, 1, m).expand(t, n, m)


def _box_samples(m: int, seed: int) -> torch.Tensor:
    """Surface samples of the same box shape `fastdo.solver.synthetic` uses (a genuine 2-manifold, so
    kNN spacing behaves like design 11.5's h^2 ~ D^2/M table, unlike a volumetric point cloud)."""
    mesh = trimesh.creation.box(extents=(1.0, 0.6, 0.4))
    mesh.apply_translation(-mesh.bounding_box.centroid)
    pts, _face_idx = trimesh.sample.sample_surface(mesh, m, seed=seed)
    return torch.as_tensor(np.asarray(pts), dtype=DT)


def test_exactness_at_k_equals_m() -> None:
    """Design 11.5: sparse with the full candidate set (k=M) reproduces the dense potentials
    and plan to 1e-9 (same eps/rho/n_iter/init, cold start)."""
    T, N, M = 2, 128, 512
    x, y = _clouds(T, N, M)
    a = torch.full((T, N), 1.0 / N, dtype=DT)
    b = torch.full((T, M), 1.0 / M, dtype=DT)
    eps, rho_a, rho_b, n_iter = 2e-3, 1.0, 0.1, 50
    C = torch.cdist(x, y) ** 2
    P_dense, f_dense, g_dense = sinkhorn_unbalanced_potentials(C, a, b, eps, rho_a, rho_b, n_iter)

    idx = _full_idx(T, N, M)
    C_k = torch.gather(C, 2, idx)
    values, (f_sp, g_sp) = sinkhorn_unbalanced_sparse(C_k, idx, a, b, eps, rho_a, rho_b, n_iter)
    P_from_sparse = torch.zeros_like(P_dense).scatter_(2, idx, values)

    assert torch.allclose(P_from_sparse, P_dense, atol=1e-9)
    assert torch.allclose(f_sp, f_dense, atol=1e-9)
    assert torch.allclose(g_sp, g_dense, atol=1e-9)


def test_exactness_at_k_equals_m_with_masked_marginal() -> None:
    """Exactness at k=M against `sinkhorn_unbalanced_potentials` on the
    SAME masked marginal (the mesh_mask mechanism - `b` has exact zeros, not just a uniform
    marginal), 1e-9. Masked columns must match too, not only the unmasked ones - `_safe_log_mass`'s
    sentinel is shared code (imported from `ot.py`), so this is exercising that shared path directly."""
    T, N, M = 2, 128, 512
    x, y = _clouds(T, N, M)
    a = torch.full((T, N), 1.0 / N, dtype=DT)
    b = torch.full((T, M), 1.0 / M, dtype=DT)
    b[:, ::3] = 0.0                                    # mask out a third of the columns (exact zeros)
    b = b / b.sum(dim=1, keepdim=True)
    eps, rho_a, rho_b, n_iter = 2e-3, 1.0, 0.1, 50
    C = torch.cdist(x, y) ** 2
    P_dense, f_dense, g_dense = sinkhorn_unbalanced_potentials(C, a, b, eps, rho_a, rho_b, n_iter)

    idx = _full_idx(T, N, M)
    C_k = torch.gather(C, 2, idx)
    values, (f_sp, g_sp) = sinkhorn_unbalanced_sparse(C_k, idx, a, b, eps, rho_a, rho_b, n_iter)
    P_from_sparse = torch.zeros_like(P_dense).scatter_(2, idx, values)

    assert torch.allclose(P_from_sparse, P_dense, atol=1e-9)
    assert torch.allclose(f_sp, f_dense, atol=1e-9)
    assert torch.allclose(g_sp, g_dense, atol=1e-9)
    assert float(P_from_sparse[:, :, ::3].abs().max()) < 1e-9   # masked columns carry ~0 mass in both


def test_locality_fine_level_tail_mass_below_1e_3() -> None:
    """Design 11.5's measured table: at the fine setting (eps=1e-4, rho_x=0.1, rho_y=1e-3, all x
    diameter^2) the dropped row mass outside k=32 nearest mesh samples has mean < 1e-3."""
    M, N, k = 2048, 512, 32
    Y = _box_samples(M, seed=3).unsqueeze(0)
    diameter2 = float((Y[0].max(0).values - Y[0].min(0).values).norm() ** 2)
    perm = torch.randperm(M)[:N]
    X = Y[:, perm] + 0.005 * torch.randn(1, N, 3, dtype=DT)   # ~5mm pose-offset proxy
    a = torch.full((1, N), 1.0 / N, dtype=DT)
    b = torch.full((1, M), 1.0 / M, dtype=DT)
    C = torch.cdist(X, Y) ** 2
    eps, rho_a, rho_b = 1e-4 * diameter2, 0.1 * diameter2, 1e-3 * diameter2
    P_dense = sinkhorn_unbalanced_potentials(C, a, b, eps, rho_a, rho_b, n_iter=300)[0]
    idx = torch.topk(-C, k, dim=2).indices
    tail = knn_tail_mass(P_dense, idx)
    assert float(tail.mean()) < 1e-3, float(tail.mean())


def test_locality_mid_level_tail_mass_exceeds_0_1() -> None:
    """Same shape at the mid setting (eps=2e-3, rho_x=1, rho_y=0.1), but with the partial-observation
    structure design 11.9 identifies as the actual mechanism: `X` samples only ONE half of the mesh
    (self-occlusion), while `b` still covers the WHOLE mesh uniformly. The near-hard mesh marginal then
    forces mass onto the far, unobserved half - long-range transport that k=64 truncation (nearest
    candidates to a *visible*-side observation) cannot reach, unlike the fine level's local plan."""
    M, N, k = 1024, 512, 64
    Y = _box_samples(M, seed=3)
    diameter2 = float((Y.max(0).values - Y.min(0).values).norm() ** 2)
    front = torch.nonzero(Y[:, 0] > Y[:, 0].median()).squeeze(-1)     # one observed half
    perm = front[torch.randperm(front.shape[0])[:N]]
    X = (Y[perm] + 0.005 * torch.randn(N, 3, dtype=DT)).unsqueeze(0)
    Y = Y.unsqueeze(0)
    a = torch.full((1, N), 1.0 / N, dtype=DT)
    b = torch.full((1, M), 1.0 / M, dtype=DT)                          # marginal covers front AND back
    C = torch.cdist(X, Y) ** 2
    eps, rho_a, rho_b = 2e-3 * diameter2, 1.0 * diameter2, 0.1 * diameter2
    P_dense = sinkhorn_unbalanced_potentials(C, a, b, eps, rho_a, rho_b, n_iter=300)[0]
    idx = torch.topk(-C, k, dim=2).indices
    tail = knn_tail_mass(P_dense, idx)
    assert float(tail.mean()) > 0.1, float(tail.mean())


def test_plan_barycenters_sparse_matches_dense_on_full_candidate_set() -> None:
    T, N, M = 2, 64, 96
    x, y = _clouds(T, N, M)
    a = torch.full((T, N), 1.0 / N, dtype=DT)
    b = torch.full((T, M), 1.0 / M, dtype=DT)
    eps, rho_a, rho_b, n_iter = 0.02, 10.0, 0.1, 100
    C = torch.cdist(x, y) ** 2
    idx = _full_idx(T, N, M)
    C_k = torch.gather(C, 2, idx)
    values, _ = sinkhorn_unbalanced_sparse(C_k, idx, a, b, eps, rho_a, rho_b, n_iter)
    P_dense = sinkhorn_unbalanced_potentials(C, a, b, eps, rho_a, rho_b, n_iter)[0]

    ybar_sp, w_sp = plan_barycenters_sparse(values, idx, y)
    ybar_dense, w_dense = plan_barycenters(P_dense, y)
    assert torch.allclose(ybar_sp, ybar_dense, atol=1e-6)
    assert torch.allclose(w_sp, w_dense, atol=1e-9)


def test_knn_tail_mass_zero_when_candidates_cover_everything() -> None:
    T, N, M = 1, 8, 12
    P = torch.full((T, N, M), 1.0 / M, dtype=DT)
    idx = _full_idx(T, N, M)
    tail = knn_tail_mass(P, idx)
    assert tail.shape == (T, N)
    assert torch.allclose(tail, torch.zeros(T, N, dtype=DT), atol=1e-12)


def test_warm_start_from_dense_converges_fast() -> None:
    """Potentials from a dense fine-level call warm-start the sparse call: <=20 iterations should reach
    marginals within 1e-6 of a 200-iteration cold sparse solve."""
    T, N, M, k = 1, 256, 1024, 32
    torch.manual_seed(5)
    y = torch.randn(T, M, 3, dtype=DT)
    x = y[:, torch.randperm(M)[:N]] + 0.005 * torch.randn(T, N, 3, dtype=DT)
    a = torch.full((T, N), 1.0 / N, dtype=DT)
    b = torch.full((T, M), 1.0 / M, dtype=DT)
    eps, rho_a, rho_b = 1e-4, 0.1, 1e-3
    C = torch.cdist(x, y) ** 2
    _, f_dense, g_dense = sinkhorn_unbalanced_potentials(C, a, b, eps, rho_a, rho_b, n_iter=200)

    idx = torch.topk(-C, k, dim=2).indices
    C_k = torch.gather(C, 2, idx)
    cold_values, _ = sinkhorn_unbalanced_sparse(C_k, idx, a, b, eps, rho_a, rho_b, n_iter=200)
    warm_values, _ = sinkhorn_unbalanced_sparse(C_k, idx, a, b, eps, rho_a, rho_b, n_iter=20,
                                                potentials=(f_dense, g_dense))

    # Primary check: same `idx` for both calls, so `values` are directly comparable entry-by-entry (no
    # need to re-aggregate by column first).
    assert torch.allclose(warm_values, cold_values, atol=1e-6)
    # Secondary check: the column marginals agree too (order-invariant, catches a hypothetical bug that
    # preserved per-entry values but shuffled which column they attach to).
    flat_idx = idx.reshape(T, -1)
    warm_col = torch.zeros(T, M, dtype=DT).scatter_add_(1, flat_idx, warm_values.reshape(T, -1))
    cold_col = torch.zeros(T, M, dtype=DT).scatter_add_(1, flat_idx, cold_values.reshape(T, -1))
    assert torch.allclose(warm_col, cold_col, atol=1e-6)


def test_warm_start_across_candidate_change_matches_cold_solve() -> None:
    """Two consecutive refreshes with DIFFERENT
    visibility masks (hence different candidate sets) warm-start correctly across the change, converging
    to the same marginals as an independent cold solve on the second candidate set - not corrupting state
    via stale candidate-slot indexing, since `g` is kept GLOBAL-indexed throughout."""
    T, N, M, k = 1, 256, 1024, 32
    torch.manual_seed(7)
    y = torch.randn(T, M, 3, dtype=DT)
    x = y[:, torch.randperm(M)[:N]] + 0.005 * torch.randn(T, N, 3, dtype=DT)
    a = torch.full((T, N), 1.0 / N, dtype=DT)
    b1 = torch.full((T, M), 1.0 / M, dtype=DT)
    # A modest visibility drift between two consecutive accepted LM steps (~5% of columns newly
    # excluded), not a wholesale marginal change - matching what an actual refresh looks like.
    flip = torch.randperm(M)[: M // 20]
    b2 = b1.clone()
    b2[:, flip] = 0.0
    b2 = b2 / b2.sum(dim=1, keepdim=True)
    eps, rho_a, rho_b = 1e-4, 0.1, 1e-3

    C = torch.cdist(x, y) ** 2
    idx1 = torch.topk(-C, k, dim=2).indices
    C1_k = torch.gather(C, 2, idx1)
    # Step 1 well-converged (300 iters), mimicking the fine level's steady state at an accepted step
    # after several LM iterations, not a single freshly-cold-started recompute.
    _, pots1 = sinkhorn_unbalanced_sparse(C1_k, idx1, a, b1, eps, rho_a, rho_b, n_iter=300)

    C2_search = torch.where(b2[:, None, :] > 0, C, torch.full_like(C, float("inf")))
    idx2 = torch.topk(-C2_search, k, dim=2).indices
    C2_k = torch.gather(C, 2, idx2)
    warm_values, _ = sinkhorn_unbalanced_sparse(C2_k, idx2, a, b2, eps, rho_a, rho_b, n_iter=150,
                                                potentials=pots1)
    cold_values, _ = sinkhorn_unbalanced_sparse(C2_k, idx2, a, b2, eps, rho_a, rho_b, n_iter=300)

    assert torch.isfinite(warm_values).all()
    # Primary check: both calls share `idx2`, so `values` are directly comparable entry-by-entry.
    assert torch.allclose(warm_values, cold_values, atol=1e-6)
    # Secondary check: column marginals (order-invariant) agree too.
    flat_idx2 = idx2.reshape(T, -1)
    warm_col = torch.zeros(T, M, dtype=DT).scatter_add_(1, flat_idx2, warm_values.reshape(T, -1))
    cold_col = torch.zeros(T, M, dtype=DT).scatter_add_(1, flat_idx2, cold_values.reshape(T, -1))
    assert torch.allclose(warm_col, cold_col, atol=1e-6)


@pytest.mark.timing
@pytest.mark.skipif(not torch.cuda.is_available(), reason="timing gate needs the GPU")
def test_sparse_fine_level_20_iters_under_40ms() -> None:
    """Design 11.5: sparse fine-level call at (100,1024,2048,k=32) for 20 iterations under 40ms
    (vs. dense's ~1s/50 iterations, the headline speedup)."""
    device = "cuda"
    T, N, M, k = 100, 1024, 2048, 32
    torch.manual_seed(0)
    y = torch.randn(T, M, 3, dtype=torch.float32, device=device)
    x = torch.randn(T, N, 3, dtype=torch.float32, device=device)
    a = torch.full((T, N), 1.0 / N, dtype=torch.float32, device=device)
    b = torch.full((T, M), 1.0 / M, dtype=torch.float32, device=device)
    C = torch.cdist(x, y) ** 2
    idx = torch.topk(-C, k, dim=2).indices
    C_k = torch.gather(C, 2, idx)
    sinkhorn_unbalanced_sparse(C_k, idx, a, b, 1e-4, 0.1, 1e-3, 5)   # warm-up: skip allocator/kernel setup
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    sinkhorn_unbalanced_sparse(C_k, idx, a, b, 1e-4, 0.1, 1e-3, 20)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.04, elapsed


@pytest.mark.timing
@pytest.mark.skipif(not torch.cuda.is_available(), reason="timing gate needs the GPU")
def test_fast_refresh_under_60ms() -> None:
    """A full fine-level 'refresh' - float64-matmul-mode cdist (not the ~100x
    slower precision-mode kernel `pointmap_term` used before), topk(k=32), and the
    production warm-start budget of 10 sparse Sinkhorn iterations - targets under 60ms on the H100 at
    (100,1024,2048,k=32) (informational under contention)."""
    device = "cuda"
    T, N, M, k = 100, 1024, 2048, 32
    torch.manual_seed(0)
    x = torch.randn(T, N, 3, dtype=torch.float64, device=device)
    y = torch.randn(T, M, 3, dtype=torch.float64, device=device)
    a = torch.full((T, N), 1.0 / N, dtype=torch.float32, device=device)
    b = torch.full((T, M), 1.0 / M, dtype=torch.float32, device=device)

    def refresh() -> None:
        C64 = torch.cdist(x, y) ** 2                    # default (matmul) mode, float64
        C = C64.to(torch.float32)
        idx = torch.topk(-C, k, dim=2).indices
        C_k = torch.gather(C, 2, idx)
        sinkhorn_unbalanced_sparse(C_k, idx, a, b, 1e-4, 0.1, 1e-3, 10)

    refresh()                                            # warm-up: skip allocator/kernel setup
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    refresh()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.06, elapsed
