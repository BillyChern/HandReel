import math

import torch

from fastdo.geometry.lie import apply_left, se3_exp
from fastdo.solver.ot_sparse import SparsePlan
from fastdo.solver.terms import (
    _squared_distances,
    pointmap_term,
    project,
    smoothness_term,
    track_term,
    transform_points,
)

torch.manual_seed(0)
DT = torch.float64
DT32 = torch.float32


def _scene(T: int = 3, M: int = 40, J: int = 12):
    Y = torch.randn(M, 3, dtype=DT) * 0.3
    poses = se3_exp(torch.randn(T, 6, dtype=DT) * 0.1)
    poses[:, 2, 3] += 0.8
    log_scale = torch.tensor(math.log(0.5), dtype=DT)
    K = torch.tensor([[300.0, 0.0, 160.0], [0.0, 300.0, 120.0], [0.0, 0.0, 1.0]], dtype=DT)
    anchors = Y[:J]
    return Y, poses, log_scale, K, anchors


def _perturb(poses: torch.Tensor, log_scale: torch.Tensor, delta: torch.Tensor):
    """delta: (T*6+1,) -> perturbed (poses, log_scale)."""
    T = poses.shape[0]
    new = torch.stack([apply_left(poses[k], delta[6 * k:6 * k + 6]) for k in range(T)])
    return new, log_scale + delta[-1]


def _fd_check(residual_fn, jac, poses, log_scale, atol: float) -> None:
    """Compare analytic jac (T,n,d,7) against central differences on the frame's own params."""
    T = poses.shape[0]
    h = 1e-6
    for k in range(T):
        for i in range(7):
            d = torch.zeros(T * 6 + 1, dtype=DT)
            idx = T * 6 if i == 6 else 6 * k + i
            d[idx] = h
            rp = residual_fn(*_perturb(poses, log_scale, d))[k]
            rm = residual_fn(*_perturb(poses, log_scale, -d))[k]
            num = (rp - rm) / (2 * h)
            assert torch.allclose(jac[k, :, :, i], num, atol=atol), f"frame {k} col {i}"


def test_pointmap_residual_zero_at_truth_and_jacobian() -> None:
    Y, poses, log_scale, _K, _ = _scene()
    X = transform_points(Y, poses, log_scale.exp())              # perfect observations
    X_valid = torch.ones(X.shape[:2], dtype=torch.bool)
    out, plan = pointmap_term(X, X_valid, Y, poses, log_scale, eps=1e-4, rho_x=float("inf"),
                              rho_y=float("inf"), n_sinkhorn=500)
    assert isinstance(plan, torch.Tensor)                         # no `knn`: always the dense plan
    assert out.residual.abs().max() < 2e-3                       # entropic blur only
    assert plan.shape == (3, 40, 40)

    def res(p, s):
        return pointmap_term(X, X_valid, Y, p, s, 1e-4, float("inf"), float("inf"), 500, plan=plan)[0].residual

    _fd_check(res, out.jacobian, poses, log_scale, atol=1e-5)


def test_pointmap_invalid_rows_have_zero_weight() -> None:
    Y, poses, log_scale, _K, _ = _scene()
    X = transform_points(Y, poses, log_scale.exp())
    X_valid = torch.ones(X.shape[:2], dtype=torch.bool)
    X_valid[:, :10] = False
    X[:, :10] = float("nan")
    out, plan = pointmap_term(X, X_valid, Y, poses, log_scale, 1e-2, 10.0, 0.05, 100)
    assert isinstance(plan, torch.Tensor)                         # no `knn`: always the dense plan
    assert torch.all(out.weight[:, :10] == 0)
    assert torch.isfinite(out.residual).all() and torch.isfinite(out.jacobian).all()
    assert torch.allclose(plan[:, :10].sum(dim=(1, 2)), torch.zeros(3, dtype=DT))


def test_pointmap_term_float32_matches_float64_cost() -> None:
    """The cost matrix is always computed in float64 regardless of X/Y's own
    dtype, so a float32 caller doesn't suffer catastrophic cancellation for nearby points any more than
    a float64 caller does; compare fresh (plan=None) solves cross-dtype."""
    Y, poses, log_scale, _K, _ = _scene()
    X = transform_points(Y, poses, log_scale.exp())
    X_valid = torch.ones(X.shape[:2], dtype=torch.bool)
    out64, _ = pointmap_term(X, X_valid, Y, poses, log_scale, eps=1e-3, rho_x=float("inf"),
                             rho_y=float("inf"), n_sinkhorn=300)

    Y32 = Y.to(DT32)
    poses32 = poses.to(DT32)
    log_scale32 = log_scale.to(DT32)
    X32 = X.to(DT32)
    out32, _ = pointmap_term(X32, X_valid, Y32, poses32, log_scale32, eps=1e-3, rho_x=float("inf"),
                             rho_y=float("inf"), n_sinkhorn=300)

    norm64 = out64.residual.double().norm()
    norm32 = out32.residual.double().norm()
    assert (norm32 - norm64).abs() / norm64 < 1e-4


def test_matmul_cdist_float64_matches_precision_mode() -> None:
    """The default (matmul-based) cdist in float64 must match the slow, explicit
    non-mm kernel to high relative precision - the empirical justification for pointmap_term no longer
    using the non-mm kernel by default (design 11.5's cancellation concern was real in float32, ~1e-7
    relative, but negligible in float64, ~1e-16 relative, next to eps as small as 1e-4 * diameter^2)."""
    torch.manual_seed(0)
    n, m = 300, 400
    x = torch.randn(n, 3, dtype=DT)
    y = torch.randn(m, 3, dtype=DT)
    pts = torch.cat([x, y])
    diameter = (pts.max(0).values - pts.min(0).values).norm()
    x, y = x / diameter, y / diameter                            # unit-diameter canonical points
    C_mm = torch.cdist(x, y) ** 2
    C_precise = torch.cdist(x, y, compute_mode="donot_use_mm_for_euclid_dist") ** 2
    assert torch.allclose(C_mm, C_precise, rtol=1e-10, atol=1e-15)


def test_pointmap_point_to_plane_jacobian_and_precision() -> None:
    Y, poses, log_scale, _K, _ = _scene()
    normals = torch.nn.functional.normalize(Y, dim=-1)                 # scene points ~ sphere-like; any unit field works for FD
    X = transform_points(Y, poses, log_scale.exp())
    X_valid = torch.ones(X.shape[:2], dtype=torch.bool)
    out, plan = pointmap_term(X, X_valid, Y, poses, log_scale, 1e-4, float("inf"), float("inf"), 300,
                              normals=normals, point_to_plane=True)
    assert out.residual.shape == (3, 40, 1) and out.jacobian.shape == (3, 40, 1, 7)

    def res(p, s):
        return pointmap_term(X, X_valid, Y, p, s, 1e-4, float("inf"), float("inf"), 300, plan=plan,
                             normals=normals, point_to_plane=True)[0].residual

    _fd_check(res, out.jacobian, poses, log_scale, atol=1e-5)


def test_point_to_plane_ignores_tangential_sampling_error() -> None:
    torch.manual_seed(1)
    Y = torch.nn.functional.normalize(torch.randn(400, 3, dtype=DT), dim=-1) * 0.5   # unit-ish sphere samples
    normals = torch.nn.functional.normalize(Y, dim=-1)
    poses = se3_exp(torch.zeros(1, 6, dtype=DT)); poses[0, 2, 3] = 1.0
    log_scale = torch.tensor(0.0, dtype=DT)
    X_c = torch.nn.functional.normalize(torch.randn(1, 300, 3, dtype=DT), dim=-1) * 0.5   # different samples on the same sphere
    X = transform_points(X_c[0], poses, 1.0)
    X_valid = torch.ones(1, 300, dtype=torch.bool)
    p2p, _ = pointmap_term(X, X_valid, Y, poses, log_scale, 1e-4, float("inf"), 1e-3, 300)
    p2pl, _ = pointmap_term(X, X_valid, Y, poses, log_scale, 1e-4, float("inf"), 1e-3, 300,
                            normals=normals, point_to_plane=True)
    assert p2pl.residual.abs().mean() < 0.3 * p2p.residual.norm(dim=-1).mean()


def test_pointmap_mesh_mask_all_true_matches_unmasked() -> None:
    """An all-true `mesh_mask` is the same restriction as no mask at all: same plan/barycenters."""
    Y, poses, log_scale, _K, _ = _scene()
    X = transform_points(Y, poses, log_scale.exp())
    X_valid = torch.ones(X.shape[:2], dtype=torch.bool)
    mask = torch.ones(poses.shape[0], Y.shape[0], dtype=torch.bool)
    out_ref, plan_ref = pointmap_term(X, X_valid, Y, poses, log_scale, 1e-2, 10.0, 0.05, 100)
    out_masked, plan_masked = pointmap_term(X, X_valid, Y, poses, log_scale, 1e-2, 10.0, 0.05, 100,
                                            mesh_mask=mask)
    assert isinstance(plan_ref, torch.Tensor) and isinstance(plan_masked, torch.Tensor)
    assert torch.allclose(plan_ref, plan_masked, atol=1e-9)
    assert torch.allclose(out_ref.residual, out_masked.residual, atol=1e-9)


def test_pointmap_mesh_mask_excludes_masked_columns() -> None:
    """Masked-out mesh samples must receive (numerically) zero plan mass regardless of cost."""
    Y, poses, log_scale, _K, _ = _scene(M=40)
    X = transform_points(Y, poses, log_scale.exp())
    X_valid = torch.ones(X.shape[:2], dtype=torch.bool)
    mask = torch.ones(poses.shape[0], Y.shape[0], dtype=torch.bool)
    mask[:, :20] = False           # mask out the first half of the mesh samples for every frame
    _out, plan = pointmap_term(X, X_valid, Y, poses, log_scale, 1e-2, 10.0, 0.05, 200, mesh_mask=mask)
    assert isinstance(plan, torch.Tensor)                         # no `knn`: always the dense plan
    col_mass = plan.sum(dim=1)                        # (T,M)
    assert torch.all(col_mass[:, :20] < 1e-12), col_mass[:, :20]
    assert torch.all(col_mass[:, 20:].sum(dim=1) > 0.5)


def test_pointmap_mesh_mask_all_false_row_falls_back_to_full_template() -> None:
    """A frame whose `mesh_mask` is all-False (e.g. `visible_samples` degenerate case reaching here
    directly, bypassing its own `_MIN_VISIBLE` fallback) must fall back to the uniform full template for
    that frame, not silently zero its whole mesh-side marginal - and must produce finite outputs."""
    Y, poses, log_scale, _K, _ = _scene(T=3, M=40)
    X = transform_points(Y, poses, log_scale.exp())
    X_valid = torch.ones(X.shape[:2], dtype=torch.bool)
    mask = torch.ones(poses.shape[0], Y.shape[0], dtype=torch.bool)
    mask[0] = False                # frame 0 has no visible samples at all
    out, plan = pointmap_term(X, X_valid, Y, poses, log_scale, 1e-2, 10.0, 0.05, 100, mesh_mask=mask)
    assert isinstance(plan, torch.Tensor)                         # no `knn`: always the dense plan
    assert torch.isfinite(out.residual).all() and torch.isfinite(out.jacobian).all()
    assert torch.isfinite(plan).all()
    # Frame 0 falls back to the full (uniform) template: its column mass should sum close to 1 (finite
    # rho_x/rho_y here means "close to", not exactly, balanced: row-mass diagnostics showed the same ~90%+
    # transported-mass pattern with finite rho), spread over
    # all M columns, not concentrated in/excluded from any particular subset.
    assert float(plan[0].sum()) > 0.95
    assert bool((plan[0].sum(dim=0) > 0).all())          # every mesh column gets some mass, frame 0


def test_pointmap_mesh_mask_zero_weight_row_falls_back_to_full_template() -> None:
    """A frame whose *continuous* `mesh_mask` (e.g. mask times a foreshortening weight that happens to
    clamp everything to 0) sums to zero must fall back the same way as an all-False boolean row -
    finite outputs, uniform full-template mass for that frame - even though the mask itself, as a
    boolean, is not empty."""
    Y, poses, log_scale, _K, _ = _scene(T=3, M=40)
    X = transform_points(Y, poses, log_scale.exp())
    X_valid = torch.ones(X.shape[:2], dtype=torch.bool)
    weight = torch.ones(poses.shape[0], Y.shape[0], dtype=DT)
    weight[1] = 0.0                 # frame 1: every "visible" sample foreshortened to exactly 0 weight
    out, plan = pointmap_term(X, X_valid, Y, poses, log_scale, 1e-2, 10.0, 0.05, 100, mesh_mask=weight)
    assert isinstance(plan, torch.Tensor)                         # no `knn`: always the dense plan
    assert torch.isfinite(out.residual).all() and torch.isfinite(out.jacobian).all()
    assert torch.isfinite(plan).all()
    assert float(plan[1].sum()) > 0.95


def test_pointmap_term_knn_at_k_equals_m_matches_dense() -> None:
    """`knn=M` (every column a candidate) must reproduce the dense plan/residual through the
    full `pointmap_term` API, and the returned plan is a `SparsePlan`."""
    Y, poses, log_scale, _K, _ = _scene(T=2, M=48)
    X = transform_points(Y, poses, log_scale.exp())
    X_valid = torch.ones(X.shape[:2], dtype=torch.bool)
    out_dense, _plan_dense = pointmap_term(X, X_valid, Y, poses, log_scale, 1e-2, 10.0, 0.05, 100)
    out_knn, plan_knn = pointmap_term(X, X_valid, Y, poses, log_scale, 1e-2, 10.0, 0.05, 100, knn=48)
    assert isinstance(plan_knn, SparsePlan)
    assert plan_knn.indices.shape == (2, X.shape[1], 48)
    assert torch.allclose(out_knn.residual, out_dense.residual, atol=1e-8)
    assert torch.allclose(out_knn.jacobian, out_dense.jacobian, atol=1e-8)


def test_pointmap_term_knn_cached_sparse_plan_reused() -> None:
    """The cached-plan (`plan=` supplied directly) path must accept a `SparsePlan` unchanged, exactly
    like it already accepts a dense plan."""
    Y, poses, log_scale, _K, _ = _scene(T=2, M=40)
    X = transform_points(Y, poses, log_scale.exp())
    X_valid = torch.ones(X.shape[:2], dtype=torch.bool)
    out1, plan = pointmap_term(X, X_valid, Y, poses, log_scale, 1e-2, 10.0, 0.05, 50, knn=16)
    assert isinstance(plan, SparsePlan)
    out2, plan2 = pointmap_term(X, X_valid, Y, poses, log_scale, 1e-2, 10.0, 0.05, 50, plan=plan)
    assert plan2 is plan                                  # no Sinkhorn recompute, passed through as-is
    assert torch.allclose(out1.residual, out2.residual, atol=1e-12)


def test_pointmap_term_knn_never_picks_a_masked_out_column() -> None:
    """Candidate search only over samples with nonzero `mesh_mask` weight - a hidden
    column must never appear in the candidate set, however close it is."""
    Y, poses, log_scale, _K, _ = _scene(T=1, M=40)
    X = transform_points(Y, poses, log_scale.exp())
    X_valid = torch.ones(X.shape[:2], dtype=torch.bool)
    mask = torch.ones(1, 40, dtype=torch.bool)
    mask[:, :20] = False                                   # hide the first half, including near neighbours
    _out, plan = pointmap_term(X, X_valid, Y, poses, log_scale, 1e-2, 10.0, 0.05, 100, mesh_mask=mask,
                               knn=8)
    assert isinstance(plan, SparsePlan)
    assert bool((plan.indices >= 20).all())


def test_pointmap_term_knn_filler_columns_get_zero_mass_when_visible_lt_k() -> None:
    """terms.py's own comment on the knn branch: a frame with FEWER mesh_mask-visible samples than `knn`
    (here 16 visible out of 64, knn=32) still fills its candidate slots via torch.topk with masked-out
    (filler) columns - this must not corrupt anything: every filler candidate carries (numerically) zero
    plan mass, and every output stays finite."""
    Y, poses, log_scale, _K, _ = _scene(T=1, M=64)
    X = transform_points(Y, poses, log_scale.exp())
    X_valid = torch.ones(X.shape[:2], dtype=torch.bool)
    mask = torch.zeros(1, 64, dtype=torch.bool)
    mask[:, :16] = True                                    # only 16 of 64 columns visible
    out, plan = pointmap_term(X, X_valid, Y, poses, log_scale, 1e-2, 10.0, 0.05, 100, mesh_mask=mask,
                              knn=32)
    assert isinstance(plan, SparsePlan)
    assert torch.isfinite(out.residual).all() and torch.isfinite(out.jacobian).all()
    assert torch.isfinite(plan.values).all()
    hidden = ~mask[0]                                      # (64,): columns 16..63 are masked out
    is_filler = hidden[plan.indices[0]]                    # (N,32) bool: candidate is a filler column
    assert bool(is_filler.any())                           # sanity: some rows actually needed fillers
    assert float(plan.values[0][is_filler].abs().max()) < 1e-9


def test_pointmap_term_knn_with_potentials_returns_dense_shaped_duals() -> None:
    Y, poses, log_scale, _K, _ = _scene(T=2, M=32)
    X = transform_points(Y, poses, log_scale.exp())
    X_valid = torch.ones(X.shape[:2], dtype=torch.bool)
    out, plan, (f, g) = pointmap_term(X, X_valid, Y, poses, log_scale, 1e-2, 10.0, 0.05, 50, knn=8,
                                      potentials=None)
    assert isinstance(plan, SparsePlan)
    assert f is not None and g is not None
    assert f.shape == (2, X.shape[1]) and g.shape == (2, 32)
    assert torch.isfinite(out.residual).all()


def test_pointmap_term_knn_point_to_plane_at_k_equals_m_matches_dense() -> None:
    Y, poses, log_scale, _K, _ = _scene()
    normals = torch.nn.functional.normalize(Y, dim=-1)
    X = transform_points(Y, poses, log_scale.exp())
    X_valid = torch.ones(X.shape[:2], dtype=torch.bool)
    out_dense, _ = pointmap_term(X, X_valid, Y, poses, log_scale, 1e-4, float("inf"), float("inf"), 300,
                                 normals=normals, point_to_plane=True)
    out_knn, plan = pointmap_term(X, X_valid, Y, poses, log_scale, 1e-4, float("inf"), float("inf"), 300,
                                  normals=normals, point_to_plane=True, knn=Y.shape[0])
    assert isinstance(plan, SparsePlan)
    assert torch.allclose(out_knn.residual, out_dense.residual, atol=1e-7)


def test_track_term_zero_at_truth_and_jacobian() -> None:
    _Y, poses, log_scale, K, anchors = _scene()
    p = transform_points(anchors, poses, log_scale.exp())
    tracks = project(p, K)
    vis = torch.ones(tracks.shape[:2], dtype=torch.bool)
    out = track_term(anchors, tracks, vis, poses, log_scale, K, cauchy_c=4.0)
    assert out.residual.abs().max() < 1e-9
    assert torch.allclose(out.weight, torch.ones_like(out.weight))
    noisy = tracks + torch.randn_like(tracks) * 2.0
    out2 = track_term(anchors, noisy, vis, poses, log_scale, K, cauchy_c=4.0)

    def res(pp, s):
        return track_term(anchors, noisy, vis, pp, s, K, 4.0).residual

    _fd_check(res, out2.jacobian, poses, log_scale, atol=1e-4)
    assert (out2.weight < 1.0).any() and (out2.weight > 0.0).all()


def test_track_term_nan_tracks_when_invisible() -> None:
    _Y, poses, log_scale, K, anchors = _scene()
    p = transform_points(anchors, poses, log_scale.exp())
    tracks = project(p, K)
    vis = torch.ones(tracks.shape[:2], dtype=torch.bool)
    vis[:, :4] = False
    tracks[:, :4] = float("nan")
    out = track_term(anchors, tracks, vis, poses, log_scale, K, cauchy_c=4.0)
    assert torch.all(out.weight[:, :4] == 0)
    assert torch.isfinite(out.residual).all()
    assert torch.isfinite(out.jacobian).all()


def test_smoothness_jacobian_small_motion() -> None:
    poses = se3_exp(torch.randn(4, 6, dtype=DT) * 0.02)
    r, Jk, Jk1 = smoothness_term(poses, lam_v=1.0)
    assert r.shape == (3, 6) and Jk.shape == (3, 6, 6)
    h = 1e-6
    for k in range(3):
        for i in range(6):
            d = torch.zeros(6, dtype=DT); d[i] = h
            pp = poses.clone(); pp[k] = apply_left(poses[k], d)
            pm = poses.clone(); pm[k] = apply_left(poses[k], -d)
            num = (smoothness_term(pp, 1.0)[0][k] - smoothness_term(pm, 1.0)[0][k]) / (2 * h)
            assert torch.allclose(Jk[k, :, i], num, atol=1e-6)
            pp = poses.clone(); pp[k + 1] = apply_left(poses[k + 1], d)
            pm = poses.clone(); pm[k + 1] = apply_left(poses[k + 1], -d)
            num = (smoothness_term(pp, 1.0)[0][k] - smoothness_term(pm, 1.0)[0][k]) / (2 * h)
            assert torch.allclose(Jk1[k, :, i], num, atol=1e-6)


def test_squared_distances_match_explicit_cdist() -> None:
    A = torch.randn(3, 17, 3, dtype=DT)
    B = torch.randn(3, 29, 3, dtype=DT)
    B[:, 0] = A[:, 0] + 1e-9                     # near-coincident pair: no mm-path cancellation
    expected = torch.cdist(A, B, compute_mode="donot_use_mm_for_euclid_dist") ** 2
    got = _squared_distances(A, B)
    assert got.shape == (3, 17, 29)
    assert torch.allclose(got, expected, rtol=1e-12, atol=1e-24)
    assert bool((got[:, 0, 0] < 1e-17).all())

