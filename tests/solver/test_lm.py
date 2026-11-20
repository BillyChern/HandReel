import math

import pytest
import torch

from fastdo.geometry.lie import se3_exp, se3_inv, se3_log
from fastdo.solver.lm import (
    LMConfig,
    apply_update,
    assemble_normal_equations,
    lm_optimize,
    lm_step,
    total_cost,
)
from fastdo.solver.terms import (
    pointmap_term,
    project,
    smoothness_term,
    track_term,
    transform_points,
)

torch.manual_seed(0)
DT = torch.float64


def test_assemble_matches_dense_jtj() -> None:
    T, n, d = 3, 5, 2
    J = torch.randn(T, n, d, 7, dtype=DT)
    r = torch.randn(T, n, d, dtype=DT)
    w = torch.rand(T, n, dtype=DT)
    from fastdo.solver.terms import TermOutput
    H, g = assemble_normal_equations([TermOutput(r, J, w)], None, T)
    assert H.shape == (19, 19) and g.shape == (19,)
    # brute force: expand per-frame Jacobians into the global parameter layout
    Jg = torch.zeros(T * n * d, 19, dtype=DT)
    rg = torch.zeros(T * n * d, dtype=DT)
    row = 0
    for k in range(T):
        for i in range(n):
            for j in range(d):
                sw = w[k, i].sqrt()
                Jg[row, 6 * k:6 * k + 6] = sw * J[k, i, j, :6]
                Jg[row, 18] = sw * J[k, i, j, 6]
                rg[row] = sw * r[k, i, j]
                row += 1
    assert torch.allclose(H, Jg.T @ Jg, atol=1e-10)
    assert torch.allclose(g, Jg.T @ rg, atol=1e-10)


@pytest.mark.parametrize("T", [3, 6])
def test_lm_step_banded_matches_dense_on_assembled_system(T: int) -> None:
    """`lm_optimize` solves its LM step with `solver="banded"` (an O(n) arrow-banded
    elimination) instead of the O(n^3) dense factorization, exploiting the block-tridiagonal (bandwidth
    11) + dense-arrow structure that `assemble_normal_equations` always produces. Verify the two agree
    to 1e-9 on a genuinely assembled system (a random pointmap-like term plus a random smoothness
    triple, both damped identically).

    T=3 (n = 6*3+1 = 19) is the smallest system for which `lm_step(solver="banded")` actually takes the
    banded branch rather than falling back to dense (that boundary is `n > _BAND_U + 2 = 13`, i.e.
    T >= 3; T=2's n=13 is exactly `test_lm_step_banded_falls_back_to_dense_for_small_systems`'s fallback
    case) - it covers the boundary where the band (u=11) is almost as
    wide as the matrix itself. T=6 is the original, more comfortably-banded case."""
    from fastdo.solver.terms import TermOutput

    n, d = 5, 2
    J = torch.randn(T, n, d, 7, dtype=DT)
    r = torch.randn(T, n, d, dtype=DT)
    w = torch.rand(T, n, dtype=DT)
    smooth = (torch.randn(T - 1, 6, dtype=DT), torch.randn(T - 1, 6, 6, dtype=DT), torch.randn(T - 1, 6, 6, dtype=DT))
    H, g = assemble_normal_equations([TermOutput(r, J, w)], smooth, T)
    lam = 0.7
    dense = lm_step(H, g, lam, solver="dense")
    banded = lm_step(H, g, lam, solver="banded")
    assert torch.allclose(dense, banded, atol=1e-9), (dense - banded).abs().max()


def test_lm_step_banded_falls_back_to_dense_for_small_systems() -> None:
    """Below `_BAND_U + 2` unknowns the banded path would need a wider band than the matrix has, so
    `lm_step(solver="banded")` falls back to the exact dense solve (still matches `solver="dense"`)."""
    H, g = assemble_normal_equations([], (torch.randn(1, 6, dtype=DT), torch.randn(1, 6, 6, dtype=DT),
                                          torch.randn(1, 6, 6, dtype=DT)), 2)
    dense = lm_step(H, g, 0.3, solver="dense")
    banded = lm_step(H, g, 0.3, solver="banded")
    assert torch.allclose(dense, banded, atol=1e-9)


def test_lm_step_reduces_quadratic() -> None:
    A = torch.randn(19, 19, dtype=DT)
    H = A.T @ A + torch.eye(19, dtype=DT)
    g = torch.randn(19, dtype=DT)
    delta = lm_step(H, g, lam=0.0)
    assert torch.allclose(H @ delta, -g, atol=1e-8)


def test_apply_update_roundtrip() -> None:
    poses = se3_exp(torch.randn(2, 6, dtype=DT) * 0.1)
    ls = torch.tensor(0.1, dtype=DT)
    delta = torch.zeros(13, dtype=DT)
    delta[:6] = torch.tensor([0.01, 0, 0, 0, 0.02, 0], dtype=DT)
    delta[12] = 0.05
    new, nls = apply_update(poses, ls, delta)
    assert torch.allclose(se3_log(new[0] @ se3_inv(poses[0])), delta[:6], atol=1e-9)
    assert torch.allclose(new[1], poses[1])
    assert math.isclose(float(nls), 0.15)


def test_lm_optimize_H_matches_state_after_one_accepted_step() -> None:
    """`LMResult.H` used to be assembled at the TOP of the loop's last
    iteration - i.e. from the state BEFORE that iteration's trial step - which is only correct if the
    step was rejected. Whenever the loop terminates right after an ACCEPTED step (as here: a single,
    well-inside-the-quadratic-region step that both reduces cost and hits `max_iter=1`), `H` was one
    step stale relative to the returned `poses`/`log_scale`, so `TrajectorySolution.covariance` (a
    Laplace approximation via `H`'s pseudo-inverse) was evaluated at the wrong pose. Verify directly:
    run exactly one accepted iteration on a small problem and check `result.H` equals
    `assemble_normal_equations` evaluated fresh at `result.poses`/`result.log_scale`, to 1e-12."""
    T, M, J = 3, 60, 12
    Y = torch.randn(M, 3, dtype=DT) * 0.3
    poses_gt = se3_exp(torch.randn(T, 6, dtype=DT) * 0.05)
    poses_gt[:, 2, 3] += 0.8
    ls_gt = torch.tensor(math.log(0.5), dtype=DT)
    K = torch.tensor([[300.0, 0, 160.0], [0, 300.0, 120.0], [0, 0, 1.0]], dtype=DT)
    anchors = Y[:J]
    X = transform_points(Y, poses_gt, ls_gt.exp())
    X_valid = torch.ones(T, M, dtype=torch.bool)
    tracks = project(transform_points(anchors, poses_gt, ls_gt.exp()), K)
    vis = torch.ones(T, J, dtype=torch.bool)
    # A small perturbation keeps the very first (default lam0=1e-2, near-Gauss-Newton) LM step inside
    # the problem's local quadratic region, so it is virtually guaranteed to reduce cost (asserted below,
    # not just assumed).
    poses0 = torch.stack([se3_exp(torch.tensor([0.005, -0.004, 0.006, 0.01, -0.008, 0.007], dtype=DT)) @ p
                         for p in poses_gt])
    ls0 = ls_gt + 0.01
    state: dict[str, torch.Tensor | None] = {"plan": None}

    def evaluate(poses, ls, refresh: bool):
        out, plan = pointmap_term(X, X_valid, Y, poses, ls, 1e-3, float("inf"), float("inf"), 200,
                                  plan=None if refresh else state["plan"])
        state["plan"] = plan
        trk = track_term(anchors, tracks, vis, poses, ls, K, cauchy_c=4.0)
        return [out, trk], smoothness_term(poses, lam_v=1e-4)

    res = lm_optimize(poses0, ls0, evaluate, LMConfig(max_iter=1))
    assert not torch.equal(res.poses, poses0), "the single step must have been accepted, not rejected"
    assert res.cost < total_cost(*evaluate(poses0, ls0, True))
    terms, smooth = evaluate(res.poses, res.log_scale, True)
    H_expected, _g_expected = assemble_normal_equations(terms, smooth, T)
    assert torch.allclose(res.H, H_expected, atol=1e-12), (res.H - H_expected).abs().max()


def test_lm_recovers_trajectory_from_tracks_and_pointmaps() -> None:
    T, M, J = 5, 200, 40
    Y = torch.randn(M, 3, dtype=DT) * 0.3
    poses_gt = se3_exp(torch.randn(T, 6, dtype=DT) * 0.05)
    poses_gt[:, 2, 3] += 0.8
    ls_gt = torch.tensor(math.log(0.5), dtype=DT)
    K = torch.tensor([[300.0, 0, 160.0], [0, 300.0, 120.0], [0, 0, 1.0]], dtype=DT)
    anchors = Y[:J]
    X = transform_points(Y, poses_gt, ls_gt.exp())
    X_valid = torch.ones(T, M, dtype=torch.bool)
    tracks = project(transform_points(anchors, poses_gt, ls_gt.exp()), K)
    vis = torch.ones(T, J, dtype=torch.bool)
    poses0 = torch.stack([se3_exp(torch.tensor([0.03, -0.02, 0.04, 0.1, -0.08, 0.05], dtype=DT)) @ p for p in poses_gt])
    ls0 = ls_gt + 0.2
    state: dict[str, torch.Tensor | None] = {"plan": None}

    def evaluate(poses, ls, refresh: bool):
        out, plan = pointmap_term(X, X_valid, Y, poses, ls, 1e-3, float("inf"), float("inf"), 200,
                                  plan=None if refresh else state["plan"])
        state["plan"] = plan
        trk = track_term(anchors, tracks, vis, poses, ls, K, cauchy_c=4.0)
        return [out, trk], smoothness_term(poses, lam_v=1e-4)

    res = lm_optimize(poses0, ls0, evaluate, LMConfig(max_iter=40))
    err_t = (res.poses[:, :3, 3] - poses_gt[:, :3, 3]).norm(dim=1).max()
    err_r = se3_log(res.poses @ se3_inv(poses_gt))[:, 3:].norm(dim=1).max()
    assert err_t < 3e-3 and err_r < math.radians(1.0)
    assert abs(float(res.log_scale - ls_gt)) < 0.01
    assert res.cost < total_cost(*evaluate(poses0, ls0, True))


def test_lm_result_hessian_matches_final_state() -> None:
    # Isolated on a forked RNG state: this test's extra torch.randn() draws must not shift what
    # later test modules see from the shared global generator (they rely on their own
    # module-level torch.manual_seed(0) plus incidental draw order, not per-test reseeding).
    with torch.random.fork_rng(devices=[]):
        T, M, J = 5, 200, 40
        Y = torch.randn(M, 3, dtype=DT) * 0.3
        poses_gt = se3_exp(torch.randn(T, 6, dtype=DT) * 0.05)
        poses_gt[:, 2, 3] += 0.8
        ls_gt = torch.tensor(math.log(0.5), dtype=DT)
        K = torch.tensor([[300.0, 0, 160.0], [0, 300.0, 120.0], [0, 0, 1.0]], dtype=DT)
        anchors = Y[:J]
        X = transform_points(Y, poses_gt, ls_gt.exp())
        X_valid = torch.ones(T, M, dtype=torch.bool)
        tracks = project(transform_points(anchors, poses_gt, ls_gt.exp()), K)
        vis = torch.ones(T, J, dtype=torch.bool)
        poses0 = torch.stack(
            [se3_exp(torch.tensor([0.03, -0.02, 0.04, 0.1, -0.08, 0.05], dtype=DT)) @ p for p in poses_gt]
        )
        ls0 = ls_gt + 0.2
        state: dict[str, torch.Tensor | None] = {"plan": None}

        def evaluate(poses, ls, refresh: bool):
            out, plan = pointmap_term(X, X_valid, Y, poses, ls, 1e-3, float("inf"), float("inf"), 200,
                                      plan=None if refresh else state["plan"])
            state["plan"] = plan
            trk = track_term(anchors, tracks, vis, poses, ls, K, cauchy_c=4.0)
            return [out, trk], smoothness_term(poses, lam_v=1e-4)

        res = lm_optimize(poses0, ls0, evaluate, LMConfig(max_iter=40))
        final_terms, final_smooth = evaluate(res.poses, res.log_scale, True)
        H_recomputed, _ = assemble_normal_equations(final_terms, final_smooth, T)
        assert torch.allclose(H_recomputed, res.H, rtol=1e-6, atol=1e-9)


def test_assemble_smoothness_matches_dense() -> None:
    # See test_lm_result_hessian_matches_final_state: keep this test's RNG draws from leaking
    # into the shared global generator that later test modules implicitly depend on.
    with torch.random.fork_rng(devices=[]):
        T = 4
        r = torch.randn(T - 1, 6, dtype=DT)
        Jk = torch.randn(T - 1, 6, 6, dtype=DT)
        Jk1 = torch.randn(T - 1, 6, 6, dtype=DT)
        H, g = assemble_normal_equations([], (r, Jk, Jk1), T)
        assert H.shape == (6 * T + 1, 6 * T + 1) and g.shape == (6 * T + 1,)
        # brute force: expand the tridiagonal smoothness Jacobians into the global parameter layout
        Jg = torch.zeros(6 * (T - 1), 6 * T + 1, dtype=DT)
        rg = torch.zeros(6 * (T - 1), dtype=DT)
        for k in range(T - 1):
            Jg[6 * k:6 * k + 6, 6 * k:6 * k + 6] = Jk[k]
            Jg[6 * k:6 * k + 6, 6 * (k + 1):6 * (k + 1) + 6] = Jk1[k]
            rg[6 * k:6 * k + 6] = r[k]
        assert torch.allclose(H, Jg.T @ Jg, atol=1e-10)
        assert torch.allclose(g, Jg.T @ rg, atol=1e-10)


def _behind_camera_problem() -> tuple[torch.Tensor, torch.Tensor, object]:
    """Pointmap-only problem whose data sit BEHIND the camera (z = -0.3), from an init in front of it (z = 0.8)."""
    T, M = 4, 150
    Y = torch.randn(M, 3, dtype=DT) * 0.05
    target = se3_exp(torch.zeros(T, 6, dtype=DT))
    target[:, 2, 3] = -0.3
    X = transform_points(Y, target, torch.tensor(1.0, dtype=DT))
    X_valid = torch.ones(T, M, dtype=torch.bool)
    poses0 = se3_exp(torch.zeros(T, 6, dtype=DT))
    poses0[:, 2, 3] = 0.8
    state: dict[str, torch.Tensor | None] = {"plan": None}

    def evaluate(poses, ls, refresh: bool):
        out, plan = pointmap_term(X, X_valid, Y, poses, ls, 1e-3, float("inf"), float("inf"), 200,
                                  plan=None if refresh else state["plan"])
        state["plan"] = plan
        return [out], smoothness_term(poses, lam_v=1e-4)

    return poses0, torch.tensor(0.0, dtype=DT), evaluate


def test_lm_without_the_guard_follows_data_behind_the_camera() -> None:
    poses0, ls0, evaluate = _behind_camera_problem()
    res = lm_optimize(poses0, ls0, evaluate, LMConfig(max_iter=40, fix_scale=True))
    assert float(res.poses[:, 2, 3].median()) < 0.0                     # the unguarded optimizer crosses the camera


def test_lm_min_median_depth_rejects_steps_that_cross_the_camera() -> None:
    """Collapses put the object behind the camera; the guard rejects any trial step whose per-clip median depth
    is at or below `min_median_depth`, so the accepted trajectory never gets there."""
    poses0, ls0, evaluate = _behind_camera_problem()
    res = lm_optimize(poses0, ls0, evaluate, LMConfig(max_iter=40, fix_scale=True, min_median_depth=0.05))
    assert float(res.poses[:, 2, 3].median()) > 0.05
