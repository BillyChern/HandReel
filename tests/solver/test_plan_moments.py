"""The pointmap term from a plan's canonical moments equals the term from the plan (`fastdo.solver.moments`).

Term level: dense and kNN plans, point-to-point and point-to-plane, float64 and float32, per-frame templates and
scales, at the refresh pose and at trial poses away from it, including rows of zero and sub-clamp mass. Callback level
(`batch_eval.make_evaluate`): the cache holds moments (never a (T,N,M) or (T,N,k) tensor) with `plan_moments` on and
the plan with it off, and the terms match through refreshes, chunked refreshes, subsets and trials. Solve level: a
batched solve with the flag on reproduces the off solve's LM histories and poses to round-off. Memory: the model
still bounds the measured CPU allocator peak with the flag on.
"""

from __future__ import annotations

import logging
import math
from dataclasses import replace

import pytest
import torch
from torch import Tensor

from fastdo.geometry.lie import se3_exp
from fastdo.solver import batch_eval
from fastdo.solver.api import SolverConfig, _levels, clip_dims, solve_trajectories
from fastdo.solver.batch import pack_clips, prepare_batch
from fastdo.solver.batch_eval import EvalCache, make_evaluate
from fastdo.solver.memory import batch_peak_bytes
from fastdo.solver.moments import PlanMoments, plan_moments, transported_points
from fastdo.solver.ot import plan_barycenters
from fastdo.solver.ot_sparse import SparsePlan, plan_barycenters_sparse
from fastdo.solver.terms import TermOutput, pointmap_term, transform_points
from tests.solver.test_batch import DEVICE, _clip, _inputs, _unzip
from tests.solver.test_memory import _measured_peak
from tests.solver.test_terms import _fd_check

logger = logging.getLogger(__name__)
DT = torch.float64


def _scene(T: int, M: int, N: int, dtype: torch.dtype, seed: int, per_frame: bool
           ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Noisy partial observations of a random template: X (T,N,3) with some invalid rows, X_valid, Y, normals,
    poses (T,4,4) and log_scale (scalar, or (T,) when `per_frame`, which also makes Y and normals (T,M,3))."""
    g = torch.Generator().manual_seed(seed)
    Y = torch.randn(M, 3, generator=g, dtype=DT) * 0.3
    normals = torch.nn.functional.normalize(Y + 0.1 * torch.randn(M, 3, generator=g, dtype=DT), dim=-1)
    poses = se3_exp(torch.randn(T, 6, generator=g, dtype=DT) * 0.1)
    poses[:, 2, 3] += 0.8
    log_scale = torch.tensor(math.log(0.5), dtype=DT)
    if per_frame:
        Y = Y.expand(T, -1, -1) * (1.0 + 0.05 * torch.rand(T, 1, 1, generator=g, dtype=DT))
        normals = normals.expand(T, -1, -1).clone()
        log_scale = log_scale + 0.02 * torch.randn(T, generator=g, dtype=DT)
    cam = transform_points(Y, poses, log_scale.exp())
    pick = torch.randint(0, M, (T, N), generator=g)
    X = torch.gather(cam, 1, pick[..., None].expand(-1, -1, 3)) + 0.01 * torch.randn(T, N, 3, generator=g, dtype=DT)
    X_valid = torch.rand(T, N, generator=g) > 0.15
    X = torch.where(X_valid[..., None], X, torch.full_like(X, float("nan")))
    return (X.to(dtype), X_valid, Y.to(dtype), normals.to(dtype), poses.to(dtype), log_scale.to(dtype))


def _trial_poses(poses: Tensor, log_scale: Tensor, k: int) -> tuple[Tensor, Tensor]:
    g = torch.Generator().manual_seed(100 + k)
    xi = (0.03 * k * torch.randn(poses.shape[0], 6, generator=g, dtype=DT)).to(poses.dtype)
    return se3_exp(xi) @ poses, log_scale + 0.03 * k


def _assert_terms_close(got: TermOutput, want: TermOutput, rtol: float, atol: float) -> float:
    worst = 0.0
    for name in ("residual", "jacobian", "weight", "extra"):
        a, b = getattr(got, name), getattr(want, name)
        assert (a is None) == (b is None), name
        if a is None:
            continue
        assert a.shape == b.shape and a.dtype == b.dtype, name
        assert torch.allclose(a, b, rtol=rtol, atol=atol), (name, float((a - b).abs().max()))
        worst = max(worst, float(((a - b).abs() / (b.abs() + atol / rtol)).max()))
    return worst


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("knn", [None, 12])
@pytest.mark.parametrize("point_to_plane", [False, True])
@pytest.mark.parametrize("per_frame", [False, True])
def test_moments_term_equals_plan_term_at_trial_poses(dtype: torch.dtype, knn: int | None, point_to_plane: bool,
                                                      per_frame: bool) -> None:
    X, X_valid, Y, normals, poses, log_scale = _scene(3, 40, 64, dtype, seed=7, per_frame=per_frame)
    eps, rho_x, rho_y = 1e-2, 0.1, 1e-2
    kw = {"normals": normals, "point_to_plane": point_to_plane, "with_deficiency": True}
    out_plan, plan = pointmap_term(X, X_valid, Y, poses, log_scale, eps, rho_x, rho_y, 80, knn=knn, **kw)
    assert isinstance(plan, SparsePlan if knn else Tensor)
    out_mom, mom = pointmap_term(X, X_valid, Y, poses, log_scale, eps, rho_x, rho_y, 80, knn=knn, moments=True, **kw)
    assert isinstance(mom, PlanMoments) and (mom.normal is not None) == point_to_plane
    assert mom.bary.shape == (3, 64, 3) and mom.mass.shape == (3, 64)
    rtol, atol = (1e-10, 1e-13) if dtype == torch.float64 else (1e-5, 1e-6)
    worst = _assert_terms_close(out_mom, out_plan, rtol, atol)          # the refresh itself
    for k in (1, 2, 3):                                                  # trial poses away from the refresh
        p_k, ls_k = _trial_poses(poses, log_scale, k)
        want, _ = pointmap_term(X, X_valid, Y, p_k, ls_k, eps, rho_x, rho_y, 0, plan=plan, **kw)
        got, same = pointmap_term(X, X_valid, Y, p_k, ls_k, eps, rho_x, rho_y, 0, plan=mom, **kw)
        assert same is mom                                               # passed through, like a cached plan
        worst = max(worst, _assert_terms_close(got, want, rtol, atol))
    logger.info("moments vs plan (%s, knn %s, p2p %s, per-frame %s): worst relative difference %.2e", dtype, knn,
                point_to_plane, per_frame, worst)


def test_moments_mass_and_deficiency_are_exact() -> None:
    """w is the same `P.sum(-1)` reduction, so the deficiency (unbalanced_merit) is bit-identical."""
    X, X_valid, Y, _normals, poses, log_scale = _scene(2, 32, 48, DT, seed=3, per_frame=False)
    for knn in (None, 8):
        _out, plan = pointmap_term(X, X_valid, Y, poses, log_scale, 1e-2, 0.1, 1e-2, 60, knn=knn,
                                   with_deficiency=True)
        mom = plan_moments(plan, Y)
        w = plan.values.sum(2) if isinstance(plan, SparsePlan) else plan.sum(2)
        assert torch.equal(mom.mass, w)
        p_k, ls_k = _trial_poses(poses, log_scale, 2)
        a, _ = pointmap_term(X, X_valid, Y, p_k, ls_k, 1e-2, 0.1, 1e-2, 0, plan=plan, with_deficiency=True)
        b, _ = pointmap_term(X, X_valid, Y, p_k, ls_k, 1e-2, 0.1, 1e-2, 0, plan=mom, with_deficiency=True)
        assert a.extra is not None and b.extra is not None and torch.equal(a.extra, b.extra)


def test_zero_and_sub_clamp_mass_rows_mirror_the_barycenter_convention() -> None:
    """`plan_barycenters` divides by max(w, 1e-12): a zero-mass row transports to 0 (not t) and a row with
    0 < w < 1e-12 to (w / 1e-12) of its barycenter. `transported_points` reproduces both, dense and sparse."""
    g = torch.Generator().manual_seed(11)
    T, N, M, k = 2, 6, 10, 4
    Y = torch.randn(M, 3, generator=g, dtype=DT)
    poses = se3_exp(torch.randn(T, 6, generator=g, dtype=DT))
    s = torch.tensor([0.7, 1.3], dtype=DT)
    P = torch.rand(T, N, M, generator=g, dtype=DT)
    P[0, 1] = 0.0                                                        # zero mass
    P[1, 2] *= 1e-15                                                     # below the clamp
    P[1, 3] = 0.0
    P[1, 3, 4] = 1e-13                                                   # one entry below the clamp
    want, w = plan_barycenters(P, transform_points(Y, poses, s))
    got = transported_points(plan_moments(P, Y), poses, s)
    assert float(w[0, 1]) == 0.0 and float(want[0, 1].abs().max()) == 0.0 and float(got[0, 1].abs().max()) == 0.0
    assert torch.allclose(got, want, rtol=1e-12, atol=1e-15), float((got - want).abs().max())
    idx = torch.stack([torch.randperm(M, generator=g)[:k] for _ in range(T * N)]).reshape(T, N, k)
    values = torch.gather(P, 2, idx)
    want_s, _ = plan_barycenters_sparse(values, idx, transform_points(Y, poses, s))
    got_s = transported_points(plan_moments(SparsePlan(indices=idx, values=values), Y), poses, s)
    assert float(got_s[0, 1].abs().max()) == 0.0
    assert torch.allclose(got_s, want_s, rtol=1e-12, atol=1e-15), float((got_s - want_s).abs().max())


@pytest.mark.parametrize("point_to_plane", [False, True])
def test_moments_jacobian_matches_finite_differences(point_to_plane: bool) -> None:
    """The analytic Jacobian of the moments path is the derivative of its own residual (plan held fixed)."""
    X, X_valid, Y, normals, poses, log_scale = _scene(3, 40, 40, DT, seed=5, per_frame=False)
    X_valid = torch.ones_like(X_valid)
    X = transform_points(Y, poses, log_scale.exp())[:, :40] + 0.005
    kw = {"normals": normals, "point_to_plane": point_to_plane}
    out, mom = pointmap_term(X, X_valid, Y, poses, log_scale, 1e-3, float("inf"), float("inf"), 200, moments=True,
                             **kw)
    assert isinstance(mom, PlanMoments)

    def res(p: Tensor, s: Tensor) -> Tensor:
        return pointmap_term(X, X_valid, Y, p, s, 1e-3, float("inf"), float("inf"), 0, plan=mom, **kw)[0].residual

    _fd_check(res, out.jacobian, poses, log_scale, atol=1e-5)


def test_point_to_plane_refuses_moments_without_normals() -> None:
    X, X_valid, Y, normals, poses, log_scale = _scene(2, 24, 32, DT, seed=2, per_frame=False)
    _, mom = pointmap_term(X, X_valid, Y, poses, log_scale, 1e-2, 0.1, 1e-2, 20, moments=True)
    assert isinstance(mom, PlanMoments) and mom.normal is None
    with pytest.raises(ValueError, match="PlanMoments reduced with normals"):
        pointmap_term(X, X_valid, Y, poses, log_scale, 1e-2, 0.1, 1e-2, 0, plan=mom, normals=normals,
                      point_to_plane=True)


# --- the batched evaluate callback -------------------------------------------------------------------------------

N_OBS, N_MESH = 96, 192


def _cache_tensors(cache: EvalCache) -> list[Tensor]:
    plan = cache.plan
    parts: list[Tensor | None] = (
        [plan.bary, plan.mass, plan.normal] if isinstance(plan, PlanMoments)
        else [plan.indices, plan.values] if isinstance(plan, SparsePlan) else [plan])
    return [x for x in (*parts, *(cache.potentials or ()), cache.mask) if x is not None]


@pytest.mark.parametrize(("correspondence", "level_index", "chunk_frames", "unbalanced"),
                         [("ot", 0, None, False), ("ot", 2, None, False), ("ot", 0, 4, False), ("ot", 2, 4, True),
                          ("ot", 1, 4, True), ("nn", 1, None, False)])
def test_evaluate_with_moments_matches_plan_cache(monkeypatch: pytest.MonkeyPatch, correspondence: str,
                                                  level_index: int, chunk_frames: int | None, unbalanced: bool) -> None:
    if chunk_frames is not None:
        monkeypatch.setattr(batch_eval, "_REFRESH_ELEMENTS", chunk_frames * N_OBS * N_MESH)
    off = SolverConfig(plan_moments=False, n_obs=N_OBS, n_mesh=N_MESH, device=DEVICE, sinkhorn_float32=False,
                       correspondence=correspondence, unbalanced_merit=unbalanced)
    on = replace(off, plan_moments=True)
    db = prepare_batch(pack_clips(*_unzip([_inputs(_clip(T, N_MESH, seed=140 + T), seed=T) for T in (5, 6, 7)])), off)
    T, M = db.layout.n_frames, db.Y.shape[1]
    level = _levels(off)[level_index][0]
    gen = torch.Generator().manual_seed(level_index)
    xi = (0.01 * torch.randn(T, 6, generator=gen, dtype=torch.float64)).to(DEVICE)
    cache_off, cache_on = EvalCache(), EvalCache()
    ev_off, ev_on = make_evaluate(db, off, level, cache_off), make_evaluate(db, on, level, cache_on)
    frames = db.layout.frames_of([0, 2])
    ls = db.log_scale0 + 0.01
    moved = se3_exp(xi) @ db.poses0
    steps: list[tuple[bool, Tensor, Tensor | None]] = [
        (True, db.poses0, None), (False, moved, None), (True, moved, frames), (False, se3_exp(0.5 * xi) @ moved, frames),
        (False, se3_exp(-xi) @ moved, None)]
    for refresh, poses, sub in steps:
        (pm_off, tr_off), _ = ev_off(poses, ls, refresh, sub)
        (pm_on, tr_on), _ = ev_on(poses, ls, refresh, sub)
        _assert_terms_close(pm_on, pm_off, 1e-9, 1e-12)
        assert torch.equal(tr_on.residual, tr_off.residual)
        assert isinstance(cache_on.plan, PlanMoments) and cache_on.plan.normal is None
        assert isinstance(cache_off.plan, SparsePlan if level.knn and correspondence == "ot" else Tensor)
        for x in _cache_tensors(cache_on):                               # no plan-shaped tensor survives a refresh
            assert x.shape[:2] != (T, N_OBS) or x.dim() < 3 or x.shape[2] == 3, x.shape
            assert x.numel() < T * N_OBS * min(M, level.knn or M)
        w_off = (cache_off.plan.values.sum(2) if isinstance(cache_off.plan, SparsePlan)
                 else cache_off.plan.sum(2))
        assert torch.equal(cache_on.plan.mass, w_off)


def test_cache_holds_the_plan_when_off() -> None:
    cfg = SolverConfig(plan_moments=False, n_obs=N_OBS, n_mesh=N_MESH, device=DEVICE, sinkhorn_float32=False)
    assert not cfg.plan_moments
    db = prepare_batch(pack_clips(*_unzip([_inputs(_clip(6, N_MESH, seed=150), seed=1)])), cfg)
    for level_index, kind in ((0, Tensor), (2, SparsePlan)):
        cache = EvalCache()
        make_evaluate(db, cfg, _levels(cfg)[level_index][0], cache)(db.poses0, db.log_scale0, True)
        assert isinstance(cache.plan, kind) and not isinstance(cache.plan, PlanMoments)
        if kind is Tensor:
            assert isinstance(cache.plan, Tensor) and cache.plan.shape == (6, N_OBS, N_MESH)


# --- the solve ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize(("float32", "chunk_frames"), [(False, 4), (True, None)])
def test_batched_solve_with_moments_matches_plan_solve(monkeypatch: pytest.MonkeyPatch, float32: bool,
                                                       chunk_frames: int | None) -> None:
    """Three clips (one occluded), float64 LM with a float64 Sinkhorn in 4-frame chunks or the shipped float32
    Sinkhorn in one chunk: identical LM acceptance histories and iteration counts, poses and scales to round-off (the
    returned poses are float32)."""
    if chunk_frames is not None:
        monkeypatch.setattr(batch_eval, "_REFRESH_ELEMENTS", chunk_frames * 128 * 256)
    inputs = [_inputs(_clip(T, 256, seed=160 + T, occlusion=occ), seed=T) for T, occ in ((7, 0.0), (9, 0.5), (11, 0.0))]
    off = SolverConfig(plan_moments=False, n_obs=128, n_mesh=256, device=DEVICE, sinkhorn_float32=float32)
    base = solve_trajectories(*_unzip(inputs), off)
    mom = solve_trajectories(*_unzip(inputs), replace(off, plan_moments=True))
    for i, (a, b) in enumerate(zip(base, mom, strict=True)):
        # `TrajectorySolution.poses` is float32 (the se3_log of a float32 product reads ~1e-7 m even for identical
        # matrices, so the entries are compared directly); 1e-6 is a few float32 ulps of a ~0.5 m translation.
        d_pose = float((a.poses - b.poses).abs().max())
        logger.info("clip %d moments-vs-plan: max pose entry %.2e, scale %.2e, confidence %.2e, accepted steps %s, "
                    "fine iterations %d", i, d_pose, abs(a.scale - b.scale),
                    float((a.confidence - b.confidence).abs().max()), [len(x) for x in a.accepted], a.iterations)
        assert b.accepted == a.accepted and b.iterations == a.iterations, i
        assert d_pose < 1e-6 and abs(a.scale - b.scale) < 1e-9, (i, d_pose, a.scale, b.scale)
        assert torch.allclose(b.confidence, a.confidence, atol=1e-6)
        assert torch.allclose(b.per_frame_cost, a.per_frame_cost, rtol=1e-6, atol=1e-9)


# --- the memory model --------------------------------------------------------------------------------------------

@pytest.mark.parametrize(("lengths", "n_obs", "n_mesh", "float32", "chunk_frames"),
                         [((6,), 128, 256, True, None), ((5, 6, 7), 128, 256, True, None),
                          ((5, 6, 7), 128, 256, False, 2), ((6,), 256, 2048, False, 2), ((6,), 256, 2048, True, 2)])
def test_moments_estimate_bounds_measured_cpu_peak(monkeypatch: pytest.MonkeyPatch, lengths: tuple[int, ...],
                                                   n_obs: int, n_mesh: int, float32: bool,
                                                   chunk_frames: int | None) -> None:
    """`memory.batch_peak_bytes` with `plan_moments` still bounds the allocator peak of a real solve (the chunking
    decisions it drives stay safe), and it is below the plan model wherever a plan was held beside a chunk. A one-shot
    refresh of one clip is Sinkhorn-bound either way."""
    inputs = [_inputs(_clip(T, n_mesh, seed=170 + T, image_hw=(120, 160), n_tracks=16), seed=T) for T in lengths]
    off = SolverConfig(plan_moments=False, n_obs=n_obs, n_mesh=n_mesh, iters_coarse=4, iters_mid=2, iters_fine=3, device="cpu",
                       sinkhorn_float32=float32)
    on = replace(off, plan_moments=True)
    refresh = (1 << 28) if chunk_frames is None else chunk_frames * n_obs * n_mesh
    measured = _measured_peak(monkeypatch, inputs, on, refresh)
    dims = clip_dims(*_unzip(inputs)[:3], on)
    estimate, plan_model = batch_peak_bytes(dims, on, refresh), batch_peak_bytes(dims, off, refresh)
    logger.info("moments peak: measured %.1f MiB, estimate %.1f MiB (plan model %.1f MiB)", measured / 2**20,
                estimate / 2**20, plan_model / 2**20)
    assert measured <= estimate <= 2.0 * measured, (measured, estimate, estimate / measured)
    if len(lengths) > 1 or chunk_frames is not None:                     # a one-shot refresh keeps its bound
        assert estimate < plan_model
