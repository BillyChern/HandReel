"""Anisotropic similarity registration (`SolverConfig.anisotropic_scale`, `fastdo.solver.aniso`).

Flag off, byte identity (a). Every anisotropic op sits behind a branch the flag-off path never takes: a (B,) log-scale
state, a 7-column Jacobian, `stretch=None` and `DeviceBatch.shape_axes is None`. The tests below pin that down inside
this tree: the omitted field solves bit-identically to an explicit False, no stretch code runs, and `solve.npz` keeps
its historical keys. A stored digest was rejected as the permanent check: it binds to one torch/BLAS build and thread
count, and any later intentional change to the solver would break it. The comparison against the tree BEFORE this
change was run once by hand. It solved the same fixtures (B = 1 and 3; dense-plan chunked and
moments chunked refreshes; nn correspondences; visibility off with unbalanced merit, IRLS merit, track validity, dense
fine level and float64 Sinkhorn; no velocity prior; `score_trajectory`) under both trees and compares every output
with `torch.equal`: 121/121 tensors were identical.

Flag on: (b) an isotropic truth solves to equal l on every axis and to flag-off's poses and scale; (c) a template
whose true object is stretched (1.3, 1.0, 0.8) in its principal frame, seen tumbling, recovers l and cuts the pose
error; (d) the stretch Jacobian columns match finite differences in every term; (e) the plan moments equal the plan
under the stretch; (f) an axis the clip never measures stays at mean l under the prior. Also: the bordered linear
algebra against a dense solve, the principal frame's determinism, the output convention and the memory estimate.

The recovery fixtures mirror the box template over its three symmetry planes, so its principal frame is exactly the
box's axes and the truth lies inside the model. With 768 random samples the sampled axes are ~5 deg off, and a
stretch along the box's axes is then not representable, which biased l by up to 0.04 and rotation by 2 deg.
"""

from __future__ import annotations

import math
from dataclasses import fields, replace

import pytest
import torch
from torch import Tensor

from fastdo.engine.workers_qc import read_solve_artifact, write_solve_artifact
from fastdo.geometry.lie import apply_left, se3_exp, se3_inv, se3_log
from fastdo.solver import aniso, api, batch, batch_eval, synthetic, terms
from fastdo.solver.aniso import (
    Stretch,
    corrected_template_normals,
    corrected_template_points,
    principal_frame,
    stretch_jacobian,
)
from fastdo.solver.api import (
    MeshSamples,
    SolverConfig,
    TrajectoryObservations,
    _levels,
    solve_trajectories,
)
from fastdo.solver.banded import (
    BlockTridiagonal,
    ClipLayout,
    densify,
    marginal_covariances,
    solve_block_tridiagonal,
)
from fastdo.solver.batch import pack_clips, prepare_batch
from fastdo.solver.batch_eval import EvalCache, make_evaluate
from fastdo.solver.lm import clip_costs
from fastdo.solver.memory import batch_peak_bytes
from fastdo.solver.moments import plan_moments
from fastdo.solver.ot_sparse import SparsePlan
from fastdo.solver.terms import pointmap_term, track_term
from tests.solver.test_batch import _clip, _inputs, _unzip
from tests.solver.test_memory import _measured_peak

_SMALL = {"n_obs": 128, "n_mesh": 256, "device": "cpu"}
_FIX = {"n_obs": 192, "n_mesh": 512, "device": "cpu"}
_SOLUTION_TENSORS = ("poses", "covariance", "per_frame_cost", "confidence")


def _small_inputs() -> list:
    return [_inputs(_clip(T, 256, seed=170 + T, image_hw=(120, 160), n_tracks=16), seed=T) for T in (5, 7)]


def _assert_same_solutions(a: list, b: list) -> None:
    for x, y in zip(a, b, strict=True):
        for name in _SOLUTION_TENSORS:
            assert torch.equal(getattr(x, name), getattr(y, name)), name
        assert x.scale == y.scale and x.iterations == y.iterations and x.accepted == y.accepted


# ---------------------------------------------------------------------------------------------------------------- (a)

def test_flag_off_is_the_default_and_identical_to_an_explicit_false() -> None:
    assert SolverConfig().anisotropic_scale is False
    inputs = _small_inputs()
    omitted = solve_trajectories(*_unzip(inputs), SolverConfig(**_SMALL))
    explicit = solve_trajectories(*_unzip(inputs), SolverConfig(**_SMALL, anisotropic_scale=False, lam_aniso=123.0))
    _assert_same_solutions(omitted, explicit)
    assert all(s.shape_log_stretch is None and s.shape_axes is None and s.shape_centroid is None for s in omitted)


def test_flag_off_never_runs_a_stretch_op(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every anisotropic entry point raises; the flag-off solve (B = 2, then the dense-plan and nn paths) and
    `score_trajectory` still run and reproduce the unpatched results bit for bit."""
    inputs = _small_inputs()
    configs = [SolverConfig(**_SMALL), SolverConfig(**_SMALL, plan_moments=False, correspondence="nn")]
    before = [solve_trajectories(*_unzip(inputs), cfg) for cfg in configs]
    m, o, p, s = inputs[0]
    score_before = api.score_trajectory(m, o, p, s, configs[0])

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("an anisotropic-scale op ran with anisotropic_scale off")

    for owner, name in ((batch, "principal_frame"), (batch, "_aniso_fields"), (batch_eval, "prior_residuals"),
                        (batch_eval.FrameInputs, "stretch"), (terms, "stretch_jacobian"), (api, "_isotropic_outputs"),
                        (aniso.Stretch, "apply"), (aniso.Stretch, "apply_normals"), (aniso.Stretch, "matrix")):
        monkeypatch.setattr(owner, name, forbidden)
    db = prepare_batch(pack_clips(*_unzip(inputs)), configs[0])
    assert db.shape_axes is None and db.shape_centroid is None and db.aniso_w is None and db.log_scale0.dim() == 1
    for cfg, want in zip(configs, before, strict=True):
        _assert_same_solutions(solve_trajectories(*_unzip(inputs), cfg), want)
    for got, want in zip(api.score_trajectory(m, o, p, s, configs[0]), score_before, strict=True):
        assert torch.equal(got, want)


def test_solve_npz_gains_the_shape_fields_only_with_the_flag(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import numpy as np

    m, o, p, s = _small_inputs()[0]
    historical = {"poses", "scale", "covariance", "per_frame_cost", "confidence", "iterations", "seconds", "frame_idx",
                  "track_weight"}
    for flag in (False, True):
        sol = solve_trajectories([m], [o], [p], [s], SolverConfig(**_SMALL, anisotropic_scale=flag))[0]
        out = tmp_path / str(flag)
        write_solve_artifact(out, sol, np.arange(p.shape[0]), torch.zeros(p.shape[0], 3))
        with np.load(out / "solve.npz") as data:
            keys = set(data.files)
        extra = {"shape_log_stretch", "shape_axes", "shape_centroid"} if flag else set()
        assert keys == historical | extra
        back = read_solve_artifact(out).solution
        for name in ("shape_log_stretch", "shape_axes", "shape_centroid"):
            got, want = getattr(back, name), getattr(sol, name)
            assert (got is None and want is None) or torch.equal(got, want.double())


# ---------------------------------------------------------------------------------------------------------------- fixtures

def _rot(axis: int, angle: float) -> Tensor:
    c, s = math.cos(angle), math.sin(angle)
    i, j = [(1, 2), (2, 0), (0, 1)][axis]
    R = torch.eye(3, dtype=torch.float64)
    R[i, i], R[i, j], R[j, i], R[j, j] = c, -s, s, c
    return R


def _trajectory(n: int, kind: str) -> Tensor:
    """`tumble`: +-55 deg about the camera's y axis and +-35 deg about x, so every box face is seen extended. `planar`:
    rotation about the optical axis only (the front face always faces the camera) and translation in the image plane."""
    poses = []
    for k in range(n):
        u = k / (n - 1)
        if kind == "tumble":
            R = _rot(1, math.radians(-55 + 110 * u)) @ _rot(0, math.radians(35 * math.sin(2 * math.pi * u)))
            t = [0.02 * math.sin(2 * math.pi * u), 0.01 * math.cos(2 * math.pi * u), 0.55]
        else:
            R = _rot(2, math.radians(-40 + 80 * u))
            t = [0.03 * math.sin(2 * math.pi * u), 0.02 * u - 0.01, 0.55]
        T = torch.eye(4, dtype=torch.float64)
        T[:3, :3], T[:3, 3] = R, torch.tensor(t, dtype=torch.float64)
        poses.append(T)
    return torch.stack(poses).float()


def _stretched_box(monkeypatch: pytest.MonkeyPatch, kind: str, n: int, sigma: tuple[float, float, float],
                   depth_noise: float = 0.002) -> tuple[synthetic.SyntheticClip, MeshSamples, TrajectoryObservations]:
    """A synthetic box clip on a fixed `_trajectory` whose TRUE object is the template stretched by `sigma` along the
    template's principal axes (x, y, z: the template's extents stay ordered). The template is the clip's samples
    mirrored over the box's symmetry planes (module docstring), divided by sigma; normals by the inverse transpose."""
    monkeypatch.setattr(synthetic, "_smooth_trajectory", lambda n_, *_args: _trajectory(n_, kind))
    clip = synthetic.make_synthetic_clip(synthetic.SyntheticConfig(
        n_frames=n, image_hw=(160, 214), focal=200.0, n_surface=768, n_tracks=32, depth_noise=depth_noise,
        shape="box"), seed=5)
    sig = torch.tensor(sigma)
    flips = torch.tensor([[a, b, c] for a in (1.0, -1.0) for b in (1.0, -1.0) for c in (1.0, -1.0)])
    points = (clip.surface_points[None] * flips[:, None]).reshape(-1, 3) / sig
    normals = torch.nn.functional.normalize((clip.surface_normals[None] * flips[:, None]).reshape(-1, 3) * sig, dim=-1)
    obs = TrajectoryObservations(intrinsics=clip.intrinsics, masks=clip.masks, hand_masks=clip.hand_masks,
                                 pointmaps=clip.pointmaps, tracks=clip.tracks, track_vis=clip.track_vis,
                                 anchors=clip.anchors / sig)
    return clip, MeshSamples(points, normals), obs


def _perturbed(poses: Tensor, seed: int = 1, trans: float = 0.02, rot_deg: float = 8.0) -> Tensor:
    g = torch.Generator().manual_seed(seed)
    xi = torch.zeros(poses.shape[0], 6)
    xi[:, :3] = trans * torch.nn.functional.normalize(torch.randn(poses.shape[0], 3, generator=g), dim=1)
    xi[:, 3:] = math.radians(rot_deg) * torch.nn.functional.normalize(torch.randn(poses.shape[0], 3, generator=g), dim=1)
    return se3_exp(xi.double()).float() @ poses


def _pose_errors(poses: Tensor, truth: Tensor) -> tuple[float, float]:
    """Mean object-origin translation error (mm) and mean rotation error (deg) over frames."""
    rel = se3_log(poses.double() @ se3_inv(truth.double()))
    dt = (poses[:, :3, 3].double() - truth[:, :3, 3].double()).norm(dim=1)
    return float(dt.mean()) * 1e3, math.degrees(float(rel[:, 3:].norm(dim=1).mean()))


def _solve(mesh: MeshSamples, obs: TrajectoryObservations, init: Tensor, s0: float, **cfg: object):  # type: ignore[no-untyped-def]
    return solve_trajectories([mesh], [obs], [init], [s0], SolverConfig(**_FIX, **cfg))[0]  # type: ignore[arg-type]


# ---------------------------------------------------------------------------------------------------------------- (b)

def test_isotropic_truth_solves_isotropic_and_matches_flag_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """Measured: l = (-2.1262, -2.1262, -2.1261), scale 0.1193 in both arms, pose errors 3.27 / 3.28 mm, 0.016 deg."""
    clip, mesh, obs = _stretched_box(monkeypatch, "tumble", 16, (1.0, 1.0, 1.0))
    init, s0 = _perturbed(clip.poses_gt), clip.scale_gt * 1.1
    off = _solve(mesh, obs, init, s0)
    on = _solve(mesh, obs, init, s0, anisotropic_scale=True)
    ell = on.shape_log_stretch
    assert ell is not None and float(ell.max() - ell.min()) < 2e-3, ell
    assert abs(on.scale / off.scale - 1.0) < 2e-3 and abs(on.scale / clip.scale_gt - 1.0) < 0.02, (on.scale, off.scale)
    assert float((on.poses[:, :3, 3] - off.poses[:, :3, 3]).norm(dim=1).max()) < 1e-3
    assert math.degrees(float(se3_log(on.poses.double() @ se3_inv(off.poses.double()))[:, 3:].norm(dim=1).max())) < 0.1


# ---------------------------------------------------------------------------------------------------------------- (c)

def test_recovers_an_anisotropic_stretch_and_the_poses(monkeypatch: pytest.MonkeyPatch) -> None:
    """sigma = (1.3, 1.0, 0.8). Measured (16 frames, 192 observed points): true l = log(0.12) + log sigma =
    (-1.8579, -2.1203, -2.3434), recovered (-1.8638, -2.1262, -2.3492), a common -0.006 that the isotropic solve of
    the same fixture shares (-0.006 in (b)); pose error 3.2 mm / 0.013 deg on against 22.4 mm / 10.6 deg off."""
    sigma = (1.3, 1.0, 0.8)
    clip, mesh, obs = _stretched_box(monkeypatch, "tumble", 16, sigma)
    init, s0 = _perturbed(clip.poses_gt), clip.scale_gt * 1.1
    off = _solve(mesh, obs, init, s0)
    on = _solve(mesh, obs, init, s0, anisotropic_scale=True)
    truth = torch.tensor([math.log(clip.scale_gt * s) for s in sigma], dtype=torch.float64)
    assert on.shape_log_stretch is not None and on.shape_axes is not None
    assert torch.allclose(on.shape_axes.abs(), torch.eye(3, dtype=torch.float64), atol=1e-9)
    assert float((on.shape_log_stretch - truth).abs().max()) < 0.015, (on.shape_log_stretch, truth)
    assert abs(math.log(on.scale) - float(truth.mean())) < 0.015
    t_on, r_on = _pose_errors(on.poses, clip.poses_gt)
    t_off, r_off = _pose_errors(off.poses, clip.poses_gt)
    assert t_on < 5.0 and r_on < 0.1, (t_on, r_on)
    assert t_on < 0.3 * t_off and r_on < 0.05 * r_off, (t_on, t_off, r_on, r_off)


# ---------------------------------------------------------------------------------------------------------------- (f)

def test_prior_holds_an_axis_the_clip_never_measures(monkeypatch: pytest.MonkeyPatch) -> None:
    """A planar trajectory that only ever shows the front face (side-face pixels and anchors dropped, exact depths):
    the depth stretch l_z moves only the front face along the camera ray, which the translation absorbs frame for
    frame, so the data does not measure it. The truth has sigma_z = 0.75; the solve keeps l_z at mean l instead.
    Measured: l = (-2.1203, -2.1203, -2.1292), |l_z - mean l| 0.006 (0.021 at lam_aniso = 0, where only the weak
    mesh-side pull of the visible template's side faces moves it), true l_z -2.408."""
    sigma = (1.0, 1.0, 0.75)
    clip, mesh, obs = _stretched_box(monkeypatch, "planar", 16, sigma, depth_noise=0.0)
    R, t = clip.poses_gt[:, :3, :3], clip.poses_gt[:, :3, 3]
    canonical = torch.einsum("tji,thwj->thwi", R, clip.pointmaps - t[:, None, None, :]) / clip.scale_gt
    front = clip.masks & (canonical[..., 2] < -0.2 + 2e-3)                 # the true box's z = -0.2 face
    pointmaps = torch.where(front[..., None], clip.pointmaps, torch.full_like(clip.pointmaps, float("nan")))
    on_face = (clip.anchors[:, 2] < -0.2 + 1e-3)[None]
    obs = replace(obs, masks=front, pointmaps=pointmaps, track_vis=obs.track_vis & on_face)
    init, s0 = _perturbed(clip.poses_gt), clip.scale_gt * 1.1
    held = _solve(mesh, obs, init, s0, anisotropic_scale=True)
    free = _solve(mesh, obs, init, s0, anisotropic_scale=True, lam_aniso=0.0)
    true_xy = math.log(clip.scale_gt)
    for sol in (held, free):
        assert sol.shape_log_stretch is not None
        assert float((sol.shape_log_stretch[:2] - true_xy).abs().max()) < 0.01, sol.shape_log_stretch
    ell = held.shape_log_stretch
    assert ell is not None and free.shape_log_stretch is not None
    gap = abs(float(ell[2] - ell.mean()))
    assert gap < 0.012, ell
    assert gap < 0.5 * abs(float(free.shape_log_stretch[2] - free.shape_log_stretch.mean())), (ell, free.shape_log_stretch)
    assert abs(float(ell[2]) - math.log(clip.scale_gt * sigma[2])) > 0.2        # the depth really was not measured


# ---------------------------------------------------------------------------------------------------------------- (d)

def _random_stretch(T: int, gen: torch.Generator) -> Stretch:
    U, _ = torch.linalg.qr(torch.randn(3, 3, generator=gen, dtype=torch.float64))
    return Stretch(axes=U.expand(T, 3, 3).clone(), centroid=0.05 * torch.randn(3, generator=gen, dtype=torch.float64)
                   .expand(T, 3).clone(), log_stretch=math.log(0.12) + 0.2 * torch.randn(T, 3, generator=gen,
                                                                                            dtype=torch.float64))


def _random_poses(T: int, gen: torch.Generator) -> Tensor:
    xi = 0.3 * torch.randn(T, 6, generator=gen, dtype=torch.float64)
    poses = se3_exp(xi)
    poses[:, 2, 3] += 0.6
    return poses


def _fd_columns(residual, poses: Tensor, st: Stretch, h: float = 1e-6) -> Tensor:  # type: ignore[no-untyped-def]
    """Central differences of `residual(poses, stretch)` (T,n,d) over [rho, theta, l]: (T,n,d,9)."""
    cols = []
    for i in range(9):
        e = torch.zeros(poses.shape[0], 9, dtype=torch.float64)
        e[:, i] = h
        plus = residual(apply_left(poses, e[:, :6]), replace(st, log_stretch=st.log_stretch + e[:, 6:]))
        minus = residual(apply_left(poses, -e[:, :6]), replace(st, log_stretch=st.log_stretch - e[:, 6:]))
        cols.append((plus - minus) / (2 * h))
    return torch.stack(cols, dim=-1)


def test_stretch_jacobian_matches_finite_differences() -> None:
    gen = torch.Generator().manual_seed(211)
    T, M, N, J = 3, 40, 30, 12
    st, poses = _random_stretch(T, gen), _random_poses(T, gen)
    Y = 0.5 * torch.randn(M, 3, generator=gen, dtype=torch.float64)

    def points(p: Tensor, s: Stretch) -> Tensor:
        return terms.transform_points(s.apply(Y), p, 1.0)

    p = points(poses, st)
    analytic = stretch_jacobian(p, poses[:, :3, 3], st.rotated_axes(poses))
    assert torch.allclose(analytic, _fd_columns(points, poses, st)[..., 6:], rtol=1e-6, atol=1e-8)
    assert torch.allclose(analytic.sum(-1), p - poses[:, None, :3, 3], atol=1e-12)   # sums to today's scale column

    X = p[:, :N] + 0.01 * torch.randn(T, N, 3, generator=gen, dtype=torch.float64)
    X_valid = torch.ones(T, N, dtype=torch.bool)
    plan = torch.rand(T, N, M, generator=gen, dtype=torch.float64) / (N * M)
    zero = torch.zeros(T, dtype=torch.float64)
    for fixed in (plan, plan_moments(plan, Y)):
        def pm_residual(pp: Tensor, s: Stretch, fixed=fixed) -> Tensor:  # type: ignore[no-untyped-def]
            return pointmap_term(X, X_valid, Y, pp, zero, 1.0, 1.0, 1.0, 5, plan=fixed, stretch=s)[0].residual

        term, _ = pointmap_term(X, X_valid, Y, poses, zero, 1.0, 1.0, 1.0, 5, plan=fixed, stretch=st)
        assert term.jacobian.shape == (T, N, 3, 9)
        assert torch.allclose(term.jacobian, _fd_columns(pm_residual, poses, st), rtol=1e-6, atol=1e-8)

    anchors = 0.5 * torch.randn(J, 3, generator=gen, dtype=torch.float64)
    K = torch.tensor([[300.0, 0, 160], [0, 300.0, 120], [0, 0, 1]], dtype=torch.float64)
    tracks = terms.project(terms.transform_points(st.apply(anchors), poses, 1.0), K) + 2.0
    vis = torch.ones(T, J, dtype=torch.bool)

    def tr_residual(pp: Tensor, s: Stretch) -> Tensor:
        return track_term(anchors, tracks, vis, pp, zero, K, 4.0, stretch=s).residual

    tr = track_term(anchors, tracks, vis, poses, zero, K, 4.0, stretch=st)
    assert tr.jacobian.shape == (T, J, 2, 9)
    assert torch.allclose(tr.jacobian, _fd_columns(tr_residual, poses, st), rtol=1e-6, atol=1e-6)


def test_point_to_plane_refuses_a_stretch() -> None:
    gen = torch.Generator().manual_seed(3)
    st, poses = _random_stretch(2, gen), _random_poses(2, gen)
    Y = torch.randn(8, 3, dtype=torch.float64)
    with pytest.raises(ValueError, match="point_to_plane"):
        pointmap_term(torch.randn(2, 5, 3, dtype=torch.float64), torch.ones(2, 5, dtype=torch.bool), Y, poses,
                      torch.zeros(2, dtype=torch.float64), 1.0, 1.0, 1.0, 5, plan=torch.rand(2, 5, 8,
                      dtype=torch.float64), normals=Y, point_to_plane=True, stretch=st)


# ---------------------------------------------------------------------------------------------------------------- (e)

def test_plan_moments_equal_the_plan_under_the_stretch() -> None:
    """pbar = R A (B - c) + t from the canonical moments equals the plan's barycenter of the stretched template, for a
    dense and a kNN plan, including a zero-mass row; then end to end, a trial at a new stretch from the cached moments
    equals the trial from the cached plan (`batch_eval`)."""
    gen = torch.Generator().manual_seed(204)
    T, M, N = 3, 50, 20
    st, poses = _random_stretch(T, gen), _random_poses(T, gen)
    Y = torch.randn(T, M, 3, generator=gen, dtype=torch.float64)
    X = torch.randn(T, N, 3, generator=gen, dtype=torch.float64) * 0.05 + poses[:, None, :3, 3]
    X_valid = torch.ones(T, N, dtype=torch.bool)
    dense = torch.rand(T, N, M, generator=gen, dtype=torch.float64) / (N * M)
    dense[0, 3] = 0.0                                                      # a zero-mass row
    idx = torch.topk(torch.rand(T, N, M, generator=gen), 7, dim=2).indices
    sparse = SparsePlan(indices=idx, values=torch.gather(dense, 2, idx))
    zero = torch.zeros(T, dtype=torch.float64)
    for plan in (dense, sparse):
        want, _ = pointmap_term(X, X_valid, Y, poses, zero, 1.0, 1.0, 1.0, 5, plan=plan, stretch=st)
        got, _ = pointmap_term(X, X_valid, Y, poses, zero, 1.0, 1.0, 1.0, 5, plan=plan_moments(plan, Y), stretch=st)
        assert torch.allclose(got.residual, want.residual, rtol=1e-10, atol=1e-12)
        assert torch.allclose(got.jacobian, want.jacobian, rtol=1e-10, atol=1e-12)

    inputs = _small_inputs()
    trial_terms = []
    for moments in (False, True):
        cfg = SolverConfig(**_SMALL, anisotropic_scale=True, plan_moments=moments)
        db = prepare_batch(pack_clips(*_unzip(inputs)), cfg)
        evaluate = make_evaluate(db, cfg, _levels(cfg)[1][0], EvalCache())
        evaluate(db.poses0, db.log_scale0, True)
        stretched = db.log_scale0 + torch.tensor([[0.05, -0.02, -0.04], [-0.03, 0.01, 0.02]], dtype=torch.float64)
        trial_terms.append(evaluate(db.poses0, stretched, False)[0])
    for got, want in zip(trial_terms[1], trial_terms[0], strict=True):
        assert torch.allclose(got.residual, want.residual, rtol=1e-9, atol=1e-12)
        assert torch.allclose(got.jacobian, want.jacobian, rtol=1e-9, atol=1e-12)


# ---------------------------------------------------------------------------------------------------------------- prior, LM

def test_prior_is_in_the_merit_and_zero_when_isotropic() -> None:
    inputs = _small_inputs()
    cfg = SolverConfig(**_SMALL, anisotropic_scale=True, lam_aniso=0.5)
    db = prepare_batch(pack_clips(*_unzip(inputs)), cfg)
    evaluate = make_evaluate(db, cfg, _levels(cfg)[0][0], EvalCache())
    ell = db.log_scale0 + torch.tensor([[0.1, 0.0, -0.1], [0.0, 0.0, 0.0]], dtype=torch.float64)
    found, _ = evaluate(db.poses0, ell, True)
    assert [t.jacobian.shape[-1] for t in found] == [9, 9, 9]
    prior = clip_costs(found[2:], None, db.layout)
    lam_b = torch.stack([db.aniso_w[sl].sum() for sl in db.layout.frame_slices()])   # type: ignore[index]
    dev = ((ell - ell.mean(1, keepdim=True)) ** 2).sum(1)
    assert torch.allclose(prior, lam_b * dev, rtol=1e-12) and float(prior[1]) < 1e-25     # P l is 0 up to round-off
    supports = [batch.prior_support(db.X_valid[sl]) for sl in db.layout.frame_slices()]
    want = [0.5 * sup * s2 for sup, s2 in zip(supports, db.scale2, strict=True)]
    assert torch.allclose(lam_b, torch.tensor(want, dtype=torch.float64), rtol=1e-12)


def _bordered_system(lengths: tuple[int, ...], velocity: bool, gen: torch.Generator) -> BlockTridiagonal:
    layout = ClipLayout.build(lengths, "cpu")
    T, B, K = sum(lengths), len(lengths), 3
    G = torch.randn(T, 6, 6, generator=gen, dtype=torch.float64)
    D = G @ G.mT + 6.0 * torch.eye(6, dtype=torch.float64)
    U = torch.zeros(T - 1, 6, 6, dtype=torch.float64)
    U[layout.pair_rows] = 0.3 * torch.randn(layout.pair_rows.shape[0], 6, 6, generator=gen, dtype=torch.float64)
    U2 = None
    if velocity:
        U2 = torch.zeros(T - 2, 6, 6, dtype=torch.float64)
        U2[layout.triple_rows] = 0.2 * torch.randn(layout.triple_rows.shape[0], 6, 6, generator=gen, dtype=torch.float64)
    Hs = torch.randn(B, K, K, generator=gen, dtype=torch.float64)
    return BlockTridiagonal(D=D, U=U, c=0.3 * torch.randn(T, 6, K, generator=gen, dtype=torch.float64),
                            h_ss=Hs @ Hs.mT + 30.0 * torch.eye(K, dtype=torch.float64),
                            g=torch.randn(T, 6, generator=gen, dtype=torch.float64),
                            g_s=torch.randn(B, K, generator=gen, dtype=torch.float64), U2=U2, layout=layout)


@pytest.mark.parametrize("velocity", [False, True])
def test_bordered_solve_and_covariances_match_the_dense_system(velocity: bool) -> None:
    """The 3-column stretch border through block-Thomas + the per-clip 3x3 Schur complement equals the dense damped
    solve (per-clip damping), and the marginal covariances equal the dense inverse's diagonal blocks."""
    gen = torch.Generator().manual_seed(7)
    lengths = (4, 5)
    sys = _bordered_system(lengths, velocity, gen)
    T, B, K = sum(lengths), len(lengths), 3
    H, g = densify(sys)
    lam = torch.tensor([0.1, 0.02], dtype=torch.float64)
    clip_of = torch.repeat_interleave(torch.arange(B), torch.tensor(lengths))
    lam_rows = torch.cat([lam[clip_of].repeat_interleave(6), lam.repeat_interleave(K)])
    A = H + torch.diag(lam_rows * torch.diagonal(H)) + 1e-9 * torch.eye(H.shape[0], dtype=torch.float64)
    want = torch.linalg.solve(A, -g)
    d_pose, d_l = solve_block_tridiagonal(sys, lam)
    assert d_l.shape == (B, K)
    assert torch.allclose(d_pose.reshape(-1), want[:6 * T], rtol=1e-9, atol=1e-11)
    assert torch.allclose(d_l.reshape(-1), want[6 * T:], rtol=1e-9, atol=1e-11)
    inv = torch.linalg.inv(H + 1e-9 * torch.eye(H.shape[0], dtype=torch.float64))
    blocks = torch.stack([inv[6 * k:6 * k + 6, 6 * k:6 * k + 6] for k in range(T)])
    assert torch.allclose(marginal_covariances(sys), blocks, rtol=1e-9, atol=1e-12)
    one = replace(_bordered_system((6,), velocity, gen), layout=None)
    one = replace(one, h_ss=one.h_ss[0], g_s=one.g_s[0])                   # single-clip form
    H1, g1 = densify(one)
    want1 = torch.linalg.solve(H1 + 0.05 * torch.diag(torch.diagonal(H1)) + 1e-9 * torch.eye(H1.shape[0],
                               dtype=torch.float64), -g1)
    d_pose1, d_l1 = solve_block_tridiagonal(one, 0.05)
    assert d_l1.shape == (K,) and torch.allclose(torch.cat([d_pose1.reshape(-1), d_l1]), want1, rtol=1e-9, atol=1e-11)


# ---------------------------------------------------------------------------------------------------------------- frame, outputs

def test_principal_frame_is_deterministic() -> None:
    gen = torch.Generator().manual_seed(9)
    R, _ = torch.linalg.qr(torch.randn(3, 3, generator=gen, dtype=torch.float64))
    pts = (torch.randn(500, 3, generator=gen, dtype=torch.float64) * torch.tensor([3.0, 2.0, 1.0])) @ R.T + 0.4
    U, c = principal_frame(pts)
    assert torch.allclose(U.T @ U, torch.eye(3, dtype=torch.float64), atol=1e-12) and abs(float(torch.det(U)) - 1) < 1e-12
    var = ((pts - c) @ U).var(0)
    assert bool((var[:-1] > var[1:]).all()), var                          # descending extents
    assert all(float(U[int(U[:, i].abs().argmax()), i]) > 0 for i in range(2))
    for other in (pts.flip(0), pts.float()):                               # sample order and dtype do not matter
        U2, c2 = principal_frame(other)
        assert torch.allclose(U2, U, atol=1e-5) and torch.allclose(c2, c, atol=1e-5)


def test_outputs_reproduce_the_solver_model_through_the_corrected_template() -> None:
    """Reported poses/scale with `corrected_template_points` give exactly the solver's p = R A (y - c) + t; flag-off
    solutions pass through unchanged."""
    inputs = _small_inputs()
    cfg = SolverConfig(**_SMALL, anisotropic_scale=True)
    db = prepare_batch(pack_clips(*_unzip(inputs)), cfg)
    gen = torch.Generator().manual_seed(4)
    ell = db.log_scale0 + 0.1 * torch.randn(2, 3, generator=gen, dtype=torch.float64)
    poses = apply_left(db.poses0, 0.05 * torch.randn(db.poses0.shape[0], 6, generator=gen, dtype=torch.float64))
    out_poses, log_scale, shape = api._isotropic_outputs(db, poses, ell)
    for b, sl in enumerate(db.layout.frame_slices()):
        stretch = Stretch(axes=db.shape_axes[sl], centroid=db.shape_centroid[sl],   # type: ignore[index]
                          log_stretch=ell[b].expand(sl.stop - sl.start, 3))
        y = db.Y[sl][:, :20]
        want = terms.transform_points(stretch.apply(y), poses[sl], 1.0)
        solution = api.TrajectorySolution(poses=out_poses[sl], scale=float(log_scale[b].exp()), covariance=poses[sl],
                                          per_frame_cost=poses[sl], confidence=poses[sl], iterations=0, seconds=0.0,
                                          shape_log_stretch=shape[b][0], shape_axes=shape[b][1],
                                          shape_centroid=shape[b][2])
        got = terms.transform_points(corrected_template_points(y, solution), out_poses[sl], solution.scale)
        assert torch.allclose(got, want, atol=1e-12)
        metric = corrected_template_points(y, solution, metric=True)
        assert torch.allclose(terms.transform_points(metric, out_poses[sl], 1.0), want, atol=1e-12)
        n = torch.nn.functional.normalize(torch.randn(20, 3, generator=gen, dtype=torch.float64), dim=-1)
        assert torch.allclose(corrected_template_normals(n, solution), stretch.take(slice(0, 1)).apply_normals(n)[0],
                              atol=1e-12)
    flag_off = replace(solution, shape_log_stretch=None, shape_axes=None, shape_centroid=None)
    assert corrected_template_points(y, flag_off) is y and corrected_template_normals(n, flag_off) is n
    assert {f.name for f in fields(api.TrajectorySolution)} >= {"shape_log_stretch", "shape_axes", "shape_centroid"}


def test_memory_estimate_bounds_the_anisotropic_peak(monkeypatch: pytest.MonkeyPatch) -> None:
    """The stretch transients are in `memory.batch_peak_bytes` (upper bound, within 2x of the CPU allocator peak), and
    they add nothing with the flag off."""
    inputs = [_inputs(_clip(T, 512, seed=211 + T, image_hw=(120, 160), n_tracks=16), seed=T) for T in (5, 6)]
    for moments in (True, False):
        cfg = SolverConfig(n_obs=128, n_mesh=512, iters_coarse=3, iters_mid=2, iters_fine=2, device="cpu",
                           anisotropic_scale=True, plan_moments=moments)
        refresh = 1 << 28
        measured = _measured_peak(monkeypatch, inputs, cfg, refresh)
        estimate = batch_peak_bytes(api.clip_dims(*_unzip(inputs)[:3], cfg), cfg, refresh)
        assert measured <= estimate <= 2.0 * measured, (moments, measured, estimate, estimate / measured)
        off = replace(cfg, anisotropic_scale=False)
        assert batch_peak_bytes(api.clip_dims(*_unzip(inputs)[:3], off), off, refresh) < estimate
