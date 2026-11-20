"""Dead-zone absolute depth anchor (`SolverConfig.lam_depth_anchor`, `terms.depth_anchor_term`).

(1) Off (the default, lam = 0): the term is never built -- every anchor entry point raises under monkeypatch and the
solve still reproduces the unpatched one bit for bit, B = 2. (2) On but in band: every frame of every trial stays
inside the dead zone, so the term adds exact zeros LAST and the solve is bit-identical to lam = 0 (the spy proves the
term ran and was zero). (3) Restoring force: a synthetic clip whose init places EVERY frame at 1.8x its depth along
the ray (the sustained-offset regime of sequence 135250) solved at the fine level only, where the unbalanced-OT
pointmap has abandoned the far template (an absorbing state): lam = 0 stays out at d ~0.54 with the tracks
co-scaling the object (scale 1.65x), lam = 100 brings every frame inside the band and the scale back. (4) The
Jacobian against central differences outside the band, both signs, with and without a stretch, and exact zeros
inside. (5) Config validation and the engine's `solve_stage.solver` path (Hydra needs `+`/`++`: struct mode).

The comparison of the flag-off path against the tree BEFORE this change was run once by hand (same fixtures under
both trees, `torch.equal` on every output); a stored digest would bind to one torch/BLAS build.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from torch import Tensor

from fastdo.engine.solve_stage_config import SolveStageConfig
from fastdo.geometry.lie import apply_left
from fastdo.solver import batch, batch_eval, terms
from fastdo.solver.aniso import Stretch
from fastdo.solver.api import MeshSamples, SolverConfig, TrajectoryObservations, solve_trajectories
from fastdo.solver.batch import frame_depths, valid_pixels
from fastdo.solver.terms import depth_anchor_term
from tests.solver.test_anisotropic_scale import _random_poses, _random_stretch
from tests.solver.test_batch import _clip, _inputs, _unzip

_REPO = Path(__file__).resolve().parents[2]
_SMALL = {"n_obs": 128, "n_mesh": 256, "device": "cpu"}
_SOLUTION_TENSORS = ("poses", "covariance", "per_frame_cost", "confidence")


def _small_inputs() -> list:
    return [_inputs(_clip(T, 256, seed=170 + T, image_hw=(120, 160), n_tracks=16), seed=T) for T in (5, 7)]


def _assert_same_solutions(a: list, b: list) -> None:
    for x, y in zip(a, b, strict=True):
        for name in _SOLUTION_TENSORS:
            assert torch.equal(getattr(x, name), getattr(y, name)), name
        assert x.scale == y.scale and x.iterations == y.iterations and x.accepted == y.accepted


def _observed_depths(clip) -> Tensor:  # type: ignore[no-untyped-def]
    pts, cnt = valid_pixels(clip.masks, clip.pointmaps)
    return frame_depths(pts, torch.cat([cnt.new_zeros(1), cnt.cumsum(0)]))


def _log_ratio(clip, poses: Tensor, scale: float, z_obs: Tensor) -> Tensor:  # type: ignore[no-untyped-def]
    """d_t = log z_sol - log z_obs at the template centroid (the term's definition, computed independently)."""
    c = clip.surface_points.double().mean(0)
    z = (scale * poses[:, :3, :3].double() @ c + poses[:, :3, 3].double())[:, 2]
    return z.log() - z_obs.log()


# ---------------------------------------------------------------------------------------------------------------- (1)

def test_off_is_the_default_and_never_builds_the_term(monkeypatch: pytest.MonkeyPatch) -> None:
    assert SolverConfig().lam_depth_anchor == 0.0 and SolverConfig().depth_anchor_band == 0.2
    inputs = _small_inputs()
    before = solve_trajectories(*_unzip(inputs), SolverConfig(**_SMALL))

    def boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("a depth-anchor op ran with lam_depth_anchor = 0")

    for module, name in ((batch_eval, "depth_anchor_term"), (terms, "depth_anchor_term"),
                         (batch, "frame_depths"), (batch, "_depth_anchor_fields")):
        monkeypatch.setattr(module, name, boom)
    omitted = solve_trajectories(*_unzip(inputs), SolverConfig(**_SMALL))
    explicit = solve_trajectories(*_unzip(inputs), SolverConfig(**_SMALL, lam_depth_anchor=0.0, depth_anchor_band=0.01))
    _assert_same_solutions(before, omitted)
    _assert_same_solutions(before, explicit)
    db = batch.prepare_batch(batch.pack_clips(*_unzip(inputs)), SolverConfig(**_SMALL))
    assert db.depth_centroid is None and db.depth_log_obs is None and db.depth_valid is None


# ---------------------------------------------------------------------------------------------------------------- (2)

def test_in_band_is_bit_identical_to_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """lam = 100 with every frame of every evaluation (trials included) inside the band: same bits as lam = 0."""
    inputs = _small_inputs()
    off = solve_trajectories(*_unzip(inputs), SolverConfig(**_SMALL))
    seen: list[Tensor] = []
    real = batch_eval.depth_anchor_term

    def spy(*args, **kwargs):  # type: ignore[no-untyped-def]
        out = real(*args, **kwargs)
        seen.append(out.residual.abs().max() + out.jacobian.abs().max())
        return out

    monkeypatch.setattr(batch_eval, "depth_anchor_term", spy)
    on = solve_trajectories(*_unzip(inputs), SolverConfig(**_SMALL, lam_depth_anchor=100.0))
    assert len(seen) > 10                                      # the term ran on every evaluation ...
    assert all(float(x) == 0.0 for x in seen)                  # ... and was exactly zero every time
    _assert_same_solutions(off, on)


# ---------------------------------------------------------------------------------------------------------------- (3)

def test_restores_a_sustained_depth_offset_the_pointmap_abandoned() -> None:
    clip = _clip(8, 256, seed=178, image_hw=(120, 160), n_tracks=16)
    obs = TrajectoryObservations(intrinsics=clip.intrinsics, masks=clip.masks, hand_masks=clip.hand_masks,
                                 pointmaps=clip.pointmaps, tracks=clip.tracks, track_vis=clip.track_vis,
                                 anchors=clip.anchors)
    mesh = MeshSamples(clip.surface_points, clip.surface_normals)
    z_obs = _observed_depths(clip)
    assert bool(torch.isfinite(z_obs).all())
    init = clip.poses_gt.clone()
    init[:, :3, 3] *= 1.8                                      # every frame 1.8x deeper along its ray
    assert float(_log_ratio(clip, init, clip.scale_gt, z_obs).min()) > 0.55
    # Fine level only: its unbalanced OT (rho_x = 0.1 diameter^2) abandons a template ~0.45 m away.
    base = SolverConfig(**_SMALL, iters_coarse=0, iters_mid=0)
    band = base.depth_anchor_band
    sol = {lam: solve_trajectories([mesh], [obs], [init], [clip.scale_gt], replace(base, lam_depth_anchor=lam))[0]
           for lam in (0.0, 100.0)}
    d_off = _log_ratio(clip, sol[0.0].poses, sol[0.0].scale, z_obs)
    d_on = _log_ratio(clip, sol[100.0].poses, sol[100.0].scale, z_obs)
    assert float(d_off.min()) > 0.4                            # absorbed: no restoring force without the anchor
    assert float(d_on.abs().max()) <= band                     # every frame back inside the band
    assert abs(sol[100.0].scale / clip.scale_gt - 1.0) < 0.1 < abs(sol[0.0].scale / clip.scale_gt - 1.0)
    err = {lam: float((s.poses[:, :3, 3] - clip.poses_gt[:, :3, 3]).norm(dim=1).mean()) for lam, s in sol.items()}
    assert err[100.0] < 0.1 * err[0.0]

    # Batched path: the offset clip between two in-band clips solves as it does alone (clip independence).
    m, o, p, s = _unzip(_small_inputs())
    cfg = replace(base, lam_depth_anchor=100.0)
    batched = solve_trajectories([m[0], mesh, m[1]], [o[0], obs, o[1]], [p[0], init, p[1]], [s[0], clip.scale_gt, s[1]],
                                 cfg)
    assert torch.allclose(batched[1].poses, sol[100.0].poses, atol=1e-6)
    assert batched[1].scale == pytest.approx(sol[100.0].scale, rel=1e-9)
    assert batched[1].accepted == sol[100.0].accepted


# ---------------------------------------------------------------------------------------------------------------- (4)

def _fd(residual, poses: Tensor, log_scale: Tensor, h: float = 1e-6) -> Tensor:  # type: ignore[no-untyped-def]
    """Central differences of `residual(poses, log_scale)` (T,1,1) over [rho, theta, sigma]: (T,1,1,7)."""
    cols = []
    for i in range(7):
        e = torch.zeros(poses.shape[0], 7, dtype=torch.float64)
        e[:, i] = h
        plus = residual(apply_left(poses, e[:, :6]), log_scale + e[:, 6])
        minus = residual(apply_left(poses, -e[:, :6]), log_scale - e[:, 6])
        cols.append((plus - minus) / (2 * h))
    return torch.stack(cols, dim=-1)


def test_jacobian_matches_finite_differences_outside_the_band() -> None:
    gen = torch.Generator().manual_seed(225)
    T, lam, band = 6, 7.0, 0.2
    poses = _random_poses(T, gen)
    poses[:, 2, 3] += 0.6                                      # keep every centroid well in front
    centroid = 0.3 * torch.randn(T, 3, generator=gen, dtype=torch.float64)
    log_scale = math.log(0.15) + 0.1 * torch.randn(T, generator=gen, dtype=torch.float64)
    z_sol = (log_scale.exp()[:, None] * (poses[:, :3, :3] @ centroid[..., None])[..., 0] + poses[:, :3, 3])[:, 2]
    assert bool((z_sol > 0.05).all())
    offset = torch.tensor([0.5, -0.5, 0.3, -0.35, 0.8, -0.9], dtype=torch.float64)   # |d| > band, both signs
    log_obs = z_sol.log() - offset
    valid = torch.ones(T, dtype=torch.bool)

    def residual(p: Tensor, ls: Tensor) -> Tensor:
        return depth_anchor_term(centroid, log_obs, valid, p, ls, lam, band).residual

    out = depth_anchor_term(centroid, log_obs, valid, poses, log_scale, lam, band)
    assert out.residual.shape == (T, 1, 1) and out.jacobian.shape == (T, 1, 1, 7) and out.weight.shape == (T, 1)
    expected = lam ** 0.5 * torch.sign(offset) * (offset.abs() - band)
    assert torch.allclose(out.residual[:, 0, 0], expected, atol=1e-12)
    fd = _fd(residual, poses, log_scale)
    # pose columns match finite differences; the size columns (sigma, stretch) are zeroed by design (placement only)
    assert torch.allclose(out.jacobian[..., :6], fd[..., :6], rtol=1e-6, atol=1e-8)
    assert torch.equal(out.jacobian[..., 6:], torch.zeros_like(out.jacobian[..., 6:]))
    assert fd[..., 6].abs().max() > 0          # the residual DOES depend on scale: the zero column is a choice

    # anisotropic stretch: the centroid is a template point under R A (c - c_p) + t; nine columns.
    st = _random_stretch(T, gen)
    zero = torch.zeros(T, dtype=torch.float64)
    z_st = terms.transform_points(st.apply(centroid[:, None, :]), poses, 1.0)[:, 0, 2]
    assert bool((z_st > 0.05).all())
    log_obs_st = z_st.log() - offset

    def residual_st(p: Tensor, s: Stretch) -> Tensor:
        return depth_anchor_term(centroid, log_obs_st, valid, p, zero, lam, band, stretch=s).residual

    out_st = depth_anchor_term(centroid, log_obs_st, valid, poses, zero, lam, band, stretch=st)
    assert out_st.jacobian.shape == (T, 1, 1, 9)
    cols = []
    for i in range(9):
        e = torch.zeros(T, 9, dtype=torch.float64)
        e[:, i] = 1e-6
        plus = residual_st(apply_left(poses, e[:, :6]), replace(st, log_stretch=st.log_stretch + e[:, 6:]))
        minus = residual_st(apply_left(poses, -e[:, :6]), replace(st, log_stretch=st.log_stretch - e[:, 6:]))
        cols.append((plus - minus) / 2e-6)
    fd_st = torch.stack(cols, dim=-1)
    assert torch.allclose(out_st.jacobian[..., :6], fd_st[..., :6], rtol=1e-6, atol=1e-8)
    assert torch.equal(out_st.jacobian[..., 6:], torch.zeros_like(out_st.jacobian[..., 6:]))   # stretch: size, zeroed


def test_exact_zeros_inside_the_band_and_on_unobserved_frames() -> None:
    gen = torch.Generator().manual_seed(2251)
    T = 5
    poses = _random_poses(T, gen)
    poses[:, 2, 3] += 0.6                                      # keep every centroid well in front
    centroid = 0.3 * torch.randn(T, 3, generator=gen, dtype=torch.float64)
    log_scale = torch.full((T,), math.log(0.15), dtype=torch.float64)
    z_sol = (0.15 * (poses[:, :3, :3] @ centroid[..., None])[..., 0] + poses[:, :3, 3])[:, 2]
    offset = torch.tensor([0.0, 0.19, -0.19, 0.9, -0.9], dtype=torch.float64)
    valid = torch.tensor([True, True, True, False, False])
    out = depth_anchor_term(centroid, z_sol.log() - offset, valid, poses, log_scale, 1e6, 0.2)
    assert torch.equal(out.residual, torch.zeros_like(out.residual))
    assert torch.equal(out.jacobian, torch.zeros_like(out.jacobian))
    assert out.weight[:, 0].tolist() == [1.0, 1.0, 1.0, 0.0, 0.0]


def test_frame_depths_is_the_median_of_the_in_front_points() -> None:
    rows = [[0.0, 0.0, z] for z in (0.5, 0.7, 0.6, 0.9, -1.0, float("nan"))] * 2 + [[float("nan"), 0.0, 0.8]]
    pts = torch.tensor(rows, dtype=torch.float32)             # 8 finite in-front points; the last drops on its x
    offsets = torch.tensor([0, 13, 13])
    z = frame_depths(pts, offsets, min_points=8)
    assert z[0].item() == pytest.approx(0.65) and math.isnan(z[1].item())      # even count: mean of 0.6 and 0.7
    assert math.isnan(frame_depths(pts, offsets, min_points=9)[0].item())      # 8 in-front points < 9


# ---------------------------------------------------------------------------------------------------------------- (5)

@pytest.mark.parametrize("bad", [{"lam_depth_anchor": -1.0}, {"lam_depth_anchor": float("nan")},
                                 {"lam_depth_anchor": float("inf")}, {"depth_anchor_band": 0.0},
                                 {"depth_anchor_band": -0.1}, {"depth_anchor_band": float("nan")},
                                 {"depth_anchor_band": float("inf")}])
def test_config_validation(bad: dict) -> None:
    with pytest.raises(ValueError, match="lam_depth_anchor|depth_anchor_band"):
        SolverConfig(**bad)


def test_engine_config_path_reaches_solver_config() -> None:
    cfg = SolveStageConfig.from_mapping({"device": "cpu", "solver": {"lam_depth_anchor": 50.0, "depth_anchor_band": 0.3}})
    solver = cfg.solver_config()
    assert (solver.lam_depth_anchor, solver.depth_anchor_band) == (50.0, 0.3)
    with pytest.raises(ValueError, match="depth_anchor_band"):
        SolveStageConfig.from_mapping({"solver": {"depth_anchor_band": 0.0}}).solver_config()

    def composed(*overrides: str) -> SolverConfig:
        with initialize_config_dir(config_dir=str(_REPO / "run/conf"), version_base=None):
            # the always-on term and the certificate gate are exclusive; production has the gate on
            raw = compose(config_name="config", overrides=[f"paths.root={_REPO}", "solve_stage.depth_anchor_gate=null",
                                                           *overrides])
        return SolveStageConfig.from_mapping(raw.solve_stage).solver_config()

    assert composed().lam_depth_anchor == 0.0                  # run/conf/config.yaml keeps the term OFF
    for prefix in ("+", "++"):                                 # `solver: {}` is a struct node: a new key needs + / ++
        on = composed(f"{prefix}solve_stage.solver.lam_depth_anchor=50", f"{prefix}solve_stage.solver.depth_anchor_band=0.25")
        assert (on.lam_depth_anchor, on.depth_anchor_band) == (50.0, 0.25)
    with pytest.raises(Exception, match="lam_depth_anchor"):
        composed("solve_stage.solver.lam_depth_anchor=50")
