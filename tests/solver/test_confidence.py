"""The support-invariant, noise-floor-normalized per-frame confidence (`fastdo.solver.confidence`).

The function reads the fine-level pointmap term as the solver produces it: `residual = sqrt(N w_i) (x_i - pbar_i)`
with `N` the padded row count, `w_i` the plan's row mass and invalid rows zeroed (`terms.pointmap_term`).
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import pytest
import torch
from torch import Tensor

from fastdo.solver import api
from fastdo.solver.api import (
    MeshSamples,
    SolverConfig,
    TrajectoryObservations,
    score_trajectory,
    solve_trajectories,
    solve_trajectory,
)
from fastdo.solver.confidence import MAD_TO_SIGMA, FrameConfidence, frame_confidence
from fastdo.solver.synthetic import SyntheticConfig, make_synthetic_clip

_MIN_SUPPORT = 20


def _residual(dist: Tensor, mass: Tensor, gen: torch.Generator) -> Tensor:
    """`sqrt(N w) (x - pbar)` for matched distances `dist` (T,N) along random unit directions."""
    direction = torch.nn.functional.normalize(torch.randn(*dist.shape, 3, generator=gen, dtype=torch.float64), dim=-1)
    return (dist.shape[1] * mass).sqrt()[..., None] * dist[..., None] * direction


def _frames(T: int, N: int, sigma: float, gen: torch.Generator, n_valid: int | None = None
            ) -> tuple[Tensor, Tensor, Tensor]:
    """T frames of N rows (the first `n_valid` valid) whose distance is |N(0, sigma)| (depth-like 1D noise), with the
    uniform row mass 1/n_valid of a balanced plan; returns (residual, row_mass, valid)."""
    n = N if n_valid is None else n_valid
    valid = torch.zeros(T, N, dtype=torch.bool)
    valid[:, :n] = True
    dist = (sigma * torch.randn(T, N, generator=gen, dtype=torch.float64)).abs()
    mass = torch.where(valid, torch.full((T, N), 1.0 / n, dtype=torch.float64), torch.zeros(T, N, dtype=torch.float64))
    res = torch.where(valid[..., None], _residual(dist, mass, gen), torch.zeros(T, N, 3, dtype=torch.float64))
    return res, mass, valid


def _score(res: Tensor, mass: Tensor, valid: Tensor, slices: list[slice] | None = None, scale: float = 1.0,
           mass_floor: float = 0.1) -> FrameConfidence:
    return frame_confidence(res, mass, valid, slices or [slice(0, res.shape[0])], scale=scale, mass_floor=mass_floor,
                            min_support=_MIN_SUPPORT)


def test_frame_at_the_noise_floor_scores_e_minus_one() -> None:
    """Depth-like noise of 2 mm: the noise floor 1.4826 x median distance recovers sigma and a frame at that floor
    has statistic ~1, so confidence ~e^-1 at scale 1 (and exp(-1/scale) at any other scale)."""
    gen = torch.Generator().manual_seed(0)
    res, mass, valid = _frames(4, 4096, 0.002, gen)
    out = _score(res, mass, valid)
    assert MAD_TO_SIGMA == pytest.approx(1.4826)
    assert out.noise_floor.shape == (1,) and float(out.noise_floor[0]) == pytest.approx(0.002, rel=0.03)
    assert torch.allclose(out.statistic, torch.ones(4, dtype=torch.float64), atol=0.04), out.statistic
    assert torch.allclose(out.confidence, torch.full((4,), math.exp(-1.0), dtype=torch.float64), atol=0.02)
    scaled = _score(res, mass, valid, scale=2.5)
    assert torch.equal(scaled.statistic, out.statistic)
    assert torch.allclose(scaled.confidence, torch.exp(-out.statistic / 2.5), rtol=1e-12, atol=0.0)


def test_statistic_is_support_invariant() -> None:
    """A frame whose 100 valid rows are padded to 1024 scores exactly like a frame holding the same distances tiled 10
    times (and like the same frame with every row mass halved): the old sqrt(N w)-weighted RMS grew as
    sqrt(N / N_valid) instead."""
    gen = torch.Generator().manual_seed(1)
    base, base_mass, base_valid = _frames(3, 1024, 0.002, gen)                # frames 0-2 fix the clip's floor
    dist = (0.003 * torch.randn(100, generator=gen, dtype=torch.float64)).abs()
    sparse_dist = torch.zeros(1024, dtype=torch.float64)
    sparse_dist[:100] = dist
    tiled_dist = torch.zeros(1024, dtype=torch.float64)
    tiled_dist[:1000] = dist.repeat(10)
    sparse_valid = torch.arange(1024) < 100
    tiled_valid = torch.arange(1024) < 1000
    sparse_mass = sparse_valid.double() / 100
    tiled_mass = tiled_valid.double() / 1000
    rows = [(sparse_dist, sparse_mass, sparse_valid), (tiled_dist, tiled_mass, tiled_valid),
            (sparse_dist, 0.5 * sparse_mass, sparse_valid)]
    res = torch.cat([base] + [_residual(d[None], m[None], gen) for d, m, _ in rows])
    mass = torch.cat([base_mass] + [m[None] for _, m, _ in rows])
    valid = torch.cat([base_valid] + [v[None] for _, _, v in rows])
    out = _score(res, mass, valid)
    stat = out.statistic
    assert float(stat[3]) == pytest.approx(float(stat[4]), rel=1e-12) and float(stat[3]) == pytest.approx(
        float(stat[5]), rel=1e-12), stat
    old_rms = (res.norm(dim=-1).pow(2).sum(1) / valid.sum(1)).sqrt()      # the earlier per-frame statistic
    assert float(old_rms[3] / old_rms[4]) > 3.0


def test_frames_without_pointmap_support_score_exactly_zero() -> None:
    gen = torch.Generator().manual_seed(2)
    res, mass, valid = _frames(5, 256, 0.002, gen)
    valid[1, _MIN_SUPPORT - 1:] = False                                        # 19 valid rows: below the support gate
    valid[3] = False                                                           # fully occluded
    res = torch.where(valid[..., None], res, torch.zeros_like(res))
    out = _score(res, mass, valid)
    assert float(out.confidence[1]) == 0.0 and float(out.confidence[3]) == 0.0
    assert bool((out.confidence[[0, 2, 4]] > 0.0).all()) and bool(torch.isfinite(out.confidence).all())
    none = _score(torch.zeros_like(res), mass, torch.zeros_like(valid))
    assert bool((none.confidence == 0.0).all()) and bool(torch.isfinite(none.confidence).all())


def test_rows_below_the_mass_floor_are_excluded() -> None:
    """Rows whose plan mass is under `mass_floor` x the frame's mean valid-row mass (observations the partial OT left
    unmatched) do not enter the frame statistic: the frame scores as if they were absent."""
    gen = torch.Generator().manual_seed(3)
    res, mass, valid = _frames(4, 512, 0.002, gen)
    mass_out = mass.clone()
    mass_out[2, 400:] = 1e-4 * mass[2, 0]                                     # 112 unmatched rows
    res_out = res.clone()
    res_out[2, 400:] = _residual(torch.full((1, 112), 0.05, dtype=torch.float64), mass_out[2:3, 400:], gen)[0]
    clean = _score(res, mass, valid)
    dirty = _score(res_out, mass_out, valid)
    absent_valid = valid.clone()
    absent_valid[2, 400:] = False
    absent = _score(torch.where(absent_valid[..., None], res, 0.0), mass.masked_fill(~absent_valid, 0.0), absent_valid)
    assert float(dirty.statistic[2]) == pytest.approx(float(absent.statistic[2]), rel=0.03)
    assert float(dirty.confidence[2]) == pytest.approx(float(clean.confidence[2]), abs=0.05)


def test_wrong_frame_scores_low_and_leaves_the_other_frames_alone() -> None:
    """One frame of 30 at 20 mm against a 2 mm floor: it scores near 0; the clip median moves by ~4%, so the other
    frames' statistics stay within 10%."""
    gen = torch.Generator().manual_seed(4)
    res, mass, valid = _frames(30, 512, 0.002, gen)
    good = _score(res, mass, valid, scale=2.0)
    bad_res = res.clone()
    bad_res[6] = _residual(torch.full((1, 512), 0.02, dtype=torch.float64), mass[6:7], gen)[0]
    bad = _score(bad_res, mass, valid, scale=2.0)
    assert float(bad.confidence[6]) < 0.05
    keep = torch.arange(30) != 6
    assert torch.allclose(bad.statistic[keep], good.statistic[keep], rtol=0.1)


def test_batched_clips_score_bitwise_like_one_clip() -> None:
    """Three clips of different lengths, noise levels and supports (N = 37, so clip offsets are not SIMD-aligned):
    scoring them as one batch is bitwise equal to scoring each clip alone, and no clip reads another's noise floor."""
    gen = torch.Generator().manual_seed(5)
    clips = [_frames(5, 37, 0.002, gen), _frames(7, 37, 0.01, gen, n_valid=30), _frames(3, 37, 0.004, gen)]
    res, mass, valid = (torch.cat([c[i] for c in clips]) for i in range(3))
    valid[6, :] = False
    slices = [slice(0, 5), slice(5, 12), slice(12, 15)]
    batched = _score(res, mass, valid, slices, scale=1.7)
    for b, sl in enumerate(slices):
        alone = _score(res[sl].clone(), mass[sl].clone(), valid[sl].clone(), scale=1.7)
        assert torch.equal(batched.confidence[sl], alone.confidence), b
        assert torch.equal(batched.statistic[sl], alone.statistic), b
        assert torch.equal(batched.noise_floor[b:b + 1], alone.noise_floor), b
    assert float(batched.noise_floor[1]) > 2.0 * float(batched.noise_floor[0])


def _tiny_inputs(seed: int, n_frames: int, occlusion: float) -> tuple[MeshSamples, TrajectoryObservations, Tensor,
                                                                      float]:
    clip = make_synthetic_clip(SyntheticConfig(n_frames=n_frames, image_hw=(120, 160), focal=150.0, n_surface=256,
                                               n_tracks=16, occlusion=occlusion, shape="box"), seed=seed)
    obs = TrajectoryObservations(intrinsics=clip.intrinsics, masks=clip.masks, hand_masks=clip.hand_masks,
                                 pointmaps=clip.pointmaps, tracks=clip.tracks, track_vis=clip.track_vis,
                                 anchors=clip.anchors)
    return MeshSamples(clip.surface_points, clip.surface_normals), obs, clip.poses_gt, clip.scale_gt


def test_solver_scores_each_clip_of_a_batch_like_its_own_one_clip_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """`solve_trajectories` hands `frame_confidence` its fine-level refresh with one slice per clip, and each returned
    `TrajectorySolution.confidence` is bitwise the one-clip computation on that clip's own rows (float32 on the host),
    using `SolverConfig.confidence_scale` / `confidence_mass_floor` and the pointmap support gate."""
    calls: list[tuple[Tensor, Tensor, Tensor, list[slice], dict[str, Any]]] = []

    def spy(residual: Tensor, row_mass: Tensor, valid: Tensor, slices: list[slice], **kw: Any) -> FrameConfidence:
        calls.append((residual.clone(), row_mass.clone(), valid.clone(), list(slices), dict(kw)))
        return frame_confidence(residual, row_mass, valid, slices, **kw)

    monkeypatch.setattr(api, "frame_confidence", spy)
    inputs = [_tiny_inputs(7, 5, 0.0), _tiny_inputs(8, 7, 0.5)]
    cfg = replace(SolverConfig(n_obs=64, n_mesh=128, device="cpu", iters_coarse=2, iters_mid=1, iters_fine=2),
                  confidence_scale=1.9, confidence_mass_floor=0.2)
    sols = solve_trajectories([i[0] for i in inputs], [i[1] for i in inputs], [i[2] for i in inputs],
                              [i[3] for i in inputs], cfg)
    assert len(calls) == 1
    res, mass, valid, slices, kw = calls[0]
    assert slices == [slice(0, 5), slice(5, 12)]
    assert kw == {"scale": 1.9, "mass_floor": 0.2, "min_support": api._MIN_VALID_OBS}
    for sol, sl in zip(sols, slices, strict=True):
        alone = frame_confidence(res[sl].clone(), mass[sl].clone(), valid[sl].clone(), [slice(0, sl.stop - sl.start)],
                                 **kw)
        assert torch.equal(sol.confidence, alone.confidence.float().cpu())


def test_solve_trajectories_matches_one_clip_solves_bitwise() -> None:
    """End to end (batched solve, shipped confidence): a two-clip `solve_trajectories` returns, for each clip,
    exactly the poses, per-frame cost and confidence of that clip's own `solve_trajectory` (CPU, shipped confidence
    defaults; clip 1 has fully occluded frames that score 0.0)."""
    inputs = [_tiny_inputs(7, 5, 0.0), _tiny_inputs(8, 7, 0.5)]
    cfg = SolverConfig(n_obs=64, n_mesh=128, device="cpu", iters_coarse=2, iters_mid=1, iters_fine=2)
    batched = solve_trajectories([i[0] for i in inputs], [i[1] for i in inputs], [i[2] for i in inputs],
                                 [i[3] for i in inputs], cfg)
    for b, (mesh, obs, poses, scale) in enumerate(inputs):
        alone = solve_trajectory(mesh, obs, poses, scale, cfg)
        assert torch.equal(batched[b].poses, alone.poses), b
        assert torch.equal(batched[b].per_frame_cost, alone.per_frame_cost), b
        assert torch.equal(batched[b].confidence, alone.confidence), (b, batched[b].confidence, alone.confidence)
    assert bool((batched[1].confidence == 0.0).any()) and bool((batched[0].confidence > 0.0).all())


def test_statistic_is_invariant_to_the_visible_template_sample_count() -> None:
    """The statistic of frames at their ground-truth poses (`score_trajectory`, shipped scale and mass floor) stays at
    the noise floor as the template is sampled 8x more densely: its clip median moves by less than 15% over n_mesh
    256 -> 2048 (measured 0.82 -> 0.92 on this box), where the earlier residual scale moved with
    the plan's row masses. The floor reads ~1, so a frame there scores ~exp(-1 / confidence_scale)."""
    clip = make_synthetic_clip(SyntheticConfig(n_frames=12, image_hw=(240, 320), n_surface=2048, n_tracks=48,
                                               occlusion=0.0, depth_noise=0.002, shape="box"), seed=11)
    obs = TrajectoryObservations(intrinsics=clip.intrinsics, masks=clip.masks, hand_masks=clip.hand_masks,
                                 pointmaps=clip.pointmaps, tracks=clip.tracks, track_vis=clip.track_vis,
                                 anchors=clip.anchors)
    mesh = MeshSamples(clip.surface_points, clip.surface_normals)
    medians: dict[int, float] = {}
    for n_mesh in (256, 1024, 2048):
        cfg = SolverConfig(n_obs=512, n_mesh=n_mesh, device="cpu")
        _, conf = score_trajectory(mesh, obs, clip.poses_gt, clip.scale_gt, cfg)
        assert bool((conf > 0.0).all()), (n_mesh, conf)
        medians[n_mesh] = float((-cfg.confidence_scale * conf.double().log()).median())
    assert max(medians.values()) <= 1.15 * min(medians.values()), medians
    assert all(0.7 <= z <= 1.1 for z in medians.values()), medians
