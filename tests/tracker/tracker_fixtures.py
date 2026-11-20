"""Shared fixtures for the tracker model tests: small config, seeded parameters and inputs, fixture-built inputs."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import torch
from torch import Tensor, nn

from fastdo.solver.banded import ClipLayout
from fastdo.solver.ot_sparse import SparsePlan
from fastdo.solver.terms import TermOutput, pointmap_term, track_term
from fastdo.tracker.clusters import mesh_cluster_ids
from fastdo.tracker.config import TrackerConfig
from fastdo.tracker.features import build_frame_tokens, track_tokens
from fastdo.tracker.heads import TrackerOutput
from fastdo.tracker.normalize import cluster_geometry
from fastdo.tracker.pack import TrackerInputs, build_tracker_inputs
from tests.tracker.synthetic_terms import CAUCHY_C, TermsFixture, make_terms_fixture

__all__ = ["SMALL_CONFIG", "Scene", "base_scene", "fixture_tracker_inputs", "max_rel", "nan_injection",
           "pixel_rescaled_scene", "random_tracker_inputs", "randomize_parameters", "scaled_scene", "tracker_loss",
           "two_clip_args"]

F64 = torch.float64
SMALL_CONFIG = TrackerConfig(width=48, n_layers=2, n_heads=4, head_hidden=16, max_distance=8)


def randomize_parameters(model: nn.Module, std: float = 0.05, seed: int = 0) -> None:
    """Every parameter ~ N(0, std^2), LayerNorm scales ~ N(1, std^2), drawn on CPU float64 in named order."""
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            base = 1.0 if (p.dim() == 1 and name.endswith("weight")) else 0.0
            p.copy_((torch.randn(p.shape, generator=gen, dtype=F64) * std + base).to(device=p.device, dtype=p.dtype))


def _some_true(shape: tuple[int, ...], gen: torch.Generator) -> Tensor:   # ~75% True, >= 1 True per (B,W)
    keep = torch.randint(shape[-1], shape[:-1] + (1,), generator=gen)
    return (torch.rand(shape, generator=gen) < 0.75).scatter(-1, keep, True)


def random_tracker_inputs(B: int, lengths: tuple[int, ...], J: int, N: int, C: int = 64, seed: int = 0,
                          width: int | None = None, dtype: torch.dtype = F64) -> TrackerInputs:
    """Seeded Gaussian tokens, `frame_mask` True on each clip's first `lengths[b]` frames, random set masks."""
    if len(lengths) != B:
        raise ValueError(f"need {B} lengths, got {lengths}")
    W = max(lengths) if width is None else width
    gen = torch.Generator().manual_seed(seed)
    frame = torch.randn(B, W, 110, generator=gen, dtype=F64)
    track = torch.randn(B, W, J, 6, generator=gen, dtype=F64)
    visible = _some_true((B, W, J), gen)
    track[..., 5] = visible.to(F64)
    track_mask = _some_true((B, W, J), gen) | visible
    point = torch.randn(B, W, N, 7, generator=gen, dtype=F64)
    cluster = torch.randn(B, W, C, 5, generator=gen, dtype=F64)
    frame_mask = torch.arange(W)[None, :] < torch.tensor(lengths)[:, None]
    return TrackerInputs(frame.to(dtype), frame_mask, track.to(dtype), track_mask, point.to(dtype),
                         _some_true((B, W, N), gen), cluster.to(dtype), _some_true((B, W, C), gen))


def tracker_loss(out: TrackerOutput) -> Tensor:
    """The loss of test M10: sum of weights, duals, increments, log damping and log precision."""
    return (out.point_weights.sum() + out.track_weights.sum() + out.dual_f.sum() + out.dual_g.sum()
            + out.xi_hat.sum() + out.sigma_hat.sum() + out.damping.log().sum() + out.precision_diag.log().sum())


def max_rel(a: Tensor, ref: Tensor) -> float:
    """max|a - ref| / max|ref| (0.0 when equal, inf when ref is all zero and a is not)."""
    diff = float((a.to(F64) - ref.to(F64)).abs().max()) if a.numel() else 0.0
    scale = float(ref.to(F64).abs().max()) if ref.numel() else 0.0
    return 0.0 if diff == 0.0 else (diff / scale if scale > 0 else math.inf)


@dataclass(frozen=True)
class Scene:
    """A fixture's per-frame state and the terms evaluated at it (plan held fixed)."""
    X: Tensor
    poses: Tensor
    log_scale: Tensor
    K: Tensor
    pm: TermOutput
    tr_raw: TermOutput


def base_scene(fx: TermsFixture) -> Scene:
    return Scene(fx.X, fx.poses, fx.log_scale, fx.K, fx.pm, fx.tr_raw)


def scaled_scene(fx: TermsFixture, a: float) -> Scene:
    """Scene similarity as feature-token test 11b: X, pose translations and the scale by `a`, K fixed."""
    poses = fx.poses.clone()
    poses[:, :3, 3] *= a
    ls = fx.log_scale + math.log(a)
    pm = pointmap_term(a * fx.X, fx.X_valid, fx.Y, poses, ls, a**2 * fx.eps, a**2 * fx.rho_x, a**2 * fx.rho_y, 0,
                       plan=fx.plan)[0]
    tr = track_term(fx.anchors, fx.tracks, fx.track_vis, poses, ls, fx.K, CAUCHY_C)
    return Scene(a * fx.X, poses, ls, fx.K, pm, tr)


def pixel_rescaled_scene(fx: TermsFixture, b: float) -> Scene:
    """Pixel rescale: K rows 0-1, the tracks and cauchy_c by `b`."""
    K = fx.K.clone()
    K[:2] *= b
    tr = track_term(fx.anchors, b * fx.tracks, fx.track_vis, fx.poses, fx.log_scale, K, b * CAUCHY_C)
    return Scene(fx.X, fx.poses, fx.log_scale, K, fx.pm, tr)


def fixture_tracker_inputs(fx: TermsFixture, scene: Scene | None = None, plan: Tensor | SparsePlan | None = None,
                           cluster_id: Tensor | None = None, point_pm: TermOutput | None = None) -> TrackerInputs:
    """Fixture -> feature tokens -> `build_tracker_inputs` (one clip); `point_pm` feeds only the point tokens."""
    s = base_scene(fx) if scene is None else scene
    plan = fx.plan if plan is None else plan
    ids = mesh_cluster_ids(fx.Y, 64, 0) if cluster_id is None else cluster_id
    frame = build_frame_tokens(s.pm, s.tr_raw, plan, ids, fx.template_diameter, s.poses, s.log_scale)
    centroids, fraction = cluster_geometry(fx.Y, ids, fx.template_diameter)
    return build_tracker_inputs(frame, track_tokens(fx.anchors, s.tr_raw, fx.template_diameter),
                                s.pm if point_pm is None else point_pm, s.X, plan, s.poses, s.log_scale,
                                fx.template_diameter, s.K, centroids, fraction)


def nan_injection(fs: TermsFixture) -> tuple[TermOutput, TermOutput, SparsePlan]:
    """Token test 14's pattern rebuilt through `pointmap_term`: (clean pm, NaN pm, NaN plan), NaN in frame 4 only."""
    assert isinstance(fs.plan, SparsePlan)
    values = fs.plan.values.clone()
    values[4, 0, 0] = float("nan")
    nan_plan = SparsePlan(indices=fs.plan.indices, values=values)
    pm_c, pm_n = (pointmap_term(fs.X, fs.X_valid, fs.Y, fs.poses, fs.log_scale, fs.eps, fs.rho_x, fs.rho_y, 0,
                                plan=plan, knn=16)[0] for plan in (fs.plan, nan_plan))
    jac = pm_n.jacobian.clone()
    jac[4, 0, 0, 0] = float("nan")
    return pm_c, replace(pm_n, jacobian=jac), nan_plan


def two_clip_args(frames0: int = 6) -> dict[str, Any]:
    """P2's batch as `build_tracker_inputs` kwargs: sparse seeds 0 and 1, seed 0 cut to `frames0` frames."""
    parts: dict[str, list[Tensor]] = {}
    geometry: list[tuple[Tensor, Tensor]] = []
    for seed, n in ((0, frames0), (1, 6)):
        fx = make_terms_fixture(knn=16, seed=seed)
        assert isinstance(fx.plan, SparsePlan)
        ids = mesh_cluster_ids(fx.Y, 64, 0)
        geometry.append(cluster_geometry(fx.Y, ids, fx.template_diameter))
        per_frame = {"frame_tokens": build_frame_tokens(fx.pm, fx.tr_raw, fx.plan, ids, fx.template_diameter,
                                                        fx.poses, fx.log_scale),
                     "track_tokens": track_tokens(fx.anchors, fx.tr_raw, fx.template_diameter),
                     "res": fx.pm.residual, "jac": fx.pm.jacobian, "w": fx.pm.weight, "X": fx.X,
                     "idx": fx.plan.indices, "val": fx.plan.values, "poses": fx.poses,
                     "log_scale": fx.log_scale.expand(6), "K": fx.K.expand(6, 3, 3),
                     "template_diameter": torch.full((6,), fx.template_diameter, dtype=F64)}
        for key, value in per_frame.items():
            parts.setdefault(key, []).append(value[:n])
    cat = {key: torch.cat(value) for key, value in parts.items()}
    layout = ClipLayout.build((frames0, 6), "cpu")
    pm, plan = TermOutput(cat.pop("res"), cat.pop("jac"), cat.pop("w")), SparsePlan(cat.pop("idx"), cat.pop("val"))
    cen, frac = (torch.stack([g[i] for g in geometry])[layout.clip_index] for i in (0, 1))
    return dict(cat, pm=pm, plan=plan, centroids=cen, fraction=frac, layout=layout)
