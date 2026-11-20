"""Synthetic solver terms for the tracker feature tests: one fixture built from the solver's own terms."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from fastdo.solver.batch import sample_points, valid_pixels
from fastdo.solver.ot_sparse import SparsePlan
from fastdo.solver.synthetic import SyntheticConfig, make_synthetic_clip
from fastdo.solver.terms import TermOutput, pointmap_term, track_term

__all__ = ["TermsFixture", "make_terms_fixture", "random_terms"]

CAUCHY_C = 4.0   # SolverConfig.cauchy_c default (px)


@dataclass(frozen=True)
class TermsFixture:
    """A synthetic box clip's pointmap term and plan, and its raw and solver-weighted track terms, at GT poses.

    `tr` (weight / L2, i.e. lam_track = 1 and scale2 = L2) exists only for the solver-weighting checks; it is never
    a `build_frame_tokens` input.
    """

    X: Tensor
    X_valid: Tensor
    Y: Tensor
    poses: Tensor
    log_scale: Tensor
    K: Tensor
    anchors: Tensor
    tracks: Tensor
    track_vis: Tensor
    template_diameter: float
    eps: float
    rho_x: float
    rho_y: float
    pm: TermOutput
    plan: Tensor | SparsePlan
    tr_raw: TermOutput
    tr: TermOutput


def make_terms_fixture(T: int = 6, N: int = 96, M: int = 256, J: int = 16, knn: int | None = None,
                       seed: int = 0, occluded_frame: int | None = None) -> TermsFixture:
    """Build the fixture exactly as the feature-token tests specify (float64, CPU)."""
    clip = make_synthetic_clip(SyntheticConfig(n_frames=T, image_hw=(120, 160), focal=150.0, shape="box",
                                               n_surface=M, n_tracks=J), seed)
    points, counts = valid_pixels(clip.masks, clip.pointmaps)
    X, X_valid = sample_points(points, torch.cat([counts.new_zeros(1), counts.cumsum(0)]), N,
                               torch.Generator().manual_seed(seed))
    Y = clip.surface_points.double()
    template_diameter = float((Y.max(0).values - Y.min(0).values).norm())
    poses = clip.poses_gt.double()
    log_scale = torch.tensor(math.log(clip.scale_gt), dtype=torch.float64)
    L2 = (clip.scale_gt * template_diameter) ** 2
    eps, rho_x, rho_y = 1e-3 * L2, 0.1 * L2, 1e-3 * L2
    track_vis = clip.track_vis.clone()
    if occluded_frame is not None:
        X_valid[occluded_frame] = False
        track_vis[occluded_frame] = False
    pm, plan = pointmap_term(X, X_valid, Y, poses, log_scale, eps, rho_x, rho_y, 50, knn=knn)
    anchors, tracks, K = clip.anchors.double(), clip.tracks.double(), clip.intrinsics.double()
    tr_raw = track_term(anchors, tracks, track_vis, poses, log_scale, K, CAUCHY_C)
    tr = TermOutput(tr_raw.residual, tr_raw.jacobian, tr_raw.weight / L2)
    return TermsFixture(X=X, X_valid=X_valid, Y=Y, poses=poses, log_scale=log_scale, K=K, anchors=anchors,
                        tracks=tracks, track_vis=track_vis, template_diameter=template_diameter, eps=eps,
                        rho_x=rho_x, rho_y=rho_y, pm=pm, plan=plan, tr_raw=tr_raw, tr=tr)


def random_terms(T: int, n: int, d: int, P: int, seed: int) -> TermOutput:
    """Gaussian residuals/Jacobians and U(-0.5, 1) weights (negative weights exercise the clamp), float64."""
    gen = torch.Generator().manual_seed(seed)
    jacobian = torch.randn(T, n, d, P, generator=gen, dtype=torch.float64)
    residual = torch.randn(T, n, d, generator=gen, dtype=torch.float64)
    weight = torch.rand(T, n, generator=gen, dtype=torch.float64) * 1.5 - 0.5
    return TermOutput(residual=residual, jacobian=jacobian, weight=weight)
