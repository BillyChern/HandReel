"""Packing feature tokens into the tracker's batched, masked, scale-free inputs.

`build_tracker_inputs` normalizes per packed frame in float64, marks a frame invalid when any of its frame, point or
cluster tokens (or its apparent size) is non-finite, zeroes it with `torch.where` (never a 0/1 multiply) and packs
the solver's `ClipLayout` into `(B,W,...)` tensors. Every `TrackerInputs` tensor is finite. No host read except the
layout's host ints.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch import Tensor

from fastdo.solver.banded import ClipLayout
from fastdo.solver.ot_sparse import SparsePlan
from fastdo.solver.terms import TermOutput
from fastdo.tracker.features import frame_feature_dim, plan_row_mass
from fastdo.tracker.normalize import (
    _normalize_track,
    _per_frame,
    cluster_tokens,
    normalize_frame_tokens,
    point_tokens,
    track_image_scale,
)

__all__ = ["TrackerInputs", "build_tracker_inputs", "clip_frame_mask", "clips_to_frames", "frames_to_clips"]

logger = logging.getLogger(__name__)

F64 = torch.float64


@dataclass(frozen=True)
class TrackerInputs:
    """The network's only inputs: scale-free tokens and bool masks, all `(B,W,...)` over clips and frame slots."""

    frame: Tensor          # (B,W,F)   normalized frame tokens
    frame_mask: Tensor     # (B,W)     bool: a real frame whose tokens are finite
    track: Tensor          # (B,W,J,6) normalized track tokens
    track_mask: Tensor     # (B,W,J)   bool: a real track slot
    point: Tensor          # (B,W,N,7) point tokens
    point_mask: Tensor     # (B,W,N)   bool: pm.weight > 0
    cluster: Tensor        # (B,W,C,5) cluster tokens
    cluster_mask: Tensor   # (B,W,C)   bool: a non-empty cluster


def frames_to_clips(x: Tensor, layout: ClipLayout) -> Tensor:
    """(T,...) -> (B,W,...) placed by `layout.slot`, zeros (False) in padded slots."""
    if x.shape[0] != layout.n_frames:
        raise ValueError(f"x has {x.shape[0]} frames, the layout {layout.n_frames}")
    B, W = layout.n_clips, layout.width
    out = x.new_zeros((B * W,) + tuple(x.shape[1:])).index_copy_(0, layout.slot, x)
    return out.view((B, W) + tuple(x.shape[1:]))


def clips_to_frames(x: Tensor, layout: ClipLayout) -> Tensor:
    """(B,W,...) -> (T,...) in packed frame order."""
    B, W = layout.n_clips, layout.width
    if tuple(x.shape[:2]) != (B, W):
        raise ValueError(f"x must start with ({B},{W}), got {tuple(x.shape)}")
    return x.reshape((B * W,) + tuple(x.shape[2:])).index_select(0, layout.slot)


def clip_frame_mask(layout: ClipLayout) -> Tensor:
    """(B,W) bool, True on real frames."""
    return frames_to_clips(torch.ones(layout.n_frames, dtype=torch.bool, device=layout.slot.device), layout)


def build_tracker_inputs(frame_tokens: Tensor, track_tokens: Tensor, pm: TermOutput, X: Tensor,
                         plan: Tensor | SparsePlan, poses: Tensor, log_scale: Tensor,
                         template_diameter: Tensor | float, K: Tensor, centroids: Tensor, fraction: Tensor,
                         layout: ClipLayout | None = None, track_valid: Tensor | None = None,
                         n_params: int = 7, n_bins: int = 16) -> TrackerInputs:
    """Normalize the raw feature tokens per packed frame and pack them per clip.

    `frame_tokens` (T,F) and `track_tokens` (T,J,6) are the features' raw outputs; `pm`, `X`, `plan`, `poses`,
    `log_scale` and `template_diameter` are the state the tokens were built at; `K` (3,3) or (T,3,3);
    `centroids`/`fraction` from `cluster_geometry`, (C,3)/(C,) or gathered per frame (T,C,3)/(T,C). `layout=None` packs
    one clip.
    A frame is masked, with all its tokens 0 and set masks False, when its frame tokens, apparent size, any point
    token row or any cluster token row is non-finite; a non-finite track row clears only its slot.
    """
    if frame_tokens.dim() != 2:
        raise ValueError(f"frame_tokens must be (T,F), got {tuple(frame_tokens.shape)}")
    T, dev = frame_tokens.shape[0], frame_tokens.device
    if K.shape != (3, 3) and K.shape != (T, 3, 3):
        raise ValueError(f"K must be (3,3) or ({T},3,3), got {tuple(K.shape)}")
    if centroids.dim() not in (2, 3) or centroids.shape[-1] != 3:
        raise ValueError(f"centroids must be (C,3) or (T,C,3), got {tuple(centroids.shape)}")
    C = centroids.shape[-2]
    if fraction.shape[-1] != C or frame_tokens.shape[1] != frame_feature_dim(n_params, n_bins, C):
        raise ValueError(f"C disagrees: centroids {tuple(centroids.shape)}, fraction {tuple(fraction.shape)}, "
                         f"frame tokens {tuple(frame_tokens.shape)}")
    if track_tokens.dim() != 3 or track_tokens.shape[0] != T:
        raise ValueError(f"track_tokens must be ({T},J,6), got {tuple(track_tokens.shape)}")
    J = track_tokens.shape[1]
    if track_valid is not None and track_valid.shape != (T, J):
        raise ValueError(f"track_valid must be ({T},{J}), got {tuple(track_valid.shape)}")
    if layout is None:
        layout = ClipLayout.build((T,), dev)
    if sum(layout.lengths) != T:
        raise ValueError(f"layout lengths {layout.lengths} do not sum to T={T}")
    if log_scale.dim() > 1 or (log_scale.dim() == 1 and log_scale.shape[0] != T):
        raise ValueError(f"log_scale must be 0-dim or ({T},), got {tuple(log_scale.shape)}")
    L = log_scale.to(device=dev, dtype=F64).exp() * _per_frame(template_diameter, T, dev, "template_diameter")
    d_img = track_image_scale(K, poses, L)
    frame_n, frame_ok = normalize_frame_tokens(frame_tokens, d_img, n_params, n_bins, C)
    track_n, track_ok = _normalize_track(track_tokens, d_img)
    point_r = point_tokens(X, pm, plan_row_mass(plan), poses, L)
    clus_r = cluster_tokens(frame_tokens, centroids, fraction, n_params, n_bins)
    point_ok = torch.isfinite(point_r).all(-1)
    clus_ok = torch.isfinite(clus_r).all(-1)
    finite = frame_ok & point_ok.all(1) & clus_ok.all(1)
    zero = torch.zeros((), dtype=F64, device=dev)
    point = torch.where((finite[:, None] & point_ok)[..., None], point_r, zero)
    cluster = torch.where((finite[:, None] & clus_ok)[..., None], clus_r, zero)
    track = torch.where(finite[:, None, None], track_n, zero)
    frame = torch.where(finite[:, None], frame_n, zero)   # I9: a frame masked by point_ok/clus_ok is zeroed too
    tv = torch.ones(T, J, dtype=torch.bool, device=dev) if track_valid is None else track_valid.to(dev)
    frac_ok = fraction.to(dev).expand(T, C) > 0
    return TrackerInputs(
        frame=frames_to_clips(frame, layout), frame_mask=frames_to_clips(finite, layout),
        track=frames_to_clips(track, layout), track_mask=frames_to_clips(tv & track_ok & finite[:, None], layout),
        point=frames_to_clips(point, layout),
        point_mask=frames_to_clips((pm.weight > 0) & point_ok & finite[:, None], layout),
        cluster=frames_to_clips(cluster, layout),
        cluster_mask=frames_to_clips(frac_ok & clus_ok & finite[:, None], layout))
