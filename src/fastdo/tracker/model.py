"""The amortized tracker network: validation, entry masking, backbone and heads.

The only inputs are `TrackerInputs` (scale-free tokens and bool masks): no intrinsics, pose, scale or apparent size
reaches the network. The forward checks shapes only, casts floating inputs to the parameter dtype and masks the bundle
at entry with `torch.where`, so NaN or inf in a masked slot reaches no activation, output or gradient. No host read
and no value-dependent Python branch, so the forward is CUDA-graph capturable.
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor, nn

from fastdo.tracker.backbone import TemporalTrackerBackbone
from fastdo.tracker.config import TrackerConfig
from fastdo.tracker.features import TRACK_TOKEN_DIM
from fastdo.tracker.heads import TrackerHeads, TrackerOutput
from fastdo.tracker.normalize import CLUSTER_TOKEN_DIM, POINT_TOKEN_DIM
from fastdo.tracker.pack import TrackerInputs

__all__ = ["AmortizedTracker"]

logger = logging.getLogger(__name__)


def _check_inputs(inputs: TrackerInputs, cfg: TrackerConfig) -> None:
    """Shape, width and mask-dtype validation (ValueError); reads shapes and dtypes only."""
    frame = inputs.frame
    if frame.dim() != 3 or frame.shape[-1] != cfg.feature_dim:
        raise ValueError(f"frame must be (B,W,{cfg.feature_dim}), got {tuple(frame.shape)}")
    bw = tuple(frame.shape[:2])
    sets = (("track", inputs.track, inputs.track_mask, TRACK_TOKEN_DIM),
            ("point", inputs.point, inputs.point_mask, POINT_TOKEN_DIM),
            ("cluster", inputs.cluster, inputs.cluster_mask, CLUSTER_TOKEN_DIM))
    if inputs.frame_mask.dtype != torch.bool or tuple(inputs.frame_mask.shape) != bw:
        raise ValueError(f"frame_mask must be bool {bw}, got {inputs.frame_mask.dtype} "
                         f"{tuple(inputs.frame_mask.shape)}")
    for name, tok, mask, dim in sets:
        if tok.dim() != 4 or tuple(tok.shape[:2]) != bw or tok.shape[-1] != dim:
            raise ValueError(f"{name} must be {bw + ('S', dim)}, got {tuple(tok.shape)}")
        if mask.dtype != torch.bool or mask.shape != tok.shape[:-1]:
            raise ValueError(f"{name}_mask must be bool {tuple(tok.shape[:-1])}, got {mask.dtype} {tuple(mask.shape)}")
        if not tok.is_floating_point():
            raise ValueError(f"{name} tokens must be floating point, got {tok.dtype}")
    if inputs.cluster.shape[2] != cfg.n_clusters:
        raise ValueError(f"need {cfg.n_clusters} clusters, got {inputs.cluster.shape[2]}")
    if inputs.track.shape[2] > cfg.max_tracks:
        raise ValueError(f"at most {cfg.max_tracks} tracks, got {inputs.track.shape[2]}")
    if not frame.is_floating_point():
        raise ValueError(f"frame tokens must be floating point, got {frame.dtype}")


def _mask_at_entry(inputs: TrackerInputs, dtype: torch.dtype) -> TrackerInputs:
    """Cast tokens to `dtype`; zero masked frames and every slot whose set mask (AND frame mask) is False."""
    fm = inputs.frame_mask
    zero = torch.zeros((), dtype=dtype, device=fm.device)
    frame = torch.where(fm[..., None], inputs.frame.to(dtype), zero)
    masked: list[Tensor] = []
    for tok, mask in ((inputs.track, inputs.track_mask), (inputs.point, inputs.point_mask),
                      (inputs.cluster, inputs.cluster_mask)):
        m = mask & fm[..., None]
        masked += [torch.where(m[..., None], tok.to(dtype), zero), m]
    return TrackerInputs(frame, fm, *masked)


class AmortizedTracker(nn.Module):
    """Temporal transformer over scale-free per-frame tokens emitting solver-step controls (design Sections 2-5)."""

    def __init__(self, cfg: TrackerConfig | None = None) -> None:
        super().__init__()
        self.cfg = TrackerConfig() if cfg is None else cfg
        self.backbone = TemporalTrackerBackbone(self.cfg)
        self.heads = TrackerHeads(self.cfg)
        n_params = sum(p.numel() for p in self.parameters())
        logger.info("AmortizedTracker: %d parameters (width %d, layers %d, heads %d)", n_params, self.cfg.width,
                    self.cfg.n_layers, self.cfg.n_heads)

    def forward(self, inputs: TrackerInputs) -> TrackerOutput:
        _check_inputs(inputs, self.cfg)
        masked = _mask_at_entry(inputs, next(self.parameters()).dtype)
        return self.heads(self.backbone(masked), masked)
