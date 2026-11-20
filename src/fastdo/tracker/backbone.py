"""Temporal transformer backbone of the amortized tracker.

Frame scalars, a set-attention pool over tracks and one over template clusters are summed per frame, then
`n_layers` pre-norm blocks attend over frame index with a learned relative position bias keyed by `min(|i-j|, R)`
(no absolute position, no causal mask, so the backbone is time-reversal equivariant). Explicit matmul attention so
float64 CPU tests see exact mask semantics. Every mask is applied with `torch.where`, never a 0/1 multiply.

Called directly, the backbone assumes masked slots hold finite values; `AmortizedTracker` masks its inputs at entry.
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from fastdo.tracker.config import TrackerConfig
from fastdo.tracker.features import TRACK_TOKEN_DIM
from fastdo.tracker.normalize import CLUSTER_TOKEN_DIM
from fastdo.tracker.pack import TrackerInputs

__all__ = ["RelativePositionBias", "SetPool", "TemporalBlock", "TemporalTrackerBackbone"]

logger = logging.getLogger(__name__)


class RelativePositionBias(nn.Module):
    """(h,W,W) attention bias `E[min(|i-j|, R)]` from an `(R+1, h)` table."""

    def __init__(self, max_distance: int, n_heads: int) -> None:
        super().__init__()
        self.max_distance = max_distance
        self.table = nn.Embedding(max_distance + 1, n_heads)

    def forward(self, n_frames: int, device: torch.device) -> Tensor:
        pos = torch.arange(n_frames, device=device)
        dist = (pos[:, None] - pos[None, :]).abs().clamp_max(self.max_distance)
        return self.table(dist).permute(2, 0, 1)


class SetPool(nn.Module):
    """Masked multi-head attention pooling of a set `(B,W,S,d_in)` to `(B,W,D)`; an all-masked set gives 0."""

    def __init__(self, d_in: int, width: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.embed_in = nn.Linear(d_in, width)
        self.embed_out = nn.Linear(width, width)
        self.score = nn.Linear(width, n_heads)
        self.out = nn.Linear(width, width)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        B, W, S, _ = x.shape
        D = self.out.in_features
        m = mask[..., None]
        e = self.embed_out(F.gelu(self.embed_in(x)))
        a = self.score(e)
        e = torch.where(m, e, torch.zeros((), dtype=e.dtype, device=e.device))
        a = a.masked_fill(~m, torch.finfo(a.dtype).min)
        a = torch.where(m, torch.softmax(a, dim=-2), torch.zeros((), dtype=a.dtype, device=a.device))
        pooled = (a[..., None] * e.view(B, W, S, self.n_heads, D // self.n_heads)).sum(-3).reshape(B, W, D)
        return self.out(pooled)


class TemporalBlock(nn.Module):
    """Pre-norm self-attention over frames with a relative position bias, then a GELU MLP."""

    def __init__(self, cfg: TrackerConfig) -> None:
        super().__init__()
        D = cfg.width
        self.n_heads = cfg.n_heads
        self.norm1 = nn.LayerNorm(D)
        self.qkv = nn.Linear(D, 3 * D)
        self.proj = nn.Linear(D, D)
        self.norm2 = nn.LayerNorm(D)
        self.mlp_in = nn.Linear(D, cfg.ffn_ratio * D)
        self.mlp_out = nn.Linear(cfg.ffn_ratio * D, D)
        self.bias = RelativePositionBias(cfg.max_distance, cfg.n_heads)

    def forward(self, x: Tensor, frame_mask: Tensor) -> Tensor:
        B, W, D = x.shape
        h = self.n_heads
        q, k, v = (t.view(B, W, h, D // h).transpose(1, 2) for t in self.qkv(self.norm1(x)).split(D, dim=-1))
        v = torch.where(frame_mask[:, None, :, None], v, torch.zeros((), dtype=v.dtype, device=v.device))
        logits = q @ k.transpose(-1, -2) / math.sqrt(D // h) + self.bias(W, x.device)[None]
        logits = logits.masked_fill(~frame_mask[:, None, None, :], torch.finfo(logits.dtype).min)
        attn = torch.softmax(logits, dim=-1)
        x = x + self.proj((attn @ v).transpose(1, 2).reshape(B, W, D))
        return x + self.mlp_out(F.gelu(self.mlp_in(self.norm2(x))))


class TemporalTrackerBackbone(nn.Module):
    """`TrackerInputs` -> per-frame embedding `(B,W,D)`, exactly 0 on masked frames."""

    def __init__(self, cfg: TrackerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self._mass_lo = 4 * cfg.n_params + cfg.n_bins
        self._mass_hi = self._mass_lo + cfg.n_clusters
        D = cfg.width
        self.frame_enc = nn.Linear(cfg.feature_dim - cfg.n_clusters, D)
        self.track_pool = SetPool(TRACK_TOKEN_DIM, D, cfg.n_heads)
        self.cluster_pool = SetPool(CLUSTER_TOKEN_DIM, D, cfg.n_heads)
        self.ln_in = nn.LayerNorm(D)
        self.blocks = nn.ModuleList(TemporalBlock(cfg) for _ in range(cfg.n_layers))
        self.ln_out = nn.LayerNorm(D)

    def forward(self, inputs: TrackerInputs) -> Tensor:
        fm = inputs.frame_mask
        frame = inputs.frame
        scalars = torch.cat([frame[..., :self._mass_lo], frame[..., self._mass_hi:]], dim=-1)
        x = (self.frame_enc(scalars) + self.track_pool(inputs.track, inputs.track_mask)
             + self.cluster_pool(inputs.cluster, inputs.cluster_mask))
        zero = torch.zeros((), dtype=x.dtype, device=x.device)
        x = torch.where(fm[..., None], self.ln_in(x), zero)
        for block in self.blocks:
            x = block(x, fm)
        return torch.where(fm[..., None], self.ln_out(x), zero)
