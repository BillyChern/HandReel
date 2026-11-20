"""The interaction gate network (engine stage 2): a per-frame MLP on 8 pooled DINOv2 tokens.

Each of the `n_frames` sampled rows is `[token (token_dim), hand_conf (1)]`. The token part is layer-normalized (no
affine), the hand_conf column is appended back, and a shared two-layer GELU MLP gives `h_t`. Heads:

- `z_t`, a per-frame interaction logit, pooled into the clip's `interact_logit` by an LSE smooth max (multiple-instance
  learning: one manipulating frame makes the clip positive); `p_interact = sigmoid(interact_logit)` is the only output
  the accept decision uses.
- `keyframe_scores_t`, a per-frame score (a later keyframe or detector stage reads its argmax).
- `rigid_logit` and `visibility` from `cat(mean_t h_t, max_t h_t)`. No source labels rigidity yet, so `p_rigid` is
  uncalibrated and never used.

Every clip output is invariant to the order of the frames. About 2.8e5 parameters at `token_dim` 2048.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import Tensor, nn

__all__ = ["GateConfig", "GateOutput", "InteractionGate", "lse_pool"]


@dataclass(frozen=True)
class GateConfig:
    """Architecture of `InteractionGate`: token width, MLP width, frames per clip, LSE temperature and input dropout."""

    token_dim: int
    hidden: int = 128
    n_frames: int = 8
    lse_temperature: float = 5.0
    dropout: float = 0.1


class GateOutput(NamedTuple):
    """Clip outputs `(B,)` and per-frame keyframe scores `(B, n_frames)`; the logits are kept for stable BCE."""

    p_interact: Tensor
    p_rigid: Tensor
    visibility: Tensor
    keyframe_scores: Tensor
    interact_logit: Tensor
    rigid_logit: Tensor


def lse_pool(z: Tensor, temperature: float) -> Tensor:
    """`(1 / r) * log(mean(exp(r * z)))` over the last dim, via `logsumexp - log(n)`: between the mean and the max.

    Raises:
        ValueError: If `temperature <= 0`.
    """
    if temperature <= 0:
        raise ValueError(f"lse temperature must be > 0, got {temperature}")
    return (torch.logsumexp(temperature * z, dim=-1) - math.log(int(z.shape[-1]))) / temperature


class InteractionGate(nn.Module):
    """The gate network (module docstring). Built weights come from the global torch RNG: seed before building."""

    def __init__(self, cfg: GateConfig) -> None:
        super().__init__()
        if cfg.token_dim < 1 or cfg.hidden < 1 or cfg.n_frames < 1:
            raise ValueError(f"GateConfig needs token_dim, hidden and n_frames >= 1, got {cfg}")
        self.cfg = cfg
        self.norm = nn.LayerNorm(cfg.token_dim, elementwise_affine=False)
        self.dropout = nn.Dropout(cfg.dropout)
        self.frame_encoder = nn.Sequential(nn.Linear(cfg.token_dim + 1, cfg.hidden), nn.GELU(),
                                           nn.Linear(cfg.hidden, cfg.hidden), nn.GELU())
        self.interact_head = nn.Linear(cfg.hidden, 1)
        self.keyframe_head = nn.Linear(cfg.hidden, 1)
        self.rigid_head = nn.Linear(2 * cfg.hidden, 1)
        self.visibility_head = nn.Linear(2 * cfg.hidden, 1)

    def forward(self, features: Tensor) -> GateOutput:
        """Gate outputs for `(B, n_frames, token_dim + 1)` features.

        Raises:
            ValueError: On other feature dims; the message names the expected ones.
        """
        n_frames, width = self.cfg.n_frames, self.cfg.token_dim + 1
        if features.ndim != 3 or int(features.shape[1]) != n_frames or int(features.shape[2]) != width:
            raise ValueError(f"gate features must be (B, {n_frames}, {width}) (n_frames, token_dim + 1 hand_conf "
                             f"column), got {tuple(features.shape)}")
        x = torch.cat([self.norm(features[..., :-1]), features[..., -1:]], dim=-1)
        h = self.frame_encoder(self.dropout(x))
        z = self.interact_head(h).squeeze(-1)
        keyframe_scores = self.keyframe_head(h).squeeze(-1)
        interact_logit = lse_pool(z, self.cfg.lse_temperature)
        pooled = torch.cat([h.mean(dim=1), h.amax(dim=1)], dim=-1)
        rigid_logit = self.rigid_head(pooled).squeeze(-1)
        visibility = torch.sigmoid(self.visibility_head(pooled).squeeze(-1))
        return GateOutput(p_interact=torch.sigmoid(interact_logit), p_rigid=torch.sigmoid(rigid_logit),
                          visibility=visibility, keyframe_scores=keyframe_scores, interact_logit=interact_logit,
                          rigid_logit=rigid_logit)
