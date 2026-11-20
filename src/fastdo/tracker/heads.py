"""Output heads of the amortized tracker.

Per-observation weights and Sinkhorn duals come from small MLPs that read each observation's own token together with
its frame embedding; per-clip damping and log-scale increment are masked means over frames; per-frame increments
and diagonal precision come from a frame MLP. The last layer of every head is zero-initialized with biases that make
an untrained model start as the solver (damping `damping_init`, weights `weight_init`, zero increments and duals).
Masked entries are exact: weights and duals 0, `xi_hat` 0, `precision_diag` 1. Masks use `torch.where` only.

Called directly, the heads assume masked slots hold finite values; `AmortizedTracker` masks its inputs at entry.
"""

from __future__ import annotations

import logging
import math
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from fastdo.tracker.config import TrackerConfig
from fastdo.tracker.features import TRACK_TOKEN_DIM
from fastdo.tracker.normalize import CLUSTER_TOKEN_DIM, POINT_TOKEN_DIM
from fastdo.tracker.pack import TrackerInputs

__all__ = ["ObservationHead", "TrackerHeads", "TrackerOutput"]

logger = logging.getLogger(__name__)

_FRAME_CHANNELS = 14    # damping (1) | xi_hat (6) | precision (6) | sigma_hat (1)


class TrackerOutput(NamedTuple):
    """Per-clip `(B,)` and per-frame `(B,W,...)` outputs in the units the LM glue and training consume."""

    point_weights: Tensor    # (B,W,N) multiplicative override of pm.weight
    track_weights: Tensor    # (B,W,J) multiplicative override of the raw Cauchy weight
    damping: Tensor          # (B,)    Marquardt lam of solve_block_tridiagonal
    xi_hat: Tensor           # (B,W,6) body-frame canonical increment [rho / L, theta]
    sigma_hat: Tensor        # (B,)    log-scale increment
    precision_diag: Tensor   # (B,W,6) diagonal precision of the body-frame canonical error
    dual_g: Tensor           # (B,W,C) cluster Sinkhorn potential / eps
    dual_f: Tensor           # (B,W,N) point Sinkhorn potential / eps


class ObservationHead(nn.Module):
    """`(B,W,S,d_obs)` tokens with their `(B,W,D)` frame embedding -> `(B,W,S,n_out)`."""

    def __init__(self, d_obs: int, width: int, hidden: int, n_out: int) -> None:
        super().__init__()
        self.obs = nn.Linear(d_obs, hidden)
        self.ctx = nn.Linear(width, hidden)
        self.hidden = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, n_out)

    def forward(self, obs: Tensor, frame_embed: Tensor) -> Tensor:
        u = F.gelu(self.obs(obs) + self.ctx(frame_embed)[:, :, None, :])
        return self.out(F.gelu(self.hidden(u)))


class _FrameHead(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.hidden = nn.Linear(width, width)
        self.out = nn.Linear(width, _FRAME_CHANNELS)

    def forward(self, frame_embed: Tensor) -> Tensor:
        return self.out(F.gelu(self.hidden(frame_embed)))


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


class TrackerHeads(nn.Module):
    """Frame embedding `(B,W,D)` and the (entry-masked) `TrackerInputs` -> `TrackerOutput`."""

    def __init__(self, cfg: TrackerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        D, H = cfg.width, cfg.head_hidden
        self.frame_head = _FrameHead(D)
        self.track_head = ObservationHead(TRACK_TOKEN_DIM, D, H, 1)
        self.point_head = ObservationHead(POINT_TOKEN_DIM, D, H, 2)
        self.cluster_head = ObservationHead(CLUSTER_TOKEN_DIM, D, H, 1)
        lo, hi = cfg.damping_log10_range
        weight_bias = _logit((cfg.weight_init - cfg.weight_floor) / (1.0 - cfg.weight_floor))
        with torch.no_grad():
            for last in (self.frame_head.out, self.track_head.out, self.point_head.out, self.cluster_head.out):
                last.weight.zero_()
                last.bias.zero_()
            self.frame_head.out.bias[0] = _logit((math.log10(cfg.damping_init) - lo) / (hi - lo))
            self.track_head.out.bias[0] = weight_bias
            self.point_head.out.bias[0] = weight_bias

    def _weights(self, logits: Tensor, mask: Tensor) -> Tensor:
        floor = self.cfg.weight_floor
        w = floor + (1.0 - floor) * torch.sigmoid(logits)
        return torch.where(mask, w, torch.zeros((), dtype=w.dtype, device=w.device))

    def forward(self, frame_embed: Tensor, inputs: TrackerInputs) -> TrackerOutput:
        cfg, fm, z = self.cfg, inputs.frame_mask, frame_embed
        zero = torch.zeros((), dtype=z.dtype, device=z.device)
        count = fm.sum(-1).clamp_min(1).to(z.dtype)

        def masked_mean(v: Tensor) -> Tensor:
            return torch.where(fm, v, zero).sum(-1) / count

        f = self.frame_head(z)
        lo, hi = cfg.damping_log10_range
        damping = torch.pow(10.0, lo + (hi - lo) * torch.sigmoid(masked_mean(f[..., 0])))
        fm6 = fm[..., None]
        xi_hat = torch.where(fm6, f[..., 1:7], zero)
        precision = torch.where(fm6, F.softplus(f[..., 7:13]) + cfg.precision_floor, torch.ones((), dtype=z.dtype,
                                                                                               device=z.device))
        sigma_hat = masked_mean(f[..., 13])
        track_valid = inputs.track_mask & (inputs.track[..., 5] > 0) & fm6
        track_weights = self._weights(self.track_head(inputs.track, z)[..., 0], track_valid)
        point_valid = inputs.point_mask & fm6
        point_out = self.point_head(inputs.point, z)
        point_weights = self._weights(point_out[..., 0], point_valid)
        dual_f = torch.where(point_valid, point_out[..., 1], zero)
        cluster_valid = inputs.cluster_mask & fm6
        dual_g = torch.where(cluster_valid, self.cluster_head(inputs.cluster, z)[..., 0], zero)
        return TrackerOutput(point_weights=point_weights, track_weights=track_weights, damping=damping,
                             xi_hat=xi_hat, sigma_hat=sigma_hat, precision_diag=precision, dual_g=dual_g,
                             dual_f=dual_f)
