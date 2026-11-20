# Copyright 2026 The HandReel authors. SPDX-License-Identifier: Apache-2.0
# Portions adapted from DO AS I DO (https://github.com/malik-group/do-as-i-do, commit 824591b),
# Copyright (c) 2026 malik-group, MIT License; see licenses/DO-AS-I-DO-MIT.txt and THIRD_PARTY_LICENSES.md.
"""The D1 update: antithetic knot noise, Blom rank weights, acceptance, and the registered update for equivalence.

Linear-regime argument (design doc section 4, D1). For f(eps) ~ g^T eps, a rank-weighted estimator
sum_i w_i eps_(i) has signal sum_i w_i E[z_(i)] along g and perpendicular variance (d-1) sum_i w_i^2. Weights
proportional to the expected normal order statistics maximise the ratio; the registered uniform top-q mean reaches
phi(z_{1-q})^2 / q of it (0.308 at q = 0.1). `rank_weights` uses Blom's approximation of the expected order
statistics, centred, and scales them so that sum_i w_i b_i = phi(z_{1-q}) / q: the same expected step along g as the
registered top-q mean, with lower variance.
"""

from __future__ import annotations

import math
from statistics import NormalDist

import torch
import torch.nn.functional as F

__all__ = [
    "antithetic_knot_noise",
    "blom_scores",
    "knot_interp",
    "noisy_rows",
    "rank_update",
    "rank_weights",
    "registered_expected_step",
    "registered_update",
    "registered_weights",
]


def registered_expected_step(q: float = 0.1) -> float:
    """E[mean of the top-q fraction of N(0,1) samples] = phi(z_{1-q}) / q (large N)."""
    z = NormalDist().inv_cdf(1.0 - q)
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi) / q


def blom_scores(n: int) -> torch.Tensor:
    """Blom's approximation of the expected order statistics of n N(0,1) draws, ascending, centred (sum 0)."""
    if n < 2:
        raise ValueError(f"need >= 2 ranked samples, got {n}")
    nd = NormalDist()
    s = torch.tensor([nd.inv_cdf((i - 0.375) / (n + 0.25)) for i in range(1, n + 1)], dtype=torch.float64)
    return s - s.mean()


def noisy_rows(noise_knots: torch.Tensor) -> torch.Tensor:
    """(N,) bool: rows whose knot noise is not identically zero."""
    return noise_knots.reshape(noise_knots.shape[0], -1).abs().amax(dim=1) > 0


def rank_weights(rews: torch.Tensor, noisy: torch.Tensor, q: float = 0.1) -> torch.Tensor:
    """(N,) weights: Blom scores by reward rank over the noisy rows (0 elsewhere), scaled to the registered step.

    Non-finite rewards rank lowest. Ties keep their input order (stable sort)."""
    r = rews.detach().to(torch.float64).clone()
    r[~torch.isfinite(r)] = -math.inf
    idx = torch.nonzero(noisy, as_tuple=False).reshape(-1)
    b = blom_scores(int(idx.numel())).to(r.device)
    order = torch.argsort(r[idx], stable=True)          # ascending reward
    w_sub = torch.empty_like(b)
    w_sub[order] = b
    c = registered_expected_step(q) / float((b * b).sum())
    w = torch.zeros(rews.shape[0], dtype=torch.float64, device=rews.device)
    w[idx] = c * w_sub
    return w.to(rews.dtype)


def rank_update(base: torch.Tensor, samples: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """base + sum_i w_i (samples_i - base); base (H, nu), samples (N, H, nu), weights (N,)."""
    return base + torch.einsum("n,nhu->hu", weights.to(samples.dtype), samples - base.unsqueeze(0))


def antithetic_knot_noise(noise_scale: torch.Tensor, gns: torch.Tensor | float,
                          generator: torch.Generator | None = None) -> torch.Tensor:
    """(N, K, nu) knot noise: rows [0, N/2) drawn with the registered per-row scale, rows [N/2, N) their negation.

    Row 0 carries zero noise (the registered scale zeroes it), so row N/2 does too: the two nominal slots."""
    n = noise_scale.shape[0]
    if n % 2:
        raise ValueError(f"antithetic sampling needs an even sample count, got {n}")
    half = n // 2
    z = torch.randn(noise_scale[:half].shape, generator=generator, device=noise_scale.device,
                    dtype=noise_scale.dtype)
    base = z * noise_scale[:half] * gns
    base[0] = 0.0
    return torch.cat([base, -base], dim=0)


def knot_interp(knots: torch.Tensor, n: int) -> torch.Tensor:
    """DO AS I DO's `interp(src, n, order=1)`: (N, K, D) -> (N, K*n, D), linear, align_corners=True."""
    big_n, k, _ = knots.shape
    if k <= 1:
        return knots.repeat(1, n, 1)
    src = knots if knots.is_floating_point() else knots.to(torch.float32)
    out = F.interpolate(src.permute(0, 2, 1), size=k * n, mode="linear", align_corners=True)
    return out.permute(0, 2, 1)


def registered_weights(rews: torch.Tensor, num_samples: int, temperature: float) -> tuple[torch.Tensor, torch.Tensor]:
    """DO AS I DO's `_compute_weights_impl`, op for op (equivalence mode and KA-E)."""
    nan_mask = torch.isnan(rews) | torch.isinf(rews)
    safe_rews = torch.where(nan_mask, torch.tensor(torch.inf, device=rews.device), rews)
    rews_min = safe_rews.min()
    rews_min = torch.where(torch.isinf(rews_min), torch.tensor(-1000.0, device=rews.device), rews_min)
    rews = torch.where(nan_mask, rews_min, rews)
    top_k = max(1, int(0.1 * num_samples))
    top_indices = torch.topk(rews, k=top_k, largest=True).indices
    weights = torch.zeros_like(rews)
    top_rews = rews[top_indices]
    top_rews_normalized = (top_rews - top_rews.mean()) / (top_rews.std() + 1e-2)
    weights[top_indices] = F.softmax(top_rews_normalized / temperature, dim=0)
    return weights, nan_mask


def registered_update(samples: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """DO AS I DO's update: the weighted mean of the samples."""
    return (weights[:, None, None] * samples).sum(dim=0)
