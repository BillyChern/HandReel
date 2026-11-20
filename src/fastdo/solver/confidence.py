"""Per-frame confidence of a trajectory solve: support-invariant and normalized by the clip's noise floor.

The earlier `exp(-rms / (0.05 D))` read the pointmap term's own residual, `sqrt(N w_i) (x_i - pbar_i)`, whose RMS
depends on how many observations a frame has (it grows as `sqrt(N / N_valid)` below the `n_obs` cap) and on the
plan's row masses. The visible, foreshortening-weighted template changed those masses and barycenters, and the
confidence of well-converged DexYCB frames fell from ~0.6 to 0.17-0.32 with no pose-accuracy change. This module
reads the same fine-level refresh, but through statistics that neither support nor mass level can move.

For frame k with valid rows i (observations `x_i`, plan barycenters `pbar_i`, row masses `w_i`):

* kept rows: valid, `w_i > 0` and `w_i >= mass_floor * mean_valid(w)` (observations the partial OT left unmatched
  carry almost no mass and do not enter the statistics);
* matched distance `r_i = |x_i - pbar_i|` (read back from the residual as `|residual_i| / sqrt(N w_i)`);
* `m_k = sqrt(sum_kept w_i r_i^2 / sum_kept w_i)`, the mass-weighted RMS matched distance (metres);
* the clip's noise floor `sigma = 1.4826 * median r_i` over the kept rows of every supported frame of that clip: the
  median absolute deviation about zero (the residual of a perfect fit), scaled to a Gaussian standard deviation;
  a median over the whole clip, so a minority of badly posed frames does not move it;
* `statistic_k = m_k / sigma` (~1 for a frame at the noise floor; +inf without support);
* `confidence_k = exp(-statistic_k / scale)`, and exactly 0.0 for a frame with fewer than `min_support` valid rows.

Every per-clip quantity is computed on a copy of that clip's own rows, so a batch of clips scores each clip bitwise
as a one-clip call would (`solve_trajectory` is the B = 1 case of `solve_trajectories`).

Semantics:

* A WITHIN-CLIP relative signal. Normalizing by the clip's own floor removes support and template-density effects
  and also removes every error all of a clip's frames share: a clip-wide bias along the camera ray (DexYCB cracker box
  ~6 mm, sugar box ~4 mm, reached even from the exact ground-truth start) or a whole-clip shift reads as the floor.
  Downstream QC (features, acceptance calibration) must not accept or rank clips across clips on it; a clip-level
  confidence needs a gauge uncertainty (a conformal gauge interval). Over clips it does not rank error
  (pooled DexYCB oracle Spearman -0.21); within clips it does weakly (+0.109 can, +0.143 cracker, -0.211 sugar).
* At the floor: `statistic` ~1 by construction for 1-D Gaussian distances (a frame there scores exp(-1 / scale),
  0.49 at the shipped 1.4). Real matched distances are 3-D and template-spacing dominated: well-converged synthetic
  box frames read 0.82-0.92 (confidence 0.52-0.56 at 1.4), drifting ~12% as the template is sampled 256 -> 2048
  points and ~3% as observations go 128 -> 1024 (mass floor 0.7). DexYCB frames sit in a narrow band (0.36-0.72 at
  1.4, clip floors 8.3-9.5 mm), so the 90% coverage of accurate frames holds by placing the scale just below that
  band (coverage 0.68 at 1.3, 0.92 at 1.4, 0.96 at 1.5), not by separating accurate from inaccurate frames (91% of
  frames >= 5 mm also score > 0.5).
* Detection limits within a clip (shipped defaults, one frame of the synthetic box held at its ground-truth pose,
  frame confidence 0.50-0.55): 2 cm along the camera ray -> 0.05-0.07; 2 cm across it -> 0.34-0.37 (4 cm 0.20-0.26);
  10 / 20 deg about the box's z axis -> 0.46-0.52 / 0.40-0.46, about x or y -> 0.35-0.46 / 0.22-0.37. Lateral and
  rotation errors are weakly visible; QC must not rely on the confidence for them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

__all__ = ["MAD_TO_SIGMA", "FrameConfidence", "frame_confidence"]

MAD_TO_SIGMA = 1.4826   # median |e| of a zero-mean Gaussian residual e, times this, estimates its standard deviation


@dataclass(frozen=True)
class FrameConfidence:
    """`confidence` (T,) in [0, 1]; `statistic` (T,) `m_k / sigma` (+inf without support); `matched_rms` (T,) `m_k` in
    metres (0 without support); `noise_floor` (B,) each clip's `sigma` in metres (NaN for a clip with no support)."""

    confidence: Tensor
    statistic: Tensor
    matched_rms: Tensor
    noise_floor: Tensor


def frame_confidence(residual: Tensor, row_mass: Tensor, valid: Tensor, frame_slices: Sequence[slice], *,
                     scale: float, mass_floor: float, min_support: int) -> FrameConfidence:
    """The per-frame confidence of packed clips (see the module docstring for the definition).

    Args:
        residual: (T,N,d) fine-level pointmap residual `sqrt(N w_i) (x_i - pbar_i)`, zero on invalid rows
            (`terms.pointmap_term`, `point_to_plane=False`).
        row_mass: (T,N) the plan's row sums `w_i` at that refresh.
        valid: (T,N) bool valid observation rows.
        frame_slices: one slice of frames per clip, covering `0..T` in order.
        scale: the calibrated divisor of the statistic (`SolverConfig.confidence_scale`), > 0.
        mass_floor: kept rows need `w_i >= mass_floor * mean_valid(w)` (`SolverConfig.confidence_mass_floor`),
            in [0, 1].
        min_support: frames with fewer valid rows score exactly 0.0.

    Raises:
        ValueError: `scale` not positive or `mass_floor` outside [0, 1].
    """
    if not scale > 0.0:
        raise ValueError(f"confidence scale must be positive, got {scale}")
    if not 0.0 <= mass_floor <= 1.0:
        raise ValueError(f"confidence mass floor must be in [0, 1], got {mass_floor}")
    parts = [_clip_confidence(residual[sl].clone(), row_mass[sl].clone(), valid[sl].clone(), scale, mass_floor,
                              min_support) for sl in frame_slices]
    return FrameConfidence(confidence=torch.cat([p.confidence for p in parts]),
                           statistic=torch.cat([p.statistic for p in parts]),
                           matched_rms=torch.cat([p.matched_rms for p in parts]),
                           noise_floor=torch.cat([p.noise_floor for p in parts]))


def _clip_confidence(residual: Tensor, row_mass: Tensor, valid: Tensor, scale: float, mass_floor: float,
                     min_support: int) -> FrameConfidence:
    """One clip's frames (no host synchronization)."""
    n_rows = residual.shape[1]
    dtype = residual.dtype
    tiny = torch.finfo(dtype).tiny
    zero = torch.zeros((), dtype=dtype, device=residual.device)
    n_valid = valid.sum(1)
    supported = n_valid >= min_support
    w = torch.where(valid, row_mass.to(dtype), zero)
    mean_w = w.sum(1) / n_valid.clamp_min(1).to(dtype)
    kept = valid & (w > 0) & (w >= mass_floor * mean_w[:, None]) & supported[:, None]
    usable = kept.any(1)                                                 # implies supported
    sq = residual.square().sum(-1)                                       # N w_i r_i^2
    kept_mass = n_rows * torch.where(kept, w, zero).sum(1)
    rms = torch.where(usable, (torch.where(kept, sq, zero).sum(1) / kept_mass.clamp_min(tiny)).sqrt(), zero)
    dist = torch.where(kept, (sq / (n_rows * w).clamp_min(tiny)).sqrt(), torch.full((), float("nan"), dtype=dtype,
                                                                                   device=residual.device))
    sigma = MAD_TO_SIGMA * torch.nanmedian(dist.flatten())
    statistic = torch.where(usable, rms / sigma.clamp_min(tiny), torch.full((), float("inf"), dtype=dtype,
                                                                            device=residual.device))
    confidence = torch.where(usable, torch.exp(-statistic / scale), zero)
    return FrameConfidence(confidence=confidence, statistic=statistic, matched_rms=rms, noise_floor=sigma.reshape(1))
