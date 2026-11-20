"""Reporting quantities of the policy evaluation (design Section 9b amendments b-c; budget Section 2 amendment).

- Fallback rates are measured directly on held-out clips, f_BoK(K) for best-of-K and f_SMC(K) for twisted SMC, with
  Wilson score intervals (`wilson_interval`, `rate_estimate`); the fitted law is never the only number (at 300 clips
  x 8 rollouts the sd of pi_0 is about 0.06).
- `smallest_k_excess` inverts the fitted best-of-K law for the smallest K whose fallback rate exceeds its floor pi_0 by
  at most `excess` (5 % in amendment c); `smallest_measured_k` reads the same off the measured rates.
- Throughput (budget Section 2, amended): expected seconds per clip E[t] = t_rest + t_pi(K) + f t_fallback, where the
  throughput configuration (twisted SMC, rejected clips count against yield) has no fallback term, and accepted clips
  per GPU-hour = 3600 (1 - reject rate) / E[t].
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from fastdo.policy.fallback_law import smallest_k

__all__ = ["Z_95", "RateEstimate", "accepted_clips_per_gpu_hour", "expected_clip_seconds", "rate_estimate",
           "smallest_k_excess", "smallest_measured_k", "wilson_interval"]

Z_95 = 1.959963984540054          # two-sided 95 % standard normal quantile


def wilson_interval(failures: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion `failures / n` (Wilson 1927), clamped to [0, 1].

    Raises:
        ValueError: If `n < 1`, `failures` lies outside [0, n] or `z` is not positive.
    """
    if n < 1 or not 0 <= failures <= n or not z > 0.0:
        raise ValueError(f"need n >= 1, 0 <= failures <= n and z > 0, got {failures}, {n}, {z}")
    p, z2 = failures / n, z * z
    denominator = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denominator
    low = 0.0 if failures == 0 else max(0.0, centre - half)          # the bounds at 0 and n are exact, not rounded
    high = 1.0 if failures == n else min(1.0, centre + half)
    return low, high


@dataclass(frozen=True)
class RateEstimate:
    """A fallback rate measured on `n_clips` held-out clips: `failures` clips had no successful trajectory with `k`
    rollouts (`method` "best_of_k") or particles (`method` "smc")."""

    method: str
    k: int
    failures: int
    n_clips: int

    @property
    def rate(self) -> float:
        return self.failures / self.n_clips

    @property
    def interval(self) -> tuple[float, float]:
        return wilson_interval(self.failures, self.n_clips)

    def as_dict(self) -> dict[str, Any]:
        low, high = self.interval
        return {"method": self.method, "k": self.k, "failures": self.failures, "n_clips": self.n_clips,
                "rate": self.rate, "wilson95_low": low, "wilson95_high": high}


def rate_estimate(method: str, k: int, failed: Sequence[bool]) -> RateEstimate:
    """The `RateEstimate` of per-clip failure flags.

    Raises:
        ValueError: On an unknown method, `k < 1` or no clips.
    """
    if not (method == "best_of_k" or method.startswith("smc")) or k < 1 or len(failed) == 0:
        raise ValueError(f"need method best_of_k or smc[_variant], k >= 1 and at least one clip, got {method}, {k}, "
                         f"{len(failed)}")
    return RateEstimate(method=method, k=k, failures=sum(bool(f) for f in failed), n_clips=len(failed))


def smallest_k_excess(alpha: float, beta: float, pi_0: float, excess: float = 0.05, k_max: int = 4096) -> int | None:
    """Smallest K in [1, k_max] with f_BoK(K) - pi_0 = (1 - pi_0) B(alpha, beta + K) / B(alpha, beta) <= `excess`
    (`fastdo.policy.fallback_law.fallback_rate`), or None if even K = k_max does not reach it.

    Raises:
        ValueError: On invalid parameters or `excess` outside (0, 1).
    """
    if not 0.0 < excess < 1.0 or not 0.0 <= pi_0 < 1.0:
        raise ValueError(f"need excess in (0, 1) and pi_0 in [0, 1), got {excess}, {pi_0}")
    target = excess / (1.0 - pi_0)
    return 1 if target >= 1.0 else smallest_k(alpha, beta, 0.0, target, k_max)


def smallest_measured_k(estimates: Sequence[RateEstimate], pi_0: float, excess: float = 0.05) -> int | None:
    """Smallest measured K from which on the target holds: every measured K' >= K has rate - `pi_0` <= `excess`, so a
    noisy grid whose rate rises again at a larger K never yields an optimistic K. None if the largest K misses it."""
    found = None
    for estimate in sorted(estimates, key=lambda e: e.k, reverse=True):
        if estimate.rate - pi_0 > excess:
            break
        found = estimate.k
    return found


def expected_clip_seconds(t_rest: float, t_policy: float, fallback_rate: float = 0.0, t_fallback: float = 0.0
                          ) -> float:
    """E[t] = t_rest + t_policy + fallback_rate t_fallback (budget Section 2; the throughput configuration passes no
    fallback: its failures are rejected, not re-run).

    Raises:
        ValueError: On negative times or a rate outside [0, 1].
    """
    if min(t_rest, t_policy, t_fallback) < 0.0 or not 0.0 <= fallback_rate <= 1.0:
        raise ValueError(f"times must be >= 0 and the rate in [0, 1], got {t_rest}, {t_policy}, {fallback_rate}, "
                         f"{t_fallback}")
    return t_rest + t_policy + fallback_rate * t_fallback


def accepted_clips_per_gpu_hour(expected_seconds: float, reject_rate: float) -> float:
    """3600 (1 - reject_rate) / E[t]: clips that pass the success check per GPU-hour of a single stream.

    Raises:
        ValueError: If `expected_seconds` is not positive or the rate lies outside [0, 1].
    """
    if not expected_seconds > 0.0 or not 0.0 <= reject_rate <= 1.0:
        raise ValueError(f"need E[t] > 0 and a rate in [0, 1], got {expected_seconds}, {reject_rate}")
    return 3600.0 * (1.0 - reject_rate) / expected_seconds
