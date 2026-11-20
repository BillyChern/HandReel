"""Reporting quantities (design 9b amendments b-c; budget Section 2 amendment)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from fastdo.policy.fallback_law import fallback_rate, smallest_k
from fastdo.policy.report import (
    Z_95,
    accepted_clips_per_gpu_hour,
    expected_clip_seconds,
    rate_estimate,
    smallest_k_excess,
    smallest_measured_k,
    wilson_interval,
)


@pytest.mark.parametrize(("failures", "n"), [(0, 10), (5, 10), (1, 3), (3, 300), (300, 300)])
def test_wilson_interval_solves_the_score_equation(failures: int, n: int) -> None:
    """Each finite bound p solves (p_hat - p)^2 = z^2 p (1 - p) / n (Wilson's definition), the interval contains
    p_hat, and the textbook 5/10 case is [0.2366, 0.7634]."""
    low, high = wilson_interval(failures, n)
    p_hat = failures / n
    assert 0.0 <= low <= p_hat <= high <= 1.0
    for bound in (low, high):
        if 0.0 < bound < 1.0 or bound == p_hat:
            assert (p_hat - bound) ** 2 == pytest.approx(Z_95**2 * bound * (1.0 - bound) / n, abs=1e-12)
    if (failures, n) == (5, 10):
        assert (round(low, 4), round(high, 4)) == (0.2366, 0.7634)
    if failures == 0:
        assert low == 0.0 and high == pytest.approx(Z_95**2 / (n + Z_95**2), rel=1e-12)
    with pytest.raises(ValueError):
        wilson_interval(4, 3)


def test_wilson_interval_covers_at_nominal_rate() -> None:
    """Monte Carlo: at p = 0.02 and n = 300 the 95 % Wilson interval covers p for 93-97.5 % of 4000 draws."""
    rng = np.random.default_rng(0)
    draws = rng.binomial(300, 0.02, size=4000)
    covered = np.mean([lo <= 0.02 <= hi for lo, hi in (wilson_interval(int(d), 300) for d in draws)])
    assert 0.93 <= covered <= 0.975, covered


def test_rate_estimate_counts_failures() -> None:
    est = rate_estimate("smc", 16, [True, False, False, True])
    assert (est.failures, est.n_clips, est.rate) == (2, 4, 0.5)
    row = est.as_dict()
    assert row["method"] == "smc" and row["wilson95_low"] < 0.5 < row["wilson95_high"]
    assert rate_estimate("smc_no_floor", 4, [False]).rate == 0.0
    with pytest.raises(ValueError):
        rate_estimate("mpc", 4, [True])
    with pytest.raises(ValueError):
        rate_estimate("best_of_k", 4, [])


def test_smallest_k_excess_matches_brute_force_and_the_design_example() -> None:
    """At (alpha, beta, pi_0) = (0.6, 1.4, 0.001) the smallest K with f_BoK(K) - pi_0 <= 5 % is 179. Design 9b amendment
    (c) quotes K = 185 for this target, which is the smallest K with f_BoK(K) <= 5 % itself (pinned below). A
    brute-force scan over K agrees for other floors; the measured-grid variant picks the first qualifying K."""
    assert smallest_k_excess(0.6, 1.4, 0.001, 0.05) == 179
    assert smallest_k(0.6, 1.4, 0.001, 0.05) == 185
    for pi_0 in (0.0, 0.02, 0.1):
        k = smallest_k_excess(0.6, 1.4, pi_0, 0.05)
        assert k is not None
        assert fallback_rate(0.6, 1.4, pi_0, k) - pi_0 <= 0.05 < fallback_rate(0.6, 1.4, pi_0, k - 1) - pi_0
    assert smallest_k_excess(0.6, 1.4, 0.0, 0.003) is None                 # K = 19,560 > k_max 4096
    rows = [rate_estimate("best_of_k", k, [True] * f + [False] * (100 - f)) for k, f in ((1, 70), (16, 22), (64, 6))]
    assert smallest_measured_k(rows, pi_0=0.02, excess=0.05) == 64
    assert smallest_measured_k(rows, pi_0=0.02, excess=0.01) is None
    # Non-monotone measured rates: K = 16 meets 5 % but K = 64 misses it again, so no K holds from 16 on (review 2).
    noisy = [rate_estimate("best_of_k", k, [True] * f + [False] * (100 - f)) for k, f in ((1, 70), (16, 5), (64, 9))]
    assert smallest_measured_k(noisy, pi_0=0.02, excess=0.05) is None
    assert smallest_measured_k([*noisy, rate_estimate("best_of_k", 256, [False] * 100)], pi_0=0.02, excess=0.05) == 256


def test_throughput_law_of_budget_section_2() -> None:
    """E[t] = t_rest + t_pi + f t_MPC, accepted clips per GPU-hour = 3600 (1 - reject) / E[t]; at f = 5 % and a 900 s
    MPC the fallback alone adds 45 s per clip (the Section 2 cap)."""
    assert expected_clip_seconds(18.0, 1.5) == 19.5
    assert expected_clip_seconds(18.0, 1.5, 0.05, 900.0) == pytest.approx(64.5)
    assert accepted_clips_per_gpu_hour(19.5, 0.1) == pytest.approx(3600.0 * 0.9 / 19.5)
    assert math.isclose(accepted_clips_per_gpu_hour(2.0, 0.0), 1800.0)
    with pytest.raises(ValueError):
        expected_clip_seconds(1.0, 1.0, 1.5, 1.0)
    with pytest.raises(ValueError):
        accepted_clips_per_gpu_hour(0.0, 0.1)
