import math

import numpy as np
import pytest

from fastdo.qc.conformal import conformal_rank, split_conformal_threshold

# 20 distinct scores, deliberately unsorted; sorted they are 0.01, 0.02, ..., 0.20 (i-th smallest = i/100).
_SCORES_20 = np.array([0.13, 0.02, 0.19, 0.07, 0.11, 0.20, 0.01, 0.16, 0.05, 0.09,
                       0.14, 0.03, 0.18, 0.08, 0.12, 0.06, 0.17, 0.04, 0.15, 0.10])


def test_split_conformal_threshold_matches_formula() -> None:
    # alpha = 0.1: ceil((20+1)(1-0.1)) = ceil(18.9) = 19 -> 19th smallest = 0.19
    assert split_conformal_threshold(_SCORES_20, 0.1) == 0.19
    # alpha = 0.2: ceil(21 * 0.8) = ceil(16.8) = 17 -> 0.17
    assert split_conformal_threshold(_SCORES_20, 0.2) == 0.17
    # alpha = 0.5: ceil(21 * 0.5) = ceil(10.5) = 11 -> 0.11
    assert split_conformal_threshold(_SCORES_20, 0.5) == 0.11
    assert conformal_rank(20, 0.1) == 19


def test_conformal_rank_is_exact_at_integer_products() -> None:
    # (n+1)(1-alpha) is an exact integer here; floating-point error must not bump the rank by one.
    assert conformal_rank(99, 0.1) == 90
    assert conformal_rank(9, 0.1) == 9
    assert conformal_rank(19, 0.05) == 19
    assert conformal_rank(299, 0.3) == 210


def test_split_conformal_threshold_clips_to_max_when_n_too_small() -> None:
    scores = np.array([0.3, 0.1, 0.2, 0.5, 0.4])
    assert conformal_rank(5, 0.1) == 6                # ceil(6 * 0.9) = 6 > n
    assert split_conformal_threshold(scores, 0.1) == 0.5


def test_split_conformal_threshold_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        split_conformal_threshold(np.array([]), 0.1)
    with pytest.raises(ValueError):
        split_conformal_threshold(np.array([0.1, math.nan]), 0.1)
    for alpha in (0.0, 1.0, -0.1):
        with pytest.raises(ValueError):
            split_conformal_threshold(_SCORES_20, alpha)


def test_split_conformal_coverage_guarantee() -> None:
    """500 Monte Carlo (n_cal=100, n_test=1) splits of one exchangeable (i.i.d.) score distribution."""
    alpha, n_splits, n_cal = 0.1, 500, 100
    rng = np.random.default_rng(20260913)
    misses = 0
    for _ in range(n_splits):
        scores = rng.gamma(shape=2.0, scale=0.3, size=n_cal + 1)
        threshold = split_conformal_threshold(scores[:n_cal], alpha)
        misses += int(scores[n_cal] > threshold)
    miscoverage = misses / n_splits
    band = 3.0 * math.sqrt(alpha * (1.0 - alpha) / n_splits)
    assert miscoverage <= alpha + band
    # Non-vacuous: with continuous scores the exact miscoverage is 1 - k/(n+1) = 10/101, so a threshold
    # that accepted (almost) everything would fall below this lower band edge.
    assert miscoverage >= 1.0 - conformal_rank(n_cal, alpha) / (n_cal + 1) - band
