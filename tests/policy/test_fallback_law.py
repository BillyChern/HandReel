"""The fallback-rate law of design Section 9b.

Per-clip success probability of one stochastic rollout: p = 0 with probability pi_0, else p ~ Beta(alpha, beta). The
best-of-K fallback rate is f(K) = E[(1 - p)^K] = pi_0 + (1 - pi_0) B(alpha, beta + K) / B(alpha, beta).
"""

from __future__ import annotations

import itertools
import logging
import math

import numpy as np
import pytest
from scipy import integrate, stats
from scipy.special import expit, gammaln, logit

from fastdo.policy.fallback_law import (
    bootstrap_fallback_law,
    fallback_rate,
    smallest_k,
    zero_inflated_beta_binomial_loglik,
    zero_inflated_beta_binomial_mle,
)

logger = logging.getLogger(__name__)

ALPHA, BETA, PI_0 = 0.6, 1.4, 0.02


def _clip_probabilities(n_clips: int, rng: np.random.Generator) -> np.ndarray:
    """p per clip: 0 with probability PI_0 (a reference the policy can never track), else Beta(ALPHA, BETA)."""
    never = rng.random(n_clips) < PI_0
    return np.where(never, 0.0, rng.beta(ALPHA, BETA, size=n_clips))


def test_fallback_rate_matches_monte_carlo() -> None:
    """400k clips, K rollouts per clip simulated as Binomial(K, p) successes; a clip falls back when none succeeds.
    The design's exact values are 0.7060 / 0.2204 / 0.0595 at K = 1 / 16 / 256."""
    rng = np.random.default_rng(0)
    p = _clip_probabilities(400_000, rng)
    for k, design in ((1, 0.7060), (16, 0.2204), (256, 0.0595)):
        exact = fallback_rate(ALPHA, BETA, PI_0, k)
        monte_carlo = float(np.mean(rng.binomial(k, p) == 0))
        assert abs(exact - monte_carlo) < 0.002, (k, exact, monte_carlo)
        assert exact == pytest.approx(design, abs=5e-5)


def test_fallback_rate_is_the_beta_integral_and_a_power_law() -> None:
    """Independent of betaln: f(K) = pi_0 + (1 - pi_0) int (1 - p)^K Beta(p; alpha, beta) dp by quadrature, f(0) = 1,
    and f(K) - pi_0 ~ (1 - pi_0) Gamma(alpha + beta) / Gamma(beta) K^-alpha as K -> infinity."""
    for k in (0, 1, 5, 64):
        integral, _ = integrate.quad(lambda p, k=k: (1.0 - p) ** k * stats.beta.pdf(p, ALPHA, BETA), 0.0, 1.0,
                                     limit=200)
        assert fallback_rate(ALPHA, BETA, PI_0, k) == pytest.approx(PI_0 + (1.0 - PI_0) * integral, rel=1e-6)
    assert fallback_rate(ALPHA, BETA, PI_0, 0) == 1.0
    k = 10**7
    asymptote = (1.0 - PI_0) * math.exp(gammaln(ALPHA + BETA) - gammaln(BETA)) * k**-ALPHA
    assert fallback_rate(ALPHA, BETA, PI_0, k) - PI_0 == pytest.approx(asymptote, rel=1e-5)
    with pytest.raises(ValueError):
        fallback_rate(ALPHA, BETA, PI_0, -1)
    with pytest.raises(ValueError):
        fallback_rate(0.0, BETA, PI_0, 4)


def test_zero_inflated_beta_binomial_loglik_matches_quadrature() -> None:
    """P(s | n) = pi_0 [s = 0] + (1 - pi_0) int C(n, s) p^s (1 - p)^(n - s) Beta(p) dp, summed in log over clips."""
    successes, trials = np.array([0, 3, 8, 1]), np.array([8, 8, 8, 5])
    expected = 0.0
    for s, n in zip(successes, trials, strict=True):
        mixture, _ = integrate.quad(lambda p, s=s, n=n: stats.binom.pmf(s, n, p) * stats.beta.pdf(p, ALPHA, BETA),
                                    0.0, 1.0, limit=200)
        expected += math.log(PI_0 * (s == 0) + (1.0 - PI_0) * mixture)
    got = zero_inflated_beta_binomial_loglik(successes, trials, ALPHA, BETA, PI_0)
    assert got == pytest.approx(expected, rel=1e-7)


def _counts_300_by_8(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """300 clips x 8 rollouts from (ALPHA, BETA, PI_0): Beta draws, then the never-trackable mask, then the counts."""
    rng = np.random.default_rng(seed)
    p = rng.beta(ALPHA, BETA, size=300)
    p = np.where(rng.random(300) < PI_0, 0.0, p)
    trials = np.full(300, 8)
    return rng.binomial(trials, p), trials


def test_bootstrap_interval_covers_the_truth_over_seeds() -> None:
    """300 clips x 8 rollouts from (0.6, 1.4, 0.02) on 40 seeds: the 90 % percentile bootstrap interval (100 clip
    resamples) of each of alpha, beta and pi_0 covers the generating value on at least 70 % of the seeds.

    This replaces a single-draw recovery test: at this sample size the fit's sd is about 0.18 (alpha), 0.26 (beta) and
    0.065 (pi_0), so fixed bounds of 0.15 / 0.5 / 0.01 held jointly for only 5-8 % of draws (design 9b amendment b).
    The bootstrap's warm-started refits reach the 27-start optimum (scratch check: log-likelihood within 7e-6)."""
    truth = np.array([ALPHA, BETA, PI_0])
    covered, widths = np.zeros(3), np.zeros(3)
    for seed in range(40):
        successes, trials = _counts_300_by_8(seed)
        fit = bootstrap_fallback_law(successes, trials, n_boot=100, seed=seed)
        low, high = fit.interval(0.90).T
        assert fit.samples.shape == (100, 3) and bool(np.all(low <= high))
        assert (fit.alpha, fit.beta, fit.pi_0) == zero_inflated_beta_binomial_mle(successes, trials)
        covered += (low <= truth) & (truth <= high)
        widths += (high - low) / 40.0
    coverage = covered / 40.0
    logger.info("90 %% bootstrap coverage over 40 seeds (alpha, beta, pi_0): %s; mean widths %s",
                coverage.round(3).tolist(), widths.round(3).tolist())
    assert bool(np.all(coverage >= 0.7)), coverage


def test_beta_binomial_mle_is_the_likelihood_maximum() -> None:
    """On the seed-0 draw the fit scores at least as high as the generating parameters and every point of a 21^3 grid
    over (log alpha, log beta, logit pi_0), and no coordinate step of 1e-3 from it improves the likelihood."""
    successes, trials = _counts_300_by_8(0)
    fit = zero_inflated_beta_binomial_mle(successes, trials)

    def loglik(theta: np.ndarray) -> float:
        return zero_inflated_beta_binomial_loglik(successes, trials, math.exp(theta[0]), math.exp(theta[1]),
                                                  float(expit(theta[2])))

    best = zero_inflated_beta_binomial_loglik(successes, trials, *fit)
    assert best >= zero_inflated_beta_binomial_loglik(successes, trials, ALPHA, BETA, PI_0)
    grid = itertools.product(np.linspace(-2.0, 2.0, 21), np.linspace(-2.0, 3.0, 21), np.linspace(-10.0, 2.0, 21))
    assert best >= max(loglik(np.array(point)) for point in grid)
    theta = np.array([math.log(fit[0]), math.log(fit[1]), float(logit(fit[2]))])
    for i, delta in itertools.product(range(3), (-1e-3, 1e-3)):
        moved = theta.copy()
        moved[i] += delta
        assert best >= loglik(moved) - 1e-6, (i, delta)


def test_beta_binomial_mle_validates_inputs() -> None:
    with pytest.raises(ValueError):
        zero_inflated_beta_binomial_mle(np.array([9]), np.array([8]))
    with pytest.raises(ValueError):
        zero_inflated_beta_binomial_mle(np.array([1, 2]), np.array([8]))
    with pytest.raises(ValueError):
        zero_inflated_beta_binomial_mle(np.array([], dtype=int), np.array([], dtype=int))


def test_smallest_k_none_when_pi0_exceeds_target() -> None:
    """f(K) > pi_0 for every K, so a target below pi_0 is a yield cost that no K buys."""
    assert smallest_k(ALPHA, BETA, 0.01, target=0.003) is None
    assert smallest_k(ALPHA, BETA, 0.003, target=0.003) is None


def test_smallest_k_is_the_first_k_meeting_the_target() -> None:
    """With alpha = 0.6 the power law is slow: f(K) - pi_0 ~ 1.13 K^-0.6, so 0.3% is out of reach of k_max = 4096 even
    at pi_0 = 0 (f(4096) = 0.77%; the smallest K is 19560), while 5% at pi_0 = 0.001 takes K = 185."""
    assert smallest_k(ALPHA, BETA, 0.0, target=0.003) is None
    k = smallest_k(ALPHA, BETA, 0.001, target=0.05)
    assert k is not None and 100 < k < 4096
    assert fallback_rate(ALPHA, BETA, 0.001, k) <= 0.05 < fallback_rate(ALPHA, BETA, 0.001, k - 1)
    assert smallest_k(ALPHA, BETA, 0.001, target=0.05, k_max=k - 1) is None
    assert smallest_k(ALPHA, BETA, 0.001, target=0.05, k_max=k) == k
    assert smallest_k(ALPHA, BETA, 0.0, target=0.99) == 1                 # f(1) = beta / (alpha + beta) = 0.7
