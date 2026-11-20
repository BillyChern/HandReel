"""Fallback-rate law of best-of-K rollouts.

Let p_c be the probability that one stochastic policy rollout (action noise plus reference noise) tracks clip c. Over
clips p is zero-inflated Beta: p = 0 with probability pi_0 (a reference the policy can never track), otherwise
p ~ Beta(alpha, beta). A clip falls back when none of K independent rollouts succeeds, so the fallback rate is

    f(K) = E[(1 - p)^K] = pi_0 + (1 - pi_0) B(alpha, beta + K) / B(alpha, beta)
         ~ pi_0 + (1 - pi_0) Gamma(alpha + beta) / Gamma(beta) K^-alpha        (K -> infinity),

a power law whose exponent alpha is the density of hard clips near p = 0, above the floor pi_0. `fallback_rate`
evaluates it exactly in log space (`scipy.special.betaln`) and `smallest_k` inverts it. `zero_inflated_beta_binomial_mle`
fits (alpha, beta, pi_0) to per-clip success counts s_c out of n_c rollouts, whose likelihood is

    P(s | n) = [s = 0] pi_0 + (1 - pi_0) C(n, s) B(s + alpha, n - s + beta) / B(alpha, beta),

by maximizing the mean log-likelihood over (log alpha, log beta, logit pi_0) with L-BFGS-B from a grid of starts: the
likelihood is not concave in these coordinates, and pi_0 trades off against a small alpha (both put mass on s = 0).
"""

from __future__ import annotations

import itertools
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize
from scipy.special import betaln, expit, gammaln, log_expit

logger = logging.getLogger(__name__)

__all__ = ["FallbackLawFit", "bootstrap_fallback_law", "fallback_rate", "smallest_k",
           "zero_inflated_beta_binomial_loglik", "zero_inflated_beta_binomial_mle"]

_LOG_BOUND = 10.0            # |log alpha|, |log beta| <= 10
_LOGIT_BOUND = 30.0          # |logit pi_0| <= 30, pi_0 in [9.4e-14, 1 - 9.4e-14]
_STARTS = tuple(itertools.product((0.3, 1.0, 3.0), (0.5, 2.0, 8.0), (1e-3, 0.05, 0.3)))   # (alpha, beta, pi_0)


def _check_parameters(alpha: float, beta: float, pi_0: float) -> None:
    if not (math.isfinite(alpha) and alpha > 0.0 and math.isfinite(beta) and beta > 0.0):
        raise ValueError(f"alpha and beta must be finite and positive, got alpha={alpha}, beta={beta}")
    if not 0.0 <= pi_0 <= 1.0:
        raise ValueError(f"pi_0 must lie in [0, 1], got {pi_0}")


def fallback_rate(alpha: float, beta: float, pi_0: float, k: int) -> float:
    """Best-of-K fallback rate f(K) = pi_0 + (1 - pi_0) exp(betaln(alpha, beta + K) - betaln(alpha, beta)), K >= 0.

    Raises:
        ValueError: On invalid parameters or a negative or non-integer K.
    """
    _check_parameters(alpha, beta, pi_0)
    if isinstance(k, bool) or int(k) != k or k < 0:
        raise ValueError(f"k must be a non-negative integer, got {k}")
    if k == 0:
        return 1.0
    return float(pi_0 + (1.0 - pi_0) * math.exp(betaln(alpha, beta + k) - betaln(alpha, beta)))


def smallest_k(alpha: float, beta: float, pi_0: float, target: float, k_max: int = 4096) -> int | None:
    """Smallest K in [1, k_max] with f(K) <= target, by bisection (f decreases strictly in K).

    None when pi_0 >= target (f(K) > pi_0 for every K: the fallback floor is a yield cost that no K buys) or when
    f(k_max) > target.

    Raises:
        ValueError: On invalid parameters, a target outside (0, 1) or k_max < 1.
    """
    _check_parameters(alpha, beta, pi_0)
    if not 0.0 < target < 1.0:
        raise ValueError(f"target must lie in (0, 1), got {target}")
    if k_max < 1:
        raise ValueError(f"k_max must be >= 1, got {k_max}")
    if pi_0 >= target or fallback_rate(alpha, beta, pi_0, k_max) > target:
        return None
    lo, hi = 0, k_max                     # invariant: f(lo) > target >= f(hi); f(0) = 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if fallback_rate(alpha, beta, pi_0, mid) <= target:
            hi = mid
        else:
            lo = mid
    return hi


def _counts(successes: ArrayLike, trials: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    s, n = np.asarray(successes, dtype=np.float64), np.asarray(trials, dtype=np.float64)
    if s.ndim != 1 or s.shape != n.shape or s.size == 0:
        raise ValueError(f"successes and trials must be non-empty (C,) arrays of one shape, got {s.shape}, {n.shape}")
    if not (np.all(np.isfinite(s)) and np.all(np.isfinite(n)) and np.all(s == np.round(s))
            and np.all(n == np.round(n))):
        raise ValueError("successes and trials must be finite integers")
    if np.any(n < 1) or np.any(s < 0) or np.any(s > n):
        raise ValueError("every clip needs trials >= 1 and 0 <= successes <= trials")
    return s, n


def _clip_log_likelihood(s: np.ndarray, n: np.ndarray, alpha: float, beta: float, log_pi_0: float,
                         log_rest: float) -> np.ndarray:
    """log P(s_c | n_c) per clip (module docstring), given log pi_0 and log(1 - pi_0)."""
    log_bb = (gammaln(n + 1.0) - gammaln(s + 1.0) - gammaln(n - s + 1.0) + betaln(s + alpha, n - s + beta)
              - betaln(alpha, beta))
    return np.where(s == 0, np.logaddexp(log_pi_0, log_rest + log_bb), log_rest + log_bb)


def zero_inflated_beta_binomial_loglik(successes: ArrayLike, trials: ArrayLike, alpha: float, beta: float,
                                       pi_0: float) -> float:
    """Total log-likelihood sum_c log P(s_c | n_c) of per-clip success counts (module docstring)."""
    _check_parameters(alpha, beta, pi_0)
    s, n = _counts(successes, trials)
    log_pi_0 = math.log(pi_0) if pi_0 > 0.0 else -math.inf
    log_rest = math.log1p(-pi_0) if pi_0 < 1.0 else -math.inf
    return float(_clip_log_likelihood(s, n, alpha, beta, log_pi_0, log_rest).sum())


def _fit(s: np.ndarray, n: np.ndarray, weights: np.ndarray,
         starts: Sequence[tuple[float, float, float]]) -> tuple[float, float, float]:
    """L-BFGS-B on the weighted mean negative log-likelihood over (log alpha, log beta, logit pi_0) from every start
    (clipped into the bounds); the best finite optimum wins."""
    total = float(weights.sum())

    def objective(theta: np.ndarray) -> float:
        log_alpha, log_beta, logit = (float(x) for x in theta)
        terms = _clip_log_likelihood(s, n, math.exp(log_alpha), math.exp(log_beta), float(log_expit(logit)),
                                     float(log_expit(-logit)))
        return -float(terms @ weights) / total

    bounds = [(-_LOG_BOUND, _LOG_BOUND), (-_LOG_BOUND, _LOG_BOUND), (-_LOGIT_BOUND, _LOGIT_BOUND)]
    best_x: np.ndarray | None = None
    best_f = math.inf
    for alpha_0, beta_0, pi_0_start in starts:
        p = min(max(pi_0_start, 1e-300), 1.0 - 1e-16)
        start = np.clip([math.log(alpha_0), math.log(beta_0), math.log(p) - math.log1p(-p)],
                        [b[0] for b in bounds], [b[1] for b in bounds])
        result = minimize(objective, start, method="L-BFGS-B", bounds=bounds)
        if math.isfinite(float(result.fun)) and float(result.fun) < best_f:
            best_x, best_f = np.asarray(result.x, dtype=np.float64), float(result.fun)
    if best_x is None:
        raise RuntimeError("zero_inflated_beta_binomial_mle: no start reached a finite likelihood")
    logger.debug("zero-inflated beta-binomial fit over %d clips: theta %s (mean nll %.6f)", int(total),
                 np.round(best_x, 4).tolist(), best_f)
    return math.exp(best_x[0]), math.exp(best_x[1]), float(expit(best_x[2]))


def _distinct(s: np.ndarray, n: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The distinct (successes, trials) pairs and how many clips share each: the likelihood's sufficient statistic."""
    pairs, counts = np.unique(np.stack([s, n], axis=1), axis=0, return_counts=True)
    return pairs[:, 0], pairs[:, 1], counts.astype(np.float64)


def zero_inflated_beta_binomial_mle(successes: ArrayLike, trials: ArrayLike,
                                    starts: Sequence[tuple[float, float, float]] | None = None
                                    ) -> tuple[float, float, float]:
    """(alpha, beta, pi_0) maximizing the zero-inflated beta-binomial likelihood of per-clip success counts.

    L-BFGS-B on the mean negative log-likelihood over theta = (log alpha, log beta, logit pi_0), within
    |log alpha|, |log beta| <= 10 and |logit pi_0| <= 30, from each start (alpha, beta, pi_0); by default the 27 starts
    of the grid alpha in {0.3, 1, 3}, beta in {0.5, 2, 8}, pi_0 in {1e-3, 0.05, 0.3}; the best optimum wins. Without a
    zero-success clip pi_0 goes to its lower bound (~1e-13).

    Raises:
        ValueError: On malformed counts.
        RuntimeError: If no start reaches a finite likelihood.
    """
    s, n, weights = _distinct(*_counts(successes, trials))
    return _fit(s, n, weights, _STARTS if starts is None else starts)


@dataclass(frozen=True)
class FallbackLawFit:
    """The fitted (alpha, beta, pi_0) and `samples (B, 3)`, its fits on B bootstrap resamples of the clips."""

    alpha: float
    beta: float
    pi_0: float
    samples: np.ndarray
    n_clips: int

    @property
    def at_parameter_bound(self) -> bool:
        """Whether alpha or beta sits at the optimizer's bound e^(+-10). The likelihood is flat there (for example when
        every clip has the same success fraction, so no overdispersion identifies them): the fit is not an estimate."""
        return max(abs(math.log(self.alpha)), abs(math.log(self.beta))) >= _LOG_BOUND - 1e-6

    def interval(self, level: float = 0.95) -> np.ndarray:
        """`(3, 2)` percentile bootstrap intervals of (alpha, beta, pi_0) at `level`."""
        if not 0.0 < level < 1.0:
            raise ValueError(f"level must lie in (0, 1), got {level}")
        return np.quantile(self.samples, [(1.0 - level) / 2.0, (1.0 + level) / 2.0], axis=0).T


def bootstrap_fallback_law(successes: ArrayLike, trials: ArrayLike, n_boot: int = 1000, seed: int = 0
                           ) -> FallbackLawFit:
    """The MLE and its clip bootstrap: `n_boot` resamples of the clips with replacement (a multinomial draw of the
    distinct (s, n) pairs, which is the same distribution), each refit from the full-data fit and from it with pi_0 set
    to 1e-3 and 0.3 (the ridge along which pi_0 trades off against a small alpha).

    Raises:
        ValueError: On malformed counts or `n_boot < 1`.
    """
    if n_boot < 1:
        raise ValueError(f"n_boot must be >= 1, got {n_boot}")
    s, n, weights = _distinct(*_counts(successes, trials))
    alpha, beta, pi_0 = _fit(s, n, weights, _STARTS)
    starts = ((alpha, beta, pi_0), (alpha, beta, 1e-3), (alpha, beta, 0.3))
    rng = np.random.default_rng(seed)
    n_clips = int(weights.sum())
    samples = np.empty((n_boot, 3))
    for b in range(n_boot):
        draw = rng.multinomial(n_clips, weights / n_clips).astype(np.float64)
        keep = draw > 0
        samples[b] = _fit(s[keep], n[keep], draw[keep], starts)
    fit = FallbackLawFit(alpha=alpha, beta=beta, pi_0=pi_0, samples=samples, n_clips=n_clips)
    low, high = fit.interval(0.95)[2]                  # pi_0 is printed only with its interval
    logger.info("fallback law over %d clips: alpha %.3f, beta %.3f, pi_0 %.4f [%.4f, %.4f] (bootstrap 95 %%, %d refits)",
                n_clips, alpha, beta, pi_0, low, high, n_boot)
    return fit
