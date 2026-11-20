"""Split-conformal threshold with the finite-sample guarantee of Angelopoulos & Bates (2021).

Given calibration nonconformity scores `s_1..s_n` and a test score `s_{n+1}` that are exchangeable,
the threshold `q = s_(k)`, the `k = ceil((n+1)(1-alpha))`-th smallest calibration score, satisfies

    1 - alpha  <=  P(s_{n+1} <= q)  <=  1 - alpha + 1/(n+1)

(the upper bound when scores have no ties), marginally over the calibration draw and the test point,
with no assumption on the model that produced the scores. When `k > n` the guarantee is unattainable
for this `n` and `alpha` (the exact answer is `q = +inf`); this module clips to the largest score, as
the acceptance-rule interface specifies, and logs a warning that coverage is then not guaranteed.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["ConformalCalibration", "conformal_rank", "split_conformal_threshold"]


@dataclass(frozen=True)
class ConformalCalibration:
    threshold: float     # accept iff the test score is <= threshold
    alpha: float         # target miscoverage
    n_calibration: int   # number of calibration scores the threshold was taken from


def _check_alpha(alpha: float) -> None:
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")


def conformal_rank(n: int, alpha: float) -> int:
    """`ceil((n+1)(1-alpha))`, computed in exact rational arithmetic on `alpha`'s decimal repr so that an
    exactly-integer product (e.g. n=99, alpha=0.1 -> 90) is never bumped by floating-point error."""
    _check_alpha(alpha)
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    return math.ceil((n + 1) * (1 - Fraction(repr(float(alpha)))))


def split_conformal_threshold(scores: np.ndarray, alpha: float) -> float:
    """The `ceil((n+1)(1-alpha))`-th smallest of `scores` (n = len), clipped to the max when that rank
    exceeds `n`."""
    s = np.asarray(scores, dtype=np.float64).ravel()
    if s.size == 0:
        raise ValueError("split_conformal_threshold needs at least one calibration score")
    if not np.isfinite(s).all():
        raise ValueError("calibration scores must be finite")
    n = int(s.size)
    k = conformal_rank(n, alpha)
    if k > n:
        logger.warning("split_conformal_threshold: rank %d > n=%d at alpha=%g; clipping to the max score, "
                       "coverage is NOT guaranteed (need n >= %d)", k, n, alpha,
                       math.ceil((1 - Fraction(repr(float(alpha)))) / Fraction(repr(float(alpha)))))
        return float(s.max())
    return float(np.partition(s, k - 1)[k - 1])
