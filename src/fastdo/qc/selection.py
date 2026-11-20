"""Conformal selection with Benjamini-Hochberg FDR control for clip acceptance.

Why not D5's threshold. `fastdo.qc.acceptance` guarantees recall of good clips,
`P(accept | error < tau) >= 1 - alpha`. The data product is the accepted set, so the guarantee that
matters is its purity: the false-discovery rate `FDR = E[#bad accepted / max(#accepted, 1)]`. This module
gives `FDR <= q` from the same kind of held-out calibration clips (Jin & Candes 2023, "Selection by
prediction with conformal p-values"; Bates et al. 2023, "Testing for outliers with conformal p-values").

p-values. Test clip j carries the null "clip j is bad" (pose error >= tau). With scores V (higher =
worse; D5's `1 - P(good)` from a rule fitted on its fit split only) and n calibration clips with labels
`bad_i`,

    p_j = (1 + #{i : bad_i, V_i <= V_j}) / (n + 1)                                  (deterministic)
    p_j = (#{i : bad_i, V_i < V_j} + U_j (1 + #{i : bad_i, V_i = V_j})) / (n + 1)    (randomized)

with `U_j = 1 - rng.random()` i.i.d. uniform on (0, 1] (the randomized form stays exact under score
ties; the deterministic one, U = 1, is pointwise larger). Good calibration clips count in n but never in
the numerator (score +inf in Jin & Candes' clipped score). Because the denominator runs over *all*
calibration clips, validity is the joint `P(p_j <= t and clip j bad) <= t`, not super-uniformity
conditional on the clip being bad; the two coincide only when every calibration clip is bad (with good
clips in the pool and an uninformative score, `P(p_j <= t | bad)` is about `t / P(bad)`). The joint form is what
the FDR proof needs: with calibration and test clips exchangeable, Benjamini-Hochberg at level q on
`p_1..p_m` gives `FDR <= q`. tests/qc/test_selection.py checks both forms by Monte Carlo.

Use. The calibration clips must not have been used to fit the rule (`fit_calibration_split` returns the
held-out split of `fit_acceptance_rule`), and the guarantee is marginal over calibration and test draws
exchangeable with them (DexYCB -> internet clips is a distribution shift). One BH per manifest shard of m
clips keeps `FDR <= q` per shard.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from fastdo.qc.acceptance import AcceptanceRule, score_clips
from fastdo.qc.features import QcFeatures

logger = logging.getLogger(__name__)

__all__ = ["SelectionResult", "benjamini_hochberg", "conformal_pvalues", "select_clips"]

# Relative slack on the BH step-up comparison p_(k) <= q k / m: absorbs float rounding of q k / m (e.g.
# 0.05 * 43 / 43 = 0.049999999999999996), far below the gap between distinct p-values a / (n + 1) and
# q k / m for n, m up to ~1e5.
_BH_REL_SLACK = 1e-12


def _check_level(q: float) -> None:
    if not 0.0 < q < 1.0:
        raise ValueError(f"q must be in (0, 1), got {q}")


def _finite_1d(x: np.ndarray, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {a.shape}")
    if not np.isfinite(a).all():
        raise ValueError(f"{name} must be finite")
    return a


def conformal_pvalues(cal_scores: np.ndarray, cal_bad: np.ndarray, test_scores: np.ndarray,
                      rng: np.random.Generator | None = None) -> np.ndarray:
    """Conformal p-values (m,) for the nulls "test clip is bad" (module docstring). Deterministic with
    ties counted as <= when `rng` is None; otherwise ties are broken by `U = 1 - rng.random(m)`."""
    cal = _finite_1d(cal_scores, "cal_scores")
    bad = np.asarray(cal_bad)
    if bad.dtype != np.bool_ or bad.shape != cal.shape:
        raise ValueError(f"cal_bad must be a boolean array of shape {cal.shape}, got {bad.dtype} {bad.shape}")
    test = _finite_1d(test_scores, "test_scores")
    if cal.size == 0:
        raise ValueError("conformal_pvalues needs at least one calibration clip")
    bad_sorted = np.sort(cal[bad])
    n_le = np.searchsorted(bad_sorted, test, side="right").astype(np.float64)
    if rng is None:
        return (1.0 + n_le) / (cal.size + 1)
    n_lt = np.searchsorted(bad_sorted, test, side="left").astype(np.float64)
    u = 1.0 - rng.random(test.size)
    return (n_lt + u * (1.0 + n_le - n_lt)) / (cal.size + 1)


def benjamini_hochberg(pvalues: np.ndarray, q: float) -> np.ndarray:
    """Benjamini-Hochberg step-up at level q: with sorted `p_(1) <= .. <= p_(m)` and the largest k such
    that `p_(k) <= q k / m`, reject (select) every j with `p_j <= p_(k)`; nothing when no k qualifies."""
    _check_level(q)
    p = np.asarray(pvalues, dtype=np.float64)
    if p.ndim != 1:
        raise ValueError(f"pvalues must be 1-D, got shape {p.shape}")
    if np.isnan(p).any() or (p < 0.0).any() or (p > 1.0).any():
        raise ValueError("pvalues must lie in [0, 1]")
    m = p.size
    p_sorted = np.sort(p)
    passing = np.flatnonzero(p_sorted <= q * np.arange(1, m + 1) / m * (1.0 + _BH_REL_SLACK))
    if passing.size == 0:
        return np.zeros(m, dtype=bool)
    return p <= p_sorted[passing[-1]]


def _frozen(x: np.ndarray, dtype: npt.DTypeLike) -> np.ndarray:
    a: np.ndarray = np.array(x, dtype=dtype, copy=True)
    a.setflags(write=False)
    return a


@dataclass(frozen=True, eq=False)   # eq=False: ndarray fields have no scalar equality
class SelectionResult:
    selected: np.ndarray     # (m,) bool: BH rejections of "clip is bad", i.e. the accepted test clips
    pvalues: np.ndarray      # (m,) conformal p-values
    q: float                 # target FDR
    n_calibration: int       # calibration clips (the n in n + 1)
    n_bad_calibration: int   # calibration clips with pose error >= tau (the ones counted)

    def __post_init__(self) -> None:
        selected, pvalues = _frozen(self.selected, bool), _frozen(self.pvalues, np.float64)
        if selected.ndim != 1 or selected.shape != pvalues.shape:
            raise ValueError(f"selected {selected.shape} and pvalues {pvalues.shape} must be equal 1-D shapes")
        if not 0 <= self.n_bad_calibration <= self.n_calibration:
            raise ValueError(f"need 0 <= n_bad_calibration <= n_calibration, got {self.n_bad_calibration}, "
                             f"{self.n_calibration}")
        object.__setattr__(self, "selected", selected)
        object.__setattr__(self, "pvalues", pvalues)


def select_clips(rule: AcceptanceRule, cal_qc: list[QcFeatures], cal_error_m: np.ndarray,
                 test_qc: list[QcFeatures], q: float = 0.1, tau: float = 0.02,
                 rng: np.random.Generator | None = None) -> SelectionResult:
    """Accept the test clips selected by BH at level q on conformal p-values of the already-fitted
    `rule`'s scores (never refitted). `cal_qc`/`cal_error_m` must be labelled clips the rule was not fitted
    on (e.g. the `fit_calibration_split` held-out split); `FDR <= q` holds per call, so per shard."""
    _check_level(q)
    if not (np.isfinite(tau) and tau > 0.0):
        raise ValueError(f"tau must be a positive pose-error bound (m), got {tau}")
    err = _finite_1d(cal_error_m, "cal_error_m")
    if err.size != len(cal_qc):
        raise ValueError(f"{len(cal_qc)} calibration QC feature sets but {err.size} pose errors")
    if tau != rule.tau:
        logger.warning("select_clips: tau=%g differs from the rule's tau=%g; FDR control still holds at "
                       "tau=%g, but the score was trained for the rule's tau (power may drop)", tau, rule.tau, tau)
    cal_bad = err >= tau
    pvalues = conformal_pvalues(score_clips(rule, cal_qc), cal_bad, score_clips(rule, test_qc), rng=rng)
    selected = benjamini_hochberg(pvalues, q)
    n, n_bad = int(err.size), int(cal_bad.sum())
    if n_bad == 0:
        logger.warning("select_clips: the calibration set has no bad clips at tau=%g m, so every p-value is at "
                       "most 1/(n+1) and BH may accept every test clip; pass labelled calibration clips of both "
                       "kinds (e.g. the fit_calibration_split held-out split), not the good-only set the recall "
                       "threshold uses", tau)
    if (n + 1) * q < 1.0 and rng is None:
        logger.warning("select_clips: n=%d calibration clips < 1/q - 1 at q=%g; the smallest deterministic "
                       "p-value 1/(n+1) exceeds q, so nothing can be selected", n, q)
    elif (n + 1) * q < 1.0:
        logger.warning("select_clips: n=%d calibration clips < 1/q - 1 at q=%g; only randomized p-values "
                       "U/(n+1) can fall below q, so selection is rare and power is low", n, q)
    logger.info("select_clips: %d of %d test clips selected at q=%g (calibration n=%d, %d bad at tau=%g m)",
                int(selected.sum()), selected.size, q, n, n_bad, tau)
    return SelectionResult(selected=selected, pvalues=pvalues, q=q, n_calibration=n, n_bad_calibration=n_bad)
