"""Clip acceptance: a logistic score on QC features, thresholded by class-conditional split conformal.

Model: a plain numpy Newton/IRLS logistic regression maps the
five standardized `QcFeatures` to `P(good)`, `good = true_pose_error < tau`; the nonconformity score is
`1 - P(good)`; a clip is accepted iff its score is <= the split-conformal threshold.

Guarantee. `fit_acceptance_rule` splits the labelled clips (stratified by label, seeded) into a fit set
(standardization statistics + logistic weights) and a calibration set, and takes the threshold from the
calibration set's *good* clips only (class-conditional split conformal; Vovk 2012, Sadinle, Lei &
Wasserman 2019, Angelopoulos & Bates 2021). Because every model parameter comes from the fit set, a new
good clip's score is exchangeable with the calibration good clips' scores, so

    P(accept | true_pose_error < tau) >= 1 - alpha

in finite samples, whenever `ceil((n+1)(1-alpha)) <= n` for `n = n_calibration` good calibration clips
and the deployment clips are exchangeable with the calibration clips (DexYCB -> internet clips is a
distribution shift; see theory landscape Section 5). Bad clips are rejected only as well as the
logistic score separates them -- that rate is measured and logged on the calibration split, not
guaranteed. Calibrating on all clips instead would only guarantee accepting 1 - alpha of *all* clips,
whatever their quality, which is not an acceptance criterion.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.special import expit

from fastdo.bench.metrics import retarget_errors
from fastdo.qc.conformal import ConformalCalibration, split_conformal_threshold
from fastdo.qc.features import QcFeatures

logger = logging.getLogger(__name__)

__all__ = [
    "AcceptanceRule",
    "accept",
    "calibrate",
    "fit_acceptance_rule",
    "fit_calibration_split",
    "fit_logistic_irls",
    "pose_error_m",
    "score_clips",
]

_N_FEATURES = 5


def _frozen_array(x: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    a = np.array(x, dtype=np.float64, copy=True)
    if a.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {a.shape}")
    if not np.isfinite(a).all():
        raise ValueError(f"{name} must be finite")
    a.setflags(write=False)
    return a


@dataclass(frozen=True, eq=False)   # eq=False: ndarray fields have no scalar equality
class AcceptanceRule:
    calibration: ConformalCalibration
    logistic_weights: np.ndarray   # (6,) [bias, w_1..w_5] on standardized features
    feature_mean: np.ndarray       # (5,) fit-set feature means
    feature_std: np.ndarray        # (5,) fit-set feature stds (1.0 where a feature was constant)
    tau: float = 0.02              # "good clip" pose-error bound (m) the coverage statement refers to

    def __post_init__(self) -> None:
        object.__setattr__(self, "logistic_weights",
                           _frozen_array(self.logistic_weights, (_N_FEATURES + 1,), "logistic_weights"))
        object.__setattr__(self, "feature_mean", _frozen_array(self.feature_mean, (_N_FEATURES,), "feature_mean"))
        std = _frozen_array(self.feature_std, (_N_FEATURES,), "feature_std")
        if (std <= 0.0).any():
            raise ValueError("feature_std must be positive")
        object.__setattr__(self, "feature_std", std)


def fit_logistic_irls(X: np.ndarray, y: np.ndarray, l2: float = 1e-3, max_iter: int = 100,
                      tol: float = 1e-10) -> np.ndarray:
    """Newton/IRLS logistic regression with an intercept: returns `[b, w]` (d+1,) maximizing
    `sum_i log P(y_i | x_i) - l2/2 |w|^2`. The tiny ridge (intercept unpenalized) only keeps the weights
    finite on (quasi-)separable data; each Newton step is backtracked on the penalized objective."""
    X = np.asarray(X, dtype=np.float64)
    t = np.asarray(y, dtype=np.float64).ravel()
    if X.ndim != 2 or X.shape[0] != t.shape[0]:
        raise ValueError(f"X must be (n,d) with n = len(y): {X.shape} vs {t.shape}")
    A = np.hstack([np.ones((X.shape[0], 1)), X])
    penalty = l2 * np.diag(np.r_[0.0, np.ones(X.shape[1])])

    def objective(w: np.ndarray) -> float:
        eta = A @ w
        return float(np.sum(np.logaddexp(0.0, eta) - t * eta) + 0.5 * w @ penalty @ w)

    w = np.zeros(A.shape[1])
    f = objective(w)
    for it in range(max_iter):
        p = expit(A @ w)
        grad = A.T @ (p - t) + penalty @ w
        hess = (A * (p * (1.0 - p))[:, None]).T @ A + penalty + 1e-12 * np.eye(A.shape[1])
        step = np.linalg.solve(hess, grad)
        size = 1.0
        while True:
            w_new = w - size * step
            f_new = objective(w_new)
            if f_new <= f - 1e-4 * size * float(grad @ step) or size < 1e-8:
                break
            size *= 0.5
        w, f = w_new, f_new
        if float(np.max(np.abs(size * step))) < tol:
            break
        if it == max_iter - 1:
            logger.warning("fit_logistic_irls: no convergence in %d Newton steps (last step %.3g)", max_iter,
                           float(np.max(np.abs(size * step))))
    return w


def _score(w: np.ndarray, mean: np.ndarray, std: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Nonconformity score `1 - P(good)` = `expit(-logit)` (no cancellation near P = 1)."""
    Z = (X - mean) / std
    return np.asarray(expit(-(w[0] + Z @ w[1:])), dtype=np.float64)


def _stratified_split(good: np.ndarray, fit_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Seeded per-class split -> (fit indices, calibration indices). Keeps >= 1 clip of each class in
    the fit set and >= 1 good clip in the calibration set."""
    rng = np.random.default_rng(seed)
    fit, cal = [], []
    for label in (True, False):
        idx = rng.permutation(np.flatnonzero(good == label))
        hi = len(idx) - 1 if label else len(idx)
        n_fit = min(max(round(fit_fraction * len(idx)), 1), hi)
        fit.append(idx[:n_fit])
        cal.append(idx[n_fit:])
    return np.sort(np.concatenate(fit)), np.sort(np.concatenate(cal))


def fit_calibration_split(true_pose_error: np.ndarray, tau: float = 0.02, fit_fraction: float = 0.5,
                          seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """`(fit indices, calibration indices)`: the seeded label-stratified split `fit_acceptance_rule` makes
    for the same errors, `tau`, `fit_fraction` and `seed`. Only calibration-split clips may feed a later
    conformal step on the fitted rule (its threshold here, or `fastdo.qc.selection.select_clips`)."""
    err = np.asarray(true_pose_error, dtype=np.float64).ravel()
    if not np.isfinite(err).all():
        raise ValueError("pose errors must be finite")
    if not 0.0 < fit_fraction < 1.0:
        raise ValueError(f"fit_fraction must be in (0, 1), got {fit_fraction}")
    good = err < tau
    n_good, n_bad = int(good.sum()), int((~good).sum())
    if n_good < 2 or n_bad < 1:
        raise ValueError(f"need >= 2 good and >= 1 bad clips at tau={tau}: got {n_good} good, {n_bad} bad")
    return _stratified_split(good, fit_fraction, seed)


def fit_acceptance_rule(qc: list[QcFeatures], true_pose_error: np.ndarray, alpha: float = 0.1,
                        tau: float = 0.02, fit_fraction: float = 0.5, seed: int = 0,
                        l2: float = 1e-3) -> AcceptanceRule:
    """Fit the logistic score on a stratified `fit_fraction` of the clips and conformally calibrate its
    threshold on the remaining good clips (see module docstring for the guarantee)."""
    X = np.stack([q.to_array() for q in qc]) if qc else np.zeros((0, _N_FEATURES))
    err = np.asarray(true_pose_error, dtype=np.float64).ravel()
    if X.shape[0] != err.shape[0]:
        raise ValueError(f"{X.shape[0]} QC feature sets but {err.shape[0]} pose errors")
    if not (np.isfinite(X).all() and np.isfinite(err).all()):
        raise ValueError("QC features and pose errors must be finite")
    fit_idx, cal_idx = fit_calibration_split(err, tau=tau, fit_fraction=fit_fraction, seed=seed)
    good = err < tau
    mean = X[fit_idx].mean(axis=0)
    std = X[fit_idx].std(axis=0)
    std = np.where(std > 1e-12, std, 1.0)
    w = fit_logistic_irls((X[fit_idx] - mean) / std, good[fit_idx], l2=l2)
    cal_good = cal_idx[good[cal_idx]]
    cal_bad = cal_idx[~good[cal_idx]]
    threshold = split_conformal_threshold(_score(w, mean, std, X[cal_good]), alpha)
    calibration = ConformalCalibration(threshold=threshold, alpha=alpha, n_calibration=int(cal_good.size))
    false_accept = float((_score(w, mean, std, X[cal_bad]) <= threshold).mean()) if cal_bad.size else float("nan")
    logger.info("fit_acceptance_rule: fit %d clips, calibrated on %d good (threshold %.4f, alpha %g, tau %g m); "
                "held-out bad clips accepted: %.3f of %d", fit_idx.size, cal_good.size, threshold, alpha, tau,
                false_accept, cal_bad.size)
    return AcceptanceRule(calibration=calibration, logistic_weights=w, feature_mean=mean, feature_std=std,
                          tau=tau)


def calibrate(qc: list[QcFeatures], true_pose_error: np.ndarray, alpha: float = 0.1, tau: float = 0.02,
              fit_fraction: float = 0.5, seed: int = 0) -> tuple[ConformalCalibration, np.ndarray]:
    """The conformal calibration and fitted logistic weights `(6,)` of `fit_acceptance_rule` (the same
    split and fit). The weights act on standardized features, so a deployable rule needs the fit-set
    statistics too: use `fit_acceptance_rule` to get the full `AcceptanceRule`."""
    rule = fit_acceptance_rule(qc, true_pose_error, alpha=alpha, tau=tau, fit_fraction=fit_fraction, seed=seed)
    return rule.calibration, rule.logistic_weights


def score_clips(rule: AcceptanceRule, qc: list[QcFeatures]) -> np.ndarray:
    """Nonconformity scores `1 - P(good | qc)` (m,) of the already-fitted `rule` (higher = worse); nothing
    is refitted, so held-out clips' scores stay exchangeable with those of new clips."""
    X = np.stack([c.to_array() for c in qc]) if qc else np.zeros((0, _N_FEATURES))
    finite = np.isfinite(X).all(axis=1)
    if not finite.all():
        first = int(np.flatnonzero(~finite)[0])
        raise ValueError(f"non-finite QC feature in clip {first}: {qc[first]}")
    return _score(rule.logistic_weights, rule.feature_mean, rule.feature_std, X)


def accept(rule: AcceptanceRule, qc: QcFeatures) -> tuple[bool, float]:
    """`(score <= threshold, 1 - score)` with `score = 1 - P(good | qc)`."""
    score = float(score_clips(rule, [qc])[0])
    return bool(score <= rule.calibration.threshold), 1.0 - score


def pose_error_m(pred_poses: np.ndarray, gt_poses: np.ndarray) -> float:
    """Clip pose error for calibration labels: mean per-frame translation error (m) between (T,4,4)
    trajectories, i.e. `fastdo.bench.metrics.retarget_errors(...).e_pos` (no new error definition)."""
    return retarget_errors(np.asarray(pred_poses, dtype=np.float64), np.asarray(gt_poses, dtype=np.float64)).e_pos
