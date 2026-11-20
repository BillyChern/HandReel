import dataclasses
import math

import numpy as np
import pytest
from scipy.stats import rankdata

from fastdo.qc.acceptance import (
    AcceptanceRule,
    accept,
    calibrate,
    fit_acceptance_rule,
    fit_logistic_irls,
    pose_error_m,
)
from fastdo.qc.conformal import ConformalCalibration
from fastdo.qc.features import QcFeatures

_ALPHA = 0.1
_TAU = 0.02


def _synthetic_batch(n: int, rng: np.random.Generator) -> tuple[list[QcFeatures], np.ndarray]:
    """Noisy monotone (features, pose error) pairs: every feature degrades as the error grows."""
    err = rng.gamma(shape=2.0, scale=0.008, size=n)                   # ~71% below tau = 2 cm
    iou = np.clip(0.97 - 6.0 * err + rng.normal(0.0, 0.04, n), 0.0, 1.0)
    std_ok = np.clip(1.0 - 12.0 * err + rng.normal(0.0, 0.12, n), 0.0, 1.0)
    ot = 0.01 + 0.4 * err + np.abs(rng.normal(0.0, 0.004, n))
    pen = np.abs(rng.normal(0.0, 0.0015, n)) + 0.15 * err
    inl = np.clip(0.95 - 8.0 * err + rng.normal(0.0, 0.06, n), 0.0, 1.0)
    qc = [QcFeatures(reprojection_iou=float(a), laplace_std_ok_frac=float(b), ot_residual_rms_frac=float(c),
                     penetration_depth_m=float(d), track_inlier_ratio=float(e))
          for a, b, c, d, e in zip(iou, std_ok, ot, pen, inl, strict=True)]
    return qc, err


def test_logistic_irls_recovers_known_weights() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20000, 3))
    w_true = np.array([0.5, 1.5, -2.0, 0.0])
    p = 1.0 / (1.0 + np.exp(-(w_true[0] + X @ w_true[1:])))
    y = rng.random(20000) < p
    w = fit_logistic_irls(X, y)
    assert w.shape == (4,)
    assert np.allclose(w, w_true, atol=0.1)


def test_accept_rejects_out_of_distribution_clip() -> None:
    qc, err = _synthetic_batch(200, np.random.default_rng(1))
    rule = fit_acceptance_rule(qc, err, alpha=_ALPHA, tau=_TAU)
    ood = QcFeatures(reprojection_iou=0.1, laplace_std_ok_frac=0.9, ot_residual_rms_frac=0.015,
                     penetration_depth_m=0.05, track_inlier_ratio=0.9)
    accepted, conf = accept(rule, ood)
    assert accepted is False
    assert 0.0 <= conf < 1.0 - rule.calibration.threshold
    typical = dataclasses.replace(ood, reprojection_iou=0.95, penetration_depth_m=0.001)
    assert accept(rule, typical)[0] is True


def test_calibrate_end_to_end_on_synthetic() -> None:
    qc_cal, err_cal = _synthetic_batch(200, np.random.default_rng(2))
    calibration, weights = calibrate(qc_cal, err_cal, alpha=_ALPHA)
    rule = fit_acceptance_rule(qc_cal, err_cal, alpha=_ALPHA, tau=_TAU)
    assert isinstance(calibration, ConformalCalibration) and calibration == rule.calibration
    assert np.array_equal(weights, rule.logistic_weights) and weights.shape == (6,)
    assert calibration.alpha == _ALPHA and calibration.n_calibration >= 9   # rank <= n: non-vacuous

    qc_new, err_new = _synthetic_batch(200, np.random.default_rng(3))
    decisions = [accept(rule, q) for q in qc_new]
    accepted = np.array([a for a, _ in decisions])
    conf = np.array([c for _, c in decisions])
    assert ((conf >= 0.0) & (conf <= 1.0)).all()
    assert np.array_equal(accepted, 1.0 - conf <= calibration.threshold)
    good = err_new < _TAU
    n_good = int(good.sum())
    miscoverage = float((~accepted[good]).mean())                     # good clips wrongly rejected
    assert miscoverage <= _ALPHA + 3.0 * math.sqrt(_ALPHA * (1.0 - _ALPHA) / n_good)
    # the rule is not vacuous: the score ranks clips by error, most bad clips are rejected, and accepted
    # clips have lower error (individual weight signs are not asserted: the features are collinear)
    rank_corr = float(np.corrcoef(rankdata(1.0 - conf), rankdata(err_new))[0, 1])
    assert rank_corr > 0.5
    false_accept = float(accepted[~good].mean())
    assert false_accept < 0.5
    assert err_new[accepted].mean() < err_new[~accepted].mean()


def test_calibrate_requires_both_classes() -> None:
    qc, err = _synthetic_batch(50, np.random.default_rng(4))
    with pytest.raises(ValueError):
        calibrate(qc, np.zeros_like(err))
    with pytest.raises(ValueError):
        calibrate(qc, err[:10])


def test_acceptance_rule_is_immutable_and_validated() -> None:
    cal = ConformalCalibration(threshold=0.3, alpha=0.1, n_calibration=50)
    rule = AcceptanceRule(calibration=cal, logistic_weights=np.zeros(6), feature_mean=np.zeros(5),
                          feature_std=np.ones(5))
    with pytest.raises(ValueError):
        rule.logistic_weights[0] = 1.0
    with pytest.raises(ValueError):
        AcceptanceRule(calibration=cal, logistic_weights=np.zeros(5), feature_mean=np.zeros(5),
                       feature_std=np.ones(5))
    with pytest.raises(ValueError):
        accept(rule, QcFeatures(math.nan, 1.0, 0.0, 0.0, 1.0))


def test_pose_error_m_reuses_bench_metrics() -> None:
    gt = np.tile(np.eye(4), (3, 1, 1))
    pred = gt.copy()
    pred[:, 0, 3] += np.array([0.01, 0.02, 0.03])
    assert math.isclose(pose_error_m(pred, gt), 0.02, rel_tol=1e-12)
