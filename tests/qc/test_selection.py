import logging
import math
from collections.abc import Callable

import numpy as np
import pytest

import fastdo.qc as qc_pkg
import fastdo.qc.acceptance as acceptance_mod
import fastdo.qc.selection as selection_mod
from fastdo.qc.acceptance import fit_acceptance_rule, fit_calibration_split, score_clips
from fastdo.qc.conformal import split_conformal_threshold
from fastdo.qc.selection import SelectionResult, benjamini_hochberg, conformal_pvalues, select_clips
from tests.qc.test_acceptance import _synthetic_batch

logger = logging.getLogger(__name__)

_Q = 0.1
_ALPHA = 0.1
_TAU = 0.02
_TS = (0.05, 0.1, 0.2)


def _band(t: float, n: int) -> float:
    return 3.0 * math.sqrt(t * (1.0 - t) / n)


def test_conformal_pvalues_formula() -> None:
    cal = np.array([0.10, 0.40, 0.25, 0.70, 0.55, 0.90])
    bad = np.array([False, True, False, True, False, True])       # bad scores {0.40, 0.70, 0.90}, n = 6
    test = np.array([0.05, 0.70, 0.95])
    # deterministic: (1 + #{bad, V_i <= V_j}) / 7; the good 0.10/0.25/0.55 never count; the tie at 0.70 counts
    assert np.array_equal(conformal_pvalues(cal, bad, test), np.array([1.0, 3.0, 4.0]) / 7.0)
    # randomized: (#{bad, V_i < V_j} + U (1 + #{bad, V_i = V_j})) / 7 with U = 1 - rng.random(m) on (0, 1]
    u = 1.0 - np.random.default_rng(7).random(3)
    expected = np.array([0.0 + u[0], 1.0 + 2.0 * u[1], 3.0 + u[2]]) / 7.0
    p = conformal_pvalues(cal, bad, test, rng=np.random.default_rng(7))
    assert np.allclose(p, expected, rtol=0.0, atol=1e-15)
    assert (p > np.array([0.0, 1.0, 3.0]) / 7.0).all() and (p <= np.array([1.0, 3.0, 4.0]) / 7.0).all()
    with pytest.raises(ValueError):
        conformal_pvalues(cal, bad.astype(float), test)             # labels must be boolean
    with pytest.raises(ValueError):
        conformal_pvalues(cal[:5], bad, test)
    with pytest.raises(ValueError):
        conformal_pvalues(cal, bad, np.array([math.nan]))
    with pytest.raises(ValueError):
        conformal_pvalues(np.zeros(0), np.zeros(0, dtype=bool), test)


@pytest.mark.parametrize(("tied", "randomized"), [(False, False), (False, True), (True, False), (True, True)])
def test_pvalues_superuniform_under_null(tied: bool, randomized: bool) -> None:
    """20k bad test clips, each exchangeable with its own draw of 200 bad calibration clips. One shared
    calibration set would not do: conditional on it, P(p <= t) is Beta-distributed with sd ~0.021 at
    t = 0.1, far wider than the 3-sigma binomial band (0.0064). `tied` scores take 8 discrete levels."""
    n_draws, n_cal = 20_000, 200
    rng = np.random.default_rng(20260913 + 2 * int(tied) + int(randomized))
    all_bad = np.ones(n_cal, dtype=bool)
    p = np.empty(n_draws)
    for d in range(n_draws):
        v = rng.integers(0, 8, n_cal + 1).astype(np.float64) if tied else rng.standard_normal(n_cal + 1)
        p[d] = conformal_pvalues(v[:n_cal], all_bad, v[n_cal:], rng=rng if randomized else None)[0]
    for t in _TS:
        frac = float(np.mean(p <= t))
        logger.info("superuniform tied=%s randomized=%s: P(p <= %.2f) = %.4f", tied, randomized, t, frac)
        assert frac <= t + _band(t, n_draws)
        if randomized:                                   # randomized tie-breaking: exactly uniform
            assert frac >= t - _band(t, n_draws)
        elif not tied:                                   # no ties: uniform on the grid {1..n+1}/(n+1)
            assert frac >= math.floor(t * (n_cal + 1)) / (n_cal + 1) - _band(t, n_draws)


def test_pvalues_joint_validity_with_mixed_calibration() -> None:
    """Good clips in the calibration pool keep the denominator at n + 1, so the Jin-Candes guarantee is the
    joint P(p <= t and test clip bad) <= t -- tight for a score that carries no information."""
    n_draws, n_cal, pi_bad = 20_000, 200, 0.4
    rng = np.random.default_rng(11)
    p, test_bad = np.empty(n_draws), np.empty(n_draws, dtype=bool)
    for d in range(n_draws):
        v = rng.standard_normal(n_cal + 1)
        bad = rng.random(n_cal + 1) < pi_bad
        p[d] = conformal_pvalues(v[:n_cal], bad[:n_cal], v[n_cal:], rng=rng)[0]
        test_bad[d] = bad[n_cal]
    for t in _TS:
        joint = float(np.mean((p <= t) & test_bad))
        logger.info("mixed calibration: P(p <= %.2f, bad) = %.4f; P(p <= %.2f | bad) = %.4f", t, joint, t,
                    float(np.mean(p[test_bad] <= t)))
        assert t - _band(t, n_draws) <= joint <= t + _band(t, n_draws)


def _bh_reference(p: np.ndarray, q: float) -> np.ndarray:
    """Self-consistency form of BH: the largest k with #{p <= qk/m} >= k; reject {p <= qk/m}."""
    m = p.size
    for k in range(m, 0, -1):
        if int(np.sum(p <= q * k / m * (1.0 + 1e-12))) >= k:
            return p <= q * k / m * (1.0 + 1e-12)
    return np.zeros(m, dtype=bool)


def test_benjamini_hochberg_reference() -> None:
    # sorted: 0.001 0.015 0.035 0.038 0.050 0.061 0.075 0.300 0.600 0.900 against q k / m = 0.01 k.
    # k = 3 fails (0.035 > 0.03), k = 5 passes with equality (0.05 <= 0.05), k = 6, 7, 8 fail -> reject the
    # five smallest; the step-up at k = 5 rescues 0.035 and 0.038, which a step-down pass would stop before.
    p = np.array([0.600, 0.035, 0.001, 0.300, 0.050, 0.075, 0.015, 0.900, 0.061, 0.038])
    expected = np.array([False, True, True, False, True, False, True, False, False, True])
    assert np.array_equal(benjamini_hochberg(p, 0.1), expected)
    # float boundary: 43 p-values of exactly 0.05 at q = 0.05 satisfy p_(43) <= q 43 / 43 in exact arithmetic,
    # but the float product 0.05 * 43 / 43 is 0.049999999999999996 -- equality must still reject all 43
    assert 0.05 * 43 / 43 < 0.05
    assert benjamini_hochberg(np.full(43, 0.05), 0.05).all()
    assert np.array_equal(benjamini_hochberg(np.array([0.2, 0.3]), 0.1), [False, False])
    assert benjamini_hochberg(np.zeros(0), 0.1).shape == (0,)
    ties = np.array([0.04, 0.04, 0.04, 0.04, 0.9])                   # k = 4: 0.04 <= 0.08, all ties rejected
    assert np.array_equal(benjamini_hochberg(ties, 0.1), [True, True, True, True, False])
    rng = np.random.default_rng(3)
    for _ in range(300):
        m = int(rng.integers(1, 40))
        v = np.round(rng.beta(0.3, 2.0, m), 2)                      # many ties and near-boundary values
        assert np.array_equal(benjamini_hochberg(v, 0.2), _bh_reference(v, 0.2))
    for q in (0.0, 1.0, -0.1):
        with pytest.raises(ValueError):
            benjamini_hochberg(p, q)
    with pytest.raises(ValueError):
        benjamini_hochberg(np.array([0.1, math.nan]), 0.1)


def _controller_draw(rng: np.random.Generator, k: int, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruction of the original simulation (its script was not kept): pose error ~ Gamma(2, 1 cm),
    so P(bad) = 3 e^-2 = 0.406 at tau = 2 cm, and score V = log(error / tau) + sigma N(0, 1)."""
    err = rng.gamma(2.0, 0.01, k)
    return np.log(err / _TAU) + sigma * rng.standard_normal(k), err >= _TAU


def _fdp_power(selected: np.ndarray, bad: np.ndarray) -> tuple[float, float]:
    fdp = float((selected & bad).sum()) / max(int(selected.sum()), 1)
    power = float((selected & ~bad).sum()) / max(int((~bad).sum()), 1)
    return fdp, power


def _simulate(sigma: float, reps: int, seed: int, randomized: bool) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    out = np.zeros(4)
    for _ in range(reps):
        cal_v, cal_bad = _controller_draw(rng, 200, sigma)
        test_v, test_bad = _controller_draw(rng, 200, sigma)
        p = conformal_pvalues(cal_v, cal_bad, test_v, rng=rng if randomized else None)
        accepted = test_v <= split_conformal_threshold(cal_v[~cal_bad], _ALPHA)   # D5 class-conditional
        out += (*_fdp_power(benjamini_hochberg(p, _Q), test_bad), *_fdp_power(accepted, test_bad))
    fdr, power, fdr_cc, power_cc = out / reps
    return {"fdr": fdr, "power": power, "fdr_cc": fdr_cc, "power_cc": power_cc}


@pytest.mark.parametrize("randomized", [False, True])
def test_fdr_controlled_monte_carlo(randomized: bool) -> None:
    r = _simulate(sigma=1.0, reps=1000, seed=5 + int(randomized), randomized=randomized)
    logger.info("reference simulation (n = m = 200, 1000 reps, randomized=%s): selection FDR %.4f power %.4f "
                "| class-conditional alpha=0.1 FDR %.4f power %.4f", randomized, r["fdr"], r["power"],
                r["fdr_cc"], r["power_cc"])
    assert r["fdr"] <= 0.11


def test_power_increases_with_score_quality() -> None:
    sharp = _simulate(sigma=0.3, reps=300, seed=17, randomized=False)
    noisy = _simulate(sigma=1.0, reps=300, seed=17, randomized=False)
    logger.info("power: sigma 0.3 -> %.4f (FDR %.4f), sigma 1.0 -> %.4f (FDR %.4f)", sharp["power"], sharp["fdr"],
                noisy["power"], noisy["fdr"])
    assert sharp["power"] > noisy["power"]


def test_select_clips_does_not_refit(monkeypatch: pytest.MonkeyPatch) -> None:
    qc, err = _synthetic_batch(400, np.random.default_rng(21))
    rule = fit_acceptance_rule(qc[:200], err[:200], alpha=_ALPHA, tau=_TAU, seed=0)
    weights = rule.logistic_weights.copy()
    _, cal_idx = fit_calibration_split(err[:200], tau=_TAU, seed=0)
    cal_qc, cal_err, test_qc = [qc[i] for i in cal_idx], err[cal_idx], qc[200:]
    calls: list[str] = []

    def spy(name: str) -> Callable[..., None]:
        def _spy(*args: object, **kwargs: object) -> None:
            calls.append(name)
            raise AssertionError(f"select_clips called {name}")
        return _spy

    for name in ("fit_acceptance_rule", "fit_logistic_irls", "calibrate"):
        for module in (acceptance_mod, qc_pkg, selection_mod):
            monkeypatch.setattr(module, name, spy(name), raising=False)
    result = select_clips(rule, cal_qc, cal_err, test_qc, q=_Q, tau=_TAU)
    assert calls == []
    assert isinstance(result, SelectionResult) and result.q == _Q
    assert np.array_equal(rule.logistic_weights, weights)
    expected_p = conformal_pvalues(score_clips(rule, cal_qc), cal_err >= _TAU, score_clips(rule, test_qc))
    assert np.array_equal(result.pvalues, expected_p)
    assert np.array_equal(result.selected, benjamini_hochberg(expected_p, _Q))
    assert result.n_calibration == len(cal_idx) and result.n_bad_calibration == int((cal_err >= _TAU).sum())
    assert 0 < int(result.selected.sum()) < len(test_qc)
    with pytest.raises(ValueError):
        result.selected[0] = not result.selected[0]
    empty = select_clips(rule, cal_qc, cal_err, [], q=_Q, tau=_TAU)
    assert empty.selected.shape == (0,) and empty.pvalues.shape == (0,)
    with pytest.raises(ValueError):
        select_clips(rule, cal_qc, cal_err[:-1], test_qc)
    with pytest.raises(ValueError):
        select_clips(rule, cal_qc, cal_err, test_qc, q=0.0)


def test_fit_calibration_split_matches_fit_acceptance_rule() -> None:
    qc, err = _synthetic_batch(200, np.random.default_rng(22))
    rule = fit_acceptance_rule(qc, err, alpha=_ALPHA, tau=_TAU, fit_fraction=0.5, seed=3)
    fit_idx, cal_idx = fit_calibration_split(err, tau=_TAU, fit_fraction=0.5, seed=3)
    assert np.array_equal(np.sort(np.concatenate([fit_idx, cal_idx])), np.arange(200))
    cal_good = cal_idx[err[cal_idx] < _TAU]
    assert rule.calibration.n_calibration == cal_good.size
    assert rule.calibration.threshold == split_conformal_threshold(score_clips(rule, [qc[i] for i in cal_good]),
                                                                   _ALPHA)
    assert np.allclose(rule.feature_mean, np.stack([qc[i].to_array() for i in fit_idx]).mean(axis=0))
    with pytest.raises(ValueError):
        fit_calibration_split(np.zeros(10), tau=_TAU)                   # no bad clip


def test_select_clips_fdr_on_d5_synthetic_features() -> None:
    """End to end on D5's synthetic QC features: per repetition fit a rule on 200 clips, select among 200
    fresh clips with its held-out calibration split (~100 clips). The guarantee is marginal over draws."""
    reps = 200
    rng = np.random.default_rng(23)
    fdp, power, fdp_cc, power_cc = (np.empty(reps) for _ in range(4))
    for r in range(reps):
        qc, err = _synthetic_batch(400, rng)
        rule = fit_acceptance_rule(qc[:200], err[:200], alpha=_ALPHA, tau=_TAU, seed=r)
        _, cal_idx = fit_calibration_split(err[:200], tau=_TAU, seed=r)
        result = select_clips(rule, [qc[i] for i in cal_idx], err[cal_idx], qc[200:], q=_Q, tau=_TAU)
        test_bad = err[200:] >= _TAU
        accepted = score_clips(rule, qc[200:]) <= rule.calibration.threshold
        fdp[r], power[r] = _fdp_power(result.selected, test_bad)
        fdp_cc[r], power_cc[r] = _fdp_power(accepted, test_bad)
    se = float(fdp.std(ddof=1)) / math.sqrt(reps)
    logger.info("D5 synthetic features (%d reps): selection FDR %.4f (se %.4f) power %.4f | class-conditional "
                "FDR %.4f power %.4f", reps, fdp.mean(), se, power.mean(), fdp_cc.mean(), power_cc.mean())
    assert float(fdp.mean()) <= _Q + 3.0 * se
    assert float(power.mean()) > 0.5


def test_select_clips_warns_when_calibration_has_no_bad_clips(caplog: pytest.LogCaptureFixture) -> None:
    """A good-only calibration set (the D5 threshold's convention) makes every p-value 1/(n+1), so BH can
    accept every test clip: select_clips must say so instead of silently accepting."""
    qc, err = _synthetic_batch(200, np.random.default_rng(7))
    rule = fit_acceptance_rule(qc, err)
    good = [f for f, e in zip(qc, err) if e < rule.tau][:50]
    caplog.set_level("WARNING", logger="fastdo.qc.selection")
    res = selection_mod.select_clips(rule, good, np.full(len(good), 0.001), qc[:20], q=0.1)
    assert res.n_bad_calibration == 0
    assert any("no bad clips" in r.getMessage() for r in caplog.records)
