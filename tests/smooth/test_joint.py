"""Known answers for the REGISTERED estimator and the joint arms.

KA-MAP-REG     the registered config (estimate noise reference, Huber, 3 reweightings) returns the minimiser of the
               independently written objective with S_k and the Huber weights FROZEN at its own output; mutation: the
               centre covariance evaluated at the MEASURED ray (`rts._ref_ray`) is not that minimiser.
KA-GAUGE-HO    equivariance holds on the DIAGONAL gauge subgroup (one common lambda for object and hand) for C1 and
               AR1D-HO; a separate lambda per body is NOT an equivariance of C1 (the grasp factor mixes the bodies'
               metric scales) and the size of the break is bounded on seen data;
               the per-body hold keeps each body's weighted range mean exactly.
KA-LEMMA1-HO   AR1D-HO keeps each body's weighted range mean without the hold (Lemma 1 per body).
KA-MAP-HO      AR1D-HO equals scipy's minimiser of its independently written linear objective.
KA-COMMON-HO   common along-ray jitter on object and wrist: AR1D-HO keeps the gap and reduces both bodies' error;
               object-only AR1D opens the gap (mutation).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import least_squares

from fastdo.smooth import SmootherConfig, alongray_joint, alongray_ranges, smooth_window
from fastdo.smooth import rts as rts_mod
from fastdo.smooth.rts import Noise, smooth_se3
from tests.smooth.objective import check_fixed_point
from tests.smooth.synth import along_jitter, rms, rot_noise, twist_trajectory

NZ = Noise(s_rot=0.05, a_par=0.06, a_perp=0.01, q_w=0.0125, q_v=0.009, a_par_h=0.025, a_perp_h=0.007,
           q_h=0.004, q_c=0.004, reweights=3, gn_iters=40, noise_ref="estimate")


def _window(T: int = 12, spike: bool = True):
    R, c = twist_trajectory(T=T, xi=(0.0, 0.03, 0.0, 0.003, 0.0, 0.002))
    h = c + np.einsum("kij,j->ki", R, np.array([0.0, 0.06, -0.03]))
    Rm = rot_noise(R, 0.04)
    cm = along_jitter(c, 0.05, 0.008, seed=3)
    hm = along_jitter(h, 0.02, 0.006, seed=4)
    if spike:                                       # Huber must act (weights < 1) for the check to mean anything
        cm[5] = cm[5] + 0.3 * cm[5] / np.linalg.norm(cm[5])
        hm[8] = hm[8] + 0.1 * hm[8] / np.linalg.norm(hm[8])
    con = np.zeros(T, bool)
    con[2:10] = True
    return Rm, cm, hm, con


def _reg_gap() -> tuple[float, object]:
    Rm, cm, hm, con = _window()
    dt = np.ones(len(cm) - 1)
    res = rts_mod.smooth_se3(Rm, cm, dt, NZ, hand_m=hm, contact=con, gauge_hold=False)
    return check_fixed_point(res, Rm, cm, hm, con, NZ, dt), res


def test_ka_map_reg_registered_estimator_is_the_frozen_objective_minimiser() -> None:
    gap, res = _reg_gap()
    assert np.nanmin(res.w_c) < 0.5 and np.nanmin(res.w_h) < 1.0          # Huber is active in this window
    assert gap < 1e-6


def test_ka_map_reg_mutation_measured_ray_reference_fires(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rts_mod, "_ref_ray", lambda c_est, c_meas: c_meas / np.linalg.norm(c_meas))
    gap, _ = _reg_gap()
    assert gap > 1e-4


def _poses(R: np.ndarray, c: np.ndarray, s: float, p_bar: np.ndarray) -> np.ndarray:
    P = np.tile(np.eye(4), (len(c), 1, 1))
    P[:, :3, :3] = R
    P[:, :3, 3] = c - s * R @ p_bar
    return P


def _gauge_case(arm: str, lam_o: float, lam_h: float) -> tuple[float, float, float, float]:
    """Max |S(g Y) - g S(Y)| (m) for object centres and wrists, and each body's held weighted-range ratio."""
    T = 30
    R, c = twist_trajectory(T=T, xi=(0.0, 0.03, 0.0, 0.003, 0.0, 0.002))
    h = c + np.einsum("kij,j->ki", R, np.array([0.0, 0.06, -0.03]))
    Rm, cm, hm = rot_noise(R, 0.04), along_jitter(c, 0.06, 0.004, seed=3), along_jitter(h, 0.025, 0.004, seed=4)
    s, p_bar = 0.1, np.array([0.01, -0.02, 0.03])
    con = np.ones(T, bool)
    cfg = SmootherConfig(arm=arm)
    fr = np.arange(T)

    def run(lo: float, lh: float):
        o = smooth_window(_poses(Rm, lo * cm, lo * s, p_bar), lo * s, p_bar, fr, cfg, wrist=lh * hm, contact=con,
                          source_domain="third_person")
        cc = lo * s * o.poses[:, :3, :3] @ p_bar + o.poses[:, :3, 3]
        return cc, lh * hm + o.hand_shift

    c0, h0 = run(1.0, 1.0)
    c1, h1 = run(lam_o, lam_h)
    d_o = float(np.abs(c1 - lam_o * c0).max())
    d_h = float(np.abs(h1 - lam_h * h0).max())
    r_o = float(np.sum(np.linalg.norm(c1, axis=1)) / np.sum(np.linalg.norm(lam_o * cm, axis=1)))
    r_h = float(np.sum(np.linalg.norm(h1, axis=1)) / np.sum(np.linalg.norm(lam_h * hm, axis=1)))
    return d_o, d_h, r_o, r_h


@pytest.mark.parametrize("arm", ["se3_contact", "alongray_contact"])
@pytest.mark.parametrize("lam", [0.9, 1.1])
def test_ka_gauge_ho_diagonal_subgroup_is_exact(arm: str, lam: float) -> None:
    d_o, d_h, _, _ = _gauge_case(arm, lam, lam)
    assert d_o < 1e-8 and d_h < 1e-8


@pytest.mark.parametrize("arm", ["se3_contact", "alongray_contact"])
def test_ka_gauge_ho_per_body_lambda_is_not_an_equivariance(arm: str) -> None:
    """An object-only lambda changes the within-window shape of BOTH bodies (the
    seen-data size of this break is measured separately); the unweighted range mean stays within 0.5%."""
    d_o, d_h, r_o, r_h = _gauge_case(arm, 1.1, 1.0)
    assert d_o > 1e-6 and d_h > 1e-6
    assert abs(r_o - 1.0) < 5e-3 and abs(r_h - 1.0) < 5e-3


def _joint_inputs(T: int = 40, common: bool = True):
    _, c = twist_trajectory(T=T, xi=(0.0, 0.0, 0.0, 0.003, 0.0, 0.002), wobble=(0.0, 0.0002))
    h = c + np.array([0.0, 0.06, -0.03])
    rng = np.random.default_rng(7)
    j = rng.normal(0, 0.05, T)
    jh = j if common else rng.normal(0, 0.02, T)
    return c, h, c * (1 + j)[:, None], h * (1 + jh)[:, None]


NZ1 = Noise(s_rot=0.05, a_par=0.06, a_perp=0.01, q_w=0.0125, q_v=0.009, a_par_h=0.025, a_perp_h=0.007,
            q_h=0.00375, q_c=0.0158)


@pytest.mark.parametrize("rw", [0, 3])
def test_ka_lemma1_ho_per_body_range_mean_kept_without_hold(rw: int) -> None:
    _, _, cm, hm = _joint_inputs(common=False)
    cm[[3, 17]] = np.nan
    hm[[9]] = np.nan
    nz = Noise(**{**NZ1.__dict__, "reweights": rw})
    jr = alongray_joint(cm, hm, np.ones(len(cm), bool), np.ones(len(cm) - 1), nz, gauge_hold=False)
    assert abs(jr.gamma) < 1e-9 and abs(jr.gamma_hand) < 1e-9
    assert np.isnan(jr.c[[3, 17]]).all() and np.isnan(jr.hand[9]).all()


def test_ka_map_ho_equals_the_independent_linear_minimiser() -> None:
    _, _, cm, hm = _joint_inputs(T=20, common=False)
    T = len(cm)
    con = np.zeros(T, bool)
    con[4:15] = True
    nz = Noise(**{**NZ1.__dict__, "reweights": 0})
    jr = alongray_joint(cm, hm, con, np.ones(T - 1), nz, gauge_hold=False)
    ro, rh = np.linalg.norm(cm, axis=1), np.linalg.norm(hm, axis=1)
    so, sh = nz.a_par * np.median(ro), nz.a_par_h * np.median(rh)
    Lq = lambda q: np.linalg.cholesky(q ** 2 * np.array([[1 / 3, 1 / 2], [1 / 2, 1.0]]))

    def f(x: np.ndarray) -> np.ndarray:
        a, va, b, vb = x[:T], x[T:2 * T], x[2 * T:3 * T], x[3 * T:]
        out = [(a - ro) / so, (b - rh) / sh, (va - vb)[con] / nz.q_c]
        for k in range(T - 1):
            out.append(np.linalg.solve(Lq(nz.q_v), [a[k + 1] - a[k] - va[k], va[k + 1] - va[k]]))
            out.append(np.linalg.solve(Lq(nz.q_h), [b[k + 1] - b[k] - vb[k], vb[k + 1] - vb[k]]))
        return np.concatenate([np.ravel(o) for o in out])

    x0 = np.r_[ro, np.zeros(T), rh, np.zeros(T)]
    sol = least_squares(f, x0, method="lm", xtol=1e-15, ftol=1e-15, gtol=1e-15)
    assert np.abs(sol.x[:T] - np.linalg.norm(jr.c, axis=1)).max() < 1e-8
    assert np.abs(sol.x[2 * T:3 * T] - np.linalg.norm(jr.hand, axis=1)).max() < 1e-8


def test_ka_common_ho_keeps_the_gap_and_object_only_opens_it() -> None:
    c, h, cm, hm = _joint_inputs()
    T = len(c)
    jr = alongray_joint(cm, hm, np.ones(T, bool), np.ones(T - 1), NZ1)
    gap = lambda co, ho: rms(np.linalg.norm((co - ho) - (c - h), axis=1))
    assert gap(jr.c, jr.hand) < 0.25 * rms(np.linalg.norm(cm - c, axis=1))
    assert rms(np.linalg.norm(jr.c - c, axis=1)) < 0.6 * rms(np.linalg.norm(cm - c, axis=1))
    assert rms(np.linalg.norm(jr.hand - h, axis=1)) < 0.6 * rms(np.linalg.norm(hm - h, axis=1))
    alone, _, _ = alongray_ranges(cm, np.ones(T - 1), NZ1.a_par, NZ1.q_v / (NZ1.a_par * np.median(
        np.linalg.norm(cm, axis=1))))
    assert gap(alone, hm) > 4.0 * gap(jr.c, jr.hand)


def test_joint_fallback_without_a_hand_is_object_only() -> None:
    _, _, cm, _ = _joint_inputs()
    jr = alongray_joint(cm, None, None, np.ones(len(cm) - 1), NZ1)
    assert jr.hand is None and np.isfinite(jr.c).all()


def test_se3_contact_reports_block_weights() -> None:
    Rm, cm, hm, con = _window(spike=False)
    res = smooth_se3(Rm, cm, np.ones(len(cm) - 1), NZ, hand_m=hm, contact=con)
    assert res.w_h is not None and np.isfinite(res.w_g[con]).all() and np.isnan(res.w_g[~con]).all()
