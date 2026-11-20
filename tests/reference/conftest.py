"""Reference-conditioner tests. `FASTDO_S299_MUTATION=<name>` applies one MUTATION of the module (a mutation runner
sets it); every mutation must make at least one test here fail."""

from __future__ import annotations

import dataclasses
import os

import numpy as np
import pytest

from fastdo.reference.config import ReferenceConfig

MUTATION = os.environ.get("FASTDO_S299_MUTATION", "")
CFG_MUTATIONS = {
    "d_off_eq_d_on": {"release_m": 0.015},
    "no_guard": {"guard_rows": 0},
    "near_tier": {"sym_tol_abs_m": 0.010, "sym_tol_rel": 1.0},
    "no_unary": {"flip_unary_deg": 0.0},
    "no_action_cap": {"cap_pos_m": 1e9, "cap_rot_deg": 1e9},
    # rev 2 (the vote and motion floors)
    "min_vote_4": {"min_vote": 4},
    "no_motion_floor": {"motion_floor_m": 0.0, "motion_floor_deg": 0.0},
    # rev 2: the two-sided reading of alpha (p <= 0.025)
    "alpha_half": {"drift_alpha": 0.025},
}


@pytest.fixture
def cfg() -> ReferenceConfig:
    return dataclasses.replace(ReferenceConfig(), **CFG_MUTATIONS.get(MUTATION, {}))


@pytest.fixture(autouse=True)
def _mutate(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from fastdo.reference import conditioner as CN
    from fastdo.reference import frames as FR
    from fastdo.reference import frechet as FM
    from fastdo.reference import motion as MT
    from fastdo.reference import segmentation as SG

    if MUTATION == "no_transport":
        monkeypatch.setattr(FR, "transport_covariance", lambda cov, poses, c, ws: np.asarray(cov, dtype=np.float64))
    elif MUTATION == "no_sigma_transport":
        orig = FR.transport_covariance

        def no_sigma(cov, poses, c, ws):
            return orig(cov, poses, c, dataclasses.replace(ws, sigma=np.ones_like(ws.sigma)))
        monkeypatch.setattr(FR, "transport_covariance", no_sigma)
    elif MUTATION == "cauchy":
        monkeypatch.setattr(FM, "_rho", lambda e, k: 0.5 * k * k * np.log1p((e / k) ** 2))
        monkeypatch.setattr(FM, "_weight", lambda e, k: 1.0 / (1.0 + (e / k) ** 2))
    elif MUTATION == "skip_drift":                  # the conditioner never refuses `moved`
        def never_moved(times, r, cfg=None, seed=0, cov=None, scale=None):
            return dataclasses.replace(MT.motion_test_residuals(times, r, cfg, seed, cov=cov, scale=scale),
                                       moved=False)
        monkeypatch.setattr(CN, "MT", SimpleNamespace(motion_test_residuals=never_moved, MotionTest=MT.MotionTest))
    elif MUTATION == "floor_only":                  # the supplied covariance never reaches the mean
        monkeypatch.setattr(CN, "effective_covariance",
                            lambda cov_cam, poses, c_star, ws, alpha, cfg: np.broadcast_to(
                                FM.floor_covariance(cfg), (len(poses), 6, 6)).copy())
    elif MUTATION == "drift_trend_only":            # no split-half test
        monkeypatch.setattr(MT, "_split_half", lambda z, zp, B, min_rows: None)
    elif MUTATION == "drift_no_clip":               # no Huber clip of the whitened residuals
        monkeypatch.setattr(MT, "_clip", lambda z, k: z)
    elif MUTATION == "no_bonferroni":               # OR of two level-alpha tests
        monkeypatch.setattr(MT, "_combine", lambda p_t, p_h: p_t if p_h is None else min(p_t, p_h))
    elif MUTATION == "scale_from_levels":           # robust scale from residual levels, not first differences
        def levels(r, cfg):
            r = np.asarray(r, dtype=np.float64)
            out = np.empty(2)
            for b, sl in enumerate((slice(0, 3), slice(3, 6))):
                dc = r[:, sl] - r[:, sl].mean(axis=0)
                out[b] = max(np.sqrt(float(np.median(np.sum(dc * dc, axis=1))) / 2.365974), 1e-4 if b == 0 else 1.7e-4)
            return out
        monkeypatch.setattr(MT, "robust_scale", levels)
    elif MUTATION == "conditioner_weighted_drift":  # the conditioner passes Sigma_eff to its drift test
        orig_cr = CN.consolidate_rows

        def weighted(rows, p, O, sigma_eff, conf, times, elements, cfg, estimator="cond"):
            rows_ = np.asarray(rows, dtype=int)
            vote = rows_[np.asarray(conf)[rows_] >= cfg.min_confidence]
            orig_mt = MT.motion_test_residuals
            CN.MT = SimpleNamespace(MotionTest=MT.MotionTest, motion_test_residuals=lambda t, r, c=None, s=0, **kw:
                                    orig_mt(t, r, c, s, cov=np.asarray(sigma_eff)[vote]))
            try:
                return orig_cr(rows, p, O, sigma_eff, conf, times, elements, cfg, estimator)
            finally:
                CN.MT = MT
        monkeypatch.setattr(CN, "consolidate_rows", weighted)
    elif MUTATION == "cap_position_only":
        monkeypatch.setattr(CN, "_cap_mask", lambda p_rows, O_rows, pbar, Obar, elements, cfg:
                            np.linalg.norm(p_rows - pbar, axis=1) <= cfg.cap_pos_m)
    elif MUTATION == "cap_no_quotient":
        orig_cap = CN._cap_mask
        monkeypatch.setattr(CN, "_cap_mask", lambda p_rows, O_rows, pbar, Obar, elements, cfg:
                            orig_cap(p_rows, O_rows, pbar, Obar, elements[:1], cfg))
    elif MUTATION == "merge_across_touch":
        def merged(states):
            idx = np.flatnonzero(np.asarray(states) == "rest")
            return [idx] if idx.size else []
        monkeypatch.setattr(SG, "rest_segments", merged)
    elif MUTATION == "fill_nonfinite":
        orig_rs = SG.row_states

        def filled(statuses, pose_ok, cfg):
            return orig_rs(statuses, np.ones_like(np.asarray(pose_ok, dtype=bool)), cfg)
        monkeypatch.setattr(SG, "row_states", filled)
    elif MUTATION == "unseen_hand_certified":
        orig_hs = SG.hand_status

        def certified(dist, times, declared_absent, cfg):
            return orig_hs(dist, times, declared_absent or not np.isfinite(dist).any(), cfg)
        monkeypatch.setattr(SG, "hand_status", certified)
    elif MUTATION == "no_quotient":
        orig_rm = FM.robust_mean

        def no_q(p, O, Sigma, elements, cfg, huber=True):
            return orig_rm(p, O, Sigma, elements[:1], cfg, huber)
        monkeypatch.setattr(FM, "robust_mean", no_q)
