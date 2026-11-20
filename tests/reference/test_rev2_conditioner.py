"""rev 2: conditioner known answers that kill the mutants rev 1 left alive: covariance weighting, the cap's
rotation and quotient, min_vote, and the drift refusal's magnitude floor. (The drift test's split-half and Huber
clip are killed in `test_motion.py`, where the conditioner's drift test now lives.)"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from fastdo.reference import condition_window, consolidate_rows
from fastdo.reference import frechet as FM
from fastdo.reference.so3 import exp

from .synth import noisy, pose, window

BASE = pose([0.2, -0.3, 0.4], [0.05, -0.02, 0.6])


def _static(n, rng, pos_sd=0.0005, rot_deg=0.05, O0=None):
    O0 = np.eye(3) if O0 is None else O0
    p = rng.normal(0, pos_sd, (n, 3))
    return p, exp(rng.normal(0, np.radians(rot_deg), (n, 3))) @ O0


def test_supplied_covariance_weights_the_mean(cfg) -> None:
    """Rows alternate between a 1 mm and a 20 mm camera-frame position sd (identity rotation, rotation variance at
    the sanitiser's floor): the consolidated centre is the inverse-variance mean of Sigma_eff = Sigma^w + Sigma_0
    (closed form, 1e-7 m), 2.65 mm from the floor-only (equal-weight) mean."""
    n = 20
    P = np.tile(np.eye(4), (n, 1, 1))
    P[:, :3, 3] = [0.05, -0.02, 0.6]
    x = np.where(np.arange(n) % 2 == 0, 0.003, -0.003)
    P[:, 0, 3] += x
    var = np.where(np.arange(n) % 2 == 0, 0.001 ** 2, 0.020 ** 2)
    cov = np.zeros((n, 6, 6))
    cov[:, :3, :3] = var[:, None, None] * np.eye(3)
    cov[:, 3:, 3:] = 1e-12 * np.eye(3)
    ref = condition_window(window(P, np.full(n, 0.3), cov=cov), cfg)
    s = ref.segments[0]
    assert s.status == "consolidated" and len(s.replaced) == n
    w = 1.0 / (var + cfg.floor_pos_m ** 2)
    want = float(np.sum(w * x) / np.sum(w))
    c0 = ref.c_star + np.array([0.05, -0.02, 0.6])
    assert abs((s.p - c0)[0] - want) < 1e-7 and abs(want) > 0.0025


def test_the_cap_checks_the_rotation(cfg) -> None:
    """A row at the mean's centre but rotated 30 deg is kept (the cap is centre AND angle)."""
    rng = np.random.default_rng(1)
    n = 14
    p, O = _static(n, rng)
    O[6] = exp(np.radians(30.0) * np.array([0.0, 1.0, 0.0])) @ O[6]
    S0 = np.broadcast_to(FM.floor_covariance(cfg), (n, 6, 6))
    rep = consolidate_rows(np.arange(n), p, O, S0, np.ones(n), np.arange(n) / 10, np.eye(3)[None], cfg)
    assert rep.status == "consolidated"
    assert 6 in rep.outlier_kept and 6 not in rep.replaced and len(rep.replaced) == n - 1


def test_the_cap_uses_the_symmetry_quotient(cfg) -> None:
    """Half the rows carry a 180 deg symmetry relabelling (the same physical state): every row is within the cap in
    the quotient and is replaced."""
    rng = np.random.default_rng(2)
    G = np.stack([np.eye(3), Rotation.from_rotvec([0, 0, np.pi]).as_matrix()])
    n = 12
    p, O = _static(n, rng, O0=Rotation.from_rotvec([0.2, 0.4, -0.1]).as_matrix())
    O[::2] = O[::2] @ G[1]
    S0 = np.broadcast_to(FM.floor_covariance(cfg), (n, 6, 6))
    rep = consolidate_rows(np.arange(n), p, O, S0, np.ones(n), np.arange(n) / 10, G, cfg)
    assert rep.status == "consolidated" and len(rep.replaced) == n and len(rep.outlier_kept) == 0


def test_min_vote_is_five(cfg) -> None:
    rng = np.random.default_rng(3)
    S0 = FM.floor_covariance(cfg)
    for n, want in ((4, "rest_short"), (5, "consolidated")):
        p, O = _static(n, rng)
        rep = consolidate_rows(np.arange(n), p, O, np.broadcast_to(S0, (n, 6, 6)), np.ones(n), np.arange(n) / 10,
                               np.eye(3)[None], cfg)
        assert rep.status == want, (n, rep.status, rep.reason)


def test_a_significant_sub_floor_drift_is_consolidated(cfg) -> None:
    """Rev 1 refused clean GT segments whose GT moved 0.04-1.6 mm. A 1.5 mm linear drift under 0.05 mm noise is
    significant but below the 20 mm / 15 deg floor: consolidated."""
    rng = np.random.default_rng(5)
    n = 30
    P = noisy(BASE, n, rng, pos_sd=0.00005, rot_sd_deg=0.005)
    P[:, :3, 3] += np.linspace(0, 0.0015, n)[:, None] * [1, 0, 0]
    ref = condition_window(window(P, np.full(n, 0.3)), cfg)
    s = ref.segments[0]
    assert s.status == "consolidated" and ref.changed.all()
    assert s.drift.significant and not s.drift.moved


def test_the_conditioner_drift_test_is_deliberately_unweighted(cfg) -> None:
    """The conditioner's `moved` refusal whitens by the
    voting rows' ROBUST RESIDUAL SCALE even when a per-row covariance is supplied (variant W); Sigma_eff weights the
    MEAN only. A supplied covariance must not reach the drift test: rows with a 5 cm step declared 1 m uncertain are
    still refused `moved`."""
    n = 20
    P = np.tile(np.eye(4), (n, 1, 1))
    P[:, :3, 3] = [0.05, -0.02, 0.6]
    rng = np.random.default_rng(13)
    P[:, :3, 3] += rng.normal(0, 0.002, (n, 3))
    P[14:, 0, 3] += 0.05
    cov = np.zeros((n, 6, 6))
    cov[:, :3, :3] = 0.002 ** 2 * np.eye(3)
    cov[:, 3:, 3:] = 1e-8 * np.eye(3)
    cov[14:, :3, :3] = np.eye(3)
    ref = condition_window(window(P, np.full(n, 0.3), cov=cov), cfg)
    s = ref.segments[0]
    assert s.drift is not None and s.drift.scale_source == "robust_residual"
    assert s.status == "refused" and s.reason == "moved"
