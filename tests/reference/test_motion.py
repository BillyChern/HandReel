"""rev 2 noise-calibrated motion test: known answers."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from fastdo.reference import motion as MT
from fastdo.reference.so3 import exp

N11 = 11
TAU11 = np.arange(N11) / 10.0


def _iid(rng, n, pos_sd, rot_sd_deg):
    return np.concatenate([rng.normal(0, pos_sd, (n, 3)), rng.normal(0, np.radians(rot_sd_deg), (n, 3))], axis=1)


def _poses(rng, n, drift_m=0.0, drift_deg=0.0, pos_sd=0.002, rot_sd_deg=0.5, u=(1.0, 0.0, 0.0), a=(0.0, 0.0, 1.0)):
    """World centres and rotations of a lead with a linear drift (centre and rotation about the centre) plus iid
    noise."""
    f = np.linspace(0.0, 1.0, n)
    p0 = np.array([0.3, -0.1, 0.8])
    O0 = Rotation.from_rotvec([0.4, -0.2, 0.9]).as_matrix()
    p = p0 + f[:, None] * drift_m * np.asarray(u) + rng.normal(0, pos_sd, (n, 3))
    drift = exp(f[:, None] * np.radians(drift_deg) * np.asarray(a))
    noise = exp(rng.normal(0, np.radians(rot_sd_deg), (n, 3)))
    return p, noise @ drift @ O0


def test_level_under_iid_static_noise(cfg) -> None:
    """The Bonferroni-combined test keeps its joint level: <= 0.07 rejections in 1,000 static iid leads of 11 rows
    (rev 1's OR of two 0.05 tests rejects ~10%). Lower bound: >= 0.035,
    so the test is not run at half its level (the two-sided reading alpha/2 = 0.025 rejects 0.031 here; the
    pre-registered one-sided 0.05 rejects 0.053)."""
    rng = np.random.default_rng(1)
    rej = 0
    for i in range(1000):
        m = MT.motion_test_residuals(TAU11, _iid(rng, N11, 0.01, 3.0), cfg, seed=i)
        rej += m.significant
    assert 0.035 <= rej / 1000 <= 0.07


def test_alpha_is_one_sided_005(cfg, monkeypatch) -> None:
    """Known answer pinning the pre-registered ONE-SIDED alpha = 0.05: a combined p of 0.04
    (inside (0.025, 0.05]) is significant and 0.0501 is not. A two-sided reading (p <= 0.025) fails the first."""
    rng = np.random.default_rng(12)
    r = _iid(rng, 20, 0.002, 0.3)
    tau = np.arange(20) / 10.0
    assert cfg.drift_alpha == 0.05
    for p, want in ((0.04, True), (0.05, True), (0.0501, False)):
        monkeypatch.setattr(MT, "_combine", lambda p_t, p_h, _p=p: _p)
        m = MT.motion_test_residuals(tau, r, cfg, seed=0)
        assert m.p_value == p and m.significant is want


def test_robust_scale_is_taken_from_first_differences(cfg) -> None:
    """The robust scale uses first differences, so a linear drift does not inflate it. 100 rows, a 1 m / 60 deg
    linear drift under 2 mm / 0.5 deg iid noise: the scale stays within 20% of the true noise sd; a scale from the
    residual LEVELS would be ~0.29 m / ~17 deg."""
    rng = np.random.default_rng(11)
    n = 100
    r = _iid(rng, n, 0.002, 0.5)
    r[:, 0] += np.linspace(0.0, 1.0, n)
    r[:, 4] += np.linspace(0.0, np.radians(60.0), n)
    s = MT.robust_scale(r, cfg)
    assert abs(s[0] - 0.002) <= 0.2 * 0.002
    assert abs(np.degrees(s[1]) - 0.5) <= 0.2 * 0.5


def test_large_linear_drift_is_moved_and_its_size_recovered(cfg) -> None:
    rng = np.random.default_rng(2)
    for i in range(5):
        p, O = _poses(rng, N11, drift_m=0.10)
        m = MT.motion_test(TAU11, p, O, cfg, seed=i)
        assert m.decided and m.significant and m.moved
        assert 90.0 < m.magnitude_mm < 110.0 and 90.0 < m.trend_mm < 110.0
        p, O = _poses(rng, N11, drift_deg=30.0)
        m = MT.motion_test(TAU11, p, O, cfg, seed=i)
        assert m.moved and 27.0 < m.magnitude_deg < 33.0 and m.trend_mm < 5.0


def test_a_significant_drift_below_the_floor_is_not_moved(cfg) -> None:
    """Rev 1's KA-noop defect: a 5 mm trend under 0.2 mm noise is significant, but below the 20 mm / 15 deg floor."""
    rng = np.random.default_rng(3)
    for i in range(5):
        p, O = _poses(rng, 20, drift_m=0.005, pos_sd=0.0002, rot_sd_deg=0.02)
        m = MT.motion_test(np.arange(20) / 10.0, p, O, cfg, seed=i)
        assert m.significant and not m.exceeds_floor and not m.moved
        assert 4.0 < m.magnitude_mm < 6.0


def test_one_gross_row_has_bounded_influence(cfg) -> None:
    """A 1 m outlier at the END of a 100-row static segment does not make the trend significant (Huber clip): <= 12%
    of 200 segments reach p_trend <= 0.05. Unclipped, the outlier alone gives p ~ 2/n at every seed."""
    rng = np.random.default_rng(4)
    tau = np.arange(100) / 10.0
    hits = 0
    for i in range(200):
        r = _iid(rng, 100, 0.002, 0.5)
        r[-1, 0] += 1.0
        hits += MT.motion_test_residuals(tau, r, cfg, seed=i).p_trend <= 0.05
    assert hits / 200 <= 0.12


def test_a_trendless_shift_is_caught_by_the_split_half(cfg) -> None:
    """Halves 4 cm apart with the linear component projected out (a sawtooth, 40 rows, whitened by a given 3 cm /
    2 deg scale so no row is clipped): the trend test is blind (p > 0.5), the split-half test is not (p <= 0.005);
    the shift exceeds the floor, so the lead is moved. A trend-only test would call it static."""
    n = 40
    tau = np.arange(n) / 10.0
    c = tau - tau.mean()
    h = np.where(np.arange(n) < n // 2, 1.0, -1.0)
    x = h - (h @ c) / (c @ c) * c
    x *= 0.04 / abs(x[: n // 2].mean() - x[n // 2:].mean())
    rng = np.random.default_rng(5)
    r = _iid(rng, n, 0.0005, 0.05)
    r[:, 0] += x
    m = MT.motion_test_residuals(tau, r, cfg, seed=0, scale=np.array([0.03, np.radians(2.0)]))
    assert m.scale_source == "given" and m.p_trend > 0.5 and m.p_half <= 0.005
    assert m.significant and m.moved and 35.0 < m.half_disc_mm < 45.0 and m.trend_mm < 5.0


def test_covariance_whitening(cfg) -> None:
    """(a) A constant diagonal covariance equal to the robust scale gives the robust result exactly. (b) The
    covariance is USED: a 5 cm shift confined to rows declared 1 m uncertain is not motion; whitened by the rows' own
    scale it is."""
    rng = np.random.default_rng(6)
    tau = np.arange(20) / 10.0
    r = _iid(rng, 20, 0.003, 0.5)
    r[:, 0] += np.linspace(0, 0.03, 20)
    s = MT.robust_scale(r, cfg)
    cov = np.broadcast_to(np.diag(np.repeat(s ** 2, 3)), (20, 6, 6))
    a = MT.motion_test_residuals(tau, r, cfg, seed=3)
    b = MT.motion_test_residuals(tau, r, cfg, seed=3, cov=cov)
    assert a.scale_source == "robust_residual" and b.scale_source == "covariance"
    for f in ("p_trend", "p_half", "trend_stat", "half_stat", "trend_mm", "half_disc_mm", "magnitude_deg"):
        assert abs(getattr(a, f) - getattr(b, f)) <= 1e-9 * max(1.0, abs(getattr(a, f)))
    r2 = _iid(rng, 20, 0.002, 0.3)
    r2[14:, 0] += 0.05
    cov2 = np.tile(np.diag([0.002 ** 2] * 3 + [np.radians(0.3) ** 2] * 3), (20, 1, 1))
    cov2[14:, :3, :3] = np.eye(3) * 1.0
    assert MT.motion_test_residuals(tau, r2, cfg, seed=1).moved
    assert not MT.motion_test_residuals(tau, r2, cfg, seed=1, cov=cov2).moved


def test_invariance_to_a_world_rigid_motion(cfg) -> None:
    rng = np.random.default_rng(7)
    p, O = _poses(rng, N11, drift_m=0.03, drift_deg=10.0, pos_sd=0.01, rot_sd_deg=3.0)
    Q = Rotation.from_rotvec([1.1, -0.4, 0.3]).as_matrix()
    u = np.array([2.0, -1.0, 0.5])
    a = MT.motion_test(TAU11, p, O, cfg, seed=9)
    b = MT.motion_test(TAU11, p @ Q.T + u, Q @ O, cfg, seed=9)
    assert a.p_trend == b.p_trend and a.p_half == b.p_half and a.moved == b.moved
    for f in ("trend_mm", "trend_deg", "half_disc_mm", "half_disc_deg", "scale_mm", "scale_deg"):
        assert abs(getattr(a, f) - getattr(b, f)) <= 1e-6 * max(1.0, abs(getattr(a, f)))


def test_symmetry_quotient_in_the_residuals(cfg) -> None:
    """Every other row relabelled by a 180 deg symmetry: the same physical state, not motion."""
    rng = np.random.default_rng(8)
    G = np.stack([np.eye(3), Rotation.from_rotvec([0, 0, np.pi]).as_matrix()])
    p, O = _poses(rng, 20, pos_sd=0.002, rot_sd_deg=0.5)
    O[::2] = O[::2] @ G[1]
    m = MT.motion_test(np.arange(20) / 10.0, p, O, cfg, seed=0, elements=G)
    assert m.magnitude_deg < 2.0 and not m.moved


def test_too_few_rows_is_undecided(cfg) -> None:
    rng = np.random.default_rng(10)
    p, O = _poses(rng, 4, drift_m=0.2)
    m = MT.motion_test(np.arange(4) / 10.0, p, O, cfg)
    assert not m.decided and not m.moved and m.reason == "too_few_rows"
