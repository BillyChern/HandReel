"""A4 (b): the identifiable-component motion test (`fastdo.reference.a4`), known answers on synthetic
leads. The block length and noise are the committed A4 (a) values unless a test sets its own."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from fastdo.reference import a4 as A4
from fastdo.reference.rayframe import ray_frames
from fastdo.reference.so3 import exp

CENTRE = np.array([0.15, -0.05, 0.9])
O0 = Rotation.from_rotvec([0.4, -0.2, 0.9]).as_matrix()


def _ar1(rng, n, phi, shape=(3,)):
    e = rng.normal(size=(n, *shape))
    out = np.empty_like(e)
    out[0] = e[0]
    for t in range(1, n):
        out[t] = phi * out[t - 1] + np.sqrt(1 - phi ** 2) * e[t]
    return out


def _lead(rng, n, drift=(0.0, 0.0, 0.0), drift_deg=0.0, axis=(0.0, 0.0, 1.0), pos_sd=0.006, rot_sd_deg=2.0, phi=0.0,
          ray_sd=0.0):
    """A lead seen by a static camera at the origin (identity rotation): linear drift plus noise (AR(1) phi)."""
    f = np.linspace(0.0, 1.0, n)
    u = CENTRE / np.linalg.norm(CENTRE)
    p = CENTRE + f[:, None] * np.asarray(drift) + pos_sd * _ar1(rng, n, phi) + ray_sd * rng.normal(size=(n, 1)) * u
    O = exp(np.radians(rot_sd_deg) * _ar1(rng, n, phi)) @ exp(f[:, None] * np.radians(drift_deg) * np.asarray(axis)) @ O0
    cams = np.zeros((n, 3))
    rots = np.tile(np.eye(3), (n, 1, 1))
    return np.arange(n) / 10.0, p, O, cams, rots


def _perp_dir():
    u = CENTRE / np.linalg.norm(CENTRE)
    v = np.cross(u, [0.0, 1.0, 0.0])
    return v / np.linalg.norm(v)


def test_whiteners_drop_the_along_ray_component() -> None:
    rng = np.random.default_rng(0)
    A = ray_frames(rng.normal(size=(4, 3)), Rotation.random(4, random_state=2).as_matrix(), CENTRE)
    W = A4.a4_whiteners(A)
    for i in range(4):
        u, e1 = A[i, :, 0], A[i, :, 1]
        assert np.allclose(W[i] @ np.r_[u, 0, 0, 0], 0.0, atol=1e-12)
        assert np.allclose(W[i] @ np.r_[e1, 0, 0, 0], np.r_[e1 / A4.A4_NOISE.sd[1], 0, 0, 0])
        assert np.linalg.matrix_rank(W[i]) == 5
    C = A4.a4_covariance(A)
    assert np.allclose(np.einsum("ji,njk,kl->nil", np.eye(3), A.transpose(0, 2, 1) @ C[:, :3, :3] @ A, np.eye(3)),
                       np.diag(np.square(A4.A4_NOISE.sd[:3])))


def test_block_orders() -> None:
    o, exact = A4.block_orders(20, 8, 1999, np.random.default_rng(0))
    assert exact and len(o) == 6 and len({tuple(x) for x in o}) == 6
    for x in o:
        assert sorted(x) == list(range(20))
        assert any(list(x[:8]) == list(range(s, s + 8)) for s in (0, 8)) or list(x[:4]) == [16, 17, 18, 19]
    o, exact = A4.block_orders(80, 8, 1999, np.random.default_rng(0))          # 10 blocks: 10! > 2000
    assert not exact and len(o) == 2000 and list(o[0]) == list(range(80))


def test_a_short_lead_cannot_be_significant() -> None:
    """12 rows in blocks of 8: two blocks, two orders, smallest combined p 1.0: never significant, flagged."""
    rng = np.random.default_rng(1)
    t, p, O, C, R = _lead(rng, 12, drift=0.30 * _perp_dir())
    m = A4.a4_motion_test(t, p, O, C, R)
    assert m.decided and m.n_blocks == 2 and m.min_p == 1.0 and not m.significant and not m.moved
    assert m.reason == "too_few_blocks" and m.perp_mm > 250.0


def test_perpendicular_drift_is_moved_along_ray_drift_is_not() -> None:
    """48 rows (6 blocks, 720 exact orders, smallest p 2/720): a 10 cm perpendicular drift is moved; a 30 cm drift
    ALONG the ray is not seen at all (perp magnitude ~0)."""
    rng = np.random.default_rng(2)
    u = CENTRE / np.linalg.norm(CENTRE)
    for i in range(3):
        m = A4.a4_motion_test(*_lead(rng, 48, drift=0.10 * _perp_dir()), seed=i)
        assert m.exact and m.min_p == 2.0 / 720 and m.significant and m.moved
        assert 80.0 < m.perp_mm < 120.0
        m = A4.a4_motion_test(*_lead(rng, 48, drift=0.30 * u), seed=i)
        assert not m.moved and m.perp_mm < 10.0


def test_rotation_drift_and_floor() -> None:
    rng = np.random.default_rng(3)
    m = A4.a4_motion_test(*_lead(rng, 48, drift_deg=30.0, rot_sd_deg=1.0))
    assert m.moved and 25.0 < m.rot_deg < 35.0
    m = A4.a4_motion_test(*_lead(rng, 48, drift_deg=8.0, rot_sd_deg=0.2, pos_sd=0.001))
    assert m.significant and not m.exceeds_floor and not m.moved


def test_block_null_holds_the_level_under_correlated_noise() -> None:
    """Static leads of 48 rows under AR(1) 0.5 noise (tau_int 3 rows, inside one block of 8) at the measured scale:
    the block-permutation test is called significant in <= 0.08 of 200 leads; an iid permutation (block length 1) in
    >= 0.40 of them. (When the correlation time approaches the block length the block null inflates too; the
    adoption gate (d)(i) measures the level with measured residual blocks.)"""
    rng = np.random.default_rng(4)
    iid = dataclasses.replace(A4.A4_NOISE, block_len=1, permutations=499)
    hits_b = hits_1 = 0
    for i in range(200):
        lead = _lead(rng, 48, pos_sd=0.006, rot_sd_deg=2.0, phi=0.5)
        hits_b += A4.a4_motion_test(*lead, seed=i).significant
        hits_1 += A4.a4_motion_test(*lead, noise=iid, seed=i).significant
    assert hits_b / 200 <= 0.08 and hits_1 / 200 >= 0.40


def test_invariance_to_a_world_rigid_motion() -> None:
    rng = np.random.default_rng(5)
    t, p, O, C, R = _lead(rng, 40, drift=0.04 * _perp_dir(), drift_deg=10.0)
    Q = Rotation.from_rotvec([1.1, -0.4, 0.3]).as_matrix()
    s = np.array([2.0, -1.0, 0.5])
    a = A4.a4_motion_test(t, p, O, C, R, seed=3)
    b = A4.a4_motion_test(t, p @ Q.T + s, Q @ O, C @ Q.T + s, Q @ R, seed=3)
    assert a.p_trend == b.p_trend and a.p_half == b.p_half and a.moved == b.moved
    for f in ("perp_mm", "rot_deg", "trend_perp_mm", "half_perp_mm"):
        assert abs(getattr(a, f) - getattr(b, f)) <= 1e-6 * max(1.0, abs(getattr(a, f)))


def test_too_few_rows_is_undecided() -> None:
    rng = np.random.default_rng(6)
    m = A4.a4_motion_test(*_lead(rng, 4))
    assert not m.decided and not m.moved and m.reason == "too_few_rows"


def test_alpha_is_one_sided_005(monkeypatch) -> None:
    """Both permutation p-values 0.02 -> Bonferroni-combined 0.04, inside (0.025, 0.05]: significant. A two-sided
    reading (p <= 0.025) would not be."""
    rng = np.random.default_rng(7)
    monkeypatch.setattr(A4, "_p", lambda stat_perm, obs: 0.02)
    m = A4.a4_motion_test(*_lead(rng, 48))
    assert m.p_value == pytest.approx(0.04) and m.significant


def test_the_perpendicular_magnitude_excludes_the_mean_ray() -> None:
    """The camera orbits the object by 60 deg over the lead, so a drift along the MEAN ray is partly seen in the
    rows' perpendicular axes; the reported perpendicular magnitude still excludes the mean-ray direction (a 30 cm
    mean-ray drift reports < 0.3 of its size, a 10 cm drift across the mean ray reports ~10 cm)."""
    rng = np.random.default_rng(8)
    n = 48
    ang = np.radians(np.linspace(-30.0, 30.0, n))
    cams = CENTRE + 0.9 * np.stack([np.sin(ang), np.zeros(n), -np.cos(ang)], axis=1)
    rots = np.tile(np.eye(3), (n, 1, 1))
    ubar = np.array([0.0, 0.0, 1.0])
    f = np.linspace(0.0, 1.0, n)
    O = np.tile(O0, (n, 1, 1))
    t = np.arange(n) / 10.0
    p = CENTRE + f[:, None] * 0.30 * ubar + rng.normal(0, 0.002, (n, 3))
    m = A4.a4_motion_test(t, p, O, cams, rots)
    assert m.perp_mm < 90.0
    p = CENTRE + f[:, None] * 0.10 * np.array([0.0, 1.0, 0.0]) + rng.normal(0, 0.002, (n, 3))
    m = A4.a4_motion_test(t, p, O, cams, rots)
    assert 80.0 < m.perp_mm < 120.0


def test_the_huber_clip_bounds_gross_outlier_rows() -> None:
    """Added because the mutant `no_huber_clip` survived. Heavy tails are central to (a): 13-18% of the
    measured rows lie beyond 3 robust sd. Each whitened row is clipped to norm <= k = 3.55, so the trend statistic is
    bounded by k^2 (sum |c_i|)^2 / sum c_i^2 however gross the outliers are. A static 48-row lead with six rows thrown
    1 m perpendicular (about 170 sd) must respect that bound; without the clip it is exceeded many times over."""
    rng = np.random.default_rng(9)
    t, p, O, C, R = _lead(rng, 48, pos_sd=0.003, rot_sd_deg=0.5)
    out = np.array([3, 17, 30, 44, 45, 47])
    p = p.copy()
    p[out] += 1.0 * _perp_dir()
    O = O.copy()
    O[out] = exp(np.tile(np.radians([0.0, 90.0, 0.0]), (len(out), 1))) @ O[out]
    m = A4.a4_motion_test(t, p, O, C, R)
    c = t - t.mean()
    k = 3.55
    bound = k ** 2 * np.sum(np.abs(c)) ** 2 / np.sum(c ** 2)
    assert m.decided and m.extra["trend_stat"] <= bound + 1e-9
    clean = A4.a4_motion_test(*_lead(np.random.default_rng(9), 48, pos_sd=0.003, rot_sd_deg=0.5))
    assert clean.extra["trend_stat"] <= bound
