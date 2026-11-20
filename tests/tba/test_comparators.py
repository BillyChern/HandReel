"""Comparator arms (C-SMOOTH, C-CONS): known answers, each with a mutation that fires."""

from __future__ import annotations

import math

import numpy as np
import pytest

from fastdo.bench.metric_frame import along_ray_components
from fastdo.tba.comparators import c_cons, c_smooth, consensus_gauge_base, rts_smooth
from fastdo.tba.geometry import pixel_rays, rotvec_to_matrix

K = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])


def _poses(n: int = 30, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    P = np.tile(np.eye(4), (n, 1, 1))
    for k in range(n):
        P[k, :3, :3] = rotvec_to_matrix(np.array([0.1, 0.02 * k, 0.0]))
        P[k, :3, 3] = (0.03 * math.sin(k / 6.0), -0.02, 0.6 + 0.004 * k) + rng.normal(0.0, 0.001, 3)
    return P


def test_rts_smoother_recovers_a_constant_velocity_series_and_skips_nans() -> None:
    rng = np.random.default_rng(1)
    truth = 0.6 + 0.004 * np.arange(60)
    y = truth + rng.normal(0.0, 0.02, 60)
    y[[5, 17]] = np.nan
    sm = rts_smooth(y, 0.02, 0.005)
    assert np.isfinite(sm).all()
    assert np.sqrt(np.mean((sm - truth) ** 2)) < 0.4 * np.sqrt(np.nanmean((y - truth) ** 2))
    # mutation: a huge process noise makes the smoother an identity on the observed rows -> no improvement
    ident = rts_smooth(y, 0.02, 10.0)
    ok = np.isfinite(y)
    assert np.sqrt(np.mean((ident[ok] - truth[ok]) ** 2)) > 0.9 * np.sqrt(np.mean((y[ok] - truth[ok]) ** 2))


def test_c_smooth_moves_each_centre_along_its_own_ray_only(monkeypatch: pytest.MonkeyPatch) -> None:
    P = _poses()
    cm = np.array([0.01, -0.02, 0.03])
    scale = 0.12
    rng = np.random.default_rng(2)
    noisy = P.copy()
    c = scale * P[:, :3, :3] @ cm + P[:, :3, 3]
    u = c / np.linalg.norm(c, axis=1, keepdims=True)
    noisy[:, :3, 3] += rng.normal(0.0, 0.03, (len(P), 1)) * u                 # along-ray noise only
    out = c_smooth(noisy, scale, cm, meas_sd_m=0.03)
    assert np.array_equal(out[:, :3, :3], noisy[:, :3, :3])
    c_in = scale * noisy[:, :3, :3] @ cm + noisy[:, :3, 3]
    c_out = scale * out[:, :3, :3] @ cm + out[:, :3, 3]
    assert np.allclose(along_ray_components(c_out, c_in).perp, 0.0, atol=1e-12)
    e_in = np.sqrt(np.mean(along_ray_components(c_in, c).signed_along ** 2))
    e_out = np.sqrt(np.mean(along_ray_components(c_out, c).signed_along ** 2))
    assert e_out < 0.6 * e_in
    # mutation: the smoother replaced by the identity -> the noise stays and the check fires
    monkeypatch.setattr("fastdo.tba.comparators.rts_smooth", lambda y, *a, **k: np.asarray(y, np.float64))
    stiff = c_smooth(noisy, scale, cm, meas_sd_m=0.03)
    c_st = scale * stiff[:, :3, :3] @ cm + stiff[:, :3, 3]
    assert np.sqrt(np.mean(along_ray_components(c_st, c).signed_along ** 2)) > 0.9 * e_in


def test_c_cons_reads_the_pointmap_gauge_exactly() -> None:
    P = _poses(12)
    rng = np.random.default_rng(3)
    uv = rng.uniform([200, 150], [440, 330], (40, 2))
    rays = pixel_rays(uv, K)
    d = rng.uniform(0.55, 0.65, 40)
    vis = rng.uniform(size=(12, 40)) > 0.2
    from fastdo.tba.bundle import pair_depths
    from fastdo.tba.geometry import relative_poses

    zb = pair_depths(relative_poses(P, 0), d, rays)
    lam = 1.07
    z_obs = np.where(vis, lam * zb, np.nan)
    ell = consensus_gauge_base(P, 0, rays, d, vis, z_obs)
    assert ell == pytest.approx(math.log(lam), abs=1e-12)
    out, s = c_cons(P, 0.1, ell)
    assert s == pytest.approx(0.1 * lam) and np.allclose(out[:, :3, 3], lam * P[:, :3, 3])
    # mutation: a pointmap at BASE's depths reads no gauge
    assert consensus_gauge_base(P, 0, rays, d, vis, np.where(vis, zb, np.nan)) == pytest.approx(0.0, abs=1e-12)
