"""Known answers and mutations for fastdo.bench.metric_source."""

from __future__ import annotations

import math

import numpy as np
import pytest

from fastdo.bench.metric_source import (
    caliper_extents,
    caliper_log_sd,
    e1_split,
    expected_e1,
    gls_fuse,
    known_size_correction,
    mpjpe_root_aligned,
    object_cd_after_scaled_icp,
    pa_mpjpe,
    size_reading,
)


def _box(dims: tuple[float, float, float], n: int = 4000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = rng.uniform(-0.5, 0.5, (n, 3)) * np.asarray(dims)
    face = rng.integers(0, 3, n)
    p[np.arange(n), face] = np.sign(p[np.arange(n), face]) * np.asarray(dims)[face] / 2
    return p


def _rot(seed: int) -> np.ndarray:
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


def _traj(n: int = 30, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.normal(0, 0.05, n), rng.normal(0, 0.05, n), rng.uniform(0.45, 0.75, n)])


def test_caliper_extents_rotation_invariant() -> None:
    p = _box((0.04, 0.10, 0.18))
    e0 = caliper_extents(p)
    e1 = caliper_extents(p @ _rot(3).T + 0.3)
    np.testing.assert_allclose(e0, [0.04, 0.10, 0.18], atol=2e-3)
    np.testing.assert_allclose(e0, e1, atol=2e-3)


@pytest.mark.parametrize("kind", ["obb_gm", "obb_max", "rms_radius", "diameter"])
def test_size_reading_is_degree_one_homogeneous(kind: str) -> None:
    p = _box((0.05, 0.08, 0.12))
    assert size_reading(1.37 * p, kind) == pytest.approx(1.37 * size_reading(p, kind), rel=2e-3)


def test_known_size_recovers_a_pure_gauge_exactly() -> None:
    """KA: true shape as template, solve on the orbit (scale lam, centres lam c): kappa = 1/lam and E1 -> 0."""
    shape = _box((0.06, 0.09, 0.15))
    cg = _traj()
    for lam in (0.8, 1.07, 1.25):
        kappa = known_size_correction(shape @ _rot(1).T, lam, shape, "obb_gm")
        assert kappa == pytest.approx(1.0 / lam, rel=3e-3)
        sp = e1_split(lam * cg, cg, kappa)
        assert sp.e1 < 3e-3 * float(np.sqrt(np.mean(np.sum(cg ** 2, 1))))


def test_known_size_mutation_inverted_correction_fires() -> None:
    shape = _box((0.06, 0.09, 0.15))
    cg = _traj()
    lam = 1.12
    kappa = known_size_correction(shape, lam, shape, "obb_gm")
    good = e1_split(lam * cg, cg, kappa).e1
    bad = e1_split(lam * cg, cg, 1.0 / kappa).e1        # mutation: correction applied the wrong way
    assert bad > 50 * max(good, 1e-6)


def test_e1_split_is_exact_pythagorean() -> None:
    rng = np.random.default_rng(5)
    cg = _traj(40, 5)
    cp = 1.09 * cg + rng.normal(0, 0.02, cg.shape)
    for kappa in (1.0, 0.93, 1.2):
        sp = e1_split(cp, cg, kappa)
        assert sp.e1 ** 2 == pytest.approx(sp.gauge_part ** 2 + sp.offorbit_part ** 2, rel=1e-12)
    # mutation: dropping the orbit point (lam := 1) breaks the identity whenever lam != 1
    z = np.linalg.norm(cg, axis=1)
    a = np.einsum("ij,ij->i", cp - cg, cg / z[:, None])
    assert np.sqrt(np.mean(a ** 2)) ** 2 != pytest.approx(e1_split(cp, cg).offorbit_part ** 2, rel=1e-3)


def test_oracle_gauge_leaves_only_the_off_orbit_floor() -> None:
    rng = np.random.default_rng(6)
    cg = _traj(40, 6)
    cp = 0.9 * cg + rng.normal(0, 0.01, cg.shape)
    sp0 = e1_split(cp, cg)
    sp = e1_split(cp, cg, 1.0 / sp0.lam)
    assert sp.gauge_part == pytest.approx(0.0, abs=1e-12)
    assert sp.e1 == pytest.approx(sp0.offorbit_part / sp0.lam, rel=1e-9)


def test_expected_e1_matches_monte_carlo_and_zero_noise() -> None:
    rng = np.random.default_rng(7)
    cg = _traj(25, 7)
    cp = 1.05 * cg + rng.normal(0, 0.01, cg.shape)
    k0 = 0.97
    assert expected_e1(cp, cg, k0, 0.0) == pytest.approx(e1_split(cp, cg, k0).e1, rel=1e-9)
    eps = np.random.default_rng(8).normal(0, 0.08, 40_000)
    mc = math.sqrt(np.mean([e1_split(cp, cg, k0 * math.exp(e)).e1 ** 2 for e in eps[:4000]]))
    assert expected_e1(cp, cg, k0, 0.08) == pytest.approx(mc, rel=0.03)
    tp = math.sqrt(0.5 * (e1_split(cp, cg, k0 * 1.05).e1 ** 2 + e1_split(cp, cg, k0 / 1.05).e1 ** 2))
    assert expected_e1(cp, cg, k0, two_point=0.05) == pytest.approx(tp, rel=1e-9)


def test_gls_fuse_theorem2_weights_and_monotone_variance() -> None:
    est, var = gls_fuse([0.0, 0.1], [0.01, 0.01])
    assert est == pytest.approx(0.05) and var == pytest.approx(0.005)
    est, var = gls_fuse([0.0, 0.1], [0.01, 0.0])
    assert est == pytest.approx(0.1) and var == 0.0
    _, v1 = gls_fuse([0.0], [0.01])
    _, v2 = gls_fuse([0.0, 0.3], [0.01, 0.5])
    assert v2 < v1
    with pytest.raises(ValueError):
        gls_fuse([0.0, 0.1], [0.0, 0.0])


def test_caliper_log_sd_delta_method() -> None:
    e = np.array([0.05, 0.10, 0.20])
    assert caliper_log_sd(e, 0.001, "obb_max") == pytest.approx(0.005)
    assert caliper_log_sd(e, 0.001, "obb_gm") == pytest.approx(0.001 * math.sqrt(400 + 100 + 25) / 3)


def test_scaled_icp_object_cd_is_blind_to_a_gauge() -> None:
    """The literature's object protocol is exactly blind to g_k (scale about the camera centre)."""
    shape = _box((0.05, 0.08, 0.12), n=1500)
    rng = np.random.default_rng(9)
    gt = [shape @ _rot(i).T + np.array([0.0, 0.0, 0.6]) for i in range(3)]
    pred = [g + rng.normal(0, 0.002, g.shape) for g in gt]
    cd1, s1 = object_cd_after_scaled_icp(pred, gt)
    cdk, sk = object_cd_after_scaled_icp([1.2 * p for p in pred], gt)
    assert cdk == pytest.approx(cd1, rel=0.05)
    assert sk == pytest.approx(s1 / 1.2, rel=0.01)
    cdn1, _ = object_cd_after_scaled_icp(pred, gt, normalise_init=True)
    cdnk, snk = object_cd_after_scaled_icp([1.2 * p for p in pred], gt, normalise_init=True)
    assert cdnk == pytest.approx(cdn1, rel=1e-9)
    assert snk * 1.2 == pytest.approx(object_cd_after_scaled_icp(pred, gt, normalise_init=True)[1], rel=1e-9)
    # counter-example kept on purpose: uncentred ICP from the identity stalls 12 cm away along the ray
    cd_u, _ = object_cd_after_scaled_icp([1.2 * p for p in pred], gt, centre=False)
    assert cd_u > 10 * cd1


def test_mpjpe_root_aligned_and_pa() -> None:
    rng = np.random.default_rng(10)
    g = rng.normal(0, 0.04, (5, 21, 3)) + np.array([0.0, 0.0, 0.5])
    assert np.allclose(mpjpe_root_aligned(g + 0.1, g), 0.0)
    m = mpjpe_root_aligned(1.1 * g, g)          # a whole-scene gauge scales the root-relative hand by 1.1
    rel = np.linalg.norm(g - g[:, :1], axis=2).mean(1)
    np.testing.assert_allclose(m, 0.1 * rel, rtol=1e-9)
    np.testing.assert_allclose(pa_mpjpe(1.1 * g, g), 0.0, atol=1e-9)
