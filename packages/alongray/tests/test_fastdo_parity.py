# Copyright 2026 The HandReel authors
# SPDX-License-Identifier: Apache-2.0
"""alongray equals fastdo's metric_frame / metric_source / alignment, bit for bit, on fixtures from fastdo's tests.

The fixtures (_traj, _poses, _box, _rot and the constructions) are copied from tests/bench/test_metric_frame.py,
tests/bench/test_metric_source.py and tests/bench/test_alignment.py of the HandReel repository. These tests
need ``fastdo`` importable; outside that repository they skip, unless ALONGRAY_REQUIRE_FASTDO=1, which turns a
missing ``fastdo`` into a failure (set on the development host so that the parity check cannot pass by skipping).

Bit-for-bit equality holds on VALID inputs only. Where the inputs are degenerate the two deliberately differ:
alongray raises on a non-positive or non-finite scale, a bad kappa and non-(T, 3) or mismatched joint shapes (fastdo
reshapes or broadcasts), and ``pa_mpjpe`` scores a frame with an undefined similarity as NaN where fastdo raises.
"""

from __future__ import annotations

import os
from dataclasses import astuple

import numpy as np
import pytest

import alongray

if os.environ.get("ALONGRAY_REQUIRE_FASTDO") == "1":
    import fastdo.bench.alignment as fa
    import fastdo.bench.metric_frame as fm
    import fastdo.bench.metric_source as fs
else:
    fm = pytest.importorskip("fastdo.bench.metric_frame")
    fs = pytest.importorskip("fastdo.bench.metric_source")
    fa = pytest.importorskip("fastdo.bench.alignment")


# ---- fixtures copied from tests/bench/test_metric_frame.py
def _traj_mf(n: int = 20, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.array([0.05, -0.03, 0.55])
    return base + 0.08 * rng.standard_normal((n, 3)) * np.array([1.0, 1.0, 0.5])


def _rot_mf(seed: int) -> np.ndarray:
    q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((3, 3)))
    return q * np.sign(np.linalg.det(q))


def _poses(centres: np.ndarray, seed: int = 0) -> np.ndarray:
    out = np.tile(np.eye(4), (len(centres), 1, 1))
    for t in range(len(centres)):
        out[t, :3, :3] = _rot_mf(seed + t)
    out[:, :3, 3] = centres
    return out


# ---- fixtures copied from tests/bench/test_metric_source.py
def _box(dims: tuple[float, float, float], n: int = 4000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = rng.uniform(-0.5, 0.5, (n, 3)) * np.asarray(dims)
    face = rng.integers(0, 3, n)
    p[np.arange(n), face] = np.sign(p[np.arange(n), face]) * np.asarray(dims)[face] / 2
    return p


def _rot_ms(seed: int) -> np.ndarray:
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


def _traj_ms(n: int = 30, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.normal(0, 0.05, n), rng.normal(0, 0.05, n), rng.uniform(0.45, 0.75, n)])


def _cases() -> list[tuple[str, np.ndarray, np.ndarray]]:
    """(name, pred, gt) centre trajectories built as fastdo's tests build them."""
    gt = _traj_mf()
    u = gt / np.linalg.norm(gt, axis=1, keepdims=True)
    perp = np.cross(u, np.array([0.0, 1.0, 0.0]))
    perp /= np.linalg.norm(perp, axis=1, keepdims=True)
    nan_pred = 1.1 * gt
    nan_pred[3] = np.nan
    poses, _ = fm.apply_gauge(_poses(gt), 0.12, 1.17)
    rng5 = np.random.default_rng(5)
    cg5 = _traj_ms(40, 5)
    rng6 = np.random.default_rng(6)
    cg6 = _traj_ms(40, 6)
    rng7 = np.random.default_rng(7)
    cg7 = _traj_ms(25, 7)
    return [
        ("gauge_0.8", 0.8 * gt, gt),
        ("gauge_1.25", 1.25 * gt, gt),
        ("perp_shift", gt + 0.03 * perp, gt),
        ("far_50mm", gt + 0.05 * u, gt),
        ("near_50mm", gt - 0.05 * u, gt),
        ("nan_frame", nan_pred, gt),
        ("apply_gauge_1.17", poses[:, :3, 3], gt),
        ("pythagorean_s5", 1.09 * cg5 + rng5.normal(0, 0.02, cg5.shape), cg5),
        ("oracle_s6", 0.9 * cg6 + rng6.normal(0, 0.01, cg6.shape), cg6),
        ("expected_s7", 1.05 * cg7 + rng7.normal(0, 0.01, cg7.shape), cg7),
    ]


CASE_NAMES = [c[0] for c in _cases()]


@pytest.mark.parametrize("name", CASE_NAMES)
def test_along_ray_components_equal(name: str) -> None:
    _, pred, gt = next(c for c in _cases() if c[0] == name)
    a, b = alongray.along_ray_components(pred, gt), fm.along_ray_components(pred, gt)
    for x, y in zip(astuple(a), astuple(b), strict=True):
        np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize("name", CASE_NAMES)
def test_gauge_decomposition_equal(name: str) -> None:
    _, pred, gt = next(c for c in _cases() if c[0] == name)
    for sizes in ((None, None), (0.12 * 1.17, 0.12), (0.12, 0.12)):
        a = alongray.gauge_decomposition(pred, gt, *sizes)
        b = fm.gauge_decomposition(pred, gt, *sizes)
        np.testing.assert_array_equal(np.array(astuple(a), dtype=float), np.array(astuple(b), dtype=float))


@pytest.mark.parametrize("name", CASE_NAMES)
@pytest.mark.parametrize("kappa", [1.0, 0.93, 1.2])
def test_e_par_split_equals_e1_split(name: str, kappa: float) -> None:
    _, pred, gt = next(c for c in _cases() if c[0] == name)
    a = alongray.e_par_split(pred, gt, kappa)
    b = fs.e1_split(pred, gt, kappa)
    assert (a.e_par, a.gauge_part, a.off_orbit_part, a.lam, a.z_rms) == (
        b.e1, b.gauge_part, b.offorbit_part, b.lam, b.z_rms)
    assert a.oracle_floor == b.offorbit_part / b.lam


def test_apply_gauge_equal() -> None:
    p = _poses(_traj_mf())
    a, sa = alongray.apply_gauge(p, 0.12, 1.17)
    b, sb = fm.apply_gauge(p, 0.12, 1.17)
    np.testing.assert_array_equal(a, b)
    assert sa == sb


def test_score_object_equals_fastdo_composition() -> None:
    """The one-call API reports exactly e1_split's numbers for the object, from (T, 4, 4) poses."""
    gt = _poses(_traj_mf(), seed=3)
    pred = gt.copy()
    pred[:, :3, 3] = 1.07 * gt[:, :3, 3] + np.random.default_rng(2).normal(0, 0.01, (len(gt), 3))
    b = alongray.score(pred, gt)["object"]
    r = fs.e1_split(pred[:, :3, 3], gt[:, :3, 3])
    assert (b.e_par, b.gauge_part, b.off_orbit_part, b.lam) == (r.e1, r.gauge_part, r.offorbit_part, r.lam)


def test_hand_protocols_equal() -> None:
    rng = np.random.default_rng(10)
    g = rng.normal(0, 0.04, (5, 21, 3)) + np.array([0.0, 0.0, 0.5])
    for pred in (g + 0.1, 1.1 * g, g + rng.normal(0, 0.005, g.shape)):
        np.testing.assert_array_equal(alongray.mpjpe_root_aligned(pred, g), fs.mpjpe_root_aligned(pred, g))
        np.testing.assert_array_equal(alongray.pa_mpjpe(pred, g), fs.pa_mpjpe(pred, g))


def test_object_cd_after_scaled_icp_equal() -> None:
    shape = _box((0.05, 0.08, 0.12), n=1500)
    rng = np.random.default_rng(9)
    gt = [shape @ _rot_ms(i).T + np.array([0.0, 0.0, 0.6]) for i in range(3)]
    pred = [g + rng.normal(0, 0.002, g.shape) for g in gt]
    for p, kw in ((pred, {}), ([1.2 * q for q in pred], {}), ([1.2 * q for q in pred], {"normalise_init": True}),
                  ([1.2 * q for q in pred], {"centre": False})):
        assert alongray.object_cd_after_scaled_icp(p, gt, **kw) == fs.object_cd_after_scaled_icp(p, gt, **kw)


def test_chamfer_and_umeyama_equal() -> None:
    rng = np.random.default_rng(3)
    a, b = rng.normal(size=(300, 3)), rng.normal(size=(250, 3))
    assert alongray.chamfer_cm2(a, b) == fm.hold_chamfer_cm2(a, b)
    src = rng.normal(size=(50, 3))
    dst = 1.3 * src @ _rot_ms(4).T + 0.2
    for x, y in zip(alongray.umeyama_similarity(src, dst), fa.umeyama_similarity(src, dst), strict=True):
        np.testing.assert_array_equal(x, y)


def test_points_and_nan_rows_equal_e1_split() -> None:
    """(T, N, 3) points: fastdo flattens frames x points; alongray takes the flattened (T*N, 3) array. NaN rows are
    dropped by both, with the same count."""
    rng = np.random.default_rng(12)
    g = rng.normal(0, 0.04, (15, 21, 3)) + np.array([0.0, 0.0, 0.55])
    p = 0.95 * g + rng.normal(0, 0.005, g.shape)
    p[3, 4] = np.nan
    a, b = alongray.e_par_split(p.reshape(-1, 3), g.reshape(-1, 3)), fs.e1_split(p, g)
    assert (a.e_par, a.gauge_part, a.off_orbit_part, a.lam, a.z_rms) == (
        b.e1, b.gauge_part, b.offorbit_part, b.lam, b.z_rms)
    assert a.n_frames == fm.gauge_decomposition(p.reshape(-1, 3), g.reshape(-1, 3)).n_frames == 15 * 21 - 1
