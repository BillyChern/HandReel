# Copyright 2026 The alongray authors
# SPDX-License-Identifier: Apache-2.0
"""The constant-shift reading (shift_part, shift_scale_part, collinearity) and the few-frames flag low_n."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alongray import MIN_ROWS, e_par_split, score


def _traj(n: int = 30, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.normal(0, 0.05, n), rng.normal(0, 0.05, n), rng.uniform(0.45, 0.75, n)])


def _shift(g: np.ndarray, d: float) -> np.ndarray:
    return g + d * g / np.linalg.norm(g, axis=1, keepdims=True)


def test_pure_shift_puts_collinearity_times_its_mean_square_in_the_gauge_part() -> None:
    g = _traj(seed=1)
    sp = e_par_split(_shift(g, 0.03), g)
    z = np.linalg.norm(g, axis=1)
    c = float(np.mean(z) ** 2 / np.mean(z * z))
    assert sp.collinearity == pytest.approx(c, rel=1e-14)
    assert sp.shift_part == pytest.approx(0.03, rel=1e-12)
    assert sp.gauge_part ** 2 == pytest.approx(c * 0.03 ** 2, rel=1e-10)
    assert sp.shift_scale_part == pytest.approx(sp.e_par, rel=1e-12)


def test_flat_range_makes_shift_and_scale_indistinguishable() -> None:
    g = _traj(seed=2)
    g[:, 2] = 0.6 + 0.001 * np.arange(len(g)) / len(g)
    sp = e_par_split(_shift(g, 0.02), g)
    assert sp.collinearity > 0.999 and sp.gauge_part ** 2 / sp.e_par ** 2 > 0.999


def test_wide_range_separates_them() -> None:
    g = _traj(seed=4)
    g[:, 2] = np.linspace(0.3, 1.5, len(g))
    sp = e_par_split(_shift(g, 0.02), g)
    assert sp.collinearity < 0.9 and sp.gauge_part ** 2 / sp.e_par ** 2 < 0.9


def test_pure_scale_has_zero_unique_shift_share() -> None:
    g = _traj(seed=5)
    sp = e_par_split(1.2 * g, g)
    assert sp.shift_scale_part == pytest.approx(sp.gauge_part, rel=1e-12)
    assert sp.shift_part ** 2 == pytest.approx(sp.collinearity * sp.gauge_part ** 2, rel=1e-10)


def test_constant_range_is_defined() -> None:
    g = np.tile([[0.0, 0.0, 0.7]], (10, 1))
    sp = e_par_split(1.1 * g, g)
    assert sp.collinearity == pytest.approx(1.0) and sp.shift_scale_part == pytest.approx(sp.e_par, rel=1e-12)


def test_low_n_flag_and_warning() -> None:
    g = _traj(n=5)
    rng = np.random.default_rng(0)
    p = 1.2 * g + rng.normal(scale=0.02, size=g.shape)
    for n in (1, 2):
        with pytest.warns(UserWarning, match="low_n"):
            sp = e_par_split(p[:n], g[:n])
        assert sp.low_n and MIN_ROWS == 3
    with pytest.warns(UserWarning, match="low_n"):
        one = e_par_split(p[:1], g[:1])
    assert one.off_orbit_part == pytest.approx(0.0, abs=1e-15) and one.e_par == pytest.approx(one.gauge_part)
    assert not e_par_split(p[:3], g[:3]).low_n


def test_body_score_carries_the_new_fields() -> None:
    g = _traj(seed=7)
    b = score(_shift(g, 0.02), g)["object"]
    sp = e_par_split(_shift(g, 0.02), g)
    for k in ("shift_part", "shift_scale_part", "collinearity", "low_n"):
        assert getattr(b, k) == getattr(sp, k)
    assert not math.isnan(b.collinearity)
