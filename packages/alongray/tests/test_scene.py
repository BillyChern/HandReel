# Copyright 2026 The alongray authors
# SPDX-License-Identifier: Apache-2.0
"""Per-body, shared and relative gauges of several bodies (alongray.scene)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alongray import scene_gauge


def _traj(n: int = 30, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.normal(0, 0.05, n), rng.normal(0, 0.05, n), rng.uniform(0.45, 0.75, n)])


def test_shared_gauge_removes_a_common_scale() -> None:
    c, o = _traj(seed=1), _traj(seed=2)
    s = scene_gauge({"hand": (1.2 * c, c), "object": (1.2 * o, o)})
    assert s.shared_lam == pytest.approx(1.2, rel=1e-12)
    assert s.relative_log_gauge["hand/object"] == pytest.approx(0.0, abs=1e-12)
    assert max(s.residual_after_shared_scale.values()) == pytest.approx(0.0, abs=1e-12)


def test_residual_after_shared_scale_known_answer() -> None:
    """Bodies on gauges 1.2 and 0.8 at clearly different ranges (the object twice as far): dividing every prediction
    by the ONE least-squares scale leaves each body at (lam_b / lam_s - 1) times its truth, all along its rays."""
    c, o = _traj(seed=3), 2.0 * _traj(seed=4)
    s = scene_gauge({"hand": (1.2 * c, c), "object": (0.8 * o, o)})
    lam_s = (1.2 * np.sum(c * c) + 0.8 * np.sum(o * o)) / (np.sum(c * c) + np.sum(o * o))
    assert s.shared_lam == pytest.approx(lam_s, rel=1e-12)
    assert abs(lam_s - 1.0) > 0.1                       # far from 1: the residuals are not near-zero by accident
    assert s.relative_log_gauge["hand/object"] == pytest.approx(math.log(1.5), rel=1e-12)
    for name, lam_b, g in (("hand", 1.2, c), ("object", 0.8, o)):
        want = abs(lam_b / lam_s - 1.0) * math.sqrt(float(np.mean(np.sum(g * g, axis=1))))
        assert s.residual_after_shared_scale[name] == pytest.approx(want, rel=1e-12)
        assert s.shared_off_orbit[name] == pytest.approx(lam_s * want, rel=1e-12)
    assert s.lam == pytest.approx({"hand": 1.2, "object": 0.8})


def test_joints_use_the_root_and_empty_is_refused() -> None:
    c = _traj(seed=5)
    j = c[:, None, :] + np.random.default_rng(0).normal(scale=0.03, size=(len(c), 21, 3))
    s = scene_gauge({"object": (1.1 * c, c), "hand": (1.1 * j, j)}, hand_root=0)
    assert s.lam["hand"] == pytest.approx(1.1, rel=1e-12)
    with pytest.raises(ValueError):
        scene_gauge({})
