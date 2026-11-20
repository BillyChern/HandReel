# Copyright 2026 The alongray authors. SPDX-License-Identifier: Apache-2.0
"""The no-GT scale-gauge illustration (observability education, not error estimation). The projection used for the
expected values is written here independently of the module. The tests write no files."""

from __future__ import annotations

import numpy as np
import pytest

from alongray import gauge_demo as GD

K = np.array([[600.0, 0.0, 320.0], [0.0, 610.0, 240.0], [0.0, 0.0, 1.0]])


def _project(K, X):
    """Independent float64 pinhole projection: (u, v) = (K X)[:2] / (K X)[2]."""
    h = np.einsum("ij,nj->ni", K, X)
    return h[:, :2] / h[:, 2:3]


def _scene(n=40, seed=7):
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.uniform(-0.3, 0.3, n), rng.uniform(-0.2, 0.2, n), rng.uniform(0.4, 1.2, n)])


@pytest.mark.parametrize("lam", [0.5, 2.0])
def test_coherently_scaled_scenes_project_identically(lam):
    X = _scene()
    r = GD.scaled_scene_pair(X, lam, K)
    assert np.array_equal(r["scaled_points"], lam * X)
    d = np.abs(_project(K, X) - _project(K, lam * X)).max()
    assert d <= 1e-8 and r["max_pixel_difference"] <= 1e-8
    assert r["metric_depth_ratio"] == pytest.approx(lam, abs=1e-15)
    assert "not error estimation" in r["label"] and "Triggs" in r["reference"]


def test_near_zero_or_negative_depth_and_bad_lambda_are_refused():
    X = _scene()
    X[3, 2] = 1e-9
    with pytest.raises(ValueError, match="depth"):
        GD.scaled_scene_pair(X, 2.0, K)
    X[3, 2] = -0.5
    with pytest.raises(ValueError, match="depth"):
        GD.scaled_scene_pair(X, 2.0, K)
    for lam in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError, match="lam"):
            GD.scaled_scene_pair(_scene(), lam, K)
    with pytest.raises(ValueError, match="depth"):
        GD.scaled_scene_pair(_scene(), 1e-9, K)              # scaled depths near zero


def test_negative_control_a_fixed_metric_hand_anchor_breaks_the_gauge():
    """A hand of fixed metric size cannot follow the scaling: placed at the scaled depth it projects 1/lam as large."""
    X = _scene()
    hand = np.array([[0.0, 0.0, 0.0], [0.08, 0.0, 0.0], [0.0, 0.09, 0.0]]) + np.array([0.05, 0.02, 0.6])
    for lam in (0.5, 2.0):
        c = GD.fixed_anchor_control(X, hand, lam, K)
        assert c["scene_max_pixel_difference"] <= 1e-8
        assert c["anchor_max_pixel_difference"] > 1.0
        # independent check: the anchor's centroid moves to lam times its depth with its metric shape unchanged
        moved = hand - hand.mean(0) + lam * hand.mean(0)
        ref = np.abs(_project(K, hand) - _project(K, moved)).max()
        assert c["anchor_max_pixel_difference"] == pytest.approx(ref, rel=1e-12)


def test_negative_control_a_fixed_second_camera_baseline_breaks_the_gauge():
    X = _scene()
    for lam in (0.5, 2.0):
        c = GD.second_camera_control(X, lam, K, baseline=np.array([0.1, 0.0, 0.0]))
        assert c["camera1_max_pixel_difference"] <= 1e-8
        ref = np.abs(_project(K, X - [0.1, 0, 0]) - _project(K, lam * X - [0.1, 0, 0])).max()
        assert c["camera2_max_pixel_difference"] == pytest.approx(ref, rel=1e-12) and ref > 1.0
    z = GD.second_camera_control(X, 2.0, K, baseline=np.zeros(3))
    assert z["camera2_max_pixel_difference"] <= 1e-8          # a zero baseline is the positive control
