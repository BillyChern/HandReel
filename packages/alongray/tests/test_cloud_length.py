# Copyright 2026 The alongray authors. SPDX-License-Identifier: Apache-2.0
"""Object clouds must have one cloud per trajectory frame, not merely as many as each other."""

from __future__ import annotations

import numpy as np
import pytest

import alongray


def _traj(T: int) -> np.ndarray:
    return np.column_stack([np.linspace(-0.05, 0.05, T), np.zeros(T), np.linspace(0.5, 0.7, T)])


def test_cloud_lists_shorter_than_the_trajectory_are_refused():
    T = 6
    g = _traj(T)
    cloud = [np.random.default_rng(t).normal(0, 0.02, (20, 3)) + g[t] for t in range(T)]
    with pytest.raises(ValueError, match="one cloud per frame"):
        alongray.score(1.05 * g, g, pred_object_points=cloud[:4], gt_object_points=cloud[:4])


def test_cloud_lists_of_trajectory_length_are_scored():
    T = 6
    g = _traj(T)
    cloud = [np.random.default_rng(t).normal(0, 0.02, (20, 3)) + g[t] for t in range(T)]
    s = alongray.score(1.05 * g, g, pred_object_points=cloud, gt_object_points=cloud)
    assert s["object"].aligned_n_frames["chamfer_cm2_after_scaled_icp"] == T
