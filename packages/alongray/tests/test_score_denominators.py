# Copyright 2026 The alongray authors
# SPDX-License-Identifier: Apache-2.0
"""Every metric of a BodyScore carries the number of frames it was computed on (A5).

The centre error uses the frames whose reference point is finite in both prediction and truth; root-aligned MPJPE
and PA-MPJPE use the frames whose every joint is finite; the Chamfer after scaled ICP uses the frames whose clouds
are finite. These sets differ, so a single `n_frames` cannot describe them.
"""

from __future__ import annotations

import numpy as np
import pytest

from alongray import score


def _hand(t: int = 10, j: int = 21, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    wrist = np.column_stack([rng.normal(0, 0.05, t), rng.normal(0, 0.05, t), rng.uniform(0.45, 0.75, t)])
    return wrist[:, None, :] + 0.03 * rng.standard_normal((t, j, 3))


def test_each_metric_reports_its_own_frame_count() -> None:
    gt = _hand()
    pred = 1.05 * gt + 0.002
    pred[3, 7] = np.nan                       # a finger joint: the centre (wrist) frame survives, MPJPE frames do not
    pred[5, 0] = np.nan                       # the wrist: every metric loses frame 5
    obj_gt = gt[:, 0] + [0.0, 0.0, 0.1]
    res = score(1.05 * obj_gt, obj_gt, pred, gt)
    h = res["hand"]
    assert h.n_frames_input == 10 and h.n_frames == 9
    assert h.aligned_n_frames == {"centre_rms_after_lam": 9, "centre_ate_sim3": 9, "mpjpe_root_aligned": 8,
                                  "pa_mpjpe": 8}
    assert res["object"].aligned_n_frames == {"centre_rms_after_lam": 10, "centre_ate_sim3": 10}
    assert set(h.as_dict()) >= {"aligned_n_frames", "n_frames_input"}


def test_the_chamfer_after_scaled_icp_has_its_own_denominator() -> None:
    rng = np.random.default_rng(1)
    gt = _hand()[:, 0] + [0.0, 0.0, 0.1]
    cloud = rng.normal(scale=0.03, size=(40, 3))
    gpts = [c + cloud for c in gt]
    ppts = [1.1 * p for p in gpts]
    ppts[2] = np.full_like(ppts[2], np.nan)
    o = score(1.1 * gt, gt, pred_object_points=ppts, gt_object_points=gpts)["object"]
    assert o.aligned_n_frames["chamfer_cm2_after_scaled_icp"] == 9 and o.n_frames == 10
    none = score(1.1 * gt, gt, pred_object_points=[np.full((40, 3), np.nan)] * 10, gt_object_points=gpts)["object"]
    assert none.aligned_n_frames["chamfer_cm2_after_scaled_icp"] == 0
    assert np.isnan(none.aligned["chamfer_cm2_after_scaled_icp"])


def test_a_metric_with_no_finite_frame_reports_zero_and_nan() -> None:
    gt = _hand()
    pred = gt.copy()
    pred[:, 5] = np.nan                       # one joint missing everywhere: no frame for the joint metrics
    h = score(gt[:, 0] + 0.1, gt[:, 0] + 0.1, pred, gt)["hand"]
    assert h.aligned_n_frames["mpjpe_root_aligned"] == 0 and np.isnan(h.aligned["mpjpe_root_aligned"])
    assert h.aligned_n_frames["pa_mpjpe"] == 0 and h.n_frames == 10
    assert h.aligned["centre_rms_after_lam"] == pytest.approx(0.0, abs=1e-12)
