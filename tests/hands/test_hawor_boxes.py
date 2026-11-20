"""Tests for `fastdo.hands.hawor_boxes`: top-1 boxes, detection chunks, padded predict chunks, the Kalman replica."""

from __future__ import annotations

import json
import subprocess

import numpy as np
import pytest

from fastdo.hands.hawor_boxes import (
    HAND_CLASS,
    detection_chunks,
    filter_hand_boxes,
    kalman_xywh_filter,
    padded_chunks,
    select_hand_boxes,
)
from tests.hands.hawor_env import HAWOR_PYTHON, cpu_subprocess_env


def _xyxy(xywh: np.ndarray) -> np.ndarray:
    return np.stack([xywh[:, 0] - xywh[:, 2] / 2, xywh[:, 1] - xywh[:, 3] / 2,
                     xywh[:, 0] + xywh[:, 2] / 2, xywh[:, 1] + xywh[:, 3] / 2], axis=1)


def test_select_hand_boxes_top1_per_class() -> None:
    xyxy0 = np.array([[10, 20, 30, 40], [11, 21, 31, 41], [50, 60, 70, 80]], np.float32)
    conf0 = np.array([0.5, 0.9, 0.3], np.float32)
    cls0 = np.array([1.0, 1.0, 0.0], np.float32)
    frame1 = (np.array([[1, 2, 3, 4]], np.float32), np.array([0.15], np.float32), np.array([1.0], np.float32))

    out = select_hand_boxes([(xyxy0, conf0, cls0), frame1])

    assert HAND_CLASS == {"left": 0, "right": 1}
    assert set(out) == {"left", "right"}
    for side in ("left", "right"):
        assert out[side].shape == (2, 5) and out[side].dtype == np.float64
        assert np.isnan(out[side][1]).all()
    np.testing.assert_array_equal(out["right"][0], np.r_[xyxy0[1], conf0[1]].astype(np.float64))
    np.testing.assert_array_equal(out["left"][0], np.r_[xyxy0[2], conf0[2]].astype(np.float64))


def test_detection_chunks_split_at_gaps() -> None:
    boxes = np.full((10, 5), np.nan)
    boxes[[0, 1, 2, 5, 6, 9]] = 1.0

    chunks = detection_chunks(boxes)

    assert [c.tolist() for c in chunks] == [[0, 1, 2], [5, 6], [9]]
    assert all(c.dtype == np.int64 for c in chunks)
    assert detection_chunks(np.full((4, 5), np.nan)) == []


def test_padded_chunks() -> None:
    assert padded_chunks(46, 16) == [(0, 16, 0), (16, 32, 0), (32, 46, 2)]
    assert padded_chunks(32, 16) == [(0, 16, 0), (16, 32, 0)]
    assert padded_chunks(5, 16) == [(0, 5, 11)]


def test_kalman_static_box_is_fixed_point() -> None:
    z = np.tile([320.0, 240.0, 80.0, 100.0], (50, 1))
    out = kalman_xywh_filter(z)
    assert out.shape == (50, 4) and out.dtype == np.float64
    assert np.abs(out - z).max() <= 1e-9


def test_kalman_nan_semantics() -> None:
    box = np.array([200.0, 150.0, 60.0, 90.0])
    z = np.full((13, 4), np.nan)
    z[3:10] = box
    z[11:13] = box

    out = kalman_xywh_filter(z)

    assert np.isnan(out[:3]).all()
    np.testing.assert_array_equal(out[3], box)                   # the first finite row returns itself
    assert np.abs(out[10] - box).max() <= 1e-9                   # a NaN row returns the prediction
    assert np.isfinite(out[3:]).all()


def test_kalman_constant_velocity_converges() -> None:
    u = np.arange(61, dtype=np.float64)
    z = np.stack([100.0 + 2.0 * u, np.full_like(u, 300.0), np.full_like(u, 200.0), np.full_like(u, 200.0)], 1)
    out = kalman_xywh_filter(z)
    assert np.abs(out[30:, 0] - z[30:, 0]).max() <= 0.05


def _seeded_sequence() -> np.ndarray:
    rng = np.random.default_rng(0)
    u = np.arange(40, dtype=np.float64)
    z = np.stack([400.0 + 3.0 * u + rng.normal(0, 2, 40), 300.0 - 1.5 * u + rng.normal(0, 2, 40),
                  120.0 + rng.normal(0, 1, 40), 150.0 + rng.normal(0, 1, 40)], axis=1)
    z[[0, 1, 7, 8, 21]] = np.nan
    return z


_ULTRALYTICS_KF = r"""
import json, sys
import numpy as np
from ultralytics.trackers.utils.kalman_filter import KalmanFilterXYWH
z = np.array(json.loads(sys.stdin.read()), dtype=np.float64)
kf = KalmanFilterXYWH()
out = np.full((len(z), 4), np.nan)
mean, cov = kf.initiate(z[2])
out[2] = mean[:4]
for u in range(3, len(z)):
    mean, cov = kf.predict(mean, cov)
    if np.isfinite(z[u]).all():
        mean, cov = kf.update(mean, cov, z[u])
    out[u] = mean[:4]
print(json.dumps(out.tolist()))
"""


@pytest.mark.skipif(not HAWOR_PYTHON.is_file(), reason="hawor python not available")
def test_kalman_matches_ultralytics() -> None:
    z = _seeded_sequence()
    proc = subprocess.run([str(HAWOR_PYTHON), "-c", _ULTRALYTICS_KF], input=json.dumps(z.tolist()),
                          capture_output=True, text=True, env=cpu_subprocess_env(), timeout=300, check=False)
    assert proc.returncode == 0, proc.stderr[-2000:]
    reference = np.array(json.loads(proc.stdout.strip().splitlines()[-1]), dtype=np.float64)

    ours = kalman_xywh_filter(z)

    assert np.isnan(ours[:2]).all()
    assert np.abs(ours[2:] - reference[2:]).max() <= 1e-6


def _stride3_boxes(rows: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    xywh = np.stack([300 + 5 * np.arange(rows) + rng.normal(0, 3, rows), 200 + rng.normal(0, 3, rows),
                     90 + rng.normal(0, 2, rows), 110 + rng.normal(0, 2, rows)], axis=1)
    boxes = np.concatenate([_xyxy(xywh), rng.uniform(0.3, 0.9, (rows, 1))], axis=1)
    return boxes, np.arange(rows, dtype=np.int64) * 3


def test_filter_hand_boxes_restarts_per_chunk() -> None:
    boxes, frame_idx = _stride3_boxes(12)
    boxes[5] = np.nan

    out = filter_hand_boxes(boxes, frame_idx)
    changed = boxes.copy()
    changed[6:12, :4] += 25.0
    out_changed = filter_hand_boxes(changed, frame_idx)

    assert out.shape == (12, 5) and out.dtype == np.float64
    assert np.isnan(out[5]).all()
    assert np.abs(out[6] - boxes[6]).max() <= 1e-9
    np.testing.assert_array_equal(out_changed[:5], out[:5])
    np.testing.assert_array_equal(out[:, 4][np.isfinite(boxes[:, 4])], boxes[:, 4][np.isfinite(boxes[:, 4])])


def test_filter_hand_boxes_equals_explicit_grid() -> None:
    boxes, frame_idx = _stride3_boxes(10, seed=1)
    x = boxes[:, :4]
    xywh = np.stack([(x[:, 0] + x[:, 2]) / 2, (x[:, 1] + x[:, 3]) / 2, x[:, 2] - x[:, 0], x[:, 3] - x[:, 1]], 1)
    grid = np.zeros((28, 4))
    for k in range(9):
        for step in range(3):
            a = step / 3.0
            grid[3 * k + step] = (1 - a) * xywh[k] + a * xywh[k + 1]
    grid[27] = xywh[9]

    expected = _xyxy(kalman_xywh_filter(grid)[0::3])
    out = filter_hand_boxes(boxes, frame_idx)

    assert np.abs(out[:, :4] - expected).max() <= 1e-9
    np.testing.assert_array_equal(out[:, 4], boxes[:, 4])
