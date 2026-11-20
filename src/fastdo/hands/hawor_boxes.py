"""Per-frame hand boxes for batched HaWoR (D3b): top-1 detections, fixed-size predict chunks and a Kalman box filter.

Shared module: the hawor env's driver loads this file by path (Python 3.10, numpy 1.26), and the uv env imports it. It
imports only numpy and the standard library, and nothing from `fastdo` or the other shared modules.

HaWoR tracks hands with ultralytics BoT-SORT at the source frame rate and crops the tracker's Kalman state. At 10 fps
the same boxes come from this pipeline:
1. raw detections, top-1 per class (`select_hand_boxes`);
2. maximal runs of detected rows (`detection_chunks`);
3. per run, the box centre and size linearly interpolated onto the source frame grid and filtered by a constant-velocity
   Kalman filter restarted per run (`filter_hand_boxes`, `kalman_xywh_filter`).

The filter implements the standard predict/update equations with the noise model of ultralytics' `KalmanFilterXYWH`:
position and velocity standard deviations of 1/20 and 1/160 of the box width or height.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = [
    "HAND_CLASS",
    "SIGMA_POSITION",
    "SIGMA_VELOCITY",
    "detection_chunks",
    "filter_hand_boxes",
    "kalman_xywh_filter",
    "padded_chunks",
    "select_hand_boxes",
]

HAND_CLASS: dict[str, int] = {"left": 0, "right": 1}
SIGMA_POSITION = 1.0 / 20.0
SIGMA_VELOCITY = 1.0 / 160.0

_F = np.eye(8)
_F[:4, 4:] = np.eye(4)
_H = np.eye(4, 8)


def select_hand_boxes(detections: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
                      min_conf: float = 0.2) -> dict[str, np.ndarray]:
    """Top-1 box per hand class per frame.

    Args:
        detections: One `(xyxy (K,4), conf (K,), cls (K,))` per frame.
        min_conf: Boxes below this confidence are ignored.

    Returns:
        Per side (`HAND_CLASS` keys), `(T,5)` float64 `[x1,y1,x2,y2,conf]`: the highest-confidence box of that class with
        `conf >= min_conf`, or a NaN row when there is none.
    """
    t = len(detections)
    out = {side: np.full((t, 5), np.nan, dtype=np.float64) for side in HAND_CLASS}
    for row, (xyxy, conf, cls) in enumerate(detections):
        boxes = np.asarray(xyxy, dtype=np.float64).reshape(-1, 4)
        scores = np.asarray(conf, dtype=np.float64).reshape(-1)
        labels = np.rint(np.asarray(cls, dtype=np.float64).reshape(-1)).astype(np.int64)
        for side, label in HAND_CLASS.items():
            candidates = np.flatnonzero((labels == label) & (scores >= min_conf))
            if candidates.size:
                best = int(candidates[np.argmax(scores[candidates])])
                out[side][row, :4] = boxes[best]
                out[side][row, 4] = scores[best]
    return out


def detection_chunks(boxes: np.ndarray) -> list[np.ndarray]:
    """int64 row indices of the maximal runs of consecutive rows whose values are all finite."""
    arr = np.asarray(boxes, dtype=np.float64)
    rows = np.flatnonzero(np.isfinite(arr.reshape(arr.shape[0], -1)).all(axis=1)).astype(np.int64)
    if rows.size == 0:
        return []
    return [run.astype(np.int64) for run in np.split(rows, np.flatnonzero(np.diff(rows) != 1) + 1)]


def padded_chunks(n_frames: int, chunk: int) -> list[tuple[int, int, int]]:
    """`(start, stop, n_pad)` per predict call: `start = 0, chunk, ...`, `stop = min(start + chunk, n_frames)`, and
    `n_pad = chunk - (stop - start)` copies of the last frame complete the call.

    Raises:
        ValueError: If `chunk < 1` or `n_frames < 0`.
    """
    if chunk < 1 or n_frames < 0:
        raise ValueError(f"need chunk >= 1 and n_frames >= 0, got chunk={chunk}, n_frames={n_frames}")
    return [(start, min(start + chunk, n_frames), chunk - (min(start + chunk, n_frames) - start))
            for start in range(0, n_frames, chunk)]


def _position_std(w: float, h: float, weight: float) -> np.ndarray:
    return np.array([weight * w, weight * h, weight * w, weight * h])


def kalman_xywh_filter(measurements: np.ndarray) -> np.ndarray:
    """Constant-velocity Kalman filter over `(U,4)` `cx, cy, w, h` measurements on a unit time grid.

    Returns `(U,4)` float64: NaN before the first finite row; the first finite row itself; the updated mean at later
    finite rows; the predicted mean at later NaN rows.

    Raises:
        ValueError: If `measurements` is not `(U,4)`.
    """
    z = np.asarray(measurements, dtype=np.float64)
    if z.ndim != 2 or z.shape[1] != 4:
        raise ValueError(f"measurements must be (U,4), got {z.shape}")
    out = np.full(z.shape, np.nan)
    finite = np.isfinite(z).all(axis=1)
    measured = np.flatnonzero(finite)
    if measured.size == 0:
        return out
    first = int(measured[0])
    w, h = z[first, 2], z[first, 3]
    mean = np.concatenate([z[first], np.zeros(4)])
    cov = np.diag(np.square(np.concatenate([_position_std(w, h, 2 * SIGMA_POSITION),
                                            _position_std(w, h, 10 * SIGMA_VELOCITY)])))
    out[first] = z[first]
    for u in range(first + 1, z.shape[0]):
        w, h = mean[2], mean[3]
        process = np.diag(np.square(np.concatenate([_position_std(w, h, SIGMA_POSITION),
                                                    _position_std(w, h, SIGMA_VELOCITY)])))
        mean = _F @ mean
        cov = _F @ cov @ _F.T + process
        if finite[u]:
            innovation_cov = _H @ cov @ _H.T + np.diag(np.square(_position_std(mean[2], mean[3], SIGMA_POSITION)))
            gain = np.linalg.solve(innovation_cov, _H @ cov).T          # P H^T S^-1 (P and S are symmetric)
            mean = mean + gain @ (z[u] - _H @ mean)
            cov = cov - gain @ innovation_cov @ gain.T
        out[u] = mean[:4]
    return out


def filter_hand_boxes(boxes: np.ndarray, frame_idx: np.ndarray) -> np.ndarray:
    """Kalman-filtered `(T,5)` boxes, one filter per detection chunk on that chunk's source frame grid.

    For each chunk, the centre and size of its rows are placed at `u = frame_idx[r] - frame_idx[chunk[0]]`, linearly
    interpolated at the source frames in between, filtered, and read back at the chunk's rows as xyxy; `conf` is copied.
    NaN rows stay NaN.

    Raises:
        ValueError: If `boxes` is not `(T,5)`, or `frame_idx` is not `(T,)` and strictly increasing.
    """
    arr = np.asarray(boxes, dtype=np.float64)
    idx = np.asarray(frame_idx, dtype=np.int64).reshape(-1)
    if arr.ndim != 2 or arr.shape[1] != 5 or idx.shape != (arr.shape[0],):
        raise ValueError(f"boxes must be (T,5) with frame_idx (T,), got {arr.shape} and {idx.shape}")
    if idx.size > 1 and not bool((np.diff(idx) > 0).all()):
        raise ValueError("frame_idx must be strictly increasing")
    xywh = np.stack([(arr[:, 0] + arr[:, 2]) / 2, (arr[:, 1] + arr[:, 3]) / 2, arr[:, 2] - arr[:, 0],
                     arr[:, 3] - arr[:, 1]], axis=1)
    out = np.full(arr.shape, np.nan)
    for chunk in detection_chunks(arr):
        u = idx[chunk] - idx[chunk[0]]
        grid_u = np.arange(int(u[-1]) + 1, dtype=np.float64)
        grid = np.stack([np.interp(grid_u, u.astype(np.float64), xywh[chunk, c]) for c in range(4)], axis=1)
        filtered = kalman_xywh_filter(grid)[u]
        out[chunk, 0] = filtered[:, 0] - filtered[:, 2] / 2
        out[chunk, 1] = filtered[:, 1] - filtered[:, 3] / 2
        out[chunk, 2] = filtered[:, 0] + filtered[:, 2] / 2
        out[chunk, 3] = filtered[:, 1] + filtered[:, 3] / 2
        out[chunk, 4] = arr[chunk, 4]
    return out
