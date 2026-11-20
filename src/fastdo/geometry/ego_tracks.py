"""Background correspondences for the ego-motion estimate (`fastdo.geometry.ego_motion` has the model).

`exclusion_masks` marks, per decode row and at the tracker's working resolution, the pixels that are NOT static
scene: the object mask (dilated) and each valid hand's image region (the box of its projected joints, padded, and
extended to the bottom image edge, where an egocentric forearm enters). What it misses -- other people, a moving
vehicle, a forearm entering from the side -- is left to the robust registration.

`plan_correspondences` detects Shi-Tomasi corners outside the exclusion on each row i and tracks them with pyramidal
Lucas-Kanade to rows i+k (k in `cfg.skips`), keeping a pair only when the backward track returns within
`cfg.fb_max_px` and the target also lies outside row j's exclusion. Pixels are mapped back to full resolution (nearest
pixel), where the pointmaps are sampled: `EgoPlan.frame/v/u` is the flat list of pointmap samples the plan needs, and
`EgoPlan.edge_i/edge_j` index it per edge. The geometry stage gathers exactly those samples on the device
(`sample_rows`), so the ego-motion needs no full-resolution pointmap on the host.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import cv2
import numpy as np

from fastdo.geometry.ego_motion import EgoMotion, EgoMotionConfig, solve_ego_motion

__all__ = ["EgoPlan", "exclusion_masks", "plan_correspondences", "solve_from_samples", "working_size"]


def working_size(height: int, width: int, long_side: int) -> tuple[int, int, float]:
    """`(h, w, factor)`: the tracker's frame size (long side at most `long_side`) and full/working scale."""
    factor = max(height, width) / float(long_side)
    if factor <= 1.0:
        return height, width, 1.0
    return max(1, round(height / factor)), max(1, round(width / factor)), factor


def exclusion_masks(n_rows: int, height: int, width: int, cfg: EgoMotionConfig,
                    object_masks: np.ndarray | None = None,
                    joints: Mapping[str, np.ndarray] | None = None,
                    joint_valid: Mapping[str, np.ndarray] | None = None,
                    K: np.ndarray | None = None) -> np.ndarray:
    """(N,h,w) bool at working resolution: True = not static scene (module docstring).

    Args:
        object_masks: (N,H,W) bool full-resolution object masks, or None.
        joints: side -> (N,J,3) camera-frame hand joints (metres), or None.
        joint_valid: side -> (N,) bool.
        K: (3,3) pixel intrinsics of the full-resolution frame (needed with `joints`).
    """
    h, w, factor = working_size(height, width, cfg.long_side)
    out = np.zeros((n_rows, h, w), bool)
    diag = float(np.hypot(h, w))
    if object_masks is not None:
        object_masks = np.asarray(object_masks, bool)
        if object_masks.shape != (n_rows, height, width):
            raise ValueError(f"object_masks must be {(n_rows, height, width)}, got {object_masks.shape}")
        pad = max(1, round(cfg.object_pad_frac * diag))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad + 1, 2 * pad + 1))
        for t in range(n_rows):
            small = cv2.resize(object_masks[t].astype(np.uint8), (w, h), interpolation=cv2.INTER_AREA) > 0
            out[t] |= cv2.dilate(small.astype(np.uint8), kernel) > 0
    if joints is not None:
        if K is None or joint_valid is None:
            raise ValueError("hand exclusion needs K and joint_valid")
        fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
        for side, jts in joints.items():
            jts = np.asarray(jts, float)
            ok_rows = np.asarray(joint_valid[side], bool)
            for t in range(n_rows):
                if not ok_rows[t]:
                    continue
                p = jts[t]
                p = p[np.isfinite(p).all(1) & (p[:, 2] > 1e-3)]
                if len(p) == 0:
                    continue
                u = (fx * p[:, 0] / p[:, 2] + cx) / factor
                v = (fy * p[:, 1] / p[:, 2] + cy) / factor
                u0, u1, v0, v1 = u.min(), u.max(), v.min(), v.max()
                pad_u = cfg.hand_pad_frac * max(u1 - u0, 1.0)
                pad_v = cfg.hand_pad_frac * max(v1 - v0, 1.0)
                a0 = int(np.clip(np.floor(u0 - pad_u), 0, w))
                a1 = int(np.clip(np.ceil(u1 + pad_u), 0, w))
                b0 = int(np.clip(np.floor(v0 - pad_v), 0, h))
                if a1 > a0 and b0 < h:
                    out[t, b0:, a0:a1] = True           # the hand's box, down to the bottom edge (forearm)
    return out


@dataclass(frozen=True)
class EgoPlan:
    """The correspondences of one window and the pointmap samples they need.

    Attributes:
        n_rows: Decode rows N.
        height, width: Full-resolution frame size.
        edges: (E,2) int64 (i, j).
        frame, v, u: (A,) int64 flat list of full-resolution pointmap samples (row, pixel row, pixel column).
        edge_i, edge_j: Per edge, (P_e,) int64 indices into the flat list for rows i and j.
        n_corners: (N,) corners detected per row.
    """

    n_rows: int
    height: int
    width: int
    edges: np.ndarray
    frame: np.ndarray
    v: np.ndarray
    u: np.ndarray
    edge_i: list[np.ndarray]
    edge_j: list[np.ndarray]
    n_corners: np.ndarray

    def samples_of_rows(self, first: int, end: int) -> np.ndarray:
        """Indices of the flat samples on rows [first, end)."""
        return np.nonzero((self.frame >= first) & (self.frame < end))[0]


def _grey(frame: np.ndarray, h: int, w: int) -> np.ndarray:
    g = cv2.cvtColor(np.ascontiguousarray(frame), cv2.COLOR_RGB2GRAY)
    if g.shape != (h, w):
        g = cv2.resize(g, (w, h), interpolation=cv2.INTER_AREA)
    return g


def plan_correspondences(frames: np.ndarray, exclude: np.ndarray, cfg: EgoMotionConfig) -> EgoPlan:
    """Tracked background pixel pairs for every edge (i, i+k) (module docstring).

    Args:
        frames: (N,H,W,3) uint8 RGB decode rows.
        exclude: (N,h,w) bool from `exclusion_masks` (working resolution).
        cfg: The settings.
    """
    frames = np.asarray(frames)
    n, height, width = int(frames.shape[0]), int(frames.shape[1]), int(frames.shape[2])
    h, w, factor = working_size(height, width, cfg.long_side)
    if exclude.shape != (n, h, w):
        raise ValueError(f"exclude must be {(n, h, w)}, got {exclude.shape}")
    grey = [_grey(frames[t], h, w) for t in range(n)]
    lk = {"winSize": (21, 21), "maxLevel": 3,
          "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)}
    corners: list[np.ndarray] = []
    for t in range(n):
        allowed = (~exclude[t]).astype(np.uint8) * 255
        pts = cv2.goodFeaturesToTrack(grey[t], maxCorners=cfg.max_corners, qualityLevel=cfg.corner_quality,
                                      minDistance=cfg.min_distance_px, mask=allowed)
        corners.append(np.zeros((0, 2), np.float32) if pts is None else pts.reshape(-1, 2).astype(np.float32))
    edges: list[tuple[int, int]] = []
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(n):
        p0 = corners[i]
        for k in cfg.skips:
            j = i + k
            if j >= n:
                continue
            edges.append((i, j))
            if len(p0) == 0:
                pairs.append((np.zeros((0, 2)), np.zeros((0, 2))))
                continue
            p1, st1, _ = cv2.calcOpticalFlowPyrLK(grey[i], grey[j], p0.reshape(-1, 1, 2), None, **lk)
            pb, st2, _ = cv2.calcOpticalFlowPyrLK(grey[j], grey[i], p1, None, **lk)
            p1, pb = p1.reshape(-1, 2), pb.reshape(-1, 2)
            fb = np.linalg.norm(pb - p0, axis=1)
            ok = (st1.reshape(-1) == 1) & (st2.reshape(-1) == 1) & (fb < cfg.fb_max_px)
            ok &= (p1[:, 0] >= 0) & (p1[:, 0] <= w - 1) & (p1[:, 1] >= 0) & (p1[:, 1] <= h - 1)
            if ok.any():
                ui = np.clip(np.round(p1[:, 0]).astype(int), 0, w - 1)
                vi = np.clip(np.round(p1[:, 1]).astype(int), 0, h - 1)
                ok &= ~exclude[j][vi, ui]
            pairs.append((p0[ok].astype(float), p1[ok].astype(float)))

    def full(p: np.ndarray, size: int) -> np.ndarray:
        return np.clip(np.round((p + 0.5) * factor - 0.5).astype(np.int64), 0, size - 1)

    frame_l, v_l, u_l, e_i, e_j = [], [], [], [], []
    at = 0
    for (i, j), (a, b) in zip(edges, pairs, strict=True):
        m = len(a)
        for row, p in ((i, a), (j, b)):
            frame_l.append(np.full(m, row, np.int64))
            u_l.append(full(p[:, 0], width) if m else np.zeros(0, np.int64))
            v_l.append(full(p[:, 1], height) if m else np.zeros(0, np.int64))
        e_i.append(np.arange(at, at + m, dtype=np.int64))
        e_j.append(np.arange(at + m, at + 2 * m, dtype=np.int64))
        at += 2 * m
    cat = (lambda xs: np.concatenate(xs) if xs else np.zeros(0, np.int64))  # noqa: E731
    return EgoPlan(n, height, width, np.asarray(edges, np.int64).reshape(-1, 2), cat(frame_l), cat(v_l), cat(u_l),
                   e_i, e_j, np.array([len(c) for c in corners], np.int64))


def solve_from_samples(plan: EgoPlan, samples: np.ndarray, cfg: EgoMotionConfig) -> EgoMotion:
    """`solve_ego_motion` on the plan's pointmap samples ((A,3) metres, in the plan's flat order)."""
    samples = np.asarray(samples, dtype=np.float64)
    if samples.shape != (len(plan.frame), 3):
        raise ValueError(f"samples must be {(len(plan.frame), 3)}, got {samples.shape}")
    xs = [samples[ix] for ix in plan.edge_i]
    ys = [samples[ix] for ix in plan.edge_j]
    return solve_ego_motion(plan.n_rows, plan.edges, xs, ys, cfg)
