"""Contact-state segmentation of the reference conditioner.

Per hand and row: CONTACT if the hand-object distance d <= d_on (15 mm), FAR if d > d_off (100 mm), NEAR in
between, UNKNOWN on rows where the hand is not valid (or infilled) or the object pose is missing. An UNKNOWN hand is
certified FAR only if the dataset declares it absent, or if every valid row of that hand within `bridge_s` before and
after is FAR with at least one such row on each side. A never-seen hand that the dataset does NOT declare absent is
`unknown_hand` on every row.

Row states, by priority: `no_pose` > `touched` > `guard` (within `guard_rows` of a touched row) > `near` >
`unknown_hand` > `rest` (eligible). Rest segments are maximal runs of consecutive `rest` rows; any other row breaks a
run, so no segment spans a touched span.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fastdo.reference.config import ReferenceConfig
from fastdo.robot.kinematic_rows import hand_object_distance

__all__ = [
    "CONTACT",
    "FAR",
    "NEAR",
    "UNKNOWN",
    "HandInput",
    "hand_distance",
    "hand_status",
    "rest_segments",
    "row_states",
    "segment_window",
]

CONTACT, NEAR, FAR, UNKNOWN = 0, 1, 2, 3
STATES = ("no_pose", "touched", "guard", "near", "unknown_hand", "rest")


@dataclass(frozen=True)
class HandInput:
    vertices: np.ndarray | None   # (T,778,3) in the conditioning (ego-scaled camera) frame, or None
    valid: np.ndarray             # (T,) bool: valid and not infilled
    declared_absent: bool = False  # the DATASET says this hand is not in the scene
    distance: np.ndarray | None = None  # optional precomputed (T,) distances (m, NaN where not computed)


def hand_distance(hand: HandInput, poses_scaled: np.ndarray, pose_ok: np.ndarray, mesh_vertices: np.ndarray,
                  mesh_faces: np.ndarray, scale: float, cfg: ReferenceConfig) -> np.ndarray:
    """`(T,)` distance (m) on rows where the hand is valid and the object pose finite; NaN elsewhere."""
    rows = np.asarray(hand.valid, dtype=bool) & pose_ok
    if hand.distance is not None:
        d = np.asarray(hand.distance, dtype=np.float64).copy()
        d[~rows] = np.nan
        return d
    if hand.vertices is None or not rows.any():
        return np.full(len(rows), np.nan)
    return hand_object_distance(hand.vertices, poses_scaled, mesh_vertices, mesh_faces, scale, rows, cfg.vertex_stride)


def hand_status(dist: np.ndarray, times: np.ndarray, declared_absent: bool, cfg: ReferenceConfig) -> np.ndarray:
    """`(T,)` CONTACT / NEAR / FAR / UNKNOWN with the certification of UNKNOWN rows (module docstring)."""
    d = np.asarray(dist, dtype=np.float64)
    known = np.isfinite(d)
    st = np.full(len(d), UNKNOWN, dtype=np.int64)
    st[known & (d <= cfg.contact_m)] = CONTACT
    st[known & (d > cfg.contact_m) & (d <= cfg.release_m)] = NEAR
    st[known & (d > cfg.release_m)] = FAR
    if declared_absent:
        st[~known] = FAR
        return st
    t = np.asarray(times, dtype=np.float64)
    vrows = np.flatnonzero(known)
    for i in np.flatnonzero(~known):
        before = vrows[(t[vrows] >= t[i] - cfg.bridge_s - 1e-9) & (vrows < i)]
        after = vrows[(t[vrows] <= t[i] + cfg.bridge_s + 1e-9) & (vrows > i)]
        if before.size and after.size and np.all(st[before] == FAR) and np.all(st[after] == FAR):
            st[i] = FAR
    return st


def row_states(statuses: list[np.ndarray], pose_ok: np.ndarray, cfg: ReferenceConfig) -> np.ndarray:
    """`(T,)` state names (module docstring) from the per-hand statuses."""
    n = len(pose_ok)
    S = np.stack(statuses) if statuses else np.full((0, n), FAR)
    touched = (S == CONTACT).any(axis=0)
    guard = np.zeros(n, dtype=bool)
    for r in np.flatnonzero(touched):
        guard[max(0, r - cfg.guard_rows): r + cfg.guard_rows + 1] = True
    out = np.full(n, "rest", dtype=object)
    out[(S == UNKNOWN).any(axis=0)] = "unknown_hand"
    out[(S == NEAR).any(axis=0)] = "near"
    out[guard] = "guard"
    out[touched] = "touched"
    out[~np.asarray(pose_ok, dtype=bool)] = "no_pose"
    return out.astype(str)


def rest_segments(states: np.ndarray) -> list[np.ndarray]:
    """Maximal runs of consecutive `rest` rows (index arrays)."""
    m = np.asarray(states) == "rest"
    out, start = [], None
    for i, v in enumerate(np.append(m, False)):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append(np.arange(start, i))
            start = None
    return out


def segment_window(hands: dict[str, HandInput], poses_scaled: np.ndarray, times: np.ndarray, mesh_vertices: np.ndarray,
                   mesh_faces: np.ndarray, scale: float, cfg: ReferenceConfig
                   ) -> tuple[np.ndarray, list[np.ndarray], dict[str, np.ndarray]]:
    """`(row states, rest segments, per-hand distances)` of one window."""
    pose_ok = np.isfinite(np.asarray(poses_scaled)).all(axis=(1, 2))
    dists, statuses = {}, []
    for side, hand in hands.items():
        d = hand_distance(hand, poses_scaled, pose_ok, mesh_vertices, mesh_faces, scale, cfg)
        dists[side] = d
        statuses.append(hand_status(d, times, hand.declared_absent, cfg))
    states = row_states(statuses, pose_ok, cfg)
    return states, rest_segments(states), dists
