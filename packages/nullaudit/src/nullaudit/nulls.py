# Copyright 2026 The HandReel authors. SPDX-License-Identifier: Apache-2.0
"""The standard null trajectories and null control sequences.

Kinematic nulls (no simulator):
- static-at-start: the object held at the reference's first pose on every row;
- time-shifted: the reference delayed by `shift` rows (the first pose fills the gap);
- reversed: the reference played backwards.
The last two are sanity nulls: a delayed or reversed trajectory that passes meets the test's tolerances on this clip.

Simulated nulls (need a simulator callback, see `audit.Simulator`):
- hand-absent (N0): the clip executed with no hand-object contact force on any post-release row;
- plan-then-freeze (N1): the stored controls up to the release row, then the last pre-release command held,
  u_t = u_{W-1} for t >= W.
"""

from __future__ import annotations

import numpy as np

from .poses import as_pose7

__all__ = ["move_hand_away", "plan_then_freeze_controls", "reversed_reference", "static_at_start", "time_shifted"]


def static_at_start(reference: np.ndarray, rows: int | None = None) -> np.ndarray:
    """The object held at the reference's first pose for `rows` rows (default: the reference length)."""
    ref = as_pose7(reference)
    n = ref.shape[0] if rows is None else int(rows)
    return np.repeat(ref[:1], n, axis=0)


def time_shifted(reference: np.ndarray, shift: int) -> np.ndarray:
    """The reference delayed by `shift` >= 0 rows: row t is reference row max(0, t - shift); same length."""
    ref = as_pose7(reference)
    if shift < 0:
        raise ValueError("shift must be >= 0")
    idx = np.clip(np.arange(ref.shape[0]) - int(shift), 0, None)
    return ref[idx]


def reversed_reference(reference: np.ndarray) -> np.ndarray:
    """The reference played backwards (row t is reference row n - 1 - t)."""
    return as_pose7(reference)[::-1].copy()


def plan_then_freeze_controls(controls: np.ndarray, release_row: int) -> np.ndarray:
    """N1: rows < `release_row` unchanged, every later row = the command of row `release_row` - 1."""
    u = np.asarray(controls, dtype=np.float64).copy()
    if not 0 < release_row <= u.shape[0]:
        raise ValueError(f"release_row {release_row} outside (0, {u.shape[0]}]")
    u[release_row:] = u[release_row - 1]
    return u


def move_hand_away(controls: np.ndarray, column: int, offset: float = 2.0) -> np.ndarray:
    """N0 helper: add `offset` to one control column (e.g. the hand-root x position target) on every row.

    Our studies move the hand root 2 m along x during the welded warm-up so that it never returns. Moving the
    hand is not enough on its own, and neither is a model edit on its own: the simulator must also report the
    hand-object force so that the audit can check N0 (`audit.Rollout.hand_object_force`).
    """
    u = np.asarray(controls, dtype=np.float64).copy()
    u[:, column] += offset
    return u
