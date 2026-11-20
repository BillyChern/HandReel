# Copyright 2026 The HandReel authors. SPDX-License-Identifier: Apache-2.0
"""Reference implementation of DO AS I DO's object-tracking success test (arXiv 2606.19333, section 4.1).

The test, as specified in the paper and re-executed in our studies:

    E_pos = (1/n) sum_{t<n} |p_t - p^r_t|  <  0.1 m      and      E_rot = (1/n) sum_{t<n} ang(q_t, q^r_t)  <  0.5 rad,

where the mean runs over the n EXECUTED rows of the simulator, warm-up rows included. During the warm-up the object
is welded at the reference start pose, so those rows have zero error by construction. Our PRIMARY reading
adds that the run executed the whole clip (n >= T); `required_rows` switches that on.

This file is a clean-room port of our own study scorer (its `reference_steps`, `daid` and
`null_flags`). It contains no third-party code. Two separate development records stand behind it. A post hoc check
using DO AS I DO's tracking-error function reproduced 112 stored whole-trace errors with a maximum absolute
difference of 1.8e-7. A separate protocol specified a tolerance of 5.1e-5 for a later known-answer check on
reference clips; that number is a specified tolerance, not a measured difference. This package's own parity with
our scorer is a third, separate check, made in the development repository on stored cases (that test does not
ship and skips where its resources are missing).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .poses import as_pose7, quat_angle

__all__ = ["DAID_POS_M", "DAID_ROT_RAD", "DoAsIDoObjectTest", "ObjectTestResult", "object_errors",
           "reference_on_executor_grid"]

DAID_POS_M = 0.1
DAID_ROT_RAD = 0.5


def reference_on_executor_grid(frames: np.ndarray, ref_dt: float, sim_dt: float, warmup_rows: int,
                               n_rows: int) -> np.ndarray:
    """The reference as DO AS I DO's loader places it on the simulator grid (float32, as they run it).

    Each reference frame is repeated k = max(1, round(ref_dt / sim_dt)) times by linear interpolation over
    h * k evenly spaced points; frame 0 is then held for `warmup_rows` rows in front; the last row is repeated, or the
    result cut, to exactly `n_rows` rows. Columns are interpolated independently, so a quaternion that flips sign
    between two raw frames is interpolated through a short (non-unit) quaternion; `quat_angle` is scale-free.

    Args:
        frames: (h, D) raw reference rows, e.g. (h, 7) object poses at the video rate.
        ref_dt: seconds per reference frame.
        sim_dt: seconds per simulator row.
        warmup_rows: W, the welded warm-up rows.
        n_rows: rows of the returned grid.

    Returns:
        (n_rows, D) float32 array.
    """
    q = np.asarray(frames).astype(np.float32)
    k = max(1, round(ref_dt / sim_dt))
    h = q.shape[0]
    x = np.linspace(0.0, h - 1, h * k)
    cols = [np.interp(x, np.arange(h), q[:, j]) for j in range(q.shape[1])]
    out = np.stack(cols, axis=1).astype(np.float32)
    out = np.concatenate([np.repeat(out[:1], warmup_rows, axis=0), out], axis=0)
    if out.shape[0] < n_rows:
        out = np.concatenate([out, np.repeat(out[-1:], n_rows - out.shape[0], axis=0)], axis=0)
    return out[:n_rows]


def object_errors(executed: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    """(E_pos in m, E_rot in rad): means over every executed row against the reference on the same rows.

    The reference is cut to the executed length, or padded with its last row when the run is longer.

    Raises:
        ValueError: If a compared reference row is non-finite or has a zero quaternion.
    """
    ex = as_pose7(executed)
    ref = as_pose7(reference)
    n = ex.shape[0]
    if n == 0:
        return float("nan"), float("nan")
    if n <= ref.shape[0]:
        ref = ref[:n]
    else:
        ref = np.concatenate([ref, np.repeat(ref[-1:], n - ref.shape[0], axis=0)], axis=0)
    # A zero reference quaternion has no orientation: quat_angle would read atan2(0, 0) = 0, a perfect match.
    # Component-wise interpolation of equivalent raw signs (+q, -q) can produce one, so it is refused here,
    # before any verdict, whatever built the reference.
    undefined = np.flatnonzero(~(np.linalg.norm(ref[:, 3:], axis=1) > 0.0) | ~np.isfinite(ref).all(axis=1))
    if undefined.size:
        raise ValueError(f"the reference orientation is undefined on {undefined.size} compared row(s) (first rows "
                         f"{undefined[:5].tolist()}): a zero or non-finite reference quaternion")
    pos = float(np.linalg.norm(ex[:, :3] - ref[:, :3], axis=1).mean())
    rot = float(quat_angle(ex[:, 3:], ref[:, 3:]).mean())
    return pos, rot


@dataclass(frozen=True)
class ObjectTestResult:
    """One verdict of the object test. `full` = executed rows >= required rows and every row finite, with a
    non-zero quaternion (a zero quaternion is no rotation and is treated like a non-finite row)."""

    passed: bool
    pos_err_m: float
    rot_err_rad: float
    executed_rows: int
    full: bool

    def __bool__(self) -> bool:
        return self.passed


@dataclass(frozen=True)
class DoAsIDoObjectTest:
    """DO AS I DO's object test bound to one clip's reference; call it on an executed object trajectory.

    Args:
        reference: (n, 7) or (n, 4, 4) reference on the executor grid, warm-up rows included
            (see `reference_on_executor_grid`).
        required_rows: T, the clip's simulator rows. A run with fewer executed rows fails (our PRIMARY reading).
            None scores any prefix, as the released return value does (the LITERAL reading).
        pos_m, rot_rad: the published thresholds, strict `<`.
    """

    reference: np.ndarray
    required_rows: int | None = None
    pos_m: float = DAID_POS_M
    rot_rad: float = DAID_ROT_RAD

    def __call__(self, executed: np.ndarray) -> ObjectTestResult:
        ex = as_pose7(executed)
        n = int(ex.shape[0])
        finite = bool(np.isfinite(ex).all() and (np.linalg.norm(ex[:, 3:], axis=1) > 0.0).all())
        full = bool((self.required_rows is None or n >= self.required_rows) and n > 0 and finite)
        pos, rot = object_errors(ex, self.reference)
        return ObjectTestResult(passed=bool(full and pos < self.pos_m and rot < self.rot_rad), pos_err_m=pos,
                                rot_err_rad=rot, executed_rows=n, full=full)
