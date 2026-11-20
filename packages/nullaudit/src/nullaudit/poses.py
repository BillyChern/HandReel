# Copyright 2026 The HandReel authors. SPDX-License-Identifier: Apache-2.0
"""Object-pose trajectories: (T, 7) rows (x, y, z, qw, qx, qy, qz) or (T, 4, 4) homogeneous matrices.

Every function here returns the (T, 7) form with float64 values. Quaternions are scalar-first (wxyz, MuJoCo's
convention) and need not be unit length: `quat_angle` is invariant to the scale and sign of either argument.
"""

from __future__ import annotations

import numpy as np

__all__ = ["ROTATION_TOL", "as_pose7", "matrix_to_quat", "quat_angle"]

ROTATION_TOL = 1e-4        # max |R^T R - I| and |det R - 1| for a block to count as a proper rotation


def matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrices (T, 3, 3) -> unit wxyz quaternions (T, 4) with w >= 0 (Shepperd's method).

    Raises:
        ValueError: If a block is not a proper rotation to within `ROTATION_TOL` (non-finite, not orthonormal, or
            of determinant other than +1). A zero block, a scaled rotation and a reflection have no quaternion;
            converting one would return an arbitrary orientation, so none is converted.
    """
    R = np.asarray(R, dtype=np.float64)
    if R.ndim != 3 or R.shape[1:] != (3, 3):
        raise ValueError(f"expected rotation matrices of shape (T, 3, 3), got {R.shape}")
    if R.shape[0] == 0:                  # an empty run in the matrix form is as valid an input as (0, 7)
        return np.empty((0, 4))
    with np.errstate(invalid="ignore"):
        ortho = np.abs(np.swapaxes(R, 1, 2) @ R - np.eye(3)).reshape(R.shape[0], -1).max(axis=1, initial=0.0)
        bad = ~np.isfinite(R).all(axis=(1, 2)) | ~(ortho <= ROTATION_TOL) | \
            ~(np.abs(np.linalg.det(R) - 1.0) <= ROTATION_TOL)
    if bad.any():
        rows = np.flatnonzero(bad)
        raise ValueError(f"{len(rows)} rotation block(s) are not proper rotations (first rows {rows[:5].tolist()}): "
                         "no orientation is defined for them")
    tr = np.trace(R, axis1=1, axis2=2)
    diag = np.stack([tr, R[:, 0, 0], R[:, 1, 1], R[:, 2, 2]], axis=1)
    k = np.argmax(diag, axis=1)
    q = np.empty((R.shape[0], 4))
    for i in range(R.shape[0]):
        m = R[i]
        if k[i] == 0:
            s = 2.0 * np.sqrt(max(1.0 + tr[i], 0.0))
            q[i] = (0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s)
        elif k[i] == 1:
            s = 2.0 * np.sqrt(max(1.0 + m[0, 0] - m[1, 1] - m[2, 2], 0.0))
            q[i] = ((m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s)
        elif k[i] == 2:
            s = 2.0 * np.sqrt(max(1.0 + m[1, 1] - m[0, 0] - m[2, 2], 0.0))
            q[i] = ((m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s)
        else:
            s = 2.0 * np.sqrt(max(1.0 + m[2, 2] - m[0, 0] - m[1, 1], 0.0))
            q[i] = ((m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return q * np.where(q[:, :1] < 0.0, -1.0, 1.0)


def as_pose7(traj: np.ndarray) -> np.ndarray:
    """A (T, 7) or (T, 4, 4) object trajectory as a new (T, 7) float64 array (position, wxyz quaternion)."""
    x = np.asarray(traj, dtype=np.float64)
    if x.ndim == 2 and x.shape[1] == 7:
        return x.copy()
    if x.ndim == 3 and x.shape[1:] == (4, 4):
        return np.concatenate([x[:, :3, 3], matrix_to_quat(x[:, :3, :3])], axis=1)
    raise ValueError(f"expected an object trajectory of shape (T, 7) or (T, 4, 4), got {x.shape}")


def quat_angle(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Geodesic angle |log(r^-1 q)| in radians, row by row, for wxyz quaternion rows (T, 4).

    Shortest path (q and -q are the same rotation) and scale-free: the relative quaternion's vector and scalar parts
    scale together, so 2 atan2(|v|, |w|) does not depend on |q| or |r|.
    """
    q = np.asarray(q, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    w1, v1 = q[:, :1], q[:, 1:]
    w2, v2 = r[:, :1], -r[:, 1:]
    w = w1 * w2 - np.sum(v1 * v2, axis=1, keepdims=True)
    v = w1 * v2 + w2 * v1 + np.cross(v1, v2)
    return 2.0 * np.arctan2(np.linalg.norm(v, axis=1), np.abs(w[:, 0]))
