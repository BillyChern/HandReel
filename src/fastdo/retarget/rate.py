"""Engine-rate composition of the kinematic reference's inputs.

Perception (object solve, hands) runs at the engine rate, 10 fps. The kinematic reference is then solved at the
full rate on inputs upsampled from the engine rate, never upsampled itself: on the whisking demo a 10 fps
solve spline-upsampled to 138 frames penetrates on 40.6% of frames (min -17.5 mm), while the full-rate solve on
spline-upsampled 10 fps inputs penetrates on none. Clearance is a per-frame
constraint that a smooth curve between keys cannot keep around thin geometry.

- `upsample_object_poses`: positions by C2 natural cubic spline; rotations by natural spline on sign-continuous
  unit quaternions, renormalized. The approximation needs every key-to-key rotation well under 90 degrees, so
  `quaternion_spline` logs a WARNING naming the intervals of 45 degrees or more.
- `upsample_hand_track`: the wrist SE(3) the same way. Hand articulation would be interpolated per joint by
  slerp/lerp of MANO parameters, not by splines (splines lose to slerp on HaWoR parameters). No MANO layer is
  importable in the uv environment (`smplx` and `manopth` are absent; MANO is a runtime licence), so the MANO
  vertices are lerped per vertex in the
  wrist frame between keys and mapped back with the upsampled wrist. Approximation: a linear blend of the two key
  meshes in the wrist frame moves a vertex that rotates by theta about a joint at lever arm r along the chord
  instead of the arc, at most r (1 - cos(theta / 2)) short of it mid-interval (0.25 mm for r = 5 cm and
  theta = 0.2 rad); it also blends whatever wrist-frame jitter the per-frame hand fits carry.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation
from torch import Tensor

logger = logging.getLogger(__name__)

__all__ = ["engine_key_frames", "natural_spline", "quaternion_spline", "upsample_hand_track",
           "upsample_object_poses"]

_WARN_INTERVAL_DEG = 45.0


def engine_key_frames(n_frames: int, stride: int) -> np.ndarray:
    """int64 `arange(0, n_frames, stride)`, with `n_frames - 1` appended when it is not already the last key."""
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    if n_frames < 2:
        raise ValueError(f"n_frames must be >= 2, got {n_frames}")
    keys = np.arange(0, n_frames, stride, dtype=np.int64)
    if keys[-1] != n_frames - 1:
        keys = np.append(keys, np.int64(n_frames - 1))
    return keys


def _check_keys(key_frames: np.ndarray) -> np.ndarray:
    keys = np.asarray(key_frames)
    if keys.ndim != 1 or keys.shape[0] < 2:
        raise ValueError(f"need at least 2 key frames, got shape {keys.shape}")
    if not bool(np.all(np.diff(keys) > 0)):
        raise ValueError("key frames must strictly increase")
    return keys


def natural_spline(values: np.ndarray, key_frames: np.ndarray, out_frames: np.ndarray) -> np.ndarray:
    """C2 natural cubic spline through `values (K, ...)` at `key_frames (K,)`, evaluated at `out_frames`."""
    keys = _check_keys(key_frames)
    vals = np.asarray(values, dtype=np.float64)
    if vals.shape[0] != keys.shape[0]:
        raise ValueError(f"{vals.shape[0]} values for {keys.shape[0]} key frames")
    spline = CubicSpline(keys.astype(np.float64), vals, axis=0, bc_type="natural")
    return np.asarray(spline(np.asarray(out_frames).astype(np.float64)), dtype=np.float64)


def quaternion_spline(quat_wxyz: np.ndarray, key_frames: np.ndarray, out_frames: np.ndarray) -> np.ndarray:
    """Natural spline of sign-continuous wxyz quaternions `(K, 4)`, renormalized per output row.

    Key k is negated when its dot product with the (already flipped) key k - 1 is negative; key 0 keeps its sign,
    so negating key 0 negates the whole output."""
    quat = np.array(quat_wxyz, dtype=np.float64, copy=True)
    for k in range(1, quat.shape[0]):
        if float(np.dot(quat[k], quat[k - 1])) < 0.0:
            quat[k] = -quat[k]
    dots = np.clip(np.abs(np.sum(quat[1:] * quat[:-1], axis=1)), 0.0, 1.0)
    angles = np.degrees(2.0 * np.arccos(dots))
    large = [(int(k), int(k) + 1) for k in np.nonzero(angles >= _WARN_INTERVAL_DEG)[0]]
    if large:
        logger.warning("quaternion spline: key intervals %s rotate by >= %.0f degrees (max %.1f); the renormalized "
                       "spline is only accurate well under 90 degrees", large, _WARN_INTERVAL_DEG, float(angles.max()))
    out = natural_spline(quat, key_frames, out_frames)
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def _as_float64(x: np.ndarray | Tensor) -> np.ndarray:
    if isinstance(x, Tensor):
        return x.detach().cpu().numpy().astype(np.float64)
    return np.asarray(x, dtype=np.float64)


def _check_upsample(n_rows: int, key_frames: np.ndarray, n_frames: int) -> np.ndarray:
    keys = _check_keys(key_frames)
    if n_rows != keys.shape[0]:
        raise ValueError(f"{n_rows} rows for {keys.shape[0]} key frames")
    if int(keys[0]) != 0 or int(keys[-1]) != n_frames - 1:
        raise ValueError(f"key frames must start at 0 and end at n_frames - 1 = {n_frames - 1} (no extrapolation), "
                         f"got {int(keys[0])}..{int(keys[-1])}")
    return keys


def upsample_object_poses(poses: np.ndarray | Tensor, key_frames: np.ndarray, n_frames: int) -> np.ndarray:
    """`(K, 4, 4)` SE(3) poses at `key_frames` -> `(n_frames, 4, 4)` float64 at every frame (module docstring).

    Key 0's quaternion is taken with w >= 0, so the output does not depend on the matrix-to-quaternion sign."""
    mats = _as_float64(poses)
    if mats.ndim != 3 or mats.shape[1:] != (4, 4):
        raise ValueError(f"expected (K, 4, 4) poses, got {mats.shape}")
    keys = _check_upsample(mats.shape[0], key_frames, n_frames)
    out_frames = np.arange(n_frames)
    quat = Rotation.from_matrix(mats[:, :3, :3]).as_quat()[:, [3, 0, 1, 2]]
    if quat[0, 0] < 0.0:
        quat[0] = -quat[0]
    out = np.tile(np.eye(4), (n_frames, 1, 1))
    out[:, :3, 3] = natural_spline(mats[:, :3, 3], keys, out_frames)
    out[:, :3, :3] = Rotation.from_quat(quaternion_spline(quat, keys, out_frames)[:, [1, 2, 3, 0]]).as_matrix()
    return out


def upsample_hand_track(wrist_poses: np.ndarray | Tensor, hand_verts: np.ndarray | Tensor, key_frames: np.ndarray,
                        n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """`(K, 4, 4)` wrist poses and `(K, V, 3)` hand vertices at `key_frames` -> `(n_frames, 4, 4)` wrist poses by
    `upsample_object_poses` and `(n_frames, V, 3)` vertices lerped per vertex in the wrist frame (module docstring)."""
    wrist = _as_float64(wrist_poses)
    verts = _as_float64(hand_verts)
    if wrist.ndim != 3 or wrist.shape[1:] != (4, 4) or verts.ndim != 3 or verts.shape[2] != 3:
        raise ValueError(f"expected (K, 4, 4) wrist poses and (K, V, 3) vertices, got {wrist.shape}, {verts.shape}")
    keys = _check_upsample(wrist.shape[0], key_frames, n_frames)
    if verts.shape[0] != keys.shape[0]:
        raise ValueError(f"{verts.shape[0]} vertex rows for {keys.shape[0]} key frames")
    wrist_out = upsample_object_poses(wrist, keys, n_frames)
    local = np.einsum("kji,kvj->kvi", wrist[:, :3, :3], verts - wrist[:, None, :3, 3])       # R_k^T (v - p_k)
    frames = np.arange(n_frames)
    seg = np.clip(np.searchsorted(keys, frames, side="right") - 1, 0, keys.shape[0] - 2)
    alpha = ((frames - keys[seg]) / (keys[seg + 1] - keys[seg]))[:, None, None]
    blend = (1.0 - alpha) * local[seg] + alpha * local[seg + 1]
    verts_out = np.einsum("tij,tvj->tvi", wrist_out[:, :3, :3], blend) + wrist_out[:, None, :3, 3]
    return wrist_out, verts_out
