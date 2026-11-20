"""Engine-rate to reference-rate composition of the retargeting stage's inputs.

The engine's perception stages (object solve, hands) run on the decode rows, every `stride`-th source frame (10 fps).
The kinematic reference is solved at the full source rate, one reference frame per source frame from the first to the
last decode row, on inputs upsampled from the engine rows. The reference itself is never upsampled: on the whisking
demo a 10 fps solve spline-upsampled penetrates on 40.6% of frames, the full-rate solve on upsampled inputs on none.
`upsample_reference_inputs` uses `fastdo.retarget.rate`'s merged helpers:

- object pose: `upsample_object_poses`, C2 natural cubic splines on positions and on sign-continuous quaternions;
- wrist SE(3) and hand vertices: `upsample_hand_track`, the wrist the same way and the vertices lerped per vertex in
  the wrist frame between keys (the helper's stand-in for per-joint slerp/lerp of MANO parameters, since no MANO layer
  is importable in the uv environment).

Masking (invalid hand rows must not create contacts or enter the retargeting cost). Batched HaWoR fills a hand row
whose `valid` is False with a finite copy of the nearest valid row, so its content is not evidence. Only valid rows are
spline knots; frames before the first or after the last valid row hold that row's values. A reference frame is valid
only when every engine row its interpolation brackets is valid: the row itself at a key, else the two rows around it.
`frame_valid` carries that to `retarget_kinematic`, which drops those frames' contacts and wrist targets. When every
source frame of a span is a knot, its rows are copied rather than splined.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from fastdo.retarget.rate import upsample_hand_track, upsample_object_poses

__all__ = ["ReferenceInputs", "reference_frame_valid", "reference_key_frames", "upsample_reference_inputs"]

_DT = torch.float64


@dataclass(frozen=True)
class ReferenceInputs:
    """The kinematic reference's inputs at the reference rate, in the one world frame of their engine rows."""

    obj_poses: Tensor        # (N,4,4) float64
    wrist_poses: Tensor      # (N,4,4) float64
    hand_verts: Tensor       # (N,V,3) float64
    frame_valid: Tensor      # (N,) bool; False where the hand input brackets an invalid row or is held past the ends
    key_frames: np.ndarray   # (T,) int64 reference frame of each engine row
    masked_rows: int         # engine rows with `valid` False
    masked_frames: int       # reference frames with `frame_valid` False

    @property
    def n_frames(self) -> int:
        return int(self.obj_poses.shape[0])


def reference_key_frames(frame_idx: np.ndarray) -> np.ndarray:
    """`(T,)` int64 reference frame of each engine row, `frame_idx - frame_idx[0]`.

    Raises:
        ValueError: If `frame_idx` is empty or not strictly increasing.
    """
    idx = np.asarray(frame_idx, dtype=np.int64).reshape(-1)
    if idx.size == 0:
        raise ValueError("frame_idx is empty")
    if not bool(np.all(np.diff(idx) > 0)):
        raise ValueError("frame_idx must strictly increase")
    return idx - idx[0]


def _check_keys(key_frames: np.ndarray) -> np.ndarray:
    keys = np.asarray(key_frames, dtype=np.int64).reshape(-1)
    if keys.size == 0 or int(keys[0]) != 0 or not bool(np.all(np.diff(keys) > 0)):
        raise ValueError(f"key frames must start at 0 and strictly increase, got {keys[:8].tolist()}")
    return keys


def reference_frame_valid(key_frames: np.ndarray, row_valid: np.ndarray) -> np.ndarray:
    """`(key_frames[-1] + 1,)` bool: at a key frame its row's validity, elsewhere both bracketing rows' validity.

    Raises:
        ValueError: On bad key frames or a validity count that differs from the key count.
    """
    keys = _check_keys(key_frames)
    valid = np.asarray(row_valid, dtype=bool).reshape(-1)
    if valid.shape != keys.shape:
        raise ValueError(f"{valid.shape[0]} validity flags for {keys.shape[0]} key frames")
    frames = np.arange(int(keys[-1]) + 1)
    right = np.searchsorted(keys, frames, side="left")                  # first key >= frame, always a valid index
    left = np.where(keys[right] == frames, right, right - 1)
    return np.asarray(valid[left] & valid[right], dtype=bool)


def upsample_reference_inputs(obj_poses: Tensor, wrist_poses: Tensor, hand_verts: Tensor, row_valid: np.ndarray,
                              key_frames: np.ndarray) -> ReferenceInputs:
    """Engine rows -> reference frames, with the invalid hand rows masked (module docstring).

    Args:
        obj_poses: `(T,4,4)` object poses of the engine rows, world frame.
        wrist_poses: `(T,4,4)` wrist poses of the engine rows, same frame.
        hand_verts: `(T,V,3)` hand vertices of the engine rows, same frame.
        row_valid: `(T,)` bool hand validity of the engine rows.
        key_frames: `(T,)` reference frame of each engine row (`reference_key_frames`).

    Raises:
        ValueError: On mismatched row counts, bad key frames, or no valid hand row.
    """
    keys = _check_keys(key_frames)
    t, n = int(keys.shape[0]), int(keys[-1]) + 1
    valid = np.asarray(row_valid, dtype=bool).reshape(-1)
    if (obj_poses.shape != (t, 4, 4) or wrist_poses.shape != (t, 4, 4) or hand_verts.dim() != 3
            or hand_verts.shape[0] != t or hand_verts.shape[2] != 3 or valid.shape[0] != t):
        raise ValueError(f"expected {t} engine rows: got obj_poses {tuple(obj_poses.shape)}, wrist_poses "
                         f"{tuple(wrist_poses.shape)}, hand_verts {tuple(hand_verts.shape)}, row_valid {valid.shape}")
    rows = np.flatnonzero(valid)
    if rows.size == 0:
        raise ValueError("no valid hand row to compose the reference inputs from")
    knots = keys[rows]
    lo, hi = int(knots[0]), int(knots[-1])
    index = torch.from_numpy(rows)
    wrist_knots, verts_knots = wrist_poses.to(_DT)[index], hand_verts.to(_DT)[index]
    wrist = torch.empty(n, 4, 4, dtype=_DT)
    verts = torch.empty(n, hand_verts.shape[1], 3, dtype=_DT)
    if knots.shape[0] == hi - lo + 1:                                   # every frame of [lo, hi] is a knot: copy
        wrist[lo:hi + 1], verts[lo:hi + 1] = wrist_knots, verts_knots
    else:
        wrist_up, verts_up = upsample_hand_track(wrist_knots, verts_knots, knots - lo, hi - lo + 1)
        wrist[lo:hi + 1], verts[lo:hi + 1] = torch.from_numpy(wrist_up), torch.from_numpy(verts_up)
    wrist[:lo], verts[:lo] = wrist[lo], verts[lo]                       # held past the ends; masked below
    wrist[hi + 1:], verts[hi + 1:] = wrist[hi], verts[hi]
    if t == n:
        obj = obj_poses.to(_DT).clone()
    else:
        obj = torch.from_numpy(upsample_object_poses(obj_poses, keys, n))
    frame_valid = reference_frame_valid(keys, valid)
    return ReferenceInputs(obj_poses=obj, wrist_poses=wrist, hand_verts=verts, frame_valid=torch.from_numpy(frame_valid),
                           key_frames=keys, masked_rows=int((~valid).sum()), masked_frames=int((~frame_valid).sum()))
