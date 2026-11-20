"""Gate labels: 100DOH contact states, synthetic stand-ins, DexYCB contact windows and the DexYCB partition.

Label sources (design §3: stationary contact and no contact are negatives):

- 100DOH train clips: `interact` from the anchor annotation, positive iff some hand is in portable-object contact
  (`contact_state == 3`, legend at 100DOH/official/file/README_100K.md:73). These clips only calibrate the threshold.
- DexYCB (the fit): a positive window over the whole sequence when some frame has hand-object contact, and a negative
  window over the longest no-contact, object-still prefix of the same sequence. `visibility` and keyframe targets come
  from the grasped object's pixel count and hand-object distance.
- Synthetic clips have no RGB, so they only fix label semantics in tests.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from fastdo.bench.bench_clips import doh_clip_name
from fastdo.gate.features import N_GATE_FRAMES
from fastdo.io.video import sample_frame_indices

if TYPE_CHECKING:
    from fastdo.solver.synthetic import SyntheticClip

__all__ = [
    "HEADS",
    "PORTABLE_OBJECT_CONTACT",
    "DexycbFrameLabels",
    "GateLabels",
    "dexycb_frame_labels",
    "dexycb_partition",
    "dexycb_windows",
    "doh_interact_label",
    "synthetic_stand_in_labels",
]

HEADS: tuple[str, ...] = ("interact", "rigid", "visibility", "keyframe")
PORTABLE_OBJECT_CONTACT = 3
STAND_IN_MIN_AFFECTED = 0.2      # the plan's rule: at least this fraction of frames has the occluder touching the object


@dataclass(frozen=True)
class GateLabels:
    """`interact`, `rigid`, `visibility` `(B,)`, `keyframe` `(B, 8)` and `mask` `(B, 4)` bool in `HEADS` order."""

    interact: Tensor
    rigid: Tensor
    visibility: Tensor
    keyframe: Tensor
    mask: Tensor


def doh_interact_label(entry: Mapping[str, Any]) -> bool:
    """True iff some hand of the 100DOH entry's anchor annotation is in portable-object contact (state 3).

    Raises:
        ValueError: Naming `doh_clip_name(entry)` when `hands` or a hand's `contact_state` is missing.
        TypeError: If `hands` is not a list.
    """
    name = doh_clip_name(entry)
    hands = entry.get("hands")
    if hands is None:
        raise ValueError(f"100DOH entry {name} has no hands annotation")
    if not isinstance(hands, list):
        raise TypeError(f"100DOH entry {name}: hands must be a list, got {type(hands).__name__}")
    states: list[int] = []
    for i, hand in enumerate(hands):
        if not isinstance(hand, Mapping) or hand.get("contact_state") is None:
            raise ValueError(f"100DOH entry {name}: hand {i} has no contact_state")
        states.append(int(hand["contact_state"]))
    return PORTABLE_OBJECT_CONTACT in states


def synthetic_stand_in_labels(clips: list[SyntheticClip]) -> GateLabels:
    """The plan's stand-in labels: a frame is occluder-affected iff the 3x3-dilated `hand_masks[k]` overlaps
    `masks[k]`; `interact = hand_masks.any() and affected_fraction >= 0.2`; `rigid = 1`;
    `visibility = mean_k(|masks[k]| / max_k |masks[k]|)`. Keyframe is unlabelled.

    Raises:
        ValueError: If `clips` is empty.
    """
    if not clips:
        raise ValueError("synthetic_stand_in_labels needs at least one clip")
    interact: list[float] = []
    visibility: list[float] = []
    for clip in clips:
        masks, hands = clip.masks.bool(), clip.hand_masks.bool()
        dilated = F.max_pool2d(hands.float()[:, None], kernel_size=3, stride=1, padding=1)[:, 0] > 0
        affected = (dilated & masks).flatten(1).any(dim=1)
        fraction = int(affected.sum()) / int(affected.numel())
        interact.append(float(bool(hands.any()) and fraction >= STAND_IN_MIN_AFFECTED))
        areas = masks.flatten(1).sum(dim=1).double()
        peak = float(areas.max())
        visibility.append(float((areas / peak).mean()) if peak > 0 else 0.0)
    b = len(clips)
    return GateLabels(interact=torch.tensor(interact), rigid=torch.ones(b), visibility=torch.tensor(visibility),
                      keyframe=torch.zeros(b, N_GATE_FRAMES),
                      mask=torch.tensor([True, True, True, False]).expand(b, len(HEADS)).clone())


@dataclass(frozen=True)
class DexycbFrameLabels:
    """Per source frame `(T,)`: hand-object distance (NaN: object pose invalid; +inf: hand absent), grasped-object
    displacement from its first valid pose (NaN: pose invalid), and pixel visibility relative to the clip maximum."""

    distance_m: np.ndarray
    displacement_m: np.ndarray
    visibility: np.ndarray


def _invalid_poses(poses: np.ndarray) -> np.ndarray:
    """All 0, all -1, or any NaN: a superset of `fastdo.io.datasets.dexycb._is_invalid` (all -1 or NaN)."""
    flat = poses.reshape(poses.shape[0], -1)
    return (flat == 0).all(axis=1) | (flat == -1).all(axis=1) | np.isnan(flat).any(axis=1)


def dexycb_frame_labels(joints: np.ndarray, object_points: np.ndarray, poses: np.ndarray,
                        object_pixels: np.ndarray) -> DexycbFrameLabels:
    """Frame labels of one DexYCB camera sequence.

    Args:
        joints: `(T, 21, 3)` camera-frame hand joints in metres; a frame with any -1 has no hand.
        object_points: `(P, 3)` grasped-object surface points in the object frame, metres.
        poses: `(T, 3, 4)` grasped-object `[R | t]` object-to-camera (`pose_y[ycb_grasp_ind]`).
        object_pixels: `(T,)` count of `seg == ycb_ids[ycb_grasp_ind]`.

    Raises:
        ValueError: On inconsistent shapes, or if the object is never visible (`max(object_pixels) == 0`).
    """
    joints = np.asarray(joints, dtype=np.float64)
    points = np.asarray(object_points, dtype=np.float64)
    poses = np.asarray(poses, dtype=np.float64)
    pixels = np.asarray(object_pixels, dtype=np.float64)
    t = poses.shape[0] if poses.ndim == 3 else -1
    if (poses.shape[1:] != (3, 4) or joints.shape != (t, 21, 3) or points.ndim != 2 or points.shape[1] != 3
            or points.shape[0] < 1 or pixels.shape != (t,)):
        raise ValueError(f"expected joints (T,21,3), object_points (P,3), poses (T,3,4) and object_pixels (T,), got "
                         f"{joints.shape}, {points.shape}, {poses.shape} and {pixels.shape}")
    pose_bad = _invalid_poses(poses)
    hand_bad = (joints == -1).any(axis=(1, 2))
    distance = np.full(t, np.nan)
    for i in np.flatnonzero(~pose_bad):
        if hand_bad[i]:
            distance[i] = np.inf
            continue
        posed = points @ poses[i, :, :3].T + poses[i, :, 3]
        distance[i] = float(np.sqrt(((joints[i][:, None, :] - posed[None, :, :]) ** 2).sum(axis=-1).min()))
    displacement = np.full(t, np.nan)
    valid = np.flatnonzero(~pose_bad)
    if valid.size:
        displacement[valid] = np.linalg.norm(poses[valid, :, 3] - poses[valid[0], :, 3], axis=1)
    peak = float(pixels.max()) if t > 0 else 0.0
    if peak <= 0:
        raise ValueError("the grasped object is never visible (object_pixels max is 0)")
    return DexycbFrameLabels(distance_m=distance, displacement_m=displacement, visibility=pixels / peak)


def dexycb_windows(labels: DexycbFrameLabels, stride: int, contact_m: float = 0.015, free_m: float = 0.030,
                   still_m: float = 0.005, min_negative_frames: int = 4) -> tuple[list[int] | None, list[int] | None]:
    """`(positive_rows, negative_rows)` of one sequence.

    Positive rows are `sample_frame_indices(T, stride)`, or None without a contact frame (`distance_m < contact_m`).
    A frame is free when `distance_m > free_m` (inf counts) and `displacement_m <= still_m`; a NaN frame is not free.
    Negative rows are the longest free prefix `[0, t)`, or None when `t < min_negative_frames`.

    Raises:
        ValueError: If `stride < 1` or `min_negative_frames < 1`.
    """
    if stride < 1 or min_negative_frames < 1:
        raise ValueError(f"stride and min_negative_frames must be >= 1, got {stride} and {min_negative_frames}")
    distance, displacement = labels.distance_m, labels.displacement_m
    t = int(distance.shape[0])
    positive = sample_frame_indices(t, stride) if bool((distance < contact_m).any()) else None
    blocked = np.flatnonzero(~((distance > free_m) & (displacement <= still_m)))
    prefix = int(blocked[0]) if blocked.size else t
    return positive, (list(range(prefix)) if prefix >= min_negative_frames else None)


def dexycb_partition(seq_ids: Sequence[str], val_subject: str, val_every: int = 5) -> dict[str, str]:
    """`seq_id -> "val" | "val_seq" | "fit"`: every id of `val_subject` is `val` (subject-held-out, the promotion
    metric); of the other ids, sorted, index `i % val_every == val_every - 1` is `val_seq` (reported only), else `fit`.

    Raises:
        ValueError: If no id or every id belongs to `val_subject`, an id repeats, or `val_every < 1`.
    """
    ids = [str(seq_id) for seq_id in seq_ids]
    if val_every < 1:
        raise ValueError(f"val_every must be >= 1, got {val_every}")
    if len(set(ids)) != len(ids):
        raise ValueError("DexYCB ids repeat")
    val = {seq_id for seq_id in ids if seq_id.split("/")[0] == val_subject}
    if not val:
        raise ValueError(f"no DexYCB id belongs to val_subject {val_subject!r}")
    if len(val) == len(ids):
        raise ValueError(f"every DexYCB id belongs to val_subject {val_subject!r}: nothing is left to fit on")
    rest = sorted(seq_id for seq_id in ids if seq_id not in val)
    parts = dict.fromkeys(val, "val") | {seq_id: "val_seq" if i % val_every == val_every - 1 else "fit"
                                         for i, seq_id in enumerate(rest)}
    return {seq_id: parts[seq_id] for seq_id in ids}
