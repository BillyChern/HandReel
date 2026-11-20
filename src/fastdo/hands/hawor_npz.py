"""HaWoR per-clip hand-mesh npz -> ``fastdo.io.clip_record.HandTrack``, plus MANO fingertip ids.

Format verified against the baseline's real demo data (``third_party/do-as-i-do/reconstruction/
whisking/whisking/all_hand_meshes.npz``, 138 frames): keys ``{left,right}_vertices`` (T,778,3),
``{left,right}_faces`` (1552,3), ``{left,right}_joints`` (T,21,3), ``{left,right}_trans`` (T,3),
``{left,right}_rot`` (T,3), ``{left,right}_hand_pose`` (T,45), ``{left,right}_betas`` (T,10), and
``{left,right}_valid`` (T,) bool.

``HandTrack.params`` (``fastdo.io.clip_record``) is global-orient axis-angle (3) + 15-joint
axis-angle (45) = 48. HaWoR's ``rot``/``hand_pose`` are already that convention (axis-angle, not
PCA) -- contrast ``fastdo.io.datasets.dexycb``, whose DexYCB source stores MANO pose as PCA
coefficients and needs a basis matrix to convert to axis-angle. ``HandTrack.betas`` is a single
``(10,)`` shape vector per clip, but HaWoR estimates it per frame; this loader averages
``{side}_betas`` over frames, per the adapter contract.

MANO tip vertex ids (thumb, index, middle, ring, pinky; shared topology for both hands) are the
standard MANO fingertip vertex ids, not derived here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from fastdo.io.clip_record import HandTrack

__all__ = ["fingertip_indices", "load_hawor_npz"]

_SIDES = ("left", "right")
_FINGERTIP_VERTEX_IDS = (745, 317, 444, 556, 673)  # thumb, index, middle, ring, pinky


def fingertip_indices() -> Tensor:
    """MANO tip vertex ids (thumb, index, middle, ring, pinky), ``(5,)`` int64."""
    return torch.tensor(_FINGERTIP_VERTEX_IDS, dtype=torch.int64)


def load_hawor_npz(path: Path, side: str = "right") -> HandTrack:
    """Loads one hand's trajectory from a HaWoR ``all_hand_meshes.npz`` file.

    Args:
        path: Path to the npz file.
        side: ``"left"`` or ``"right"``.

    Returns:
        A ``HandTrack`` with ``params = concat(rot, hand_pose)`` (T,48), ``betas`` averaged over
        frames (10,), camera-frame ``trans`` (T,3) and ``joints`` (T,21,3), and ``valid`` (T,)
        true where ``trans``/``rot``/``hand_pose`` are all finite (and, when the npz carries a
        ``{side}_valid`` array, additionally true there).

    Raises:
        ValueError: If ``side`` is not ``"left"``/``"right"``, or an array has an unexpected shape.
    """
    if side not in _SIDES:
        raise ValueError(f"side must be one of {_SIDES}, got {side!r}")

    with np.load(Path(path)) as data:
        rot = np.asarray(data[f"{side}_rot"], dtype=np.float32)
        hand_pose = np.asarray(data[f"{side}_hand_pose"], dtype=np.float32)
        trans = np.asarray(data[f"{side}_trans"], dtype=np.float32)
        joints = np.asarray(data[f"{side}_joints"], dtype=np.float32)
        betas_raw = np.asarray(data[f"{side}_betas"], dtype=np.float32)
        valid_key = f"{side}_valid"
        provided_valid = np.asarray(data[valid_key], dtype=bool) if valid_key in data else None

    t = rot.shape[0]
    _expect_shape(rot, (t, 3), f"{side}_rot", path)
    _expect_shape(hand_pose, (t, 45), f"{side}_hand_pose", path)
    _expect_shape(trans, (t, 3), f"{side}_trans", path)
    _expect_shape(joints, (t, 21, 3), f"{side}_joints", path)

    params = np.concatenate([rot, hand_pose], axis=1).astype(np.float32)
    betas = betas_raw.mean(axis=0).astype(np.float32) if betas_raw.ndim == 2 else betas_raw.astype(np.float32)

    finite = np.isfinite(trans).all(axis=1) & np.isfinite(rot).all(axis=1) & np.isfinite(hand_pose).all(axis=1)
    valid = finite if provided_valid is None else (finite & provided_valid)

    track = HandTrack(params=params, betas=betas, trans=trans, joints=joints, valid=valid)
    track.validate(t, f"{side}_hand")
    return track


def _expect_shape(arr: np.ndarray, shape: tuple[int, ...], name: str, path: Path) -> None:
    if tuple(arr.shape) != shape:
        raise ValueError(f"{path}: {name} expected shape {shape}, got {tuple(arr.shape)}")
