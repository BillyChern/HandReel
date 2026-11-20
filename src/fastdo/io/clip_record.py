"""ClipRecord: the unified per-clip output schema (hands + object trajectory)."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

_HAND_FIELDS = ("params", "betas", "trans", "joints", "valid")


@dataclass(frozen=True)
class HandTrack:
    """MANO trajectory of one hand. params = global orient (3) + 15 joints axis-angle (45)."""

    params: np.ndarray   # (T, 48) float32
    betas: np.ndarray    # (10,) float32
    trans: np.ndarray    # (T, 3) float32, camera frame
    joints: np.ndarray   # (T, 21, 3) float32, camera frame
    valid: np.ndarray    # (T,) bool

    def validate(self, t: int, prefix: str) -> None:
        _expect(self.params, (t, 48), np.dtype(np.float32), f"{prefix}.params")
        _expect(self.betas, (10,), np.dtype(np.float32), f"{prefix}.betas")
        _expect(self.trans, (t, 3), np.dtype(np.float32), f"{prefix}.trans")
        _expect(self.joints, (t, 21, 3), np.dtype(np.float32), f"{prefix}.joints")
        _expect(self.valid, (t,), np.dtype(np.bool_), f"{prefix}.valid")


@dataclass(frozen=True)
class ClipRecord:
    """Everything the retargeting stage needs from one clip."""

    clip_id: str
    frame_idx: np.ndarray        # (T,) int64 source frame indices
    timestamps: np.ndarray       # (T,) float64 seconds
    intrinsics: np.ndarray       # (3, 3) float32
    cam_to_world: np.ndarray     # (T, 4, 4) float32
    object_mesh_path: str
    object_pose: np.ndarray      # (T, 4, 4) float32 object-to-camera
    object_scale: float
    gravity_cam: np.ndarray      # (3,) float32 up-vector in camera frame of frame 0
    confidence: np.ndarray       # (T,) float32 per-frame tracker confidence
    gate_score: float
    hand_left: HandTrack | None
    hand_right: HandTrack | None

    @property
    def n_frames(self) -> int:
        return int(self.frame_idx.shape[0])

    def validate(self) -> None:
        t = self.n_frames
        _expect(self.frame_idx, (t,), np.dtype(np.int64), "frame_idx")
        _expect(self.timestamps, (t,), np.dtype(np.float64), "timestamps")
        _expect(self.intrinsics, (3, 3), np.dtype(np.float32), "intrinsics")
        _expect(self.cam_to_world, (t, 4, 4), np.dtype(np.float32), "cam_to_world")
        _expect(self.object_pose, (t, 4, 4), np.dtype(np.float32), "object_pose")
        _expect(self.gravity_cam, (3,), np.dtype(np.float32), "gravity_cam")
        _expect(self.confidence, (t,), np.dtype(np.float32), "confidence")
        for side, hand in (("hand_left", self.hand_left), ("hand_right", self.hand_right)):
            if hand is not None:
                hand.validate(t, side)

    def save(self, path: Path) -> None:
        self.validate()
        arrays: dict[str, np.ndarray] = {}
        for field in fields(self):
            val = getattr(self, field.name)
            if isinstance(val, np.ndarray):
                arrays[field.name] = val
            elif isinstance(val, HandTrack):
                for h in _HAND_FIELDS:
                    arrays[f"{field.name}.{h}"] = getattr(val, h)
            elif val is not None:
                arrays[field.name] = np.array(val)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            np.savez_compressed(f, **arrays)  # type: ignore[arg-type]

    @classmethod
    def load(cls, path: Path) -> ClipRecord:
        with np.load(path, allow_pickle=False) as z:
            data = {k: z[k] for k in z.files}
        hands: dict[str, HandTrack | None] = {}
        for side in ("hand_left", "hand_right"):
            keys = {h: f"{side}.{h}" for h in _HAND_FIELDS}
            hands[side] = HandTrack(**{h: data[k] for h, k in keys.items()}) if keys["params"] in data else None
        rec = cls(
            clip_id=str(data["clip_id"]), frame_idx=data["frame_idx"], timestamps=data["timestamps"],
            intrinsics=data["intrinsics"], cam_to_world=data["cam_to_world"],
            object_mesh_path=str(data["object_mesh_path"]), object_pose=data["object_pose"],
            object_scale=float(data["object_scale"]), gravity_cam=data["gravity_cam"],
            confidence=data["confidence"], gate_score=float(data["gate_score"]),
            hand_left=hands["hand_left"], hand_right=hands["hand_right"],
        )
        rec.validate()
        return rec


def _expect(arr: np.ndarray, shape: tuple, dtype: np.dtype, name: str) -> None:
    if tuple(arr.shape) != tuple(shape):
        raise ValueError(f"{name}: expected shape {shape}, got {tuple(arr.shape)}")
    if arr.dtype != dtype:
        raise ValueError(f"{name}: expected dtype {dtype}, got {arr.dtype}")
