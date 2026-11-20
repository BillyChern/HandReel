"""DexYCB world frame from the calibration extrinsics (for the DexYCB policy references).

Convention, checked on the data in `tests/io/test_dexycb_world.py`. `calibration/extrinsics_<id>/extrinsics.yml`, with
`<id>` from the sequence's `meta.yml` `extrinsics`, holds under `extrinsics`:
- one row-major 3x4 `[R | t]` per camera serial, the camera-to-master transform; the `master` camera's own entry is the
  identity;
- under `apriltag`, the tag-to-master transform of the AprilTag on the table.

dex-ycb-toolkit's `sequence_loader.py` builds its point clouds the same way: `R p + t` per serial, then the inverse of
`apriltag` to put the table on the xy-plane.

The world frame here is that tag frame: `T_world_cam = T_master_tag^-1 T_master_cam`. Measured on subject 03's calibration:
- every camera sits 0.07 to 0.89 m above the tag plane, so world z points up;
- the grasped object rests on z = 0 at frame 0 (lowest vertex at -1.8 mm);
- the GT object pose from all 8 views agrees in this frame to 1.3e-7 m.

It is not D6's static-camera fallback (`fastdo.qc.export.gravity_aligned_cam_to_world` of the `[0, -1, 0]` up vector),
which has no translation and no calibrated yaw.

The rig is calibrated and static, so the frame is one transform per sequence and serial. `with_world_frame` stores it in
a `ClipRecord`: `cam_to_world` holds one float32 row per frame, and `gravity_cam` is world +z in the camera frame.
`select_rows` keeps a subset of a record's rows, e.g. the engine's stride-3 decode rows.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from fastdo.io.clip_record import ClipRecord, HandTrack
from fastdo.io.datasets.dexycb import load_dexycb_yaml

__all__ = ["WORLD_FRAME", "dexycb_cam_to_world", "load_dexycb_extrinsics", "select_rows", "with_world_frame"]

WORLD_FRAME = "dexycb_apriltag"
_UP = np.array([0.0, 0.0, 1.0])


def _homogeneous(values: Sequence[float]) -> np.ndarray:
    rows = np.asarray(values, dtype=np.float64).reshape(-1)
    if rows.shape != (12,):
        raise ValueError(f"an extrinsics entry holds 12 values ([R | t] row-major), got {rows.shape[0]}")
    out = np.eye(4)
    out[:3, :4] = rows.reshape(3, 4)
    return out


def load_dexycb_extrinsics(root: str | Path, extrinsics_id: str) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """`({serial: (4,4) camera-to-master}, (4,4) tag-to-master)`, float64, from one calibration (module docstring).

    Raises:
        ValueError: If the file has no `apriltag` entry or an entry is not 12 numbers.
    """
    path = Path(root) / "calibration" / f"extrinsics_{extrinsics_id}" / "extrinsics.yml"
    entries = load_dexycb_yaml(path)["extrinsics"]
    if "apriltag" not in entries:
        raise ValueError(f"{path}: no apriltag transform, so no table (world) frame")
    cams = {str(name): _homogeneous(values) for name, values in entries.items() if str(name) != "apriltag"}
    return cams, _homogeneous(entries["apriltag"])


def dexycb_cam_to_world(root: str | Path, seq: str) -> np.ndarray:
    """`(4,4)` float64 camera-to-world transform of `"<subject>/<sequence>/<serial>"` (module docstring).

    Raises:
        ValueError: If the id is malformed, `meta.yml` names no extrinsics, or the calibration lacks the serial.
    """
    parts = seq.strip("/").split("/")
    if len(parts) != 3:
        raise ValueError(f"expected <subject>/<sequence>/<serial>, got {seq!r}")
    subject, sequence, serial = parts
    meta = load_dexycb_yaml(Path(root) / subject / sequence / "meta.yml")
    extrinsics_id = meta.get("extrinsics")
    if extrinsics_id is None:
        raise ValueError(f"{seq}: meta.yml names no extrinsics calibration")
    cams, tag_to_master = load_dexycb_extrinsics(root, str(extrinsics_id))
    if serial not in cams:
        raise ValueError(f"{seq}: calibration extrinsics_{extrinsics_id} has no camera {serial}")
    return np.asarray(np.linalg.solve(tag_to_master, cams[serial]), dtype=np.float64)


def with_world_frame(record: ClipRecord, cam_to_world: np.ndarray) -> ClipRecord:
    """`record` with the static `cam_to_world (4,4)` on every row (float32) and `gravity_cam` = world +z in the camera
    frame; nothing else changes and `record` is not modified.

    Raises:
        ValueError: Unless `cam_to_world` is a finite (4, 4) transform.
    """
    c2w = np.asarray(cam_to_world, dtype=np.float64)
    if c2w.shape != (4, 4) or not bool(np.isfinite(c2w).all()):
        raise ValueError(f"cam_to_world must be a finite (4, 4) transform, got shape {c2w.shape}")
    out = dataclasses.replace(record, cam_to_world=np.tile(c2w.astype(np.float32), (record.n_frames, 1, 1)),
                              gravity_cam=(c2w[:3, :3].T @ _UP).astype(np.float32))
    out.validate()
    return out


def _hand_rows(track: HandTrack | None, rows: np.ndarray) -> HandTrack | None:
    if track is None:
        return None
    return HandTrack(params=track.params[rows], betas=track.betas, trans=track.trans[rows], joints=track.joints[rows],
                     valid=track.valid[rows])


def select_rows(record: ClipRecord, rows: np.ndarray) -> ClipRecord:
    """`record` restricted to `rows` (every per-row array and hand track); per-clip fields stay.

    Raises:
        ValueError: Unless `rows` is non-empty, strictly increases and lies in [0, T).
    """
    idx = np.asarray(rows, dtype=np.int64).reshape(-1)
    if idx.size == 0 or bool(np.any(np.diff(idx) <= 0)) or int(idx[0]) < 0 or int(idx[-1]) >= record.n_frames:
        raise ValueError(f"rows must strictly increase within [0, {record.n_frames}), got {idx[:8].tolist()}")
    out = dataclasses.replace(record, frame_idx=record.frame_idx[idx], timestamps=record.timestamps[idx],
                              cam_to_world=record.cam_to_world[idx], object_pose=record.object_pose[idx],
                              confidence=record.confidence[idx], hand_left=_hand_rows(record.hand_left, idx),
                              hand_right=_hand_rows(record.hand_right, idx))
    out.validate()
    return out
