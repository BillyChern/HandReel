"""ClipRecord export and the world-frame retargeting stage hook.

`export_clip_record` assembles the per-clip `ClipRecord` from the object solve, the hands and the camera and validates
it against the existing schema, which has no defaults (so `gravity_cam` is required).

`run_retarget_stage` is the conversion step `retarget_kinematic` needs. That solver requires object poses, wrist poses
and hand vertices in one frame, the MuJoCo world frame; a camera-frame input once caused a severe regression (see
`fastdo.retarget.kinematic`). It runs in two parts, so that a caller can tell unusable inputs from a failed solve.

`prepare_retarget_stage`:
1. skips a clip whose mean tracker confidence is not positive, that has no hand track with a valid row, or that has no
   row with both a valid hand and a supported object (`None`; `retarget_skip_reason` names which);
2. rejects non-finite inputs (ValueError): `cam_to_world` or `object_pose` on any row (every row is a spline knot),
   the hand joints, global orient (for `ik_site`) or vertices on a valid row;
3. upcasts every pose, camera, joint and vertex input to float64 and derives camera-frame wrist targets
   from the hand track, `ik_site` by default (`fastdo.retarget.wrist_target`);
4. moves the object poses, wrist targets and hand vertices to the world frame with `record.cam_to_world`, per row
   (`pose_to_world`, `points_to_world`);
5. composes the reference-rate inputs from those engine rows and masks the invalid hand rows
   (`fastdo.qc.reference_rate.upsample_reference_inputs`).

`solve_retarget_stage` then calls `retarget_kinematic` once, at the reference rate, with `frame_valid`. The returned
reference is not resampled.

Camera frame when no stage estimates one. `DEFAULT_GRAVITY_CAM` = `[0, -1, 0]` is the up vector
`fastdo.io.datasets.dexycb.DexYCBReader` uses without a reliable estimate (an upright OpenCV camera).
`gravity_aligned_cam_to_world` builds a static camera-to-world transform from an up vector: the minimal rotation taking
it to world +z, no translation, the same for every frame. That assumes a static camera and puts the world origin at the
camera centre; placing the scene against a table plane (the baseline's `world_offset`) is not done here.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from fastdo.io.clip_record import ClipRecord, HandTrack
from fastdo.qc.reference_rate import (
    ReferenceInputs,
    reference_key_frames,
    upsample_reference_inputs,
)
from fastdo.retarget.kinematic import KinematicTrajectory, RetargetConfig, retarget_kinematic
from fastdo.retarget.mj_model import RobotScene
from fastdo.retarget.wrist_target import DEFAULT_WRIST_TARGET, WRIST_TARGETS, wrist_target_poses
from fastdo.solver.api import TrajectorySolution
from fastdo.solver.sdf import SdfGrid

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_GRAVITY_CAM",
    "export_clip_record",
    "gravity_aligned_cam_to_world",
    "points_to_world",
    "pose_to_world",
    "prepare_retarget_stage",
    "retarget_hand_side",
    "retarget_row_valid",
    "retarget_skip_reason",
    "run_retarget_stage",
    "solve_retarget_stage",
]

DEFAULT_GRAVITY_CAM = np.array([0.0, -1.0, 0.0], dtype=np.float32)
_DT = torch.float64
_SIDES = ("left", "right")


def _float32(x: Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, Tensor):
        return x.detach().cpu().numpy().astype(np.float32)
    return np.array(x, dtype=np.float32, copy=True)


def pose_to_world(cam_to_world: Tensor, pose_cam: Tensor) -> Tensor:
    """`(T,4,4)` `cam_to_world @ pose_cam`, per frame, in the promoted dtype of the two inputs.

    Raises:
        ValueError: Unless both are `(T,4,4)` with the same `T`.
    """
    if cam_to_world.dim() != 3 or tuple(cam_to_world.shape[1:]) != (4, 4) or pose_cam.shape != cam_to_world.shape:
        raise ValueError(f"expected two (T,4,4) stacks, got {tuple(cam_to_world.shape)} and {tuple(pose_cam.shape)}")
    dtype = torch.promote_types(cam_to_world.dtype, pose_cam.dtype)
    return cam_to_world.to(dtype) @ pose_cam.to(dtype)


def points_to_world(cam_to_world: Tensor, points_cam: Tensor) -> Tensor:
    """`(T,V,3)` points moved by each frame's rotation and translation of `cam_to_world (T,4,4)`.

    Raises:
        ValueError: Unless `points_cam` is `(T,V,3)` and `cam_to_world` is `(T,4,4)` with the same `T`.
    """
    if (cam_to_world.dim() != 3 or tuple(cam_to_world.shape[1:]) != (4, 4) or points_cam.dim() != 3
            or points_cam.shape[0] != cam_to_world.shape[0] or points_cam.shape[2] != 3):
        raise ValueError(f"expected (T,4,4) and (T,V,3), got {tuple(cam_to_world.shape)} and {tuple(points_cam.shape)}")
    dtype = torch.promote_types(cam_to_world.dtype, points_cam.dtype)
    c2w, points = cam_to_world.to(dtype), points_cam.to(dtype)
    return torch.einsum("tij,tvj->tvi", c2w[:, :3, :3], points) + c2w[:, :3, 3:4].transpose(-1, -2)


def gravity_aligned_cam_to_world(gravity_cam: Tensor | np.ndarray, n_frames: int) -> Tensor:
    """`(n_frames,4,4)` float64 static camera-to-world transform taking the up vector `gravity_cam` to world +z.

    The rotation is the minimal one (about `gravity_cam x z`); an up vector along -z turns by pi about x. No translation.

    Raises:
        ValueError: If `gravity_cam` is not a finite, non-zero 3-vector.
    """
    up = torch.as_tensor(np.asarray(gravity_cam, dtype=np.float64)).reshape(-1)
    norm = float(up.norm()) if up.numel() == 3 else 0.0
    if not (np.isfinite(norm) and norm > 1e-12):
        raise ValueError(f"gravity_cam must be a finite non-zero 3-vector, got {up.tolist()}")
    u = up / norm
    c = float(u[2])
    if c <= -1.0 + 1e-12:
        rot = torch.diag(torch.tensor([1.0, -1.0, -1.0], dtype=_DT))
    else:
        v = torch.linalg.cross(u, torch.tensor([0.0, 0.0, 1.0], dtype=_DT))
        k = torch.tensor([[0.0, -float(v[2]), float(v[1])], [float(v[2]), 0.0, -float(v[0])],
                          [-float(v[1]), float(v[0]), 0.0]], dtype=_DT)
        rot = torch.eye(3, dtype=_DT) + k + (k @ k) / (1.0 + c)
    out = torch.eye(4, dtype=_DT).repeat(n_frames, 1, 1)
    out[:, :3, :3] = rot
    return out


def export_clip_record(clip_id: str, solution: TrajectorySolution, mesh_path: str, K: Tensor, cam_to_world: Tensor,
                       gravity_cam: Tensor, hand_left: HandTrack | None, hand_right: HandTrack | None,
                       gate_score: float, frame_idx: np.ndarray, timestamps: np.ndarray, *,
                       path: Path | None = None) -> ClipRecord:
    """The clip's validated `ClipRecord` (float32 camera, pose and confidence arrays copied from the inputs).

    Args:
        clip_id: Clip identifier.
        solution: The object solve: `poses (T,4,4)` object-to-camera, `scale`, per-frame `confidence (T,)`.
        mesh_path: The canonical object mesh (the shape stage's `mesh.obj`).
        K: `(3,3)` pixel intrinsics.
        cam_to_world: `(T,4,4)` camera-to-world transforms.
        gravity_cam: `(3,)` up vector in the camera frame of frame 0 (`DEFAULT_GRAVITY_CAM` without an estimate).
        hand_left, hand_right: `HandTrack`s with the solution's `T` rows, or None.
        gate_score: The interaction gate's score.
        frame_idx: `(T,)` source frame indices.
        timestamps: `(T,)` seconds.
        path: When given, the record is also saved there.

    Raises:
        ValueError: If any field breaks the schema (`ClipRecord.validate`), e.g. hand rows that are not the solution's.
    """
    record = ClipRecord(
        clip_id=str(clip_id), frame_idx=np.array(frame_idx, dtype=np.int64, copy=True),
        timestamps=np.array(timestamps, dtype=np.float64, copy=True), intrinsics=_float32(K),
        cam_to_world=_float32(cam_to_world), object_mesh_path=str(mesh_path), object_pose=_float32(solution.poses),
        object_scale=float(solution.scale), gravity_cam=_float32(gravity_cam), confidence=_float32(solution.confidence),
        gate_score=float(gate_score), hand_left=hand_left, hand_right=hand_right,
    )
    record.validate()
    if path is not None:
        record.save(Path(path))
    return record


def retarget_hand_side(record: ClipRecord, side: str | None) -> tuple[str, HandTrack | None]:
    """`(side, track)`: the requested side (None: right when the record has it, else left) and its `HandTrack`.

    Raises:
        ValueError: If `side` is neither None, "left" nor "right".
    """
    if side is None:
        side = "left" if record.hand_right is None and record.hand_left is not None else "right"
    if side not in _SIDES:
        raise ValueError(f"side must be one of {_SIDES} or None, got {side!r}")
    return side, record.hand_right if side == "right" else record.hand_left


def _require_finite(values: np.ndarray | Tensor, valid: np.ndarray | None, name: str) -> None:
    """ValueError naming `name` when a row (a valid row, when `valid` is given) has a non-finite entry."""
    array = values.detach().cpu().numpy() if isinstance(values, Tensor) else np.asarray(values)
    finite = np.isfinite(array.reshape(array.shape[0], -1)).all(axis=1)
    bad = np.flatnonzero(~finite if valid is None else valid & ~finite)
    if bad.size:
        raise ValueError(f"{name} is non-finite on {'rows' if valid is None else 'valid rows'} {bad[:6].tolist()}")


MIN_BRIDGED_DEPTH_M = 0.05     # a bridged pose must put the object this far in front of the camera


def retarget_row_valid(record: ClipRecord, track_valid: np.ndarray, max_bridge: int = 1) -> np.ndarray:
    """`(T,)` rows the retargeting may use: the hand row is valid AND the object is supported there.

    Supported = observed (`confidence > 0`) OR bridged: an INTERIOR run of at most `max_bridge` unobserved rows, between
    two observed rows, whose object is in front of the camera (depth > `MIN_BRIDGED_DEPTH_M`).

    The solver fills rows with fewer than `MIN_VALID_OBS` pointmap points from its temporal priors and reports them with
    `confidence == 0` (`fastdo.solver.api`). Across a long gap or past either end of the clip that pose is an
    EXTRAPOLATION (~30% of an Ego4D pilot's rows, some behind the camera), so it is masked like a missing hand.
    A single row between two observed rows is an INTERPOLATION over 0.2 s — 5 Hz keys add ~1-3 mm — which is
    what geometry at `row_stride` 2 produces on every other row (measured non-inferior); masking those
    would drop half the clip.
    """
    conf = np.asarray(record.confidence)
    observed = conf > 0.0
    supported = observed.copy()
    depth = np.asarray(record.object_pose)[:, 2, 3]
    n, row = len(observed), 0
    while row < n:
        if observed[row]:
            row += 1
            continue
        end = row
        while end < n and not observed[end]:
            end += 1
        interior = row > 0 and end < n
        if interior and end - row <= max_bridge and bool((depth[row:end] > MIN_BRIDGED_DEPTH_M).all()):
            supported[row:end] = True
        row = end
    return np.asarray(track_valid, dtype=bool) & supported


def retarget_skip_reason(record: ClipRecord, side: str | None = None) -> str | None:
    """Why `prepare_retarget_stage(record, ..., side=side)` skips the clip (the text a skipped `retarget.json` records),
    or None when it does not: the mean tracker confidence is not positive; the side has no valid hand row; or no row
    has both a valid hand and a supported object (`retarget_row_valid`). Checked in that order.

    Raises:
        ValueError: If `side` is neither None, "left" nor "right".
    """
    confidence = float(np.mean(record.confidence)) if record.n_frames else 0.0
    if not confidence > 0.0:
        return "mean tracker confidence is not positive"
    side, track = retarget_hand_side(record, side)
    if track is None or not bool(np.any(track.valid)):
        return f"no valid {side} hand row"
    if not bool(retarget_row_valid(record, track.valid).any()):
        return f"no row with both a valid {side} hand and an observed (or bridged) object"
    return None


def prepare_retarget_stage(record: ClipRecord, hand_verts_cam: Tensor, *, side: str | None = None,
                           wrist_target: str = DEFAULT_WRIST_TARGET) -> ReferenceInputs | None:
    """The world-frame, reference-rate solver inputs (module docstring, steps 1-5), or `None` when skipped.

    Args:
        record: The accepted clip's `ClipRecord` (engine rows).
        hand_verts_cam: `(T,V,3)` camera-frame MANO vertices of `side`, the hands stage's `{side}_vertices`, one row per
            record row.
        side: Hand to retarget; None picks right when present, else left.
        wrist_target: Wrist-prior target convention, `ik_site` (default) or `mcp` (`fastdo.retarget.wrist_target`).

    Raises:
        ValueError: On an unknown `wrist_target`, `hand_verts_cam` without the record's rows, a `record.frame_idx`
            that does not increase, or a non-finite input (step 2).
    """
    if wrist_target not in WRIST_TARGETS:
        raise ValueError(f"wrist_target must be one of {WRIST_TARGETS}, got {wrist_target!r}")
    t = record.n_frames
    reason = retarget_skip_reason(record, side)
    if reason is not None and reason.startswith("mean tracker confidence"):
        logger.info("retarget stage: clip %r skipped, %s", record.clip_id, reason)
        return None
    side, track = retarget_hand_side(record, side)
    if hand_verts_cam.dim() != 3 or hand_verts_cam.shape[0] != t or hand_verts_cam.shape[2] != 3:
        raise ValueError(f"hand_verts_cam must be (T,V,3) with the record's T = {t} rows (hands.npz rows are the decode "
                         f"rows), got {tuple(hand_verts_cam.shape)}")
    if reason is not None or track is None:
        logger.warning("retarget stage: clip %r skipped: %s", record.clip_id, reason)
        return None
    valid = retarget_row_valid(record, track.valid)
    rotvec = np.ascontiguousarray(track.params[:, :3])
    _require_finite(record.cam_to_world, None, "cam_to_world")
    _require_finite(record.object_pose, None, "object_pose")
    _require_finite(track.joints, valid, f"hand_{side}.joints")
    if wrist_target == "ik_site":
        _require_finite(rotvec, valid, f"hand_{side}.params[:, :3] (global orient)")
    _require_finite(hand_verts_cam, valid, "hand_verts_cam")
    c2w = torch.from_numpy(record.cam_to_world).to(_DT)
    wrist_cam = wrist_target_poses(wrist_target, torch.from_numpy(track.joints).to(_DT),
                                   torch.from_numpy(rotvec).to(_DT), is_right=side == "right", valid=valid)
    inputs = upsample_reference_inputs(pose_to_world(c2w, torch.from_numpy(record.object_pose).to(_DT)),
                                       pose_to_world(c2w, wrist_cam), points_to_world(c2w, hand_verts_cam.to(_DT)),
                                       valid, reference_key_frames(record.frame_idx))
    logger.info("retarget stage: clip %r, %s hand, %s wrist target, %d engine rows -> %d reference frames; masked %d "
                "rows (%d reference frames)", record.clip_id, side, wrist_target, t, inputs.n_frames,
                inputs.masked_rows, inputs.masked_frames)
    return inputs


def solve_retarget_stage(record: ClipRecord, inputs: ReferenceInputs, hand_labels: Tensor, scene: RobotScene,
                         sdf: SdfGrid, surface: Tensor, surface_normals: Tensor, frequency: float,
                         cfg: RetargetConfig) -> KinematicTrajectory:
    """`retarget_kinematic` once on `prepare_retarget_stage`'s inputs, with their `frame_valid` (module docstring).

    Args:
        record: The clip's `ClipRecord` (for `object_scale`).
        inputs: `prepare_retarget_stage(record, ...)`.
        hand_labels: `(V,)` int8 finger label per vertex (`mano_finger_labels`).
        scene: The robot scene.
        sdf: Canonical-unit object SDF grid.
        surface: `(M,3)` canonical surface samples.
        surface_normals: `(M,3)` their outward normals.
        frequency: Reference rate in Hz, one reference frame per source frame (for `qvel`).
        cfg: Kinematic retargeting weights and schedule.
    """
    return retarget_kinematic(inputs.obj_poses, inputs.wrist_poses, record.object_scale, inputs.hand_verts, hand_labels,
                              scene, sdf, surface, surface_normals, frequency, cfg, frame_valid=inputs.frame_valid)


def run_retarget_stage(record: ClipRecord, hand_verts_cam: Tensor, hand_labels: Tensor, scene: RobotScene,
                       sdf: SdfGrid, surface: Tensor, surface_normals: Tensor, frequency: float, cfg: RetargetConfig,
                       *, side: str | None = None,
                       wrist_target: str = DEFAULT_WRIST_TARGET) -> KinematicTrajectory | None:
    """The clip's full-rate kinematic reference: `prepare_retarget_stage`, then `solve_retarget_stage`; `None` when
    skipped. Arguments and errors as in those two functions."""
    inputs = prepare_retarget_stage(record, hand_verts_cam, side=side, wrist_target=wrist_target)
    if inputs is None:
        return None
    return solve_retarget_stage(record, inputs, hand_labels, scene, sdf, surface, surface_normals, frequency, cfg)
