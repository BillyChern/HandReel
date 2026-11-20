"""The retarget stage on a whisking slice with CPU MuJoCo: the float64 solve.

`run_retarget_stage` upcasts every pose, camera, joint and vertex input to float64 before the world conversion, the
upsampling and the kinematic solve, while `ClipRecord` keeps its float32 storage. The test: the reference solved from a
float32 `ClipRecord` equals, within 1e-9 in qpos, the reference of a float64 pipeline written directly in the world
frame (IK-site wrist target, `fastdo.retarget.rate` upsampling, `retarget_kinematic` at the reference rate) fed that
record's stored values.

Why the camera is an identity rotation with a float32 translation at the scene's median position. This reference is not
a well-conditioned function of its inputs. On this slice, a float64-valued record run through the stage moves qpos by
5e-8 to 3e-7 against the pipeline on the original float64 rows, whatever the camera: generic, a quarter turn (even when
every input round-trips exactly), or identity. Those gaps come from 1e-16 rounding, in
rotation-vector composition and in computing the IK-site offset in the camera frame rather than the world frame. So 1e-9
only means something when the stage and the pipeline see bitwise-identical float64 inputs:
- float32-stored values plus a float32 translation of the same scale are exact in float64 (28 significant bits at most);
- there is no rotation to compose;
- the camera-frame and world-frame joint differences are the same numbers.
A correct stage therefore reproduces the pipeline exactly. In float32 the same additions round a few percent of the
vertex coordinates (measured 2.5%; the test requires at least 1%), which this reference amplifies, so a stage that does
its conversion in float32 does not. Rotation handling in the conversion is covered by `tests/qc/test_export.py`.

Not asserted: what float32 storage costs. The float32 record's reference is 0.21 rad from the float64 rows' reference
on the whole clip; that conditioning question is left open here. The test logs the gap on this slice.

The slice is 16 engine rows at stride 3 (source frames 45-90), giving 46 reference frames at the clip's 50 Hz label. The
hand is the keypoints file's world-frame MANO vertices, global orient and wrist, with the MCP joints posed from MANO's
rest joints (`tests/retarget/mano_joints.py`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from fastdo.io.clip_record import HandTrack
from fastdo.qc.export import DEFAULT_GRAVITY_CAM, export_clip_record, run_retarget_stage
from fastdo.retarget.baseline_io import baseline_world_inputs
from fastdo.retarget.kinematic import RetargetConfig, retarget_kinematic
from fastdo.retarget.labels import mano_finger_labels
from fastdo.retarget.rate import upsample_hand_track, upsample_object_poses
from fastdo.retarget.wrist_target import ik_site_wrist_poses
from fastdo.solver.api import TrajectorySolution
from tests.retarget.mano_joints import mano_rest_joint_model, posed_mcp_joints
from tests.retarget.whisking_data import (
    KEYPOINTS_NPZ,
    MANO_RIGHT_PKL,
    MESH_SCALE,
    TRAJECTORY_NPZ,
    build_scene,
    build_whisk_sdf_and_surface,
    whisking_data_missing,
)

logger = logging.getLogger(__name__)
DT = torch.float64
ROWS = 45 + 3 * np.arange(16, dtype=np.int64)
KEYS = ROWS - ROWS[0]
N_REF = int(KEYS[-1]) + 1
TOL = 1e-9


@dataclass(frozen=True)
class HandRows:
    """Engine rows in one frame: object poses `(T,4,4)`, hand joints `(T,21,3)`, MANO global orient `(T,3)`, vertices
    `(T,778,3)`."""

    obj: Tensor
    joints: Tensor
    rotvec: Tensor
    verts: Tensor

    def translated(self, shift: Tensor) -> HandRows:
        """The rows moved by `shift (3,)`; rotations and the global orient are unchanged."""
        obj = self.obj.clone()
        obj[:, :3, 3] = obj[:, :3, 3] + shift
        return HandRows(obj=obj, joints=self.joints + shift, rotvec=self.rotvec, verts=self.verts + shift)


def _world_rows() -> tuple[HandRows, float]:
    obj, _, verts, frequency = baseline_world_inputs(KEYPOINTS_NPZ, TRAJECTORY_NPZ)
    assert MANO_RIGHT_PKL is not None
    with np.load(KEYPOINTS_NPZ) as data:
        wrist = np.asarray(data["qpos_wrist_right"], dtype=np.float64)[:, :3]
        rotvec = np.asarray(data["mano_global_orient_right"], dtype=np.float64)
        betas = np.asarray(data["mano_betas_right"], dtype=np.float64)
    joints = posed_mcp_joints(mano_rest_joint_model(MANO_RIGHT_PKL), rotvec, betas, wrist)
    index = torch.from_numpy(ROWS)
    return HandRows(obj=obj.to(DT)[index], joints=torch.from_numpy(joints)[index], rotvec=torch.from_numpy(rotvec)[index],
                    verts=verts.to(DT)[index]), frequency


@whisking_data_missing
def test_float32_clip_record_reproduces_the_float64_pipeline_on_its_stored_values() -> None:
    assert MANO_RIGHT_PKL is not None
    world, frequency = _world_rows()
    scene, labels, cfg = build_scene(), mano_finger_labels(MANO_RIGHT_PKL), RetargetConfig()
    sdf, surface, normals = build_whisk_sdf_and_surface()

    def pipeline(rows: HandRows) -> np.ndarray:
        wrist_up, verts_up = upsample_hand_track(ik_site_wrist_poses(rows.joints, rows.rotvec), rows.verts, KEYS, N_REF)
        obj_up = torch.from_numpy(upsample_object_poses(rows.obj, KEYS, N_REF))
        return np.asarray(retarget_kinematic(obj_up, torch.from_numpy(wrist_up), MESH_SCALE, torch.from_numpy(verts_up),
                                             labels, scene, sdf, surface, normals, frequency, cfg,
                                             frame_valid=torch.ones(N_REF, dtype=torch.bool)).qpos)

    t = int(ROWS.shape[0])
    positions = torch.cat([world.obj[:, :3, 3], world.verts.reshape(-1, 3)]).numpy()
    shift = torch.from_numpy(np.median(positions, axis=0).astype(np.float32)).to(DT)
    cam = world.translated(-shift)
    cam_to_world = torch.eye(4, dtype=DT).repeat(t, 1, 1)
    cam_to_world[:, :3, 3] = shift
    params = np.zeros((t, 48), dtype=np.float32)
    params[:, :3] = cam.rotvec.numpy()
    joints32 = cam.joints.numpy().astype(np.float32)
    hand = HandTrack(params=params, betas=np.zeros(10, dtype=np.float32), trans=joints32[:, 0].copy(), joints=joints32,
                     valid=np.ones(t, dtype=bool))
    solution = TrajectorySolution(poses=cam.obj, scale=MESH_SCALE, covariance=torch.zeros(t, 6, 6),
                                  per_frame_cost=torch.zeros(t), confidence=torch.ones(t), iterations=0, seconds=0.0)
    record = export_clip_record("whisking_slice", solution, "whisk.obj", torch.eye(3), cam_to_world,
                                torch.from_numpy(DEFAULT_GRAVITY_CAM), None, hand, 1.0, ROWS, ROWS / frequency)
    verts32 = cam.verts.float()
    assert record.object_pose.dtype == record.cam_to_world.dtype == joints32.dtype == np.float32
    assert np.array_equal(record.cam_to_world.astype(np.float64), cam_to_world.numpy())        # exact in float32
    rounded = (verts32 + shift.float()).to(DT) != verts32.to(DT) + shift        # float32 conversion would change these
    assert float(rounded.to(DT).mean()) >= 0.01

    out = run_retarget_stage(record, verts32, labels, scene, sdf, surface, normals, frequency, cfg)

    assert out is not None and out.qpos.shape == (N_REF, 35)
    stored = HandRows(obj=torch.from_numpy(record.object_pose).to(DT), joints=torch.from_numpy(joints32).to(DT),
                      rotvec=torch.from_numpy(params[:, :3].copy()).to(DT), verts=verts32.to(DT))
    gap = float(np.abs(out.qpos - pipeline(stored.translated(shift))).max())
    storage = float(np.abs(out.qpos - pipeline(world)).max())
    logger.info("float32 record: stage vs float64 pipeline on the stored values, max |qpos| gap %.3g; vs the float64 "
                "rows' reference %.3g (float32 storage); float32 addition rounds %.3g of the "
                "vertex coordinates", gap, storage, float(rounded.to(DT).mean()))
    assert gap <= TOL
