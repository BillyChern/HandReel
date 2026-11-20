"""A synthetic engine clip directory for the robot-path tests, generated FROM the robot.

The video world is the robot frame itself (up +z, camera looking along +x, static), so with `table_xy` set to the
object's start position the placement is the identity up to the 1 mm table clearance. The right hand's joints are
built so that `mano_hand_frame` and the wrist equal the Dex3 base site of a known arm path, with Dex3's own FK
fingertips at a known hand pose: the kinematic path should give back both. The object, a 6 cm box, slides 10 cm in +y
in the reference, but the hand path never touches it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from fastdo.io.clip_record import ClipRecord, HandTrack
from fastdo.robot.arm_ik import site_pose
from fastdo.robot.dex3_retarget import dex3_tip_positions
from fastdo.robot.g1_model import G1Upper

TABLE_Z = 0.80
OBJECT_XY = (0.40, -0.05)
HAND_Q = np.array([0.1, -0.2, -0.5, 0.4, 0.3, 0.5, 0.4])
N_ROWS = 20


def arm_path(n: int = N_ROWS) -> np.ndarray:
    """`(n,7)` smooth right-arm path whose hand stays above the table and away from the body."""
    s = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, n))
    q0 = np.array([-0.6, -0.3, 0.0, 0.3, 0.0, 0.0, 0.0])
    q1 = np.array([-0.8, -0.5, 0.2, 0.2, 0.3, 0.2, 0.0])
    return q0 + s[:, None] * (q1 - q0)


def hand_joints(g1: G1Upper, arm: np.ndarray, hand_q: np.ndarray = HAND_Q) -> np.ndarray:
    """`(T,21,3)` OpenPose joints: MCPs placing `mano_hand_frame` on the base site, tips at Dex3's FK tips of
    `hand_q` (`(7,)` for every row, or `(T,7)` per row)."""
    poses = site_pose(g1, arm)
    hq = np.broadcast_to(np.asarray(hand_q, dtype=np.float64), (len(arm), 7))
    tips = dex3_tip_positions(g1, hq)[:, 1:]                                  # (T,3,3) thumb, index, middle
    fixed = {0: [0.0, 0.0, 0.0], 9: [0.0, -0.09, 0.0], 5: [0.0, -0.085, -0.02], 17: [0.0, -0.07, 0.04]}
    local = {k: np.broadcast_to(np.asarray(v), (len(arm), 3)) for k, v in fixed.items()}
    local.update({4: tips[:, 0], 8: tips[:, 1], 12: tips[:, 2]})
    joints = np.zeros((len(arm), 21, 3))
    for k, p in local.items():
        joints[:, k] = np.einsum("tij,tj->ti", poses[:, :3, :3], p) + poses[:, :3, 3]
    for k in range(21):                       # remaining joints: on the wrist (never read by the robot path)
        if k not in local:
            joints[:, k] = poses[:, :3, 3]
    return joints


def contact_object_path(g1: G1Upper, contact_row: int, drift: float, n: int = N_ROWS) -> np.ndarray:
    """`(n,4,4)` world poses of the 6 cm box for a clip whose index tip TOUCHES it from `contact_row` on: the top face
    1 mm under the index tip (the box is carried), and before that the box at its contact pose shifted by
    `drift * (contact_row - t) / contact_row` in +x (the pre-contact reconstruction drift of a smoke clip)."""
    tips = hand_joints(g1, arm_path(n))[:, 8]
    obj = np.tile(np.eye(4), (n, 1, 1))
    obj[:, :3, 3] = tips - np.array([0.0, 0.0, 0.5 * 0.06 + 0.001])
    obj[:contact_row, :3, 3] = obj[contact_row, :3, 3]
    obj[:contact_row, 0, 3] += drift * (contact_row - np.arange(contact_row)) / contact_row
    return obj


def write_clip(g1: G1Upper, root: Path, object_world: np.ndarray | None = None,
               confidence: np.ndarray | None = None, hand_q: np.ndarray = HAND_Q) -> Path:
    """Write `clip_record.npz` and `mesh.obj` (no `hands.npz`: tips then come from the joints) under `root`. The box
    slides 10 cm in +y on the table unless `object_world` `(N_ROWS,4,4)` gives its world poses; `confidence`
    `(N_ROWS,)` (default 1: every row observed); `hand_q` the Dex3 pose the tips come from (`(7,)` or per row)."""
    root.mkdir(parents=True, exist_ok=True)
    box = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    box.export(root / "mesh.obj")
    scale = 0.06
    t = N_ROWS
    c2w = np.eye(4)
    c2w[:3, :3] = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])   # cam z -> +x, cam -y -> +z
    obj_w = np.tile(np.eye(4), (t, 1, 1))
    obj_w[:, 0, 3] = OBJECT_XY[0]
    obj_w[:, 1, 3] = OBJECT_XY[1] + np.linspace(0.0, 0.10, t)
    obj_w[:, 2, 3] = TABLE_Z + 0.5 * scale + 0.001
    if object_world is not None:
        obj_w = np.asarray(object_world, dtype=np.float64)
    obj_cam = np.einsum("ij,tjk->tik", np.linalg.inv(c2w), obj_w)
    joints_cam = np.einsum("ij,tvj->tvi", c2w[:3, :3].T, hand_joints(g1, arm_path(t), hand_q) - c2w[:3, 3])
    hand = HandTrack(params=np.zeros((t, 48), np.float32), betas=np.zeros(10, np.float32),
                     trans=joints_cam[:, 0].astype(np.float32), joints=joints_cam.astype(np.float32),
                     valid=np.ones(t, bool))
    rec = ClipRecord(clip_id="synthetic:k000", frame_idx=np.arange(t, dtype=np.int64) * 3,
                     timestamps=np.arange(t, dtype=np.float64) * 0.1, intrinsics=np.eye(3, dtype=np.float32),
                     cam_to_world=np.repeat(c2w[None], t, axis=0).astype(np.float32), object_mesh_path="mesh.obj",
                     object_pose=obj_cam.astype(np.float32), object_scale=scale,
                     gravity_cam=(c2w[:3, :3].T @ [0.0, 0.0, 1.0]).astype(np.float32),
                     confidence=(np.ones(t) if confidence is None else np.asarray(confidence)).astype(np.float32),
                     gate_score=1.0, hand_left=None, hand_right=hand)
    rec.save(root / "clip_record.npz")
    return root
