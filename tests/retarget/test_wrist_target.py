"""Tests for `fastdo.retarget.wrist_target`: the baseline IK-site wrist target and the
MCP geometric frame.

- Synthetic hands (no data). A rigid MANO-like hand's IK-site target is its geometric frame. With joint noise the
  offset equals the baseline's quaternion average (`process_dataset.py::_wrist_frame_offset`, copied below as the
  parity reference), and the target is frame-equivariant. Only valid rows count. Dispatch and errors.
- Whisking (data). The IK-site target of the keypoints file's world-frame hand reproduces the shipped
  `qpos_wrist_right`, which C3b/C3c solved against and C4 validated in physics. That file has no MCP joints, so they are
  posed from MANO's rest joints (`tests/retarget/mano_joints.py`), a joint model first checked against HaWoR's own
  joints in `all_hand_meshes.npz`.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from fastdo.retarget.kinematic_terms import human_wrist_poses
from fastdo.retarget.wrist_target import (
    DEFAULT_WRIST_TARGET,
    WRIST_TARGETS,
    ik_site_wrist_poses,
    wrist_frame_offset,
    wrist_target_poses,
)
from tests.retarget.mano_joints import (
    MCP_OPENPOSE,
    mano_rest_joint_model,
    posed_mcp_joints,
    rest_joints,
)
from tests.retarget.whisking_data import BASELINE_ROOT, KEYPOINTS_NPZ, MANO_RIGHT_PKL

DT = torch.float64
HANDS_NPZ = BASELINE_ROOT / "reconstruction" / "whisking" / "whisking" / "all_hand_meshes.npz"


# _baseline_offset is adapted from DO AS I DO (https://github.com/malik-group/do-as-i-do, commit
# 824591b, retargeting/retargeting/pipeline/process_dataset.py), Copyright (c) 2026 malik-group,
# MIT License; see licenses/DO-AS-I-DO-MIT.txt.
def _baseline_offset(joints: np.ndarray, rotvec: np.ndarray, is_right: bool) -> Rotation:
    """`process_dataset.py::compute_wrist_rotation` + `_wrist_frame_offset`, verbatim apart from names."""
    def _norm(v: np.ndarray) -> np.ndarray:
        return v / np.linalg.norm(v, axis=-1, keepdims=True)

    z = _norm(joints[..., 9, :] - joints[..., 0, :])
    y_src = joints[..., 5, :] - joints[..., 13, :] if is_right else joints[..., 13, :] - joints[..., 5, :]
    x = _norm(np.cross(_norm(y_src), z))
    r_geom = Rotation.from_matrix(np.stack([x, _norm(np.cross(z, x)), z], axis=-1))
    q = (Rotation.from_rotvec(rotvec).inv() * r_geom).as_quat()
    q = q * np.sign(q @ q[0])[:, None]
    _, eigvecs = np.linalg.eigh(q.T @ q)
    return Rotation.from_quat(eigvecs[:, -1])


def _rigid_hand(t: int, seed: int, noise_m: float = 0.0) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """`(joints (t,21,3), rotvec (t,3), rest (21,3))`: one rest hand moved by random rigid motions, plus joint noise."""
    rng = np.random.default_rng(seed)
    rest = 0.04 * rng.normal(size=(21, 3))
    rest[0] = 0.0
    rest[MCP_OPENPOSE] = [[0.02, 0.005, 0.085], [0.0, 0.0, 0.09], [-0.02, 0.004, 0.084]]
    rot = Rotation.random(t, random_state=seed)
    trans = rng.normal(size=(t, 3))
    joints = np.einsum("tij,vj->tvi", rot.as_matrix(), rest) + trans[:, None] + noise_m * rng.normal(size=(t, 21, 3))
    return torch.from_numpy(joints), torch.from_numpy(rot.as_rotvec()), rest


@pytest.mark.parametrize("is_right", [True, False])
def test_ik_site_target_of_a_rigid_hand_is_its_geometric_frame(is_right: bool) -> None:
    joints, rotvec, rest = _rigid_hand(9, 1)
    target = ik_site_wrist_poses(joints, rotvec, is_right=is_right)
    geometric = human_wrist_poses(joints, is_right=is_right)
    assert target.dtype == DT and target.shape == (9, 4, 4)
    assert float((target - geometric).abs().max()) <= 1e-12
    rest_frame = human_wrist_poses(torch.from_numpy(rest[None]), is_right=is_right)[0, :3, :3].numpy()
    assert np.abs(wrist_frame_offset(joints, rotvec, is_right=is_right) - rest_frame).max() <= 1e-12


def test_ik_site_offset_is_the_baseline_average_and_the_target_is_frame_equivariant() -> None:
    joints, rotvec, _ = _rigid_hand(12, 2, noise_m=0.003)
    offset = wrist_frame_offset(joints, rotvec)
    assert np.abs(offset - _baseline_offset(joints.numpy(), rotvec.numpy(), True).as_matrix()).max() <= 1e-12
    target = ik_site_wrist_poses(joints, rotvec)
    fixed = np.einsum("tji,tjk->tik", Rotation.from_rotvec(rotvec.numpy()).as_matrix(), target[:, :3, :3].numpy())
    assert np.abs(fixed - offset).max() <= 1e-12 and torch.equal(target[:, :3, 3], joints[:, 0])
    assert float((target[:, :3, :3] - human_wrist_poses(joints)[:, :3, :3]).abs().max()) > 1e-3   # noise: not R_geom

    motion = torch.from_numpy(np.eye(4))
    motion[:3, :3] = torch.from_numpy(Rotation.from_rotvec([0.4, -1.1, 0.7]).as_matrix())
    motion[:3, 3] = torch.tensor([0.3, -0.2, 1.5], dtype=DT)
    moved_joints = joints @ motion[:3, :3].T + motion[:3, 3]
    moved_rotvec = (Rotation.from_matrix(motion[:3, :3].numpy()) * Rotation.from_rotvec(rotvec.numpy())).as_rotvec()
    moved = ik_site_wrist_poses(moved_joints, torch.from_numpy(moved_rotvec))
    assert float((moved - motion @ target).abs().max()) <= 1e-10


def test_ik_site_target_uses_only_valid_rows() -> None:
    joints, rotvec, _ = _rigid_hand(8, 3, noise_m=0.003)
    valid = np.array([True, False, True, True, False, True, True, True])
    bad_joints, bad_rotvec = joints.clone(), rotvec.clone()
    bad_joints[~torch.from_numpy(valid)] = float("nan")
    bad_rotvec[1] = 40.0
    target = ik_site_wrist_poses(bad_joints, bad_rotvec, valid=valid)
    rows = torch.from_numpy(np.flatnonzero(valid))
    assert torch.equal(target[rows], ik_site_wrist_poses(joints[rows], rotvec[rows]))
    assert torch.equal(target[[1, 4]], torch.eye(4, dtype=DT).expand(2, 4, 4))

    with pytest.raises(ValueError, match="no valid row"):
        ik_site_wrist_poses(joints, rotvec, valid=np.zeros(8, bool))
    with pytest.raises(ValueError, match="joints is non-finite on valid rows \\[1\\]"):
        wrist_frame_offset(bad_joints, rotvec, valid=valid | (np.arange(8) == 1))
    nan_rotvec = rotvec.clone()
    nan_rotvec[6, 2] = float("nan")
    with pytest.raises(ValueError, match="rotvec is non-finite on valid rows \\[6\\]"):
        ik_site_wrist_poses(joints, nan_rotvec, valid=valid)
    for bad_args in ((joints[:, :20], rotvec), (joints, rotvec[:-1]), (joints[0], rotvec[0])):
        with pytest.raises(ValueError, match="expected joints"):
            ik_site_wrist_poses(*bad_args)
    with pytest.raises(ValueError, match="validity flags"):
        ik_site_wrist_poses(joints, rotvec, valid=valid[:-1])


def test_wrist_target_poses_dispatches_on_the_convention() -> None:
    joints, rotvec, _ = _rigid_hand(5, 4, noise_m=0.003)
    assert WRIST_TARGETS == ("ik_site", "mcp") and DEFAULT_WRIST_TARGET == "ik_site"
    assert torch.equal(wrist_target_poses("mcp", joints, rotvec, is_right=False), human_wrist_poses(joints, is_right=False))
    assert torch.equal(wrist_target_poses("ik_site", joints, rotvec), ik_site_wrist_poses(joints, rotvec))
    with pytest.raises(ValueError, match="wrist target"):
        wrist_target_poses("palm", joints, rotvec)


@pytest.mark.skipif(MANO_RIGHT_PKL is None or not KEYPOINTS_NPZ.is_file() or not HANDS_NPZ.is_file(),
                    reason="whisking keypoints / hand meshes (third_party or $FASTDO_BASELINE_ROOT) or MANO pkl missing")
def test_ik_site_target_reproduces_the_shipped_whisking_wrist_target() -> None:
    """On whisking: the MANO joint model reproduces HaWoR's wrist and MCP joints within 1e-6 m (measured 7.1e-8), and
    the IK-site target of the keypoints file's hand reproduces `qpos_wrist_right` within 1e-4 rad (measured 4.4e-5, a
    constant bias from the baseline's float32 joints and spike cleaning). The MCP target is 0.017 rad off, and MANO's
    global orient without the offset 2.1 rad."""
    assert MANO_RIGHT_PKL is not None
    model = mano_rest_joint_model(MANO_RIGHT_PKL)
    with np.load(HANDS_NPZ) as hands:
        rot, betas, trans, hawor = (hands[f"right_{k}"].astype(np.float64) for k in ("rot", "betas", "trans", "joints"))
    posed = posed_mcp_joints(model, rot, betas, rest_joints(model, betas)[:, 0] + trans)
    used = [0, *MCP_OPENPOSE]
    assert np.abs(posed[:, used] - hawor[:, used]).max() <= 1e-6

    with np.load(KEYPOINTS_NPZ) as keypoints:
        shipped = np.asarray(keypoints["qpos_wrist_right"], dtype=np.float64)
        rotvec = np.asarray(keypoints["mano_global_orient_right"], dtype=np.float64)
        kp_betas = np.asarray(keypoints["mano_betas_right"], dtype=np.float64)
    joints = torch.from_numpy(posed_mcp_joints(model, rotvec, kp_betas, shipped[:, :3]))
    shipped_rot = Rotation.from_quat(shipped[:, [4, 5, 6, 3]])

    def angle_to_shipped(poses: torch.Tensor) -> np.ndarray:
        return np.asarray((Rotation.from_matrix(poses[:, :3, :3].numpy()).inv() * shipped_rot).magnitude())

    target = wrist_target_poses("ik_site", joints, torch.from_numpy(rotvec))
    assert angle_to_shipped(target).max() <= 1e-4
    assert np.abs(target[:, :3, 3].numpy() - shipped[:, :3]).max() <= 1e-12
    assert angle_to_shipped(wrist_target_poses("mcp", joints, torch.from_numpy(rotvec))).max() > 1e-2
    assert np.asarray((Rotation.from_rotvec(rotvec).inv() * shipped_rot).magnitude()).min() > 1.0
