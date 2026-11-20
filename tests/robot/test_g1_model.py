"""Tests for `fastdo.robot.g1_model`: pin verification (with mutations), the stationary-base edit, actuators, the Dex3
retargeting sites against an independent URDF forward kinematics, the left-arm park pose and a PD hold."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from fastdo.robot import g1_constants as C
from fastdo.robot import g1_model
from fastdo.robot.g1_model import (
    UNITREE_ROS_XML,
    PinMismatchError,
    build_g1_upper_spec,
    compile_g1_upper,
    verify_unitree_ros,
)
from fastdo.robot.trajectory import gravity_torque, left_park_torque


def test_pin_verifies_on_the_pinned_clone(unitree_root: Path) -> None:
    verify_unitree_ros(unitree_root)


def test_pin_mismatch_fails_loudly(unitree_root: Path, tmp_path: Path) -> None:
    """Mutations: a different expected SHA, a different expected digest, and a clone at another commit all raise."""
    with pytest.raises(PinMismatchError, match="HEAD is"):
        verify_unitree_ros(unitree_root, expected_sha="0" * 40)
    with pytest.raises(PinMismatchError, match="sha256"):
        verify_unitree_ros(unitree_root, expected_xml_sha256="0" * 64)
    fake = tmp_path / "unitree_ros"
    (fake / Path(UNITREE_ROS_XML).parent).mkdir(parents=True)
    shutil.copyfile(unitree_root / UNITREE_ROS_XML, fake / UNITREE_ROS_XML)
    git = ["git", "-C", str(fake), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run([*git, "commit", "-qm", "copy"], check=True)
    with pytest.raises(PinMismatchError, match="HEAD is"):
        verify_unitree_ros(fake)
    with pytest.raises(PinMismatchError):
        verify_unitree_ros(tmp_path / "not_a_repo")


def test_stationary_upper_body_structure(g1, unitree_root: Path) -> None:
    m = g1.model
    names = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt)}
    assert names == set(C.RIGHT_ARM_JOINTS + C.LEFT_ARM_JOINTS + C.RIGHT_HAND_JOINTS + C.LEFT_HAND_JOINTS)
    assert (m.nq, m.nv, m.nu) == (28, 28, 42)
    assert m.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    assert m.opt.timestep == pytest.approx(0.002)
    pelvis = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    assert m.body_weldid[pelvis] == 0 and np.allclose(m.body_pos[pelvis], [0.0, 0.0, 0.793])
    official = mujoco.MjModel.from_xml_path(str(unitree_root / UNITREE_ROS_XML))
    for name in names:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        oid = mujoco.mj_name2id(official, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert np.array_equal(m.jnt_range[jid], official.jnt_range[oid]), name
        assert np.array_equal(m.jnt_actfrcrange[jid], official.jnt_actfrcrange[oid]), name


def test_actuators_carry_the_official_gains(g1) -> None:
    m = g1.model
    kp = m.actuator_gainprm[g1.act_right_arm, 0]
    kv = -m.actuator_biasprm[g1.act_right_arm, 2]
    assert np.allclose(kp, C.ARM_KP) and np.allclose(kv, C.ARM_KD)
    assert np.allclose(m.actuator_gainprm[g1.act_right_hand, 0], C.HAND_KP)
    assert np.allclose(-m.actuator_biasprm[g1.act_right_hand, 2], C.HAND_KD)
    assert np.allclose(m.actuator_gainprm[g1.ff_right_arm, 0], 1.0)
    hand_joints = m.actuator_trnid[g1.act_right_hand, 0]
    assert [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in hand_joints] == list(C.RIGHT_HAND_JOINTS)


def _urdf_tips_in_base() -> np.ndarray:
    """Independent FK at q = 0 from the Dex3 URDF joint origins (palm frame), mapped into base_link."""
    thumb = np.array([0.0255, 0, 0]) + [-0.0025, 0.0193, 0] + np.array([0, 0.0458, 0]) + [0, 0.05, 0]
    index = np.array([0.0777, -0.0016, 0.0285]) + [0.0458, 0, 0] + np.array([0.05, 0, 0])
    middle = np.array([0.0777, -0.0016, -0.0285]) + [0.0458, 0, 0] + np.array([0.05, 0, 0])
    r_base_palm = Rotation.from_euler("xyz", [3.14, 0.0, -1.57]).as_matrix()
    return np.stack([thumb, index, middle]) @ r_base_palm.T


def _model_tips_in_base(g) -> np.ndarray:
    d = mujoco.MjData(g.model)
    d.qpos[:] = g.default_qpos()
    mujoco.mj_kinematics(g.model, d)
    rot = d.site_xmat[g.site_base].reshape(3, 3)
    return (d.site_xpos[g.site_tips] - d.site_xpos[g.site_base]) @ rot


def test_dex3_sites_match_the_urdf_frames(g1) -> None:
    tips = _model_tips_in_base(g1)
    assert np.abs(tips - _urdf_tips_in_base()).max() < 1e-6
    # Unitree hand convention (right): fingers point -y (y = middle -> wrist), index is at -z of the middle finger.
    assert tips[2, 1] < -0.1 and tips[1, 2] < tips[2, 2]


def test_dex3_site_mutation_is_caught(unitree_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation: reading the URDF rpy as intrinsic XYZ instead of URDF's extrinsic xyz moves the tips off the URDF
    answer. (Transposing it would not: rpy (3.14, 0, -1.57) is a half turn, its own inverse.)"""

    def wrong() -> tuple[np.ndarray, np.ndarray]:
        return np.asarray(C.PALM_IN_WRIST_YAW), Rotation.from_euler("XYZ", C.PALM_IN_DEX3_BASE_RPY).as_matrix().T

    monkeypatch.setattr(g1_model, "dex3_base_in_wrist_yaw", wrong)
    g = compile_g1_upper(build_g1_upper_spec(unitree_root), unitree_root)
    assert np.abs(_model_tips_in_base(g) - _urdf_tips_in_base()).max() > 0.01


def test_left_park_pose_is_contact_free_and_inside_limits(g1) -> None:
    m = g1.model
    d = mujoco.MjData(m)
    d.qpos[:] = g1.default_qpos()
    mujoco.mj_forward(m, d)
    assert d.ncon == 0
    lo = m.jnt_range[[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in C.LEFT_ARM_JOINTS]]
    assert ((np.asarray(C.LEFT_ARM_PARK) > lo[:, 0]) & (np.asarray(C.LEFT_ARM_PARK) < lo[:, 1])).all()


def test_pd_hold_with_gravity_feedforward(g1) -> None:
    """Known answer: with tau_ff = g(q) a static pose is held exactly; the fingers hold 0.4 rad of flexion within
    0.02 rad at the official kp 1.5 (the scout measured 0.009 rad mean with implicitfast)."""
    m = g1.model
    d = mujoco.MjData(m)
    q_arm = np.array([-0.5, -0.3, 0.2, 1.0, 0.1, 0.2, -0.1])
    q_hand = np.array([0.0, 0.0, -0.4, 0.4, 0.4, 0.4, 0.4])
    d.qpos[:] = g1.default_qpos()
    d.qpos[g1.right_arm_qpos] = q_arm
    d.qpos[g1.right_hand_qpos] = q_hand
    ctrl = g1.robot_ctrl(q_arm, q_hand, gravity_torque(g1, q_arm[None], q_hand[None])[0], None, left_park_torque(g1))
    for _ in range(500):
        d.ctrl[:] = ctrl
        mujoco.mj_step(m, d)
    assert np.abs(d.qpos[g1.right_arm_qpos] - q_arm).max() < 1e-3
    assert np.abs(d.qpos[g1.left_arm_qpos] - C.LEFT_ARM_PARK).max() < 1e-3
    assert np.abs(d.qpos[g1.right_hand_qpos] - q_hand).max() < 0.02
