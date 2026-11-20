"""Tests for `fastdo.robot.g1_constants`: every constant is re-read from the official file it cites (skipped when the
third-party clones are absent), and the run script's embedded copies agree with the module."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import pytest

from fastdo.robot import g1_constants as C
from fastdo.robot import g1_sdk_replay as S
from fastdo.robot.g1_model import UNITREE_ROS_XML

_THIRD = Path("/opt/fastdo/handreel/third_party")


def _need(path: Path) -> Path:
    if not path.exists():
        pytest.skip(f"{path} not available")
    return path


def _urdf_limits(path: Path) -> dict[str, tuple[float, float, float]]:
    out = {}
    for joint in ET.parse(path).getroot().iter("joint"):
        lim = joint.find("limit")
        if lim is not None and "lower" in lim.attrib:
            out[joint.attrib["name"]] = (float(lim.attrib["lower"]), float(lim.attrib["upper"]),
                                         float(lim.attrib["velocity"]))
    return out


def test_hand_command_limits_are_the_intersection_of_the_official_files(unitree_root: Path) -> None:
    mjcf = mujoco.MjModel.from_xml_path(str(unitree_root / UNITREE_ROS_XML))
    ros = _urdf_limits(unitree_root / "robots/g1_description/g1_29dof_with_hand_rev_1_0.urdf")
    xr = _urdf_limits(_need(_THIRD / "xr_teleoperate/assets/unitree_hand/unitree_dex3_right.urdf"))
    for k, name in enumerate(C.RIGHT_HAND_JOINTS):
        jr = mjcf.jnt_range[mujoco.mj_name2id(mjcf, mujoco.mjtObj.mjOBJ_JOINT, name)]
        lo = max(jr[0], ros[name][0], xr[name][0])
        hi = min(jr[1], ros[name][1], xr[name][1])
        assert C.RIGHT_HAND_COMMAND_LIMITS[k] == pytest.approx((lo, hi), abs=1e-4), name
        assert C.HAND_URDF_VELOCITY[k] == pytest.approx(min(ros[name][2], xr[name][2])), name
    for k, name in enumerate(C.RIGHT_ARM_JOINTS):
        jr = mjcf.jnt_range[mujoco.mj_name2id(mjcf, mujoco.mjtObj.mjOBJ_JOINT, name)]
        assert C.RIGHT_ARM_COMMAND_LIMITS[k] == pytest.approx(tuple(jr), abs=1e-4), name
        assert C.ARM_URDF_VELOCITY[k] == pytest.approx(ros[name][2]), name


def test_gains_order_and_frames_match_xr_teleoperate() -> None:
    arm = _need(_THIRD / "xr_teleoperate/teleop/robot_control/robot_arm.py").read_text()
    hand = (_THIRD / "xr_teleoperate/teleop/robot_control/robot_hand_unitree.py").read_text()
    for name, value in (("kp_low", 80.0), ("kd_low", 3.0), ("kp_wrist", 40.0), ("kd_wrist", 1.5), ("kp_high", 300.0)):
        assert re.search(rf"self\.{name} = {value}", arm), name
    assert re.search(r"kp = 1\.5\s+kd = 0\.2", hand)
    right = re.search(r"class Dex3_1_Right_JointIndex\(IntEnum\):(.*?)\n\n", hand, re.DOTALL)
    assert right is not None
    order = re.findall(r"kRightHand(\w+?)(\d) = (\d)", right.group(1))
    names = [f"right_hand_{f.lower()}_{i}_joint" for f, i, _ in sorted(order, key=lambda x: int(x[2]))]
    assert tuple(names) == C.RIGHT_HAND_JOINTS
    urdf = (_THIRD / "xr_teleoperate/assets/unitree_hand/unitree_dex3_right.urdf").read_text()
    assert '<origin rpy="3.14 0 -1.57" xyz="0 0 0" />' in urdf
    assert C.PALM_IN_DEX3_BASE_RPY == (3.14, 0.0, -1.57)


def test_dds_indices_match_the_sdk_example() -> None:
    example = _need(_THIRD / "unitree_sdk2_python/example/g1/high_level/g1_arm7_sdk_dds_example.py").read_text()
    for joint, idx in C.ARM_DDS_INDEX.items():
        side, rest = joint.split("_", 1)
        camel = side.capitalize() + "".join(p.capitalize() for p in rest.replace("_joint", "").split("_"))
        assert re.search(rf"{camel} = {idx}\b", example), (joint, camel, idx)
    assert re.search(r"kNotUsedJoint = 29", example)
    assert C.ARM_SDK_WEIGHT_INDEX == 29


def test_run_script_embeds_the_module_constants() -> None:
    """The run script is copied verbatim into packages and cannot import fastdo: its copies must agree."""
    assert S.RIGHT_ARM_DDS == tuple(C.ARM_DDS_INDEX[j] for j in C.RIGHT_ARM_JOINTS)
    assert S.LEFT_ARM_DDS == tuple(C.ARM_DDS_INDEX[j] for j in C.LEFT_ARM_JOINTS)
    assert S.WAIST_DDS == tuple(C.WAIST_DDS_INDEX[j] for j in C.WAIST_JOINTS)
    assert S.ARM_KP == C.ARM_KP and S.ARM_KD == C.ARM_KD
    assert (S.WAIST_KP, S.WAIST_KD, S.HAND_KP, S.HAND_KD) == (C.WAIST_KP, C.WAIST_KD, C.HAND_KP, C.HAND_KD)
    assert (S.ARM_SDK_TOPIC, S.HAND_CMD_TOPIC, S.LOWSTATE_TOPIC, S.HAND_STATE_TOPIC) == (
        C.ARM_SDK_TOPIC, C.HAND_CMD_TOPIC, C.LOWSTATE_TOPIC, C.HAND_STATE_TOPIC)
    assert S.MODE_MACHINES == C.G1_MODE_MACHINES and S.WEIGHT_INDEX == C.ARM_SDK_WEIGHT_INDEX
    assert np.isfinite(C.LEFT_ARM_PARK).all()
