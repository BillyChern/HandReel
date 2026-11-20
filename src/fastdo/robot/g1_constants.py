"""Constants of the Unitree G1 29-DoF + Dex3-1 robot path, each with the official file it comes from.

Robot: G1 29-DoF with the OLD 4010 wrist motors (`mode_machine` 5, or 6 for the lock-waist
variant) and Dex3-1 hands. Upper body only, stationary base, RIGHT hand only. The numbers below come from official
Unitree sources at the commits pinned here, except where a comment says otherwise: the hand command limits are
intersections across those files, and the left-arm park pose is a choice of the authors.

- `unitree_ros` @ 5994d4fa: `robots/g1_description/g1_29dof_with_hand_rev_1_0.{xml,urdf}` (model, limits).
- `xr_teleoperate` @ 817fb00c: `teleop/robot_control/robot_arm.py` (arm gains, 250 Hz, `rt/arm_sdk`, blend weight in
  `motor_cmd[29].q`), `robot_hand_unitree.py` (`rt/dex3/right/cmd`, kp 1.5 / kd 0.2, 100 Hz, right package enum order),
  `assets/unitree_hand/unitree_dex3_right.urdf` (retargeting frames and its own finger limits) and
  `assets/unitree_hand/unitree_dex3.yml` (DexPilot config).
- `unitree_sdk2_python` @ 814556d1: `example/g1/high_level/g1_arm7_sdk_dds_example.py` (DDS joint indices, weight
  ramp, waist included in the `arm_sdk` joint set).

COMMAND limits are the INTERSECTION of the limits of every official file that states one (scout section 4 lists the
conflicts); the model keeps its own MJCF ranges. `tests/robot/test_g1_constants.py` re-reads the files and checks it.
"""

from __future__ import annotations

__all__ = [
    "ARM_DDS_INDEX",
    "ARM_KD",
    "ARM_KP",
    "ARM_SDK_TOPIC",
    "ARM_SDK_WEIGHT_INDEX",
    "ARM_URDF_VELOCITY",
    "DEX3_TIP_OFFSETS",
    "G1_MODE_MACHINES",
    "HAND_CMD_TOPIC",
    "HAND_KD",
    "HAND_KP",
    "HAND_STATE_TOPIC",
    "HAND_URDF_VELOCITY",
    "LEFT_ARM_JOINTS",
    "LEFT_ARM_PARK",
    "LEFT_HAND_JOINTS",
    "LEG_JOINTS",
    "LOWSTATE_TOPIC",
    "PALM_IN_DEX3_BASE_RPY",
    "PALM_IN_WRIST_YAW",
    "RIGHT_ARM_COMMAND_LIMITS",
    "RIGHT_ARM_JOINTS",
    "RIGHT_HAND_COMMAND_LIMITS",
    "RIGHT_HAND_JOINTS",
    "WAIST_DDS_INDEX",
    "WAIST_JOINTS",
    "WAIST_KD",
    "WAIST_KP",
]

# --- joints (MJCF names) -------------------------------------------------------------------------------------------
_ARM = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
RIGHT_ARM_JOINTS: tuple[str, ...] = tuple(f"right_{j}_joint" for j in _ARM)
LEFT_ARM_JOINTS: tuple[str, ...] = tuple(f"left_{j}_joint" for j in _ARM)
WAIST_JOINTS: tuple[str, ...] = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
LEG_JOINTS: tuple[str, ...] = tuple(f"{s}_{j}_joint" for s in ("left", "right")
                                    for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"))
# Right Dex3-1 in the order of the enum `Dex3_1_Right_JointIndex` (xr_teleoperate `robot_hand_unitree.py`, lines
# 222-229: thumb, index, middle), which indexes `HandCmd_.motor_cmd` there. This file follows that enum and nothing
# else. The same checkout's retargeting helper lists the right hand as thumb, middle, index (`hand_retargeting.py`,
# lines 55-56), so the upstream sources do not agree with each other. Which order the hardware uses is NOT
# established by this file: check it on the robot before any hand command is sent.
RIGHT_HAND_JOINTS: tuple[str, ...] = tuple(f"right_hand_{j}_joint" for j in
                                           ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0",
                                            "middle_1"))
LEFT_HAND_JOINTS: tuple[str, ...] = tuple(f"left_hand_{j}_joint" for j in
                                          ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0",
                                           "middle_1"))

# --- DDS (unitree_hg) ------------------------------------------------------------------------------------------------
G1_MODE_MACHINES: tuple[int, ...] = (5, 6)       # with_hand_rev_1_0 (5) and its lock-waist sibling (6): 4010 wrists
ARM_SDK_TOPIC = "rt/arm_sdk"
LOWSTATE_TOPIC = "rt/lowstate"
HAND_CMD_TOPIC = "rt/dex3/right/cmd"
HAND_STATE_TOPIC = "rt/dex3/right/state"         # the SDK C++ example uses rt/lf/dex3/right/state (UNVERIFIED which)
ARM_SDK_WEIGHT_INDEX = 29                        # motor_cmd[29].q = arm_sdk blend weight in [0, 1]
ARM_DDS_INDEX: dict[str, int] = {**{j: 15 + i for i, j in enumerate(LEFT_ARM_JOINTS)},
                                 **{j: 22 + i for i, j in enumerate(RIGHT_ARM_JOINTS)}}
WAIST_DDS_INDEX: dict[str, int] = {j: 12 + i for i, j in enumerate(WAIST_JOINTS)}

# --- gains (xr_teleoperate robot_arm.py / robot_hand_unitree.py) ----------------------------------------------------
# Shoulder and elbow are its "weak" motors (kp_low 80, kd_low 3), the three wrist motors use kp_wrist 40 / kd_wrist
# 1.5, every other body joint it holds (waist) uses kp_high 300 / kd_high 3.
ARM_KP: tuple[float, ...] = (80.0, 80.0, 80.0, 80.0, 40.0, 40.0, 40.0)
ARM_KD: tuple[float, ...] = (3.0, 3.0, 3.0, 3.0, 1.5, 1.5, 1.5)
WAIST_KP = 300.0
WAIST_KD = 3.0
HAND_KP = 1.5
HAND_KD = 0.2

# --- limits ---------------------------------------------------------------------------------------------------------
# Right arm: ros MJCF ranges == ros URDF ranges (rad). URDF velocity limits (rad/s).
RIGHT_ARM_COMMAND_LIMITS: tuple[tuple[float, float], ...] = (
    (-3.0892, 2.6704), (-2.2515, 1.5882), (-2.618, 2.618), (-1.0472, 2.0944),
    (-1.97222, 1.97222), (-1.61443, 1.61443), (-1.61443, 1.61443))
ARM_URDF_VELOCITY: tuple[float, ...] = (37.0, 37.0, 37.0, 37.0, 37.0, 22.0, 22.0)
# Right Dex3-1, hardware order. thumb_1 = MJCF [-1.0472, 0.724312] ∩ ros URDF [-1.04719755, 0.61086523] ∩ xr Dex3 URDF
# [-0.920, 0.72431163]; thumb_0 velocity = min(ros URDF 3.14, xr URDF 6.857). Every other entry agrees across files.
RIGHT_HAND_COMMAND_LIMITS: tuple[tuple[float, float], ...] = (
    (-1.0472, 1.0472), (-0.920, 0.61086523), (-1.74533, 0.0), (0.0, 1.5708), (0.0, 1.74533), (0.0, 1.5708),
    (0.0, 1.74533))
HAND_URDF_VELOCITY: tuple[float, ...] = (3.14, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0)

# --- left arm park pose ------------------------------------------------------------------------------------------
# Shoulder roll +0.4 rad abducts the left arm away from the hip (the MJCF left range is [-1.5882, 2.2515]); elbow
# 1.57 rad hangs the forearm straight down (elbow 0 holds it horizontal, forward). The left hand then stays behind
# x = 0.04 m and below z = 0.6 m, clear of a table whose near edge is >= 0.18 m in front of the pelvis at any height.
# `tests/robot/test_g1_model.py` checks the pose is contact-free and inside the limits. A choice, not an official pose.
LEFT_ARM_PARK: tuple[float, ...] = (0.0, 0.4, 0.0, 1.57, 0.0, 0.0, 0.0)

# --- Dex3-1 retargeting frames (xr_teleoperate unitree_dex3_right.urdf) ---------------------------------------------
# In the G1 MJCF the palm is merged into `right_wrist_yaw_link`: the palm mesh geom sits at this offset with identity
# orientation, and the finger bodies sit at palm + the Dex3 URDF joint origins (thumb_0 0.067 = 0.0415 + 0.0255).
PALM_IN_WRIST_YAW: tuple[float, float, float] = (0.0415, -0.003, 0.0)
# The Dex3 URDF fixed joint `base` places right_hand_palm_link in base_link with rpy (3.14, 0, -1.57) at the origin.
# The literal 3.14/-1.57 are kept (not pi): the retargeting frame is whatever Unitree's file defines.
PALM_IN_DEX3_BASE_RPY: tuple[float, float, float] = (3.14, 0.0, -1.57)
# Tip frames: 5 cm along each distal link (thumb_tip_joint, index_joint, middle_joint).
DEX3_TIP_OFFSETS: dict[str, tuple[str, tuple[float, float, float]]] = {
    "thumb": ("right_hand_thumb_2_link", (0.0, 0.05, 0.0)),
    "index": ("right_hand_index_1_link", (0.05, 0.0, 0.0)),
    "middle": ("right_hand_middle_1_link", (0.05, 0.0, 0.0)),
}
