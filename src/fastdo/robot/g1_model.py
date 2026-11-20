"""Official-model loader for the G1 29-DoF + Dex3-1 upper-body robot path.

Source of truth: `unitree_ros/robots/g1_description/g1_29dof_with_hand_rev_1_0.xml` at the pinned commit
(`UNITREE_ROS_SHA`). `verify_unitree_ros` checks, and raises `PinMismatchError` on any failure, that
1. the clone's `git rev-parse HEAD` is the pinned SHA,
2. `git status --porcelain -- robots/g1_description` is empty (no edited or added mesh or model file), and
3. the MJCF's sha256 is the pinned digest.
The clone is read-only third-party code; this module never writes to it. It is looked up at `$FASTDO_UNITREE_ROS`,
then `<repo>/third_party/unitree_ros`, then the primary checkout's `third_party/unitree_ros`
(`fastdo.hostpaths.primary_root()`).

Edits applied to the loaded `MjSpec` (the file itself is untouched):
- **Stationary base.** The pelvis free joint is deleted, so the pelvis is welded to the world at the file's own
  standing height (0.793 m). The 12 leg joints and the 3 waist joints are deleted, so legs and torso are rigid at joint
  angle 0. This is the configuration of Unitree's own fixed-base Dex3 sim task (`unitree_sim_isaaclab`
  `robot_configs.py::g1_29dof_dex3_base_fix`: legs and waist locked at 0). On the real robot (DDS mode A) Unitree's
  locomotion controller owns legs and waist; torso sway under it is NOT modelled.
- **Actuators.** The 43 official torque motors are replaced by, for each kept joint, a MuJoCo `position` actuator with
  the official command gains (`kp`, `kv` = kd; `g1_constants`), plus a `motor` per arm joint for the feed-forward
  torque `tau_ff` that xr_teleoperate also sends. The motor law `tau = tau_ff + kp(q* - q) - kd dq` is the real
  arm_sdk command with dq* = 0, as xr_teleoperate sends it. The joints keep the file's `actuatorfrcrange` torque limits.
- **Integrator `implicitfast`** at dt = 2 ms: the scout measured the official Dex3 gains as explicit PD at 2 ms to be
  unstable (0.60 rad mean error) and the implicit position actuator to track at 0.009 rad (scout section 3.3).
- **Active joints.** Right arm (7) and right Dex3-1 (7) follow the trajectory. The left arm holds `LEFT_ARM_PARK` and
  the left hand holds 0 (the left-hand motor order is ambiguous in the official sources, so it is never commanded on
  hardware either).
- **Sites** (no tip sites exist in the official MJCF): `right_dex3_base` (the Dex3 URDF `base_link` retargeting frame,
  on the merged palm) and `right_{thumb,index,middle}_tip` (5 cm along each distal link, as the Dex3 URDF).
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from fastdo import hostpaths
from fastdo.robot import g1_constants as C

__all__ = [
    "UNITREE_ROS_SHA",
    "UNITREE_ROS_XML",
    "UNITREE_ROS_XML_SHA256",
    "G1Upper",
    "PinMismatchError",
    "build_g1_upper_spec",
    "compile_g1_upper",
    "dex3_base_in_wrist_yaw",
    "load_g1_upper",
    "resolve_unitree_ros",
    "verify_unitree_ros",
]

UNITREE_ROS_SHA = "5994d4faef0a9cadd3287f8de0199a67eeb2a259"
UNITREE_ROS_XML = "robots/g1_description/g1_29dof_with_hand_rev_1_0.xml"
UNITREE_ROS_XML_SHA256 = "77c98e7d34e428c6cdbd1e0d665e989cbc43bf17abb8a7c048494b4bc51ead88"
_MAIN_CHECKOUT = hostpaths.primary_root() / "third_party" / "unitree_ros"   # $FASTDO_PRIMARY_ROOT/third_party/...
DEFAULT_TIMESTEP = 0.002


class PinMismatchError(RuntimeError):
    """The unitree_ros clone is not the pinned, unmodified official source."""


def resolve_unitree_ros(root: str | Path | None = None) -> Path:
    """The unitree_ros clone: `root`, else `$FASTDO_UNITREE_ROS`, else the repo's or the main checkout's third_party.

    Raises:
        FileNotFoundError: If no candidate holds `UNITREE_ROS_XML`.
    """
    if root is not None:
        candidates = [Path(root)]
    else:
        env = os.environ.get("FASTDO_UNITREE_ROS")
        repo = Path(__file__).resolve().parents[3] / "third_party" / "unitree_ros"
        candidates = [Path(env)] if env else [repo, _MAIN_CHECKOUT]
    for cand in candidates:
        if (cand / UNITREE_ROS_XML).is_file():
            return cand
    raise FileNotFoundError(f"no unitree_ros clone with {UNITREE_ROS_XML} at {[str(c) for c in candidates]}: set "
                            f"FASTDO_UNITREE_ROS to an official unitree_ros clone at commit {UNITREE_ROS_SHA}")


def _git(root: Path, *args: str) -> str:
    try:
        out = subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise PinMismatchError(f"cannot verify {root} with git {' '.join(args)}: {exc}") from exc
    return out.stdout


def verify_unitree_ros(root: Path, expected_sha: str = UNITREE_ROS_SHA,
                       expected_xml_sha256: str = UNITREE_ROS_XML_SHA256) -> None:
    """Fail loudly unless `root` is the pinned, unmodified clone (module docstring, checks 1-3).

    Raises:
        PinMismatchError: On a different HEAD, a dirty `robots/g1_description`, or a different MJCF digest.
    """
    head = _git(root, "rev-parse", "HEAD").strip()
    if head != expected_sha:
        raise PinMismatchError(f"{root}: HEAD is {head}, the pinned unitree_ros commit is {expected_sha}")
    dirty = _git(root, "status", "--porcelain", "--", "robots/g1_description").strip()
    if dirty:
        raise PinMismatchError(f"{root}: robots/g1_description differs from the pinned commit:\n{dirty[:2000]}")
    digest = hashlib.sha256((root / UNITREE_ROS_XML).read_bytes()).hexdigest()
    if digest != expected_xml_sha256:
        raise PinMismatchError(f"{root / UNITREE_ROS_XML}: sha256 {digest} != pinned {expected_xml_sha256}")


def dex3_base_in_wrist_yaw() -> tuple[np.ndarray, np.ndarray]:
    """`(pos (3,), R (3,3))` of the Dex3 URDF `base_link` in the MJCF's `right_wrist_yaw_link` frame.

    The URDF places the palm in base_link by rpy `PALM_IN_DEX3_BASE_RPY` (URDF rpy = Rz(y) Ry(p) Rx(r), extrinsic xyz)
    at the origin, and the MJCF palm sits at `PALM_IN_WRIST_YAW` with identity orientation. So base_link in the palm is
    the inverse rotation at the palm origin.
    """
    r_base_palm = Rotation.from_euler("xyz", C.PALM_IN_DEX3_BASE_RPY).as_matrix()
    return np.asarray(C.PALM_IN_WRIST_YAW, dtype=np.float64), r_base_palm.T


def _quat_wxyz(rot: np.ndarray) -> list[float]:
    x, y, z, w = Rotation.from_matrix(rot).as_quat()
    return [float(w), float(x), float(y), float(z)]


def build_g1_upper_spec(root: Path, *, timestep: float = DEFAULT_TIMESTEP, verify: bool = True) -> mujoco.MjSpec:
    """The edited `MjSpec` of the official MJCF (module docstring). `verify` runs `verify_unitree_ros` first."""
    if verify:
        verify_unitree_ros(root)
    spec = mujoco.MjSpec.from_file(str(root / UNITREE_ROS_XML))
    for act in list(spec.actuators):
        spec.delete(act)
    for name in ("floating_base_joint", *C.LEG_JOINTS, *C.WAIST_JOINTS):
        joint = spec.joint(name)
        if joint is None:
            raise ValueError(f"official MJCF has no joint {name!r}")
        spec.delete(joint)
    spec.option.timestep = timestep
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST

    wrist = spec.body("right_wrist_yaw_link")
    pos, rot = dex3_base_in_wrist_yaw()
    wrist.add_site(name="right_dex3_base", pos=pos.tolist(), quat=_quat_wxyz(rot), size=[0.008, 0, 0])
    for finger, (body, offset) in C.DEX3_TIP_OFFSETS.items():
        spec.body(body).add_site(name=f"right_{finger}_tip", pos=list(offset), size=[0.005, 0, 0])

    def _position(joint: str, kp: float, kd: float) -> None:
        act = spec.add_actuator(name=f"pos_{joint}", target=joint, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        act.set_to_position(kp=kp, kv=kd)

    def _motor(joint: str) -> None:
        act = spec.add_actuator(name=f"ff_{joint}", target=joint, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        act.set_to_motor()

    for arm in (C.RIGHT_ARM_JOINTS, C.LEFT_ARM_JOINTS):
        for joint, kp, kd in zip(arm, C.ARM_KP, C.ARM_KD, strict=True):
            _position(joint, kp, kd)
    for hand in (C.RIGHT_HAND_JOINTS, C.LEFT_HAND_JOINTS):
        for joint in hand:
            _position(joint, C.HAND_KP, C.HAND_KD)
    for joint in (*C.RIGHT_ARM_JOINTS, *C.LEFT_ARM_JOINTS):
        _motor(joint)
    return spec


@dataclass(frozen=True)
class G1Upper:
    """The compiled upper-body model plus the index maps every robot-path module uses (all `(7,)` int arrays)."""

    model: mujoco.MjModel
    root: Path
    right_arm_qpos: np.ndarray
    right_arm_dof: np.ndarray
    right_hand_qpos: np.ndarray       # hardware order (`C.RIGHT_HAND_JOINTS`)
    right_hand_dof: np.ndarray
    left_arm_qpos: np.ndarray
    left_hand_qpos: np.ndarray
    act_right_arm: np.ndarray         # position actuators
    act_right_hand: np.ndarray
    act_left_arm: np.ndarray
    act_left_hand: np.ndarray
    ff_right_arm: np.ndarray          # feed-forward motors
    ff_left_arm: np.ndarray
    site_base: int
    site_tips: np.ndarray             # (3,) thumb, index, middle

    @property
    def right_arm_range(self) -> np.ndarray:
        """`(7,2)` MJCF joint ranges of the right arm."""
        return self.model.jnt_range[self._joint_ids(C.RIGHT_ARM_JOINTS)].copy()

    @property
    def right_hand_range(self) -> np.ndarray:
        """`(7,2)` MJCF joint ranges of the right hand, package enum order."""
        return self.model.jnt_range[self._joint_ids(C.RIGHT_HAND_JOINTS)].copy()

    def _joint_ids(self, names: tuple[str, ...]) -> np.ndarray:
        return np.array([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names])

    def default_qpos(self) -> np.ndarray:
        """`(nq,)` qpos with the left arm parked, every other joint at its `qpos0` (0 for the robot's hinges; a free
        object body at its spec pose)."""
        q = self.model.qpos0.copy()
        q[self.left_arm_qpos] = C.LEFT_ARM_PARK
        return q

    def robot_ctrl(self, q_arm: np.ndarray, q_hand: np.ndarray, tau_ff: np.ndarray | None = None,
                   ctrl: np.ndarray | None = None, tau_left: np.ndarray | None = None) -> np.ndarray:
        """`(nu,)` controls: right arm/hand position targets, left arm park, left hand 0, and the feed-forward
        torques of the right arm (`tau_ff`) and the parked left arm (`tau_left`)."""
        out = np.zeros(self.model.nu) if ctrl is None else ctrl
        out[self.act_right_arm] = q_arm
        out[self.act_right_hand] = q_hand
        out[self.act_left_arm] = C.LEFT_ARM_PARK
        out[self.act_left_hand] = 0.0
        out[self.ff_right_arm] = 0.0 if tau_ff is None else tau_ff
        out[self.ff_left_arm] = 0.0 if tau_left is None else tau_left
        return out


def compile_g1_upper(spec: mujoco.MjSpec, root: Path) -> G1Upper:
    """Compile `spec` (from `build_g1_upper_spec`, possibly with more bodies added) and derive the index maps."""
    model = spec.compile()

    def ids(kind: mujoco.mjtObj, names: tuple[str, ...] | list[str]) -> np.ndarray:
        out = np.array([mujoco.mj_name2id(model, kind, n) for n in names], dtype=np.int64)
        if (out < 0).any():
            raise ValueError(f"missing {kind.name} {[n for n, i in zip(names, out, strict=True) if i < 0]}")
        return out

    j = mujoco.mjtObj.mjOBJ_JOINT
    a = mujoco.mjtObj.mjOBJ_ACTUATOR
    return G1Upper(
        model=model, root=root,
        right_arm_qpos=model.jnt_qposadr[ids(j, C.RIGHT_ARM_JOINTS)],
        right_arm_dof=model.jnt_dofadr[ids(j, C.RIGHT_ARM_JOINTS)],
        right_hand_qpos=model.jnt_qposadr[ids(j, C.RIGHT_HAND_JOINTS)],
        right_hand_dof=model.jnt_dofadr[ids(j, C.RIGHT_HAND_JOINTS)],
        left_arm_qpos=model.jnt_qposadr[ids(j, C.LEFT_ARM_JOINTS)],
        left_hand_qpos=model.jnt_qposadr[ids(j, C.LEFT_HAND_JOINTS)],
        act_right_arm=ids(a, [f"pos_{n}" for n in C.RIGHT_ARM_JOINTS]),
        act_right_hand=ids(a, [f"pos_{n}" for n in C.RIGHT_HAND_JOINTS]),
        act_left_arm=ids(a, [f"pos_{n}" for n in C.LEFT_ARM_JOINTS]),
        act_left_hand=ids(a, [f"pos_{n}" for n in C.LEFT_HAND_JOINTS]),
        ff_right_arm=ids(a, [f"ff_{n}" for n in C.RIGHT_ARM_JOINTS]),
        ff_left_arm=ids(a, [f"ff_{n}" for n in C.LEFT_ARM_JOINTS]),
        site_base=int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_dex3_base")),
        site_tips=ids(mujoco.mjtObj.mjOBJ_SITE, ["right_thumb_tip", "right_index_tip", "right_middle_tip"]),
    )


@lru_cache(maxsize=4)
def _load_cached(root: str, timestep: float) -> G1Upper:
    path = Path(root)
    return compile_g1_upper(build_g1_upper_spec(path, timestep=timestep, verify=True), path)


def load_g1_upper(root: str | Path | None = None, *, timestep: float = DEFAULT_TIMESTEP) -> G1Upper:
    """The verified, compiled upper-body model (cached per clone and timestep). Errors as `verify_unitree_ros`."""
    return _load_cached(str(resolve_unitree_ros(root)), float(timestep))
