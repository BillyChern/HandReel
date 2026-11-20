"""MuJoCo CPU replay harness for G1 + Dex3-1 robot trajectories.

Scene: the official upper-body model (`fastdo.robot.g1_model`: fixed base, official command gains as implicit position
actuators, `implicitfast`, dt = 2 ms) + a static table box + the object as a FREE body whose collision geometry is our
reconstructed mesh (metres; MuJoCo collides with its convex hull) at the placed start pose.

The replay streams commands as the robot would receive them: arm targets + `tau_ff` held for 4 ms (250 Hz), hand
targets held for 10 ms (100 Hz). It first holds the first command for `settle_s` (the object settles on the table),
then plays the trajectory. The same harness takes any `RobotTrajectory`, so the physics-refined arm can reuse it.

Metrics (`ReplayMetrics`):
- joint tracking error |q - q_cmd| (rad) of the right arm and hand, RMS and max, sampled at each arm tick;
- joint-limit violations: ticks where a right-arm / right-hand joint leaves its MJCF range by more than 0.01 rad;
- self-collision: robot-robot contacts deeper than `self_tol`; count of physics steps and max depth (m);
- table penetration: robot-table contact depth, max (m) and steps deeper than `table_tol`;
- object contact: physics steps with any robot-object contact (0 = the robot never touched the object);
- settle: the object's translation (`settle_drift`, m) and rotation (`settle_rot_deg`) while it settles on the table
  before the robot moves (the execution gate caps both: an object placed where it cannot rest is refused);
- object: displacement from the settled start, the reference displacement (the reconstruction placed in the robot
  frame, between the first and last row the retarget used: `ref_rows`), the final position error to the reference
  end, and whether it fell below the table top;
- success: every criterion of `SuccessCriteria` holds (the object must MOVE: displacement >= `min_move_frac` of the
  reference, with a reference displacement of at least `min_ref_move`).

`simulate_run` runs the package's run script itself (`g1_sdk_replay.Replayer`, the same object `--execute` drives) on
this scene through `RunIO`: its rt/arm_sdk and Dex3 commands become the position targets and feed-forward torques of
both arms and the right hand, its state reads come from the simulation, its clock is the simulation's. So the moves
to the hold pose and home, the playback, operator stops and the release are simulated exactly as the script commands
them, from a given home. The arm_sdk blend weight w mixes the script's targets with the pose the robot held when the
run started (a stand-in for Unitree's own controller, which is not modelled: target = w q_cmd + (1 - w) q_start). The
waist is welded at 0 (the official model has no waist joints here). Per phase (`PhaseStats`): robot-table and
robot-object contact steps, the deepest robot-table, robot-robot and robot-object penetration, the right-arm tracking
max (at full weight) and the object displacement. Contact STEPS, not depth, say whether the arm touched the table: a
hand driven into the table top is stopped by it (an early straight TO_HOLD from the hanging home, 68 mm into the
table kinematically, penetrates 0.8 mm in this simulation before the tracking abort).
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, replace

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from fastdo.robot import g1_constants as C
from fastdo.robot import g1_sdk_replay as S
from fastdo.robot.g1_model import (
    G1Upper,
    build_g1_upper_spec,
    compile_g1_upper,
    resolve_unitree_ros,
)
from fastdo.robot.trajectory import RobotTrajectory, left_park_torque

__all__ = ["ReplayMetrics", "ReplayScene", "RunIO", "SceneConfig", "SuccessCriteria", "build_ik_model",
           "build_replay_scene", "perturbed_table_scene", "replay", "run_options", "simulate_run"]


@dataclass(frozen=True)
class SceneConfig:
    table_z: float = 0.80                  # table top height (m)
    table_x: tuple[float, float] = (0.18, 0.78)
    table_y: tuple[float, float] = (-0.60, 0.40)
    table_thickness: float = 0.04
    object_mass: float = 0.2               # kg, UNVERIFIED placeholder until the lab weighs the look-alike object
    friction: float = 1.0
    settle_s: float = 0.5


@dataclass(frozen=True)
class SuccessCriteria:
    max_arm_rms: float = 0.05              # rad
    max_self_depth: float = 0.005          # m
    max_table_depth: float = 0.01          # m
    min_ref_move: float = 0.02             # m: the reference must itself move the object
    min_move_frac: float = 0.5             # sim displacement >= frac * reference displacement
    max_final_err_frac: float = 0.5        # final error <= max(final_err_floor, frac * reference displacement)
    final_err_floor: float = 0.03          # m


@dataclass(frozen=True)
class ReplayScene:
    g1: G1Upper
    object_qpos: int                       # qpos address of the object free joint
    object_body: int
    table_geom: int
    robot_geoms: np.ndarray                # bool (ngeom,)
    cfg: SceneConfig


@dataclass(frozen=True)
class ReplayMetrics:
    duration_s: float
    arm_rms: float
    arm_max: float
    hand_rms: float
    hand_max: float
    limit_violation_ticks: int
    self_collision_steps: int
    self_collision_max_depth: float
    table_max_depth: float
    table_penetration_steps: int
    object_contact_steps: int
    settle_drift: float
    object_displacement: float
    reference_displacement: float
    object_final_error: float
    object_final_rot_err_deg: float
    object_fell: bool
    success: bool
    failed_criteria: tuple[str, ...]
    table_max_force: float = 0.0           # N: largest summed robot-table normal force of one physics step
    settle_rot_deg: float = float("nan")   # deg: object rotation during the settle (`settle_drift` is its translation)

    def to_dict(self) -> dict:
        out = asdict(self)
        out["failed_criteria"] = list(self.failed_criteria)
        return out


def _quat_wxyz(rot: np.ndarray) -> list[float]:
    x, y, z, w = Rotation.from_matrix(rot).as_quat()
    return [float(w), float(x), float(y), float(z)]


def _rot_deg(rot0: np.ndarray, quat_wxyz: np.ndarray) -> float:
    """Angle (deg) between the rotation `rot0` (3,3) and the MuJoCo quaternion `quat_wxyz` (4,)."""
    r1 = Rotation.from_quat(np.roll(np.asarray(quat_wxyz, dtype=np.float64), -1)).as_matrix()
    return float(np.degrees(np.arccos(np.clip((np.trace(np.asarray(rot0).T @ r1) - 1.0) / 2.0, -1.0, 1.0))))


def _add_table(spec: mujoco.MjSpec, cfg: SceneConfig) -> None:
    (x0, x1), (y0, y1) = cfg.table_x, cfg.table_y
    table = spec.worldbody.add_body(name="table", pos=[(x0 + x1) / 2, (y0 + y1) / 2,
                                                      cfg.table_z - cfg.table_thickness / 2])
    table.add_geom(name="table_top", type=mujoco.mjtGeom.mjGEOM_BOX,
                   size=[(x1 - x0) / 2, (y1 - y0) / 2, cfg.table_thickness / 2], friction=[cfg.friction, 0.005, 1e-4],
                   rgba=[0.6, 0.5, 0.4, 1.0])


def build_ik_model(cfg: SceneConfig | None = None, root: str | None = None) -> G1Upper:
    """The robot + table (no object): the model the collision-aware arm IK runs on."""
    cfg = SceneConfig() if cfg is None else cfg
    path = resolve_unitree_ros(root)
    spec = build_g1_upper_spec(path, verify=True)
    _add_table(spec, cfg)
    return compile_g1_upper(spec, path)


def build_replay_scene(mesh_vertices: np.ndarray, mesh_faces: np.ndarray, scale: float, object_start: np.ndarray,
                       cfg: SceneConfig | None = None, root: str | None = None) -> ReplayScene:
    """The replay scene with the object mesh (`(V,3)` canonical vertices x `scale`) at `object_start` (4,4)."""
    cfg = SceneConfig() if cfg is None else cfg
    path = resolve_unitree_ros(root)
    spec = build_g1_upper_spec(path, verify=True)
    _add_table(spec, cfg)
    spec.add_mesh(name="object_mesh", uservert=(scale * np.asarray(mesh_vertices, dtype=np.float64)).ravel().tolist(),
                  userface=np.asarray(mesh_faces, dtype=np.int64).ravel().tolist())
    obj = spec.worldbody.add_body(name="object", pos=object_start[:3, 3].tolist(),
                                  quat=_quat_wxyz(object_start[:3, :3]))
    obj.add_freejoint(name="object_free")
    obj.add_geom(name="object_geom", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="object_mesh", mass=cfg.object_mass,
                 friction=[cfg.friction, 0.005, 1e-4], rgba=[0.2, 0.6, 0.9, 1.0])
    g1 = compile_g1_upper(spec, path)
    m = g1.model
    table_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
    obj_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "object")
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "object_free")
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    robot = np.ones(m.ngeom, dtype=bool)
    robot[[table_geom, floor]] = False
    robot[m.geom_bodyid == obj_body] = False
    return ReplayScene(g1=g1, object_qpos=int(m.jnt_qposadr[jid]), object_body=obj_body, table_geom=table_geom,
                       robot_geoms=robot, cfg=cfg)


def _table_force(scene: ReplayScene, d: mujoco.MjData) -> float:
    """Summed normal force (N) of the current robot-table contacts (`mj_contactForce`; 0 when none)."""
    rob, tg = scene.robot_geoms, scene.table_geom
    f6 = np.zeros(6)
    total = 0.0
    for i, c in enumerate(d.contact[: d.ncon]):
        g1, g2 = int(c.geom[0]), int(c.geom[1])
        if (rob[g1] and g2 == tg) or (rob[g2] and g1 == tg):
            mujoco.mj_contactForce(scene.g1.model, d, i, f6)
            total += abs(float(f6[0]))
    return total


def perturbed_table_scene(scene: ReplayScene, dz: float, grow: float) -> ReplayScene:
    """`scene` with its table top moved by `dz` (m, + up) and every edge moved outward by `grow` (m), thickness kept:
    a lab table inside the package's placement tolerance (`fastdo.robot.transfer`, "Table tolerance"). The robot
    and the object are unchanged (the caller raises the object's start pose by `dz`)."""
    m = copy.deepcopy(scene.g1.model)
    g, tb = scene.table_geom, m.geom_bodyid[scene.table_geom]
    m.body_pos[tb][2] += dz
    m.geom_size[g][0] += grow
    m.geom_size[g][1] += grow
    m.geom_aabb[g] = [0.0, 0.0, 0.0, *m.geom_size[g]]
    m.geom_rbound[g] = float(np.linalg.norm(m.geom_size[g]))
    (x0, x1), (y0, y1) = scene.cfg.table_x, scene.cfg.table_y
    cfg = replace(scene.cfg, table_z=scene.cfg.table_z + dz, table_x=(x0 - grow, x1 + grow),
                  table_y=(y0 - grow, y1 + grow))
    return replace(scene, g1=replace(scene.g1, model=m), cfg=cfg)


def _contacts(scene: ReplayScene, d: mujoco.MjData) -> tuple[float, float, bool, float]:
    """(max robot-robot depth, max robot-table depth, any robot-object contact, max robot-object depth) of the
    current contacts (depths 0 when none penetrate). Vectorised over the contact list (the per-contact
    Python loop was ~60% of a simulated run); the same maxima of the same depths, so the values are identical."""
    n = d.ncon
    if n == 0:
        return 0.0, 0.0, False, 0.0
    con = d.contact
    g = np.asarray(con.geom).reshape(n, 2)
    depth = -np.asarray(con.dist, dtype=np.float64).reshape(n)
    rob, bodies, ob, tg = scene.robot_geoms, scene.g1.model.geom_bodyid, scene.object_body, scene.table_geom
    r1, r2 = rob[g[:, 0]], rob[g[:, 1]]
    obj = (r1 & (bodies[g[:, 1]] == ob)) | (r2 & (bodies[g[:, 0]] == ob))
    slf = r1 & r2
    tab = ~slf & ((r1 & (g[:, 1] == tg)) | (r2 & (g[:, 0] == tg)))

    def deepest(mask: np.ndarray) -> float:
        return max(0.0, float(depth[mask].max())) if mask.any() else 0.0

    return deepest(slf), deepest(tab), bool(obj.any()), deepest(obj)


def _finger_depths(scene: ReplayScene, d: mujoco.MjData, finger_bodies: tuple[np.ndarray, ...]) -> np.ndarray:
    """`(F,)` deepest penetration (m, >= 0) of each finger's bodies into the object in the current contacts."""
    out = np.zeros(len(finger_bodies))
    n = d.ncon
    if n == 0:
        return out
    g = np.asarray(d.contact.geom).reshape(n, 2)
    depth = -np.asarray(d.contact.dist, dtype=np.float64).reshape(n)
    b = scene.g1.model.geom_bodyid
    b1, b2 = b[g[:, 0]], b[g[:, 1]]
    ob = scene.object_body
    for k, mask in enumerate(finger_bodies):
        sel = (mask[b1] & (b2 == ob)) | (mask[b2] & (b1 == ob))
        if sel.any():
            out[k] = max(0.0, float(depth[sel].max()))
    return out


def replay(scene: ReplayScene, traj: RobotTrajectory, object_ref: np.ndarray, object_start: np.ndarray,
           criteria: SuccessCriteria | None = None, self_tol: float = 0.001, table_tol: float = 0.005,
           ref_rows: np.ndarray | None = None, finger_bodies: tuple[np.ndarray, ...] | None = None
           ) -> tuple[ReplayMetrics, dict[str, np.ndarray]]:
    """Replay `traj` (module docstring). `object_ref` `(K,4,4)`: reference object poses in the robot frame at the
    trajectory's source frames (`traj.source_t`). `ref_rows` (bool `(K,)`, default all): the rows the reference may be
    read on, the retarget's USED rows; its first and last define the reference start, end and displacement. Rows
    outside it (past the clip end, unobserved) are the solver's extrapolations, not the human's motion.
    Returns the metrics and per-arm-tick logs; `object_depth` is the deepest robot-object penetration of the physics
    step after each logged tick (m, 0 = none), and with `finger_bodies` (one bool `(nbody,)` mask per finger)
    `finger_depth` `(ticks, F)` the deepest penetration of each finger's bodies into the object.
    Raises ValueError when `ref_rows` selects no row."""
    criteria = SuccessCriteria() if criteria is None else criteria
    rows = np.arange(len(object_ref)) if ref_rows is None else np.flatnonzero(np.asarray(ref_rows, dtype=bool))
    if rows.size == 0:
        raise ValueError("no reference row to measure the object's motion on")
    g1, m = scene.g1, scene.g1.model
    d = mujoco.MjData(m)
    d.qpos[:] = g1.default_qpos()
    d.qpos[g1.right_arm_qpos] = traj.arm_q[0]
    d.qpos[g1.right_hand_qpos] = traj.hand_q[0]
    oq = scene.object_qpos
    d.qpos[oq:oq + 3] = object_start[:3, 3]
    d.qpos[oq + 3:oq + 7] = _quat_wxyz(object_start[:3, :3])
    mujoco.mj_forward(m, d)
    ctrl = np.zeros(m.nu)
    tau_left = left_park_torque(g1)
    dt = m.opt.timestep
    for _ in range(round(scene.cfg.settle_s / dt)):
        g1.robot_ctrl(traj.arm_q[0], traj.hand_q[0], traj.arm_tau_ff[0], ctrl, tau_left)
        d.ctrl[:] = ctrl
        mujoco.mj_step(m, d)
    start = d.qpos[oq:oq + 3].copy()
    settle_drift = float(np.linalg.norm(start - object_start[:3, 3]))
    settle_rot = _rot_deg(object_start[:3, :3], d.qpos[oq + 3:oq + 7])

    arm_rate = 1.0 / float(traj.arm_t[1] - traj.arm_t[0])
    hand_rate = 1.0 / float(traj.hand_t[1] - traj.hand_t[0])
    n_steps = round(traj.duration / dt)
    arm_err, hand_err, lim_ticks = [], [], 0
    self_steps = table_steps = touch_steps = 0
    self_max = table_max = table_force = 0.0
    arm_rng, hand_rng = g1.right_arm_range, g1.right_hand_range
    log_t, log_qa, log_qh, log_obj, log_depth, log_finger = [], [], [], [], [], []
    logged = False
    for step in range(n_steps + 1):
        t = step * dt
        ia = min(int(np.floor(t * arm_rate + 1e-9)), len(traj.arm_t) - 1)
        ih = min(int(np.floor(t * hand_rate + 1e-9)), len(traj.hand_t) - 1)
        if step % max(round(1.0 / (arm_rate * dt)), 1) == 0:
            qa, qh = d.qpos[g1.right_arm_qpos], d.qpos[g1.right_hand_qpos]
            arm_err.append(np.abs(qa - traj.arm_q[ia]))
            hand_err.append(np.abs(qh - traj.hand_q[ih]))
            out = ((qa < arm_rng[:, 0] - 0.01) | (qa > arm_rng[:, 1] + 0.01)).any() or \
                  ((qh < hand_rng[:, 0] - 0.01) | (qh > hand_rng[:, 1] + 0.01)).any()
            lim_ticks += int(out)
            log_t.append(t)
            log_qa.append(qa.copy())
            log_qh.append(qh.copy())
            log_obj.append(d.qpos[oq:oq + 7].copy())
            logged = True
        g1.robot_ctrl(traj.arm_q[ia], traj.hand_q[ih], traj.arm_tau_ff[ia], ctrl, tau_left)
        d.ctrl[:] = ctrl
        mujoco.mj_step(m, d)
        s_d, t_d, touch, o_d = _contacts(scene, d)
        if logged:
            log_depth.append(o_d)
            if finger_bodies is not None:
                log_finger.append(_finger_depths(scene, d, finger_bodies))
            logged = False
        touch_steps += int(touch)
        self_steps += int(s_d > self_tol)
        table_steps += int(t_d > table_tol)
        self_max, table_max = max(self_max, s_d), max(table_max, t_d)
        if t_d > 0.0:
            table_force = max(table_force, _table_force(scene, d))

    end = d.qpos[oq:oq + 3].copy()
    ref0, ref1 = object_ref[rows[0]], object_ref[rows[-1]]
    ref_disp = float(np.linalg.norm(ref1[:3, 3] - ref0[:3, 3]))
    disp = float(np.linalg.norm(end - start))
    final_err = float(np.linalg.norm(end - ref1[:3, 3]))
    r_end = Rotation.from_quat(np.roll(d.qpos[oq + 3:oq + 7], -1)).as_matrix()
    rot_err = float(np.degrees(np.arccos(np.clip((np.trace(r_end.T @ ref1[:3, :3]) - 1) / 2, -1, 1))))
    fell = bool(end[2] < scene.cfg.table_z - 0.05)
    ae, he = np.asarray(arm_err), np.asarray(hand_err)
    arm_rms = float(np.sqrt(np.mean(ae ** 2)))
    failed = []
    if arm_rms > criteria.max_arm_rms:
        failed.append("arm_tracking")
    if lim_ticks:
        failed.append("joint_limits")
    if self_max > criteria.max_self_depth:
        failed.append("self_collision")
    if table_max > criteria.max_table_depth:
        failed.append("table_penetration")
    if ref_disp < criteria.min_ref_move:
        failed.append("reference_does_not_move_object")
    if disp < criteria.min_move_frac * ref_disp:
        failed.append("object_not_moved")
    if final_err > max(criteria.final_err_floor, criteria.max_final_err_frac * ref_disp):
        failed.append("object_final_error")
    if fell:
        failed.append("object_fell")
    metrics = ReplayMetrics(
        duration_s=float(traj.duration), arm_rms=arm_rms, arm_max=float(ae.max()),
        hand_rms=float(np.sqrt(np.mean(he ** 2))), hand_max=float(he.max()), limit_violation_ticks=int(lim_ticks),
        self_collision_steps=int(self_steps), self_collision_max_depth=float(self_max), table_max_depth=float(table_max),
        table_penetration_steps=int(table_steps), object_contact_steps=int(touch_steps), settle_drift=settle_drift, object_displacement=disp,
        reference_displacement=ref_disp, object_final_error=final_err, object_final_rot_err_deg=rot_err,
        object_fell=fell, success=not failed, failed_criteria=tuple(failed), table_max_force=float(table_force),
        settle_rot_deg=settle_rot)
    logs = {"t": np.asarray(log_t), "arm_q": np.asarray(log_qa), "hand_q": np.asarray(log_qh),
            "object_qpos": np.asarray(log_obj), "object_depth": np.asarray(log_depth)}
    if finger_bodies is not None:
        logs["finger_depth"] = np.asarray(log_finger).reshape(len(log_t), len(finger_bodies))
    return metrics, logs


@dataclass
class PhaseStats:
    """Per-phase metrics of `simulate_run` (module docstring)."""
    steps: int = 0
    table_contact_steps: int = 0
    table_max_depth: float = 0.0
    table_max_force: float = 0.0
    self_collision_max_depth: float = 0.0
    object_contact_steps: int = 0
    object_max_depth: float = 0.0
    arm_tracking_max: float = 0.0
    object_start: list | None = None
    object_end: list | None = None

    def to_dict(self) -> dict:
        disp = None if self.object_start is None else float(np.linalg.norm(np.subtract(self.object_end,
                                                                                         self.object_start)))
        return {"steps": self.steps, "table_contact_steps": self.table_contact_steps,
                "table_max_depth": self.table_max_depth, "table_max_force": self.table_max_force,
                "self_collision_max_depth": self.self_collision_max_depth,
                "object_contact_steps": self.object_contact_steps, "object_max_depth": self.object_max_depth,
                "arm_tracking_max": self.arm_tracking_max, "object_displacement": disp}


class RunIO:
    """`g1_sdk_replay`'s robot I/O on the replay scene (module docstring): `now`, `sleep_until` (steps the physics),
    `state`, `send_arm`, `send_hand`. `stop = (phase, fraction)`: an operator stop when that phase is that far along
    (a drill). `pin_hand`: the Dex3 joints are held kinematically at `hand0` while the object settles, so the run
    script MEASURES exactly `hand0` at its start (the measured-state runs at the edge of its hand tolerance; the weak
    finger gains would otherwise let the hand drift past that edge before the script reads it)."""

    def __init__(self, scene: ReplayScene, home: np.ndarray, hand0: np.ndarray, object_start: np.ndarray,
                 settle_s: float = 0.5, stop: tuple[str, float] | None = None, pin_hand: bool = False) -> None:
        self.scene, self.g1 = scene, scene.g1
        m, g1 = self.g1.model, self.g1
        self.m, self.d = m, mujoco.MjData(m)
        d = self.d
        d.qpos[:] = g1.default_qpos()
        d.qpos[g1.right_arm_qpos], d.qpos[g1.left_arm_qpos] = home[0:7], home[7:14]
        d.qpos[g1.right_hand_qpos] = hand0
        oq = scene.object_qpos
        d.qpos[oq:oq + 3] = object_start[:3, 3]
        d.qpos[oq + 3:oq + 7] = _quat_wxyz(object_start[:3, :3])
        mujoco.mj_forward(m, d)
        res = np.zeros(m.nv)
        mujoco.mj_rne(m, d, 0, res)
        left_dof = m.jnt_dofadr[[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in C.LEFT_ARM_JOINTS]]
        self.q_start = np.concatenate([home[0:7], home[7:14]])
        self.tau_start = np.concatenate([res[g1.right_arm_dof], res[left_dof]])
        self.target = self.q_start.copy()
        self.tau = self.tau_start.copy()
        self.hand_target = np.asarray(hand0, dtype=np.float64).copy()
        self.weight = 0.0
        self.cmd_right: np.ndarray | None = None
        self.dt = m.opt.timestep
        self.step = 0
        self.phases: dict[str, PhaseStats] = {}
        self.stop, self.stopped = stop, False
        self.replayer: S.Replayer | None = None
        for _ in range(round(settle_s / self.dt)):     # the object settles while the robot holds its start pose
            self._physics()
            if pin_hand:
                d.qpos[g1.right_hand_qpos] = hand0
                d.qvel[g1.right_hand_dof] = 0.0
        self.step = 0

    def _physics(self) -> None:
        g1, d = self.g1, self.d
        d.ctrl[g1.act_right_arm], d.ctrl[g1.act_left_arm] = self.target[:7], self.target[7:]
        d.ctrl[g1.act_right_hand], d.ctrl[g1.act_left_hand] = self.hand_target, 0.0
        d.ctrl[g1.ff_right_arm], d.ctrl[g1.ff_left_arm] = self.tau[:7], self.tau[7:]
        mujoco.mj_step(self.m, d)

    def now(self) -> float:
        return self.step * self.dt

    def sleep_until(self, t: float) -> None:
        rp = self.replayer
        while self.step * self.dt < t - 1e-9:
            self._physics()
            self.step += 1
            if rp is None:
                continue
            ph = self.phases.setdefault(rp.phase, PhaseStats())
            s_d, t_d, touch, o_d = _contacts(self.scene, self.d)
            obj = self.d.qpos[self.scene.object_qpos:self.scene.object_qpos + 3].tolist()
            ph.steps += 1
            ph.table_contact_steps += int(t_d > 0.0)
            ph.table_max_depth = max(ph.table_max_depth, t_d)
            if t_d > 0.0:
                ph.table_max_force = max(ph.table_max_force, _table_force(self.scene, self.d))
            ph.self_collision_max_depth = max(ph.self_collision_max_depth, s_d)
            ph.object_contact_steps += int(touch)
            ph.object_max_depth = max(ph.object_max_depth, o_d)
            if ph.object_start is None:
                ph.object_start = obj
            ph.object_end = obj
            if self.weight >= 1.0 and self.cmd_right is not None:
                err = float(np.abs(self.d.qpos[self.g1.right_arm_qpos] - self.cmd_right).max())
                ph.arm_tracking_max = max(ph.arm_tracking_max, err)
            self._drill(rp)

    def _drill(self, rp: S.Replayer) -> None:
        if self.stop is None or self.stopped or rp.phase != self.stop[0]:
            return
        if rp.phase == "play":
            ticks = float(rp.plan.arm_t[-1]) / (rp.dt * rp.cfg.speed_scale)
        else:
            ticks = float(rp.path.n) if rp.path is not None else 1.0
        if rp.k >= self.stop[1] * ticks:
            self.stopped = True
            rp.request_stop(f"drill: operator stop at {self.stop[1]:g} of {self.stop[0]}")

    def state(self) -> S.State:
        q = np.zeros(35)
        q[list(S.RIGHT_ARM_DDS)] = self.d.qpos[self.g1.right_arm_qpos]
        q[list(S.LEFT_ARM_DDS)] = self.d.qpos[self.g1.left_arm_qpos]
        t = self.now()
        return S.State(t, 5, q, self.d.qpos[self.g1.right_hand_qpos].copy(), t)

    def send_arm(self, q: dict[int, float], kp: dict, kd: dict, tau: dict[int, float], weight: float,
                 dt: float) -> None:
        cmd = np.array([q[i] for i in S.RIGHT_ARM_DDS + S.LEFT_ARM_DDS])
        ff = np.array([tau.get(i, 0.0) for i in S.RIGHT_ARM_DDS + S.LEFT_ARM_DDS])
        self.weight = float(weight)
        self.target = weight * cmd + (1.0 - weight) * self.q_start
        self.tau = weight * ff + (1.0 - weight) * self.tau_start
        self.cmd_right = cmd[:7]

    def send_hand(self, q: np.ndarray, dt: float) -> None:
        self.hand_target = np.asarray(q, dtype=np.float64).copy()


def simulate_run(scene: ReplayScene, plan: S.Plan, home_index: int, object_start: np.ndarray,
                 cfg: S.Config | None = None, stop: tuple[str, float] | None = None,
                 hand0: np.ndarray | None = None, settle_s: float = 0.5, start_q: np.ndarray | None = None,
                 pin_hand: bool = False) -> dict:
    """The run script on the scene from checked home `home_index` (module docstring): `{"status", "reason",
    "abort_phase", "home", "started_home" (the home the script picked; None when refused), "stop", "run_options",
    "measured_start" (the right arm, left arm, waist (17) and Dex3 hand (7) the script measured at its start),
    "object" (positions at the playback's start and end and at the run's end, m), "phases": {phase:
    PhaseStats.to_dict()}}`. `hand0`: the Dex3 start pose (default 0); `start_q` (17,): the measured right arm, left
    arm and waist the robot starts at (default: the checked home itself; a pose inside its tolerance box);
    `pin_hand`: `RunIO`'s."""
    cfg = S.Config() if cfg is None else cfg
    home = plan.home_q[home_index] if start_q is None else np.asarray(start_q, dtype=np.float64)
    io = RunIO(scene, home, np.zeros(7) if hand0 is None else hand0, object_start, settle_s, stop, pin_hand)
    rp = S.Replayer(plan, io, cfg)
    io.replayer = rp
    st = io.state()                                       # what the run script reads first (no physics step between)
    measured = {"q17": np.concatenate([st.q[list(S.RIGHT_ARM_DDS)], st.q[list(S.LEFT_ARM_DDS)],
                                       st.q[list(S.WAIST_DDS)]]).tolist(), "hand": st.hand_q.tolist()}
    res = rp.run()
    oq = scene.object_qpos
    play = io.phases.get("play")
    return {"status": res["status"], "reason": res.get("reason"), "abort_phase": res.get("abort_phase"),
            "home": home_index, "started_home": rp.home_k, "stop": None if stop is None else f"{stop[0]}@{stop[1]:.2f}",
            "run_options": run_options(cfg), "measured_start": measured,
            "object": {"play_start_m": None if play is None else play.object_start,
                       "play_end_m": None if play is None else play.object_end,
                       "run_end_m": io.d.qpos[oq:oq + 3].tolist()},
            "phases": {k: v.to_dict() for k, v in io.phases.items()}}


def run_options(cfg: S.Config) -> dict:
    """The run-script options a simulated run used (recorded per run, `g1_sdk_replay.SIM_GATE` compares them with the
    package's: a drill simulated at another speed scale than the run)."""
    return {"hold_speed": float(cfg.hold_speed), "speed_scale": float(cfg.speed_scale),
            "send_tau_ff": bool(cfg.send_tau_ff), "hold_waist": bool(cfg.hold_waist),
            "max_track_err": float(cfg.max_track_err), "hold_timeout_s": float(cfg.hold_timeout_s)}
