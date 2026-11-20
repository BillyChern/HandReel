"""The Sharpa hand scene configured for the batched tracking environments (policy design, Sections 5-8).

`load_policy_scene` loads the baseline's generated `scene.xml` through `fastdo.retarget.mj_model.RobotScene` (the
C1 layout: wrist slides + XYZ Euler hinges `0:6`, 22 finger hinges `6:28`, the object free joint `28:35`) and
applies the baseline's physics setup (`retargeting/utils/mjwp.py::setup_mj_model`: implicit-fast integrator, 100
solver and 200 line-search iterations; the warm-up weld's solref/solimp of `set_weld_active`), with the simulator
step `1 / (hz n_substeps)` and PD gains from the environment config (fingers `kp, kd`, wrist `wrist_kp,
wrist_kd`; force = kp (u - q) - kd qdot through MuJoCo's affine bias).

The torch kinematics below are the single implementation both environments observe through: the wrist pose is
`T = [Rx(q3) Ry(q4) Rz(q5), q[0:3]]` (the chain has no offsets, checked against MuJoCo's FK in the tests) and
the object pose comes from the free joint (world position, wxyz quaternion). Twists are `[v, omega]` in the
world frame at the body origin; the free joint's angular velocity is body-local in MuJoCo and is rotated here.

`label_contact_forces` decodes per-finger-label normal forces against the object from raw contact and
constraint arrays (pyramidal cones: the normal force is the sum of the 2 (condim - 1) edge forces; elliptic: the
first row), so the CPU twin (`mjdata_label_forces`) and the Warp environment share it.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch
from torch import Tensor

from fastdo.retarget.labels import FINGER_LABEL_NAMES, robot_link_labels
from fastdo.retarget.mj_model import RobotScene

logger = logging.getLogger(__name__)

__all__ = [
    "N_LABELS",
    "PolicyScene",
    "check_gravity",
    "label_contact_forces",
    "load_policy_scene",
    "mjdata_label_forces",
    "object_pose",
    "object_qpos",
    "object_twist",
    "rotation_6d",
    "weld_eq_data",
    "wrist_pose",
    "wrist_twist",
]

N_LABELS = len(FINGER_LABEL_NAMES)
_WELD_SOLREF = (0.01, 1.0)                      # baseline set_weld_active
_WELD_SOLIMP = (0.9995, 0.9995, 0.001, 0.5, 2.0)
_SOLVER_ITERATIONS, _LS_ITERATIONS = 100, 200   # baseline setup_mj_model
_GRAVITY_TOL = 1e-9


@dataclass(frozen=True)
class PolicyScene:
    """A configured `MjModel` and the indices the environments need. Arrays are read-only by convention."""

    model: mujoco.MjModel
    xml_path: Path
    nq_robot: int                  # wrist + hand qpos/dofs, also the actuator count (28)
    obj_qpos: int                  # first qpos address of the object free joint
    obj_dof: int                   # first dof address of the object free joint
    obj_body: int
    wrist_body: int
    weld_eq: int
    geom_label: np.ndarray         # (ngeom,) finger label of robot geoms, -1 elsewhere
    geom_is_object: np.ndarray     # (ngeom,) bool
    ctrl_low: np.ndarray           # (nu,)
    ctrl_high: np.ndarray          # (nu,)
    obj_extents: np.ndarray        # (3,) axis-aligned extents of the object's collision meshes, body frame (m)
    obj_mass: float
    gravity: np.ndarray            # (3,) nominal gravity, checked to be (0, 0, -|g|)


def check_gravity(model: mujoco.MjModel) -> None:
    """Raise unless gravity is non-zero along world -z: the reward's drop test and the canonical frame use world z,
    so references must be gravity-aligned in the simulator frame."""
    g = np.asarray(model.opt.gravity, dtype=np.float64)
    norm = float(np.linalg.norm(g))
    if norm <= 0.0 or abs(g[0]) > _GRAVITY_TOL * norm or abs(g[1]) > _GRAVITY_TOL * norm or g[2] >= 0.0:
        raise ValueError(f"gravity must point along world -z (references are gravity-aligned), got {g.tolist()}")


def _object_extents(model: mujoco.MjModel, body: int) -> np.ndarray:
    points = []
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != body or model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh = int(model.geom_dataid[g])
        start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, model.geom_quat[g])
        points.append(model.mesh_vert[start:start + count] @ rot.reshape(3, 3).T + model.geom_pos[g])
    if not points:
        return np.zeros(3)
    allp = np.concatenate(points)
    return allp.max(0) - allp.min(0)


def load_policy_scene(xml_path: str | Path, hz: float, n_substeps: int, kp: float, kd: float, wrist_kp: float,
                      wrist_kd: float) -> PolicyScene:
    """Load and configure the scene (module docstring). Raises ValueError on an unexpected layout or gravity."""
    robot = RobotScene.from_xml(xml_path)
    model = robot.model
    layout = robot.layout
    check_gravity(model)
    nq_robot = layout["hand"].stop
    if (layout["wrist"], robot.robot_dof) != (slice(0, 6), slice(0, nq_robot)) or model.nu != nq_robot:
        raise ValueError(f"expected wrist 0:6, robot dofs 0:{nq_robot} and one actuator per robot joint, got "
                         f"{layout}, nu={model.nu}")
    act_qpos = model.jnt_qposadr[model.actuator_trnid[:, 0]]
    if not np.array_equal(act_qpos, np.arange(nq_robot)):
        raise ValueError(f"actuator i must drive robot qpos i, got {act_qpos.tolist()}")

    model.opt.timestep = 1.0 / (hz * n_substeps)
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.opt.iterations, model.opt.ls_iterations = _SOLVER_ITERATIONS, _LS_ITERATIONS
    gains = np.where(np.arange(nq_robot) < 6, wrist_kp, kp)
    damping = np.where(np.arange(nq_robot) < 6, wrist_kd, kd)
    model.actuator_gainprm[:, 0] = gains
    model.actuator_biasprm[:, :3] = np.stack([np.zeros(nq_robot), -gains, -damping], axis=1)

    obj_joint = next(j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE)
    obj_body = int(model.jnt_bodyid[obj_joint])
    welds = [e for e in range(model.neq) if model.eq_type[e] == mujoco.mjtEq.mjEQ_WELD
             and int(model.eq_obj1id[e]) == obj_body and int(model.eq_obj2id[e]) == 0]
    if len(welds) != 1:
        raise ValueError(f"expected one weld of the object body to the world, found {welds}")
    model.eq_solref[welds[0]] = _WELD_SOLREF
    model.eq_solimp[welds[0]] = _WELD_SOLIMP

    geom_body = model.geom_bodyid.astype(np.int64)
    is_object = geom_body == obj_body
    robot_bodies = {int(model.jnt_bodyid[j]) for j in range(model.njnt) if j != obj_joint}
    in_robot = np.array([_in_subtree(model, int(b), robot_bodies) for b in geom_body])
    labels = np.where(in_robot & ~is_object, robot_link_labels(model, geom_body).numpy().astype(np.int64), -1)
    scene = PolicyScene(
        model=model, xml_path=Path(xml_path), nq_robot=nq_robot, obj_qpos=int(model.jnt_qposadr[obj_joint]),
        obj_dof=int(model.jnt_dofadr[obj_joint]), obj_body=obj_body,
        wrist_body=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_hand_C_MC"), weld_eq=welds[0],
        geom_label=labels, geom_is_object=is_object, ctrl_low=model.actuator_ctrlrange[:, 0].copy(),
        ctrl_high=model.actuator_ctrlrange[:, 1].copy(), obj_extents=_object_extents(model, obj_body),
        obj_mass=float(model.body_mass[obj_body]), gravity=np.asarray(model.opt.gravity, dtype=np.float64).copy())
    if scene.wrist_body < 0:
        raise ValueError("the scene has no right_hand_C_MC body (the wrist frame)")
    logger.info("policy scene %s: dt %.4g s, %d robot joints, object mass %.3f kg, extents %s m", xml_path,
                model.opt.timestep, nq_robot, scene.obj_mass, np.round(scene.obj_extents, 3).tolist())
    return scene


def _in_subtree(model: mujoco.MjModel, body: int, roots: set[int]) -> bool:
    while body > 0:
        if body in roots:
            return True
        body = int(model.body_parentid[body])
    return False


def wrist_pose(q_wrist: Tensor) -> Tensor:
    """(..., 6) wrist joints `[x, y, z, rx, ry, rz]` -> (..., 4, 4) `[Rx Ry Rz, p]`."""
    c, s = torch.cos(q_wrist[..., 3:]), torch.sin(q_wrist[..., 3:])
    one, zero = torch.ones_like(c[..., 0]), torch.zeros_like(c[..., 0])

    def rot(axis: int) -> Tensor:
        ca, sa = c[..., axis], s[..., axis]
        rows = {0: [one, zero, zero, zero, ca, -sa, zero, sa, ca],
                1: [ca, zero, sa, zero, one, zero, -sa, zero, ca],
                2: [ca, -sa, zero, sa, ca, zero, zero, zero, one]}[axis]
        return torch.stack(rows, dim=-1).reshape(*ca.shape, 3, 3)

    T = torch.zeros(*q_wrist.shape[:-1], 4, 4, dtype=q_wrist.dtype, device=q_wrist.device)
    T[..., :3, :3] = rot(0) @ rot(1) @ rot(2)
    T[..., :3, 3] = q_wrist[..., :3]
    T[..., 3, 3] = 1.0
    return T


def wrist_twist(q_wrist: Tensor, qd_wrist: Tensor) -> Tensor:
    """World twist `[v, omega]` (..., 6) of the wrist: omega = e_x rx' + Rx e_y ry' + Rx Ry e_z rz'."""
    zero = torch.zeros_like(q_wrist[..., :3])
    eye = torch.eye(3, dtype=q_wrist.dtype, device=q_wrist.device)
    rx = wrist_pose(torch.cat([zero, q_wrist[..., 3:4], zero[..., :2]], dim=-1))[..., :3, :3]
    rxy = wrist_pose(torch.cat([zero, q_wrist[..., 3:5], zero[..., :1]], dim=-1))[..., :3, :3]
    omega = eye[0] * qd_wrist[..., 3:4] + rx[..., :, 1] * qd_wrist[..., 4:5] + rxy[..., :, 2] * qd_wrist[..., 5:6]
    return torch.cat([qd_wrist[..., :3], omega], dim=-1)


def object_pose(qpos_obj: Tensor) -> Tensor:
    """(..., 7) free-joint qpos `[p, quat_wxyz]` -> (..., 4, 4)."""
    from fastdo.geometry.lie import quat_wxyz_to_matrix

    T = torch.zeros(*qpos_obj.shape[:-1], 4, 4, dtype=qpos_obj.dtype, device=qpos_obj.device)
    T[..., :3, :3] = quat_wxyz_to_matrix(qpos_obj[..., 3:7])
    T[..., :3, 3] = qpos_obj[..., :3]
    T[..., 3, 3] = 1.0
    return T


def object_twist(qpos_obj: Tensor, qvel_obj: Tensor) -> Tensor:
    """World twist `[v, omega]` (..., 6) from the free joint's qvel (world linear, body-local angular)."""
    R = object_pose(qpos_obj)[..., :3, :3]
    return torch.cat([qvel_obj[..., :3], (R @ qvel_obj[..., 3:6, None])[..., 0]], dim=-1)


def rotation_6d(R: Tensor) -> Tensor:
    """(..., 3, 3) -> (..., 6): the first two columns, the continuous rotation representation."""
    return R[..., :, :2].transpose(-1, -2).reshape(*R.shape[:-2], 6)


def object_qpos(obj_pose: np.ndarray) -> np.ndarray:
    """(M, 4, 4) poses -> (M, 7) free-joint qpos `[p, quat_wxyz]` (float64, MuJoCo's own conversion)."""
    out = np.zeros((obj_pose.shape[0], 7))
    for i, T in enumerate(np.asarray(obj_pose, dtype=np.float64)):
        out[i, :3] = T[:3, 3]
        mujoco.mju_mat2Quat(out[i, 3:], T[:3, :3].reshape(9).copy())
    return out


def weld_eq_data(scene: PolicyScene, obj_pose: np.ndarray) -> np.ndarray:
    """(M, 11) weld `eq_data` rows holding the object at `obj_pose (M, 4, 4)` (baseline `set_weld_target`:
    relpose position `R^T (anchor - p)`, relpose quaternion `conj(q)`)."""
    rows = np.tile(np.asarray(scene.model.eq_data[scene.weld_eq], dtype=np.float64), (obj_pose.shape[0], 1))
    qpos = object_qpos(obj_pose)
    anchor = rows[:, 0:3]
    rows[:, 3:6] = np.einsum("mji,mj->mi", obj_pose[:, :3, :3], anchor - qpos[:, :3])
    rows[:, 6:10] = qpos[:, 3:7] * np.array([1.0, -1.0, -1.0, -1.0])
    return rows


def label_contact_forces(geom: Tensor, dim: Tensor, efc_address: Tensor, worldid: Tensor, valid: Tensor,
                         efc_force: Tensor, pyramidal: bool, geom_label: Tensor, geom_is_object: Tensor,
                         n_worlds: int) -> Tensor:
    """Per-world, per-label normal force (n_worlds, N_LABELS) of hand-object contacts, without host syncs.

    `geom (C, 2)`, `dim (C,)`, `efc_address (C,)` (first constraint row, -1 if excluded), `worldid (C,)`, `valid
    (C,)` bool; `efc_force (n_worlds, J)`; `geom_label (ngeom,)` (-1 off the robot), `geom_is_object (ngeom,)`."""
    dtype = efc_force.dtype
    g = geom.long().clamp_min(0)
    obj0, obj1 = geom_is_object[g[:, 0]], geom_is_object[g[:, 1]]
    label = torch.where(obj0, geom_label[g[:, 1]], geom_label[g[:, 0]])
    adr, d, w = efc_address.long(), dim.long(), worldid.long().clamp(0, n_worlds - 1)
    keep = valid & (obj0 != obj1) & (label >= 0) & (adr >= 0) & (geom.min(dim=1).values >= 0)
    n_rows = torch.where(d > 1, 2 * (d - 1), torch.ones_like(d)) if pyramidal else torch.ones_like(d)
    offsets = torch.arange(10, device=adr.device)
    rows = adr[:, None] + offsets[None]
    row_ok = (offsets[None] < n_rows[:, None]) & (rows < efc_force.shape[1]) & keep[:, None]
    forces = efc_force[w[:, None], rows.clamp(0, efc_force.shape[1] - 1)]
    normal = torch.where(row_ok, forces, torch.zeros((), dtype=dtype, device=forces.device)).sum(dim=1)
    out = torch.zeros(n_worlds * N_LABELS, dtype=dtype, device=efc_force.device)
    out.index_add_(0, w * N_LABELS + label.clamp_min(0), torch.where(keep, normal, torch.zeros_like(normal)))
    return out.view(n_worlds, N_LABELS)


def mjdata_label_forces(scene: PolicyScene, datas: Sequence[mujoco.MjData], dtype: torch.dtype = torch.float64
                        ) -> Tensor:
    """`label_contact_forces` over a list of CPU `MjData` (one per world), (len(datas), N_LABELS)."""
    n = len(datas)
    n_efc = max((d.nefc for d in datas), default=0)
    efc = torch.zeros(n, max(n_efc, 1), dtype=dtype)
    parts: list[tuple[np.ndarray, ...]] = []
    for w, d in enumerate(datas):
        efc[w, :d.nefc] = torch.as_tensor(d.efc_force[:d.nefc], dtype=dtype)
        c = d.contact
        parts.append((c.geom[:d.ncon], c.dim[:d.ncon], c.efc_address[:d.ncon], np.full(d.ncon, w)))
    geom, dim, adr, world = (torch.as_tensor(np.concatenate([p[i] for p in parts])) for i in range(4))
    return label_contact_forces(geom.reshape(-1, 2), dim, adr, world, torch.ones(geom.numel() // 2, dtype=torch.bool),
                                efc, scene.model.opt.cone == mujoco.mjtCone.mjCONE_PYRAMIDAL,
                                torch.as_tensor(scene.geom_label), torch.as_tensor(scene.geom_is_object), n)
