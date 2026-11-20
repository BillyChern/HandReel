"""Batched forward kinematics for the fast kinematic reference.

`RobotScene.point_kinematics` runs `mj_kinematics` + `mj_comPos` per frame and one `mj_jac` per (frame, point).
`BatchedKinematics.forward` runs one `mj_kinematics` per frame on a private `MjData` and keeps every body pose plus
the world anchor and axis of each robot dof's joint. World points are then `xpos + xmat p`, and the positional
Jacobian column of a point on body b for robot dof d is `axis_d x (p - anchor_d)` for a hinge and `axis_d` for a
slide when d's joint lies on b or an ancestor of b, and exactly 0 otherwise (MuJoCo's `mj_jac` translational rows,
without the centre-of-mass detour). `canonical_jacobian` evaluates those columns directly in the object's
canonical frame, `R^T (a x r) = (R^T a) x (R^T r)`, on the static (point, ancestor dof) pairs only.

The wrist of the baseline's scenes is three world slides then intrinsic X, Y, Z hinges on bodies with zero offsets,
so its terminal pose is `Trans(d0, d1, d2) Rx(d3) Ry(d4) Rz(d5)` with `d = q - qpos0` (`wrist_pose_closed_form`):
MuJoCo moves a hinge or slide by `qpos - qpos0`, and a joint `ref` compiles to `qpos0` (`wrist_qpos0`, zero on the
baseline's scenes). `wrist_chain_is_closed_form` checks that structure on the compiled model before a caller relies
on it.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import torch
from torch import Tensor

from fastdo.geometry.lie import quat_wxyz_to_matrix
from fastdo.retarget.mj_model import RobotScene

__all__ = ["BatchedKinematics", "KinematicsBatch", "canonical_points", "wrist_chain_is_closed_form",
           "wrist_pose_closed_form", "wrist_qpos0"]

_HINGE = int(mujoco.mjtJoint.mjJNT_HINGE)
_SLIDE = int(mujoco.mjtJoint.mjJNT_SLIDE)
_EXACT = 1e-12


@dataclass(frozen=True)
class KinematicsBatch:
    """Per-frame kinematics of a `(T, nq)` qpos batch (float64, world frame)."""

    xpos: Tensor      # (T, nbody, 3)
    xmat: Tensor      # (T, nbody, 3, 3)
    xanchor: Tensor   # (T, n_dof, 3) world anchor of joint model.dof_jntid[d] for robot dof d
    xaxis: Tensor     # (T, n_dof, 3) world axis of robot dof d


class BatchedKinematics:
    """One `mj_kinematics` per frame for a `RobotScene` whose robot dofs are all hinges or slides."""

    def __init__(self, model: mujoco.MjModel, robot_dof: slice, joint_ids: np.ndarray, hinge: np.ndarray,
                 ancestors: np.ndarray) -> None:
        self._model = model
        self._data = mujoco.MjData(model)
        self._robot_dof = robot_dof
        self._joint_ids = joint_ids
        self._hinge = torch.from_numpy(hinge)
        self._ancestors = ancestors
        self._pairs: dict[bytes, tuple[np.ndarray, np.ndarray]] = {}
        self._split: dict[bytes, tuple[Tensor, Tensor, Tensor, Tensor]] = {}

    @classmethod
    def from_scene(cls, scene: RobotScene) -> BatchedKinematics:
        model = scene.model
        dof = scene.robot_dof
        joint_ids = np.array([int(model.dof_jntid[d]) for d in range(dof.start, dof.stop)], dtype=np.int64)
        for d, j in zip(range(dof.start, dof.stop), joint_ids, strict=True):
            if int(model.jnt_type[j]) not in (_HINGE, _SLIDE):
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(j))
                raise ValueError(f"robot dof {d} (joint {name!r}) is neither hinge nor slide")
        hinge = np.array([int(model.jnt_type[j]) == _HINGE for j in joint_ids], dtype=bool)
        ancestors = np.zeros((model.nbody, len(joint_ids)), dtype=bool)
        for body in range(model.nbody):
            b = body
            while b != 0:
                start = int(model.body_jntadr[b])
                for j in range(start, start + int(model.body_jntnum[b])):
                    d = int(model.jnt_dofadr[j])
                    if dof.start <= d < dof.stop:
                        ancestors[body, d - dof.start] = True
                b = int(model.body_parentid[b])
        return cls(model, dof, joint_ids, hinge, ancestors)

    @property
    def n_dof(self) -> int:
        return self._robot_dof.stop - self._robot_dof.start

    def forward(self, q: np.ndarray) -> KinematicsBatch:
        """`q (T, nq)` -> `KinematicsBatch`; `scene.data` is never written."""
        q2 = np.atleast_2d(np.asarray(q, dtype=np.float64))
        model, data = self._model, self._data
        if q2.shape[1] != model.nq:
            raise ValueError(f"expected q with {model.nq} columns, got {q2.shape}")
        T, nbody, joints = q2.shape[0], model.nbody, self._joint_ids
        xpos = np.empty((T, nbody, 3))
        xmat = np.empty((T, nbody, 9))
        xanchor = np.empty((T, joints.shape[0], 3))
        xaxis = np.empty((T, joints.shape[0], 3))
        for t in range(T):
            data.qpos[:] = q2[t]
            mujoco.mj_kinematics(model, data)
            xpos[t] = data.xpos
            xmat[t] = data.xmat
            xanchor[t] = data.xanchor[joints]
            xaxis[t] = data.xaxis[joints]
        return KinematicsBatch(xpos=torch.from_numpy(xpos), xmat=torch.from_numpy(xmat.reshape(T, nbody, 3, 3)),
                               xanchor=torch.from_numpy(xanchor), xaxis=torch.from_numpy(xaxis))

    def ancestor_pairs(self, body_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """`(point index, dof index)` int64 of every point on a body whose ancestry holds that dof's joint."""
        ids = np.ascontiguousarray(body_ids, dtype=np.int64)
        key = ids.tobytes()
        if key not in self._pairs:
            pu, pd = np.nonzero(self._ancestors[ids])
            self._pairs[key] = (pu.astype(np.int64), pd.astype(np.int64))
        return self._pairs[key]

    def world_points(self, batch: KinematicsBatch, body_ids: np.ndarray, local_pts: np.ndarray) -> Tensor:
        """`(T, U, 3)` world positions of body-local `local_pts (U, 3)` on `body_ids (U,)`."""
        ids = torch.from_numpy(np.asarray(body_ids, dtype=np.int64))
        local = torch.as_tensor(np.asarray(local_pts, dtype=np.float64))
        return batch.xpos[:, ids] + torch.einsum("tuij,uj->tui", batch.xmat[:, ids], local)

    def canonical_jacobian(self, batch: KinematicsBatch, body_ids: np.ndarray, canon_pts: Tensor, obj_block: Tensor,
                           mesh_scale: float, frames: Tensor | None = None) -> Tensor:
        """`(T, U, 3, n_dof)` Jacobian of `canon_pts (T, U, 3)` (`canonical_points` of the same batch).

        With `frames` (a `(T_batch,)` bool mask), only those frames are evaluated: `canon_pts` and `obj_block` then
        hold those frames' rows only, and the result is `(frames.sum(), U, 3, n_dof)` with the same per-frame values."""
        xaxis, xanchor = (batch.xaxis, batch.xanchor) if frames is None else \
            (batch.xaxis[frames], batch.xanchor[frames])
        hu, hd, su, sd = self._split_pairs(body_ids)
        rot = quat_wxyz_to_matrix(obj_block[:, 3:7].to(torch.float64))
        pos = obj_block[:, :3].to(torch.float64)
        axis_c = torch.einsum("tji,tdj->tdi", rot, xaxis)
        anchor_c = torch.einsum("tji,tdj->tdi", rot, xanchor - pos[:, None, :]) / mesh_scale
        T, U = canon_pts.shape[0], canon_pts.shape[1]
        # Each pair's column is evaluated by its own branch only (hinge: axis x (p - anchor); slide:
        # axis / scale), the elementwise values the former `torch.where` picked, written straight into a contiguous
        # (T, U, 3, n_dof) tensor instead of a permuted (T, U, n_dof, 3) one.
        n = self.n_dof
        jac = torch.zeros(T, U * 3 * n, dtype=torch.float64)
        if hu.numel():
            cols = torch.linalg.cross(axis_c[:, hd], canon_pts[:, hu] - anchor_c[:, hd], dim=-1)    # (T, H, 3)
            jac.index_copy_(1, self._flat_index(hu, hd, n), cols.reshape(T, -1))
        if su.numel():
            jac.index_copy_(1, self._flat_index(su, sd, n), (axis_c[:, sd] / mesh_scale).reshape(T, -1))
        return jac.view(T, U, 3, n)

    @staticmethod
    def _flat_index(pu: Tensor, pd: Tensor, n: int) -> Tensor:
        """Flat (U, 3, n) positions of the (pair, component) entries `(pu[p], c, pd[p])`, pair-major then component."""
        return ((pu[:, None] * 3 + torch.arange(3)[None, :]) * n + pd[:, None]).reshape(-1)

    def _split_pairs(self, body_ids: np.ndarray) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """`ancestor_pairs` split into hinge-dof pairs `(hu, hd)` and slide-dof pairs `(su, sd)`, as tensors (cached)."""
        ids = np.ascontiguousarray(body_ids, dtype=np.int64)
        key = ids.tobytes()
        if key not in self._split:
            pu, pd = self.ancestor_pairs(ids)
            hinge = self._hinge.numpy()[pd]
            self._split[key] = (torch.from_numpy(pu[hinge]), torch.from_numpy(pd[hinge]),
                                torch.from_numpy(pu[~hinge]), torch.from_numpy(pd[~hinge]))
        return self._split[key]

    def body_pose(self, batch: KinematicsBatch, body_id: int) -> Tensor:
        """`(T, 4, 4)` world pose of `body_id`."""
        pose = torch.zeros(batch.xpos.shape[0], 4, 4, dtype=torch.float64)
        pose[:, :3, :3] = batch.xmat[:, body_id]
        pose[:, :3, 3] = batch.xpos[:, body_id]
        pose[:, 3, 3] = 1.0
        return pose


def canonical_points(world: Tensor, obj_block: Tensor, mesh_scale: float) -> Tensor:
    """`(T, U, 3)` world points -> canonical object frame `R^T (world - p) / mesh_scale`, `obj_block (T, 7)`."""
    rot = quat_wxyz_to_matrix(obj_block[:, 3:7].to(torch.float64))
    pos = obj_block[:, :3].to(torch.float64)
    return torch.einsum("tji,tuj->tui", rot, world.to(torch.float64) - pos[:, None, :]) / mesh_scale


def _rotation(angle: Tensor, axis: int) -> Tensor:
    c, s = torch.cos(angle), torch.sin(angle)
    one, zero = torch.ones_like(c), torch.zeros_like(c)
    if axis == 0:
        rows = (one, zero, zero, zero, c, -s, zero, s, c)
    elif axis == 1:
        rows = (c, zero, s, zero, one, zero, -s, zero, c)
    else:
        rows = (c, -s, zero, s, c, zero, zero, zero, one)
    return torch.stack(rows, dim=-1).reshape(*angle.shape, 3, 3)


def wrist_pose_closed_form(q_wrist: Tensor, qpos0: Tensor) -> Tensor:
    """`(..., 6)` wrist qpos `[x, y, z, rot_x, rot_y, rot_z]` -> `(..., 4, 4)` `Trans(d0, d1, d2) Rx(d3) Ry(d4) Rz(d5)`
    with `d = q_wrist - qpos0`, `qpos0 (6,)` the wrist joints' reference (`wrist_qpos0`): MuJoCo moves hinges and
    slides by `qpos - qpos0`."""
    q = q_wrist.to(torch.float64) - qpos0.to(torch.float64)
    pose = torch.zeros(*q.shape[:-1], 4, 4, dtype=torch.float64)
    pose[..., :3, :3] = _rotation(q[..., 3], 0) @ _rotation(q[..., 4], 1) @ _rotation(q[..., 5], 2)
    pose[..., :3, 3] = q[..., :3]
    pose[..., 3, 3] = 1.0
    return pose


def wrist_qpos0(scene: RobotScene) -> Tensor:
    """`(n_wrist_dof,)` float64 `model.qpos0` of the wrist dofs' joints in dof order (what joint `ref` compiles to;
    hinge values in radians). `wrist_pose_closed_form` subtracts it."""
    model = scene.model
    wrist = scene.joint_layout()["wrist_dof"]
    adr = [int(model.jnt_qposadr[int(model.dof_jntid[d])]) for d in range(wrist.start, wrist.stop)]
    return torch.from_numpy(np.array(model.qpos0[adr], dtype=np.float64))


def wrist_chain_is_closed_form(scene: RobotScene) -> bool:
    """True iff the 6 wrist dofs are slides x, y, z then hinges x, y, z with zero joint `pos`, on 6 distinct bodies
    forming one chain from the world body to `wrist_terminal_body`, each with zero `body_pos`, identity `body_quat`
    and exactly one joint. Joint reference offsets (`model.qpos0`) may be nonzero: the closed form subtracts them."""
    # Imported here: kinematic_terms imports this module at import time.
    from fastdo.retarget.kinematic_terms import wrist_terminal_body

    model = scene.model
    wrist = scene.joint_layout()["wrist_dof"]
    if wrist.stop - wrist.start != 6:
        return False
    joints = [int(model.dof_jntid[d]) for d in range(wrist.start, wrist.stop)]
    for i, j in enumerate(joints):
        expected_axis = np.eye(3)[i % 3]
        if int(model.jnt_type[j]) != (_SLIDE if i < 3 else _HINGE):
            return False
        if np.abs(model.jnt_axis[j] - expected_axis).max() > _EXACT or np.abs(model.jnt_pos[j]).max() > _EXACT:
            return False
    bodies = [int(model.jnt_bodyid[j]) for j in joints]
    if len(set(bodies)) != 6 or int(model.body_parentid[bodies[0]]) != 0:
        return False
    if any(int(model.body_parentid[bodies[i + 1]]) != bodies[i] for i in range(5)):
        return False
    if bodies[-1] != mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, wrist_terminal_body(scene)):
        return False
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    return all(np.abs(model.body_pos[b]).max() <= _EXACT and np.abs(model.body_quat[b] - identity).max() <= _EXACT
               and int(model.body_jntnum[b]) == 1 for b in bodies)
