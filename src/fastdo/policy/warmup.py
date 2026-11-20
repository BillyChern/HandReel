# Copyright 2026 The HandReel authors. SPDX-License-Identifier: Apache-2.0
# Portions adapted from DO AS I DO (https://github.com/malik-group/do-as-i-do, commit 824591b),
# Copyright (c) 2026 malik-group, MIT License; see licenses/DO-AS-I-DO-MIT.txt and THIRD_PARTY_LICENSES.md.
"""The baseline's warm-up start for the batched tracking environments (C5-3).

Baseline (`retargeting/pipeline/optimize_physics.py`, `utils/mjwp.py::warmup_analytical_init`,
`utils/io.py::warmup_ref_interp`; do_as_i_do config: `warmup_duration = warmup_ref_base_interp_duration =
warmup_ref_finger_interp_duration = 3.0 s`, `warmup_min_clearance = 0.1 m`). The simulation starts with the finger
joints at 0 and the wrist translated by `-t n` along the world palm normal `n = R_xyz(q_wrist) n_local`, `n_local`
the x-axis of the `<side>_palm` site in its parent body frame (`[0, -1, 0]` without one). `t >= 0` is the smallest
offset that moves every hand mesh vertex (forward kinematics at the closed reference frame 0, at most 4000
stride-subsampled points) out of the `clearance` ball of every object mesh vertex (at most 4000) that it starts
inside: for `d = h - v`, `a = n.d`, `perp^2 = |d|^2 - a^2`, a pair with `|d| < c` and `perp^2 < c^2` needs
`t >= a + sqrt(c^2 - perp^2)`. During the warm-up the wrist and finger references are interpolated linearly from
that start to frame 0 (the object stays welded at frame 0).

Here the offsets of all worlds reset together are computed in one batched, host-sync-free torch pass on the
environment device (the object points follow each world's possibly perturbed `obj_pose[0]`; the hand cloud
depends only on `q_ref[0]`, which randomization does not perturb, and is cached per clip).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import mujoco
import numpy as np
import torch
from torch import Tensor

from fastdo.policy.scene import PolicyScene, wrist_pose

logger = logging.getLogger(__name__)

__all__ = ["DEFAULT_PALM_NORMAL", "MAX_POINTS", "HandCloud", "backoff_distance", "hand_cloud", "object_points",
           "open_hand_start", "palm_normal_local"]

MAX_POINTS = 4000                                  # baseline stride subsampling bound per point set
DEFAULT_PALM_NORMAL = (0.0, -1.0, 0.0)             # baseline _PALM_NORMAL_LOCAL
_PAIR_BLOCK = 2**22                                # (worlds x hand points x object points) per chunk


@dataclass(frozen=True)
class HandCloud:
    """Hand mesh points (Vh, 3) at a clip's reference frame 0 and its world palm normal (3,), float64."""

    points: Tensor
    normal: Tensor


def _subsample(points: np.ndarray) -> np.ndarray:
    n = points.shape[0]
    return points[:: (n + MAX_POINTS - 1) // MAX_POINTS] if n > MAX_POINTS else points


def _body_mesh_points(model: mujoco.MjModel, body: int) -> np.ndarray:
    """Mesh vertices of `body`'s mesh geoms in the body frame (baseline `extract_body_mesh_verts`)."""
    parts = []
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) != body or int(model.geom_type[g]) != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh = int(model.geom_dataid[g])
        if mesh < 0:
            continue
        start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, model.geom_quat[g])
        parts.append(np.asarray(model.mesh_vert[start:start + count], dtype=np.float64) @ rot.reshape(3, 3).T
                     + model.geom_pos[g])
    return np.concatenate(parts) if parts else np.zeros((0, 3))


def _side(scene: PolicyScene) -> str:
    name = mujoco.mj_id2name(scene.model, mujoco.mjtObj.mjOBJ_BODY, scene.wrist_body) or ""
    return name.split("_")[0]


def palm_normal_local(scene: PolicyScene) -> np.ndarray:
    """x-axis of the `<side>_palm` site in its parent body frame, or `DEFAULT_PALM_NORMAL`."""
    site = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_SITE, f"{_side(scene)}_palm")
    if site < 0:
        return np.asarray(DEFAULT_PALM_NORMAL)
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, scene.model.site_quat[site])
    return rot.reshape(3, 3)[:, 0].copy()


def object_points(scene: PolicyScene) -> np.ndarray:
    """Subsampled object mesh points (V, 3) in the object body frame."""
    return _subsample(_body_mesh_points(scene.model, scene.obj_body))


def hand_cloud(scene: PolicyScene, q_robot: Tensor, device: torch.device) -> HandCloud:
    """World hand mesh points at robot joints `q_robot (nq_robot,)` (bodies `<side>_*` with meshes, in body order)
    and the world palm normal `R_xyz(q_robot[3:6]) n_local`."""
    model = scene.model
    q = q_robot.detach().cpu().double().numpy()
    data = mujoco.MjData(model)
    data.qpos[:scene.nq_robot] = q
    mujoco.mj_kinematics(model, data)
    prefix = f"{_side(scene)}_"
    parts = []
    for body in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or ""
        if body == scene.obj_body or not name.startswith(prefix):
            continue
        local = _body_mesh_points(model, body)
        if local.shape[0]:
            parts.append(local @ data.xmat[body].reshape(3, 3).T + data.xpos[body])
    if not parts:
        raise ValueError(f"no {prefix}* body has mesh geoms: the warm-up back-off needs the hand meshes")
    points = torch.as_tensor(_subsample(np.concatenate(parts)), dtype=torch.float64, device=device)
    R = wrist_pose(torch.as_tensor(q[:6], dtype=torch.float64))[:3, :3]
    normal = R @ torch.as_tensor(palm_normal_local(scene), dtype=torch.float64)
    return HandCloud(points=points, normal=normal.to(device))


def backoff_distance(cloud: HandCloud, obj_local: Tensor, obj_pose: Tensor, clearance: float) -> Tensor:
    """Smallest back-off `t (M,)` along `-cloud.normal` for object points `obj_local (V, 3)` placed at each of
    `obj_pose (M, 4, 4)` (module docstring). float64 on the device of `cloud`; no host sync."""
    h = cloud.points
    n = cloud.normal
    obj = obj_pose.to(h)
    v = obj_local.to(h) @ obj[:, :3, :3].transpose(1, 2) + obj[:, None, :3, 3]            # (M, V, 3)
    centre = v.mean(dim=(0, 1))                                                          # conditioning only
    h, v = h - centre, v - centre[None, None]
    c2 = clearance * clearance
    m, n_obj = int(v.shape[0]), int(v.shape[1])
    block = max(1, _PAIR_BLOCK // max(m * n_obj, 1))
    t = torch.zeros(m, dtype=h.dtype, device=h.device)
    v_dot_n, v_sq = v @ n, (v * v).sum(-1)                                                # (M, V)
    for start in range(0, int(h.shape[0]), block):
        hb = h[start:start + block]                                                       # (B, 3)
        along = (hb @ n)[None, :, None] - v_dot_n[:, None, :]                             # (M, B, V)
        dist_sq = ((hb * hb).sum(-1)[None, :, None] + v_sq[:, None, :]
                   - 2.0 * torch.einsum("bi,mvi->mbv", hb, v)).clamp_min(0.0)
        slack = c2 - (dist_sq - along * along)
        active = (slack > 0.0) & (dist_sq < c2)
        need = torch.where(active, (along + slack.clamp_min(0.0).sqrt()).clamp_min(0.0), torch.zeros_like(slack))
        t = torch.maximum(t, need.amax(dim=(1, 2)))
    return t


def open_hand_start(q_first: Tensor, backoff: Tensor, normal: Tensor) -> Tensor:
    """Warm-up start joints (M, nq_robot): `q_first` with the fingers (joints 6:) at 0 and the wrist position moved
    by `-backoff (M,) * normal (M, 3)`."""
    start = q_first.clone()
    start[:, 6:] = 0.0
    start[:, :3] = start[:, :3] - backoff[:, None].to(start) * normal.to(start)
    return start
