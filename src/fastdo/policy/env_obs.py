"""The tracking environments' observation as a function of logged state.

`observation_from_state(state, prev_x, prev_action, contact_window)` is the one implementation of the actor observation
and the privileged critic features: `fastdo.policy.env.BatchedTrackingEnv` builds every observation with it, and
`fastdo.policy.bc.bc_dataset` rebuilds the observations of the baseline MPC's logged states with it, so a
behaviour-cloned actor sees exactly what the environment will show it. Layout and contents: the
`fastdo.policy.env` module docstring and `fastdo.policy.env_config.obs_layout`.

Everything the observation depends on is an argument: the simulator state (`qpos`, `qvel`), the episode counters
(`k`, `t`, the warm-up length), the reference frames `k..k+H` packed by `fastdo.policy.bank.ReferenceBank.packed`, the
shape descriptor, gravity, the last contact forces (privileged only), the previous canonical x-axis `prev_x`, the
previous residual `prev_action` and the contact-plan window. `observation_state` builds the `ObservationState` of any
worlds stored in a `ReferenceBank`, so callers outside the environment use the same float32 reference storage.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from fastdo.policy.bank import PackedReference, ReferenceBank
from fastdo.policy.canonical import canonical_frame
from fastdo.policy.reference import future_window
from fastdo.policy.scene import (
    PolicyScene,
    object_pose,
    object_twist,
    rotation_6d,
    wrist_pose,
    wrist_twist,
)

__all__ = ["ObservationState", "observation_from_state", "observation_state", "shape_descriptor", "state_kinematics"]


@dataclass(frozen=True)
class ObservationState:
    """What one observation of N worlds is computed from (module docstring), on one device.

    `qpos (N, nq)`, `qvel (N, nv)` float32 simulator state (robot block first, the object free joint at `obj_qpos`,
    its velocity at `obj_dof`); `k (N,)` reference frame and `t (N,)` control steps since the episode start (long);
    `last_index (N,)` each world's last reference frame; `reference` its frames `k..k+horizon`; `plan_fraction (N, L)`
    the planned-contact fraction per label; `force (N, L)` the last per-label contact forces; `shape (4,)` object
    extents and mass; `gravity (3,)`; `warmup_steps` the welded warm-up length W in control steps."""

    qpos: Tensor
    qvel: Tensor
    k: Tensor
    t: Tensor
    last_index: Tensor
    reference: PackedReference
    plan_fraction: Tensor
    force: Tensor
    shape: Tensor
    gravity: Tensor
    horizon: int
    warmup_steps: int
    nq_robot: int
    obj_qpos: int
    obj_dof: int


def state_kinematics(qpos: Tensor, qvel: Tensor, nq_robot: int, obj_qpos: int, obj_dof: int) -> tuple[Tensor, ...]:
    """`(q, qd, T_obj, twist_obj, T_wrist, twist_wrist)` of a batch of simulator states (`fastdo.policy.scene`)."""
    q, qd = qpos[:, :nq_robot], qvel[:, :nq_robot]
    obj_q = qpos[:, obj_qpos:obj_qpos + 7]
    return (q, qd, object_pose(obj_q), object_twist(obj_q, qvel[:, obj_dof:obj_dof + 6]), wrist_pose(q[:, :6]),
            wrist_twist(q[:, :6], qd[:, :6]))


def shape_descriptor(scene: PolicyScene, device: torch.device) -> Tensor:
    """`(4,)` float32 object extents and mass, the scene part of the observation's shape field."""
    return torch.as_tensor([*scene.obj_extents, scene.obj_mass], dtype=torch.float32, device=device)


def observation_state(scene: PolicyScene, bank: ReferenceBank, qpos: Tensor, qvel: Tensor, k: Tensor, t: Tensor,
                      force: Tensor, horizon: int, warmup_steps: int, gravity: Tensor, shape: Tensor
                      ) -> ObservationState:
    """The `ObservationState` of the bank's worlds (all of them, in order) at frames `k` and steps `t`; `shape` is
    `shape_descriptor(scene, device)`."""
    return ObservationState(qpos=qpos, qvel=qvel, k=k, t=t, last_index=bank.last_index(),
                            reference=bank.packed(k, horizon), plan_fraction=bank.plan_fraction, force=force,
                            shape=shape, gravity=gravity, horizon=horizon, warmup_steps=warmup_steps,
                            nq_robot=scene.nq_robot, obj_qpos=scene.obj_qpos, obj_dof=scene.obj_dof)


def observation_from_state(state: ObservationState, prev_x: Tensor, prev_action: Tensor, contact_window: Tensor
                           ) -> tuple[Tensor, Tensor, Tensor]:
    """Observation `(N, obs_dim)`, privileged features `(N, L + 6)` (both float32) and the canonical x-axis `(N, 3)`
    of this observation's frame, which is the `prev_x` of the next one (zeros start a fresh episode).

    `prev_x (N, 3)`, `prev_action (N, nq_robot)` the last residual `delta_max tanh(action)`, `contact_window
    (N, horizon + 1, L)` the contact plan on frames `k..k+horizon` (bool or float)."""
    n, H, W = int(state.qpos.shape[0]), state.horizon, state.warmup_steps
    q, qd, T_obj, tw_obj, T_w, tw_w = state_kinematics(state.qpos, state.qvel, state.nq_robot, state.obj_qpos,
                                                       state.obj_dof)
    F = canonical_frame(T_obj, T_w[:, :3, 3], state.gravity, prev_x)
    x_axis = F[:, :3, 0].clone()
    Rt = F[:, :3, :3].transpose(1, 2)

    def rotate(v: Tensor) -> Tensor:
        return (Rt @ v[..., None])[..., 0]

    packed = state.reference
    obj_window, dq = future_window(packed.obj, packed.base, T_obj, q, H)
    wrist_window, _ = future_window(packed.wrist, packed.base, T_w, q, H)
    obj_twist_f = torch.cat([rotate(tw_obj[:, :3]), rotate(tw_obj[:, 3:])], dim=1)
    phase = torch.stack([(W - state.t).clamp_min(0).float() / max(W, 1),
                         state.k.float() / state.last_index.clamp_min(1).float()], dim=1)
    parts = [rotate(T_w[:, :3, 3] - F[:, :3, 3]), rotation_6d(Rt @ T_w[:, :3, :3]),
             rotate(tw_w[:, :3]), rotate(tw_w[:, 3:]), q[:, 6:], qd[:, 6:], rotation_6d(Rt @ T_obj[:, :3, :3]),
             obj_twist_f, obj_window.reshape(n, -1), wrist_window.reshape(n, -1), dq[:, :, 6:].reshape(n, -1),
             contact_window.reshape(n, -1).float(), state.shape.expand(n, 4), state.plan_fraction, prev_action, phase]
    obs = torch.cat([p.float() for p in parts], dim=1)
    priv = torch.cat([torch.log1p(state.force.clamp_min(0.0)), obj_twist_f], dim=1).float()
    return obs, priv, x_axis
