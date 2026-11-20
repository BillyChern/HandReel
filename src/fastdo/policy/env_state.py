"""Batched copies of world states for the tracking environments (the particle system of `fastdo.policy.smc`
copies particles between worlds).

`get_state(world_ids)` returns a `WorldState` of the listed worlds (rows in the order given; a world may be listed
more than once) and `set_state(world_ids, state)` writes its rows into the listed, distinct worlds. So
`set_state(dst, get_state(src))` copies worlds, and `set_state(ids, snapshot)` restores a snapshot any number of times.
A world's state is everything that determines its next step and what that step returns:

- the simulator. On the CPU twin this is the `MjData` integration state (`mjSTATE_INTEGRATION`: time, qpos, qvel,
  act, the solver's warm start, ctrl, applied forces, `eq_active` (the weld flag), mocap, userdata, plugin state and
  history), its warning counters (the buffer-overflow flags) and the world's weld target and gravity vector. On MuJoCo
  Warp it is the same per-world arrays (`fastdo.policy.env_warp.WarpEnv`);
- the episode: the frame index `k`, the step counter `t` (the warm-up flag is `t < W`), the warm-up start joints, the
  previous residual, the previous canonical x-axis, the last contact forces, and the observation and privileged
  features the environment last returned;
- the reference: the world's rows of the `ReferenceBank` (its possibly perturbed draw), and its clip. The clip is the
  auto-reset bookkeeping: a done world restarts it when no reference set is installed.

The environment's generator (reference-set draws and randomization at auto-reset) is shared by all worlds and is not
part of a world's state. Neither backend copies MuJoCo's sleep state (not part of `mjSTATE_INTEGRATION`), so
`get_state` raises for a scene with sleeping enabled. `set_state` replaces the environment's buffers rather than
writing into them, so tensors an earlier `step`, `reset` or `privileged` returned keep their values.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np
import torch
from torch import Tensor

from fastdo.policy.bank import BankRows, ClipRecord, ClipRegistry, ReferenceBank
from fastdo.policy.env_obs import state_kinematics
from fastdo.policy.scene import PolicyScene

__all__ = ["MjDataStateMixin", "WorldState", "WorldStateMixin"]

_INTEGRATION = int(mujoco.mjtState.mjSTATE_INTEGRATION)
_N_WARNINGS = int(mujoco.mjtWarning.mjNWARNING)
_SLEEP = int(mujoco.mjtEnableBit.mjENBL_SLEEP)
_BUFFERS = ("k", "t", "warm_start", "prev_delta", "prev_x", "force", "obs", "priv", "phi")


@dataclass(frozen=True)
class WorldState:
    """The states of M worlds (module docstring); tensors on the environment's device, rows in `get_state` order.
    `phi` is the shaping potential of each world's current state (`fastdo.policy.rewards.shaping_potential`)."""

    k: Tensor
    t: Tensor
    warm_start: Tensor
    prev_delta: Tensor
    prev_x: Tensor
    force: Tensor
    obs: Tensor
    priv: Tensor
    phi: Tensor
    reference: BankRows
    clips: tuple[ClipRecord | None, ...]          # None: the world had no episode
    backend: dict[str, Any]

    @property
    def n_worlds(self) -> int:
        return int(self.k.shape[0])


class WorldStateMixin(ABC):
    """`get_state`/`set_state` of `fastdo.policy.env.BatchedTrackingEnv` over its buffers; each backend implements
    `_backend_get_state`/`_backend_set_state`. `world_kinematics` reads every world's current joint positions and object
    pose (the particle system records the reset state with it, `fastdo.policy.smc`)."""

    device: torch.device
    scene: PolicyScene
    _k: Tensor
    _t: Tensor
    _warm_start: Tensor
    _prev_delta: Tensor
    _prev_x: Tensor
    _force: Tensor
    _obs: Tensor
    _priv: Tensor
    _phi: Tensor
    _bank: ReferenceBank
    _clips: ClipRegistry
    _clip_of: list[int]
    _library: list[int]
    _started: bool

    def _world_index(self, world_ids: Tensor) -> tuple[Tensor, list[int]]:
        ids = torch.as_tensor(world_ids).to(device=self.device, dtype=torch.long).flatten()
        rows = ids.tolist()
        if any(not 0 <= w < self._k.shape[0] for w in rows):
            raise ValueError(f"world ids must lie in [0, {self._k.shape[0]}), got {rows}")
        return ids, rows

    def get_state(self, world_ids: Tensor) -> WorldState:
        """Copies of the states of worlds `world_ids` (module docstring).

        Raises:
            ValueError: If a world id is out of range or the scene enables sleeping (its state is not copied).
        """
        if int(self.scene.model.opt.enableflags) & _SLEEP:
            raise ValueError("world-state copies do not carry MuJoCo sleep state: disable sleeping in the scene")
        ids, rows = self._world_index(world_ids)
        return WorldState(k=self._k[ids], t=self._t[ids], warm_start=self._warm_start[ids],
                          prev_delta=self._prev_delta[ids], prev_x=self._prev_x[ids], force=self._force[ids],
                          obs=self._obs[ids], priv=self._priv[ids], phi=self._phi[ids], reference=self._bank.rows(ids),
                          clips=tuple(self._clips[self._clip_of[w]] if self._clip_of[w] >= 0 else None for w in rows),
                          backend=self._backend_get_state(ids))

    def set_state(self, world_ids: Tensor, state: WorldState) -> None:
        """Write the rows of `state` into the distinct worlds `world_ids` (module docstring)."""
        ids, rows = self._world_index(world_ids)
        if len(rows) != state.n_worlds or len(set(rows)) != len(rows):
            raise ValueError(f"set_state needs {state.n_worlds} distinct world ids, got {rows}")
        for name in _BUFFERS:
            buffer = getattr(self, f"_{name}").clone()
            buffer[ids] = getattr(state, name)
            setattr(self, f"_{name}", buffer)
        self._bank.set_rows(ids, state.reference)
        for world, record in zip(rows, state.clips, strict=True):
            self._clip_of[world] = -1 if record is None else self._clips.adopt(record)
        self._clips.prune([*self._clip_of, *self._library])
        self._backend_set_state(ids, state.backend)
        self._started = min(self._clip_of) >= 0

    def world_kinematics(self) -> tuple[Tensor, Tensor]:
        """Robot joint positions `(n_worlds, nq_robot)` and object poses `(n_worlds, 4, 4)` of every world's current
        state (right after `reset`, the reset state), as `step` reports them in `info["q"]` and `info["obj_pose"]`."""
        qpos, qvel = self._read_state()
        s = self.scene
        q, _, T_obj, _, _, _ = state_kinematics(qpos, qvel, s.nq_robot, s.obj_qpos, s.obj_dof)
        return q.clone(), T_obj

    @abstractmethod
    def _read_state(self) -> tuple[Tensor, Tensor]: ...

    @abstractmethod
    def _backend_get_state(self, ids: Tensor) -> dict[str, Any]: ...

    @abstractmethod
    def _backend_set_state(self, ids: Tensor, state: dict[str, Any]) -> None: ...


class MjDataStateMixin:
    """Backend state of `fastdo.policy.env.MujocoCpuEnv`: per world, `mj_getState(mjSTATE_INTEGRATION)`, the warning
    counters (lastinfo, number), the weld target row and the gravity vector."""

    scene: PolicyScene
    _datas: list[mujoco.MjData]
    _world_weld: np.ndarray
    _world_gravity: np.ndarray

    def _backend_get_state(self, ids: Tensor) -> dict[str, Any]:
        model, rows = self.scene.model, ids.tolist()
        integration = np.empty((len(rows), mujoco.mj_stateSize(model, _INTEGRATION)))
        warning = np.empty((len(rows), _N_WARNINGS, 2), dtype=np.int64)
        for j, world in enumerate(rows):
            data = self._datas[world]
            mujoco.mj_getState(model, data, integration[j], _INTEGRATION)
            warning[j] = [(data.warning[i].lastinfo, data.warning[i].number) for i in range(_N_WARNINGS)]
        return {"integration": integration, "warning": warning, "weld": self._world_weld[rows],
                "gravity": self._world_gravity[rows]}

    def _backend_set_state(self, ids: Tensor, state: dict[str, Any]) -> None:
        model = self.scene.model
        for j, world in enumerate(ids.tolist()):
            data = self._datas[world]
            mujoco.mj_setState(model, data, state["integration"][j], _INTEGRATION)
            for i, (lastinfo, number) in enumerate(state["warning"][j].tolist()):
                data.warning[i].lastinfo, data.warning[i].number = lastinfo, number
            self._world_weld[world] = state["weld"][j]
            self._world_gravity[world] = state["gravity"][j]
