"""MuJoCo Warp backend of the batched tracking environment (install the `policy` extra).

`WarpEnv` keeps every world on the GPU: one `mujoco_warp` `Model`/`Data` pair with `nworld = n_worlds`, one CUDA
graph capturing a single `mujoco_warp.step` (the baseline's `_compile_step` pattern: two eager steps load the
kernels, then `wp.ScopedCapture`), launched `n_substeps` times per control step. Per-world state that the baseline
kept global lives in batched arrays created before the capture: `opt.gravity (n_worlds, 3)` (gravity off during a
world's warm-up) and `eq_data (n_worlds, neq, 11)` (each world's weld target), next to the per-world
`Data.eq_active`. All reads and writes go through zero-copy `wp.to_torch` views, ordered against Warp's CUDA stream
with stream waits; a step needs no host sync (the auto-reset check in `BatchedTrackingEnv.step` makes one).
Contact forces are decoded from `Data.contact`/`Data.efc` by `fastdo.policy.scene.label_contact_forces`, the
same function the CPU twin uses; the slots past `nacon` keep stale contacts of earlier steps and are ignored.

World-state copies (`fastdo.policy.env_state`) gather and scatter the rows of the per-world `Data` arrays in
`_STATE_FIELDS` (time, qpos, qvel, act, history, the solver warm start and qacc, ctrl, applied forces, eq_active,
mocap, userdata and the overflow bits; a field missing from `Data` raises at construction) and of the batched `eq_data`
and gravity; contacts and constraint rows are recomputed by the next step, and sleep state is not copied (`get_state`
refuses a scene with sleeping enabled). GPU contact detection does not promise the same contact order on every run,
so a restored world reproduces its next steps to float32 solver precision, not bit for bit.

Buffers: `nconmax` contacts per world (one pool of `nconmax * n_worlds`) and `njmax` constraint rows per world. A
pyramidal condim-6 contact takes 10 rows, a condim-3 one 4 and the weld 6. The whisking scene needs 66 rows at frame 0
and 196-284 at its contact-rich frames 60-90 on CPU MuJoCo (210 at frame 90 on MuJoCo Warp), and 156 in the model's
qpos0 state, which `put_data` checks at construction; the default is 500. A world whose rows or contacts exceed the
buffers sets its sticky `Data.overflow` bits: constraints were dropped, so the environment fails it like a non-finite
state (`info["overflow"]`), and `_backend_reset` clears its bits.

Memory: MuJoCo Warp allocates from Warp's own CUDA memory pool, outside torch's per-process cap
(`apply_gpu_budget`, applied in `WarpEnv.__init__`); `warp.get_mempool_used_mem_high` reports it (the memory test
bounds both).
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch import Tensor

from fastdo.bench.gpu_budget import apply_gpu_budget
from fastdo.policy.env import BatchedTrackingEnv, EnvConfig
from fastdo.policy.reference import PolicyReference
from fastdo.policy.scene import PolicyScene, label_contact_forces

logger = logging.getLogger(__name__)

__all__ = ["WarpEnv"]

_COMPILE_STEPS = 2
_STATE_FIELDS = ("time", "qpos", "qvel", "act", "history", "qacc_warmstart", "qacc", "ctrl", "qfrc_applied",
                 "xfrc_applied", "eq_active", "mocap_pos", "mocap_quat", "userdata", "overflow")


class WarpEnv(BatchedTrackingEnv):
    """GPU environment over `mujoco_warp` (module docstring)."""

    def __init__(self, cfg: EnvConfig, scene: PolicyScene, device: torch.device,
                 references: Sequence[PolicyReference] | None = None) -> None:
        import mujoco
        import mujoco_warp as mjw
        import warp as wp

        if device.type != "cuda":
            raise ValueError(f"WarpEnv needs a CUDA device, got {device}")
        device = torch.device("cuda", torch.cuda.current_device() if device.index is None else device.index)
        apply_gpu_budget(device=device.index)          # before the first allocation, however the env is built
        self._wp, self._mjw = wp, mjw
        self._wp_device = f"cuda:{device.index}"
        n, m = cfg.n_worlds, scene.model
        wp.init()
        with wp.ScopedDevice(self._wp_device):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                self._model = mjw.put_model(m, batch_sizes={"eq_data": n})
            if caught:   # the scene's condim-6 pairs have zero rolling friction: one line, not one per pair
                logger.info("mujoco_warp.put_model: %d warnings, first: %s", len(caught), caught[0].message)
            gravity: Any = wp.array(np.tile(np.asarray(scene.gravity, dtype=np.float32), (n, 1)), dtype=wp.vec3)
            gravity._is_batched = True
            self._model.opt.gravity = gravity
            data = mujoco.MjData(m)
            mujoco.mj_forward(m, data)
            self._data = mjw.put_data(m, data, nworld=n, nconmax=cfg.nconmax, njmax=cfg.njmax)
        d = self._data
        self._qpos, self._qvel, self._ctrl = wp.to_torch(d.qpos), wp.to_torch(d.qvel), wp.to_torch(d.ctrl)
        self._zeroed = [wp.to_torch(a) for a in (d.qacc_warmstart, d.qacc, d.qfrc_applied, d.xfrc_applied, d.time)
                        if a is not None and a.size > 0]
        self._eq_active = wp.to_torch(d.eq_active)
        self._eq_data = wp.to_torch(self._model.eq_data)
        self._gravity_view = wp.to_torch(self._model.opt.gravity)
        self._nominal_gravity = torch.as_tensor(scene.gravity, dtype=torch.float32, device=device)
        self._contact = (wp.to_torch(d.contact.geom), wp.to_torch(d.contact.dim),
                         wp.to_torch(d.contact.efc_address), wp.to_torch(d.contact.worldid))
        self._nacon, self._efc_force = wp.to_torch(d.nacon), wp.to_torch(d.efc.force)
        self._overflow = wp.to_torch(d.overflow)
        missing = [name for name in _STATE_FIELDS if not hasattr(d, name)]
        if missing:     # a renamed field would silently drop out of every world-state copy
            raise AttributeError(f"mujoco_warp Data has no {missing}: update WarpEnv._STATE_FIELDS for this version")
        arrays = {name: getattr(d, name) for name in _STATE_FIELDS}
        self._state_views = {name: wp.to_torch(a) for name, a in arrays.items() if a is not None and a.size > 0}
        self._state_views.update(eq_data=self._eq_data, gravity=self._gravity_view)
        self._pyramidal = int(m.opt.cone) == int(mujoco.mjtCone.mjCONE_PYRAMIDAL)
        self._geom_label = torch.as_tensor(scene.geom_label, device=device)
        self._geom_is_object = torch.as_tensor(scene.geom_is_object, device=device)
        self._contact_index = torch.arange(self._contact[0].shape[0], device=device)
        self._graph = self._capture()
        self._warp_stream = wp.stream_to_torch(self._wp_device)
        logger.info("WarpEnv: %d worlds on %s, naconmax %d, njmax %d", n, self._wp_device, d.naconmax, d.njmax)
        super().__init__(cfg, scene, device, references)

    def _capture(self) -> Any:
        wp, mjw = self._wp, self._mjw
        with wp.ScopedDevice(self._wp_device):
            for _ in range(_COMPILE_STEPS):              # CUDA forbids module loading inside a capture
                mjw.step(self._model, self._data)
            wp.synchronize()
            with wp.ScopedCapture() as capture:
                mjw.step(self._model, self._data)
            wp.synchronize()
        return capture.graph

    def _backend_reset(self, ids: Tensor, qpos: np.ndarray, weld: np.ndarray, warm: bool) -> None:
        s = self.scene
        q = torch.as_tensor(qpos, dtype=torch.float32, device=self.device)
        self._qpos[ids] = q
        self._qvel[ids] = 0.0
        low = torch.as_tensor(s.ctrl_low, dtype=torch.float32, device=self.device)
        high = torch.as_tensor(s.ctrl_high, dtype=torch.float32, device=self.device)
        self._ctrl[ids] = torch.minimum(torch.maximum(q[:, :s.nq_robot], low), high)
        for view in self._zeroed:
            view[ids] = 0.0
        self._overflow[ids] = 0
        self._eq_active[ids, s.weld_eq] = warm
        self._eq_data[ids, s.weld_eq] = torch.as_tensor(weld, dtype=torch.float32, device=self.device)
        self._gravity_view[ids] = 0.0 if warm else self._nominal_gravity

    def _backend_release(self, mask: Tensor) -> None:
        weld = self.scene.weld_eq
        self._eq_active[:, weld] = self._eq_active[:, weld] & ~mask
        self._gravity_view.copy_(torch.where(mask[:, None], self._nominal_gravity, self._gravity_view))

    def _backend_step(self, ctrl: Tensor) -> None:
        # Warp launches on its own CUDA stream and torch writes and reads on the current one: each waits for the
        # other's pending work on the device (no host sync) so the views are never read or stepped out of order.
        torch_stream = torch.cuda.current_stream(self.device)
        self._ctrl.copy_(ctrl)
        self._warp_stream.wait_stream(torch_stream)
        with self._wp.ScopedDevice(self._wp_device):
            for _ in range(self.cfg.n_substeps):
                self._wp.capture_launch(self._graph)
        torch_stream.wait_stream(self._warp_stream)

    def _read_state(self) -> tuple[Tensor, Tensor]:
        return self._qpos.clone(), self._qvel.clone()

    def _backend_contact_forces(self) -> Tensor:
        geom, dim, efc_address, worldid = self._contact
        valid = self._contact_index < self._nacon[0]
        return label_contact_forces(geom, dim, efc_address[:, 0], worldid, valid, self._efc_force, self._pyramidal,
                                    self._geom_label, self._geom_is_object, self.n_worlds).float()

    def _backend_overflow(self) -> Tensor:
        return self._overflow != 0

    def _backend_get_state(self, ids: Tensor) -> dict[str, Any]:
        return {name: view[ids] for name, view in self._state_views.items()}

    def _backend_set_state(self, ids: Tensor, state: dict[str, Any]) -> None:
        for name, view in self._state_views.items():
            view[ids] = state[name]
