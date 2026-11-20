"""Batched tracking environments: the shared observation/reward/episode logic and the plain-MuJoCo CPU twin.

`make_env(cfg, scene_xml, device)` returns a `fastdo.policy.protocol.TrackingEnv`: `MujocoCpuEnv` on "cpu" (at most
8 worlds, sequential `mujoco.mj_step`, for tests and debugging) or `fastdo.policy.env_warp.WarpEnv` on "cuda"
(graph-captured `mujoco_warp.step`). Both inherit `BatchedTrackingEnv`, which owns everything except stepping.

Episode. `reset` puts the object on `obj_pose[0]` (never on the object block of `q_ref`, see `fastdo.policy.bank`)
and the robot on its warm-up start (`fastdo.policy.warmup`: fingers open, wrist backed off from the object; frame 0
of `q_ref` without `warmup_open_hand` or without a warm-up), at rest. The first `W = warmup_ctrl_steps` control steps
(`EnvConfig.warmup_duration`, the baseline's 3.0 s) are the baseline's warm-up: the object is welded to `obj_pose[0]`
with gravity off in that world, `k` stays 0, only an invalid state ends the episode, and the joint reference (PD
base and joint reward) of warm-up step j is `start + (j + 1)/W (q_ref[0] - start)`, frame 0 on the last one; then the
weld is released and gravity restored (`optimize_physics.py`: `set_weld_target`, `set_weld_active`, `set_gravity`).
After the warm-up every step advances `k` by one; the PD target is `u = q_ref[k'] + delta`, `delta = delta_max
tanh(action)`, clipped to the actuator ranges, with `k' = k + 1` the frame the step moves to (the reference at the
end of the control period, design Section 7); the reward and termination compare the new state with frame `k'`, and
the action-rate term penalizes the residual (`delta` against the previous `delta`), not the PD target `u`, which would
penalize fast references. A world is done on failure (`rewards.terminated`: drop or tracking loss; or an invalid
state: non-finite, or the simulator's contact/constraint buffers overflowed, `info["overflow"]`) or when `k'` is its
last frame (a truncation: `info["time_outs"]`, which is `done & ~failure` by
construction, hence a subset of `done`); done worlds auto-reset (a fresh randomized draw when `randomize`, from the
installed reference set when one was given). The returned reward is `reward_scale * rewards.step_reward` (times
`warmup_reward_scale` on warm-up steps, the baseline's `warmup_rew_multiplier`; `-w_drop` for an invalid state):
with the default reward weights a step earns at most 2, so `reward_scale = 0.02` (the control period) bounds a clip's
discounted return (gamma 0.99) by 4, the O(1-10) range PPO's `value_clip = 0.2` is written for. Every step adds the
potential-based shaping `reward_scale (gamma Phi(s') - Phi(s))` (design Section 5, `rewards.shaping_term`, one scale
on all steps so it stays potential-based; `info["shaping"]`), Phi against the frame of each state, `Phi(s') = 0` on a
failure and kept at a time-out; the potential of every world's current state is part of its state (`_phi`).

Observation (canonical frame `F = canonical_frame(object pose, wrist position, gravity, previous x)`, Section 3;
`obs_layout` gives the slices, in this order): `wrist_pose` 9 (position in F, first two columns of `R_F^T R_wrist`),
`wrist_twist` 6 (`R_F^T v, R_F^T omega`), `finger_q` 22, `finger_qd` 22, `obj_rot` 6 (`R_F^T R_obj`; the object's
position in F is zero), `obj_twist` 6 (rotated like the wrist's), `obj_window` 6H and `wrist_window` 6H
(`future_window` body-frame twists to frames k+1..k+H), `finger_window` 22H (finger `q_ref - q`), `contact_plan`
6(H+1) (frames k..k+H), `shape` 10 (object extents 3, mass 1, per-label planned-contact fraction 6: the stand-in for
the C2 measure's label mass), `prev_action` 28 (the last `delta`), `phase` 2 (remaining warm-up fraction `(W - t)/W`,
0 after it; `k / (T-1)`). Windows past a world's last frame repeat it. Privileged (critic): `log1p` of the per-label
object contact forces 6 and `obj_twist` in F 6. Heights and gravity are world z (`scene.check_gravity`).
"""

from __future__ import annotations

import copy
import dataclasses
import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch
from torch import Tensor

from fastdo.policy.bank import ClipRegistry, ReferenceBank
from fastdo.policy.env_config import EnvConfig, obs_layout
from fastdo.policy.env_obs import observation_from_state, observation_state, state_kinematics
from fastdo.policy.env_state import MjDataStateMixin, WorldStateMixin
from fastdo.policy.protocol import TIME_OUTS_KEY
from fastdo.policy.randomize import perturb_reference, sample_reference_perturbation
from fastdo.policy.reference import PolicyReference
from fastdo.policy.rewards import (
    StepObservation,
    shaping_potential,
    shaping_term,
    step_reward,
    terminated,
)
from fastdo.policy.scene import (
    N_LABELS,
    PolicyScene,
    load_policy_scene,
    mjdata_label_forces,
    object_pose,
    object_qpos,
    weld_eq_data,
)
from fastdo.policy.warmup import backoff_distance, hand_cloud, object_points, open_hand_start

logger = logging.getLogger(__name__)

__all__ = ["CPU_MAX_WORLDS", "BatchedTrackingEnv", "EnvConfig", "MujocoCpuEnv", "make_env", "obs_layout",
           "observation_from_state"]

CPU_MAX_WORLDS = 8


class BatchedTrackingEnv(WorldStateMixin, ABC):
    """Shared episode, observation and reward logic; subclasses implement the `_backend_*`/state methods (`get_state`
    and `set_state`, the world-state copy, are `fastdo.policy.env_state`)."""

    def __init__(self, cfg: EnvConfig, scene: PolicyScene, device: torch.device,
                 references: Sequence[PolicyReference] | None = None) -> None:
        self.cfg, self.scene, self.device = cfg, scene, device
        n, nu = cfg.n_worlds, scene.nq_robot
        self._layout = obs_layout(cfg.horizon)
        self._bank = ReferenceBank(n, nu, N_LABELS, cfg.hz, device)
        self._clips = ClipRegistry(self._bank)
        self._gen = torch.Generator().manual_seed(cfg.seed)
        self._library: list[int] = []
        self._clip_of = [-1] * n
        self._started = False
        self._obj_points = torch.as_tensor(object_points(scene), dtype=torch.float64, device=device)
        self._k = torch.zeros(n, dtype=torch.long, device=device)
        self._t = torch.zeros(n, dtype=torch.long, device=device)
        self._warm_start = torch.zeros(n, nu, device=device)
        self._prev_delta = torch.zeros(n, nu, device=device)
        self._prev_x = torch.zeros(n, 3, device=device)
        self._force = torch.zeros(n, N_LABELS, device=device)
        self._obs = torch.zeros(n, self.obs_dim, device=device)
        self._priv = torch.zeros(n, self.priv_dim, device=device)
        self._phi = torch.zeros(n, device=device)
        self._gravity = torch.as_tensor(scene.gravity, dtype=torch.float32, device=device)
        self._ctrl_low = torch.as_tensor(scene.ctrl_low, dtype=torch.float32, device=device)
        self._ctrl_high = torch.as_tensor(scene.ctrl_high, dtype=torch.float32, device=device)
        self._shape = torch.as_tensor([*scene.obj_extents, scene.obj_mass], dtype=torch.float32, device=device)
        self._warned_no_covariance = False
        if references:
            self.set_references(references)

    @property
    def obs_dim(self) -> int:
        return max(s.stop for s in self._layout.values())

    @property
    def priv_dim(self) -> int:
        return N_LABELS + 6

    @property
    def act_dim(self) -> int:
        return self.scene.nq_robot

    @property
    def n_worlds(self) -> int:
        return self.cfg.n_worlds

    def obs_layout(self) -> dict[str, slice]:
        return dict(self._layout)

    def reference(self, world: int) -> PolicyReference:
        """The reference world `world` currently tracks (randomized draw, robot-only q_ref, on the env device)."""
        return self._bank.reference(world)

    def set_references(self, references: Sequence[PolicyReference]) -> None:
        """Install the reference set `reset(None, ...)` and auto-resets draw from, uniformly per world (the supported
        way to train on many clips: each is validated once)."""
        self._library = [self._clips.register(ref) for ref in references]
        self._clips.prune([*self._clip_of, *self._library])

    def _pick_clip(self, world: int) -> int:
        if self._library:
            return self._library[int(torch.randint(len(self._library), (), generator=self._gen))]
        if self._clip_of[world] < 0:
            raise RuntimeError("reset(None, ...) needs a reference: pass one to reset or install a reference set")
        return self._clip_of[world]

    def _draw(self, clip: PolicyReference) -> PolicyReference:
        if not self.cfg.randomize:
            return clip
        if clip.covariance is None:
            if not self._warned_no_covariance:
                logger.warning("randomize=True but the reference has no covariance; tracking it unperturbed")
                self._warned_no_covariance = True
            return clip
        return perturb_reference(clip, sample_reference_perturbation(clip.covariance, clip.hz, self._gen))

    def reset(self, ref: Any, world_ids: Tensor | None) -> Tensor:
        """Start new episodes (protocol `TrackingEnv.reset`); returns the observation of every world."""
        ids = list(range(self.n_worlds)) if world_ids is None else sorted(set(world_ids.flatten().tolist()))
        handles = [self._clips.register(ref)] * len(ids) if ref is not None else [self._pick_clip(i) for i in ids]
        self._start_episodes(ids, handles)
        self._clips.prune([*self._clip_of, *self._library])
        self._refresh_observation(*self._read_state())
        return self._obs

    def _start_episodes(self, ids: list[int], handles: list[int]) -> None:
        nu, o = self.scene.nq_robot, self.scene.obj_qpos
        draws = [self._draw(self._clips[h].ref) for h in handles]
        for world, handle, ref in zip(ids, handles, draws, strict=True):
            self._bank.write(world, ref)
            self._clip_of[world] = handle
        first_pose = torch.stack([ref.obj_pose[0].to(device=self.device, dtype=torch.float64) for ref in draws])
        start = self._warmup_start(handles, draws, first_pose)
        pose = first_pose.cpu().numpy()
        qpos = np.zeros((len(ids), int(self.scene.model.nq)))
        qpos[:, :nu] = start.cpu().numpy()
        qpos[:, o:o + 7] = object_qpos(pose)
        index = torch.as_tensor(ids, dtype=torch.long, device=self.device)
        self._backend_reset(index, qpos, weld_eq_data(self.scene, pose), self.cfg.warmup_ctrl_steps > 0)
        self._warm_start[index] = start.float()
        for buf in (self._k, self._t, self._prev_delta, self._prev_x, self._force):
            buf[index] = 0
        self._phi[index] = self._potential(self._read_state()[0])[index]      # Phi of the start state vs frame 0
        self._started = min(self._clip_of) >= 0

    def _warmup_start(self, handles: list[int], draws: list[PolicyReference], first_pose: Tensor) -> Tensor:
        """Warm-up start joints (M, nq_robot) float64: frame 0, or the open-hand back-off (`fastdo.policy.warmup`)
        with the hand cloud cached per clip and, without randomization (every draw is the clip), the offset too."""
        cfg, nu = self.cfg, self.scene.nq_robot
        q_first = torch.stack([ref.q_ref[0, :nu].to(device=self.device, dtype=torch.float64) for ref in draws])
        if cfg.warmup_ctrl_steps == 0 or not cfg.warmup_open_hand:
            return q_first
        backoff = torch.zeros(len(draws), dtype=torch.float64, device=self.device)
        normal = torch.zeros(len(draws), 3, dtype=torch.float64, device=self.device)
        for handle in dict.fromkeys(handles):
            rows = torch.as_tensor([j for j, h in enumerate(handles) if h == handle], device=self.device)
            record = self._clips[handle]
            if "hand" not in record.cache:
                record.cache["hand"] = hand_cloud(self.scene, record.ref.q_ref[0, :nu], self.device)
            cloud = record.cache["hand"]
            if cfg.randomize:
                backoff[rows] = backoff_distance(cloud, self._obj_points, first_pose[rows], cfg.warmup_clearance)
            else:
                if "backoff" not in record.cache:
                    record.cache["backoff"] = backoff_distance(cloud, self._obj_points, first_pose[rows[:1]],
                                                               cfg.warmup_clearance)
                backoff[rows] = record.cache["backoff"]
            normal[rows] = cloud.normal
        return open_hand_start(q_first, backoff, normal)

    def _refresh_observation(self, qpos: Tensor, qvel: Tensor) -> None:
        state = observation_state(self.scene, self._bank, qpos, qvel, self._k, self._t, self._force, self.cfg.horizon,
                                  self.cfg.warmup_ctrl_steps, self._gravity, self._shape)
        window = state.reference.obj.contact_plan.reshape(self.n_worlds, self.cfg.horizon + 1, N_LABELS)
        self._obs, self._priv, self._prev_x = observation_from_state(state, self._prev_x, self._prev_delta, window)

    def _potential(self, qpos: Tensor) -> Tensor:
        """Shaping potential of every world's state against its frame k (`rewards.shaping_potential`)."""
        frame = self._bank.obj[torch.arange(self.n_worlds, device=self.device), self._k]
        o = self.scene.obj_qpos
        return shaping_potential(self.cfg.reward, object_pose(qpos[:, o:o + 7]), frame)

    def privileged(self) -> Tensor:
        return self._priv

    def step(self, action: Tensor) -> tuple[Tensor, Tensor, Tensor, dict[str, Any]]:
        """Advance every world by one control step (protocol `TrackingEnv.step`; episode rules in the module
        docstring). `info`: `time_outs`, `terminated`, `obj_pose`, `contact_force`, `k`, `q`, `warmup` (the step
        ran welded), `overflow` (buffer overflow, a failure), `reward_terms` (unscaled `step_reward` terms), `shaping`
        (the scaled shaping term inside `reward`), all of the state before any auto-reset."""
        cfg, n, W = self.cfg, self.n_worlds, self.cfg.warmup_ctrl_steps
        if not self._started:
            raise RuntimeError("every world needs an episode before step(): call reset(ref, None) first")
        a = action.to(device=self.device, dtype=torch.float32)
        if a.shape != (n, self.act_dim):
            raise ValueError(f"action must be ({n}, {self.act_dim}), got {tuple(a.shape)}")
        delta = cfg.delta_max * torch.tanh(a)
        warm = self._t < W
        last = self._bank.last_index()
        k_next = torch.where(warm, self._k, torch.minimum(self._k + 1, last))
        alpha = ((self._t + 1).float() / max(W, 1)).clamp_max(1.0)[:, None]
        warm_ref = self._warm_start + alpha * (self._bank.q[:, 0] - self._warm_start)
        base = torch.where(warm[:, None], warm_ref, self._bank.q[torch.arange(n, device=self.device), k_next])
        self._backend_step(torch.minimum(torch.maximum(base + delta, self._ctrl_low), self._ctrl_high))
        qpos, qvel = self._read_state()
        force, overflow = self._backend_contact_forces(), self._backend_overflow()
        self._t, self._k = self._t + 1, k_next
        if W > 0:
            self._backend_release(warm & (self._t >= W))

        s = self.scene
        q, qd, T_obj, tw_obj, _, _ = state_kinematics(qpos, qvel, s.nq_robot, s.obj_qpos, s.obj_dof)
        packed = self._bank.packed(self._k, 0)                        # stride 1: flat row i is world i
        current = dataclasses.replace(packed.obj, q_ref=torch.where(warm[:, None], warm_ref, packed.obj.q_ref))
        measured = StepObservation(q=q, qd=qd, obj_pose=T_obj, obj_twist=tw_obj, contact_force=force, action=delta,
                                   prev_action=self._prev_delta, k=packed.base)
        raw, terms = step_reward(cfg.reward, current, measured)
        valid = torch.isfinite(qpos).all(dim=1) & torch.isfinite(qvel).all(dim=1) & ~overflow
        failed = (~warm & terminated(cfg.reward, current, measured)) | ~valid
        time_outs = ~warm & (self._k == last) & ~failed
        done = failed | time_outs
        scale = torch.where(warm, cfg.reward_scale * cfg.warmup_reward_scale, cfg.reward_scale)
        phi_next = shaping_potential(cfg.reward, T_obj, current.obj_pose)
        shaping = cfg.reward_scale * shaping_term(cfg.reward, self._phi, phi_next, failed)
        reward = torch.where(valid, raw, torch.full_like(raw, -cfg.reward.w_drop)) * scale + shaping
        info = {TIME_OUTS_KEY: time_outs, "terminated": failed, "obj_pose": T_obj, "contact_force": force,
                "k": self._k.clone(), "q": q, "warmup": warm, "overflow": overflow, "reward_terms": terms,
                "shaping": shaping}
        self._prev_delta, self._force = delta, force.clone()      # auto-reset zeroes rows in place; info keeps force
        self._phi = torch.where(failed, torch.zeros_like(phi_next), phi_next)
        any_done, any_overflow = torch.stack([done.any(), overflow.any()]).tolist()      # one host sync
        if any_overflow:
            logger.warning("%d world(s) overflowed the contact/constraint buffers (nconmax %d, njmax %d): failed",
                           int(overflow.sum()), cfg.nconmax, cfg.njmax)
        if any_done:
            ids = done.nonzero()[:, 0].tolist()
            self._start_episodes(ids, [self._pick_clip(i) for i in ids])
            qpos, qvel = self._read_state()
        self._refresh_observation(qpos, qvel)
        return self._obs, reward, done, info

    # Backend: `_backend_reset` puts worlds `ids` at rest on `qpos (M, nq)` with ctrl on its robot block and weld rows
    # `weld (M, 11)`, weld on and gravity off iff `warm`, overflow flags cleared; `_backend_release` turns weld and
    # gravity back for worlds in `mask`; `_backend_step` applies `ctrl (N, nu)` for `n_substeps` steps; `_read_state`
    # returns float32 copies of `qpos` and `qvel`; `_backend_contact_forces` the last step's per-label hand-object
    # normal forces (N, N_LABELS); `_backend_overflow` (N,) bool: a world's contact/constraint buffers overflowed.
    @abstractmethod
    def _backend_reset(self, ids: Tensor, qpos: np.ndarray, weld: np.ndarray, warm: bool) -> None: ...

    @abstractmethod
    def _backend_release(self, mask: Tensor) -> None: ...

    @abstractmethod
    def _backend_step(self, ctrl: Tensor) -> None: ...

    @abstractmethod
    def _read_state(self) -> tuple[Tensor, Tensor]: ...

    @abstractmethod
    def _backend_contact_forces(self) -> Tensor: ...

    @abstractmethod
    def _backend_overflow(self) -> Tensor: ...


class MujocoCpuEnv(MjDataStateMixin, BatchedTrackingEnv):
    """CPU twin: one `MjData` per world, stepped sequentially; per-world gravity and weld targets are written into
    the env's own copy of the scene's `MjModel` before each world's substeps. MuJoCo's automatic reset of a diverged
    `MjData` is disabled on that copy (MuJoCo Warp has none; the caller's scene stays valid for `put_model`, which
    rejects the flag), so in both backends a divergence surfaces as a non-finite state, which the shared episode logic
    fails and resets. A full MuJoCo arena (its contact- or constraint-full warning) counts as a buffer overflow."""

    _ARENA_FULL = (int(mujoco.mjtWarning.mjWARN_CONTACTFULL), int(mujoco.mjtWarning.mjWARN_CNSTRFULL))

    def __init__(self, cfg: EnvConfig, scene: PolicyScene, references: Sequence[PolicyReference] | None = None
                 ) -> None:
        if cfg.n_worlds > CPU_MAX_WORLDS:
            raise ValueError(f"MujocoCpuEnv supports n_worlds <= {CPU_MAX_WORLDS}, got {cfg.n_worlds}")
        scene = dataclasses.replace(scene, model=copy.deepcopy(scene.model))
        m = scene.model
        m.opt.disableflags = int(m.opt.disableflags) | int(mujoco.mjtDisableBit.mjDSBL_AUTORESET)
        self._datas = [mujoco.MjData(m) for _ in range(cfg.n_worlds)]
        self._world_gravity = np.tile(scene.gravity, (cfg.n_worlds, 1))
        self._world_weld = np.tile(np.asarray(m.eq_data[scene.weld_eq], dtype=np.float64), (cfg.n_worlds, 1))
        super().__init__(cfg, scene, torch.device("cpu"), references)

    def _backend_reset(self, ids: Tensor, qpos: np.ndarray, weld: np.ndarray, warm: bool) -> None:
        m, s = self.scene.model, self.scene
        for j, world in enumerate(ids.tolist()):
            d = self._datas[world]
            mujoco.mj_resetData(m, d)                     # also clears the arena warnings (overflow flags)
            d.qpos[:] = qpos[j]
            d.ctrl[:] = np.clip(qpos[j, :s.nq_robot], s.ctrl_low, s.ctrl_high)
            d.eq_active[s.weld_eq] = warm
            self._world_weld[world] = weld[j]
            self._world_gravity[world] = 0.0 if warm else s.gravity

    def _backend_release(self, mask: Tensor) -> None:
        for world in mask.nonzero()[:, 0].tolist():
            self._datas[world].eq_active[self.scene.weld_eq] = False
            self._world_gravity[world] = self.scene.gravity

    def _backend_step(self, ctrl: Tensor) -> None:
        m, weld = self.scene.model, self.scene.weld_eq
        u = ctrl.detach().double().numpy()
        try:
            for world, d in enumerate(self._datas):
                m.opt.gravity[:] = self._world_gravity[world]
                m.eq_data[weld] = self._world_weld[world]
                d.ctrl[:] = u[world]
                mujoco.mj_step(m, d, nstep=self.cfg.n_substeps)
        finally:
            m.opt.gravity[:] = self.scene.gravity

    def _read_state(self) -> tuple[Tensor, Tensor]:
        qpos = torch.as_tensor(np.stack([d.qpos for d in self._datas]), dtype=torch.float32)
        qvel = torch.as_tensor(np.stack([d.qvel for d in self._datas]), dtype=torch.float32)
        return qpos, qvel

    def _backend_contact_forces(self) -> Tensor:
        return mjdata_label_forces(self.scene, self._datas).float()

    def _backend_overflow(self) -> Tensor:
        return torch.tensor([any(d.warning[i].number > 0 for i in self._ARENA_FULL) for d in self._datas])


def make_env(cfg: EnvConfig, scene_xml: str | Path, device: str,
             references: Sequence[PolicyReference] | None = None) -> BatchedTrackingEnv:
    """`MujocoCpuEnv` for a CPU `device`, `WarpEnv` for CUDA (which applies `apply_gpu_budget()` before allocating);
    `references` optionally installs the reference set (`set_references`)."""
    dev = torch.device(device)
    if dev.type not in ("cpu", "cuda"):
        raise ValueError(f"device must be cpu or cuda, got {device}")
    if dev.type == "cpu" and cfg.n_worlds > CPU_MAX_WORLDS:
        raise ValueError(f"the CPU environment supports n_worlds <= {CPU_MAX_WORLDS}, got {cfg.n_worlds}")
    scene = load_policy_scene(scene_xml, cfg.hz, cfg.n_substeps, cfg.kp, cfg.kd, cfg.wrist_kp, cfg.wrist_kd)
    if dev.type == "cpu":
        return MujocoCpuEnv(cfg, scene, references)
    from fastdo.policy.env_warp import WarpEnv

    return WarpEnv(cfg, scene, dev, references)
