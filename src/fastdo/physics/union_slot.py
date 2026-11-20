# Copyright 2026 The HandReel authors. SPDX-License-Identifier: Apache-2.0
# Portions adapted from DO AS I DO (https://github.com/malik-group/do-as-i-do, commit 824591b),
# Copyright (c) 2026 malik-group, MIT License; see licenses/DO-AS-I-DO-MIT.txt and THIRD_PARTY_LICENSES.md.
"""One slot of a slot-union batch: DO AS I DO's MPC rollout / optimize for one clip on a world RANGE.

`rollout`, `optimize_once` and `optimize` are `executor.rollout/optimize_once/optimize` (themselves DO AS
I DO's `make_*_fn` bodies op for op) with two changes, both output-preserving:
- the physics step is not launched by the clip: after writing its inputs (perturbation row, controls) the clip yields
  `STEP`; the scheduler launches ONE step of the whole union once every resident clip has yielded, then resumes each
  clip, which reads its step's results (reward from the shared contact buffer) BEFORE the next launch;
- DO AS I DO's env helpers act on the clip's world range only (`save_state`, `load_state`, `save_env_params`,
  `load_env_params`, `set_weld_active`, `set_gravity`, `sync_env` below, each the slice of its namesake in
  `retargeting.utils.mjwp`), and the contact view keeps only the range's contacts (`SlotContactView`).
Host reads block (no event yields): a clip never lets another step run between restoring its state and saving it, so
its worlds see no step it did not ask for. Samples, horizon, iterations, noise, draws (through the clip's `RngSlot`),
weights and the early stop are DO AS I DO's own functions, as in FAST. Registered path only: one domain-randomization
group with zero pair-margin and xy offsets (`check_registered`), reward stride 1.

Torch only (no warp): tested on CPU with a pure-torch fake batch.
"""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import torch

from fastdo.physics.perturb import PerturbPlan, Rng, apply_row, build_schedule
from fastdo.physics.sync_free import RewardStatic, reward_step

__all__ = ["BROADCAST_FIELDS", "STATE_FIELDS", "STEP", "SlotContactView", "SlotCore", "SlotViews", "check_registered",
           "load_env_params", "load_state", "optimize", "optimize_once", "remap_static", "rollout", "save_env_params",
           "save_state", "set_gravity", "set_weld_active", "sync_env"]

STEP = "step"        # yielded by a clip whose inputs for the next union step are written

# mjwp._copy_state (save_state / load_state) and mjwp._broadcast_state (sync_env), in their order
STATE_FIELDS = ("qpos", "qvel", "qacc", "time", "ctrl", "act", "act_dot", "qacc_warmstart", "qfrc_applied",
                "xfrc_applied", "mocap_pos", "mocap_quat")
BROADCAST_FIELDS = ("qpos", "qvel", "time", "ctrl", "qacc", "act", "act_dot", "qfrc_applied", "xfrc_applied",
                    "mocap_pos", "mocap_quat", "xpos", "xquat", "xmat", "geom_xpos", "geom_xmat", "site_xpos")


class SlotContactView:
    """The union's flat contact buffer seen by one world range [lo, hi): `ContactView`'s interface (`dist`, `geom`,
    `worldid` (range-local), `nacon`, `valid()`), where entries of other worlds are invalid."""

    def __init__(self, dist: torch.Tensor, geom: torch.Tensor, worldid: torch.Tensor, nacon: torch.Tensor, lo: int,
                 hi: int) -> None:
        self.dist, self.geom, self._wid, self.nacon, self.lo, self.hi = dist, geom, worldid, nacon, lo, hi
        self.arange = torch.arange(dist.shape[0], device=dist.device)

    @property
    def worldid(self) -> torch.Tensor:
        return self._wid - self.lo

    def valid(self) -> torch.Tensor:
        return (self.arange < self.nacon.to(self.arange.dtype)) & (self._wid >= self.lo) & (self._wid < self.hi)


@dataclass
class SlotViews:
    """Torch views of one slot's worlds: current state (`cur`, BROADCAST and STATE fields), saved state (`prev`,
    STATE fields), per-world `eq_active` (data) and `gravity` (model rows), and the slot's contact view."""

    lo: int
    hi: int
    cur: dict[str, torch.Tensor]
    prev: dict[str, torch.Tensor]
    eq_active: torch.Tensor
    gravity: torch.Tensor
    cv: SlotContactView
    site_xpos_key: str = "site_xpos"

    @property
    def n(self) -> int:
        return self.hi - self.lo

    @property
    def qpos(self) -> torch.Tensor:
        return self.cur["qpos"]

    @property
    def qvel(self) -> torch.Tensor:
        return self.cur["qvel"]

    @property
    def ctrl(self) -> torch.Tensor:
        return self.cur["ctrl"]

    @property
    def xfrc(self) -> torch.Tensor:
        return self.cur["xfrc_applied"]

    @property
    def site_xpos(self) -> torch.Tensor:
        return self.cur[self.site_xpos_key]


@dataclass
class SlotCore:
    """`executor.ClipCore` for a slot: the clip's constants plus a handle `views` the scheduler re-points whenever
    the union is rebuilt (every per-step access goes through it)."""

    config: Any
    S: Any                       # retargeting.utils.sampling
    mul_quat: Any
    rslot: Any                   # lockstep.RngSlot
    rng: Rng
    plan: PerturbPlan
    rstat: RewardStatic          # geom masks in UNION geom ids (`remap_static`)
    rstat_native: RewardStatic   # the clip's own (clip geom ids)
    views: Any                   # SlotViews (None until the batch adopts the clip)
    log: Any
    stats: dict[str, float] = field(default_factory=lambda: {"rollouts": 0, "steps": 0})


def check_registered(config: Any) -> None:
    """The union path supports DO AS I DO's registered physics config only (raise otherwise)."""
    if config.embodiment_type not in ("right", "left"):
        raise NotImplementedError("union executor: single-hand embodiments only")
    if config.use_torch_compile or config.contact_guidance or config.gibbs_sampling:
        raise NotImplementedError("union executor: use_torch_compile / contact_guidance / gibbs_sampling refused")
    for row in config.env_params_list:
        for p in row:
            if set(p) - {"xy_offset", "pair_margin"} or float(p.get("pair_margin", 0.0)) != 0.0:
                raise NotImplementedError(f"union executor: env param {p} is not the registered (zero) one")
    if not config.penetration_margin >= 0.0:
        raise ValueError(f"union executor needs penetration_margin >= 0, got {config.penetration_margin}")


def remap_static(st: RewardStatic, gmap: np.ndarray, ngeom_union: int) -> RewardStatic:
    """`st` with every per-geom mask moved from the clip's geom ids to the union's (`gmap[g]`, -1 = absent). Host
    flags (`*_host`) are the clip's own: unchanged."""
    dev = st.weight.device
    idx = torch.as_tensor(gmap, device=dev)
    keep = idx >= 0

    def mv(mask: torch.Tensor | None) -> torch.Tensor | None:
        if mask is None:
            return None
        out = torch.zeros(ngeom_union, dtype=mask.dtype, device=mask.device)
        out[idx[keep].to(mask.device)] = mask[keep.to(mask.device)]
        return out

    moved: dict[str, Any] = {"is_hand": mv(st.is_hand), "is_object": mv(st.is_object),
                             "any_pedestal": mv(st.any_pedestal), "is_floor": mv(st.is_floor),
                             "body_obj_mask": {b: mv(m) for b, m in st.body_obj_mask.items()}}
    return replace(st, **moved)


# --- mjwp env helpers on a world range ------------------------------------------------------------------------------

def save_state(v: SlotViews) -> None:
    for f in STATE_FIELDS:
        v.prev[f].copy_(v.cur[f])


def load_state(v: SlotViews) -> None:
    for f in STATE_FIELDS:
        v.cur[f].copy_(v.prev[f])


def save_env_params(v: SlotViews) -> dict:
    """`mjwp.save_env_params` (eq_solref / eq_solimp are union-global constants: never changed per slot)."""
    return {"pair_margin": 0.0, "xy_offset": 0.0, "gravity": v.gravity[0].clone(), "eq_active": v.eq_active.clone()}


def load_env_params(config: Any, v: SlotViews, p: dict) -> None:
    """`mjwp.load_env_params` on the range (pair_margin: zero offset = the compiled rows, checked registered)."""
    if "xy_offset" in p:
        q = v.qpos
        q[:, -7:-5] = q[:, -7:-5] + p["xy_offset"]
    if "gravity" in p:
        v.gravity[:] = p["gravity"]
    if "eq_active" in p:
        v.eq_active[:] = p["eq_active"]


def set_weld_active(config: Any, v: SlotViews, active: bool) -> None:
    """`mjwp.set_weld_active` on the range (`eq_active0` is reset-only and the weld solver rows are union-global
    constants equal to the values `set_weld_solver` writes; both set once by the batch)."""
    for eq_id in config.warmup_weld_eq_ids or ():
        v.eq_active[:, eq_id] = active


def set_gravity(v: SlotViews, gravity: list[float]) -> None:
    v.gravity[:, 0] = gravity[0]
    v.gravity[:, 1] = gravity[1]
    v.gravity[:, 2] = gravity[2]


def sync_env(v: SlotViews) -> None:
    """`mjwp.sync_env` = `_broadcast_state`: world 0 of the range to all its worlds."""
    for f in BROADCAST_FIELDS:
        t = v.cur[f]
        t[1:] = t[:1]


# --- the MPC (executor.py mirror) ----------------------------------------------------------------------------------

def rollout(c: SlotCore, ctrls: torch.Tensor, ref_slice: tuple[torch.Tensor, ...],
            env_param: dict) -> Generator[Any, None, tuple[torch.Tensor, torch.Tensor, dict]]:
    """`executor.rollout`; yields STEP before each step and computes that step's reward after it (module doc)."""
    config = c.config
    save_state(c.views)
    init_env_param = save_env_params(c.views)
    load_env_params(config, c.views, env_param)
    N, H = ctrls.shape[:2]
    K = int(config.num_perturb_samples)
    ctrls_world = ctrls.repeat_interleave(K, dim=0) if K > 1 else ctrls
    cum_rew = torch.zeros(N * K, device=config.device)
    cum_info: dict[str, torch.Tensor] | None = None
    _warmup_transition_t = -1
    _warmup_rollout_steps = 0
    _warmup_needs_cleanup = False
    if config.warmup_steps > 0 and config.warmup_sim_step <= config.warmup_steps:
        if config.warmup_sim_step < config.warmup_steps:
            _warmup_transition_t = config.warmup_steps - config.warmup_sim_step - 1
            if _warmup_transition_t >= H:
                _warmup_rollout_steps = H
                _warmup_transition_t = -1
            else:
                _warmup_rollout_steps = max(_warmup_transition_t, 0)
            set_weld_active(config, c.views, True)
            _warmup_needs_cleanup = True
        else:
            set_weld_active(config, c.views, False)
            set_gravity(c.views, list(config.default_gravity))
            _warmup_needs_cleanup = True
    with c.rslot:
        sched = build_schedule(c.plan, c.rng, H, config.warmup_sim_step)
    for t in range(H):
        abs_step = config.warmup_sim_step + t + 1
        if t == _warmup_transition_t:
            set_weld_active(config, c.views, False)
            set_gravity(c.views, list(config.default_gravity))
        if abs_step >= config.warmup_steps:
            apply_row(c.views.xfrc, sched, c.plan, t)
        c.views.ctrl.copy_(ctrls_world[:, t])
        yield STEP
        v = c.views
        ref = tuple(r[t] for r in ref_slice)
        rew, step_info = reward_step(config, c.rstat, v.qpos, v.qvel, v.site_xpos, v.cv, ref, abs_step,  # type: ignore[arg-type]
                                     c.mul_quat)
        if t == H - 1:
            rew = config.terminal_rew_scale * rew
        if t < _warmup_rollout_steps:
            cum_rew += config.warmup_rew_multiplier * rew
        else:
            cum_rew += rew
        if cum_info is None:
            cum_info = {k: x.clone() for k, x in step_info.items()}
        else:
            for k in cum_info:
                cum_info[k] += step_info[k]
    sched.verify()
    effective_H = (H - _warmup_rollout_steps) + config.warmup_rew_multiplier * _warmup_rollout_steps
    mean_rew = (cum_rew / max(effective_H, 1)).view(N, K).mean(dim=-1)
    mean_info = {k: (x / H).view(N, K).mean(dim=-1) for k, x in cum_info.items()} if cum_info else {}
    if _warmup_needs_cleanup:
        if config.warmup_sim_step < config.warmup_steps:
            set_weld_active(config, c.views, True)
        set_gravity(c.views, list(config.default_gravity))
    load_state(c.views)
    load_env_params(config, c.views, init_env_param)
    c.stats["rollouts"] += 1
    c.stats["steps"] += H
    return ctrls, mean_rew, dict(mean_info)


def optimize_once(c: SlotCore, ctrls: torch.Tensor, ref_slice: tuple[torch.Tensor, ...], env_params: list[dict],
                  sample_params: dict | None) -> Generator[Any, None, tuple[torch.Tensor, dict]]:
    """`executor.optimize_once`."""
    config = c.config
    with c.rslot:
        ctrls_samples = c.S.sample_ctrls(config, ctrls, sample_params)
    min_rew = torch.full((config.num_samples,), float("inf"), device=config.device)
    rollout_info: dict = {}
    for env_param in env_params:
        ctrls_samples, rews, rollout_info = yield from rollout(c, ctrls_samples, ref_slice, env_param)
        min_rew = torch.minimum(min_rew, rews)
    rews = min_rew
    weights, nan_mask = c.S._compute_weights_impl(rews, config.num_samples, config.temperature)
    if nan_mask.any():
        c.log.warning(f"NaNs or infs in rews: {nan_mask.sum()}/{config.num_samples}")
    ctrls_mean = (weights[:, None, None] * ctrls_samples).sum(dim=0)
    n_uni = max(0, min(config.num_trace_uniform_samples, config.num_samples))
    n_topk = max(0, min(config.num_trace_topk_samples, config.num_samples))
    idx_uni = (torch.linspace(0, config.num_samples - 1, steps=n_uni, dtype=torch.long, device=config.device)
               if n_uni > 0 else torch.tensor([], dtype=torch.long, device=config.device))
    idx_top = (torch.topk(rews, k=n_topk, largest=True).indices
               if n_topk > 0 else torch.tensor([], dtype=torch.long, device=config.device))
    sel_idx = torch.cat([idx_uni, idx_top], dim=0).long()
    info: dict[str, Any] = {}
    for k, x in rollout_info.items():
        if k not in ["trace", "trace_sample"]:
            if isinstance(x, torch.Tensor):
                x = x.cpu().numpy()
            if x.ndim == 1:
                info[k + "_max"] = x.max()
                info[k + "_min"] = x.min()
                info[k + "_median"] = np.median(x)
                info[k + "_mean"] = x.mean()
    rews_np = rews.cpu().numpy()
    info["improvement"] = rews_np.max() - rews_np[0]
    info["rew_max"] = rews_np.max()
    info["rew_min"] = rews_np.min()
    info["rew_median"] = np.median(rews_np)
    info["rew_mean"] = rews_np.mean()
    info["trace_cost"] = -rews[sel_idx].cpu().numpy()
    return ctrls_mean, info


def optimize(c: SlotCore, ctrls: torch.Tensor,
             ref_slice: tuple[torch.Tensor, ...]) -> Generator[Any, None, tuple[torch.Tensor, dict]]:
    """`executor.optimize`."""
    config = c.config
    infos = []
    sample_params_list = [{"global_noise_scale": config.beta_traj**i} for i in range(config.max_num_iterations)]
    improvement_history = []
    i = 0
    for i in range(config.max_num_iterations):
        ctrls, info = yield from optimize_once(c, ctrls, ref_slice, config.env_params_list[i],
                                               sample_params_list[i])
        infos.append(info)
        improvement_history.append(info["improvement"])
        if len(improvement_history) >= config.improvement_check_steps:
            recent = improvement_history[-config.improvement_check_steps:]
            if all(imp < config.improvement_threshold for imp in recent):
                break
    fake_info = {k: np.zeros_like(x) for k, x in infos[0].items()}
    for _ in range(config.max_num_iterations - len(infos)):
        infos.append(fake_info)
    info_aggregated = {k: np.stack([inf[k] for inf in infos], axis=0) for k in infos[0]}
    info_aggregated["opt_steps"] = np.array([i + 1])
    return ctrls, info_aggregated
