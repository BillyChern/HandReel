# Copyright 2026 The HandReel authors. SPDX-License-Identifier: Apache-2.0
# Portions adapted from DO AS I DO (https://github.com/malik-group/do-as-i-do, commit 824591b),
# Copyright (c) 2026 malik-group, MIT License; see licenses/DO-AS-I-DO-MIT.txt and THIRD_PARTY_LICENSES.md.
"""Sync-free MPPI rollout / optimize for DO AS I DO's physics stage (`retargeting/utils/sampling.py`).

`rollout`, `optimize_once` and `optimize` below are DO AS I DO's `make_rollout_fn`, `make_optimize_once_fn` and
`make_optimize_fn` bodies, op for op, written as generators that yield after each rollout step. Differences, all
output-preserving:
- the per-step perturbation is replayed from a schedule built before the step loop with the same draws
  (`perturb.build_schedule`), the reward is `sync_free.reward_step` (no host reads), and the per-world trace that
  only feeds the visualiser's `trace_sample` (dropped by `main` before saving) is not collected;
- the host waits for the device only at the end of a rollout (`wait_event`: yields instead of blocking).
Sample count, horizon, iterations, noise schedule, early stop and weights are untouched (DO AS I DO's own functions).
Runs in the `retargeting` env (warp, mujoco_warp, DO AS I DO on sys.path).

Opt-in lever RS (reward stride), OFF by default: `reward_stride` s > 1 evaluates the step reward only on
rollout steps t with t % s == s - 1 and on the terminal step t = H - 1, scaling each non-terminal evaluated step's
reward (and its info terms) by s; the terminal step keeps its own `terminal_rew_scale` weight. This is a
`sub{s}` partial sum (a Riemann sum with an s-times coarser dt). Physics stepping, perturbations and the
executed control loop are unchanged; s = 1 runs the original code path op for op.
"""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from fastdo.physics.lockstep import RngSlot, wait_event
from fastdo.physics.perturb import PerturbPlan, Rng, apply_row, build_schedule, plan_from_env
from fastdo.physics.sync_free import ContactView, RewardStatic, reward_static, reward_step

__all__ = ["ClipCore", "build_core", "check_reward_stride", "optimize", "optimize_once", "reward_evaluated",
           "rollout"]


@dataclass
class ClipCore:
    """Per-clip handles the rollout reads every step (resolved once: views, masks, plans)."""

    config: Any
    env: Any
    M: Any                         # retargeting.utils.mjwp
    S: Any                         # retargeting.utils.sampling
    mul_quat: Any
    wp: Any
    rslot: RngSlot
    rng: Rng
    plan: PerturbPlan
    rstat: RewardStatic
    cv: ContactView
    qpos: torch.Tensor
    qvel: torch.Tensor
    ctrl: torch.Tensor
    xfrc: torch.Tensor
    site_xpos: torch.Tensor
    log: Any
    stats: dict[str, float] = field(default_factory=lambda: {"rollouts": 0, "steps": 0})
    reward_stride: int = 1         # 1 = the registered reward (every rollout step); see module docstring


def check_reward_stride(stride: Any) -> int:
    """A valid reward stride: a positive int (bool refused). Returns it."""
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise ValueError(f"reward_stride must be a positive int, got {stride!r}")
    return stride


def reward_evaluated(t: int, horizon: int, stride: int) -> bool:
    """Whether rollout step t (0-based, of `horizon`) evaluates the reward under `stride` (always, at stride 1)."""
    return stride == 1 or t == horizon - 1 or t % stride == stride - 1


def build_core(config: Any, env: Any, rslot: RngSlot, decide: str = "emulate", reward_stride: int = 1) -> ClipCore:
    """Resolve everything the per-step loop needs from a DO AS I DO env (after `precompute_hand_object_geom_mask`)."""
    reward_stride = check_reward_stride(reward_stride)
    import loguru
    import mujoco
    import warp as wp
    from retargeting.utils import mjwp as M
    from retargeting.utils import sampling as S
    from retargeting.utils.in_hand import body_has_mesh_geom
    from retargeting.utils.math import mul_quat

    dev = torch.device(config.device)
    t = wp.to_torch
    if config.use_torch_compile:
        raise NotImplementedError("fast executor: use_torch_compile=True is not the registered path")
    if not hasattr(config, "_qpos_weight_cache"):
        config._qpos_weight_cache = M._weight_diff_qpos(config)
    if not hasattr(config, "_qpos_group_slices"):
        config._qpos_group_slices = M._build_qpos_group_slices(config)
    model = env.model_cpu
    # L1: masked (inactive) contacts enter the penetration scatter as clamp(0 - margin, 0); only a non-negative
    # margin keeps them exact zeros
    if not config.penetration_margin >= 0.0:
        raise ValueError(f"fast executor needs penetration_margin >= 0, got {config.penetration_margin}")
    rstat = reward_static(config, env, lambda b: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "",
                          device=dev)
    d = env.data_wp
    cv = ContactView(dist=t(d.contact.dist).reshape(-1), geom=t(d.contact.geom).reshape(-1, 2),
                     worldid=t(d.contact.worldid).reshape(-1), nacon=t(d.nacon))
    plan = plan_from_env(config, env, body_has_mesh_geom,
                         lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n))
    return ClipCore(config=config, env=env, M=M, S=S, mul_quat=mul_quat, wp=wp, rslot=rslot,
                    rng=Rng(dev, decide=decide), plan=plan, rstat=rstat, cv=cv, qpos=t(d.qpos), qvel=t(d.qvel),
                    ctrl=t(d.ctrl), xfrc=t(d.xfrc_applied), site_xpos=t(d.site_xpos), log=loguru.logger,
                    reward_stride=reward_stride)


def _event() -> torch.cuda.Event:
    ev = torch.cuda.Event()
    ev.record()
    return ev


def rollout(c: ClipCore, ctrls: torch.Tensor, ref_slice: tuple[torch.Tensor, ...],
            env_param: dict) -> Generator[Any, None, tuple[torch.Tensor, torch.Tensor, dict]]:
    """`make_rollout_fn.rollout` (see module docstring)."""
    config, env, M = c.config, c.env, c.M
    M.save_state(env)
    init_env_param = M.save_env_params(config, env)
    env = M.load_env_params(config, env, env_param)
    for _state in (env._perturb_active, env._perturb_force, env._perturb_torque):
        for _tensor in _state.values():
            _tensor.zero_()
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
            M.set_weld_active(config, env, True)
            _warmup_needs_cleanup = True
        else:
            M.set_weld_active(config, env, False)
            M.set_gravity(env, list(config.default_gravity))
            _warmup_needs_cleanup = True
    with c.rslot:
        sched = build_schedule(c.plan, c.rng, H, config.warmup_sim_step)
    stride = c.reward_stride
    for t in range(H):
        abs_step = config.warmup_sim_step + t + 1
        if t == _warmup_transition_t:
            M.set_weld_active(config, env, False)
            M.set_gravity(env, list(config.default_gravity))
        # step_env(config, env, ctrls_world[:, t], perturb=(abs_step >= warmup_steps), sim_step=abs_step)
        if abs_step >= config.warmup_steps:
            apply_row(c.xfrc, sched, c.plan, t)
        c.ctrl.copy_(ctrls_world[:, t])
        c.wp.capture_launch(env.graph)
        if stride > 1 and not reward_evaluated(t, H, stride):
            yield
            continue
        ref = [r[t] for r in ref_slice]
        rew, step_info = reward_step(config, c.rstat, c.qpos, c.qvel, c.site_xpos, c.cv, ref, abs_step, c.mul_quat)
        if t == H - 1:
            rew = config.terminal_rew_scale * rew
        elif stride > 1:
            rew = stride * rew
            step_info = {k: stride * v for k, v in step_info.items()}
        if t < _warmup_rollout_steps:
            cum_rew += config.warmup_rew_multiplier * rew
        else:
            cum_rew += rew
        if cum_info is None:
            cum_info = {k: v.clone() for k, v in step_info.items()}
        else:
            for k in cum_info:
                cum_info[k] += step_info[k]
        yield
    yield from wait_event(_event())
    sched.verify()
    effective_H = (H - _warmup_rollout_steps) + config.warmup_rew_multiplier * _warmup_rollout_steps
    mean_rew = (cum_rew / max(effective_H, 1)).view(N, K).mean(dim=-1)
    mean_info = {k: (v / H).view(N, K).mean(dim=-1) for k, v in cum_info.items()} if cum_info else {}
    if _warmup_needs_cleanup:
        if config.warmup_sim_step < config.warmup_steps:
            M.set_weld_active(config, env, True)
        M.set_gravity(env, list(config.default_gravity))
    env = M.load_state(env, None)
    env = M.load_env_params(config, env, init_env_param)
    c.stats["rollouts"] += 1
    c.stats["steps"] += H
    return ctrls, mean_rew, dict(mean_info)


def optimize_once(c: ClipCore, ctrls: torch.Tensor, ref_slice: tuple[torch.Tensor, ...], env_params: list[dict],
                  sample_params: dict | None) -> Generator[Any, None, tuple[torch.Tensor, dict]]:
    """`make_optimize_once_fn.optimize_once` (no `trace_sample`; `trace_cost` kept: it is saved)."""
    config = c.config
    with c.rslot:
        ctrls_samples = c.S.sample_ctrls(config, ctrls, sample_params)
    min_rew = torch.full((config.num_samples,), float("inf"), device=config.device)
    rollout_info: dict = {}
    for env_param in env_params:
        ctrls_samples, rews, rollout_info = yield from rollout(c, ctrls_samples, ref_slice, env_param)
        min_rew = torch.minimum(min_rew, rews)
    rews = min_rew
    if config.use_torch_compile:
        weights, nan_mask = c.S._compute_weights_compiled(rews, config.num_samples, config.temperature)
    else:
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
    for k, v in rollout_info.items():
        if k not in ["trace", "trace_sample"]:
            if isinstance(v, torch.Tensor):
                v = v.cpu().numpy()
            if v.ndim == 1:
                info[k + "_max"] = v.max()
                info[k + "_min"] = v.min()
                info[k + "_median"] = np.median(v)
                info[k + "_mean"] = v.mean()
    rews_np = rews.cpu().numpy()
    info["improvement"] = rews_np.max() - rews_np[0]
    info["rew_max"] = rews_np.max()
    info["rew_min"] = rews_np.min()
    info["rew_median"] = np.median(rews_np)
    info["rew_mean"] = rews_np.mean()
    info["trace_cost"] = -rews[sel_idx].cpu().numpy()
    return ctrls_mean, info


def optimize(c: ClipCore, ctrls: torch.Tensor,
             ref_slice: tuple[torch.Tensor, ...]) -> Generator[Any, None, tuple[torch.Tensor, dict]]:
    """`make_optimize_fn.optimize` (no progress bar / callback)."""
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
    fake_info = {k: np.zeros_like(v) for k, v in infos[0].items()}
    for _ in range(config.max_num_iterations - len(infos)):
        infos.append(fake_info)
    info_aggregated = {k: np.stack([inf[k] for inf in infos], axis=0) for k in infos[0]}
    info_aggregated["opt_steps"] = np.array([i + 1])
    return ctrls, info_aggregated
