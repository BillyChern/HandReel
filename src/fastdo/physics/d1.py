# Copyright 2026 The HandReel authors. SPDX-License-Identifier: Apache-2.0
# Portions adapted from DO AS I DO (https://github.com/malik-group/do-as-i-do, commit 824591b),
# Copyright (c) 2026 malik-group, MIT License; see licenses/DO-AS-I-DO-MIT.txt and THIRD_PARTY_LICENSES.md.
"""D1: drop-in replacements of DO AS I DO's `make_optimize_once_fn` / `make_optimize_fn` (`retargeting/utils/sampling.py`).

Signatures, info keys, early stop, iteration padding and the noise schedule are theirs. What changes:
antithetic knot noise, Blom rank weights normalised to DO AS I DO's expected step, and never-worse acceptance
(row N/2 carries the previous iterate of the same control step; if it scores higher than the current plan under the
same perturbation draws, the current update is discarded). With `D1Config(antithetic=False, rank_weights=False,
accept_never_worse=False)` the update is DO AS I DO's op for op (KA-E).

Pure torch: the rollout is injected (DO AS I DO's `make_rollout_fn` product), so this module needs no warp.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from fastdo.physics.planner_config import D1Config
from fastdo.physics.rank_estimator import (
    antithetic_knot_noise,
    knot_interp,
    noisy_rows,
    rank_update,
    rank_weights,
    registered_update,
    registered_weights,
)

__all__ = ["D1Stats", "make_d1_optimize_fn", "make_d1_optimize_once_fn", "sample_knots"]


@dataclass
class D1Stats:
    """Counters the driver writes to its planner record (world-steps are N * K * H per rollout)."""

    rollouts: int = 0
    rollout_world_steps: int = 0
    iterations: int = 0
    reverted: int = 0
    optimize_calls: int = 0
    iteration_wall_s: list[float] = field(default_factory=list)


def sample_knots(config: Any, gns: torch.Tensor, cfg: D1Config) -> torch.Tensor:
    """(N, K, nu) knot noise. Antithetic off: DO AS I DO's `_stage_joint_delta` knot draw, op for op."""
    if cfg.antithetic:
        return antithetic_knot_noise(config.noise_scale, gns)
    return torch.randn_like(config.noise_scale, device=config.device) * config.noise_scale * gns


def _gns(config: Any, sample_params: dict | None) -> torch.Tensor:
    v = sample_params.get("global_noise_scale", 1.0) if sample_params else 1.0
    return v.to(device=config.device) if torch.is_tensor(v) else torch.tensor(float(v), device=config.device)


def _summary_info(rollout_info: dict, rews: torch.Tensor, r_base: torch.Tensor, config: Any) -> dict[str, Any]:
    """DO AS I DO's per-iteration info block (same keys); `improvement` is measured against the kept plan."""
    info: dict[str, Any] = {}
    for k, v in rollout_info.items():
        if k in ("trace", "trace_sample"):
            continue
        if isinstance(v, torch.Tensor):
            v = v.cpu().numpy()
        if v.ndim == 1:
            info[k + "_max"] = v.max()
            info[k + "_min"] = v.min()
            info[k + "_median"] = np.median(v)
            info[k + "_mean"] = v.mean()
    rews_np = rews.cpu().numpy()
    info["improvement"] = rews_np.max() - float(r_base)
    info["rew_max"] = rews_np.max()
    info["rew_min"] = rews_np.min()
    info["rew_median"] = np.median(rews_np)
    info["rew_mean"] = rews_np.mean()
    if "trace" in rollout_info:
        n = config.num_samples
        n_uni = max(0, min(config.num_trace_uniform_samples, n))
        n_top = max(0, min(config.num_trace_topk_samples, n))
        idx_uni = torch.linspace(0, n - 1, steps=n_uni, dtype=torch.long, device=rews.device)
        idx_top = torch.topk(rews, k=n_top, largest=True).indices
        sel = torch.cat([idx_uni, idx_top]).long()
        info["trace_sample"] = rollout_info["trace"][sel].cpu().numpy()
        info["trace_cost"] = -rews[sel].cpu().numpy()
    return info


def make_d1_optimize_once_fn(rollout: Callable, cfg: D1Config, stats: D1Stats | None = None) -> Callable:
    stats = stats if stats is not None else D1Stats()

    def optimize_once(config: Any, env: Any, ctrls: torch.Tensor, ref_slice: tuple, env_params: list | None = None,
                      sample_params: dict | None = None, prev_ctrls: torch.Tensor | None = None):
        env_params = env_params if env_params is not None else [{}]
        knots = sample_knots(config, _gns(config, sample_params), cfg)
        samples = ctrls + knot_interp(knots, config.knot_steps)
        half = config.num_samples // 2
        use_prev = cfg.accept_never_worse and prev_ctrls is not None
        if use_prev:
            if not cfg.antithetic:
                raise ValueError("never-worse acceptance uses the antithetic nominal slot N/2")
            samples[half] = prev_ctrls
        min_rew = torch.full((config.num_samples,), float("inf"), device=config.device)
        rollout_info: dict = {}
        for env_param in env_params:
            samples, rews, rollout_info = rollout(config, env, samples, ref_slice, env_param)
            min_rew = torch.minimum(min_rew, rews)
            stats.rollouts += 1
            stats.rollout_world_steps += int(samples.shape[0] * samples.shape[1] * int(config.num_perturb_samples))
        rews = min_rew
        if not (cfg.rank_weights or cfg.antithetic or cfg.accept_never_worse):
            weights, _ = registered_weights(rews, config.num_samples, config.temperature)
            new = registered_update(samples, weights)
            return new, _summary_info(rollout_info, rews, rews[0], config), False
        safe = torch.where(torch.isfinite(rews), rews, torch.full_like(rews, -1e9))
        kept_prev = bool(use_prev and float(safe[half]) > float(safe[0]))
        if kept_prev:
            stats.reverted += 1
            return prev_ctrls.clone(), _summary_info(rollout_info, rews, rews[half], config), True
        noisy = noisy_rows(knots)
        if cfg.rank_weights:
            w = rank_weights(rews, noisy, cfg.registered_top_q)
            new = rank_update(ctrls, samples, w)
        else:
            weights, _ = registered_weights(rews, config.num_samples, config.temperature)
            new = registered_update(samples, weights)
        return new, _summary_info(rollout_info, rews, rews[0], config), False

    optimize_once.stats = stats  # type: ignore[attr-defined]
    return optimize_once


def make_d1_optimize_fn(optimize_once: Callable, cfg: D1Config, stats: D1Stats | None = None) -> Callable:
    stats = stats if stats is not None else getattr(optimize_once, "stats", D1Stats())

    def optimize(config: Any, env: Any, ctrls: torch.Tensor, ref_slice: tuple,
                 progress_callback: Callable | None = None):
        """DO AS I DO's `optimize` loop (annealing, early stop on improvement, padding) with the D1 iteration."""
        stats.optimize_calls += 1
        infos = []
        improvement_history = []
        prev: torch.Tensor | None = None
        reverted = []
        for i in range(config.max_num_iterations):
            t0 = time.perf_counter()
            sp = {"global_noise_scale": config.beta_traj ** i}
            new, info, kept = optimize_once(config, env, ctrls, ref_slice, config.env_params_list[i], sp,
                                            prev_ctrls=prev if cfg.accept_never_worse else None)
            prev = None if kept else ctrls
            ctrls = new
            reverted.append(1.0 if kept else 0.0)
            infos.append(info)
            improvement_history.append(info["improvement"])
            stats.iterations += 1
            stats.iteration_wall_s.append(time.perf_counter() - t0)
            if progress_callback is not None:
                progress_callback(i + 1, config.max_num_iterations)
            if len(improvement_history) >= config.improvement_check_steps and all(
                    imp < config.improvement_threshold
                    for imp in improvement_history[-config.improvement_check_steps:]):
                break
        n_run = len(infos)
        fake = {k: np.zeros_like(v) for k, v in infos[0].items()}
        infos += [fake] * (config.max_num_iterations - n_run)
        agg = {k: np.stack([inf[k] for inf in infos], axis=0) for k in infos[0]}
        agg["opt_steps"] = np.array([n_run])
        agg["d1_reverted"] = np.array(reverted + [0.0] * (config.max_num_iterations - n_run))
        return ctrls, agg

    optimize.stats = stats  # type: ignore[attr-defined]
    return optimize
