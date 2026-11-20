"""Calibrated critic noise sigma_V for the tempering floor of twisted SMC.

Exact-ESS tempering is scale-free: whenever the particles' potentials spread at all, lambda shrinks until half of them
are discarded, so a near-perfect critic would resample on its own noise. The floor lambda_j = max(lambda_ESS(K/2),
c sigma_V) (`SmcConfig.lambda_floor_c`, `SmcConfig.sigma_v`) stops potentials that differ by less than the critic's
error from forcing resampling.

sigma_V is the sample sd of the chunk residual

    R_chunk + gamma^H V(s_end) - V(s_start),     R_chunk = sum_h gamma^h r_h,

(the difference-form twisted log-potential, `V(s_end) = 0` after a clip's last chunk) over held-out rollouts of the
trained policy, pooled over clips, chunks and live particles. Every particle run records these residuals
(`SmcResult.chunk_residuals`); `calibrate_sigma_v` runs one best-of-K pass (K independent rollouts with the
configuration's action and reference noise, the twist on, no resampling) per held-out reference.

Chunk-horizon residuals only (`horizon_residuals`). The pool holds the chunks j >= 1 that span
exactly the chunk horizon `c = round(chunk_seconds hz)` steps (`fastdo.policy.smc.chunk_horizon`), the horizon at which
the floor meets the difference-form potentials. Chunk 0 is excluded: it spans the warm-up plus one chunk (W + c steps
from the reset state). A last chunk shorter than c is excluded too. Their residuals are taken over other horizons, so
neither can move sigma_V.
`write_sigma_v` stores the value next to the checkpoint as `<checkpoint stem>.sigma_v.json`; `read_sigma_v` and
`with_calibrated_floor` load and apply it.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fastdo.policy.networks import ActorCritic
from fastdo.policy.protocol import StatefulTrackingEnv
from fastdo.policy.reference import PolicyReference
from fastdo.policy.smc import SmcConfig, chunk_bounds, chunk_horizon, run_best_of_k

logger = logging.getLogger(__name__)

__all__ = ["calibrate_sigma_v", "calibrate_sigma_v_scenes", "horizon_residuals", "read_sigma_v", "sigma_v_path",
           "with_calibrated_floor", "write_sigma_v"]


def horizon_residuals(chunk_residuals: np.ndarray, env: StatefulTrackingEnv, ref: PolicyReference,
                      cfg: SmcConfig) -> np.ndarray:
    """The finite residuals of `chunk_residuals (N, K)` in the chunks j >= 1 of exactly the chunk horizon, on the
    schedule of `env`, `ref` and `cfg` (module docstring, "Chunk-horizon residuals only").

    Raises:
        ValueError: If the table's chunk count differs from the schedule's.
    """
    _, bounds = chunk_bounds(env, ref, cfg)
    n_chunks, horizon = len(bounds) - 1, chunk_horizon(cfg, ref.hz)
    if chunk_residuals.shape[0] != n_chunks:
        raise ValueError(f"chunk_residuals holds {chunk_residuals.shape[0]} chunks, the schedule {n_chunks}")
    keep = [j for j in range(1, n_chunks) if bounds[j + 1] - bounds[j] == horizon]
    values = chunk_residuals[keep]
    return values[np.isfinite(values)]


def calibrate_sigma_v(env: StatefulTrackingEnv, model: ActorCritic, refs: Sequence[PolicyReference], cfg: SmcConfig,
                      generator: torch.Generator) -> tuple[float, int]:
    """`(sigma_V, number of pooled residuals)` over the held-out references `refs` in one environment: the one-group
    case of `calibrate_sigma_v_scenes`."""
    return calibrate_sigma_v_scenes([(env, refs)], model, cfg, generator)


def calibrate_sigma_v_scenes(groups: Sequence[tuple[StatefulTrackingEnv, Sequence[PolicyReference]]], model: ActorCritic,
                             cfg: SmcConfig, generator: torch.Generator) -> tuple[float, int]:
    """`(sigma_V, number of pooled residuals)` over `(environment, held-out references)` groups, one per object scene:
    the sample sd of every group's chunk-horizon residuals pooled together, chunk 0 (warm-up plus one chunk)
    and a shorter last chunk excluded (module docstring).

    Raises:
        ValueError: If fewer than two finite chunk-horizon residuals were recorded (clips of chunk 0 alone, or every
            particle failed before chunk 1), or as `horizon_residuals`.
    """
    pass_cfg = dataclasses.replace(cfg, twist=True)
    pooled: list[np.ndarray] = []
    n_refs = 0
    for env, refs in groups:
        for ref in refs:
            result = run_best_of_k(env, model, ref, cfg.n_particles, generator, cfg=pass_cfg)
            pooled.append(horizon_residuals(result.chunk_residuals, env, ref, pass_cfg))
            n_refs += 1
    residuals = np.concatenate(pooled) if pooled else np.zeros(0)
    if residuals.size < 2:
        raise ValueError(f"need at least two finite chunk-horizon residuals (chunks j >= 1 of the full horizon; chunk 0 "
                         f"excluded) to calibrate sigma_V, got {residuals.size}")
    sigma_v = float(residuals.std(ddof=1))
    logger.info("sigma_V %.5g from %d chunk-horizon residuals (chunk 0 excluded) over %d held-out clips in %d scene(s) "
                "(mean %.4g)", sigma_v, residuals.size, n_refs, len(groups), float(residuals.mean()))
    return sigma_v, int(residuals.size)


def sigma_v_path(checkpoint: str | Path) -> Path:
    """`<checkpoint stem>.sigma_v.json` next to the checkpoint."""
    path = Path(checkpoint)
    return path.with_name(f"{path.stem}.sigma_v.json")


def write_sigma_v(checkpoint: str | Path, sigma_v: float, n_residuals: int, cfg: SmcConfig,
                  extra: dict[str, Any] | None = None) -> Path:
    """Store sigma_V and the chunk horizon it was measured at next to `checkpoint`; returns the file written.

    Raises:
        ValueError: If `sigma_v` is not finite and non-negative.
    """
    if not (math.isfinite(sigma_v) and sigma_v >= 0.0):
        raise ValueError(f"sigma_v must be finite and >= 0, got {sigma_v}")
    path = sigma_v_path(checkpoint)
    record = {"sigma_v": sigma_v, "n_residuals": n_residuals, "checkpoint": Path(checkpoint).name,
              "chunk_seconds": cfg.chunk_seconds, "gamma": cfg.gamma, "n_particles": cfg.n_particles,
              "action_noise_std": cfg.action_noise_std, "reference_noise": cfg.reference_noise, **(extra or {})}
    path.write_text(json.dumps(record, indent=2))
    logger.info("wrote %s (sigma_V %.5g)", path, sigma_v)
    return path


def read_sigma_v(checkpoint: str | Path) -> float:
    """sigma_V stored next to `checkpoint`.

    Raises:
        FileNotFoundError: If the checkpoint was never calibrated.
    """
    return float(json.loads(sigma_v_path(checkpoint).read_text())["sigma_v"])


def with_calibrated_floor(cfg: SmcConfig, checkpoint: str | Path) -> SmcConfig:
    """`cfg` with `sigma_v` read from the checkpoint's calibration file; a missing file leaves the floor off (logged)."""
    path = sigma_v_path(checkpoint)
    if not path.exists():
        logger.warning("%s not found: SMC runs without a tempering floor (sigma_V = 0)", path)
        return dataclasses.replace(cfg, sigma_v=0.0)
    return dataclasses.replace(cfg, sigma_v=read_sigma_v(checkpoint))
