"""Train the universal tracking policy with PPO.

Hydra entry point over `run/conf/config.yaml` with the `policy` group:

  uv run python run/pipeline/train_policy.py +policy=default policy.run=whisking_gpu
  uv run python run/pipeline/train_policy.py +policy=default policy.device=cpu policy.env.n_worlds=8 policy.n_updates=3

`train_from_config` runs the pipeline: references (`policy.data.source`: the shipped whisking kinematic reference, or
the training side of the DexYCB object split of `load_reference_clips`, on exactly one object and one hand side per run,
`policy.data.train_objects` and `hand_side`, checked up front by `check_training_scope`; multi-object training is not
covered here), the environment on that one scene (`training_scene_xml`, `make_env`: the plain-MuJoCo CPU twin or MuJoCo
Warp), an `ActorCritic`, an optional behaviour-cloning warm start from an isolation-study MPC rollout
(`fastdo.policy.bc`), then PPO in curriculum stages. At the start of every stage (`curriculum.stage_updates`) the
reference set is re-installed with that stage's randomization strength (0 -> 1 x Sigma) and clip length (1 s -> the full
clip). The shaping discount (`RewardConfig.gamma`) is PPO's gamma. A checkpoint is written every `checkpoint_every`
updates and after the last one to `<paths.outputs>/policy/<run>/checkpoints/` (`fastdo.policy.checkpoint`), with one
JSON line of PPO statistics per update in `metrics.jsonl` and the resolved configuration in `config.yaml`.

The full run (4096 worlds, ~1e9 env-steps) is GPU work; the merge-gate smoke run is
`tests/run/test_train_policy_smoke.py` (CPU twin, 8 worlds, 3 updates).
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
import yaml
from omegaconf import DictConfig, OmegaConf

from fastdo.bench.gpu_budget import apply_cpu_budget, apply_gpu_budget
from fastdo.policy.bc import (
    BcConfig,
    bc_dataset,
    bc_pretrain,
    load_mpc_states,
    mpc_reference_frequency,
    mpc_steps_per_frame,
)
from fastdo.policy.checkpoint import checkpoint_name, save_checkpoint
from fastdo.policy.data import CurriculumConfig, ReferenceFn, curriculum_reference, curriculum_stage
from fastdo.policy.env import BatchedTrackingEnv, make_env
from fastdo.policy.networks import ActorCritic
from fastdo.policy.pipeline_config import (
    check_training_scope,
    config_dict,
    env_config,
    load_reference_clips,
    ppo_config,
    training_scene_xml,
)
from fastdo.policy.ppo import train
from fastdo.policy.reference import resample_reference
from fastdo.policy.scene import N_LABELS, object_pose

logger = logging.getLogger(__name__)

_NQ_ROBOT = 28


def bc_warm_start(model: ActorCritic, env: BatchedTrackingEnv, node: Any, seed: int) -> dict[str, float]:
    """Behaviour cloning on an isolation-study arm's executed MPC rollout: the control-grid dataset of
    `fastdo.policy.bc.bc_dataset` on the arm's kinematic reference at the environment's rate, then `bc_pretrain`."""
    arm = Path(str(node.arm_dir))
    baseline = yaml.safe_load((arm / "config.yaml").read_text())
    sim_dt, ref_dt, warmup = float(baseline["sim_dt"]), float(baseline["ref_dt"]), int(baseline["warmup_steps"])
    rollout = load_mpc_states(arm / "trajectory_mjwp.npz")
    from fastdo.retarget.kinematic import KinematicTrajectory

    with np.load(arm / "trajectory_kinematic.npz") as data:
        qpos, qvel, frequency = data["qpos"], data["qvel"], float(data["frequency"])
    n_src, cfg = int(qpos.shape[0]), env.cfg
    pose = object_pose(torch.as_tensor(qpos[:, _NQ_ROBOT:_NQ_ROBOT + 7], dtype=torch.float64))
    plan = torch.zeros(n_src, N_LABELS, dtype=torch.bool)
    mpc_kin = KinematicTrajectory(qpos=qpos, qvel=qvel, frequency=mpc_reference_frequency(n_src, ref_dt, sim_dt))
    mpc_ref = resample_reference(mpc_kin, pose, plan, None, hz=1.0 / sim_dt)
    env_ref = resample_reference(KinematicTrajectory(qpos=qpos, qvel=qvel, frequency=frequency), pose, plan, None,
                                 hz=cfg.hz)
    dataset = bc_dataset(env.scene, rollout, mpc_ref, env_ref, mpc_steps_per_frame(n_src, ref_dt, frequency, cfg.hz,
                                                                                    sim_dt),
                         BcConfig(delta_max=cfg.delta_max, horizon=cfg.horizon, env_warmup_steps=cfg.warmup_ctrl_steps,
                                  mpc_warmup_steps=warmup, saturation=float(node.saturation),
                                  clamp_ctrl=bool(node.clamp_ctrl), rescale_velocity=bool(node.rescale_velocity)))
    stats = bc_pretrain(model, (dataset.obs.to(env.device), dataset.action.to(env.device)), int(node.n_epochs),
                        float(node.lr), delta_max=cfg.delta_max, seed=seed)
    stats.update(clipped_fraction=dataset.clipped_fraction, n_pairs=float(dataset.obs.shape[0]),
                 velocity_scale=dataset.velocity_scale)
    logger.info("BC warm start from %s: %d pairs, %.2f%% clipped, MSE %.4e -> %.4e (whether it shortens PPO is "
                "measured on the GPU)", arm, dataset.obs.shape[0], 100.0 * dataset.clipped_fraction,
                stats["mse_initial"], stats["mse_final"])
    return stats


def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def train_from_config(cfg: DictConfig, reference_fn: ReferenceFn | None = None, *, reader: Any = None
                      ) -> dict[str, Any]:
    """Run the training pipeline of the module docstring; returns `run_dir`, `checkpoints`, the last update's
    statistics (`last`) and the BC statistics (`bc`, None without a warm start). `reference_fn` and `reader` go to
    `load_reference_clips` (defaults: the configured DexYCB reference function and `DexYCBReader`).

    Raises:
        ValueError: On a non-positive `n_updates` or `checkpoint_every`, a DexYCB configuration outside one object and
            one hand side (`check_training_scope`, before anything is written), or an invalid sub-configuration.
    """
    p = cfg.policy
    n_updates, every, seed = int(p.n_updates), int(p.checkpoint_every), int(p.seed)
    if n_updates < 1 or every < 1:
        raise ValueError(f"need n_updates >= 1 and checkpoint_every >= 1, got {n_updates}, {every}")
    check_training_scope(cfg)
    run_dir = Path(str(cfg.paths.outputs)) / "policy" / str(p.run)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))
    torch.manual_seed(seed)
    ppo_cfg = ppo_config(p.ppo)
    env_cfg = env_config(p.env, ppo_cfg.gamma)
    curriculum = CurriculumConfig(**config_dict(p.curriculum))
    clips = load_reference_clips(cfg, "train", reference_fn, reader)
    refs, scene_xml = [clip.ref for clip in clips], training_scene_xml(cfg, clips)
    env = make_env(env_cfg, scene_xml, str(p.device))
    model = ActorCritic(env.obs_dim, env.priv_dim, env.act_dim, hidden=ppo_cfg.hidden,
                        init_log_std=float(p.init_log_std)).to(env.device)
    bc_stats = bc_warm_start(model, env, p.bc, seed) if bool(p.bc.enabled) else None
    optimizer = torch.optim.Adam(model.parameters(), lr=ppo_cfg.lr)
    generator = torch.Generator(device=env.device).manual_seed(seed)
    meta: dict[str, Any] = {"run": str(p.run), "device": str(p.device), "env": asdict(env_cfg), "ppo": asdict(ppo_cfg),
                            "curriculum": asdict(curriculum), "bc": bc_stats, "seed": seed, "scene_xml": scene_xml,
                            "n_references": len(refs)}
    checkpoints: list[str] = []
    last: dict[str, Any] = {}
    update, start = 0, time.perf_counter()
    with (run_dir / "metrics.jsonl").open("a") as metrics:
        while update < n_updates:
            strength, seconds = curriculum_stage(update, curriculum)
            env.set_references([curriculum_reference(ref, strength, seconds) for ref in refs])
            segment = min(curriculum.stage_updates - update % curriculum.stage_updates, every - update % every,
                          n_updates - update)

            def record(stats: dict[str, float], offset: int = update, strength: float = strength,
                       seconds: float = seconds) -> None:
                row = {k: _finite_or_none(float(v)) for k, v in stats.items()}
                row.update(update=offset + int(stats["update"]) + 1, strength=strength,
                           clip_seconds=_finite_or_none(seconds))
                metrics.write(json.dumps(row) + "\n")
                last.clear()
                last.update(row)

            train(env, model, ppo_cfg, segment, logger_fn=record, generator=generator, optimizer=optimizer)
            update += segment
            metrics.flush()
            if update % every == 0 or update == n_updates:
                path = save_checkpoint(run_dir / "checkpoints" / checkpoint_name(update), model, optimizer,
                                       {**meta, "update": update, "env_steps": update * ppo_cfg.n_steps * env.n_worlds,
                                        "wall_seconds": time.perf_counter() - start})
                checkpoints.append(str(path))
    logger.info("trained %s for %d updates in %.1f s: %d checkpoints under %s", p.run, n_updates,
                time.perf_counter() - start, len(checkpoints), run_dir)
    return {"run_dir": str(run_dir), "checkpoints": checkpoints, "last": last, "bc": bc_stats}


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    if "policy" not in cfg:
        raise ValueError("compose the policy group: train_policy.py +policy=default")
    apply_cpu_budget()
    if str(cfg.policy.device).startswith("cuda"):
        apply_gpu_budget()
    summary = train_from_config(cfg)
    logger.info("last update: %s", {k: summary["last"].get(k) for k in ("update", "policy_loss", "value_loss")})


if __name__ == "__main__":
    main()
