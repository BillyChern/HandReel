"""The engine's retargeting stage on the tracking policy.

`retarget_stage_policy(clip_record, checkpoint, ...)` replaces the per-clip physics MPC in the throughput configuration
of budget Section 2 (amended): the clip's policy reference (`reference_fn`, a `ClipRecord` -> `PolicyReference` at the
policy rate, see `fastdo.policy.data`), twisted SMC with the checkpoint's policy as proposal on a
`StatefulTrackingEnv` of `SmcConfig.n_particles` worlds (the checkpoint's calibrated tempering floor applied when its
`.sigma_v.json` exists, `fastdo.policy.smc_calibration`), and the selected executed trajectory in the baseline's
`KinematicTrajectory` format. Frame j is step `W - 1 + j` of the particle path (the `EnvConfig` evaluation note).
Without a warm-up (`W = 0`) frame 0 is the actual reset state of the selected path
(`SmcResult.start_qpos`, read through the environment's `world_kinematics`) and frames 1.. are steps 0..; an environment
without `world_kinematics` is refused for `W = 0` rather than substituting the reference's frame 0. `qvel` is
`fastdo.retarget.qvel.kinematic_qvel` (central differences; the object's angular velocity through `so3_log`) and
`frequency` the reference rate. A clip no particle tracks raises `ClipRejected`: it counts against yield and is never
re-run with the MPC in this configuration.

`make_policy_stage(checkpoint, record_fn=..., reference_fn=..., env=...)` adapts it to the engine's
`StageFn = Callable[[str, Path], None]` (`fastdo.engine.orchestrator`, not imported here so the policy package stays
free of the engine's worker imports): `record_fn(clip_id, artifact_dir)` gives the clip's `ClipRecord`, and the
trajectory is written to `<artifact_dir>/retarget.npz` (`RETARGET_FILE`, the `<stage>.npz` convention of
`fastdo.engine.workers`). `ClipRejected` propagates, so `run_clip` marks the stage failed with its message. This
adapter is tested here only as a callable annotated `StageFn`, never through `fastdo.engine` `run_clip`, so it
must be validated there before relying on it.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from fastdo.io.clip_record import ClipRecord
from fastdo.policy.checkpoint import load_checkpoint
from fastdo.policy.data import ReferenceFn
from fastdo.policy.protocol import StatefulTrackingEnv
from fastdo.policy.smc import SmcConfig, run_smc
from fastdo.policy.smc_calibration import sigma_v_path, with_calibrated_floor
from fastdo.retarget.kinematic import KinematicTrajectory
from fastdo.retarget.qvel import kinematic_qvel

logger = logging.getLogger(__name__)

__all__ = ["RETARGET_FILE", "ClipRejected", "RecordFn", "make_policy_stage", "retarget_stage_policy"]

RETARGET_FILE = "retarget.npz"
RecordFn = Callable[[str, Path], ClipRecord]


class ClipRejected(RuntimeError):
    """No particle's executed trajectory passed the baseline success criterion (the clip counts against yield)."""


def retarget_stage_policy(clip_record: ClipRecord, checkpoint: str | Path, *, reference_fn: ReferenceFn,
                          env: StatefulTrackingEnv, smc_cfg: SmcConfig | None = None,
                          generator: torch.Generator | None = None) -> KinematicTrajectory:
    """The retargeting stage for one clip (module docstring). `smc_cfg` defaults to `SmcConfig` with `n_particles =
    env.n_worlds` and the checkpoint's PPO discount; its `sigma_v` is read from the checkpoint's calibration when 0.

    Raises:
        ValueError: If the environment has no warm-up (`cfg.warmup_ctrl_steps == 0`) and cannot report its reset state
            (`world_kinematics`), or the particle system returned no reset state.
        ClipRejected: If no particle tracks the clip.
    """
    warmup = int(getattr(getattr(env, "cfg", None), "warmup_ctrl_steps", 0))
    if warmup < 1 and not callable(getattr(env, "world_kinematics", None)):
        raise ValueError("an environment without warm-up (warmup_ctrl_steps 0) must report its reset state "
                         "(world_kinematics): frame 0 is that state, never the reference's frame 0")
    device = getattr(env, "device", torch.device("cpu"))
    loaded = load_checkpoint(checkpoint, device)
    gamma = float((loaded.meta.get("ppo") or {}).get("gamma", SmcConfig.gamma))
    cfg = smc_cfg if smc_cfg is not None else SmcConfig(n_particles=env.n_worlds, gamma=gamma)
    if cfg.sigma_v == 0.0 and sigma_v_path(checkpoint).exists():
        cfg = with_calibrated_floor(cfg, checkpoint)
    ref = reference_fn(clip_record)
    if ref is None:                    # e.g. a DexYCB clip whose scene or hands failed validation
        raise ClipRejected(f"{clip_record.clip_id}: the reference function declined the clip, so there is no reference")
    gen = generator if generator is not None else torch.Generator().manual_seed(0)
    result = run_smc(env, loaded.model.eval(), ref, cfg, gen)
    if not result.success:
        raise ClipRejected(f"{clip_record.clip_id}: none of {cfg.n_particles} particles tracked the reference")
    frames = ref.num_frames
    if warmup >= 1:
        q = torch.as_tensor(result.qpos[warmup - 1:warmup - 1 + frames], dtype=torch.float64)
    elif result.start_qpos.size == result.qpos.shape[1]:                  # W = 0: the reset state, then steps 0..
        q = torch.as_tensor(np.concatenate([result.start_qpos[None], result.qpos[:frames - 1]]), dtype=torch.float64)
    else:
        raise ValueError(f"{clip_record.clip_id}: W = 0 but the particle system recorded no reset state")
    qvel = kinematic_qvel(q, float(ref.hz))
    logger.info("policy retargeting stage %s: %d frames, %d of %d particles tracked it, %d resampling rounds",
                clip_record.clip_id, frames, result.n_success, cfg.n_particles, result.n_resamples)
    return KinematicTrajectory(qpos=q.numpy(), qvel=qvel.numpy(), frequency=float(ref.hz))


def make_policy_stage(checkpoint: str | Path, *, record_fn: RecordFn, reference_fn: ReferenceFn,
                      env: StatefulTrackingEnv, smc_cfg: SmcConfig | None = None, seed: int = 0
                      ) -> Callable[[str, Path], None]:
    """The engine `StageFn` of `retarget_stage_policy` (module docstring): one generator seeded with `seed` for the
    stage's lifetime, the trajectory written to `<artifact_dir>/RETARGET_FILE`."""
    generator = torch.Generator(device=getattr(env, "device", torch.device("cpu"))).manual_seed(seed)

    def stage(clip_id: str, artifact_dir: Path) -> None:
        trajectory = retarget_stage_policy(record_fn(clip_id, artifact_dir), checkpoint, reference_fn=reference_fn,
                                           env=env, smc_cfg=smc_cfg, generator=generator)
        trajectory.save(Path(artifact_dir) / RETARGET_FILE)

    return stage
