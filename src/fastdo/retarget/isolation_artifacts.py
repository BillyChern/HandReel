"""Physics isolation study, report stage: load one arm's physics-run artifacts from its processed-data directory.

`run/pipeline/_baseline_physics_driver.py` writes `arm_meta.json` and `isolation_errors.npz` after the
baseline's `optimize_physics.py` returns, and `optimize_physics.py` writes `trajectory_mjwp.npz` (its
`config.yaml` is saved before the MPC loop starts). A run killed from outside, such as by the shared-host
GPU guard, leaves none of the three: `missing_arm_artifacts` names what is absent so the report can write
a no-output row (`isolation_report.no_output_row`) instead of aborting.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf

from fastdo.retarget.isolation_report import (
    ArmTimeBase,
    flatten_executed_qpos,
    resampled_clip_steps,
)

__all__ = [
    "ARM_ARTIFACT_FILES",
    "RUN_SETTING_KEYS",
    "ArmArtifacts",
    "arm_time_base",
    "load_arm_artifacts",
    "missing_arm_artifacts",
]

ARM_ARTIFACT_FILES = ("arm_meta.json", "isolation_errors.npz", "trajectory_mjwp.npz")
RUN_SETTING_KEYS = ("max_sim_steps", "ctrl_steps", "max_num_iterations", "improvement_threshold",
                    "improvement_check_steps", "ref_dt", "sim_dt")


@dataclass(frozen=True, eq=False)
class ArmArtifacts:
    """One arm's saved driver/baseline outputs, loaded once for the report."""

    meta: dict[str, Any]
    settings: dict[str, Any]
    raw_reference_frames: int
    baseline_log_errors: tuple[float, float]  # whole trace, warmup included (the driver log line)
    pos_frames: np.ndarray
    quat_frames: np.ndarray
    qpos_exec: np.ndarray
    opt_steps: np.ndarray
    improvement: np.ndarray
    rew_max: np.ndarray


def missing_arm_artifacts(data_dir: Path) -> list[str]:
    """The `ARM_ARTIFACT_FILES` absent from `data_dir`, in that order (empty when the run wrote them all)."""
    return [name for name in ARM_ARTIFACT_FILES if not (Path(data_dir) / name).exists()]


def load_arm_artifacts(data_dir: Path) -> ArmArtifacts:
    """Read one arm's `ARM_ARTIFACT_FILES` (all must exist) from its processed-data directory.

    Run settings come from `arm_meta.json` first; runs that predate those fields fall back to the
    baseline's own `config.yaml`, which `optimize_physics.py` saves after resolving `max_sim_steps`.

    Raises:
        TypeError: when that legacy `config.yaml` is not a mapping.
    """
    data_dir = Path(data_dir)
    meta = json.loads((data_dir / "arm_meta.json").read_text())
    settings = {k: meta[k] for k in RUN_SETTING_KEYS if meta.get(k) is not None}
    if len(settings) < len(RUN_SETTING_KEYS):
        baseline_cfg = OmegaConf.load(data_dir / "config.yaml")
        if not isinstance(baseline_cfg, DictConfig):
            raise TypeError(f"{data_dir / 'config.yaml'} must be a mapping of run settings, "
                            f"got {type(baseline_cfg).__name__}")
        settings.update({k: baseline_cfg[k] for k in RUN_SETTING_KEYS if k not in settings})
    raw_frames = meta.get("raw_reference_frames")
    if raw_frames is None:  # the exact reference file this arm's physics run read
        with np.load(data_dir / "trajectory_kinematic.npz") as ref:
            raw_frames = ref["qpos"].shape[0]
    with np.load(data_dir / "isolation_errors.npz") as errs:
        pos_frames, quat_frames = np.asarray(errs["pos_err_frames"]), np.asarray(errs["quat_err_frames"])
        log_errors = (float(errs["obj_pos_err"]), float(errs["obj_quat_err"]))
    with np.load(data_dir / "trajectory_mjwp.npz") as mjwp:
        return ArmArtifacts(
            meta=meta, settings=settings, raw_reference_frames=int(raw_frames),
            baseline_log_errors=log_errors, pos_frames=pos_frames, quat_frames=quat_frames,
            qpos_exec=flatten_executed_qpos(mjwp["qpos"]), opt_steps=np.asarray(mjwp["opt_steps"]),
            improvement=np.asarray(mjwp["improvement"]), rew_max=np.asarray(mjwp["rew_max"]))


def arm_time_base(a: ArmArtifacts) -> ArmTimeBase:
    """The arm's step accounting from its artifacts (executed steps = length of the per-step error series)."""
    return ArmTimeBase(
        max_sim_steps=int(a.settings["max_sim_steps"]), warmup_steps=int(a.meta["warmup_steps"]),
        raw_reference_frames=a.raw_reference_frames, executed_steps=int(a.pos_frames.shape[0]),
        chunk_steps=int(a.settings["ctrl_steps"]),
        clip_steps=resampled_clip_steps(a.raw_reference_frames, float(a.settings["ref_dt"]),
                                        float(a.settings["sim_dt"])))
