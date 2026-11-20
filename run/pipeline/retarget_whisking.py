"""Isolation study on the baseline's physics stage, on the shipped whisking demo.

Measures what the baseline's `optimize_physics.py` (MuJoCo Warp MPC) adds on top of a kinematic
reference, using OUR contact-transport reference (`fastdo.retarget.kinematic.retarget_kinematic`)
versus THEIRS (the shipped `trajectory_kinematic.npz`).

A third arm: `isolation.our_arm` names the arm prepared from our current reference
(`ours_clearance`, the clearance-band reference; the first recorded arm is `ours`), `isolation.arms` lists the
arms `run_arm` accepts and the report compares, and `isolation.match_recorded_arm` pins a new run's env
count to a recorded arm's `arm_meta.json` so the comparison stays controlled.

`isolation.reference_rate`: `full` solves the reference on the demo's
full-rate inputs; `engine` samples the inputs at `isolation.engine_stride` (the 10 fps engine rate), upsamples them
with `fastdo.retarget.rate` (object and wrist SE(3) by C2 spline, hand vertices lerped in the wrist frame) and then
solves the reference at full rate. The robot reference itself is never upsampled.

Stages (`isolation.stage`), run one at a time via `uv run python run/pipeline/retarget_whisking.py
isolation.stage=...`:
  prepare  -- produce `{our_arm}_trajectory_kinematic.npz`, set up the scene directory of each arm
              that lacks a reference (never under the worktree or `third_party/`; recorded arms are
              left untouched), and merge kinematic-only sampled-point and primitive clearance stats
              for every arm with a reference into `prepare_stats.json`.
  dry_run  -- 90s GPU-memory probe of the baseline physics stage on one arm (a
              shared-host budget check), then kill it. Never counted as a timed arm.
  run_arm  -- the real, full physics-stage run for `isolation.arm` (~30 min; run this detached).
  report   -- log the comparison table and write `report.json` (one metric window per arm: executed
              steps in [warmup_steps, min(executed, max_sim_steps)); success also needs the whole clip; an
              arm without physics output, never run or killed, gets a failed no-output row).
  all      -- every stage above in order for every arm, in-process (blocks for hours; intended for
              unattended use, not for an interactive session with a time budget).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
import trimesh
from omegaconf import DictConfig

from fastdo.bench.gpu_budget import apply_cpu_budget, apply_gpu_budget
from fastdo.retarget.baseline_io import baseline_world_inputs
from fastdo.retarget.clearance import primitive_clearance_stats, robot_contact_frames
from fastdo.retarget.isolation import (
    gpu_dry_run_probe,
    kinematic_reference_stats,
    processed_data_dir,
    setup_arm_directory,
)
from fastdo.retarget.isolation_artifacts import (
    ArmArtifacts,
    arm_time_base,
    load_arm_artifacts,
    missing_arm_artifacts,
)
from fastdo.retarget.isolation_report import (
    SUCCESS_POS_M,
    arm_row,
    drop_onset_step,
    mpc_iteration_stats,
    no_output_row,
    post_warmup_tracking_errors,
    report_table_lines,
)
from fastdo.retarget.kinematic import RetargetConfig, retarget_kinematic
from fastdo.retarget.labels import mano_finger_labels
from fastdo.retarget.mj_model import RobotScene
from fastdo.retarget.rate import engine_key_frames, upsample_hand_track, upsample_object_poses
from fastdo.solver.sdf import SdfGrid, mesh_sdf_grid

logger = logging.getLogger(__name__)

_DRIVER_SCRIPT = Path(__file__).parent / "_baseline_physics_driver.py"
# Stages that only launch the baseline subprocess skip apply_gpu_budget's ~0.5 GB CUDA context (GPU guard).
_SUBPROCESS_ONLY_STAGES = frozenset({"dry_run", "run_arm"})


def _load_scene_and_mesh(cfg: DictConfig) -> tuple[RobotScene, SdfGrid, torch.Tensor, torch.Tensor]:
    scene = RobotScene.from_xml(cfg.isolation.demo_scene_xml)
    mesh = trimesh.load(cfg.isolation.whisk_obj, process=False)
    assert isinstance(mesh, trimesh.Trimesh), f"expected a single mesh, got {type(mesh)}"
    sdf = mesh_sdf_grid(np.asarray(mesh.vertices), np.asarray(mesh.faces), resolution=64)
    surf_pts, face_idx = trimesh.sample.sample_surface(mesh, 1024, seed=np.random.default_rng(0))
    surface = torch.as_tensor(np.asarray(surf_pts), dtype=torch.float64)
    surface_normals = torch.as_tensor(np.asarray(mesh.face_normals[face_idx]), dtype=torch.float64)
    return scene, sdf, surface, surface_normals


def _arms(cfg: DictConfig) -> tuple[str, ...]:
    return tuple(str(a) for a in cfg.isolation.arms)


def _our_trajectory_path(cfg: DictConfig) -> Path:
    return Path(cfg.isolation.output_root) / f"{cfg.isolation.our_arm}_trajectory_kinematic.npz"


def _arm_output_root(cfg: DictConfig, arm: str) -> Path:
    return Path(cfg.isolation.output_root) / arm


def _arm_data_dir(cfg: DictConfig, arm: str) -> Path:
    return processed_data_dir(_arm_output_root(cfg, arm), str(cfg.isolation.robot_type),
                              str(cfg.isolation.embodiment_type), str(cfg.isolation.task_name),
                              int(cfg.isolation.data_id))


def _stats_path(cfg: DictConfig) -> Path:
    return Path(cfg.isolation.output_root) / "prepare_stats.json"


def _stage_prepare(cfg: DictConfig) -> None:
    rate = str(cfg.isolation.get("reference_rate", "full"))
    if rate not in ("full", "engine"):
        raise ValueError(f"isolation.reference_rate must be 'full' or 'engine', got {rate!r}")
    scale, seed, our_arm = float(cfg.isolation.mesh_scale), int(cfg.isolation.seed), str(cfg.isolation.our_arm)
    scene, sdf, surface, surface_normals = _load_scene_and_mesh(cfg)

    obj_poses, wrist_poses, hand_verts, frequency = baseline_world_inputs(
        cfg.isolation.demo_keypoints_npz, cfg.isolation.demo_trajectory_kinematic_npz)
    hand_labels = mano_finger_labels(cfg.isolation.mano_right_pkl)
    rcfg = RetargetConfig(seed=seed)
    stride: int | None = None
    keys: list[int] | None = None
    upsample_s = 0.0
    if rate == "engine":
        stride = int(cfg.isolation.get("engine_stride", 3))
        n_frames = int(obj_poses.shape[0])
        key_frames = engine_key_frames(n_frames, stride)
        keys = [int(k) for k in key_frames]
        t0 = time.perf_counter()
        idx = torch.as_tensor(key_frames)
        wrist_up, verts_up = upsample_hand_track(wrist_poses[idx], hand_verts[idx], key_frames, n_frames)
        obj_poses = torch.from_numpy(upsample_object_poses(obj_poses[idx], key_frames, n_frames))
        wrist_poses, hand_verts = torch.from_numpy(wrist_up), torch.from_numpy(verts_up)
        upsample_s = time.perf_counter() - t0
        logger.info("engine-rate inputs: %d keys (stride %d) upsampled to %d frames in %.3fs", len(keys), stride,
                    n_frames, upsample_s)

    logger.info("retargeting %d frames for arm %s (reference_rate=%s)...", obj_poses.shape[0], our_arm, rate)
    t0 = time.perf_counter()
    traj = retarget_kinematic(obj_poses, wrist_poses, scale, hand_verts, hand_labels, scene, sdf,
                              surface, surface_normals, frequency, rcfg)
    solve_s = time.perf_counter() - t0
    our_npz = _our_trajectory_path(cfg)
    our_npz.parent.mkdir(parents=True, exist_ok=True)
    traj.save(our_npz)
    (our_npz.parent / f"{our_arm}_retarget_config.json").write_text(json.dumps(
        {**asdict(rcfg), "reference_rate": rate, "engine_stride": stride, "key_frames": keys,
         "solve_seconds": solve_s, "upsample_seconds": upsample_s, "wall_seconds": solve_s + upsample_s}, indent=2))
    logger.info("wrote %s (%d frames, kinematic stage %.1fs)", our_npz, traj.qpos.shape[0], solve_s + upsample_s)

    # Only arms without a reference on disk are set up; recorded arms stay byte-identical.
    sources = {"theirs": Path(cfg.isolation.demo_trajectory_kinematic_npz), our_arm: our_npz}
    for arm, src in sources.items():
        if arm != our_arm and (_arm_data_dir(cfg, arm) / "trajectory_kinematic.npz").exists():
            continue
        layout = setup_arm_directory(
            output_root=_arm_output_root(cfg, arm), demo_scene_xml=Path(cfg.isolation.demo_scene_xml),
            demo_assets_dir=Path(cfg.isolation.demo_assets_dir), trajectory_kinematic_src=src,
            task=str(cfg.isolation.task_name), robot_type=str(cfg.isolation.robot_type),
            embodiment_type=str(cfg.isolation.embodiment_type), data_id=int(cfg.isolation.data_id),
            demo_keypoints_npz=Path(cfg.isolation.demo_keypoints_npz))
        logger.info("arm %-6s: scene dir -> %s (from %s)", arm, layout.data_dir, src)

    stats: dict[str, Any] = json.loads(_stats_path(cfg).read_text()) if _stats_path(cfg).exists() else {}
    n_pen = int(cfg.isolation.n_per_link_pen)
    for arm in _arms(cfg):
        src = our_npz if arm == our_arm else _arm_data_dir(cfg, arm) / "trajectory_kinematic.npz"
        if not src.exists():
            logger.warning("arm %s has no reference at %s; keeping its previous stats", arm, src)
            continue
        with np.load(src) as data:
            qpos = np.asarray(data["qpos"])
        sampled = kinematic_reference_stats(qpos, scene, sdf, surface, scale, n_pen, seed)
        contact = robot_contact_frames(qpos, scene, sdf, surface, scale, n_pen, seed)
        primitive = primitive_clearance_stats(qpos, scene, sdf, scale, contact, rcfg.box_face_tol).as_dict()
        stats[arm] = {**sampled.as_dict(), "primitive": primitive}
        logger.info("kinematic-only [%s]: sampled %s | primitive %s", arm, sampled.as_dict(), primitive)

    _stats_path(cfg).write_text(json.dumps(stats, indent=2))
    logger.info("kinematic-only stats -> %s", _stats_path(cfg))


def _conda_run_cmd(cfg: DictConfig, *args: str) -> list[str]:
    return ["nice", "-n", "15", str(cfg.isolation.conda_bin), "run", "-n",
            str(cfg.isolation.conda_env), "--no-capture-output", "python", str(_DRIVER_SCRIPT), *args]


def _driver_args(cfg: DictConfig, arm: str, max_sim_steps: int) -> list[str]:
    """Build `_baseline_physics_driver.py`'s CLI args for one arm.

    `--output-root-dir` is the arm's *root* (what `retargeting.config.Config.output_root_dir`
    expects; the driver's own `process_config` call appends the `{robot_type}/{embodiment_type}/
    {task}/{data_id}` layout to it). `--errors-out`/`--meta-out` must land inside that same
    resolved `processed_data_dir` -- the same directory `setup_arm_directory` populated and
    `_stage_report` later reads back from -- not the arm root itself.
    """
    arm_root, data_dir = _arm_output_root(cfg, arm), _arm_data_dir(cfg, arm)
    args = [
        "--baseline-root", str(cfg.isolation.baseline_root),
        "--output-root-dir", str(arm_root),
        "--task", str(cfg.isolation.task_name),
        "--data-id", str(cfg.isolation.data_id),
        "--robot-type", str(cfg.isolation.robot_type),
        "--embodiment-type", str(cfg.isolation.embodiment_type),
        "--seed", str(cfg.isolation.seed),
        "--max-sim-steps", str(max_sim_steps),
        "--errors-out", str(data_dir / "isolation_errors.npz"),
        "--meta-out", str(data_dir / "arm_meta.json"),
    ]
    if cfg.isolation.num_samples is not None:
        args += ["--num-samples", str(cfg.isolation.num_samples)]
    if cfg.isolation.num_perturb_samples is not None:
        args += ["--num-perturb-samples", str(cfg.isolation.num_perturb_samples)]
    return args


def _stage_dry_run(cfg: DictConfig) -> None:
    arm = str(cfg.isolation.arm)
    cmd = _conda_run_cmd(cfg, *_driver_args(cfg, arm, max_sim_steps=0))
    log_path = Path(cfg.isolation.temp_dir) / f"dry_run_{arm}.log"
    logger.info("GPU dry run: %s (up to %ss, sampling every %ss) -> %s", " ".join(cmd),
               cfg.isolation.dry_run_seconds, cfg.isolation.dry_run_poll_seconds, log_path)
    result = gpu_dry_run_probe(cmd, cwd=Path(cfg.isolation.baseline_root), env=dict(os.environ),
                              seconds=float(cfg.isolation.dry_run_seconds),
                              poll_seconds=float(cfg.isolation.dry_run_poll_seconds), log_path=log_path)
    cap = float(cfg.isolation.gpu_mem_cap_mib)
    verdict = "OK (under cap)" if result.peak_mib <= cap else "OVER CAP -- reduce env count"
    logger.info("GPU dry-run peak: %.0f MiB over a %.0f MiB baseline (cap %.0f MiB) -> %s",
               result.peak_mib, result.baseline_mib, cap, verdict)
    (Path(cfg.isolation.temp_dir) / f"dry_run_{arm}_result.json").write_text(json.dumps({
        "peak_mib": result.peak_mib, "baseline_mib": result.baseline_mib, "cap_mib": cap,
        "verdict": verdict, "num_samples": cfg.isolation.num_samples,
        "num_perturb_samples": cfg.isolation.num_perturb_samples, "samples": list(result.samples),
        "confirmed_dead": result.confirmed_dead,
    }, indent=2))
    if not result.confirmed_dead:
        # Shared-host rule: never leave an unbounded physics process running unattended. Fail
        # loudly rather than silently proceeding as if the dry run were cleanly torn down.
        raise RuntimeError(
            f"dry-run process group for arm {arm!r} was NOT confirmed dead after kill -- "
            f"check for a stray optimize_physics.py process before running anything else")


def _check_matches_recorded_arm(cfg: DictConfig, arm: str) -> None:
    """Refuse a run whose env count differs from `isolation.match_recorded_arm`'s `arm_meta.json`
    (the driver records "default(...)" when it did not override the baseline's value)."""
    recorded_arm = cfg.isolation.get("match_recorded_arm")
    if recorded_arm is None or str(recorded_arm) == arm:
        return
    meta = json.loads((_arm_data_dir(cfg, str(recorded_arm)) / "arm_meta.json").read_text())
    for key in ("num_samples", "num_perturb_samples"):
        recorded = meta.get(key)
        recorded = None if isinstance(recorded, str) and recorded.startswith("default") else recorded
        if recorded != cfg.isolation[key]:
            raise ValueError(f"isolation.{key}={cfg.isolation[key]!r} differs from recorded arm "
                             f"{recorded_arm!r} ({meta.get(key)!r}); the comparison would not be controlled")


def _stage_run_arm(cfg: DictConfig) -> None:
    arm = str(cfg.isolation.arm)
    if arm not in _arms(cfg):
        raise ValueError(f"isolation.arm must be one of {_arms(cfg)}, got {arm!r}")
    _check_matches_recorded_arm(cfg, arm)
    cmd = _conda_run_cmd(cfg, *_driver_args(cfg, arm, max_sim_steps=int(cfg.isolation.max_sim_steps)))
    log_path = Path(cfg.isolation.temp_dir) / f"arm_{arm}.log"
    logger.info("running arm %s (full physics stage, ~30 min): %s -> %s", arm, " ".join(cmd), log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        # start_new_session=True (own process group), matching gpu_dry_run_probe: cmd's wrapper
        # chain (nice -> conda run -> the real python process) forks rather than exec-replacing at
        # the conda run step, so if this run is ever interrupted, a plain SIGTERM/SIGKILL on just
        # the outer PID would not reliably reach the real MPC process -- see isolation.py's
        # _kill_process_group docstring for how the dry run discovered this the hard way.
        rc = subprocess.run(cmd, cwd=str(cfg.isolation.baseline_root), env=dict(os.environ),
                            stdout=log, stderr=subprocess.STDOUT, check=False,
                            start_new_session=True).returncode
    logger.info("arm %s finished rc=%d (see %s)", arm, rc, log_path)
    if rc != 0:
        raise RuntimeError(f"physics arm {arm!r} failed with rc={rc}; see {log_path}")


def _load_arm(cfg: DictConfig, arm: str) -> ArmArtifacts:
    """One arm's physics-run artifacts (`fastdo.retarget.isolation_artifacts.load_arm_artifacts`)."""
    return load_arm_artifacts(_arm_data_dir(cfg, arm))


def _stage_report(cfg: DictConfig) -> None:
    scale = float(cfg.isolation.mesh_scale)
    scene, sdf, surface, _ = _load_scene_and_mesh(cfg)
    kinematic_stats = json.loads(_stats_path(cfg).read_text()) if _stats_path(cfg).exists() else {}
    # An arm without physics output (never run, or killed, e.g. by the shared-host GPU guard) still gets
    # a row, failed, with NaN metrics; only arms whose run wrote every artifact are loaded.
    missing = {arm: missing_arm_artifacts(_arm_data_dir(cfg, arm)) for arm in _arms(cfg)}
    for arm, files in missing.items():
        if files:
            logger.warning("arm %s has no physics output (missing %s): writing a failed no-output row",
                           arm, ", ".join(files))
    arms = {arm: _load_arm(cfg, arm) for arm in _arms(cfg) if not missing[arm]}
    time_bases = {arm: arm_time_base(a) for arm, a in arms.items()}

    # Common chunk split for the MPC-iteration comparison: the earliest step in any loaded arm's metric
    # window at which its executed object position error crosses the success threshold (else the
    # first loaded arm's warmup end).
    onsets = [s for arm, a in arms.items() if (s := drop_onset_step(
        a.pos_frames, SUCCESS_POS_M, time_bases[arm].warmup_steps, time_bases[arm].metric_end_step))
        is not None]
    split_step = min(onsets) if onsets else next((tb.warmup_steps for tb in time_bases.values()), 0)

    rows: dict[str, dict[str, Any]] = {}
    for arm, a in arms.items():
        time_base = time_bases[arm]
        warmup, end = time_base.warmup_steps, time_base.metric_end_step
        if a.qpos_exec.shape[0] != a.pos_frames.shape[0]:
            raise ValueError(f"arm {arm!r}: {a.qpos_exec.shape[0]} executed qpos rows but "
                             f"{a.pos_frames.shape[0]} per-step errors; time bases disagree")
        # Tracking error and penetration both over executed steps [warmup, min(T, max_sim_steps)).
        # An arm that ended during warmup has an empty window: NaN row, no stats on zero frames.
        tracking = post_warmup_tracking_errors(a.pos_frames, a.quat_frames, warmup, end)
        phys = (kinematic_reference_stats(a.qpos_exec[warmup:end], scene, sdf, surface, scale,
                                          int(cfg.isolation.n_per_link_pen), int(cfg.isolation.seed)).as_dict()
                if tracking.n_steps > 0 else
                {"n_frames": 0, "frac_under_2mm": float("nan"), "frac_any_penetration": float("nan")})
        mpc = mpc_iteration_stats(a.opt_steps, a.improvement, a.rew_max,
                                  int(a.settings["max_num_iterations"]),
                                  float(a.settings["improvement_threshold"]),
                                  int(a.settings["improvement_check_steps"]),
                                  split_step // time_base.chunk_steps)
        rows[arm] = arm_row(arm, a.meta["wall_seconds"], tracking, time_base, a.baseline_log_errors, phys, mpc,
                            kinematic_stats.get(arm, {}))
    ordered = [rows[arm] if arm in rows else no_output_row(arm, missing[arm], kinematic_stats.get(arm, {}))
               for arm in _arms(cfg)]
    for line in report_table_lines(ordered, kinematic_stats):
        logger.info(line)
    Path(cfg.isolation.output_root, "report.json").write_text(json.dumps(ordered, indent=2))
    logger.info("full report -> %s", Path(cfg.isolation.output_root, "report.json"))


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    from fastdo import hostpaths

    stage = str(cfg.isolation.stage)
    mano = [("isolation", "isolation.mano_right_pkl", "FASTDO_MANO_DIR", False)] if stage == "prepare" else []
    hostpaths.check_config_paths(cfg, [("isolation", *k) for k in hostpaths.ISOLATION_HOST_KEYS] + mano)
    apply_cpu_budget()
    if stage not in _SUBPROCESS_ONLY_STAGES:
        apply_gpu_budget()
    if stage == "prepare":
        _stage_prepare(cfg)
    elif stage == "dry_run":
        _stage_dry_run(cfg)
    elif stage == "run_arm":
        _stage_run_arm(cfg)
    elif stage == "report":
        _stage_report(cfg)
    elif stage == "all":
        _stage_prepare(cfg)
        _stage_dry_run(cfg)
        for arm in _arms(cfg):
            cfg.isolation.arm = arm
            _stage_run_arm(cfg)
        _stage_report(cfg)
    else:
        raise ValueError(f"unknown isolation.stage={stage!r}")


if __name__ == "__main__":
    main()
