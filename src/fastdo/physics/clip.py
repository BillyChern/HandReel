# Copyright 2026 The HandReel authors. SPDX-License-Identifier: Apache-2.0
# Portions adapted from DO AS I DO (https://github.com/malik-group/do-as-i-do, commit 824591b),
# Copyright (c) 2026 malik-group, MIT License; see licenses/DO-AS-I-DO-MIT.txt and THIRD_PARTY_LICENSES.md.
"""DO AS I DO's `optimize_physics.main` as a resumable generator, plus single- and multi-clip runners.

`setup` is `main` up to its control loop and `run` is the loop, line for line, except:
- `optimize` is the sync-free `executor.optimize` (a generator: the clip yields while its GPU work runs);
- each executed substep yields after its launch and reads the state back only once the device is done;
- the viewer-only per-substep `get_reward` + `log_reward_step` block (runs whenever `config.viewer` contains
  "viser", even headless, and changes no saved output) is skipped;
- the reference keypoint trace `trace_ref` is computed from ONE device->host copy of the horizon instead of 600.
Supported: the registered do_as_i_do path (no contact guidance, no Gibbs sampling); anything else raises.

Warm-start fallback, opt-in (`fastdo.amortised.fallback`): `setup(..., warm_start=(knots, U))` makes
the control PLAN ctrl_ref + the knot residual linearly interpolated onto the executor rows (from 0 at row 0;
`fastdo.amortised.knots.residual_rows`): the first rollout optimises it and the receding horizon appends its rows;
the reference the reward reads (`ref_data`) is untouched. A zero residual reproduces the unflagged run.
`iteration_cap` lowers `max_num_iterations` (per control step); `run(cr, hold_cap_s=...)` stops at the first
control-step boundary past the hold cap (`timings["capped"]`). Without these arguments nothing changes.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fastdo.physics import executor as X
from fastdo.physics.lockstep import Lane, RngSlot, drive_single, run_lockstep, wait_event

__all__ = ["ClipRun", "check_iteration_cap", "run", "run_many", "run_single", "setup"]

log = logging.getLogger(__name__)


@dataclass
class ClipRun:
    core: X.ClipCore
    ref_data: tuple
    qpos_ref: torch.Tensor
    ctrl_ref: torch.Tensor
    mj_model: Any
    mj_data: Any
    mj_data_ref: Any
    warmup_enabled: bool
    run_viewer: Any
    timings: dict[str, float] = field(default_factory=dict)
    ctrl_plan: Any = None          # the warm-started control plan (None = ctrl_ref)


def check_iteration_cap(cap: Any) -> int:
    if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
        raise ValueError(f"iteration cap must be a positive int, got {cap!r}")
    return cap


def setup(config: Any, rslot: RngSlot, decide: str = "emulate", reward_stride: int = 1,
          warm_start: tuple[Any, Any] | None = None, iteration_cap: int | None = None) -> ClipRun | None:
    """`optimize_physics.main` up to `with run_viewer() as viewer:`; None if its output exists (main's skip).
    `reward_stride` > 1 is the opt-in RS lever (`executor` module docstring); 1 = DO AS I DO's reward.
    `warm_start` / `iteration_cap`: the warm-start fallback (module docstring)."""
    X.check_reward_stride(reward_stride)
    if iteration_cap is not None:
        check_iteration_cap(iteration_cap)
    import loguru
    import mujoco
    import warp as wp
    from retargeting.config import process_config
    from retargeting.pipeline import optimize_physics as OP
    from retargeting.utils import mjwp as M
    from retargeting.utils.io import load_data, warmup_ref_interp
    from retargeting.utils.viewer import setup_renderer, setup_viewer

    t0 = time.perf_counter()
    config = process_config(config)
    if iteration_cap is not None:
        config.max_num_iterations = min(int(config.max_num_iterations), int(iteration_cap))
    if config.contact_guidance or config.gibbs_sampling:
        raise NotImplementedError("fast executor: contact_guidance / gibbs_sampling are not the registered path")
    if config.use_torch_compile:
        # compiled sampling draws through inductor's RNG: generator consumption would differ from the registered
        # (eager) path that the exactness gate measured, so refuse rather than silently diverge
        raise NotImplementedError("fast executor: use_torch_compile=True is not the registered path")
    out_path = Path(config.output_dir) / "trajectory_mjwp.npz"
    if not config.force and out_path.exists():
        loguru.logger.info(f"Skipping optimize_physics.py (output exists: {out_path})")
        return None
    qpos_ref, qvel_ref, ctrl_ref, contact, contact_pos = load_data(config, config.data_path)
    ref_data = (qpos_ref, qvel_ref, ctrl_ref, contact, contact_pos)
    config.max_sim_steps = (config.max_sim_steps if config.max_sim_steps > 0
                            else qpos_ref.shape[0] - config.horizon_steps - config.ctrl_steps)
    _warmup_init_enabled = False
    finger_qpos_indices: list[int] = []
    robot_nq = qpos_ref.shape[1] - config.nq_obj
    if config.warmup_steps > 0 and config.embodiment_type in ["bimanual", "right", "left"]:
        _warmup_init_enabled = True
        wrist_dof = 6
        if config.embodiment_type == "bimanual":
            half = robot_nq // 2
            finger_qpos_indices += list(range(wrist_dof, half))
            finger_qpos_indices += list(range(half + wrist_dof, robot_nq))
        else:
            finger_qpos_indices += list(range(wrist_dof, robot_nq))
        init_qpos = qpos_ref[0].clone()
        init_ctrl = ctrl_ref[0].clone()
        init_qpos[finger_qpos_indices] = 0.0
        init_ctrl[finger_qpos_indices] = 0.0
        init_ref_data = (torch.cat([init_qpos.unsqueeze(0), qpos_ref[1:]], dim=0), qvel_ref,
                         torch.cat([init_ctrl.unsqueeze(0), ctrl_ref[1:]], dim=0), contact, contact_pos)
        loguru.logger.info("Warmup: set initial hand to open ({} finger joints zeroed).", len(finger_qpos_indices))
    else:
        init_ref_data = ref_data
    env = M.setup_env(config, init_ref_data)
    M.precompute_hand_object_geom_mask(config, env.model_cpu)
    if _warmup_init_enabled:
        M.warmup_analytical_init(config, env, qpos_ref[0], ctrl_ref[0], finger_qpos_indices)
    base_interp_n = (min(config.warmup_ref_base_interp_steps, config.warmup_steps)
                     if config.warmup_ref_base_interp_steps > 0 else 0)
    finger_interp_n = (min(config.warmup_ref_finger_interp_steps, config.warmup_steps)
                       if config.warmup_ref_finger_interp_steps > 0 else 0)
    if (base_interp_n > 0 or finger_interp_n > 0) and config.warmup_steps > 0:
        max_interp_n = max(base_interp_n, finger_interp_n)
        selected_qpos = wp.to_torch(env.data_wp.qpos)[0].detach().cpu().numpy()
        selected_ctrl = wp.to_torch(env.data_wp.ctrl)[0].detach().cpu().numpy()
        interped_qpos = qpos_ref[:max_interp_n].detach().cpu().numpy().copy()
        interped_ctrl = ctrl_ref[:max_interp_n].detach().cpu().numpy().copy()
        finger_set = set(finger_qpos_indices)
        base_qpos_idx = np.array([i for i in range(robot_nq) if i not in finger_set])
        base_ctrl_idx = np.array([i for i in range(robot_nq) if i not in finger_set and i < selected_ctrl.shape[0]])
        finger_idx = np.array(finger_qpos_indices) if finger_qpos_indices else np.array([], dtype=int)
        if base_interp_n > 0 and len(base_qpos_idx) > 0:
            base_qpos = warmup_ref_interp(selected_qpos, qpos_ref[base_interp_n].detach().cpu().numpy(),
                                          base_interp_n)
            base_ctrl = warmup_ref_interp(selected_ctrl, ctrl_ref[base_interp_n].detach().cpu().numpy(),
                                          base_interp_n)
            interped_qpos[:base_interp_n, base_qpos_idx] = base_qpos[:, base_qpos_idx]
            interped_ctrl[:base_interp_n, base_ctrl_idx] = base_ctrl[:, base_ctrl_idx]
        if finger_interp_n > 0 and len(finger_idx) > 0:
            fing_qpos = warmup_ref_interp(selected_qpos, qpos_ref[finger_interp_n].detach().cpu().numpy(),
                                          finger_interp_n)
            fing_ctrl = warmup_ref_interp(selected_ctrl, ctrl_ref[finger_interp_n].detach().cpu().numpy(),
                                          finger_interp_n)
            interped_qpos[:finger_interp_n, finger_idx] = fing_qpos[:, finger_idx]
            interped_ctrl[:finger_interp_n, finger_idx] = fing_ctrl[:, finger_idx]
        qpos_ref[:max_interp_n] = torch.from_numpy(interped_qpos).to(qpos_ref)
        ctrl_ref[:max_interp_n] = torch.from_numpy(interped_ctrl).to(ctrl_ref)
    ctrl_plan = ctrl_ref
    if warm_start is not None:
        from fastdo.amortised.knots import residual_rows

        knots, U = (np.asarray(x) for x in warm_start)
        if U.ndim != 2 or U.shape[1] > ctrl_ref.shape[1]:
            raise ValueError(f"warm start residual {U.shape} does not fit ctrl_ref {tuple(ctrl_ref.shape)}")
        res = torch.from_numpy(residual_rows(knots, U, ctrl_ref.shape[0])).to(ctrl_ref)
        ctrl_plan = ctrl_ref.clone()
        ctrl_plan[:, :U.shape[1]] += res
    mj_model = M.setup_mj_model(config)
    mj_data = mujoco.MjData(mj_model)
    mj_data_ref = mujoco.MjData(mj_model)
    if _warmup_init_enabled:
        mj_data.qpos[:] = wp.to_torch(env.data_wp.qpos)[0].detach().cpu().numpy()
        mj_data.ctrl[:] = wp.to_torch(env.data_wp.ctrl)[0].detach().cpu().numpy()
    else:
        mj_data.qpos[:] = init_ref_data[0][0].detach().cpu().numpy()
        mj_data.ctrl[:] = init_ref_data[2][0].detach().cpu().numpy()
    mj_data.qvel[:] = init_ref_data[1][0].detach().cpu().numpy()
    mujoco.mj_forward(mj_model, mj_data)
    mj_data.time = 0.0
    mj_data_ref.qpos[:] = qpos_ref[0].detach().cpu().numpy()
    mujoco.mj_forward(mj_model, mj_data_ref)
    warmup_enabled = config.warmup_steps > 0 and len(config.warmup_weld_eq_ids) > 0
    if config.warmup_steps > 0 and not warmup_enabled:
        loguru.logger.warning("warmup_steps > 0 but no weld constraints found; warmup disabled.")
    OP._assert_object_actuator_gains_zero(env, config, "start")
    if warmup_enabled:
        M.set_weld_target(config, env)
        M.set_weld_active(config, env, True)
        M.set_gravity(env, [0.0, 0.0, 0.0])
    object_trace_site_ids, robot_trace_site_ids = [], []
    for sid in range(mj_model.nsite):
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_SITE, sid)
        if name is not None and name.startswith("trace"):
            (object_trace_site_ids if "object" in name else robot_trace_site_ids).append(sid)
    config.trace_site_ids = object_trace_site_ids + robot_trace_site_ids
    config.num_object_trace_sites = len(object_trace_site_ids)
    if config.num_dr <= 1:
        xy_offset_list, pair_margin_list = [0.0], [0.0]
    else:
        xy_offset_list = np.linspace(config.xy_offset_range[0], config.xy_offset_range[1], config.num_dr)
        pair_margin_list = np.linspace(config.pair_margin_range[0], config.pair_margin_range[1], config.num_dr)
    config.env_params_list = [[{"xy_offset": xy_offset_list[j], "pair_margin": pair_margin_list[j]}
                               for j in range(config.num_dr)] for _ in range(config.max_num_iterations)]
    OP._save_config_yaml(config)
    run_viewer = setup_viewer(config, mj_model, mj_data)
    setup_renderer(config, mj_model)
    core = X.build_core(config, env, rslot, decide=decide, reward_stride=reward_stride)
    return ClipRun(core=core, ref_data=ref_data, qpos_ref=qpos_ref, ctrl_ref=ctrl_ref, mj_model=mj_model,
                   mj_data=mj_data, mj_data_ref=mj_data_ref, warmup_enabled=warmup_enabled, run_viewer=run_viewer,
                   timings={"setup_s": time.perf_counter() - t0}, ctrl_plan=ctrl_plan)


def run(cr: ClipRun, hold_cap_s: float | None = None) -> Generator[Any, None, dict | None]:
    """The control loop of `optimize_physics.main` and its epilogue (npz + tracking error); returns `errors`.
    `hold_cap_s`: the warm-start fallback's hold cap (module docstring)."""
    import loguru
    import mujoco
    from retargeting.pipeline import optimize_physics as OP
    from retargeting.utils.interp import get_slice
    from retargeting.utils.tracking_error import compute_object_tracking_error

    c = cr.core
    config, env, M = c.config, c.env, c.M
    mj_model, mj_data, mj_data_ref = cr.mj_model, cr.mj_data, cr.mj_data_ref
    ref_data, qpos_ref, ctrl_ref = cr.ref_data, cr.qpos_ref, cr.ctrl_ref
    ctrl_plan = cr.ctrl_plan if cr.ctrl_plan is not None else ctrl_ref
    warmup_enabled = cr.warmup_enabled
    t0 = time.perf_counter()
    capped = False
    base_noise_scale = config.noise_scale.clone()
    ctrls = ctrl_plan[: config.horizon_steps]
    info_list = []
    with cr.run_viewer() as viewer:
        while viewer.is_running():
            sim_step = int(np.round(mj_data.time / config.sim_dt))
            if warmup_enabled:
                config.warmup_sim_step = sim_step
            ref_slice = get_slice(ref_data, sim_step + 1, sim_step + config.horizon_steps + 1)
            config.noise_scale = OP._apply_noise_mask(base_noise_scale, [])
            ctrls, infos = yield from X.optimize(c, ctrls, ref_slice)
            config.noise_scale = base_noise_scale
            if len(config.trace_site_ids) > 0:
                q_h = ref_slice[0].detach().cpu().numpy()
                trace_ref = []
                for h in range(config.horizon_steps):
                    mj_data_ref.qpos[:] = q_h[h]
                    mujoco.mj_kinematics(mj_model, mj_data_ref)
                    trace_ref.append(np.array([mj_data_ref.site_xpos[sid] for sid in config.trace_site_ids]))
                infos["trace_ref"] = np.stack(trace_ref, axis=0)[None, None, :, :, :]
            step_info: dict[str, list] = {"qpos": [], "qvel": [], "time": [], "ctrl": []}
            _substep_max_pen = torch.zeros(env.num_worlds, device=config.device)
            for i in range(config.ctrl_steps):
                ctrl_step = ctrls[i]
                abs_step = sim_step + i
                if warmup_enabled and abs_step == config.warmup_steps:
                    M.set_weld_active(config, env, False)
                    M.set_gravity(env, list(config.default_gravity))
                    loguru.logger.info("Warmup ended at step {}; weld disabled, gravity restored.", abs_step)
                M.step_env(config, env, ctrl_step, perturb=False)
                ev = torch.cuda.Event()
                ev.record()
                yield from wait_event(ev)
                _pen_i, _wid_i = M.check_penetration(config, env)
                _pen_per_world = torch.zeros(env.num_worlds, device=config.device)
                _pen_per_world.scatter_reduce_(0, _wid_i, _pen_i, reduce="amax")
                _substep_max_pen = torch.maximum(_substep_max_pen, _pen_per_world)
                mj_data.qpos[:] = M.get_qpos(config, env)[0].detach().cpu().numpy()
                mj_data.qvel[:] = M.get_qvel(config, env)[0].detach().cpu().numpy()
                mj_data.ctrl[:] = ctrl_step.detach().cpu().numpy()
                mj_data.time += config.sim_dt
                step_info["qpos"].append(mj_data.qpos.copy())
                step_info["qvel"].append(mj_data.qvel.copy())
                step_info["time"].append(mj_data.time)
                step_info["ctrl"].append(mj_data.ctrl.copy())
            infos.update({k: np.stack(v, axis=0) for k, v in step_info.items()})
            if warmup_enabled:
                infos["warmup_progress"] = min(sim_step / max(config.warmup_steps, 1), 1.0)
            M.sync_env(config, env, mj_data)
            sim_step = int(np.round(mj_data.time / config.sim_dt))
            prev_ctrl = ctrls[config.ctrl_steps:]
            new_ctrl = ctrl_plan[sim_step + prev_ctrl.shape[0]: sim_step + prev_ctrl.shape[0] + config.ctrl_steps]
            if config.blend_reference and new_ctrl.shape[0] > 0 and prev_ctrl.shape[0] > 0:
                delta = prev_ctrl[-1] - new_ctrl[0]
                new_ctrl = new_ctrl.clone()
                n = new_ctrl.shape[0]
                decay = torch.linspace(1.0, 0.0, n, device=new_ctrl.device).unsqueeze(1)
                new_ctrl += delta * decay
            ctrls = torch.cat([prev_ctrl, new_ctrl], dim=0)
            mj_data.qpos[:] = M.get_qpos(config, env)[0].detach().cpu().numpy()
            mj_data.qvel[:] = M.get_qvel(config, env)[0].detach().cpu().numpy()
            mj_data_ref.qpos[:] = qpos_ref[sim_step].detach().cpu().numpy()
            infos["sim_step"] = sim_step
            info_list.append({k: v for k, v in infos.items() if k != "trace_sample"})
            current_ref = tuple(r[sim_step] for r in ref_data)
            terminate, _term_info = M.get_terminate(config, env, current_ref)
            terminate = terminate | (_substep_max_pen > config.terminate_penetration_threshold)
            _live_qpos, _live_qvel = M.get_qpos(config, env), M.get_qvel(config, env)
            terminate = terminate | ((~torch.isfinite(_live_qpos)).any(dim=-1) | (~torch.isfinite(_live_qvel)).any(dim=-1))
            if terminate[0]:
                loguru.logger.info("Early termination at step {}/{}: terminate condition triggered",
                                   sim_step, config.max_sim_steps)
                break
            if sim_step >= config.max_sim_steps:
                break
            if hold_cap_s is not None and time.perf_counter() - t0 >= hold_cap_s:
                capped = True
                loguru.logger.info("Hold cap {:.1f} s reached at step {}/{}", hold_cap_s, sim_step,
                                   config.max_sim_steps)
                break
    if config.save_info and len(info_list) > 0:
        info_aggregated = {k: np.stack([info[k] for info in info_list], axis=0) for k in info_list[0]}
        np.savez(f"{config.output_dir}/trajectory_mjwp.npz", **info_aggregated)
    errors = None
    if info_list:
        qpos_traj = np.concatenate([info["qpos"] for info in info_list], axis=0)
        qpos_ref_np = qpos_ref[: qpos_traj.shape[0]].detach().cpu().numpy()
        errors = compute_object_tracking_error(qpos_traj, qpos_ref_np, config.embodiment_type, "mjwp")
        loguru.logger.info("Final object tracking error: pos={:.4f}, quat={:.4f}", errors["obj_pos_err"],
                           errors["obj_quat_err"])
    OP._assert_object_actuator_gains_zero(env, config, "end")
    cr.timings["loop_s"] = time.perf_counter() - t0
    cr.timings.update({k: float(v) for k, v in c.stats.items()})
    if hold_cap_s is not None:
        cr.timings["capped"] = capped
    return errors


def run_single(config: Any, decide: str = "emulate", reward_stride: int = 1, warm_start: tuple[Any, Any] | None = None,
               iteration_cap: int | None = None, hold_cap_s: float | None = None
               ) -> tuple[dict | None, dict[str, float]]:
    """One clip, default stream (the reference's stream arrangement); returns (errors, timings). Waits block on the
    event (`drive_single`), never spin. `reward_stride` > 1: the opt-in RS lever. `warm_start`,
    `iteration_cap`, `hold_cap_s`: the warm-start fallback (module docstring)."""
    cr = setup(config, RngSlot(active=False), decide=decide, reward_stride=reward_stride, warm_start=warm_start,
               iteration_cap=iteration_cap)
    if cr is None:
        return None, {}
    return drive_single(run(cr, hold_cap_s=hold_cap_s)), cr.timings


def run_many(configs: list[Any], decide: str = "emulate") -> list[Lane]:
    """K clips in lock-step, one CUDA stream each (`lockstep`). Returns ONE Lane per config, in order:
    lane.result = (errors, timings), or lane.error (setup or run failure of that clip only)."""
    import warp as wp

    lanes: list[Lane] = []
    runs: list[ClipRun | None] = []
    for i, cfg in enumerate(configs):
        slot = RngSlot(active=True, device=cfg.device)
        try:
            cr = setup(cfg, slot, decide=decide)
        except Exception as exc:
            log.error("clip %d setup failed", i, exc_info=exc)
            lanes.append(Lane(name=str(i), gen=iter(()), error=exc, done=True))
            runs.append(None)
            continue
        slot.capture()
        runs.append(cr)
        lanes.append(Lane(name=str(i), gen=iter(()), result=(None, {}), done=True) if cr is None else
                     Lane(name=str(i), gen=iter(())))
    torch.cuda.synchronize()
    wp.synchronize()
    default_ts = torch.cuda.current_stream()
    default_ws = wp.get_stream()

    def wrap(cr: ClipRun) -> Generator[Any, None, Any]:
        errors = yield from run(cr)
        return errors, cr.timings

    for ln, cr in zip(lanes, runs, strict=True):
        if cr is not None:
            ts = torch.cuda.Stream()
            ln.gen, ln.streams = wrap(cr), (ts, wp.stream_from_torch(ts))

    def set_streams(ln: Lane) -> None:
        torch.cuda.set_stream(ln.streams[0])
        wp.set_stream(ln.streams[1], sync=False)

    failure: BaseException | None = None
    try:
        run_lockstep(lanes, set_streams)
    except BaseException as exc:
        failure = exc
        raise
    finally:
        torch.cuda.set_stream(default_ts)
        wp.set_stream(default_ws, sync=False)
        try:
            torch.cuda.synchronize()
        except Exception:
            if failure is None:
                raise
            log.exception("synchronize after a failed lock-step run (original error re-raised)")
    return lanes
