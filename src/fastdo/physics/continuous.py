# Copyright 2026 The HandReel authors. SPDX-License-Identifier: Apache-2.0
# Portions adapted from DO AS I DO (https://github.com/malik-group/do-as-i-do, commit 824591b),
# Copyright (c) 2026 malik-group, MIT License; see licenses/DO-AS-I-DO-MIT.txt and THIRD_PARTY_LICENSES.md.
"""Continuous batching of DO AS I DO's physics MPC across clips on a slot-union batch.

`slot_run` is `clip.run` (the sync-free executor; DO AS I DO's `optimize_physics.main` control loop and epilogue) for a
clip that lives in one slot of a union batch: the MPC is `union_slot.optimize`, and each executed substep writes the
slot's controls, yields `STEP` and post-processes on the device (penetration per world, world-0 state kept on the device
and read back once per control step instead of once per substep: the values are the same, only the read is deferred).

`run_continuous` keeps S slots busy: every tick it resumes each resident clip until it yields `STEP` (host work and
blocking reads happen inside the resume), refills a finished clip's slot with the next queued clip (the batch is
rebuilt for the new resident set BETWEEN the resumes and the launch, so no clip loses its step's contacts), and
launches ONE union step. Slots advance independently (a clip's iteration count never waits for another's), so the
lock-step idle time of `lockstep.run_lockstep` disappears; empty slots step parked hands.

The batch is duck-typed (`UnionBatchLike`): `views(slot)`, `step()`, `assign(clips)` (rebuild for a list of
resident clips, None = empty), `adopt(slot, clip_run)` (copy a freshly set-up clip's state and rows into the slot).
Torch only: the CPU tests drive it with a pure-torch fake batch; `union_batch.UnionBatch` is the mujoco_warp one.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Generator, Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import torch

from fastdo.physics import union_slot as U
from fastdo.physics.sync_free import penetration

__all__ = ["Resident", "UnionBatchLike", "run_continuous", "slot_run"]

log = logging.getLogger(__name__)


class UnionBatchLike(Protocol):
    def views(self, slot: int) -> U.SlotViews: ...
    def step(self) -> None: ...
    def assign(self, clips: list[Any | None]) -> None: ...
    def adopt(self, slot: int, clip: Any) -> None: ...


def _terminate0(config: Any, v: U.SlotViews, rstat: Any, substep_max_pen: torch.Tensor) -> torch.Tensor:
    """`mjwp.get_terminate` (single hand) combined as `clip.run` combines it; per world (only [0] is read)."""
    qpos_sim = v.qpos
    obj_z = qpos_sim[:, -4] if config.nq_obj == 6 else qpos_sim[:, -5]
    terminate = obj_z < config.terminate_z_threshold
    pen, wid = penetration(rstat, v.cv, v.n)  # type: ignore[arg-type]  # ContactView's interface
    max_pen = torch.zeros(v.n, device=config.device)
    max_pen.scatter_reduce_(0, wid, pen, reduce="amax")
    terminate = terminate | (max_pen > config.terminate_penetration_threshold)
    terminate = terminate | (substep_max_pen > config.terminate_penetration_threshold)
    live_qpos, live_qvel = v.qpos, v.qvel
    return terminate | ((~torch.isfinite(live_qpos)).any(dim=-1) | (~torch.isfinite(live_qvel)).any(dim=-1))


def slot_run(cr: Any, c: U.SlotCore) -> Generator[Any, None, dict | None]:
    """`clip.run` on a slot (module docstring). `cr` is the clip's `clip.ClipRun` (reference, host MjData)."""
    import mujoco
    from retargeting.pipeline import optimize_physics as OP
    from retargeting.utils.interp import get_slice
    from retargeting.utils.tracking_error import compute_object_tracking_error

    config = c.config
    mj_model, mj_data, mj_data_ref = cr.mj_model, cr.mj_data, cr.mj_data_ref
    ref_data, qpos_ref, ctrl_ref = cr.ref_data, cr.qpos_ref, cr.ctrl_ref
    warmup_enabled = cr.warmup_enabled
    t0 = time.perf_counter()
    base_noise_scale = config.noise_scale.clone()
    ctrls = ctrl_ref[: config.horizon_steps]
    info_list = []
    with cr.run_viewer() as viewer:
        while viewer.is_running():
            sim_step = int(np.round(mj_data.time / config.sim_dt))
            if warmup_enabled:
                config.warmup_sim_step = sim_step
            ref_slice = get_slice(ref_data, sim_step + 1, sim_step + config.horizon_steps + 1)
            config.noise_scale = OP._apply_noise_mask(base_noise_scale, [])
            ctrls, infos = yield from U.optimize(c, ctrls, ref_slice)
            config.noise_scale = base_noise_scale
            if len(config.trace_site_ids) > 0:
                q_h = ref_slice[0].detach().cpu().numpy()
                trace_ref = []
                for h in range(config.horizon_steps):
                    mj_data_ref.qpos[:] = q_h[h]
                    mujoco.mj_kinematics(mj_model, mj_data_ref)
                    trace_ref.append(np.array([mj_data_ref.site_xpos[sid] for sid in config.trace_site_ids]))
                infos["trace_ref"] = np.stack(trace_ref, axis=0)[None, None, :, :, :]
            dev_q, dev_v = [], []
            _substep_max_pen = torch.zeros(c.views.n, device=config.device)
            for i in range(config.ctrl_steps):
                ctrl_step = ctrls[i]
                abs_step = sim_step + i
                if warmup_enabled and abs_step == config.warmup_steps:
                    U.set_weld_active(config, c.views, False)
                    U.set_gravity(c.views, list(config.default_gravity))
                    c.log.info(f"Warmup ended at step {abs_step}; weld disabled, gravity restored.")
                c.views.ctrl[:] = ctrl_step.to(torch.float32)          # mjwp.step_env(..., perturb=False)
                yield U.STEP
                v = c.views
                _pen_i, _wid_i = penetration(c.rstat, v.cv, v.n)  # type: ignore[arg-type]
                _pen_per_world = torch.zeros(v.n, device=config.device)
                _pen_per_world.scatter_reduce_(0, _wid_i, _pen_i, reduce="amax")
                _substep_max_pen = torch.maximum(_substep_max_pen, _pen_per_world)
                dev_q.append(v.qpos[0].clone())
                dev_v.append(v.qvel[0].clone())
            v = c.views
            terminate = _terminate0(config, v, c.rstat, _substep_max_pen)   # contacts of the LAST substep
            U.sync_env(v)
            host_q = torch.stack(dev_q).cpu().numpy()
            host_v = torch.stack(dev_v).cpu().numpy()
            step_info: dict[str, list] = {"qpos": [], "qvel": [], "time": [], "ctrl": []}
            for i in range(config.ctrl_steps):
                mj_data.qpos[:] = host_q[i]
                mj_data.qvel[:] = host_v[i]
                mj_data.ctrl[:] = ctrls[i].detach().cpu().numpy()
                mj_data.time += config.sim_dt
                step_info["qpos"].append(mj_data.qpos.copy())
                step_info["qvel"].append(mj_data.qvel.copy())
                step_info["time"].append(mj_data.time)
                step_info["ctrl"].append(mj_data.ctrl.copy())
            infos.update({k: np.stack(x, axis=0) for k, x in step_info.items()})
            if warmup_enabled:
                infos["warmup_progress"] = min(sim_step / max(config.warmup_steps, 1), 1.0)
            sim_step = int(np.round(mj_data.time / config.sim_dt))
            prev_ctrl = ctrls[config.ctrl_steps:]
            new_ctrl = ctrl_ref[sim_step + prev_ctrl.shape[0]: sim_step + prev_ctrl.shape[0] + config.ctrl_steps]
            if config.blend_reference and new_ctrl.shape[0] > 0 and prev_ctrl.shape[0] > 0:
                delta = prev_ctrl[-1] - new_ctrl[0]
                new_ctrl = new_ctrl.clone()
                n = new_ctrl.shape[0]
                decay = torch.linspace(1.0, 0.0, n, device=new_ctrl.device).unsqueeze(1)
                new_ctrl += delta * decay
            ctrls = torch.cat([prev_ctrl, new_ctrl], dim=0)
            mj_data.qpos[:] = v.qpos[0].detach().cpu().numpy()
            mj_data.qvel[:] = v.qvel[0].detach().cpu().numpy()
            mj_data_ref.qpos[:] = qpos_ref[sim_step].detach().cpu().numpy()
            infos["sim_step"] = sim_step
            info_list.append({k: x for k, x in infos.items() if k != "trace_sample"})
            if bool(terminate[0]):
                c.log.info(f"Early termination at step {sim_step}/{config.max_sim_steps}: terminate condition "
                           "triggered")
                break
            if sim_step >= config.max_sim_steps:
                break
    if config.save_info and len(info_list) > 0:
        info_aggregated = {k: np.stack([info[k] for info in info_list], axis=0) for k in info_list[0]}
        np.savez(f"{config.output_dir}/trajectory_mjwp.npz", **info_aggregated)  # type: ignore[arg-type]
    errors = None
    if info_list:
        qpos_traj = np.concatenate([info["qpos"] for info in info_list], axis=0)
        qpos_ref_np = qpos_ref[: qpos_traj.shape[0]].detach().cpu().numpy()
        errors = compute_object_tracking_error(qpos_traj, qpos_ref_np, config.embodiment_type, "mjwp")
        c.log.info(f"Final object tracking error: pos={errors['obj_pos_err']:.4f}, "
                   f"quat={errors['obj_quat_err']:.4f}")
    cr.timings["loop_s"] = time.perf_counter() - t0
    cr.timings.update({k: float(x) for k, x in c.stats.items()})
    return errors


def _empty_gen() -> Generator[Any, None, Any]:
    return
    yield  # pragma: no cover  -- makes this a generator


@dataclass
class Resident:
    """A clip occupying a slot: its name, generator and outcome."""

    name: str
    gen: Generator[Any, None, Any]
    clip: Any = None                   # whatever `assign` needs to know about the clip (its scene, rows)
    result: Any = None
    error: BaseException | None = None
    ticks: int = 0
    started_tick: int = 0
    finished_tick: int = -1


@dataclass
class RunStats:
    ticks: int = 0
    rebuilds: int = 0
    shrinks: int = 0
    slot_ticks_busy: int = 0
    slot_ticks_total: int = 0
    seconds: float = 0.0
    finished: list[Resident] = field(default_factory=list)


def run_continuous(batch: UnionBatchLike, n_slots: int, queue: Iterable[Any],
                   start: Callable[[int, Any], tuple[Resident, Any]],
                   on_finish: Callable[[Resident], None] | None = None, shrink: bool = True) -> RunStats:
    """Run every clip of `queue` to completion on `n_slots` slots of `batch` (module docstring).

    `start(slot, item) -> (Resident, clip_run)` sets the clip up (its own RNG slot captured) and returns its resident
    and the object `batch.adopt` copies from; the resident's generator must only touch the slot through views it
    fetches from `batch` on every access. A clip that raises is finished with `error` set (the others continue).
    `shrink`: once the queue is drained, rebuild the batch with only the occupied slots after each finish (an empty
    slot's worlds would otherwise still cost a step each tick)."""
    t0 = time.perf_counter()
    pending = iter(queue)
    slots: list[Resident | None] = [None] * n_slots
    st = RunStats()
    exhausted = False

    def refill() -> list[tuple[int, Any]]:
        nonlocal exhausted
        new: list[tuple[int, Any]] = []
        for s in range(n_slots):
            if slots[s] is None and not exhausted:
                item = next(pending, None)
                if item is None:
                    exhausted = True
                    break
                try:
                    res, clip_run = start(s, item)
                except Exception as exc:
                    log.error("clip setup failed: %r", item, exc_info=exc)
                    failed = Resident(name=str(item[0] if isinstance(item, tuple) else item),
                                      gen=_empty_gen(), error=exc, finished_tick=st.ticks)
                    st.finished.append(failed)
                    if on_finish is not None:
                        on_finish(failed)
                    continue
                res.started_tick = st.ticks
                slots[s] = res
                new.append((s, clip_run))
        return new

    def advance(s: int) -> None:
        res = slots[s]
        assert res is not None
        try:
            out = next(res.gen)
            if out != U.STEP:
                raise RuntimeError(f"slot {s} yielded {out!r}; the union path yields STEP only")
            res.ticks += 1
        except StopIteration as stop:
            res.result, res.finished_tick = stop.value, st.ticks
            _finish(s)
        except Exception as exc:
            log.error("clip %s failed", res.name, exc_info=exc)
            res.error, res.finished_tick = exc, st.ticks
            _finish(s)

    def _finish(s: int) -> None:
        res = slots[s]
        assert res is not None
        slots[s] = None
        st.finished.append(res)
        if on_finish is not None:
            on_finish(res)

    def admit() -> None:
        while True:
            new = refill()
            if not new:
                return
            batch.assign([r.clip if r is not None else None for r in slots])
            st.rebuilds += 1
            for s, clip_run in new:
                batch.adopt(s, clip_run)
            for s, _ in new:
                advance(s)

    admit()
    while any(r is not None for r in slots):
        batch.step()
        st.ticks += 1
        st.slot_ticks_total += n_slots
        st.slot_ticks_busy += sum(r is not None for r in slots)
        for s in range(n_slots):
            if slots[s] is not None:
                advance(s)
        if any(r is None for r in slots) and not exhausted:
            admit()
        if shrink and exhausted and any(r is None for r in slots) and any(r is not None for r in slots):
            slots = [r for r in slots if r is not None]          # the queue is drained: stop stepping empty slots
            n_slots = len(slots)
            batch.assign([r.clip for r in slots if r is not None])
            st.rebuilds += 1
            st.shrinks += 1
    st.seconds = time.perf_counter() - t0
    return st
