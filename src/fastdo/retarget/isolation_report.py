"""Physics isolation study, report-stage helpers: one time base for every physics-output metric, plus MPC diagnostics.

The baseline driver's per-step tracking errors (`compute_object_tracking_error` over the whole
executed `qpos` trace) include the welded, zero-gravity warmup prefix, which the reference fills
with its own frame 0, so those steps carry near-zero error. Averaging over them dilutes the mean,
more so for a run that terminates early. The MPC also executes whole control chunks, so an uncapped
run can step past its own `max_sim_steps` onto the held last reference frame. Every physics-output
metric the isolation report shows (tracking error and hand-object penetration) is therefore taken
over the same window of executed steps `[warmup_steps, min(executed_steps, max_sim_steps))`, labelled
`TIME_BASE_LABEL` ("post_warmup_to_max_sim_steps") in `report.json`. An arm that terminates during
warmup (or whose window `max_sim_steps` closes) has an empty window: its row still gets written, with
`n_steps=0`, NaN means and maxima, and `success=False`, since a failed arm is when the report matters.

An arm succeeds (`arm_success`) only when the window means pass the criterion AND the run executed its
whole clip (`ArmTimeBase.completed_clip`): `optimize_physics.py` leaves its loop early when the object
falls below its z threshold, penetrates in a sub-step, or goes NaN, and means over the executed steps
alone cannot see that. `TrackingErrors.success` is the window-means half only. An arm without any
physics output (never run, or killed before the driver wrote its files) gets `no_output_row`.

`mpc_iteration_stats` summarizes how many MPPI iterations `optimize_physics.py`'s receding-horizon
loop spent per control chunk (`opt_steps`), and whether a chunk used its whole budget without the
early-stop rule firing (the last `improvement_check_steps` improvements all below
`improvement_threshold`). Wall time scales with that iteration count, not with executed steps.

`report_table_lines` formats the report stage's rows for the log. Pure numpy; nothing here reads
from disk.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

__all__ = [
    "SUCCESS_POS_M",
    "SUCCESS_ROT_RAD",
    "TIME_BASE_LABEL",
    "ArmTimeBase",
    "MpcIterationStats",
    "TrackingErrors",
    "arm_row",
    "arm_status",
    "arm_success",
    "drop_onset_step",
    "flatten_executed_qpos",
    "meets_success_criterion",
    "mpc_iteration_stats",
    "no_output_row",
    "post_warmup_tracking_errors",
    "report_table_lines",
    "resampled_clip_steps",
]

SUCCESS_POS_M = 0.1      # the baseline's success criterion: mean E_pos < 0.1 m ...
SUCCESS_ROT_RAD = 0.5    # ... and mean E_rot < 0.5 rad
# The window every physics-output metric uses: executed steps [warmup_steps, min(T, max_sim_steps)).
TIME_BASE_LABEL = "post_warmup_to_max_sim_steps"


def meets_success_criterion(pos_mean_m: float, quat_mean_rad: float) -> bool:
    """The retargeting success criterion (strict): mean E_pos < 0.1 m and mean E_rot < 0.5 rad.

    NaN means (an empty metric window) fail, since every comparison with NaN is False."""
    return bool(pos_mean_m < SUCCESS_POS_M and quat_mean_rad < SUCCESS_ROT_RAD)


@dataclass(frozen=True)
class TrackingErrors:
    """Object tracking error over executed steps `[first_step, end_step)` (`time_base` names that
    window, `TIME_BASE_LABEL`); an empty window has `n_steps == 0` and NaN means and maxima."""

    time_base: str
    first_step: int
    end_step: int
    n_steps: int
    pos_mean_m: float
    pos_max_m: float
    quat_mean_rad: float
    quat_max_rad: float

    @property
    def success(self) -> bool:
        return meets_success_criterion(self.pos_mean_m, self.quat_mean_rad)

    def as_dict(self) -> dict[str, str | int | float | bool]:
        return {**asdict(self), "success": self.success}


def post_warmup_tracking_errors(pos_err_frames: np.ndarray, quat_err_frames: np.ndarray,
                                warmup_steps: int, end_step: int | None = None) -> TrackingErrors:
    """Mean/max object position and rotation error over executed steps `[warmup_steps, end)`, where
    `end = min(end_step, T)` (`T` when `end_step` is None; the report passes `max_sim_steps`).

    An empty window (the arm terminated during warmup, `T <= warmup_steps`, or `end <= warmup_steps`)
    is not an error: it returns `n_steps=0` with NaN means and maxima, so `success` is False."""
    pos = np.asarray(pos_err_frames, dtype=np.float64).reshape(-1)
    quat = np.asarray(quat_err_frames, dtype=np.float64).reshape(-1)
    if pos.shape != quat.shape:
        raise ValueError(f"per-step error series differ in length: {pos.shape} vs {quat.shape}")
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps={warmup_steps} must be non-negative")
    end = pos.size if end_step is None else min(int(end_step), pos.size)
    post_pos, post_quat = pos[warmup_steps:end], quat[warmup_steps:end]   # empty when end <= warmup
    if post_pos.size == 0:
        nan = float("nan")
        return TrackingErrors(time_base=TIME_BASE_LABEL, first_step=int(warmup_steps), end_step=int(end),
                              n_steps=0, pos_mean_m=nan, pos_max_m=nan, quat_mean_rad=nan, quat_max_rad=nan)
    return TrackingErrors(
        time_base=TIME_BASE_LABEL, first_step=int(warmup_steps), end_step=int(end),
        n_steps=int(post_pos.size),
        pos_mean_m=float(post_pos.mean()), pos_max_m=float(post_pos.max()),
        quat_mean_rad=float(post_quat.mean()), quat_max_rad=float(post_quat.max()),
    )


def flatten_executed_qpos(qpos: np.ndarray) -> np.ndarray:
    """`trajectory_mjwp.npz`'s `qpos (n_chunks, ctrl_steps, nq)` -> chronological `(T, nq)`.

    `optimize_physics.py` stacks one per-chunk block on a new leading axis; this is the equivalent of
    the `np.concatenate` over chunks its own tracking-error call uses. 2-D input passes through."""
    arr = np.asarray(qpos)
    return arr.reshape(-1, arr.shape[-1])


def drop_onset_step(pos_err_frames: np.ndarray, threshold_m: float, warmup_steps: int,
                    end_step: int | None = None) -> int | None:
    """First executed step in `[warmup_steps, min(end_step, T))` whose object position error exceeds
    the threshold, or None (also when the window is empty: the arm ended during warmup)."""
    pos = np.asarray(pos_err_frames).reshape(-1)
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps={warmup_steps} must be non-negative")
    end = pos.size if end_step is None else min(int(end_step), pos.size)
    if warmup_steps >= end:
        return None
    above = np.nonzero(pos[warmup_steps:end] > threshold_m)[0]
    return int(above[0]) + int(warmup_steps) if above.size else None


@dataclass(frozen=True)
class MpcIterationStats:
    """Per-chunk MPPI iteration usage, split into chunks before and from `split_chunk` on."""

    opt_steps: tuple[int, ...]
    budget_exhausted: tuple[bool, ...]
    final_rew_max: tuple[float, ...]
    max_iterations: int
    split_chunk: int
    total_iterations: int
    n_budget_exhausted: int
    mean_before: float
    mean_after: float

    def as_dict(self) -> dict[str, object]:
        return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}


def mpc_iteration_stats(opt_steps: np.ndarray, improvement: np.ndarray, rew_max: np.ndarray,
                        max_iterations: int, improvement_threshold: float, check_steps: int,
                        split_chunk: int) -> MpcIterationStats:
    """Summarize `optimize_physics.py`'s per-chunk `opt_steps (n_chunks, 1)` and per-iteration
    `improvement`/`rew_max (n_chunks, max_iterations)` (zero-padded after an early stop).

    A chunk exhausts its budget when it ran all `max_iterations` and its last `check_steps`
    improvements were not all below `improvement_threshold` (the rule can fire on the final
    iteration, which also yields `opt_steps == max_iterations`). `final_rew_max` is the best
    sampled planned-horizon reward at each chunk's last iteration."""
    steps = np.asarray(opt_steps).reshape(-1).astype(int)
    imp = np.asarray(improvement, dtype=np.float64)
    rew = np.asarray(rew_max, dtype=np.float64)
    exhausted = []
    for c, n in enumerate(steps):
        recent = imp[c, max(0, n - check_steps):n]
        early_stop_fired = recent.size >= check_steps and bool(np.all(recent < improvement_threshold))
        exhausted.append(bool(n >= max_iterations and not early_stop_fired))
    before, after = steps[:split_chunk], steps[split_chunk:]
    return MpcIterationStats(
        opt_steps=tuple(int(n) for n in steps), budget_exhausted=tuple(exhausted),
        final_rew_max=tuple(float(rew[c, n - 1]) for c, n in enumerate(steps)),
        max_iterations=int(max_iterations), split_chunk=int(split_chunk),
        total_iterations=int(steps.sum()), n_budget_exhausted=int(sum(exhausted)),
        mean_before=float(before.mean()) if before.size else float("nan"),
        mean_after=float(after.mean()) if after.size else float("nan"),
    )


def resampled_clip_steps(raw_frames: int, ref_dt: float, sim_dt: float) -> int:
    """Sim steps the baseline's `align_to_sim_dt` turns a `raw_frames`-frame reference into: upsampled
    by `round(ref_dt / sim_dt)` when `ref_dt >= sim_dt`, else strided by `round(sim_dt / ref_dt)`."""
    if ref_dt >= sim_dt:
        return int(raw_frames) * max(1, round(ref_dt / sim_dt))
    return -(-int(raw_frames) // max(1, round(sim_dt / ref_dt)))


@dataclass(frozen=True)
class ArmTimeBase:
    """One arm's own step accounting. `optimize_physics.py` resolves an uncapped `max_sim_steps` as
    `len(reference) - horizon - ctrl = warmup + clip`, where `clip_steps` is the resampled raw clip
    (`resampled_clip_steps`); with a cap, `run_steps = max_sim_steps - warmup` differs from the clip
    (`capped`). Whole-chunk execution can pass `max_sim_steps` (`overshoot_steps`, on the held last
    reference frame when uncapped); metrics use `[warmup_steps, metric_end_step)`."""

    max_sim_steps: int
    warmup_steps: int
    raw_reference_frames: int
    executed_steps: int
    chunk_steps: int
    clip_steps: int

    @property
    def run_steps(self) -> int:
        return self.max_sim_steps - self.warmup_steps

    @property
    def capped(self) -> bool:
        return self.run_steps != self.clip_steps

    @property
    def metric_end_step(self) -> int:
        return min(self.executed_steps, self.max_sim_steps)

    @property
    def metric_steps(self) -> int:
        return max(self.metric_end_step - self.warmup_steps, 0)

    @property
    def overshoot_steps(self) -> int:
        return max(self.executed_steps - self.max_sim_steps, 0)

    @property
    def post_warmup_executed_steps(self) -> int:
        return max(self.executed_steps - self.warmup_steps, 0)

    @property
    def clip_fraction_executed(self) -> float:
        return min(self.metric_steps, self.clip_steps) / self.clip_steps

    @property
    def completed_clip(self) -> bool:
        """The whole clip ran: no early termination (`executed_steps >= max_sim_steps`) and no
        `max_sim_steps` cap shorter than the clip."""
        return self.executed_steps >= self.max_sim_steps and not self.capped

    def as_dict(self) -> dict[str, int | float | bool]:
        derived = ("run_steps", "capped", "metric_end_step", "metric_steps", "overshoot_steps",
                   "post_warmup_executed_steps", "clip_fraction_executed", "completed_clip")
        return {**asdict(self), **{k: getattr(self, k) for k in derived}}


def arm_success(tracking: TrackingErrors, time_base: ArmTimeBase) -> bool:
    """The addendum's per-arm success: E_pos < 0.1 m and E_rot < 0.5 rad over the full clip with no early
    termination, i.e. the window means pass (`tracking.success`) and the whole clip ran."""
    return tracking.success and time_base.completed_clip


def arm_status(time_base: ArmTimeBase) -> str:
    """How an arm's physics run ended: "ended_in_warmup", "terminated_early", "capped" or "completed"."""
    if time_base.executed_steps <= time_base.warmup_steps:
        return "ended_in_warmup"
    if time_base.executed_steps < time_base.max_sim_steps:
        return "terminated_early"
    return "capped" if time_base.capped else "completed"


def arm_row(arm: str, wall_seconds: float, tracking: TrackingErrors, time_base: ArmTimeBase,
            baseline_log_errors: tuple[float, float], physics: dict[str, Any], mpc: MpcIterationStats,
            kinematic: dict[str, Any]) -> dict[str, Any]:
    """The report row of an arm whose physics run wrote its outputs. `physics` has `n_frames`,
    `frac_under_2mm` and `frac_any_penetration` over the metric window; `baseline_log_*` keep the
    driver's whole-trace means (warmup and overshoot included) and the criterion applied to them."""
    return {
        "arm": arm, "status": arm_status(time_base), "wall_seconds": float(wall_seconds),
        "time_base": tracking.time_base,
        "obj_pos_err_mean": tracking.pos_mean_m, "obj_quat_err_mean": tracking.quat_mean_rad,
        "obj_pos_err_max": tracking.pos_max_m, "obj_quat_err_max": tracking.quat_max_rad,
        "success": arm_success(tracking, time_base), "window_means_success": tracking.success,
        "completed_clip": time_base.completed_clip, "tracking": tracking.as_dict(),
        "baseline_log_obj_pos_err": baseline_log_errors[0], "baseline_log_obj_quat_err": baseline_log_errors[1],
        "baseline_log_success": meets_success_criterion(*baseline_log_errors),
        "physics_stats_frames": physics["n_frames"],
        "physics_penetration_under_2mm_frac": physics["frac_under_2mm"],
        "physics_any_penetration_frac": physics["frac_any_penetration"],
        "time_base_steps": time_base.as_dict(), "mpc_iterations": mpc.as_dict(), "kinematic": kinematic,
        "missing_artifacts": [],
    }


def no_output_row(arm: str, missing: list[str], kinematic: dict[str, Any]) -> dict[str, Any]:
    """The report row of an arm without physics output (never run, or killed before the driver wrote its
    files): status "no_output", success False, NaN metrics, no step accounting; `arm_row`'s keys."""
    nan = float("nan")
    return {
        "arm": arm, "status": "no_output", "wall_seconds": nan, "time_base": TIME_BASE_LABEL,
        "obj_pos_err_mean": nan, "obj_quat_err_mean": nan, "obj_pos_err_max": nan, "obj_quat_err_max": nan,
        "success": False, "window_means_success": False, "completed_clip": False, "tracking": None,
        "baseline_log_obj_pos_err": nan, "baseline_log_obj_quat_err": nan, "baseline_log_success": False,
        "physics_stats_frames": 0, "physics_penetration_under_2mm_frac": nan, "physics_any_penetration_frac": nan,
        "time_base_steps": None, "mpc_iterations": None, "kinematic": kinematic, "missing_artifacts": list(missing),
    }


def report_table_lines(rows: list[dict[str, Any]], kinematic_stats: dict[str, Any]) -> list[str]:
    """The report stage's log table: one tracking/penetration line per arm, one step-accounting and
    MPC-iteration line per arm, and each reference's kinematic-only clearance (sampled points, plus the
    collision-primitive statistics when `prepare_stats.json` has them)."""
    lines = ["\n=== Physics isolation study: whisking demo ===",
             (f"tracking error and physics penetration: time base {TIME_BASE_LABEL}, executed steps in "
              "[warmup_steps, min(executed_steps, max_sim_steps)); success = window means pass AND the whole "
              "clip ran (no early termination); baseline_log_* in report.json keep the whole-trace log-line values"),
             (f"{'arm':<16}{'status':>18}{'wall_s':>10}{'pos_err_mean':>14}{'quat_err_mean':>15}{'success':>9}"
              f"{'phys_pen<2mm':>14}{'phys_any_pen':>14}")]
    lines += [f"{r['arm']:<16}{r['status']:>18}{r['wall_seconds']:>10.1f}{r['obj_pos_err_mean']:>14.4f}"
              f"{r['obj_quat_err_mean']:>15.4f}{r['success']!s:>9}"
              f"{r['physics_penetration_under_2mm_frac']:>14.3f}{r['physics_any_penetration_frac']:>14.3f}"
              for r in rows]
    for r in rows:
        tb, mpc = r["time_base_steps"], r["mpc_iterations"]
        if tb is None or mpc is None:
            lines.append(f"  {r['arm']:<16} no physics output (not run, or killed): missing "
                         f"{', '.join(r['missing_artifacts'])}")
            continue
        lines.append(
            f"  {r['arm']:<16} max_sim_steps={tb['max_sim_steps']} warmup_steps={tb['warmup_steps']} "
            f"raw_ref_frames={tb['raw_reference_frames']} executed_steps={tb['executed_steps']} (clip "
            f"{tb['clip_steps']} steps, capped={tb['capped']}, metric window {tb['metric_steps']} steps, "
            f"overshoot={tb['overshoot_steps']}, {100.0 * tb['clip_fraction_executed']:.1f}% of clip executed) "
            f"| MPC iterations total={mpc['total_iterations']} exhausted={mpc['n_budget_exhausted']}/"
            f"{len(mpc['opt_steps'])} mean/chunk before={mpc['mean_before']:.2f} after={mpc['mean_after']:.2f} "
            f"(split chunk {mpc['split_chunk']})")
    lines.append("\nkinematic-only (pre-physics) clearance/penetration per reference:")
    for arm, ks in kinematic_stats.items():
        line = (f"  {arm:<16} sampled: thumb={ks['thumb_median_mm']:.3f}mm index={ks['index_median_mm']:.3f}mm "
                f"middle={ks['middle_median_mm']:.3f}mm <2mm_frac={ks['frac_under_2mm']:.3f} "
                f"any_pen_frac={ks['frac_any_penetration']:.3f}")
        if (prim := ks.get("primitive")) is not None:
            line += (f" | primitive: pen_frames={prim['frac_frames_penetrating']:.3f} "
                     f">=0.5mm_frames={prim['frac_frames_clear_0p5mm']:.3f} thumb={prim['thumb_median_mm']:.2f}mm "
                     f"index={prim['index_median_mm']:.2f}mm middle={prim['middle_median_mm']:.2f}mm")
        lines.append(line)
    return lines
