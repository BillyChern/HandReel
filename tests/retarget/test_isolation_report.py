"""Tests for `fastdo.retarget.isolation_report`: the report-stage helpers that
put tracking error and penetration on one post-warmup time base, summarize the MPC's per-chunk
iteration usage, and lay out each arm's step accounting. Synthetic arrays only; no baseline data.
"""

from __future__ import annotations

import numpy as np
import pytest

from fastdo.retarget.isolation_report import (
    ArmTimeBase,
    arm_status,
    arm_success,
    drop_onset_step,
    flatten_executed_qpos,
    meets_success_criterion,
    mpc_iteration_stats,
    no_output_row,
    post_warmup_tracking_errors,
    report_table_lines,
    resampled_clip_steps,
)


def test_post_warmup_tracking_errors_excludes_the_warmup_prefix() -> None:
    """The driver's per-step errors cover the whole executed trace, warmup included; a welded,
    near-zero-error warmup prefix must not dilute the mean or hide the max."""
    pos = np.array([0.0, 0.0, 0.0, 0.2, 0.4])
    quat = np.array([0.0, 0.0, 0.0, 1.0, 3.0])
    errs = post_warmup_tracking_errors(pos, quat, warmup_steps=3)

    assert errs.time_base == "post_warmup_to_max_sim_steps"
    assert errs.first_step == 3
    assert errs.n_steps == 2
    assert errs.pos_mean_m == pytest.approx(0.3)
    assert errs.pos_max_m == pytest.approx(0.4)
    assert errs.quat_mean_rad == pytest.approx(2.0)
    assert errs.quat_max_rad == pytest.approx(3.0)
    assert errs.success is False
    assert errs.as_dict()["n_steps"] == 2


def test_post_warmup_success_uses_the_baselines_strict_thresholds() -> None:
    ok = post_warmup_tracking_errors(np.array([9.0, 0.05]), np.array([9.0, 0.49]), warmup_steps=1)
    assert ok.success is True
    at_pos_bound = post_warmup_tracking_errors(np.array([0.1]), np.array([0.0]), warmup_steps=0)
    assert at_pos_bound.success is False
    at_rot_bound = post_warmup_tracking_errors(np.array([0.0]), np.array([0.5]), warmup_steps=0)
    assert at_rot_bound.success is False
    assert meets_success_criterion(0.0999, 0.4999) is True
    assert meets_success_criterion(0.1, 0.0) is False
    assert meets_success_criterion(0.0, 0.5) is False


def test_post_warmup_tracking_errors_stops_at_the_runs_own_max_sim_steps() -> None:
    """The MPC executes whole chunks, so a run can step past `max_sim_steps` onto the held last
    reference frame; those overshoot steps are not part of the clip and must not enter the window."""
    pos = np.array([0.0, 0.0, 0.1, 0.3, 9.0, 9.0])
    errs = post_warmup_tracking_errors(pos, pos, warmup_steps=2, end_step=4)
    assert (errs.first_step, errs.end_step, errs.n_steps) == (2, 4, 2)
    assert errs.pos_mean_m == pytest.approx(0.2)
    assert errs.pos_max_m == pytest.approx(0.3)
    clamped = post_warmup_tracking_errors(pos, pos, warmup_steps=2, end_step=100)
    assert (clamped.end_step, clamped.n_steps) == (6, 4)


@pytest.mark.parametrize(("n_pos", "n_quat", "warmup"), [(4, 3, 0), (4, 4, -1)])
def test_post_warmup_tracking_errors_rejects_inconsistent_inputs(
    n_pos: int, n_quat: int, warmup: int,
) -> None:
    with pytest.raises(ValueError):
        post_warmup_tracking_errors(np.zeros(n_pos), np.zeros(n_quat), warmup_steps=warmup)


@pytest.mark.parametrize(("n_steps", "warmup", "end_step"), [(4, 4, None), (4, 6, None), (6, 2, 2), (0, 0, None)])
def test_post_warmup_tracking_errors_gives_an_empty_failed_row_when_the_arm_ends_in_warmup(
    n_steps: int, warmup: int, end_step: int | None,
) -> None:
    """An arm that terminates during warmup (executed_steps <= warmup_steps), or a window closed by
    max_sim_steps, has no metric steps. The report still needs that arm's row: n_steps=0, NaN means
    and maxima, success False, instead of an exception that aborts the whole report."""
    errs = post_warmup_tracking_errors(np.zeros(n_steps), np.zeros(n_steps), warmup_steps=warmup,
                                       end_step=end_step)
    assert errs.n_steps == 0
    assert errs.first_step == warmup
    assert np.isnan(errs.pos_mean_m) and np.isnan(errs.pos_max_m)
    assert np.isnan(errs.quat_mean_rad) and np.isnan(errs.quat_max_rad)
    assert errs.success is False
    assert errs.as_dict()["success"] is False
    assert meets_success_criterion(float("nan"), float("nan")) is False


def test_flatten_executed_qpos_is_chronological() -> None:
    """`trajectory_mjwp.npz` stacks one `(ctrl_steps, nq)` block per chunk on a new leading axis;
    flattening must equal the baseline's own `np.concatenate` over its per-chunk list."""
    chunks = [np.arange(12, dtype=np.float64).reshape(3, 4) + 100 * c for c in range(2)]
    stacked = np.stack(chunks, axis=0)
    np.testing.assert_array_equal(flatten_executed_qpos(stacked), np.concatenate(chunks, axis=0))
    flat = np.zeros((6, 4))
    assert flatten_executed_qpos(flat).shape == (6, 4)


def test_drop_onset_step_finds_the_first_post_warmup_threshold_crossing() -> None:
    pos = np.array([0.5, 0.0, 0.05, 0.11, 0.02, 0.3])
    assert drop_onset_step(pos, threshold_m=0.1, warmup_steps=1) == 3
    assert drop_onset_step(pos, threshold_m=0.1, warmup_steps=0) == 0
    assert drop_onset_step(pos, threshold_m=1.0, warmup_steps=1) is None
    assert drop_onset_step(pos, threshold_m=0.25, warmup_steps=1, end_step=5) is None
    assert drop_onset_step(pos, threshold_m=0.1, warmup_steps=pos.size) is None


def test_drop_onset_step_rejects_a_negative_warmup() -> None:
    with pytest.raises(ValueError):
        drop_onset_step(np.zeros(6), threshold_m=0.1, warmup_steps=-1)


def test_drop_onset_step_is_none_when_warmup_outlasts_the_executed_series() -> None:
    """An arm that terminated during warmup has no post-warmup step, so no drop onset either."""
    assert drop_onset_step(np.full(6, 9.0), threshold_m=0.1, warmup_steps=7) is None
    assert drop_onset_step(np.full(6, 9.0), threshold_m=0.1, warmup_steps=6) is None


def test_mpc_iteration_stats_counts_budget_exhaustion_and_splits_chunks() -> None:
    """A chunk exhausts its budget only if it used every iteration AND the early-stop rule (the last
    `check_steps` improvements all below threshold) did not fire on the final iteration."""
    max_it, thr = 4, 0.01
    opt_steps = np.array([[2], [4], [4], [1]])
    improvement = np.zeros((4, max_it), dtype=np.float32)
    improvement[1] = [0.5, 0.2, 0.02, 0.005]   # used all 4; last two not both < thr -> exhausted
    improvement[2] = [0.5, 0.2, 0.005, 0.001]  # used all 4; early stop fired on the last one
    rew_max = np.arange(16, dtype=np.float32).reshape(4, max_it) * -1.0

    stats = mpc_iteration_stats(opt_steps, improvement, rew_max, max_iterations=max_it,
                                improvement_threshold=thr, check_steps=2, split_chunk=2)

    assert stats.opt_steps == (2, 4, 4, 1)
    assert stats.budget_exhausted == (False, True, False, False)
    assert stats.n_budget_exhausted == 1
    assert stats.total_iterations == 11
    assert stats.mean_before == pytest.approx(3.0)
    assert stats.mean_after == pytest.approx(2.5)
    assert stats.final_rew_max == (-1.0, -7.0, -11.0, -12.0)
    assert stats.as_dict()["split_chunk"] == 2


def test_mpc_iteration_stats_split_past_the_last_chunk_leaves_after_empty() -> None:
    stats = mpc_iteration_stats(np.array([[3], [5]]), np.ones((2, 8)), np.zeros((2, 8)),
                                max_iterations=8, improvement_threshold=0.01, check_steps=2,
                                split_chunk=5)
    assert stats.mean_before == pytest.approx(4.0)
    assert np.isnan(stats.mean_after)


def test_resampled_clip_steps_mirrors_the_baselines_align_to_sim_dt() -> None:
    """`align_to_sim_dt`: upsample by round(ref_dt / sim_dt) when ref_dt >= sim_dt, else stride."""
    assert resampled_clip_steps(135, ref_dt=0.0333, sim_dt=0.005) == 945
    assert resampled_clip_steps(138, ref_dt=0.0333, sim_dt=0.005) == 966
    assert resampled_clip_steps(10, ref_dt=0.005, sim_dt=0.005) == 10
    assert resampled_clip_steps(10, ref_dt=0.005, sim_dt=0.015) == 4  # len(arr[::3])


def test_arm_time_base_windows_metrics_at_max_sim_steps_and_flags_overshoot() -> None:
    """Real C4 accounting. Ours terminated early (500 of its 966 clip steps). Theirs executed whole
    chunks up to step 1600, 55 steps past its max_sim_steps=1545, on the held last reference frame:
    the metric window stops at 1545 and the overshoot is reported, not hidden by a capped fraction."""
    ours = ArmTimeBase(max_sim_steps=1566, warmup_steps=600, raw_reference_frames=138,
                       executed_steps=1100, chunk_steps=100, clip_steps=966)
    assert (ours.run_steps, ours.capped, ours.metric_end_step, ours.metric_steps) == (966, False, 1100, 500)
    assert ours.overshoot_steps == 0
    assert ours.clip_fraction_executed == pytest.approx(500 / 966)

    theirs = ArmTimeBase(max_sim_steps=1545, warmup_steps=600, raw_reference_frames=135,
                         executed_steps=1600, chunk_steps=100, clip_steps=945)
    assert (theirs.metric_end_step, theirs.metric_steps, theirs.overshoot_steps) == (1545, 945, 55)
    assert theirs.post_warmup_executed_steps == 1000
    assert theirs.clip_fraction_executed == pytest.approx(1.0)
    assert theirs.as_dict()["overshoot_steps"] == 55 and theirs.as_dict()["capped"] is False


def test_arm_time_base_of_an_arm_that_ended_in_warmup_has_an_empty_window() -> None:
    ended = ArmTimeBase(max_sim_steps=1566, warmup_steps=600, raw_reference_frames=138,
                        executed_steps=500, chunk_steps=100, clip_steps=966)
    assert (ended.metric_end_step, ended.metric_steps, ended.post_warmup_executed_steps) == (500, 0, 0)
    assert ended.clip_fraction_executed == 0.0


def test_arm_time_base_distinguishes_a_capped_run_from_the_clip() -> None:
    """With `isolation.max_sim_steps` set, `max_sim_steps - warmup` is the cap, not the clip."""
    capped = ArmTimeBase(max_sim_steps=1000, warmup_steps=600, raw_reference_frames=135,
                         executed_steps=1000, chunk_steps=100, clip_steps=945)
    assert capped.capped is True
    assert (capped.run_steps, capped.clip_steps, capped.metric_steps) == (400, 945, 400)
    assert capped.clip_fraction_executed == pytest.approx(400 / 945)


def test_report_table_lines_add_primitive_clearance_when_prepare_recorded_it() -> None:
    kin = {"frac_under_2mm": 1.0, "frac_any_penetration": 0.0, "thumb_median_mm": 4.1, "index_median_mm": 3.7,
           "middle_median_mm": 3.6}
    prim = {"frac_frames_penetrating": 0.0, "frac_frames_clear_0p5mm": 1.0, "thumb_median_mm": 2.6,
            "index_median_mm": 2.86, "middle_median_mm": 2.84}
    lines = report_table_lines([], {"ours": kin, "ours_clearance": {**kin, "primitive": prim}})
    ours = next(line for line in lines if line.strip().startswith("ours "))
    clearance = next(line for line in lines if line.strip().startswith("ours_clearance"))
    assert "primitive" not in ours
    assert "primitive: pen_frames=0.000 >=0.5mm_frames=1.000 thumb=2.60mm index=2.86mm middle=2.84mm" in clearance
    assert "post_warmup_to_max_sim_steps" in lines[1]


def test_arm_success_requires_the_whole_clip_not_only_low_window_means() -> None:
    """The addendum's criterion is E_pos < 0.1 m and E_rot < 0.5 rad over the full clip with no early
    termination. `optimize_physics.py` leaves its loop when the object falls below its z threshold,
    penetrates in a sub-step, or goes NaN, so a run that dropped the object after warmup can still have
    low means over the steps it executed: that run fails. A capped run never covers the clip either."""
    low = post_warmup_tracking_errors(np.full(1100, 0.05), np.full(1100, 0.1), warmup_steps=600)
    assert low.success is True
    dropped = ArmTimeBase(max_sim_steps=1566, warmup_steps=600, raw_reference_frames=138,
                          executed_steps=1100, chunk_steps=100, clip_steps=966)
    assert dropped.completed_clip is False and arm_success(low, dropped) is False
    assert arm_status(dropped) == "terminated_early"
    full = ArmTimeBase(max_sim_steps=1545, warmup_steps=600, raw_reference_frames=135,
                       executed_steps=1600, chunk_steps=100, clip_steps=945)
    assert full.completed_clip is True and arm_success(low, full) is True
    assert arm_status(full) == "completed" and full.as_dict()["completed_clip"] is True
    high = post_warmup_tracking_errors(np.full(1600, 0.2), np.zeros(1600), warmup_steps=600)
    assert arm_success(high, full) is False
    capped = ArmTimeBase(max_sim_steps=1000, warmup_steps=600, raw_reference_frames=135,
                         executed_steps=1000, chunk_steps=100, clip_steps=945)
    assert capped.completed_clip is False and arm_success(low, capped) is False
    assert arm_status(capped) == "capped"
    in_warmup = ArmTimeBase(max_sim_steps=1566, warmup_steps=600, raw_reference_frames=138,
                            executed_steps=500, chunk_steps=100, clip_steps=966)
    assert arm_status(in_warmup) == "ended_in_warmup" and in_warmup.completed_clip is False


def test_no_output_row_fails_with_nan_metrics_and_the_table_reports_it() -> None:
    """An arm the GPU guard killed (or that never ran) leaves no `arm_meta.json`; the report still gets
    its row, failed, with NaN metrics and the missing files named."""
    row = no_output_row("ours_clearance", ["arm_meta.json", "trajectory_mjwp.npz"], {"thumb_median_mm": 2.7})
    assert row["status"] == "no_output" and row["success"] is False and row["completed_clip"] is False
    assert row["window_means_success"] is False
    assert np.isnan(row["obj_pos_err_mean"]) and np.isnan(row["obj_quat_err_max"]) and np.isnan(row["wall_seconds"])
    assert row["missing_artifacts"] == ["arm_meta.json", "trajectory_mjwp.npz"]
    assert row["time_base_steps"] is None and row["mpc_iterations"] is None
    assert row["kinematic"] == {"thumb_median_mm": 2.7}
    lines = report_table_lines([row], {})
    assert any(line.startswith("ours_clearance") and "no_output" in line for line in lines)
    assert any("no physics output" in line and "arm_meta.json" in line for line in lines)
