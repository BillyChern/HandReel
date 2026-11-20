"""M7: capped MPC fallback (known answers and mutations).

Pure: the acceptance rule (only certified output ships; a certified candidate is never replaced), the corpus budget
(order by margin, admit only while spent + one cap fits, resumable ledger), the warm-start rows and file, the
driver's flag handling. The entry-point known answers are `test_m7_entry_point.py` (slow). Mutation targets here:
uncertified output shipped; the budget admitting past B_fb; an iteration cap above 8 accepted by the driver; a warm
start accepted without its caps; a hold charged its cap instead of its actual wall (a killed hold included).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from fastdo.amortised import fallback as FB
from fastdo.amortised import speed_ledger as L
from fastdo.amortised.knots import residual_rows

TREE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TREE / "run" / "pipeline"))
import _baseline_physics_driver as drv


def test_only_certified_output_ships_and_never_replaces_a_certified_candidate() -> None:
    assert FB.accept(None, fallback_certified=False) == "none"
    assert FB.accept(None, fallback_certified=True) == "fallback"
    assert FB.accept({"episode": 3}, fallback_certified=True) == "kept_certified"
    assert FB.accept({"episode": 3}, fallback_certified=False) == "kept_certified"


def test_budget_orders_by_margin_admits_within_b_fb_and_resumes(tmp_path) -> None:
    cap = FB.c_cap(800.0)
    assert cap == 200.0 and FB.corpus_budget(100, cap) == pytest.approx(1000.0)
    b = FB.FallbackBudget(tmp_path / "fb.json", 100, cap)
    order = b.order([("c", 0.9), ("a", 1.4), ("b", 0.9), ("d", 0.2)])
    assert order == ["d", "b", "c", "a"]
    spent = []
    for t in order + ["e", "f", "g"]:
        if not b.admit():
            break
        b.record(t, gpu_s=cap if t != "b" else 120.0, capped=t != "b", certified=t == "b")
        spent.append(t)
    assert b.spent_s <= b.budget_s and len(spent) == 5          # 4 x 200 + 120 = 920; a 6th would need 1120
    again = FB.FallbackBudget(tmp_path / "fb.json", 100, cap)
    assert again.spent_s == b.spent_s and not again.admit()
    assert again.order([("d", 0.2), ("z", 5.0)]) == ["z"]
    s = again.summary()
    assert s["worst_clip"]["gpu_s"] == cap and s["certified"] == 1 and s["capped"] == 4
    with pytest.raises(ValueError):
        FB.FallbackBudget(tmp_path / "fb.json", 50, cap)


def test_caps_and_driver_args() -> None:
    caps = FB.FallbackCaps(hold_cap_s=200.0)
    assert caps.iteration_cap == FB.ITERATION_CAP == 8
    args = FB.driver_args(caps, Path("/x/ws.npz"))
    assert args == ["--executor", "fast", "--warm-start", "/x/ws.npz", "--iteration-cap", "8", "--hold-cap-s", "200.0"]
    with pytest.raises(ValueError):
        FB.FallbackCaps(hold_cap_s=10.0, iteration_cap=9)


def test_warm_start_file_and_rows(tmp_path) -> None:
    knots = np.array([600, 640, 680])
    U = np.array([[0.1, -0.2], [0.3, 0.0], [0.5, 0.2]])
    p = FB.write_warm_start(tmp_path / "w.npz", knots, U, {"cand": "c1"})
    k2, U2 = FB.load_warm_start(p)
    assert np.array_equal(k2, knots) and np.array_equal(U2, U)
    ref = np.random.default_rng(0).normal(size=(700, 3))
    rows = FB.warm_start_rows(ref, knots, U)
    np.testing.assert_allclose(rows[:, :2] - ref[:, :2], residual_rows(knots, U, 700))
    np.testing.assert_array_equal(rows[:, 2], ref[:, 2])
    np.testing.assert_array_equal(FB.warm_start_rows(ref, knots, np.zeros_like(U)), ref)


def test_driver_resolves_the_flags_only_when_given(tmp_path) -> None:
    assert drv.resolve_fallback("fast", None, None, None) == ({}, None)
    p = FB.write_warm_start(tmp_path / "w.npz", np.array([10, 20]), np.ones((2, 3)))
    kw, rec = drv.resolve_fallback("fast", str(p), 8, 120.0)
    assert set(kw) == {"warm_start", "iteration_cap", "hold_cap_s"} and kw["iteration_cap"] == 8
    assert rec["knots"] == 2 and len(rec["warm_start_sha256"]) == 64
    with pytest.raises(ValueError):
        drv.resolve_fallback("reference", str(p), None, None)
    with pytest.raises(ValueError):
        drv.resolve_fallback("fast", None, 0, None)


def test_driver_refuses_an_iteration_cap_above_8() -> None:
    assert drv.resolve_fallback("fast", None, FB.ITERATION_CAP, None)[0] == {"iteration_cap": FB.ITERATION_CAP}
    for cap in (FB.ITERATION_CAP + 1, 50):
        with pytest.raises(ValueError, match="design's cap"):
            drv.resolve_fallback("fast", None, cap, None)


def test_driver_refuses_a_warm_start_without_both_caps(tmp_path) -> None:
    p = FB.write_warm_start(tmp_path / "w.npz", np.array([10, 20]), np.ones((2, 3)))
    for it, hold in ((None, None), (8, None), (None, 120.0)):
        with pytest.raises(ValueError, match="needs --iteration-cap and --hold-cap-s"):
            drv.resolve_fallback("fast", str(p), it, hold)
    args = FB.driver_args(FB.FallbackCaps(hold_cap_s=120.0), p)
    kw, _ = drv.resolve_fallback("fast", args[3], int(args[5]), float(args[7]))
    assert set(kw) == {"warm_start", "iteration_cap", "hold_cap_s"}


def test_budget_charges_the_actual_hold_wall_and_reserves_the_worst_hold(tmp_path) -> None:
    """--hold-cap-s bounds the control loop only: a hold over its cap is charged its lock-held wall, a hold killed by
    the driver's timeout is charged to the reap time, and admission then reserves the largest hold seen."""
    cap = 200.0
    b = FB.FallbackBudget(tmp_path / "fb.json", 80, cap)               # B_fb = 800
    over = L.Hold("fb#0", "fallback", t_start=0.0, t_lock=5.0, t_release=5.0 + 260.0)
    b.record_hold("a", over, capped=True, certified=False)
    killed = L.holds_from_drive([{"name": "fb#1", "phase": "running", "t_start": 300.0, "t_lock": 301.0,
                                  "t_release": None, "t_reaped": 301.0 + 330.0}], "fallback")
    assert len(killed) == 1 and killed[0].release_source == "driver_reap"
    b.record_hold("b", killed[0], capped=True, certified=False)
    assert b.spent_s == pytest.approx(590.0) and b.reserve_s() == pytest.approx(330.0)
    assert not b.admit()                                               # 590 + 330 > 800 although 590 + 200 fits
    s = FB.FallbackBudget(tmp_path / "fb.json", 80, cap).summary()
    assert s["over_cap_holds"] == 2 and s["overshoot_s"] == 0.0 and s["worst_clip"]["hold"] == "fb#1"


# ------------------------------------------------------------------------------------------------ R3 whole-hold bound


def _row(seq: int, t_launch: float, t_lock: float, t_release: float, setup_s: float | None = None,
         rc: object = 0, timeout_s: float = 1020.0) -> dict:
    return {"name": f"u#{seq}", "rc": rc, "t_launch": t_launch, "t_lock": t_lock, "t_release": t_release,
            "setup_s": setup_s, "timeout_s": timeout_s, "release_source": "unit" if rc != "timeout" else "driver_reap"}


def test_hold_bound_is_c_cap_plus_p95_of_the_first_setups_then_frozen(tmp_path) -> None:
    """R3: timeout = c_cap + p95(setup) over the FIRST n holds (numpy's linear percentile), the driver's backstop
    before that, and later holds never move it."""
    from fastdo.amortised import unit as U

    assert FB.SETUP_SAMPLES == 5 and FB.BACKSTOP_S == U.UNIT_MAX_S + 120.0
    b = FB.HoldBound(tmp_path / "bound.json", cap_s=100.0)
    setups = [10.0, 12.0, 11.0, 30.0, 14.0]
    for i, s in enumerate(setups):
        assert b.timeout_s() == FB.BACKSTOP_S and not b.measured
        b.record(_row(i, 0.0, 1.0, 1.0 + s + 50.0, setup_s=s))
    assert b.measured
    want = 100.0 + float(np.percentile(setups, 95))
    assert b.timeout_s() == pytest.approx(want) and want == pytest.approx(100.0 + 26.8)
    b.record(_row(9, 0.0, 1.0, 500.0, setup_s=400.0))                 # a later slow setup does not move the bound
    assert b.timeout_s() == pytest.approx(want)
    again = FB.HoldBound(tmp_path / "bound.json", cap_s=100.0)          # resumable
    assert again.timeout_s() == pytest.approx(want) and len(again.holds) == 6
    with pytest.raises(ValueError):
        FB.HoldBound(tmp_path / "bound.json", cap_s=90.0)


def test_hold_bound_setup_sample_from_the_loop_time_covers_the_whole_non_loop_hold(tmp_path) -> None:
    """An MPC hold's sample is launch-to-release minus the run's own loop_s (imports, scene build, graph capture and
    the epilogue); the stamp is used only without a loop time; a hold without either gives no sample."""
    b = FB.HoldBound(tmp_path / "b.json", cap_s=50.0, n_setup=2)
    b.record(_row(0, 10.0, 10.5, 110.0, setup_s=7.0), loop_s=60.0)       # 100 - 60 = 40, not the stamp's 7
    assert b.setup_s == [pytest.approx(40.0)]
    b.record(_row(1, 0.0, 0.5, 30.0, setup_s=None), loop_s=None)         # killed in setup: no sample
    assert len(b.setup_s) == 1
    b.record(_row(2, 0.0, 0.5, 30.0, setup_s=9.0))
    assert b.setup_s == [pytest.approx(40.0), pytest.approx(9.0)]


def test_a_killed_hold_gives_no_setup_sample_even_with_a_stamp(tmp_path) -> None:
    """H1: a hold killed by its timeout gives NO setup sample, whatever it
    stamped (its stamp covers only the setup that ran before it). With `loop_only` (the MPC holds), only a finished
    hold (rc 0) with a loop time gives one; the stamp is never used, and a failed unit gives none."""
    b = FB.HoldBound(tmp_path / "b.json", cap_s=None, n_setup=5)
    b.record(_row(0, 0.0, 0.5, 1020.4, setup_s=2.1, rc="timeout", timeout_s=1020.0))
    b.record(_row(1, 0.0, 0.5, 1020.4, setup_s=2.1, rc="timeout", timeout_s=1020.0), loop_s=900.0)
    assert b.setup_s == [] and len(b.holds) == 2 and all(h["killed"] and h["setup_s"] is None for h in b.holds)
    b.record(_row(2, 0.0, 0.5, 40.0, setup_s=4.0), loop_only=True)                  # no loop time: no stamp either
    b.record(_row(3, 0.0, 0.5, 40.0, setup_s=4.0, rc=1), loop_s=20.0, loop_only=True)  # the unit failed
    assert b.setup_s == []
    b.record(_row(4, 0.0, 0.5, 40.0, setup_s=4.0), loop_s=20.0, loop_only=True)
    b.record(_row(5, 0.0, 0.5, 40.0, setup_s=4.0))                                 # generic work: the stamp
    assert b.setup_s == [pytest.approx(20.0), pytest.approx(4.0)]


def test_hold_bound_records_every_overshoot_and_charges_killed_holds(tmp_path) -> None:
    """A hold the driver kills at its timeout (the timeout counts from the LAUNCH) is reaped at about launch +
    timeout: its over-timeout is read on that base (here 0.4 s of reap latency), while its CHARGED over-cap is on the
    lock-held wall (the lock was taken 3 s after the launch)."""
    b = FB.HoldBound(tmp_path / "b.json", cap_s=50.0, n_setup=1)
    b.record(_row(0, 0.0, 1.0, 41.0, setup_s=5.0))                       # within the cap
    b.record(_row(1, 100.0, 103.0, 155.4, setup_s=5.0, rc="timeout", timeout_s=55.0))   # killed at 55 s from launch
    b.record({"name": "refused", "rc": 75, "t_lock": None, "t_release": None})      # never held the lock
    s = b.summary()
    assert s["holds"] == 2 and s["killed"] == 1 and s["over_cap_holds"] == 1
    assert s["overshoot_max_s"] == pytest.approx(2.4) and s["overshoot_total_s"] == pytest.approx(2.4)
    assert b.holds[1]["wall_s"] == pytest.approx(52.4) and b.holds[1]["launch_to_release_s"] == pytest.approx(55.4)
    assert b.holds[1]["over_timeout_s"] == pytest.approx(0.4) and b.holds[1]["release_source"] == "driver_reap"
    assert s["over_timeout_max_s"] == pytest.approx(0.4)
    assert b.holds[0]["over_timeout_s"] == 0.0


def test_hold_bound_resume_refuses_a_different_sample_count(tmp_path) -> None:
    """Resume must refuse a ledger written with another n_setup (as it refuses another cap): the frozen p95 would
    otherwise be taken over a different number of samples."""
    b = FB.HoldBound(tmp_path / "b.json", cap_s=50.0, n_setup=3)
    b.record(_row(0, 0.0, 1.0, 20.0, setup_s=4.0))
    assert FB.HoldBound(tmp_path / "b.json", cap_s=50.0, n_setup=3).setup_s == [4.0]
    for n in (2, 4, 5):
        with pytest.raises(ValueError):
            FB.HoldBound(tmp_path / "b.json", cap_s=50.0, n_setup=n)
    with pytest.raises(ValueError):                                      # a capped ledger is not a record-only one
        FB.HoldBound(tmp_path / "b.json", cap_s=None, n_setup=3)


def test_record_only_bound_never_gives_an_uncapped_hold_c_cap_plus_p95(tmp_path) -> None:
    """M9's H2/H2+ holds are uncapped FAST runs with no c_cap. Their bound only
    RECORDS setup samples; the timeout stays the driver's backstop after the samples exist (the 6th hold included),
    and no over-cap is computed. The capped bound is made from it later and carries the frozen samples."""
    b = FB.HoldBound(tmp_path / "h2.json", cap_s=None)
    assert b.record_only and b.n_setup == FB.SETUP_SAMPLES == 5
    setups = [20.0, 25.0, 22.0, 40.0, 30.0, 300.0]
    for i, s in enumerate(setups):
        assert b.timeout_s() == FB.BACKSTOP_S
        b.record(_row(i, 0.0, 1.0, 600.0, setup_s=s), loop_s=600.0 - s)
    assert b.measured and b.setup_s == pytest.approx(setups[:5])          # frozen after 5; the 6th does not enter
    assert b.timeout_s() == FB.BACKSTOP_S                                   # NOT c_cap + p95
    assert all(h["over_cap_s"] is None for h in b.holds) and b.summary()["over_cap_holds"] == 0
    again = FB.HoldBound(tmp_path / "h2.json", cap_s=None)                  # resumable, still record-only
    assert again.timeout_s() == FB.BACKSTOP_S and len(again.holds) == 6
    cap = b.capped(tmp_path / "fb_bound.json", FB.CCap(120.0, "FROZEN", "test"))
    want = 120.0 + float(np.percentile(setups[:5], 95))
    assert cap.timeout_s() == pytest.approx(want) and cap.setup_from == str(tmp_path / "h2.json")
    cap.record(_row(9, 0.0, 1.0, 50.0, setup_s=999.0))                      # carried samples stay frozen
    assert FB.HoldBound(tmp_path / "fb_bound.json", cap_s=120.0).timeout_s() == pytest.approx(want)


def test_capped_bound_is_refused_before_setup_is_measured(tmp_path) -> None:
    b = FB.HoldBound(tmp_path / "h2.json", cap_s=None)
    for i in range(4):
        b.record(_row(i, 0.0, 1.0, 100.0, setup_s=10.0))
    frozen = FB.CCap(100.0, "FROZEN", "test")
    with pytest.raises(ValueError):
        b.capped(tmp_path / "fb.json", frozen)
    b.record(_row(4, 0.0, 1.0, 100.0, setup_s=10.0))
    with pytest.raises(ValueError):
        b.capped(tmp_path / "fb.json", None)  # type: ignore[arg-type]
    assert b.capped(tmp_path / "fb.json", frozen).timeout_s() == pytest.approx(110.0)
