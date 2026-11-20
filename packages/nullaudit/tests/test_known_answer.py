# Copyright 2026 The HandReel authors. SPDX-License-Identifier: Apache-2.0
"""Known answers on synthetic clips (see _ka.py for the closed forms)."""

from __future__ import annotations

import numpy as np
import pytest
from _ka import Impl, closed_form, run_ka, synthetic_clip

import nullaudit as na


def test_known_answer_suite_passes() -> None:
    assert run_ka(Impl()) == []


def test_static_null_known_answer() -> None:
    n, W, v, w = 1000, 400, 0.3 / 601, 1.2 / 601
    ref = synthetic_clip(n, W, v, w)
    r = na.DoAsIDoObjectTest(ref, required_rows=n)(na.static_at_start(ref))
    assert r.passed
    assert r.pos_err_m == pytest.approx(0.09, abs=1e-12) == pytest.approx(closed_form(n, W, v), abs=1e-12)
    assert r.rot_err_rad == pytest.approx(0.36, abs=1e-12)
    a = na.audit_clip(ref, na.DoAsIDoObjectTest(ref, required_rows=n), release_row=W)
    assert a.z0 and a.outcomes["hand_absent"].status == "not_run"
    assert a.outcomes["plan_then_freeze"].status == "not_run"


def test_sanity_nulls_on_a_moving_clip_fail() -> None:
    n, W = 1000, 400
    ref = synthetic_clip(n, W, 2e-3, 0.0)          # 0.6 m post-release mean: far outside any dilution
    a = na.audit_clip(ref, na.DoAsIDoObjectTest(ref, required_rows=n), release_row=W, nulls=na.KINEMATIC_NULLS)
    assert not a.z0 and a.reference_passes
    assert a.outcomes["reversed"].status == "fail"
    assert a.outcomes["time_shifted"].status == "fail"   # 150-row lag at 2 mm/row = 0.3 m post-release, diluted 0.18


def test_verdict_forms() -> None:
    assert na.verdict(True) is True and na.verdict(np.bool_(False)) is False
    assert na.verdict({"passed": True}) is True and na.verdict({"passed": np.bool_(False)}) is False

    class _R:
        def __init__(self, v: object) -> None:
            self.passed = v

    assert na.verdict(_R(True)) is True and na.verdict(_R(False)) is False
    for wrapped in ("false", "true", 1, 0, 0.5, None, np.float64(1.0)):       # truthy or falsy, none is a verdict
        with pytest.raises(TypeError, match="expected a bool"):
            na.verdict({"passed": wrapped})
        with pytest.raises(TypeError, match="expected a bool"):
            na.verdict(_R(wrapped))
    with pytest.raises(TypeError):
        na.verdict(0.5)


def test_cli(tmp_path) -> None:
    from nullaudit.__main__ import main

    n, W = 1000, 400
    np.savez(tmp_path / "z0.npz", reference=synthetic_clip(n, W, 0.3 / 601, 1.2 / 601), required_rows=n, release_row=W)
    np.savez(tmp_path / "z1.npz", reference=synthetic_clip(n, W, 2e-3, 0.0), required_rows=n, release_row=W)
    out = tmp_path / "r.json"
    assert main([str(tmp_path / "z0.npz"), str(tmp_path / "z1.npz"), "--json", str(out)]) == 0
    import json

    d = json.loads(out.read_text())
    assert d["z0_base_rate"] == "1/2" and d["rates"]["static_at_start"]["passed"] == 1
    assert [c["release_assumed"] for c in d["per_clip"]] == [False, False]


def _named(check: str) -> bool:
    return check not in run_ka(Impl())


@pytest.mark.parametrize("check", ["K12_invalid_rollout", "K13_ledger_length", "K14_unverified_counted",
                                   "K15_zero_quaternion"])
def test_regression_known_answers(check: str) -> None:
    """Fixes F1 (invalid rollouts never scored), F2 (ledger length), F3 (unverified passes labelled) and the
    zero-quaternion rule, each with its own closed-form check in _ka.py."""
    assert _named(check)


class _LedgerSim:
    """A simulator that returns a fixed trajectory and force ledger (for the N0 rule)."""

    def __init__(self, traj: np.ndarray, force: np.ndarray, status: str = "clip_end") -> None:
        self.traj, self.force, self.status = traj, force, status

    def __call__(self, controls: np.ndarray, *, hand_object_contact: bool = True) -> na.Rollout:
        return na.Rollout(self.traj, self.force, self.status)


@pytest.mark.parametrize("row", [0, 1, 4])
def test_n0_counts_every_post_release_row(row: int) -> None:
    """Force on a single row W + `row`, the first rows after release included, makes the hand-absent run invalid;
    force only before release does not (green control)."""
    n, W = 200, 50
    ref = synthetic_clip(n, W, 0.0, 0.0)
    test = na.DoAsIDoObjectTest(ref, required_rows=n)
    f = np.zeros(n)
    f[W + row] = 1.0
    o = na.audit_clip(ref, test, simulator=_LedgerSim(ref, f), controls=np.zeros((n, 1)), release_row=W,
                      nulls=("hand_absent",)).outcomes["hand_absent"]
    assert o.status == "invalid" and "1 post-release rows" in o.note
    f = np.zeros(n)
    f[W - 1] = 1.0
    o = na.audit_clip(ref, test, simulator=_LedgerSim(ref, f), controls=np.zeros((n, 1)), release_row=W,
                      nulls=("hand_absent",)).outcomes["hand_absent"]
    assert o.status == "pass" and o.verified_n0 is True


@pytest.mark.parametrize("why, make", [
    ("negative after release", lambda n, W: np.where(np.arange(n) == W + 3, -1.0, 0.0)),
    ("negative before release", lambda n, W: np.where(np.arange(n) == W - 3, -1.0, 0.0)),
    ("minus infinity", lambda n, W: np.where(np.arange(n) == W + 3, -np.inf, 0.0)),
    ("plus infinity", lambda n, W: np.where(np.arange(n) == W + 3, np.inf, 0.0)),
    ("not a number after release", lambda n, W: np.where(np.arange(n) == W + 3, np.nan, 0.0)),
    ("not a number before release", lambda n, W: np.where(np.arange(n) == W - 3, np.nan, 0.0)),
    ("n by 0", lambda n, W: np.zeros((n, 0))),
    ("n by 1", lambda n, W: np.zeros((n, 1))),
    ("n by 2", lambda n, W: np.zeros((n, 2))),
    ("a scalar", lambda n, W: np.float64(0.0)),
    ("a zero-dimensional array", lambda n, W: np.zeros(())),
    ("complex with a zero real part", lambda n, W: np.where(np.arange(n) == W + 3, 5j, 0j)),
    ("booleans", lambda n, W: np.zeros(n, dtype=bool)),
    ("strings", lambda n, W: np.array(["0"] * n)),
    ("objects", lambda n, W: np.array([None] * n, dtype=object)),
])
def test_n0_needs_one_finite_non_negative_force_per_row(why: str, make) -> None:
    """The ledger is one total normal force in N per row. A value that is negative or not finite, or an array of
    another shape, is not a ledger: the run is invalid and never a verified null, although its poses would pass."""
    n, W = 200, 50
    ref = synthetic_clip(n, W, 0.0, 0.0)
    test = na.DoAsIDoObjectTest(ref, required_rows=n)
    o = na.audit_clip(ref, test, simulator=_LedgerSim(ref, make(n, W)), controls=np.zeros((n, 1)), release_row=W,
                      nulls=("hand_absent",)).outcomes["hand_absent"]
    assert o.status == "invalid" and o.verified_n0 is False, why
    ok = na.audit_clip(ref, test, simulator=_LedgerSim(ref, np.zeros(n)), controls=np.zeros((n, 1)), release_row=W,
                       nulls=("hand_absent",)).outcomes["hand_absent"]
    assert ok.status == "pass" and ok.verified_n0 is True          # green control: the same poses, a valid ledger


def test_a_force_too_small_for_float64_still_counts_as_a_leak() -> None:
    """A ledger in a wider float type is checked as reported. Its smallest positive value rounds to zero in float64,
    so a check after narrowing would call the null verified."""
    tiny = np.nextafter(np.longdouble(0), np.longdouble(1))
    if float(tiny) != 0.0:
        pytest.skip("long double is not wider than float64 on this platform")
    n, W = 200, 50
    ref = synthetic_clip(n, W, 0.0, 0.0)
    test = na.DoAsIDoObjectTest(ref, required_rows=n)
    f = np.zeros(n, dtype=np.longdouble)
    f[W + 3] = tiny
    o = na.audit_clip(ref, test, simulator=_LedgerSim(ref, f), controls=np.zeros((n, 1)), release_row=W,
                      nulls=("hand_absent",)).outcomes["hand_absent"]
    assert o.status == "invalid" and "1 post-release rows" in o.note
    f[W + 3] = 0
    f[W - 3] = tiny                                             # before release: allowed, and still verified
    o = na.audit_clip(ref, test, simulator=_LedgerSim(ref, f), controls=np.zeros((n, 1)), release_row=W,
                      nulls=("hand_absent",)).outcomes["hand_absent"]
    assert o.status == "pass" and o.verified_n0 is True


def test_a_run_that_ends_before_release_is_not_a_verified_null() -> None:
    """A hand-absent run that stopped before the release row has no post-release row to check: invalid, not a
    vacuous N0 pass. A release row outside the reference is refused."""
    n, W = 200, 50
    ref = synthetic_clip(n, W, 0.0, 0.0)
    test = na.DoAsIDoObjectTest(ref)
    o = na.audit_clip(ref, test, simulator=_LedgerSim(ref[:W], np.zeros(W), "drop"), controls=np.zeros((n, 1)),
                      release_row=W, nulls=("hand_absent",)).outcomes["hand_absent"]
    assert o.status == "invalid" and o.verified_n0 is False and "release row" in o.note
    for bad in (-1, n):
        with pytest.raises(ValueError, match="release_row"):
            na.audit_clip(ref, test, release_row=bad)


def test_a_zero_reference_orientation_is_refused_before_any_verdict() -> None:
    """A zero reference quaternion has no orientation; atan2(0, 0) = 0 would read it as a perfect match. The
    component-wise midpoint of two equivalent raw signs, +q and -q, is such a row."""
    n = 9
    ref = synthetic_clip(n, 3, 0.0, 0.0)
    q = ref[0, 3:].copy()
    raw = np.stack([q, -q, -q])                                   # the same rotation three times, signs as stored
    assert np.allclose(0.5 * (raw[0] + raw[1]), 0.0)              # ... whose component midpoint is the zero vector
    bad = ref.copy()
    bad[4, 3:] = 0.5 * (raw[0] + raw[1])
    with pytest.raises(ValueError, match=r"orientation is undefined on 1 compared row\(s\) \(first rows \[4\]"):
        na.DoAsIDoObjectTest(bad, required_rows=n)(ref)
    nan = ref.copy()
    nan[2, 0] = np.nan
    with pytest.raises(ValueError, match="reference orientation is undefined"):
        na.DoAsIDoObjectTest(nan, required_rows=n)(ref)
    assert na.DoAsIDoObjectTest(bad, required_rows=4)(ref[:4]).passed          # the bad row is not compared: no refusal
    assert na.DoAsIDoObjectTest(ref, required_rows=n)(ref).passed


def test_a_matrix_that_is_not_a_proper_rotation_has_no_quaternion() -> None:
    from nullaudit.poses import ROTATION_TOL, as_pose7, matrix_to_quat

    eye = np.eye(3)[None]
    assert np.allclose(matrix_to_quat(eye), [[1.0, 0.0, 0.0, 0.0]])
    c, s = np.cos(0.3), np.sin(0.3)
    rz = np.array([[[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]])
    assert np.allclose(matrix_to_quat(rz), [[np.cos(0.15), 0.0, 0.0, np.sin(0.15)]])
    assert np.allclose(matrix_to_quat(rz.astype(np.float32)), [[np.cos(0.15), 0.0, 0.0, np.sin(0.15)]], atol=1e-6)
    for name, m in (("zero block", np.zeros((1, 3, 3))), ("reflection", np.diag([1.0, 1.0, -1.0])[None]),
                    ("scaled", 1.01 * eye), ("sheared", eye + np.array([[[0.0, 0.01, 0.0]] * 3]) * [[1], [0], [0]]),
                    ("non-finite", np.where(np.eye(3) > 0, np.nan, 0.0)[None])):
        with pytest.raises(ValueError, match="not proper rotations"):
            matrix_to_quat(m)
        assert name
    with pytest.raises(ValueError, match=r"first rows \[1\]"):
        matrix_to_quat(np.concatenate([eye, np.zeros((1, 3, 3)), rz]))
    pose = np.tile(np.eye(4), (2, 1, 1))
    pose[1, :3, :3] = 0.0                                          # a zero rotation block used to become identity
    with pytest.raises(ValueError, match="not proper rotations"):
        as_pose7(pose)
    with pytest.raises(ValueError, match=r"shape \(T, 3, 3\)"):
        matrix_to_quat(np.eye(3))
    assert matrix_to_quat((1.0 + 0.5 * ROTATION_TOL / 3) * eye).shape == (1, 4)      # inside the tolerance


def test_a_release_at_the_first_row_leaves_n1_not_run_and_keeps_the_other_outcomes() -> None:
    """W = 0: no command precedes the release, so plan-then-freeze is undefined. It is reported as not run with
    its reason; the audit does not raise and the other nulls keep their outcomes."""
    n = 40
    ref = synthetic_clip(n, 0, 0.001, 0.0)
    test = na.DoAsIDoObjectTest(ref, required_rows=n)
    a = na.audit_clip(ref, test, simulator=_LedgerSim(ref, np.zeros(n)), controls=np.zeros((n, 1)), release_row=0)
    o = a.outcomes["plan_then_freeze"]
    assert (o.status, o.note) == ("not_run", "release at row 0: no command before the release to freeze")
    assert a.outcomes["hand_absent"].status in ("pass", "fail", "invalid") and "static_at_start" in a.outcomes
    with pytest.raises(ValueError):
        na.plan_then_freeze_controls(np.zeros((n, 1)), 0)           # the helper itself still refuses row 0
    b = na.audit_clip(ref, test, simulator=_LedgerSim(ref, np.zeros(n)), controls=np.zeros((n, 1)), release_row=1)
    assert b.outcomes["plan_then_freeze"].status != "not_run"


def test_a_run_without_a_force_ledger_must_still_reach_the_release() -> None:
    """A four-row rollout that matches the reference prefix, release at row 4, longer reference, NO force reported:
    the run never reached the release, so the hand-absent null is invalid, not an unverified pass."""
    n, W = 40, 4
    ref = synthetic_clip(n, W, 0.0, 0.0)
    test = na.DoAsIDoObjectTest(ref)                              # the prefix test: no required row count

    class _Short:
        def __call__(self, controls: np.ndarray, *, hand_object_contact: bool = True) -> na.Rollout:
            return na.Rollout(ref[:W].copy(), None, "ok")

    o = na.audit_clip(ref, test, simulator=_Short(), controls=np.zeros((n, 1)), release_row=W,
                      nulls=("hand_absent",)).outcomes["hand_absent"]
    assert o.status == "invalid" and "release row 4 outside the run's rows [0, 4)" in o.note and o.verified_n0 is False

    class _Reaches:
        def __call__(self, controls: np.ndarray, *, hand_object_contact: bool = True) -> na.Rollout:
            return na.Rollout(ref[:W + 1].copy(), None, "ok")

    o = na.audit_clip(ref, test, simulator=_Reaches(), controls=np.zeros((n, 1)), release_row=W,
                      nulls=("hand_absent",)).outcomes["hand_absent"]
    assert o.status != "invalid" and "N0 unverified" in o.note     # reached: scored, and labelled unverified


def test_the_cli_refuses_row_counts_that_are_not_whole_numbers(tmp_path) -> None:
    from nullaudit.__main__ import _load

    ref = synthetic_clip(20, 5, 0.0, 0.0)
    for key, bad in (("required_rows", 4.5), ("required_rows", True), ("release_row", -1), ("release_row", np.nan),
                     ("required_rows", np.array([4, 5])), ("release_row", "4")):
        p = tmp_path / "bad.npz"
        np.savez(p, reference=ref, **{key: bad})
        with pytest.raises(ValueError, match="whole non-negative number"):
            _load(p)
    p = tmp_path / "ok.npz"
    np.savez(p, reference=ref, required_rows=4.0, release_row=np.int64(5))
    got = _load(p)
    assert (got[1], got[2]) == (4, 5) and isinstance(got[1], int)


def test_an_empty_execution_fails_in_both_representations() -> None:
    """(0, 7) and (0, 4, 4) are the same empty run: both reach the verdict, and both fail it."""
    ref = synthetic_clip(10, 3, 0.0, 0.0)
    test = na.DoAsIDoObjectTest(ref, required_rows=10)
    for empty in (np.zeros((0, 7)), np.zeros((0, 4, 4))):
        res = test(empty)
        assert not res.passed and res.executed_rows == 0 and not res.full


class _StopsAt:
    """Replays the reference and stops after `rows` rows with an ordinary (not invalid) status."""

    def __init__(self, ref: np.ndarray, rows: int) -> None:
        self.ref, self.rows, self.sent = ref, rows, []

    def __call__(self, controls: np.ndarray, *, hand_object_contact: bool = True) -> na.Rollout:
        self.sent.append(np.array(controls))
        return na.Rollout(self.ref[:self.rows].copy(), np.zeros(self.rows), "drop")


@pytest.mark.parametrize("rows", [1, 3, 4])
def test_a_freeze_run_that_stops_before_the_release_is_unevaluated_not_a_pass(rows: int) -> None:
    """Release at row 4, prefix-scoring test, rollout of at most 4 rows that matches the reference: every executed
    row is a planned row. The prefix passes the test, and that is kept in the detail; N1 itself is invalid."""
    n, W = 40, 4
    ref = synthetic_clip(n, W, 0.0, 0.0)
    test = na.DoAsIDoObjectTest(ref)                              # no required row count: a prefix can pass
    sim = _StopsAt(ref, rows)
    o = na.audit_clip(ref, test, simulator=sim, controls=np.arange(n, dtype=float)[:, None], release_row=W,
                      nulls=("plan_then_freeze",)).outcomes["plan_then_freeze"]
    assert o.status == "invalid" and not o.passed
    assert f"release row 4 outside the run's rows [0, {rows})" in o.note and "N1 is unevaluated" in o.note
    assert o.detail == {"sim_status": "drop", "executed_rows": rows, "prefix_passed": True}
    assert np.array_equal(sim.sent[0][W:, 0], np.full(n - W, W - 1.0))        # the frozen command was what was sent
    rep = na.summarize([na.audit_clip(ref, test, simulator=_StopsAt(ref, rows), controls=np.zeros((n, 1)),
                                      release_row=W, nulls=("plan_then_freeze",))])
    assert rep.to_dict()["rates"]["plan_then_freeze"]["passed"] == 0


def test_a_freeze_run_that_executes_one_frozen_row_is_scored() -> None:
    n, W = 40, 4
    ref = synthetic_clip(n, W, 0.0, 0.0)
    o = na.audit_clip(ref, na.DoAsIDoObjectTest(ref), simulator=_StopsAt(ref, W + 1), controls=np.zeros((n, 1)),
                      release_row=W, nulls=("plan_then_freeze",)).outcomes["plan_then_freeze"]
    assert o.status == "pass" and o.detail["sim_status"] == "drop" and "prefix_passed" not in o.detail


def test_controls_that_end_at_the_release_leave_n1_not_run() -> None:
    n, W = 40, 4
    ref = synthetic_clip(n, W, 0.0, 0.0)
    sim = _StopsAt(ref, n)
    o = na.audit_clip(ref, na.DoAsIDoObjectTest(ref), simulator=sim, controls=np.zeros((W, 1)), release_row=W,
                      nulls=("plan_then_freeze",)).outcomes["plan_then_freeze"]
    assert o.status == "not_run" and "controls end at row 4" in o.note and sim.sent == []


def test_the_cli_takes_the_release_from_the_grid_and_marks_an_assumed_one(tmp_path, capsys) -> None:
    import json

    from nullaudit.__main__ import _load, main

    frames = synthetic_clip(8, 0, 0.01, 0.0)
    grid = dict(frames=frames, ref_dt=0.1, sim_dt=0.1, warmup_rows=12, n_rows=20)
    p = tmp_path / "grid.npz"
    np.savez(p, **grid)
    ref, req, rel, assumed = _load(p)
    assert (ref.shape, req, rel, assumed) == ((20, 7), None, 12, False)       # W = warm-up rows, not row 0
    np.savez(p, **grid, release_row=12)
    assert _load(p)[2:] == (12, False)
    np.savez(p, **grid, release_row=5)
    with pytest.raises(ValueError, match="`release_row` 5 differs from `warmup_rows` 12"):
        _load(p)
    for key, bad in (("warmup_rows", 12.5), ("n_rows", 20.9), ("n_rows", True), ("warmup_rows", -1),
                     ("n_rows", np.nan)):
        np.savez(p, **{**grid, key: bad})
        with pytest.raises(ValueError, match="whole non-negative number"):
            _load(p)
    for key, bad in (("ref_dt", 0.0), ("sim_dt", -0.1), ("ref_dt", np.inf), ("sim_dt", np.nan), ("sim_dt", True)):
        np.savez(p, **{**grid, key: bad})
        with pytest.raises(ValueError, match="positive finite number of seconds"):
            _load(p)
    np.savez(p, **grid)
    q = tmp_path / "pregridded.npz"
    np.savez(q, reference=ref)
    assert _load(q)[2:] == (None, True)
    out = tmp_path / "r.json"
    assert main([str(p), str(q), "--json", str(out)]) == 0
    assert "release row not stored for 1 of 2 clips: assumed at row 0" in capsys.readouterr().out
    assert [c["release_assumed"] for c in json.loads(out.read_text())["per_clip"]] == [False, True]


def test_the_default_lag_is_a_quarter_of_the_post_release_rows_of_a_frame_grid(tmp_path) -> None:
    from nullaudit.__main__ import _load

    p = tmp_path / "grid.npz"
    np.savez(p, frames=synthetic_clip(8, 0, 0.01, 0.0), ref_dt=0.1, sim_dt=0.1, warmup_rows=12, n_rows=20)
    ref, req, rel, _ = _load(p)
    a = na.audit_clip(ref, na.DoAsIDoObjectTest(ref, required_rows=req), release_row=rel, nulls=na.KINEMATIC_NULLS)
    assert a.outcomes["time_shifted"].note == "shift 2 rows"                  # (20 - 12) // 4, not 20 // 4

