"""The execution gate's simulated evidence.

- the object must rest where it is placed (settle <= 5 mm, <= 2 deg);
- the run simulated at the package's speed scale must itself move the object as the reference;
- the object placement tolerance (+-1 cm x / y, +-5 deg yaw) is simulated at every corner;
- the measured states `--execute` accepts are simulated (arm-box axis extremes and seeded corners, Dex3 at +-0.1 rad);
- every simulated run records its run options, `--no-waist` etc. are refused on a 0.5 package, and
  `--max-track-err` / `--hold-timeout` are accepted only at or below the simulated values;
- every package is a DRAFT, and loading its run script leaves no `__pycache__` behind.
Gate tests edit a copy of the synthetic package's `sim_stats.json` (`tests/robot/test_sim_gate.with_replay`)."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from fastdo.robot import g1_sdk_replay as S
from tests.robot.test_sim_gate import OBJ_PASS, _home_rec, _no_input, with_replay


def edited(pkg: Path, tmp_path: Path, edit) -> Path:
    """`with_replay` (everything passes) with `edit(stats)` applied to its `sim_stats.json`."""
    out = with_replay(pkg, tmp_path)
    stats = json.loads((out / "sim_stats.json").read_text())
    edit(stats)
    (out / "sim_stats.json").write_text(json.dumps(stats))
    return out


def refused_once(pkg: Path, needle: str) -> None:
    rep = S.gate_report(pkg)
    assert rep["homes_allowed"] == [] and len(rep["failures"]) == 1, rep["failures"]
    assert rep["failures"][0].startswith("no checked home passes") and needle in rep["failures"][0], rep["failures"]


def test_with_replay_passes_the_whole_gate(built, tmp_path: Path) -> None:
    """Control for every refusal below: the synthetic package with passing object criteria passes everything."""
    rep = S.gate_report(with_replay(built[3], tmp_path))
    assert rep["failures"] == [] and rep["homes_allowed"] == [0]


# --- sim settle gate ---------------------------------------------------------------

@pytest.mark.parametrize(("key", "value", "needle"), [
    ("settle_drift", 0.0051, "moved 5.1 mm while settling"),
    ("settle_rot_deg", 2.01, "turned 2.01 deg while settling"),
    ("settle_drift", None, "while settling"),                  # a missing metric fails (NaN)
    ("settle_rot_deg", None, "while settling"),
])
def test_an_object_that_does_not_rest_refuses(built, tmp_path: Path, key: str, value: object, needle: str) -> None:
    fails = S.sim_gate_failures(with_replay(built[3], tmp_path, **{key: value}))
    assert len(fails) == 1 and needle in fails[0], fails


def test_settle_boundary_values_pass(built, tmp_path: Path) -> None:
    assert S.SIM_GATE["settle_drift_max_m"] == 0.005 and S.SIM_GATE["settle_rot_max_deg"] == 2.0
    assert S.sim_gate_failures(with_replay(built[3], tmp_path, settle_drift=0.005, settle_rot_deg=2.0)) == []


def test_the_synthetic_replay_records_its_settle(built) -> None:
    rp = json.loads((built[3] / "sim_stats.json").read_text())["replay"]
    assert 0.0 < rp["settle_drift"] < 0.005 and 0.0 <= rp["settle_rot_deg"] < 2.0


# --- the object criteria of the run itself, at the package's speed scale ------------------------------

def test_the_run_records_its_object_criteria(built) -> None:
    """The synthetic hand never touches the box: the run (speed scale 1) moves it 0 cm against the reference's 10 cm
    (the replay's own reference displacement), ends ~10 cm from the reference end and does not fall."""
    stats = json.loads((built[3] / "sim_stats.json").read_text())
    crit = _home_rec(stats)["run"]["object_criteria"]
    assert crit["reference_displacement_m"] == pytest.approx(stats["replay"]["reference_displacement"], abs=1e-9)
    assert crit["displacement_m"] < 0.002 and crit["fell"] is False
    assert crit["final_error_m"] == pytest.approx(stats["replay"]["object_final_error"], abs=0.002)


@pytest.mark.parametrize(("edit", "needle"), [
    ({"displacement_m": 0.049}, "object moved 4.9 cm"),
    ({"final_error_m": 0.051}, "object final error 5.1 cm"),
    ({"fell": True}, "the object fell"),
    ({"reference_displacement_m": 0.019}, "the reference moves the object only"),
    (None, "not recorded"),
])
def test_a_run_failing_the_object_criteria_refuses(built, tmp_path: Path, edit: dict | None, needle: str) -> None:
    """The trajectory replay passes; the run simulated at the package's speed scale does not move the object as the
    reference: the home is refused (a 0.5 package must pass the object criteria simulated at 0.5)."""
    def change(stats: dict) -> None:
        run = _home_rec(stats)["run"]
        if edit is None:
            run.pop("object_criteria")
        else:
            run["object_criteria"].update(edit)
    refused_once(edited(built[3], tmp_path, change), needle)
    assert "the simulated run (speed scale 1)" in S.gate_report(edited(built[3], tmp_path / "b", change))["failures"][0]


# --- the object placement corners ------------------------------------------------------------------------

def test_the_package_simulates_the_object_placement_corners(built) -> None:
    """Every corner of +-1 cm x / y and +-5 deg yaw is simulated and clean outside the playback; the planner's
    clearance reduction (the 6 cm box: sqrt(2) cm + 2 x 4.24 cm x sin 2.5 deg = 17.8 mm) and the tolerance are in
    sim_stats.json, package.json and the README."""
    stats = json.loads((built[3] / "sim_stats.json").read_text())
    meta = json.loads((built[3] / "package.json").read_text())
    tr = stats["transfers"]
    assert (tr["object_xy_tol_m"], tr["object_yaw_tol_deg"]) == (0.01, 5.0)
    assert tr["object_envelope_m"] == pytest.approx(np.sqrt(2) * 0.01 + 2 * 0.03 * np.sqrt(2) * np.sin(np.radians(2.5)),
                                                    abs=1e-6)
    assert meta["object_placement_tolerance"]["xy_m"] == 0.01 and meta["object_placement_tolerance"]["yaw_deg"] == 5.0
    runs = {r["case"]: r for r in _home_rec(stats)["placement_runs"]}
    assert set(runs) == set(S.PLACEMENT_CASES) and len(runs) == 8
    from scipy.spatial.transform import Rotation

    from fastdo.robot.tolerance_runs import placement_pose
    nominal = np.eye(4)
    nominal[:3, 3] = meta["object_start_robot"]["position_m"]
    nominal[:3, :3] = Rotation.from_quat(np.roll(meta["object_start_robot"]["quat_wxyz"], -1)).as_matrix()
    for case, (sx, sy, sa) in S.PLACEMENT_CASES.items():
        r = runs[case]
        assert (r["dx_m"], r["dy_m"], r["dyaw_deg"]) == pytest.approx((0.01 * sx, 0.01 * sy, 5.0 * sa))
        assert np.allclose(r["object_start"], placement_pose(nominal, 0.01 * sx, 0.01 * sy, 5.0 * sa), atol=1e-9)
        settled = np.asarray(r["object"]["play_start_m"])                   # where the simulated box actually stood
        assert np.allclose(settled[:2], nominal[:2, 3] + [0.01 * sx, 0.01 * sy], atol=0.002), (case, settled)
        assert r["status"] == "completed" and r["started_home"] == 0
        assert all(v["table_contact_steps"] == 0 and v["object_contact_steps"] == 0
                   for k, v in r["phases"].items() if k != "play")
    readme = (built[3] / "README.md").read_text()
    assert "x, y and yaw are NOT simulated" not in readme
    assert "+-1.0 cm in x and y, +-5 deg in yaw" in readme and "17.8 mm" in readme


def _corner(stats: dict, case: str = "x+y-yaw+") -> dict:
    return next(r for r in _home_rec(stats)["placement_runs"] if r["case"] == case)


@pytest.mark.parametrize(("edit", "needle"), [
    (lambda s: _corner(s)["phases"]["to_hold"].update(table_contact_steps=1), "touches the table"),
    (lambda s: _corner(s)["phases"]["wait_go"].update(object_contact_steps=2), "touches the object"),
    (lambda s: _corner(s)["phases"]["play"].update(table_max_force=20.1), "force"),
    (lambda s: _corner(s).update(status="aborted"), "did not complete"),
    (lambda s: _corner(s)["object_criteria"].update(fell=True), "fell"),
    (lambda s: _corner(s)["object_criteria"].update(final_error_m=0.06), "final error"),
    (lambda s: _corner(s).update(dyaw_deg=2.5), "not (0.01, -0.01, 5.0)"),
    (lambda s: _corner(s).update(started_home=1), "started from home #1"),
    (lambda s: _home_rec(s)["placement_runs"].remove(_corner(s)), "x+y-yaw+: not simulated"),
    (lambda s: _home_rec(s).pop("placement_runs"), "not simulated"),
])
def test_a_failing_placement_corner_refuses(built, tmp_path: Path, edit, needle: str) -> None:
    """Each corner alone refuses the home (mutation: the gate reading the nominal run only)."""
    refused_once(edited(built[3], tmp_path, edit), needle)


def test_a_placement_tolerance_below_the_ruled_one_is_refused(built, tmp_path: Path) -> None:
    for i, (xy, yaw) in enumerate(((0.005, 5.0), (0.01, 2.0), (0.0, 0.0))):
        def change(stats: dict, xy=xy, yaw=yaw) -> None:
            stats["transfers"].update(object_xy_tol_m=xy, object_yaw_tol_deg=yaw)
        pkg = edited(built[3], tmp_path / str(i), change)
        meta = json.loads((pkg / "package.json").read_text())
        meta["object_placement_tolerance"].update(xy_m=xy, yaw_deg=yaw)
        (pkg / "package.json").write_text(json.dumps(meta))
        refused_once(pkg, "object placement tolerance")


def test_package_json_must_state_the_simulated_object_tolerance(built, tmp_path: Path) -> None:
    pkg = with_replay(built[3], tmp_path)
    meta = json.loads((pkg / "package.json").read_text())
    meta["object_placement_tolerance"]["xy_m"] = 0.02
    (pkg / "package.json").write_text(json.dumps(meta))
    fails = S.sim_gate_failures(pkg)
    assert len(fails) == 1 and "allows the object +-2.0 cm" in fails[0], fails


# --- the measured states --execute accepts are simulated ----------------------------------------

def test_the_package_simulates_the_measured_states(built) -> None:
    """28 axis extremes and 4 seeded corners of the 0.08 rad arm box (0.999 of it out), and the Dex3 hand at +-0.1
    rad (0.999 of it, clipped at a joint's range limit), each completed from home #0, clean outside the playback."""
    stats = json.loads((built[3] / "sim_stats.json").read_text())
    runs = _home_rec(stats)["state_runs"]
    assert [r["name"] for r in runs] == S.state_run_names(4) and stats["transfers"]["state_corners"] == 4
    for r in runs:
        arm, hand = np.asarray(r["arm_offset"]), np.asarray(r["hand_offset"])
        if r["name"].startswith("arm_axis_"):
            j = int(r["name"][9:11])
            assert abs(arm[j]) == pytest.approx(0.999 * 0.08) and np.count_nonzero(arm) == 1 and not hand.any()
        elif r["name"].startswith("arm_corner_"):
            assert np.allclose(np.abs(arm), 0.999 * 0.08) and not hand.any()
        else:
            free = [k for k in range(7) if f"hand {k}" not in r["clipped"]]
            assert free and np.allclose(np.abs(hand[free]), 0.0999) and not arm.any() and r["hand_pinned_at_start"]
        assert r["status"] == "completed" and r["started_home"] == 0, (r["name"], r.get("reason"))
        measured = np.asarray(r["measured_start"]["q17"])[:14] - np.asarray(_home_rec(stats)["home_q"])[:14]
        # the script measured the start it was given (holding the hand at its edge moves the arm < 1 mrad)
        assert np.abs(measured - arm).max() < (1e-3 if r["name"].startswith("hand") else 1e-4), r["name"]
        if r["name"].startswith("hand"):
            assert np.allclose(r["measured_start"]["hand"], hand, atol=1e-12)
        assert all(v["table_contact_steps"] == 0 and v["object_contact_steps"] == 0
                   for k, v in r["phases"].items() if k != "play"), r["name"]


def _state(stats: dict, name: str) -> dict:
    return next(r for r in _home_rec(stats)["state_runs"] if r["name"] == name)


def _shift(run: dict, key: str, k: int, by: float) -> None:
    run["measured_start"][key][k] += by


@pytest.mark.parametrize(("edit", "needle"), [
    (lambda s: _state(s, "arm_axis_03+")["phases"]["to_hold"].update(table_contact_steps=1), "touches the table"),
    (lambda s: _state(s, "arm_corner_2")["phases"]["go_home"].update(object_contact_steps=1), "touches the object"),
    (lambda s: _state(s, "hand+").update(status="refused"), "hand+: the simulated run did not complete"),
    (lambda s: _home_rec(s)["state_runs"].remove(_state(s, "arm_corner_3")), "arm_corner_3: not simulated"),
    (lambda s: _home_rec(s)["state_runs"].remove(_state(s, "arm_axis_13-")), "arm_axis_13-: not simulated"),
    (lambda s: _shift(_state(s, "arm_axis_00+"), "q17", 0, -0.04), "edge of the tolerance"),       # measured at half
    (lambda s: _shift(_state(s, "arm_axis_00+"), "q17", 4, 0.009), "edge of the tolerance"),        # another joint off
    (lambda s: _shift(_state(s, "hand+"), "q17", 2, -0.009), "edge of the tolerance"),               # the arm off
    (lambda s: _shift(_state(s, "arm_corner_1"), "q17", 9, -0.002 * np.sign(_state(s, "arm_corner_1")
                                                                            ["arm_offset"][9])), "edge of the tolerance"),
    (lambda s: _state(s, "hand-")["measured_start"].update(hand=[-0.05] * 7), "edge of the tolerance"),
    (lambda s: _state(s, "hand+").update(clipped=[]) or _state(s, "hand+")["measured_start"].update(
        hand=[0.0999] * 6 + [0.05]), "edge of the tolerance"),
    (lambda s: _state(s, "arm_corner_0").update(home_tol=0.04), "not recorded as simulated"),
    (lambda s: _state(s, "arm_corner_0").pop("measured_start"), "not recorded as simulated"),
    (lambda s: _state(s, "arm_axis_05-").update(started_home=1), "started from home #1"),
    (lambda s: _home_rec(s).pop("state_runs"), "not simulated"),
])
def test_a_failing_measured_state_refuses(built, tmp_path: Path, edit, needle: str) -> None:
    refused_once(edited(built[3], tmp_path, edit), needle)


# --- every simulated run records the run options it used -------------------------------------------

@pytest.mark.parametrize(("edit", "needle"), [
    (lambda s: _home_rec(s)["drills"][1]["run_options"].update(speed_scale=0.5), "simulated with speed_scale 0.5"),
    (lambda s: _home_rec(s)["table_tolerance_runs"][0]["run_options"].update(hold_speed=0.2), "hold_speed 0.2"),
    (lambda s: _state(s, "hand-")["run_options"].update(max_track_err=0.25), "max_track_err 0.25"),
    (lambda s: _corner(s)["run_options"].update(hold_waist=False), "hold_waist False"),
    (lambda s: _home_rec(s)["run"]["run_options"].update(send_tau_ff=False), "send_tau_ff False"),
    (lambda s: _corner(s).pop("run_options"), "run options it was simulated with are not recorded"),
])
def test_a_run_simulated_with_other_options_refuses(built, tmp_path: Path, edit, needle: str) -> None:
    refused_once(edited(built[3], tmp_path, edit), needle)


# --- a bring-up (0.5) package refuses unsimulated options ------------------------------------------

def half_package(pkg: Path, tmp_path: Path) -> Path:
    """The passing synthetic package as if exported at speed scale 0.5 (every run records 0.5)."""
    def change(stats: dict) -> None:
        stats["transfers"]["speed_scale"] = 0.5
        for rec in stats["transfers"]["homes"]:
            for key in ("drills", "table_tolerance_runs", "placement_runs", "state_runs"):
                for r in rec.get(key) or []:
                    r["run_options"]["speed_scale"] = 0.5
            if rec.get("run"):
                rec["run"]["run_options"]["speed_scale"] = 0.5
    return edited(pkg, tmp_path, change)


@pytest.mark.parametrize(("flags", "needle"), [
    (["--no-waist"], "--no-waist"),
    (["--no-tau-ff"], "--no-tau-ff"),
    (["--hold-speed", "0.2"], "--hold-speed"),
    (["--speed-scale", "1"], "--speed-scale"),
])
def test_execute_refuses_unsimulated_options_on_a_bring_up_package(built, tmp_path: Path,
                                                                   monkeypatch: pytest.MonkeyPatch,
                                                                   capsys: pytest.CaptureFixture[str], flags: list[str],
                                                                   needle: str) -> None:
    pkg = half_package(built[3], tmp_path)
    prompts = _no_input(monkeypatch)
    assert S.main(["--package", str(pkg), "--execute", "--iface", "lo"]) == 1 and len(prompts) == 1   # control
    capsys.readouterr()
    assert S.main(["--package", str(pkg), "--execute", "--iface", "lo", *flags]) == 4
    err = capsys.readouterr().err
    assert len(prompts) == 1 and "REFUSED" in err and needle in err, err


# --- --max-track-err / --hold-timeout only at or below the simulated values ---------------------------

def test_execute_accepts_tracking_and_hold_limits_only_at_or_below_the_simulated(built, tmp_path: Path,
                                                                                monkeypatch: pytest.MonkeyPatch,
                                                                                capsys: pytest.CaptureFixture[str]) -> None:
    pkg = with_replay(built[3], tmp_path)
    tr = json.loads((pkg / "sim_stats.json").read_text())["transfers"]
    assert (tr["max_track_err_rad"], tr["hold_timeout_s"]) == (0.3, 120.0)
    prompts = _no_input(monkeypatch)
    base = ["--package", str(pkg), "--execute", "--iface", "lo"]
    for extra in (["--max-track-err", "0.2"], ["--hold-timeout", "60"], ["--hold-timeout", "120"]):
        assert S.main(base + extra) == 1, extra                                     # reaches the connect prompt
    assert len(prompts) == 3
    assert S.main(base + ["--hold-timeout", "121"]) == 4 and "--hold-timeout" in capsys.readouterr().err

    def tighter(stats: dict) -> None:                                               # a package simulated at 0.2 rad
        stats["transfers"]["max_track_err_rad"] = 0.2
        for rec in stats["transfers"]["homes"]:
            for r in [rec.get("run"), *(rec.get("drills") or []), *(rec.get("table_tolerance_runs") or []),
                      *(rec.get("placement_runs") or []), *(rec.get("state_runs") or [])]:
                if r:
                    r["run_options"]["max_track_err"] = 0.2
    pkg2 = edited(built[3], tmp_path / "t", tighter)
    assert S.gate_report(pkg2)["failures"] == []
    capsys.readouterr()
    assert S.main(["--package", str(pkg2), "--execute", "--iface", "lo"]) == 4          # the default 0.3 > 0.2
    assert "--max-track-err" in capsys.readouterr().err
    assert S.main(["--package", str(pkg2), "--execute", "--iface", "lo", "--max-track-err", "0.2"]) == 1
    assert len(prompts) == 4
    pkg3 = edited(built[3], tmp_path / "m", lambda s: s["transfers"].pop("hold_timeout_s"))
    assert S.main(["--package", str(pkg3), "--execute", "--iface", "lo"]) == 4 and len(prompts) == 4


# --- review minors: DRAFT markers, no stray __pycache__ ---------------------------------------------------------------

def test_every_package_is_a_draft(built) -> None:
    from fastdo.robot.package import DRAFT_MARKER
    readme = (built[3] / "README.md").read_text()
    assert readme.splitlines()[2] == f"**{DRAFT_MARKER}**" and "DRAFT: NOT lab-ready" in DRAFT_MARKER
    assert json.loads((built[3] / "package.json").read_text())["lab_status"] == DRAFT_MARKER
    assert "(`--dry-run`)" not in readme and "(the default dry-run)" in readme


def test_loading_the_run_script_leaves_no_bytecode(built, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`package.load_run_script` imports a package's run script without writing `__pycache__` into it (the earlier
    drafts each got one from a check that imported the script); `stray_files` names such leftovers (control: a plain
    import with bytecode writing on leaves one)."""
    from fastdo.robot.package import load_run_script, stray_files
    monkeypatch.setattr(sys, "dont_write_bytecode", False)
    pkg = tmp_path / "copy"
    shutil.copytree(built[3], pkg)
    assert stray_files(pkg) == []
    mod = load_run_script(pkg)
    assert mod.SIM_GATE == S.SIM_GATE and stray_files(pkg) == [] and sys.dont_write_bytecode is False
    ctrl = tmp_path / "ctrl"
    shutil.copytree(built[3], ctrl)
    spec = importlib.util.spec_from_file_location("run_g1_replay_ctrl", ctrl / "run_g1_replay.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_g1_replay_ctrl"] = module
    spec.loader.exec_module(module)
    assert stray_files(ctrl) == ["__pycache__"]


def test_obj_pass_is_the_replay_threshold_shape() -> None:
    """The helper's passing values are inside the registered thresholds (10 cm reference, 9 cm moved, 1 cm error)."""
    g = S.SIM_GATE
    assert OBJ_PASS["displacement_m"] >= g["min_move_fraction"] * OBJ_PASS["reference_displacement_m"]
    assert S._object_failures("x", OBJ_PASS) == []
