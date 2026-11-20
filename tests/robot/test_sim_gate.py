"""The execution gate of the package run script: `--execute` refuses a package whose own
simulated replay of ITS trajectory fails a registered threshold (`g1_sdk_replay.SIM_GATE`), before anything connects
to the robot; the thresholds live in the run script, `package.json` and the README."""

from __future__ import annotations

import builtins
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from fastdo.robot import g1_sdk_replay as S

PASSING = {"table_max_depth": 0.001, "self_collision_max_depth": 0.0, "limit_violation_ticks": 0, "arm_max": 0.05,
           "reference_displacement": 0.10, "object_displacement": 0.09, "object_final_error": 0.01,
           "object_fell": False, "settle_drift": 0.001, "settle_rot_deg": 0.1}
# the object criteria of the run itself and of the placement corners, passing
OBJ_PASS = {"reference_displacement_m": 0.10, "displacement_m": 0.09, "final_error_m": 0.01, "fell": False}


def pass_objects(stats: dict) -> dict:
    """`stats` with the object criteria of every home's run and placement corners set to `OBJ_PASS` (the synthetic
    hand never touches the box)."""
    for rec in stats["transfers"]["homes"]:
        if isinstance(rec.get("run"), dict):
            rec["run"]["object_criteria"] = dict(OBJ_PASS)
        for run in rec.get("placement_runs") or []:
            run["object_criteria"] = dict(OBJ_PASS)
    return stats


def with_replay(pkg: Path, tmp_path: Path, **overrides: object) -> Path:
    """A copy of `pkg` whose sim_stats replay metrics are `PASSING` updated by `overrides` (None = key removed), and
    whose runs' object criteria pass (`pass_objects`)."""
    out = tmp_path / "pkg"
    shutil.copytree(pkg, out)
    stats = pass_objects(json.loads((out / "sim_stats.json").read_text()))
    stats["replay"].update(PASSING)
    for key, value in overrides.items():
        if value is None:
            stats["replay"].pop(key)
        else:
            stats["replay"][key] = value
    (out / "sim_stats.json").write_text(json.dumps(stats))
    return out


def test_the_synthetic_package_fails_the_gate(built) -> None:
    """Known answer: the synthetic hand never touches the box (the reference slides it 10 cm), so the gate fails on
    the object, and only on the object (the trajectory hash matches): in the replay, and in the home's run
    at the package's speed scale and at every object placement corner."""
    rep = S.gate_report(built[3])
    failures = rep["failures"]
    assert failures[:2] == ["object moved 0.0 cm < 0.5 x reference 10.0 cm", "object final error 10.0 cm"], failures
    assert len(failures) == 3 and failures[2].startswith("no checked home passes")
    home0 = [f for f in rep["home_failures"] if f.startswith("home #0")]
    assert home0 and all(("object moved" in f or "object final error" in f) for f in home0), home0
    assert sum("placement corner" in f for f in home0) == 2 * 8 and sum("the simulated run (speed scale 1)" in f
                                                                         for f in home0) == 2


def test_registered_boundary_values_pass(built, tmp_path: Path) -> None:
    pkg = with_replay(built[3], tmp_path, table_max_depth=0.005, self_collision_max_depth=0.001, arm_max=0.1499,
                      object_displacement=0.05, object_final_error=0.05)
    assert S.sim_gate_failures(pkg) == []


@pytest.mark.parametrize(("key", "value", "needle"), [
    ("table_max_depth", 0.0051, "table penetration"),
    ("self_collision_max_depth", 0.0011, "self-collision"),
    ("limit_violation_ticks", 1, "joint-limit"),
    ("arm_max", 0.15, "tracking"),                               # strict bound
    ("reference_displacement", 0.019, "the reference moves"),
    ("object_displacement", 0.049, "object moved"),              # < 0.5 x 10 cm
    ("object_final_error", 0.051, "final error"),                # > max(3 cm, 0.5 x 10 cm)
    ("object_fell", True, "fell"),
    ("table_max_depth", None, "table penetration"),             # a missing metric fails (NaN)
    ("object_fell", None, "fell"),
])
def test_each_threshold_alone_refuses(built, tmp_path: Path, key: str, value: object, needle: str) -> None:
    failures = S.sim_gate_failures(with_replay(built[3], tmp_path, **{key: value}))
    assert len(failures) == 1 and needle in failures[0], failures


def test_stats_of_another_trajectory_are_refused(built, tmp_path: Path) -> None:
    pkg = with_replay(built[3], tmp_path)
    with np.load(pkg / "trajectory.npz") as z:
        data = dict(z)
    data["arm_q"] = data["arm_q"] * (1 - 1e-6)                   # any edit after the simulation
    np.savez(pkg / "trajectory.npz", **data)
    failures = S.sim_gate_failures(pkg)
    assert len(failures) == 1 and "not computed on this trajectory" in failures[0]
    stats = json.loads((pkg / "sim_stats.json").read_text())
    del stats["trajectory_sha256"]
    (pkg / "sim_stats.json").write_text(json.dumps(stats))
    assert "not computed on this trajectory" in S.sim_gate_failures(pkg)[0]


def _no_input(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    prompts: list[str] = []

    def fake(prompt: str = "") -> str:
        prompts.append(prompt)
        return "no"

    monkeypatch.setattr(builtins, "input", fake)
    return prompts


def test_execute_refuses_before_connecting_when_the_gate_fails(built, monkeypatch: pytest.MonkeyPatch,
                                                               capsys: pytest.CaptureFixture[str]) -> None:
    prompts = _no_input(monkeypatch)
    assert S.main(["--package", str(built[3]), "--execute", "--iface", "lo"]) == 4
    assert prompts == [] and "REFUSED" in capsys.readouterr().err


def test_execute_reaches_the_connect_prompt_when_the_gate_passes(built, tmp_path: Path,
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    prompts = _no_input(monkeypatch)
    assert S.main(["--package", str(with_replay(built[3], tmp_path)), "--execute", "--iface", "lo"]) == 1
    assert len(prompts) == 1 and "'yes'" in prompts[0]


def test_dry_run_reports_the_gate(built, capsys: pytest.CaptureFixture[str]) -> None:
    assert S.main(["--package", str(built[3])]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "completed" and out["sim_gate"]["execute_allowed"] is False
    assert out["sim_gate"]["failures"] == S.sim_gate_failures(built[3])


def test_package_records_the_gate(built) -> None:
    pkg = built[3]
    meta = json.loads((pkg / "package.json").read_text())
    assert meta["sim_gate"]["thresholds"] == S.SIM_GATE
    assert meta["sim_gate"]["execute_allowed"] is False
    assert meta["sim_gate"]["failures"] == S.sim_gate_failures(pkg)
    readme = (pkg / "README.md").read_text()
    assert "## Execution gate" in readme and "REFUSE" in readme
    for text in ("5.0 mm", "1.0 mm", "0.15 rad"):
        assert text in readme


# --- the gate also reads the run script's own simulated run from each checked home ----------------

def _home_rec(stats: dict) -> dict:
    return next(r for r in stats["transfers"]["homes"] if r["index"] == 0)


def with_transfer(pkg: Path, tmp_path: Path, where: str, phase: str, key: str, value: object) -> Path:
    """`with_replay` (trajectory passes) plus one transfer metric of home #0 set: `where` = "run", or the first drill
    whose stop starts with `where` ("play" / "to_hold")."""
    out = with_replay(pkg, tmp_path)
    stats = json.loads((out / "sim_stats.json").read_text())
    rec = _home_rec(stats)
    run = rec["run"] if where == "run" else next(d for d in rec["drills"] if d["stop"].startswith(where))
    run["phases"][phase][key] = value
    (out / "sim_stats.json").write_text(json.dumps(stats))
    return out


def test_the_synthetic_package_has_a_passing_checked_home(built, tmp_path: Path) -> None:
    """The `arms_hanging` home is in the package; it passes the gate once the object criteria pass (the run
    and the placement corners are held to them, and the synthetic hand never touches the box), and fails on the
    object criteria only otherwise; `zeros` is not in the package."""
    assert S.gate_report(with_replay(built[3], tmp_path))["homes_allowed"] == [0]
    rep = S.gate_report(built[3])
    assert rep["homes_allowed"] == []
    assert all(("object moved" in f or "object final error" in f) for f in rep["home_failures"]
               if f.startswith("home #0"))
    meta = json.loads((built[3] / "package.json").read_text())
    names = {h["name"]: h for h in meta["homes"]}
    assert names["arms_hanging"]["in_package"] and names["arms_hanging"]["index"] == 0
    assert not names["zeros"]["in_package"]                         # forearms 0.06 m above the table / the box
    assert any("'zeros' is not in the package" in f for f in rep["home_failures"])


@pytest.mark.parametrize(("where", "phase", "key", "value", "needle"), [
    ("run", "ramp_up", "table_contact_steps", 1, "touches the table"),
    ("run", "to_hold", "table_contact_steps", 1, "touches the table"),
    ("run", "wait_go", "self_collision_max_depth", 0.0011, "self-collision"),
    ("run", "go_home", "object_contact_steps", 1, "touches the object"),
    ("run", "release", "table_contact_steps", 1, "touches the table"),
    ("run", "to_hold", "arm_tracking_max", 0.15, "tracking"),
    ("run", "go_home", "arm_tracking_max", 0.15, "tracking"),
    ("to_hold", "go_home", "object_contact_steps", 1, "touches the object"),
    ("play", "go_home", "table_max_depth", 0.0051, "table contact"),
    ("play", "release", "self_collision_max_depth", 0.0011, "self-collision"),
    ("play", "go_home", "arm_tracking_max", 0.15, "tracking"),
    ("run", "to_hold", "table_contact_steps", None, "touches the table"),        # a missing metric fails (NaN)
])
def test_each_transfer_threshold_alone_refuses(built, tmp_path: Path, where: str, phase: str, key: str,
                                               value: object, needle: str) -> None:
    rep = S.gate_report(with_transfer(built[3], tmp_path, where, phase, key, value))
    assert rep["homes_allowed"] == [] and len(rep["failures"]) == 1
    assert rep["failures"][0].startswith("no checked home passes") and needle in rep["failures"][0], rep


def test_transfer_boundary_values_pass(built, tmp_path: Path) -> None:
    pkg = with_replay(built[3], tmp_path)
    stats = json.loads((pkg / "sim_stats.json").read_text())
    rec = _home_rec(stats)
    rec["run"]["phases"]["to_hold"].update(self_collision_max_depth=0.001, arm_tracking_max=0.1499)
    play = next(d for d in rec["drills"] if d["stop"].startswith("play"))
    play["phases"]["go_home"].update(table_max_depth=0.005, table_contact_steps=50)    # the playback's own poses
    (pkg / "sim_stats.json").write_text(json.dumps(stats))
    assert S.gate_report(pkg)["failures"] == []


@pytest.mark.parametrize(("edit", "needle"), [
    (lambda r: r["run"].update(status="aborted"), "did not complete"),
    (lambda r: r["drills"][0].update(status="completed"), "did not stop"),
    (lambda r: r["to_hold_plan"].update(ok=False), "to_hold_plan: not planned"),
    (lambda r: r["home_q"].__setitem__(0, r["home_q"][0] + 0.01), "differs from trajectory.npz"),
    (lambda r: r.update(drills=[]), "no operator-stop drill"),
])
def test_a_broken_home_record_refuses(built, tmp_path: Path, edit, needle: str) -> None:
    pkg = with_replay(built[3], tmp_path)
    stats = json.loads((pkg / "sim_stats.json").read_text())
    edit(_home_rec(stats))
    (pkg / "sim_stats.json").write_text(json.dumps(stats))
    fails = S.sim_gate_failures(pkg)
    assert len(fails) == 1 and needle in fails[0], fails


def test_stats_without_a_transfer_simulation_are_refused(built, tmp_path: Path) -> None:
    pkg = with_replay(built[3], tmp_path)
    stats = json.loads((pkg / "sim_stats.json").read_text())
    del stats["transfers"]
    (pkg / "sim_stats.json").write_text(json.dumps(stats))
    assert "no simulation of the moves" in S.sim_gate_failures(pkg)[0]


def test_execute_starts_only_from_a_home_that_passed(built, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A second checked home whose simulated TO_HOLD touched the table: `--execute` (trajectory gate passing) starts
    from home #0 but refuses the robot standing at home #1, with nothing sent (mutation: --execute not restricting the
    homes runs from #1)."""
    from tests.robot.plans import BODY17, body
    pkg = with_replay(built[3], tmp_path)
    with np.load(pkg / "trajectory.npz") as z:
        data = dict(z)
    plan = S.Plan.load(pkg / "trajectory.npz")
    other = plan.home_q[0].copy()
    other[3] += 0.3                                                   # right elbow 0.3 rad further bent
    row = np.concatenate([other, plan.to_hold_rows(0)[0, S.TAU], plan.hand_q[0]])
    data["home_q"] = np.vstack([plan.home_q, other])
    data["to_hold_path"] = np.vstack([plan.to_hold_path, row, plan.hold_vec()])
    data["to_hold_index"] = np.append(plan.to_hold_index, plan.to_hold_index[-1] + 2)
    back = np.concatenate([other, plan.to_hold_rows(0)[0, S.TAU], plan.hand_q[-1]])
    data["go_home_path"] = np.vstack([plan.go_home_path, plan.end_vec(), back])
    data["go_home_index"] = np.append(plan.go_home_index, plan.go_home_index[-1] + 2)
    np.savez(pkg / "trajectory.npz", **data)
    assert S.Plan.load(pkg / "trajectory.npz").validate() == []
    stats = json.loads((pkg / "sim_stats.json").read_text())
    rec = json.loads(json.dumps(_home_rec(stats)))
    rec.update(name="other", index=1, home_q=other.tolist())
    rec["run"]["phases"]["to_hold"]["table_contact_steps"] = 7
    stats["transfers"]["homes"].append(rec)
    stats["trajectory_sha256"] = S.file_sha256(pkg / "trajectory.npz")
    (pkg / "sim_stats.json").write_text(json.dumps(stats))
    rep = S.gate_report(pkg)
    assert rep["failures"] == [] and rep["homes_allowed"] == [0]
    assert any(f.startswith("home #1 'other'") and "touches the table" in f for f in rep["home_failures"])

    answers = iter(["yes", "go"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(answers))
    for start, rc in ((1, 3), (0, 0)):
        answers = iter(["yes", "go"])
        q0 = np.zeros(35)
        q0[BODY17] = data["home_q"][start]
        io = S.SimulatedIO(q0=q0, hand0=np.zeros(7))
        monkeypatch.setattr(S, "SdkIO", lambda *a, _io=io, **k: _io)
        assert S.main(["--package", str(pkg), "--execute", "--iface", "lo"]) == rc, start
        assert (io.sent_arm == 0) == (start == 1)
    assert body(S.Plan.load(pkg / "trajectory.npz"), 1)[BODY17][3] == pytest.approx(other[3])


# --- package.py: the table footprint the sim and the gate assume is a placement instruction --------

TABLE_KEYS = ("near_edge_x_m", "far_edge_x_m", "right_edge_y_m", "left_edge_y_m", "top_z_m", "thickness_m")


def test_package_states_the_simulated_table_footprint(built) -> None:
    """package.json, sim_stats.json and the README all carry the table rectangle the simulation used
    (`SceneConfig` 0.18-0.78 m x -0.60-0.40 m, 4 cm thick, top at the chosen height), with a placement tolerance."""
    _, _, result, pkg = built
    meta = json.loads((pkg / "package.json").read_text())
    stats = json.loads((pkg / "sim_stats.json").read_text())
    expect = dict(zip(TABLE_KEYS, (0.18, 0.78, -0.60, 0.40, result.placement.table_z, 0.04), strict=True))
    for key, value in expect.items():
        assert meta["table"][key] == pytest.approx(value) and stats["table"][key] == pytest.approx(value), key
    assert meta["table"]["placement_tolerance_m"] == pytest.approx(0.01)
    readme = (pkg / "README.md").read_text()
    for text in ("x = 0.180 m", "x = 0.780 m", "y = -0.600 m", "y = 0.400 m", "4.0 cm thick", "+-1.0 cm"):
        assert text in readme, text
    assert "only the waist yaw is sent" not in readme and "all three waist entries" in readme


@pytest.mark.parametrize("edit", ["moved", "missing"])
def test_a_table_other_than_the_simulated_one_is_refused(built, tmp_path: Path, edit: str) -> None:
    pkg = with_replay(built[3], tmp_path)
    assert S.sim_gate_failures(pkg) == []
    meta = json.loads((pkg / "package.json").read_text())
    if edit == "moved":
        meta["table"]["near_edge_x_m"] = 0.10                          # the lab's table 8 cm closer to the robot
    else:
        del meta["table"]
    (pkg / "package.json").write_text(json.dumps(meta))
    fails = S.sim_gate_failures(pkg)
    assert len(fails) == 1 and "table" in fails[0] and "simulated" in fails[0], fails


@pytest.mark.parametrize(("edit", "needle"), [
    (lambda d: d["to_hold_path"].__setitem__((-1, 0), d["to_hold_path"][-1, 0] + 0.1), "does not join"),
    (lambda d: d["go_home_path"].__setitem__((-1, 3), d["go_home_path"][-1, 3] + 0.1), "does not join"),
    (lambda d: d.__setitem__("home_tol", np.asarray(0.5)), "home_tol"),
    (lambda d: d.__setitem__("to_hold_index", d["to_hold_index"][:1]), "to_hold_index"),
])
def test_run_script_refuses_paths_that_do_not_join(built, tmp_path: Path, capsys: pytest.CaptureFixture[str], edit,
                                                    needle: str) -> None:
    """The stored paths must start at their home and end at the trajectory (and back), with a tolerance of at most
    0.15 rad: anything else is refused before the dry-run or the robot (mutation: validation skipping the paths)."""
    pkg = tmp_path / "pkg"
    shutil.copytree(built[3], pkg)
    with np.load(pkg / "trajectory.npz") as z:
        data = {k: np.array(v) for k, v in z.items()}
    edit(data)
    np.savez(pkg / "trajectory.npz", **data)
    assert S.main(["--package", str(pkg)]) == 2
    assert needle in capsys.readouterr().err


# --- --execute accepts only the run options the package simulated ---------------

@pytest.mark.parametrize(("flags", "needle"), [
    (["--no-tau-ff"], "--no-tau-ff"),                 # the simulation sent the gravity feed-forward
    (["--hold-speed", "0.5"], "--hold-speed"),        # simulated at 0.3 rad/s
    (["--hold-speed", "0.2"], "--hold-speed"),
    (["--speed-scale", "0.5"], "--speed-scale"),      # simulated at 1.0
    (["--no-waist"], "--no-waist"),                   # the waist is welded at 0 in the simulation
])
def test_execute_refuses_run_options_the_package_did_not_simulate(built, tmp_path: Path, monkeypatch:
                                                                   pytest.MonkeyPatch, capsys: pytest.CaptureFixture[
                                                                       str], flags: list[str], needle: str) -> None:
    """A package simulated only with the feed-forward on and at 0.3 rad/s / speed scale 1 (the passing synthetic
    package) is refused under `--execute` with any other value, before anything connects; the defaults reach the
    connect prompt (`test_execute_reaches_the_connect_prompt_when_the_gate_passes`)."""
    pkg = with_replay(built[3], tmp_path)
    stats = json.loads((pkg / "sim_stats.json").read_text())
    assert (stats["transfers"]["send_tau_ff"], stats["transfers"]["hold_speed_rad_s"],
            stats["transfers"]["speed_scale"]) == (True, 0.3, 1.0)
    prompts = _no_input(monkeypatch)
    assert S.main(["--package", str(pkg), "--execute", "--iface", "lo", *flags]) == 4
    err = capsys.readouterr().err
    assert prompts == [] and "REFUSED" in err and needle in err, err


def test_execute_refuses_a_package_that_does_not_record_its_run_options(built, tmp_path: Path,
                                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    pkg = with_replay(built[3], tmp_path)
    stats = json.loads((pkg / "sim_stats.json").read_text())
    del stats["transfers"]["send_tau_ff"]
    (pkg / "sim_stats.json").write_text(json.dumps(stats))
    prompts = _no_input(monkeypatch)
    assert S.main(["--package", str(pkg), "--execute", "--iface", "lo"]) == 4 and prompts == []
    sim = S.simulated_conditions(stats)
    assert sim["send_tau_ff"] is None and S.condition_failures(sim, S.Config())


# --- the run simulated on a table at both edges of the placement tolerance -----

def test_the_package_simulates_the_table_tolerance(built) -> None:
    """The synthetic package records its tolerance (1 cm, the same in package.json), and its checked home was also
    simulated on a table 1 cm higher with every edge 1 cm out and on a table 1 cm lower, both clean."""
    pkg = built[3]
    stats = json.loads((pkg / "sim_stats.json").read_text())
    meta = json.loads((pkg / "package.json").read_text())
    assert stats["transfers"]["table_tol_m"] == pytest.approx(0.01) == meta["table"]["placement_tolerance_m"]
    rec = _home_rec(stats)
    runs = {r["case"]: r for r in rec["table_tolerance_runs"]}
    assert set(runs) == {"table_high", "table_low"}
    assert (runs["table_high"]["dz_m"], runs["table_high"]["grow_m"]) == pytest.approx((0.01, 0.01))
    assert (runs["table_low"]["dz_m"], runs["table_low"]["grow_m"]) == pytest.approx((-0.01, 0.0))
    for r in runs.values():
        assert r["status"] == "completed"
        assert all(v["table_contact_steps"] == 0 for k, v in r["phases"].items() if k != "play")


@pytest.mark.parametrize(("edit", "needle"), [
    (lambda r: r["table_tolerance_runs"][0]["phases"]["to_hold"].update(table_contact_steps=1), "touches the table"),
    (lambda r: r["table_tolerance_runs"][1]["phases"]["go_home"].update(table_contact_steps=3), "touches the table"),
    (lambda r: r["table_tolerance_runs"][0]["phases"]["wait_go"].update(object_contact_steps=1), "touches the object"),
    (lambda r: r["table_tolerance_runs"][0]["phases"]["play"].update(table_max_depth=0.0051), "table contact"),
    (lambda r: r["table_tolerance_runs"][0]["phases"]["play"].update(table_max_force=20.1), "force"),
    (lambda r: r["table_tolerance_runs"][1].update(status="aborted"), "did not complete"),
    (lambda r: r["table_tolerance_runs"].pop(), "not simulated"),
    (lambda r: r.pop("table_tolerance_runs"), "not simulated"),
    (lambda r: r["table_tolerance_runs"][0].update(dz_m=0.005), "not +1.0 cm"),
    # each case at its OWN edge, sign and edge growth included
    (lambda r: r["table_tolerance_runs"][0].update(dz_m=-0.01), "not +1.0 cm"),
    (lambda r: r["table_tolerance_runs"][1].update(dz_m=0.01), "not -1.0 cm"),       # both runs at +1 cm
    (lambda r: r["table_tolerance_runs"][0].update(grow_m=0.0), "edges out by"),
    (lambda r: r["table_tolerance_runs"][1].update(grow_m=0.01), "edges out by"),
])
def test_a_contact_on_a_table_inside_the_tolerance_refuses(built, tmp_path: Path, edit, needle: str) -> None:
    """Each failure of a table-tolerance run alone refuses the home (mutation: the gate reading the nominal run
    only)."""
    pkg = with_replay(built[3], tmp_path)
    stats = json.loads((pkg / "sim_stats.json").read_text())
    edit(_home_rec(stats))
    (pkg / "sim_stats.json").write_text(json.dumps(stats))
    rep = S.gate_report(pkg)
    assert rep["homes_allowed"] == [] and len(rep["failures"]) == 1 and needle in rep["failures"][0], rep


def test_a_tolerance_the_moves_were_not_simulated_at_is_refused(built, tmp_path: Path) -> None:
    """package.json telling the lab +-1 cm while the moves were simulated at +-0.5 cm is refused (and a tolerance
    missing from sim_stats.json)."""
    pkg = with_replay(built[3], tmp_path)
    meta = json.loads((pkg / "package.json").read_text())
    meta["table"]["placement_tolerance_m"] = 0.02
    (pkg / "package.json").write_text(json.dumps(meta))
    fails = S.sim_gate_failures(pkg)
    assert len(fails) == 1 and "planned and simulated at" in fails[0], fails
    pkg2 = with_replay(built[3], tmp_path / "b")
    stats = json.loads((pkg2 / "sim_stats.json").read_text())
    del stats["transfers"]["table_tol_m"]
    (pkg2 / "sim_stats.json").write_text(json.dumps(stats))
    assert any("tolerance" in f for f in S.sim_gate_failures(pkg2))


# --- forceful contacts the depth under-reports -------------------------------------

@pytest.mark.parametrize(("where", "key", "value", "needle"), [
    ("replay", "table_max_force", 20.1, "table contact force"),
    ("drill", "table_max_force", 20.1, "table contact force"),
])
def test_a_forceful_table_contact_refuses(built, tmp_path: Path, where: str, key: str, value: float,
                                          needle: str) -> None:
    """An early straight TO_HOLD pressed 66 N into the table at 0.6 mm depth: the playback and the drills' way home
    are capped at 20 N as well as 5 mm (boundary 20 N passes)."""
    for v, refused in ((20.0, False), (value, True)):
        sub = tmp_path / f"{where}{v}"
        sub.mkdir()
        if where == "replay":
            pkg = with_replay(built[3], sub, **{key: v})
        else:
            pkg = with_transfer(built[3], sub, "play", "go_home", key, v)
        fails = S.sim_gate_failures(pkg)
        assert bool(fails) == refused and (not refused or needle in fails[0]), fails


# --- a package planned at a tolerance below 1 cm (or none) is refused -------------

def test_a_package_planned_without_the_table_tolerance_is_refused(built, tmp_path: Path) -> None:
    """`table_tol_m` 0 with no tolerance runs (a TransferConfig with table_tol 0; the README would say +-0.0 cm) is
    refused, as is 0.5 cm, even when package.json states the same value (mutation: tolerance 0 skips the runs)."""
    for tol in (0.0, 0.005):
        pkg = with_replay(built[3], tmp_path / f"t{tol}")
        stats = json.loads((pkg / "sim_stats.json").read_text())
        stats["transfers"]["table_tol_m"] = tol
        if tol == 0.0:
            _home_rec(stats).pop("table_tolerance_runs")
        (pkg / "sim_stats.json").write_text(json.dumps(stats))
        meta = json.loads((pkg / "package.json").read_text())
        meta["table"]["placement_tolerance_m"] = tol
        (pkg / "package.json").write_text(json.dumps(meta))
        rep = S.gate_report(pkg)
        assert rep["homes_allowed"] == [] and any("below the 1.0 cm" in f for f in rep["home_failures"]), rep


# --- --execute refuses a Dex3 hand off the simulated start, printing the measured values --------

def _execute_at_home(pkg: Path, hand0: np.ndarray, monkeypatch: pytest.MonkeyPatch,
                     capsys: pytest.CaptureFixture[str]) -> tuple[int, S.SimulatedIO, str]:
    from tests.robot.plans import BODY17
    answers = iter(["yes", "go"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(answers))
    q0 = np.zeros(35)
    q0[BODY17] = S.Plan.load(pkg / "trajectory.npz").home_q[0]
    io = S.SimulatedIO(q0=q0, hand0=hand0)
    monkeypatch.setattr(S, "SdkIO", lambda *a, _io=io, **k: _io)
    rc = S.main(["--package", str(pkg), "--execute", "--iface", "lo"])
    return rc, io, capsys.readouterr().out


def test_execute_refuses_a_hand_off_the_simulated_start(built, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                         capsys: pytest.CaptureFixture[str]) -> None:
    """The package records the Dex3 start it simulated (0); `--execute` with a measured hand 0.15 rad off on one joint
    sends nothing and prints the measured hand (exit 3, refused); 0.05 rad off runs (mutation: the start hand not
    passed to the run under --execute)."""
    pkg = with_replay(built[3], tmp_path)
    stats = json.loads((pkg / "sim_stats.json").read_text())
    assert stats["transfers"]["start_hand"] == [0.0] * 7
    off = np.zeros(7)
    off[1] = 0.15
    rc, io, out = _execute_at_home(pkg, off, monkeypatch, capsys)
    res = json.loads(out[out.index("{"):])
    assert rc == 3 and res["status"] == "refused" and io.sent_arm == 0 and io.sent_hand == 0
    assert "Dex3" in res["reason"] and "0.15" in res["reason"], res["reason"]
    near = np.zeros(7)
    near[1] = 0.05
    rc, io, _ = _execute_at_home(pkg, near, monkeypatch, capsys)
    assert rc == 0 and io.sent_arm > 0


def test_execute_refuses_a_package_without_its_simulated_start_hand(built, tmp_path: Path,
                                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    pkg = with_replay(built[3], tmp_path)
    stats = json.loads((pkg / "sim_stats.json").read_text())
    del stats["transfers"]["start_hand"]
    (pkg / "sim_stats.json").write_text(json.dumps(stats))
    prompts = _no_input(monkeypatch)
    assert S.main(["--package", str(pkg), "--execute", "--iface", "lo"]) == 4 and prompts == []


# --- the speed scale the package simulated is the default and the only one --execute accepts ---

def test_execute_runs_at_the_simulated_speed_scale_only(built, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                        capsys: pytest.CaptureFixture[str]) -> None:
    """A package simulated at 0.5: `--execute` without `--speed-scale` reaches the connect prompt (the default is the
    simulated value), `--speed-scale 1` is refused before it; the dry-run plays at 0.5 (twice the ticks)."""
    pkg = with_replay(built[3], tmp_path)
    stats = json.loads((pkg / "sim_stats.json").read_text())
    stats["transfers"]["speed_scale"] = 0.5
    for rec in stats["transfers"]["homes"]:                  # every simulated run records its speed scale
        for r in [rec.get("run"), *[x for k in ("drills", "table_tolerance_runs", "placement_runs", "state_runs")
                                    for x in rec.get(k) or []]]:
            if r:
                r["run_options"]["speed_scale"] = 0.5
    (pkg / "sim_stats.json").write_text(json.dumps(stats))
    prompts = _no_input(monkeypatch)
    assert S.main(["--package", str(pkg), "--execute", "--iface", "lo"]) == 1 and len(prompts) == 1
    assert S.main(["--package", str(pkg), "--execute", "--iface", "lo", "--speed-scale", "1"]) == 4
    assert "--speed-scale" in capsys.readouterr().err and len(prompts) == 1
    ticks = {}
    for name, p in (("half", pkg), ("full", built[3])):
        assert S.main(["--package", str(p)]) == 0
        out = json.loads(capsys.readouterr().out)
        ticks[name] = float(out["phases"]["play"].split()[0])
    assert ticks["half"] == pytest.approx(2.0 * ticks["full"], rel=0.01), ticks


# --- labels in package.json and the README (force cap, object placement) ------

def test_package_labels_the_chosen_force_cap_and_the_unsimulated_object_offsets(built) -> None:
    pkg = built[3]
    meta = json.loads((pkg / "package.json").read_text())
    assert "chosen, not sourced" in meta["sim_gate"]["notes"]["table_max_force_n"]
    readme = (pkg / "README.md").read_text()
    assert "20 N\n  cap is chosen, not sourced" in readme or "cap is chosen, not sourced" in readme
    assert "x, y and yaw are NOT simulated" not in readme                  # now simulated
    assert "Object placement tolerance: +-1.0 cm in x and y, +-5 deg in yaw" in readme
    assert "more than 0.01 rad from 0" in readme and "refuses `--no-waist`" in readme
    assert "more than 0.1 rad from it" in readme and "Playback speed scale 1" in readme
