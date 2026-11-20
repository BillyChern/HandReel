"""Regression tests of fastdo.audit for the replay control and the surviving pair rows.

Each test names its item. The item-1 tests are the kills of five mutants that survived every earlier test.
The scenes are variants of `fastdo.audit.toy`, written out here.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest
from nullaudit.mujoco_adapter import MujocoExecutor

import fastdo.audit as A
from fastdo.audit import __main__ as cli
from fastdo.audit.__main__ import main
from tests.audit import _scene as S

N, W = 160, 20
LIFT = 0.3


# ------------------------------------------------------------------------------------------------ fixtures
def _settings(m: mujoco.MjModel) -> dict:
    return {"weld_eq_ids": (), "gravity": np.array(m.opt.gravity), "ctrl_steps": 1, "terminate_z": -np.inf}


def _move(m: mujoco.MjModel, dz: float) -> dict:
    """Execute a plate move of `dz` metres after W on the carry scene (the cube rides the plate); the executed
    object rows are the reference (a perfect run)."""
    st = S.carry(m)
    u = np.full((N, 1), st.ctrl[0])
    u[W:, 0] += np.linspace(0.0, dz, N - W)
    ex = MujocoExecutor(model=m, q0=st.qpos, hand_geom=S.masks(m)[0], ctrl0=st.ctrl, warmup_rows=W)
    ro = ex(u)
    assert ro.status == "clip_end"
    return {"ref": ex.last_qpos[:, 1:8].copy(), "u": u, "q0": st.qpos, "ctrl0": st.ctrl,
            "executed": A.Executed(ex.last_qpos, ex.last_qvel, ex.last_ctrl)}


def _audit(m: mujoco.MjModel, run: dict, cert=None, **kw) -> A.RunAudit:
    cert = cert or A.make_certificate("CD", run["ref"], warmup_rows=W, required_rows=N)
    args = {"warmup_rows": W, "model": m, "hand_geom": S.masks(m)[0], "controls": run["u"], "q0": run["q0"],
            "ctrl0": run["ctrl0"], "removal": "displace", **_settings(m), **kw}
    return A.audit_run(run["ref"], cert, **args)


def _write_npz(path: Path, run: dict, scene: str, m: mujoco.MjModel, rows: bool = True, **extra) -> Path:
    arrays = {"reference": run["ref"], "warmup_rows": np.int64(W), "required_rows": np.int64(N),
              "scene": np.str_(scene), "controls": run["u"], "q0": run["q0"], "ctrl0": run["ctrl0"],
              "weld_eq_ids": np.zeros(0, np.int64), "gravity": np.array(m.opt.gravity), "ctrl_steps": np.int64(1),
              "terminate_z": np.float64(-np.inf), "hand_geom": S.masks(m)[0], **extra}
    if rows:
        arrays.update(qpos=run["executed"].qpos, qvel=run["executed"].qvel, ctrl=run["executed"].ctrl)
    np.savez(path, **arrays)
    return path


def _carry_npz(tmp: Path, rows: bool = True, **extra) -> Path:
    (tmp / "carry.xml").write_text(S.CARRY_XML)
    m = S.carry_model()
    return _write_npz(tmp / "run.npz", _move(m, LIFT), "carry.xml", m, rows=rows, **extra)


# ------------------------------------------------------------------------------------------------ item 1
def test_item1_a_replay_departing_from_the_executed_rows_fails_the_control() -> None:
    """Mutant runner.py:323-324 (the "departs" reason dropped): 1 mm on one executed row must fail the control."""
    m = S.carry_model()
    run = _move(m, LIFT)
    good = _audit(m, run, executed=run["executed"], sim_dt=m.opt.timestep).to_dict()
    assert good["positive_control"]["passed"] and good["positive_control"]["max_dev"] == 0.0   # green control
    q = run["executed"].qpos.copy()
    q[W + 40, 3] += 1e-3
    d = _audit(m, run, executed=A.Executed(q, run["executed"].qvel, run["executed"].ctrl),
               sim_dt=m.opt.timestep).to_dict()
    pc = d["positive_control"]
    assert pc["reasons"] == ("the replay departs from the executed rows by 0.001 > 1e-06",), pc["reasons"]
    assert d["contact_removed_null"]["status"] == "invalid" and d["grip_census"]["status"] == "not_run"


def test_item1_executed_rows_without_sim_dt_fail_on_max_dev() -> None:
    """The executor ran at 0.0025 s; the audit simulates the scene as stored (0.005 s) and is given the executed
    rows but no sim_dt: the control must fail on max_dev alone."""
    exe = S.carry_model()
    exe.opt.timestep = 0.0025
    run = _move(exe, LIFT)
    m = S.carry_model()
    d = _audit(m, run, executed=run["executed"]).to_dict()
    pc = d["positive_control"]
    assert pc["sim_dt"] is None and pc["replay_passes"] is True and pc["max_dev"] > A.REPLAY_TOL
    assert pc["reasons"][0].startswith("the replay departs from the executed rows")
    assert all("departs from the executed rows" in r for r in pc["reasons"])          # qpos, then qvel / ctrl
    assert d["contact_removed_null"]["status"] == "invalid"


def test_item1_a_removed_model_without_other_pairs_is_refused() -> None:
    """Mutant runner.py:153-155: a removed model that also lacks the object|pedestal pair is not the executor's
    scene minus hand|object pairs."""
    m = S.model()
    st = S.press(m)
    ref = np.repeat(S.object_rows(st.qpos[None]), N, 0)
    run = {"ref": ref, "u": np.full((N, 1), st.ctrl[0]), "q0": st.qpos, "ctrl0": st.ctrl}
    both, _ = A.remove_pairs(S.XML, ["hand_object", "object_pedestal"])
    with pytest.raises(A.InputError, match="also lacks 1 pair"):
        _audit(m, run, removal="model", removed_model=S.model(both))
    one, _ = A.remove_pairs(S.XML, ["hand_object"])
    assert _audit(m, run, removal="model", removed_model=S.model(one)).clip.outcomes["hand_absent"].passed


def test_item1_a_replay_that_stops_early_is_an_infinite_departure() -> None:
    """Mutant runner.py:321-322: the plate lowers the cube; the executed run used no drop height, the audit is given
    0.2 m, so its replay stops at row 105 with a prefix equal to the executed rows. The departure is infinite, and
    waiving the certificate's rejection (`allow_uncertified_run`) does not waive it."""
    m = S.carry_model()
    run = _move(m, -0.25)
    d = _audit(m, run, executed=run["executed"], sim_dt=m.opt.timestep, terminate_z=0.2,
               allow_uncertified_run=True).to_dict()
    pc = d["positive_control"]
    assert pc["replay_status"] == "drop" and pc["replay_rows"] < N and pc["max_dev"] == float("inf")
    assert pc["reasons"][-1] == "the replay departs from the executed rows by inf > 1e-06" and not pc["overridden"]
    assert d["contact_removed_null"]["status"] == "invalid"


LID = ('<body name="lid" pos="0 0 0.63" gravcomp="1"><joint name="lid_z" type="slide" axis="0 0 1" '
       'stiffness="2000" damping="50"/><geom name="lid_g" type="box" size="0.05 0.05 0.01" mass="0.5"/></body>')
LID_PAIR = '<pair name="lid_object" geom1="lid_g" geom2="right_object_cube" condim="3"/>'
LID_XML = S.CARRY_XML.replace("</worldbody>", LID + "</worldbody>").replace("<contact>", "<contact>" + LID_PAIR)


def test_item1_outside_mask_contact_alone_gives_exit_1(tmp_path: Path) -> None:
    """Mutant runner.py:418-420: the plate lifts the cube into a sprung lid that is not in the hand mask. The
    certified replay has lid contact; the contact-removed run does not (the cube falls away), so the null is a
    verified fail and the lid contact is the report's only problem."""
    (tmp_path / "lid.xml").write_text(LID_XML)
    m = S.model(LID_XML)
    p = _write_npz(tmp_path / "lid.npz", _move(m, LIFT), "lid.xml", m, sim_dt=np.float64(m.opt.timestep))
    out = tmp_path / "r.json"
    assert main([str(p), "--json", str(out)]) == 1
    doc = json.loads(out.read_text())
    r = doc["runs"][0]
    assert r["positive_control"]["passed"] and r["positive_control"]["outside_mask_force_n_max"] > 1.0
    assert r["contact_removed_null"]["status"] == "fail" and r["contact_removed_null"]["verified_n0"] is True
    want = "object contact from geoms outside the hand mask in the certified replay of 1 runs: lid"
    assert doc["summary"]["problems"] == [want]


def test_item1_an_invalid_replay_fails_the_control() -> None:
    """Mutant runner.py:312-313: ten rows after release the servo target jumps to -1e5 m; the replay diverges (|qvel|
    above the executor's bound). Waiving the certificate's rejection must not waive the invalid replay."""
    m = S.carry_model()
    run = _move(m, LIFT)
    run["u"][W + 10:, 0] = -1e5
    d = _audit(m, run, sim_dt=m.opt.timestep, allow_uncertified_run=True).to_dict()
    pc = d["positive_control"]
    assert pc["replay_status"] == "invalid" and pc["reasons"][0].startswith("the replay is invalid (|qvel|")
    assert pc["waived"] == (pc["reasons"][1],) and pc["overridden"] is False
    assert d["contact_removed_null"]["note"].startswith("positive control failed: the replay is invalid")


# ------------------------------------------------------------------------------------------------ item 2
def test_item2_a_certificate_only_control_is_a_problem(tmp_path: Path) -> None:
    p = _carry_npz(tmp_path, rows=False)
    out = tmp_path / "r.json"
    assert main([str(p), "--json", str(out)]) == 1
    doc = json.loads(out.read_text())
    assert doc["runs"][0]["positive_control"]["certificate_only"] is True
    assert doc["summary"]["problems"] == [A.runner.CERT_ONLY + " on 1 runs: run"]
    assert doc["summary"]["positive_control"]["certificate_only"] == ["run"]


@pytest.mark.parametrize("anchor", ["executor_model", "rows"])
def test_item2_control_any_anchor_clears_it(tmp_path: Path, anchor: str) -> None:
    """Green controls: executed rows or an executor model each tie the replay to the executor (a matching sim_dt
    alone does not: test_input_validation_regressions.py)."""
    m = S.carry_model()
    extra = {"rows": {}, "executor_model": {"executor_model": np.str_("executor.mjb")}}[anchor]
    mujoco.mj_saveModel(m, str(tmp_path / "executor.mjb"), None)
    p = _carry_npz(tmp_path, rows=anchor == "rows", **extra)
    out = tmp_path / "r.json"
    assert main([str(p), "--json", str(out)]) == 0
    pc = json.loads(out.read_text())["runs"][0]["positive_control"]
    assert pc["certificate_only"] is False and pc["anchors"] == [{"rows": "executed_rows"}.get(anchor, anchor)]


# ------------------------------------------------------------------------------------------------ item 3
def test_item3_a_c289_with_hooks_is_refused_for_nulls() -> None:
    """The hooks see only object rows: on a null they would score the certified run's hand and PLAUS terms."""
    m = S.carry_model()
    run = _move(m, LIFT)
    plaus = lambda _x: {"pen_mm_p99": 0.0, "touch_frac_on_moving": 1.0, "acc_p99": 0.0}
    hand = lambda _x: (0.0, 0.0)
    for hooks in ({"plaus": plaus, "hand_errors": hand}, {"plaus": plaus}, {"hand_errors": hand}):
        cert = A.make_certificate("C289", run["ref"], warmup_rows=W, required_rows=N, **hooks)
        with pytest.raises(A.InputError, match="hooks"):
            A.audit_run(run["ref"], cert, warmup_rows=W)
        with pytest.raises(A.InputError, match="hooks"):
            _audit(m, run, cert=cert, executed=run["executed"])
    half = A.make_certificate("C289", run["ref"], warmup_rows=W, required_rows=N)     # green control
    assert _audit(m, run, cert=half, executed=run["executed"]).certificate == "C289[obj+M_i+informative]"


# ------------------------------------------------------------------------------------------------ item 4
def test_item4_applied_thresholds_are_recorded(tmp_path: Path) -> None:
    m = S.carry_model()
    run = _move(m, LIFT)
    for spec, want in (({"preset": "CD", "pos_m": 0.05}, dataclasses.replace(A.CD, pos_m=0.05)),
                       ({"preset": "C289", "grace_rows": 40}, dataclasses.replace(A.C289, grace_rows=40)),
                       ("CD", A.CD)):
        cert = A.make_certificate(spec, run["ref"], warmup_rows=W, required_rows=N)
        assert cert.thresholds == want
        assert _audit(m, run, cert=cert).settings["thresholds"] == dataclasses.asdict(want)
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"preset": "CD", "pos_m": 0.05, "require_full": False}))
    out = tmp_path / "r.json"
    assert main([str(_carry_npz(tmp_path)), "--certificate", str(cfg), "--json", str(out)]) == 0
    doc = json.loads(out.read_text())
    want = dataclasses.asdict(dataclasses.replace(A.CD, pos_m=0.05, require_full=False))
    assert doc["inputs"]["thresholds"] == want and doc["runs"][0]["settings"]["thresholds"] == want


@pytest.mark.parametrize("doc", [{"pos_m": 0.05},                          # no preset: used to default to CD
                                 {"preset": "C_D2", "pos_m": 0.05},
                                 {"preset": "CD", "require_full": "false"},
                                 {"preset": "CD", "require_full": 0},
                                 {"preset": "CD", "pos_m": "0.1"},
                                 {"preset": "CD", "pos_m": True},
                                 {"preset": "C289", "grace_rows": 50.5},
                                 {"preset": "C289", "obj_pos_m": float("nan")},
                                 {"preset": "CD", "name": 3}])
def test_item4_configs_name_their_preset_and_have_typed_values(tmp_path: Path, doc: dict) -> None:
    ref = _move(S.carry_model(), LIFT)["ref"]
    with pytest.raises(A.InputError):
        A.make_certificate(doc, ref, warmup_rows=W, required_rows=N)
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps(doc))
    assert main([str(_carry_npz(tmp_path)), "--certificate", str(cfg)]) == 2


def test_item4_control_typed_values_are_accepted() -> None:
    ref = _move(S.carry_model(), LIFT)["ref"]
    c = A.make_certificate({"preset": "CD", "pos_m": 1, "require_full": False}, ref, warmup_rows=W, required_rows=N)
    assert c.thresholds.pos_m == 1.0 and type(c.thresholds.pos_m) is float and c.required_rows is None


# ------------------------------------------------------------------------------------------------ item 5
@pytest.mark.parametrize("hand_away", [np.array([0.0]), np.float64(2.0), np.array([0.0, 2.0, 1.0]),
                                       np.array([3.0, 2.0]), np.array([0.5, 2.0]), np.array([0.0, np.nan]),
                                       np.array(["a", "b"])])
def test_item5_an_ill_shaped_hand_away_is_exit_2(tmp_path: Path, hand_away: np.ndarray,
                                                  capsys: pytest.CaptureFixture[str]) -> None:
    p = _carry_npz(tmp_path, hand_away=hand_away)
    assert main([str(p), "--removal", "pairs"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("error: hand_away") and "Traceback" not in err


def test_item5_control_a_two_number_hand_away_runs(tmp_path: Path) -> None:
    assert main([str(_carry_npz(tmp_path, hand_away=np.array([0.0, 2.0]))), "--removal", "pairs"]) == 0


@pytest.mark.parametrize("case", ["no_T", "T_vector", "not_npz", "no_scene_file", "bad_xml", "bad_module",
                                  "bad_mask", "bad_ids"])
def test_item5_input_errors_are_exit_2(tmp_path: Path, case: str, capsys: pytest.CaptureFixture[str]) -> None:
    p = _carry_npz(tmp_path)
    with np.load(p) as f:
        z = dict(f)
    args = [str(p)]
    if case == "no_T":
        z.pop("required_rows")
    elif case == "T_vector":
        z["required_rows"] = np.array([N, N])
    elif case == "not_npz":
        p.write_text("not an archive")
    elif case == "no_scene_file":
        z["scene"] = np.str_("missing.xml")
    elif case == "bad_xml":
        (tmp_path / "carry.xml").write_text("<mujoco><worldbody><geom/></worldbody")
    elif case == "bad_module":
        args += ["--certificate", "no_such_audit_module:factory"]
    elif case == "bad_mask":
        z["hand_geom"] = np.ones(2, bool)
    elif case == "bad_ids":
        z["weld_eq_ids"] = np.array([0.5])
    if case != "not_npz":
        np.savez(p, **z)
    assert main(args) == 2, case
    err = capsys.readouterr().err
    assert err.startswith("error: ") and "Traceback" not in err, err


@pytest.mark.parametrize("exc", [KeyError("internal"), ValueError("internal"), IndexError("internal")])
def test_item5_internal_errors_are_not_input_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                    exc: Exception, capsys: pytest.CaptureFixture[str]) -> None:
    """An error from inside the audit is not an input error: it is printed with its traceback and gives exit 3, not
    exit 2 (internal errors have their own exit status)."""
    def broken(*_a, **_kw):
        raise exc

    monkeypatch.setattr(cli, "audit_run", broken)
    assert main([str(_carry_npz(tmp_path))]) == 3
    err = capsys.readouterr().err
    assert "Traceback" in err and "internal" in err and "error: internal error" in err


# ------------------------------------------------------------------------------------------------ item 7
def test_item7_allow_uncertified_run_waives_only_the_rejection() -> None:
    m = S.carry_model()
    run = _move(m, LIFT)
    moving = dict(run, ref=run["ref"].copy())
    moving["ref"][W:, 2] += 0.5                          # the replay cannot follow: the certificate rejects it
    d = _audit(m, moving, sim_dt=m.opt.timestep, allow_uncertified_run=True).to_dict()
    assert d["positive_control"]["overridden"] is True and d["contact_removed_null"]["status"] == "fail"
    d = _audit(m, moving, sim_dt=0.002, allow_uncertified_run=True).to_dict()           # plus a timestep mismatch
    assert d["positive_control"]["overridden"] is False and d["contact_removed_null"]["status"] == "invalid"
    d = _audit(m, moving, sim_dt=0.002, allow_uncertified_run=True, allow_replay_mismatch=True).to_dict()
    assert d["positive_control"]["overridden"] is True
    assert d["positive_control"]["waived"] == d["positive_control"]["reasons"]


# ------------------------------------------------------------------------------------------------ item 8
def test_item8_problems_name_runs_without_a_clip_id() -> None:
    m = S.carry_model()
    run = _move(m, LIFT)
    good = _audit(m, run, executed=run["executed"])
    q = run["executed"].qpos.copy()
    q[W + 1, 3] += 1.0
    bad = _audit(m, run, executed=A.Executed(q, run["executed"].qvel, run["executed"].ctrl))
    s = A.summarize_runs([good, bad])
    assert "positive control failed on 1 runs: <run 1>" in s["problems"]
    assert s["invalid_runs"] == ["<run 1>"] and s["grip_census"]["excluded_runs"] == {"<run 1>": "positive "
                                                                                       "control failed"}


def test_item8_the_minimum_quaternion_norm_is_reported_and_zero_is_refused() -> None:
    m = S.carry_model()
    run = _move(m, LIFT)
    ref = run["ref"].copy()
    ref[7, 3:] *= 0.5
    cert = A.make_certificate("CD", ref, warmup_rows=W, required_rows=N)
    r = A.audit_run(ref, cert, warmup_rows=W, quat_tol=None)
    assert r.settings["reference_nonunit_quaternions"] == 1
    assert abs(r.settings["reference_min_quaternion_norm"] - 0.5) <= 1e-12
    s = A.summarize_runs([r])
    assert abs(s["reference_min_quaternion_norm"] - 0.5) <= 1e-12 and "(min norm 0.5;" in s["problems"][0]
    ref[7, 3:] = 0.0
    with pytest.raises(A.InputError, match="exactly zero"):
        A.audit_run(ref, cert, warmup_rows=W, quat_tol=None)
