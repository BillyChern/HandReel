"""Regression tests of fastdo.audit for input validation and the replay anchor.

Each test names its item (F1-F11) or a mutant that survived an earlier version of the tests (m01, m03, m07, m09, m14; m05 is killed in
packages/nullaudit/tests/test_known_answer.py). The scenes are variants of `fastdo.audit.toy`.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest
from nullaudit.mujoco_adapter import QVEL_MAX, MujocoExecutor

import fastdo.audit as A
from fastdo.audit.__main__ import _json_safe, main
from tests.audit import _scene as S

N, W = 160, 20


# ------------------------------------------------------------------------------------------------ fixtures
def _settings(m: mujoco.MjModel) -> dict:
    return {"weld_eq_ids": (), "gravity": np.array(m.opt.gravity), "ctrl_steps": 1, "terminate_z": -np.inf}


def _execute(m: mujoco.MjModel, st: S.State, u: np.ndarray) -> dict:
    ex = MujocoExecutor(model=m, q0=st.qpos, hand_geom=S.masks(m)[0], ctrl0=st.ctrl, warmup_rows=W)
    ex(u)
    return {"u": u, "q0": st.qpos, "ctrl0": st.ctrl, "ref": ex.last_qpos[:, 1:8].copy(),
            "executed": A.Executed(ex.last_qpos, ex.last_qvel, ex.last_ctrl)}


def _lift(m: mujoco.MjModel) -> dict:
    st = S.carry(m)
    u = np.full((N, 1), st.ctrl[0])
    u[W:, 0] += np.linspace(0.0, 0.3, N - W)
    return _execute(m, st, u)


def _still(m: mujoco.MjModel) -> dict:
    """`pressed_still` of the worked example: the plate presses a cube that never moves."""
    st = S.press(m, depth=0.05)
    run = _execute(m, st, np.full((N, 1), st.ctrl[0]))
    run["ref"] = np.repeat(S.object_rows(st.qpos[None]), N, 0)
    return run


def _audit(m: mujoco.MjModel, run: dict, **kw) -> A.RunAudit:
    cert = A.make_certificate("CD", run["ref"], warmup_rows=W, required_rows=N)
    args = {"warmup_rows": W, "model": m, "hand_geom": S.masks(m)[0], "controls": run["u"], "q0": run["q0"],
            "ctrl0": run["ctrl0"], "removal": "displace", "executed": run["executed"], **_settings(m), **kw}
    return A.audit_run(run["ref"], cert, **args)


def _npz(tmp: Path, run: dict, xml: str, m: mujoco.MjModel, rows: bool = True, **extra) -> Path:
    (tmp / "scene.xml").write_text(xml)
    arrays = {"reference": run["ref"], "warmup_rows": np.int64(W), "required_rows": np.int64(N),
              "scene": np.str_("scene.xml"), "controls": run["u"], "q0": run["q0"], "ctrl0": run["ctrl0"],
              "weld_eq_ids": np.zeros(0, np.int64), "gravity": np.array(m.opt.gravity), "ctrl_steps": np.int64(1),
              "terminate_z": np.float64(-np.inf), "hand_geom": S.masks(m)[0], "sim_dt": np.float64(m.opt.timestep)}
    if rows:
        arrays.update(qpos=run["executed"].qpos, qvel=run["executed"].qvel, ctrl=run["executed"].ctrl)
    arrays.update(extra)
    p = tmp / "run.npz"
    np.savez(p, **arrays)
    return p


def _carry_npz(tmp: Path, **extra) -> Path:
    m = S.carry_model()
    return _npz(tmp, _lift(m), S.CARRY_XML, m, **extra)


def _strict(text: str) -> dict:
    def bad(c: str) -> None:
        raise AssertionError(f"non-standard JSON constant {c}")

    return json.loads(text, parse_constant=bad)


# ------------------------------------------------------------------------------------------------ F1
def test_f1_non_finite_controls_are_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The scenario: pressed_still with controls[20:] = NaN exited 0 with a 24.1x census."""
    m = S.model()
    run = _still(m)
    u = run["u"].copy()
    u[W:, 0] = np.nan
    assert main([str(_npz(tmp_path, dict(run, u=u), S.XML, m))]) == 2
    assert "controls has non-finite values" in capsys.readouterr().err
    u[W:, 0] = np.inf
    with pytest.raises(A.InputError, match="non-finite"):
        _audit(m, dict(run, u=u))
    assert main([str(_npz(tmp_path, run, S.XML, m))]) == 0                                 # green control


def test_f1_a_huge_control_makes_the_replay_invalid() -> None:
    m = S.carry_model()
    run = _lift(m)
    u = run["u"].copy()
    u[50, 0] = 1e12
    pc = _audit(m, dict(run, u=u)).to_dict()["positive_control"]
    assert pc["replay_status"] == "invalid" and pc["reasons"][0] == "the replay is invalid (MuJoCo warning " \
                                                                     "mjWARN_BADCTRL)"


# ------------------------------------------------------------------------------------------------ F2
@pytest.mark.parametrize(("w", "t"), [(-5, N), (N, N), (200, N), (0, 0), (-1, -3)])
def test_f2_warmup_and_required_rows_are_checked(tmp_path: Path, w: int, t: int) -> None:
    m = S.model()
    run = _still(m)
    cert = A.make_certificate("CD", run["ref"], warmup_rows=20, required_rows=N)
    with pytest.raises(A.InputError, match="warmup_rows"):
        A.audit_run(run["ref"], cert, warmup_rows=w, required_rows=t)
    p = _npz(tmp_path, run, S.XML, m, warmup_rows=np.int64(w), required_rows=np.int64(t))
    assert main([str(p)]) == 2


# ------------------------------------------------------------------------------------------------ F3
@pytest.mark.parametrize("dt", [np.nan, np.inf, 0.0, -0.005])
def test_f3_sim_dt_must_be_finite_and_positive(tmp_path: Path, dt: float) -> None:
    m = S.carry_model()
    with pytest.raises(A.InputError, match="sim_dt"):
        _audit(m, _lift(m), sim_dt=dt)
    assert main([str(_carry_npz(tmp_path, sim_dt=np.float64(dt)))]) == 2


def test_f3_the_report_is_strict_json(tmp_path: Path) -> None:
    """A replay that stops early has an infinite departure; the report stays strict JSON ("inf"), and NaN is null."""
    assert _json_safe({"a": [np.float64(np.nan), float("-inf"), np.float32(1.5)], "b": (np.int64(2),)}) == \
        {"a": [None, "-inf", 1.5], "b": [2]}
    p = _carry_npz(tmp_path, terminate_z=np.float64(0.2), controls=_lift(S.carry_model())["u"] - 0.3)
    out = tmp_path / "r.json"
    assert main([str(p), "--json", str(out)]) == 1
    doc = _strict(out.read_text())
    assert doc["runs"][0]["positive_control"]["max_dev"] == "inf"
    assert doc["runs"][0]["settings"]["terminate_z"] == 0.2


# ------------------------------------------------------------------------------------------------ F4
KP900 = S.CARRY_XML.replace('kp="300"', 'kp="900"')


def test_f4_a_sim_dt_alone_does_not_anchor_the_control(tmp_path: Path) -> None:
    """The scenario: the executor's servo has kp 900, the stored XML kp 300; with only sim_dt (no executed
    rows) the timestep matches and the certificate passes the replay, so it exited 0. Now it is a listed problem."""
    exe = S.model(KP900)
    run = _lift(exe)
    p = _npz(tmp_path, run, S.CARRY_XML, exe, rows=False)
    out = tmp_path / "r.json"
    assert main([str(p), "--json", str(out)]) == 1
    doc = _strict(out.read_text())
    pc = doc["runs"][0]["positive_control"]
    assert pc["passed"] is True and pc["anchors"] == [] and pc["certificate_only"] is True
    assert doc["summary"]["problems"] == [A.runner.CERT_ONLY + " on 1 runs: run"]
    p = _npz(tmp_path, run, S.CARRY_XML, exe)                  # with the executed rows the control fails
    assert main([str(p), "--json", str(out)]) == 1
    assert _strict(out.read_text())["runs"][0]["positive_control"]["passed"] is False


# ------------------------------------------------------------------------------------------------ F5
@pytest.mark.parametrize("field", ["qvel", "ctrl"])
def test_f5_replay_qvel_and_ctrl_are_compared(field: str) -> None:
    m = S.carry_model()
    run = _lift(m)
    ex = run["executed"]
    rows = {"qpos": ex.qpos, "qvel": ex.qvel.copy(), "ctrl": ex.ctrl.copy()}
    rows[field][N - 10, 0] += 1e-3
    pc = _audit(m, dict(run, executed=A.Executed(**rows))).to_dict()["positive_control"]
    assert pc["max_dev"] == 0.0 and pc[f"max_dev_{field}"] > 9e-4
    assert pc["reasons"] == (f"the replay's {field} departs from the executed rows by 0.001 > 1e-06",)


@pytest.mark.parametrize("field", ["qpos", "qvel", "ctrl"])
def test_f5_non_finite_executed_rows_are_refused(tmp_path: Path, field: str) -> None:
    m = S.carry_model()
    run = _lift(m)
    ex = run["executed"]
    rows = {"qpos": ex.qpos.copy(), "qvel": ex.qvel.copy(), "ctrl": ex.ctrl.copy()}
    rows[field][50, 0] = np.nan
    with pytest.raises(A.InputError, match=f"executed {field} has non-finite"):
        _audit(m, dict(run, executed=A.Executed(**rows)))
    assert main([str(_npz(tmp_path, run, S.CARRY_XML, m, **{field: rows[field]}))]) == 2


def test_f5_a_truncated_or_empty_census_is_a_problem() -> None:
    m = S.carry_model()
    r = _audit(m, _lift(m), sim_dt=m.opt.timestep)
    assert A.summarize_runs([r])["problems"] == []
    cut = dataclasses.replace(r, clip_id="cut", census=dataclasses.replace(r.census, truncated_at=90))
    none = dataclasses.replace(r, clip_id="none", census=dataclasses.replace(r.census, grip_over_weight_p50=None))
    p = A.summarize_runs([cut, none])["problems"]
    assert p == ["grip census stopped at a non-finite state on 1 runs: cut (row 90)",
                 "grip census sampled no post-release row on 1 runs: none"]


# ------------------------------------------------------------------------------------------------ F6
def _cases(base: dict, m: mujoco.MjModel) -> dict[str, dict]:
    ex = base["executed"]
    return {"q0_short": {"q0": base["q0"][:-1]}, "q0_2d": {"q0": base["q0"][None]},
            "gravity_len2": {"gravity": np.zeros(2)}, "ctrl0_len2": {"ctrl0": np.zeros(2)},
            "ctrl0_nan": {"ctrl0": np.array([np.nan])}, "qpos_cols": {"qpos": ex.qpos[:, :-1]},
            "qpos_1col": {"qpos": ex.qpos[:, :1]}, "qvel_short": {"qvel": ex.qvel[:10]},
            "ctrl_short": {"ctrl": ex.ctrl[:10]}, "qvel_cols": {"qvel": ex.qvel[:, :2]},
            "weld_oob": {"weld_eq_ids": np.array([7])}, "weld_neg": {"weld_eq_ids": np.array([-1])},
            "controls_str": {"controls": np.array([["a"]])}, "controls_1d": {"controls": base["u"][:, 0]},
            "ctrl_steps0": {"ctrl_steps": np.int64(0)}, "ctrl_steps_neg": {"ctrl_steps": np.int64(-1)},
            "terminate_z_nan": {"terminate_z": np.float64(np.nan)},
            "terminate_z_inf": {"terminate_z": np.float64(np.inf)}}


@pytest.mark.parametrize("case", sorted(_cases({"q0": np.zeros(2), "u": np.zeros((2, 1)),
                                                "executed": A.Executed(*[np.zeros((2, 2))] * 3)}, None)))
def test_f6_ill_shaped_executor_settings_are_exit_2(tmp_path: Path, case: str,
                                                    capsys: pytest.CaptureFixture[str]) -> None:
    m = S.carry_model()
    run = _lift(m)
    p = _npz(tmp_path, run, S.CARRY_XML, m, **_cases(run, m)[case])
    assert main([str(p)]) == 2, case
    err = capsys.readouterr().err
    assert err.startswith("error: ") and "Traceback" not in err, err


@pytest.mark.parametrize("stride", ["0", "-1"])
def test_f6_stride_must_be_positive(tmp_path: Path, stride: str) -> None:
    assert main([str(_carry_npz(tmp_path)), "--stride", stride]) == 2


def test_f6_ctrl0_is_not_silently_overwritten(tmp_path: Path) -> None:
    """The executor sets row 0's control before the first step, so a ctrl0 that differs from it has no effect: the
    report says so (exit 1)."""
    m = S.carry_model()
    run = _lift(m)
    out = tmp_path / "r.json"
    assert main([str(_npz(tmp_path, run, S.CARRY_XML, m, ctrl0=run["ctrl0"] + 0.1)), "--json", str(out)]) == 1
    doc = _strict(out.read_text())
    assert doc["runs"][0]["settings"]["ctrl0_overwritten"] is True
    assert doc["summary"]["problems"][0].startswith("ctrl0 differs from the first control row on 1 runs")


# ------------------------------------------------------------------------------------------------ F8-F11
@pytest.mark.parametrize("tol", ["nan", "inf", "-1e-6"])
def test_f8_replay_tol_must_be_finite_and_non_negative(tmp_path: Path, tol: str) -> None:
    assert main([str(_carry_npz(tmp_path)), f"--replay-tol={tol}"]) == 2


def test_f8_a_non_default_replay_tol_is_a_problem(tmp_path: Path) -> None:
    out = tmp_path / "r.json"
    assert main([str(_carry_npz(tmp_path)), "--replay-tol", "1e-3", "--json", str(out)]) == 1
    assert _strict(out.read_text())["summary"]["problems"] == ["non-default replay tolerance on 1 runs: run (0.001)"]
    assert main([str(_carry_npz(tmp_path)), "--replay-tol", "1e-6"]) == 0                  # green control


def test_f9_settings_record_the_constants() -> None:
    m = S.carry_model()
    st = _audit(m, _lift(m), census_stride=3).settings
    assert (st["census_stride"], st["quat_tol"], st["qvel_max"], st["timestep_tol"], st["f_rated_n"],
            st["replay_tol"]) == (3, A.QUAT_TOL, QVEL_MAX, A.runner.TIMESTEP_TOL, A.F_RATED_N, A.REPLAY_TOL)


def test_f10_npz_hand_away_conflicting_with_the_argument_is_refused(tmp_path: Path) -> None:
    p = _carry_npz(tmp_path, hand_away=np.array([0.0, 2.0]))
    assert main([str(p), "--removal", "pairs", "--hand-away", "0", "3"]) == 2
    assert main([str(p), "--removal", "pairs", "--hand-away", "0", "2"]) == 0               # the same: accepted
    assert main([str(p), "--removal", "pairs"]) == 0


def test_f10_npz_mask_conflicting_with_a_regex_is_refused(tmp_path: Path) -> None:
    p = _carry_npz(tmp_path)
    assert main([str(p), "--hand-geom", "palm"]) == 2
    assert main([str(p), "--hand-geom", "["]) == 2                                         # a bad regex


def test_f11_removal_model_needs_removed_model() -> None:
    m = S.carry_model()
    with pytest.raises(A.InputError, match="removed_model"):
        _audit(m, _lift(m), removal="model")


# ------------------------------------------------------------------------------------------------ surviving mutants
def test_m01_a_timestep_off_by_1e_4_fails_the_control() -> None:
    m = S.carry_model()
    pc = _audit(m, _lift(m), sim_dt=m.opt.timestep + 1e-4).to_dict()["positive_control"]
    assert pc["reasons"] == ("model timestep 0.005 != the run's sim_dt 0.0051",)


def test_m03_one_non_unit_quaternion_is_refused() -> None:
    m = S.carry_model()
    run = _lift(m)
    ref = run["ref"].copy()
    ref[33, 3:] *= 0.5
    with pytest.raises(A.InputError, match="1 quaternions off unit norm"):
        _audit(m, dict(run, ref=ref))
    assert _audit(m, dict(run, ref=ref), quat_tol=None).settings["reference_nonunit_quaternions"] == 1


def test_m07_a_bool_warmup_rows_is_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main([str(_carry_npz(tmp_path, warmup_rows=np.bool_(True)))]) == 2
    assert "`warmup_rows` must be an integer, got True" in capsys.readouterr().err


def test_m09_executed_rows_shorter_than_t_are_refused() -> None:
    m = S.carry_model()
    run = _lift(m)
    ex = run["executed"]
    short = A.Executed(ex.qpos[:N - 10], ex.qvel[:N - 10], ex.ctrl[:N - 10])
    with pytest.raises(A.InputError, match="executed rows has 150 rows"):
        _audit(m, dict(run, executed=short))


def test_m14_a_departure_in_the_last_rows_fails_the_control() -> None:
    m = S.carry_model()
    run = _lift(m)
    q = run["executed"].qpos.copy()
    q[N - 3, 3] += 1e-3
    pc = _audit(m, dict(run, executed=dataclasses.replace(run["executed"], qpos=q))).to_dict()["positive_control"]
    assert pc["reasons"] == ("the replay departs from the executed rows by 0.001 > 1e-06",)
