"""Regression tests of fastdo.audit for silent failures and correctness of the CLI and the executor checks.

Each test names its item. Every one failed on an earlier version of the tool and passes now; each has a
green control that passed on both. The scenes are variants of `fastdo.audit.toy`, written out here so that the tests
do not depend on the code they check.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import mujoco
import numpy as np
import pytest
from nullaudit.mujoco_adapter import MujocoExecutor

import fastdo.audit as A
from fastdo.audit.__main__ import main
from tests.audit import _scene as S

TREE = Path(__file__).resolve().parents[2]
N, W = 160, 20


# ------------------------------------------------------------------------------------------------ fixtures
def _lift(m: mujoco.MjModel, hand: np.ndarray, q0: np.ndarray, ctrl0: np.ndarray) -> tuple[np.ndarray, MujocoExecutor]:
    """Execute a 0.3 m lift; return the controls and the executor holding the executed rows."""
    u = np.full((N, 1), ctrl0[0])
    u[W:, 0] += np.linspace(0.0, 0.3, N - W)
    ex = MujocoExecutor(model=m, q0=q0, hand_geom=hand, ctrl0=ctrl0, warmup_rows=W)
    ex(u)
    return u, ex


def _settings(m: mujoco.MjModel) -> dict:
    return {"weld_eq_ids": (), "gravity": np.array(m.opt.gravity), "ctrl_steps": 1, "terminate_z": -np.inf}


def _carry_run(m: mujoco.MjModel, hand: np.ndarray) -> dict:
    st = S.carry(m)
    u, ex = _lift(m, hand, st.qpos, st.ctrl)
    ref = ex.last_qpos[:, 1:8].copy()
    return {"ref": ref, "u": u, "q0": st.qpos, "ctrl0": st.ctrl,
            "executed": A.Executed(ex.last_qpos, ex.last_qvel, ex.last_ctrl)}


def _write_npz(path: Path, run: dict, scene: str, m: mujoco.MjModel, **extra) -> Path:
    arrays = {"reference": run["ref"], "warmup_rows": np.int64(W), "required_rows": np.int64(N),
              "scene": np.str_(scene), "controls": run["u"], "q0": run["q0"], "ctrl0": run["ctrl0"],
              "weld_eq_ids": np.zeros(0, np.int64), "gravity": np.array(m.opt.gravity), "ctrl_steps": np.int64(1),
              "terminate_z": np.float64(-np.inf), "hand_geom": S.masks(m)[0],
              "qpos": run["executed"].qpos, "qvel": run["executed"].qvel, "ctrl": run["executed"].ctrl, **extra}
    np.savez(path, **arrays)
    return path


def _loader_edited(m: mujoco.MjModel) -> mujoco.MjModel:
    """What a pipeline's loader does to the compiled scene (DO AS I DO: timestep, integrator)."""
    m.opt.timestep = 0.0025
    m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    return m


# ------------------------------------------------------------------------------------------------ items 1 and 2
def test_item1_cli_refuses_a_stored_xml_the_executor_did_not_run(tmp_path: Path) -> None:
    (tmp_path / "carry.xml").write_text(S.CARRY_XML)
    exe = _loader_edited(S.carry_model())
    run = _carry_run(exe, S.masks(exe)[0])
    p = _write_npz(tmp_path / "run.npz", run, "carry.xml", exe, sim_dt=np.float64(exe.opt.timestep))
    out = tmp_path / "r.json"
    assert main([str(p), "--json", str(out)]) == 1
    r = json.loads(out.read_text())["runs"][0]
    pc = r["positive_control"]
    assert pc["passed"] is False and pc["max_dev"] > A.REPLAY_TOL
    assert any("timestep" in x for x in pc["reasons"])
    assert r["contact_removed_null"]["status"] == "invalid" and r["grip_census"]["status"] == "not_run"


@pytest.mark.parametrize("removal", ["displace", "xml"])
def test_item2_cli_simulates_the_executor_model(tmp_path: Path, removal: str) -> None:
    (tmp_path / "carry.xml").write_text(S.CARRY_XML)
    exe = _loader_edited(S.carry_model())
    mujoco.mj_saveModel(exe, str(tmp_path / "executor.mjb"), None)
    run = _carry_run(exe, S.masks(exe)[0])
    p = _write_npz(tmp_path / "run.npz", run, "carry.xml", exe, sim_dt=np.float64(exe.opt.timestep),
                   executor_model=np.str_("executor.mjb"))
    out = tmp_path / "r.json"
    assert main([str(p), "--json", str(out), "--removal", removal]) == 0
    doc = json.loads(out.read_text())
    r = doc["runs"][0]
    assert r["positive_control"]["passed"] is True and r["positive_control"]["max_dev"] == 0.0
    assert r["contact_removed_null"]["status"] == "fail" and r["contact_removed_null"]["verified_n0"] is True
    if removal == "xml":
        assert {"opt.timestep", "opt.integrator"} <= set(doc["inputs"]["runs"][0]["matched_from_executor"])


def test_items1_2_control_unedited_executor(tmp_path: Path) -> None:
    """Green control: the executor ran the XML as stored; the CLI audits it with exit 0."""
    (tmp_path / "carry.xml").write_text(S.CARRY_XML)
    m = S.carry_model()
    p = _write_npz(tmp_path / "run.npz", _carry_run(m, S.masks(m)[0]), "carry.xml", m)
    out = tmp_path / "r.json"
    assert main([str(p), "--json", str(out)]) == 0
    r = json.loads(out.read_text())["runs"][0]
    assert not r["z0"] and r["contact_removed_null"]["status"] == "fail"


# ------------------------------------------------------------------------------------------------ item 3
PAD = '<geom name="palm_pad" type="box" size="0.04 0.04 0.002" pos="0 0 0.012" mass="0.01"/>'
PAD_PAIR = '<pair name="pad_object" geom1="palm_pad" geom2="right_object_cube" condim="3"/>'


def _pad_model() -> mujoco.MjModel:
    xml = S.CARRY_XML.replace('<geom name="collision_hand_palm"', PAD + '\n      <geom name="collision_hand_palm"')
    return mujoco.MjModel.from_xml_string(xml.replace("<contact>", "<contact>\n    " + PAD_PAIR))


def _pad_run(m: mujoco.MjModel, hand: np.ndarray, **kw):
    q = np.array(m.qpos0)
    q[0], q[1:4], q[4:8] = 0.3, (0.0, 0.0, 0.3 + 0.014 + S.HALF + 0.001), (1, 0, 0, 0)
    d = mujoco.MjData(m)
    d.qpos[:] = q
    d.ctrl[0] = 0.3
    for _ in range(4000):
        mujoco.mj_step(m, d)
    u, ex = _lift(m, hand, d.qpos.copy(), d.ctrl.copy())
    ref = ex.last_qpos[:, 1:8].copy()
    cert = A.make_certificate("CD", ref, warmup_rows=W, required_rows=N)
    return A.audit_run(ref, cert, warmup_rows=W, model=m, hand_geom=hand, controls=u, q0=d.qpos.copy(),
                       ctrl0=d.ctrl.copy(), removal="displace", **_settings(m), **kw)


def test_item3_a_hand_geom_outside_the_mask_is_not_a_null() -> None:
    m = _pad_model()
    palm_only = S.masks(m)[0]
    d = _pad_run(m, palm_only).to_dict()
    assert d["contact_removed_null"]["status"] == "invalid", d["contact_removed_null"]
    assert d["contact_removed_null"]["other_object_force"] > 0
    assert 0.9 < d["grip_census"]["grip_over_weight_p50"] < 1.1        # the pad carries the cube: 1x weight
    assert d["positive_control"]["outside_mask_force_n_max"] > 0
    paths = A.contact_paths(m, palm_only, A.free_object_geom(m))
    assert paths["explicit_pairs"] == 2 and paths["outside_mask_explicit_pairs"] == 1


def test_item3_control_full_hand_mask() -> None:
    m = _pad_model()
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(m.ngeom)]
    hand = np.array([n in ("collision_hand_palm", "palm_pad") for n in names])
    d = _pad_run(m, hand).to_dict()
    assert d["contact_removed_null"]["status"] == "fail" and d["contact_removed_null"]["verified_n0"] is True


# ------------------------------------------------------------------------------------------------ item 4
def _pair_id(m: mujoco.MjModel, name: str) -> int:
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_PAIR, name)


def test_item4_surviving_pair_rows_and_stat_come_from_the_executor(tmp_path: Path) -> None:
    (tmp_path / "s.xml").write_text(S.XML)
    exe = S.model()
    exe.pair_friction[_pair_id(exe, "object_pedestal")] = (0.1, 0.1, 0.001, 0.0001, 0.0001)
    exe.stat.meaninertia *= 2.0
    h, o = S.masks(exe)
    removed, rec = A.compile_without_pairs(tmp_path / "s.xml", h, o, executor_model=exe)
    p = _pair_id(removed, "object_pedestal")
    assert np.array_equal(removed.pair_friction[p], exe.pair_friction[_pair_id(exe, "object_pedestal")])
    assert removed.stat.meaninertia == exe.stat.meaninertia
    assert "stat.meaninertia" in rec["matched_from_executor"]
    assert any(k.startswith("pair[") and k.endswith(".friction") for k in rec["matched_from_executor"])


def test_item4_a_plain_xml_removal_is_refused_against_an_edited_executor(tmp_path: Path) -> None:
    exe = S.model()
    exe.pair_friction[_pair_id(exe, "object_pedestal")] = (0.1, 0.1, 0.001, 0.0001, 0.0001)
    edited, _ = A.remove_pairs(S.XML, ["hand_object"])
    st = S.press(exe)
    ref = np.repeat(S.object_rows(st.qpos[None]), N, 0)
    cert = A.make_certificate("CD", ref, warmup_rows=W, required_rows=N)
    with pytest.raises(ValueError, match="friction"):
        A.audit_run(ref, cert, warmup_rows=W, model=exe, hand_geom=S.masks(exe)[0],
                    controls=np.full((N, 1), st.ctrl[0]), q0=st.qpos, ctrl0=st.ctrl, removal="model",
                    removed_model=S.model(edited), **_settings(exe))


def test_item4_layout_mismatch_raises(tmp_path: Path) -> None:
    (tmp_path / "s.xml").write_text(S.XML)
    other = S.model(S.XML.replace("</worldbody>", '<geom name="extra" type="sphere" size="0.01" pos="1 1 1"/>'
                                                  "</worldbody>"))
    with pytest.raises(ValueError, match="layout"):
        A.compile_without_pairs(tmp_path / "s.xml", *S.masks(S.model()), executor_model=other)


def test_item4_control_same_scene(tmp_path: Path) -> None:
    (tmp_path / "s.xml").write_text(S.XML)
    m = S.model()
    removed, rec = A.compile_without_pairs(tmp_path / "s.xml", *S.masks(m))
    assert rec["pairs_removed"] == 1 and removed.npair == m.npair - 1


# ------------------------------------------------------------------------------------------------ item 5
def _press_inputs() -> tuple[mujoco.MjModel, S.State, np.ndarray, object]:
    m = S.model()
    st = S.press(m)
    ref = np.repeat(S.object_rows(st.qpos[None]), N, 0)
    return m, st, ref, A.make_certificate("CD", ref, warmup_rows=W, required_rows=N)


@pytest.mark.parametrize("missing", ["q0", "weld_eq_ids"])
def test_item5_required_executor_settings(missing: str) -> None:
    m, st, ref, cert = _press_inputs()
    kw = {"q0": st.qpos, "weld_eq_ids": ()}
    kw.pop(missing)
    with pytest.raises(ValueError, match=missing):
        A.audit_run(ref, cert, warmup_rows=W, model=m, hand_geom=S.masks(m)[0],
                    controls=np.full((N, 1), st.ctrl[0]), ctrl0=st.ctrl, **kw)


def test_item5_defaults_are_reported() -> None:
    m, st, ref, cert = _press_inputs()
    r = A.audit_run(ref, cert, warmup_rows=W, model=m, hand_geom=S.masks(m)[0],
                    controls=np.full((N, 1), st.ctrl[0]), q0=st.qpos, weld_eq_ids=())
    assert set(r.to_dict()["defaulted"]) == {"ctrl0", "gravity", "ctrl_steps", "terminate_z"}
    assert any("defaults used" in p for p in A.summarize_runs([r])["problems"])


def test_item5_cli_pairs_needs_hand_away(tmp_path: Path) -> None:
    m = S.carry_model()
    (tmp_path / "carry.xml").write_text(S.CARRY_XML)
    p = _write_npz(tmp_path / "run.npz", _carry_run(m, S.masks(m)[0]), "carry.xml", m)
    assert main([str(p), "--removal", "pairs"]) == 2
    assert main([str(p), "--removal", "pairs", "--hand-away", "0", "2.0"]) == 0          # green control


# ------------------------------------------------------------------------------------------------ item 6
def test_item6_incomplete_c289_is_labelled_and_mt_null_is_not_a_warning() -> None:
    _, _, ref, _ = _press_inputs()
    cert = A.make_certificate("C289", ref, warmup_rows=W, required_rows=N)
    r = A.audit_run(ref, cert, warmup_rows=W, clip_id="mt", certificate_name="C289")
    d = r.to_dict()
    assert d["certificate"] == "C289[obj+M_i+informative]" and d["certificate_complete"] is False
    assert d["certificate_checked"] == ["obj", "motion", "informative"] and d["reference_informative"] is False
    s = A.summarize_runs([r])
    assert "C289[obj+M_i+informative]" in s["headline"] and "WARNING" not in s["headline"]
    assert s["reference_rejected"] == [] and s["reference_rejected_by_design"] == ["mt"]


# ------------------------------------------------------------------------------------------------ item 7
def test_item7_lengths_and_quaternions() -> None:
    m, st, ref, cert = _press_inputs()
    with pytest.raises(ValueError, match="rows"):
        A.audit_run(ref[:N - 1], A.make_certificate("CD", ref[:N - 1], warmup_rows=W, required_rows=None),
                    warmup_rows=W, required_rows=N)
    with pytest.raises(ValueError, match="controls"):
        A.audit_run(ref, cert, warmup_rows=W, required_rows=N, model=m, hand_geom=S.masks(m)[0],
                    controls=np.full((N - 1, 1), st.ctrl[0]), q0=st.qpos, weld_eq_ids=())
    with pytest.raises(ValueError, match="T > W"):
        A.make_certificate("C289", ref, warmup_rows=W, required_rows=W + 50)
    bad = ref.copy()
    bad[5, 3:] = 0.0
    with pytest.raises(ValueError, match="quaternion"):
        A.audit_run(bad, A.make_certificate("CD", bad, warmup_rows=W, required_rows=N), warmup_rows=W)
    assert A.audit_run(ref, cert, warmup_rows=W).clip.z0                                  # green control


# ------------------------------------------------------------------------------------------------ item 8
def test_item8_kinematic_clips_count_as_not_run() -> None:
    m, st, ref, cert = _press_inputs()
    sim = A.audit_run(ref, cert, warmup_rows=W, model=m, hand_geom=S.masks(m)[0],
                      controls=np.full((N, 1), st.ctrl[0]), q0=st.qpos, ctrl0=st.ctrl, **_settings(m))
    kin = A.audit_run(ref, cert, warmup_rows=W, clip_id="kin")
    s = A.summarize_runs([sim, kin])
    assert s["rates"]["hand_absent"]["not_run"] == 1 and "1 not run" in s["headline"]


# ------------------------------------------------------------------------------------------------ item 9
def test_item9_exit_codes(tmp_path: Path) -> None:
    _, _, ref, _ = _press_inputs()
    p = tmp_path / "k.npz"
    np.savez(p, reference=ref, warmup_rows=np.int64(W), required_rows=np.int64(N))
    cfg = tmp_path / "strict.json"
    cfg.write_text(json.dumps({"preset": "CD", "pos_m": -1.0}))     # rejects everything, its reference included
    assert main([str(p), "--certificate", str(cfg)]) == 1
    assert main([str(p), "--certificate", str(cfg), "--allow-warnings"]) == 0
    assert main([str(p)]) == 0                                                            # green control


# ------------------------------------------------------------------------------------------------ item 10
AFTER_OBJECT = ('<body name="slider" pos="1 0 0"><joint name="sx" type="slide" axis="1 0 0"/>'
                '<geom name="slider_g" type="sphere" size="0.01" mass="0.1"/></body>')


def test_item10_census_fields_qadr_weld_masks_gravity() -> None:
    m = S.carry_model()
    st = S.carry(m)
    h, o = S.masks(m)
    q, v, c = S.rows([st] * 6)
    q[4, 0] = np.nan
    g = A.grip_census(m, q, v, c, hand_geom=h, object_geom=o, release_row=0, stride=1)
    assert g.truncated_at == 4 and g.rows_sampled == 4 and g.replay_status == "executed"
    # the object is not the last joint: tracking must read the free joint's columns
    m2 = S.model(S.CARRY_XML.replace("</worldbody>", AFTER_OBJECT + "</worldbody>"))
    st2 = S.carry(m2)
    q2, v2, c2 = S.rows([st2] * 3)
    g2 = A.grip_census(m2, q2, v2, c2, hand_geom=S.masks(m2)[0], object_geom=A.free_object_geom(m2),
                       release_row=0, stride=1, ref_pos=q2[:, 1:4])
    assert g2.frac_rows_tracked == 1.0
    # the warm-up weld is off after release
    m3 = S.model(S.CARRY_XML.replace("</actuator>", '</actuator><equality><weld body1="right_object"/></equality>'))
    q3, v3, c3 = S.rows([st])                          # the weld-free carry state: with the weld off, exactly 1x
    g3 = A.grip_census(m3, q3, v3, c3, hand_geom=S.masks(m3)[0], object_geom=A.free_object_geom(m3),
                       release_row=0, stride=1, weld_eq_ids=(0,))
    assert abs(g3.grip_over_weight_p50 - 1.0) <= 1e-9, g3.grip_over_weight_p50
    # an object mask that is not the free body is refused; the model's gravity is restored
    with pytest.raises(ValueError, match="free joint"):
        A.grip_census(m, q, v, c, hand_geom=h, object_geom=o | ~h, release_row=0)     # the pedestal is not the object
    before = np.array(m.opt.gravity)
    A.grip_census(m, q[:3], v[:3], c[:3], hand_geom=h, object_geom=o, release_row=0, stride=1,
                  gravity=np.array([0.0, 0.0, -1.0]))
    assert np.array_equal(m.opt.gravity, before)


def test_item10_object_submask_must_not_hide_contact() -> None:
    """A geom of the object body may be left out of the object mask only if nothing dynamic can touch it."""
    marker = '<geom name="marker_g" type="sphere" size="0.005" pos="0 0 0.03"/>'
    m = S.model(S.XML.replace('<geom name="right_object_cube"', marker + '\n      <geom name="right_object_cube"'))
    h, o = S.masks(m)
    assert A.check_object_mask(m, o, h).sum() == 1                                 # unpaired marker: accepted
    lip = S.XML.replace('<geom name="right_object_cube"', marker + '\n      <geom name="right_object_cube"').replace(
        "<contact>", '<contact><pair name="hand_marker" geom1="collision_hand_palm" geom2="marker_g"/>')
    m2 = S.model(lip)
    with pytest.raises(ValueError, match="inconsistent"):
        A.check_object_mask(m2, S.masks(m2)[1], S.masks(m2)[0])


def test_item10_summary_counts_excluded_runs() -> None:
    m, st, ref, cert = _press_inputs()
    sim = A.audit_run(ref, cert, warmup_rows=W, model=m, hand_geom=S.masks(m)[0],
                      controls=np.full((N, 1), st.ctrl[0]), q0=st.qpos, ctrl0=st.ctrl, **_settings(m))
    kin = A.audit_run(ref, cert, warmup_rows=W, clip_id="kin")
    g = A.summarize_runs([sim, kin])["grip_census"]
    assert g["runs"] == 1 and g["excluded"] == 1 and "kin" in g["excluded_runs"]


# ------------------------------------------------------------------------------------------------ item 11
OBJ = "v 0 0 0\nv 0.02 0 0\nv 0 0.02 0\nv 0 0 0.02\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n"


def test_item11_no_temporary_file_next_to_the_scene(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "tet.obj").write_text(OBJ)
    xml = S.XML.replace("<worldbody>", '<asset><mesh name="tet" file="tet.obj"/></asset>\n  <worldbody>').replace(
        '<geom name="right_pedestal_start"', '<geom name="deco" type="mesh" mesh="tet" pos="1 1 0"/>\n    '
        '<geom name="right_pedestal_start"')
    (tmp_path / "s.xml").write_text(xml)
    real = tempfile.mkstemp

    def guarded(*a, **kw):
        assert Path(kw.get("dir") or tempfile.gettempdir()).resolve() != tmp_path.resolve(), "temp file in scene dir"
        return real(*a, **kw)

    monkeypatch.setattr(tempfile, "mkstemp", guarded)
    m = mujoco.MjModel.from_xml_path(str(tmp_path / "s.xml"))
    removed, rec = A.compile_without_pairs(tmp_path / "s.xml", *S.masks(m))
    assert rec["pairs_removed"] == 1 and removed.nmesh == 1
    assert sorted(p.name for p in tmp_path.iterdir()) == ["s.xml", "tet.obj"]


LID = ('<body name="lid" pos="0 0.2 0.2"><joint name="ly" type="slide" axis="0 1 0"/>'
       '<geom name="lid_g" type="box" size="0.02 0.02 0.02" mass="0.1"/></body>')
LID_PAIR = '<pair name="lid_object" geom1="lid_g" geom2="right_object_cube"/>'


def test_item11_a_third_dynamic_body_is_refused() -> None:
    m = S.model(S.XML.replace("</worldbody>", LID + "</worldbody>").replace("<contact>", "<contact>" + LID_PAIR))
    with pytest.raises(ValueError, match="third dynamic body"):
        A.classify_pairs(m, S.masks(m)[0], A.free_object_geom(m))
    two = S.XML.replace("</worldbody>", '<body name="ball" pos="1 0 1"><freejoint/>'
                                        '<geom name="ball_g" type="sphere" size="0.01"/></body></worldbody>')
    with pytest.raises(ValueError, match="free joint"):
        A.classify_pairs(S.model(two))
    assert A.hand_object_pair_names(A.classify_pairs(S.model())) == ["hand_object"]          # green control


def test_item11_nullaudit_is_a_path_dependency() -> None:
    text = (TREE / "pyproject.toml").read_text()
    assert '"nullaudit @ {root:uri}/packages/nullaudit"' in text


# ------------------------------------------------------------------------------------ duplicate run names
def test_cli_refuses_two_run_files_with_one_name(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A run is named by its file stem and the pooled report keys per-run entries by that name."""
    m = S.carry_model()
    run = _carry_run(m, S.masks(m)[0])
    paths = []
    for d in ("a", "b"):
        (tmp_path / d).mkdir()
        (tmp_path / d / "carry.xml").write_text(S.CARRY_XML)
        paths.append(str(_write_npz(tmp_path / d / "run.npz", run, "carry.xml", m)))
    assert main(paths) == 2
    assert "run files share a name: run" in capsys.readouterr().err
    assert main(paths[:1]) == 0


def test_summary_refuses_two_runs_with_one_clip_id() -> None:
    m = S.carry_model()
    run = _carry_run(m, S.masks(m)[0])
    cert = A.make_certificate("CD", run["ref"], warmup_rows=W, required_rows=N)

    def one(clip_id: str) -> A.RunAudit:
        return A.audit_run(run["ref"], cert, clip_id=clip_id, warmup_rows=W, model=m, hand_geom=S.masks(m)[0],
                           controls=run["u"], q0=run["q0"], ctrl0=run["ctrl0"], removal="displace",
                           executed=run["executed"], **_settings(m))

    with pytest.raises(ValueError, match="runs share an id: same"):
        A.summarize_runs([one("same"), one("same")])
    assert A.summarize_runs([one("same"), one("other")])["problems"] == []
    assert A.summarize_runs([one(""), one("")])["problems"] == []      # empty ids fall back to positions


# ------------------------------------------------------------------------- fields nothing would read
@pytest.mark.parametrize("field", ["controls", "q0", "sim_dt", "qpos", "hand_geom"])
def test_cli_refuses_a_simulation_field_without_a_scene(tmp_path: Path, field: str,
                                                        capsys: pytest.CaptureFixture[str]) -> None:
    """Without `scene` or `executor_model` only the motionless-object null runs; a simulation field would go unread."""
    m = S.carry_model()
    run = _carry_run(m, S.masks(m)[0])
    base = {"reference": run["ref"], "warmup_rows": np.int64(W), "required_rows": np.int64(N)}
    value = {"controls": run["u"], "q0": run["q0"], "sim_dt": np.float64(m.opt.timestep),
             "qpos": run["executed"].qpos, "hand_geom": S.masks(m)[0]}[field]
    np.savez(tmp_path / "bad.npz", **base, **{field: value})
    assert main([str(tmp_path / "bad.npz")]) == 2
    err = capsys.readouterr().err
    assert field in err and "neither `scene` nor `executor_model`" in err
    np.savez(tmp_path / "kin.npz", **base)
    assert main([str(tmp_path / "kin.npz")]) in (0, 1)              # control: the kinematic file is still audited


@pytest.mark.parametrize("dropped", ["qpos", "qvel", "ctrl"])
def test_cli_refuses_a_partial_stored_trace(tmp_path: Path, dropped: str, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "carry.xml").write_text(S.CARRY_XML)
    m = S.carry_model()
    p = _write_npz(tmp_path / "run.npz", _carry_run(m, S.masks(m)[0]), "carry.xml", m)
    with np.load(p) as z:
        arrays = {k: z[k] for k in z.files if k != dropped}
    np.savez(tmp_path / "part.npz", **arrays)
    assert main([str(tmp_path / "part.npz")]) == 2
    assert f"['{dropped}'] missing" in capsys.readouterr().err
    assert main([str(p)]) == 0                                       # control: the full trace is accepted


def test_cli_accepts_a_scene_free_frame_grid_reference_with_its_sim_dt(tmp_path: Path,
                                                                     capsys: pytest.CaptureFixture[str]) -> None:
    """`frames` with `ref_dt`, `sim_dt` and `n_rows` is a supported reference without a scene, and
    `sim_dt` is read there. With `reference` given instead, a scene-free `sim_dt` is still refused, and so is a
    scene-free `hand_away` under --removal pairs, which only the simulated removal reads."""
    m = S.carry_model()
    run = _carry_run(m, S.masks(m)[0])
    dt = float(m.opt.timestep)
    grid = {"frames": run["ref"][W:], "ref_dt": np.float64(dt), "sim_dt": np.float64(dt),
            "n_rows": np.int64(run["ref"].shape[0]), "warmup_rows": np.int64(W), "required_rows": np.int64(N)}
    np.savez(tmp_path / "grid.npz", **grid)
    out = tmp_path / "r.json"
    assert main([str(tmp_path / "grid.npz"), "--json", str(out)]) in (0, 1)
    assert json.loads(out.read_text())["runs"][0]["contact_removed_null"]["status"] == "not_run"
    capsys.readouterr()
    np.savez(tmp_path / "pairs.npz", **grid, hand_away=np.array([0.0, 0.5]))
    assert main([str(tmp_path / "pairs.npz"), "--removal", "pairs"]) == 2
    assert "hand_away" in capsys.readouterr().err


@pytest.mark.parametrize("extra", ["frames", "ref_dt", "n_rows"])
def test_cli_refuses_two_encodings_of_the_reference(tmp_path: Path, extra: str,
                                                    capsys: pytest.CaptureFixture[str]) -> None:
    """With `reference` present the frame-grid fields would be recognised and never read."""
    m = S.carry_model()
    run = _carry_run(m, S.masks(m)[0])
    base = {"reference": run["ref"], "warmup_rows": np.int64(W), "required_rows": np.int64(N)}
    value = {"frames": run["ref"][W:], "ref_dt": np.float64(m.opt.timestep), "n_rows": np.int64(run["ref"].shape[0])}
    np.savez(tmp_path / "both.npz", **base, **{extra: value[extra]})
    assert main([str(tmp_path / "both.npz")]) == 2
    err = capsys.readouterr().err
    assert "two encodings of the reference" in err and extra in err


@pytest.mark.parametrize(("field", "bad", "says"), [
    ("ref_dt", 0.0, "positive finite number of seconds"), ("ref_dt", -0.01, "positive finite number of seconds"),
    ("sim_dt", 0.0, "positive finite number of seconds"), ("sim_dt", -0.002, "positive finite number of seconds"),
    ("sim_dt", np.inf, "positive finite number of seconds"), ("sim_dt", np.nan, "positive finite number of seconds"),
    ("ref_dt", np.inf, "positive finite number of seconds"),
    ("n_rows", -1, "`n_rows` >= 1"), ("n_rows", 0, "`n_rows` >= 1"), ("warmup_rows", -1, "`warmup_rows` must be >= 0"),
])
def test_cli_refuses_an_impossible_frame_grid_before_building_it(tmp_path: Path, field: str, bad: float, says: str,
                                                                 capsys: pytest.CaptureFixture[str]) -> None:
    """A zero, negative or non-finite step, or an impossible row count, is refused as bad input
    (exit 2) with one message; none reaches the interpolation, where it would give a grid or an internal error."""
    m = S.carry_model()
    run = _carry_run(m, S.masks(m)[0])
    dt = float(m.opt.timestep)
    grid = {"frames": run["ref"][W:], "ref_dt": np.float64(dt), "sim_dt": np.float64(dt),
            "n_rows": np.int64(run["ref"].shape[0]), "warmup_rows": np.int64(W), "required_rows": np.int64(N)}
    np.savez(tmp_path / "ok.npz", **grid)
    assert main([str(tmp_path / "ok.npz")]) in (0, 1)                # control: the unedited grid is accepted
    capsys.readouterr()
    np.savez(tmp_path / "bad.npz", **{**grid, field: np.asarray(bad)})
    assert main([str(tmp_path / "bad.npz")]) == 2
    err = capsys.readouterr().err
    assert says in err and "Traceback" not in err
