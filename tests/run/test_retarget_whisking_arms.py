"""Tests for `run/pipeline/retarget_whisking.py` for the kinematic-retargeting arm: a third arm prepared from our
current reference (`isolation.our_arm`) without touching the recorded C4 arms; `run_arm`'s guard that
pins a new arm's env count to a recorded arm's `arm_meta.json`; and the C4 review follow-ups (a report
row for an arm that ended during warmup, TypeError on a non-mapping legacy config). No baseline
execution; the heavy retargeting and statistics calls are replaced by stand-ins.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import DictConfig

from fastdo.retarget.isolation import KinematicStats, processed_data_dir
from fastdo.retarget.kinematic import KinematicTrajectory
from tests.run.test_retarget_whisking import _FAKE_DRIVER, _make_cfg, _write_arm_artifacts, rw


@pytest.fixture
def fake_driver(tmp_path: Path) -> Path:
    path = tmp_path / "fake_driver.py"
    path.write_text(_FAKE_DRIVER)
    return path


def _write_recorded_meta(cfg: DictConfig, arm: str, **fields: object) -> None:
    data_dir = processed_data_dir(rw._arm_output_root(cfg, arm), "sharpa", "right", "whisking_c4", 0)
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "arm_meta.json").write_text(json.dumps({"task": "whisking_c4", **fields}))


def test_stage_run_arm_runs_a_configured_third_arm_with_the_recorded_env_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_driver: Path,
) -> None:
    """The re-run adds an arm; `match_recorded_arm` pins its env count to a recorded arm's
    `arm_meta.json` ("default(...)" there means no override), so the comparison stays controlled."""
    cfg = _make_cfg(tmp_path, arm="ours_clearance", arms=["theirs", "ours", "ours_clearance"],
                    match_recorded_arm="theirs", num_perturb_samples=2)
    _write_recorded_meta(cfg, "theirs", num_samples="default(1024)", num_perturb_samples=2)
    seen: list[list[str]] = []

    def _fake_conda_run_cmd(cfg: DictConfig, *args: str) -> list[str]:
        seen.append(list(args))
        return [sys.executable, str(fake_driver), *args]

    monkeypatch.setattr(rw, "_conda_run_cmd", _fake_conda_run_cmd)
    rw._stage_run_arm(cfg)
    args = seen[0]
    assert args[args.index("--output-root-dir") + 1] == str(rw._arm_output_root(cfg, "ours_clearance"))
    assert args[args.index("--num-perturb-samples") + 1] == "2" and "--num-samples" not in args


@pytest.mark.parametrize(("overrides", "match"), [
    ({"num_perturb_samples": 4}, "num_perturb_samples"),
    ({"num_perturb_samples": 2, "num_samples": 512}, "num_samples"),
])
def test_stage_run_arm_refuses_env_settings_that_differ_from_the_recorded_arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, overrides: dict[str, object], match: str,
) -> None:
    cfg = _make_cfg(tmp_path, arm="ours_clearance", arms=["theirs", "ours_clearance"],
                    match_recorded_arm="theirs", **overrides)
    _write_recorded_meta(cfg, "theirs", num_samples="default(1024)", num_perturb_samples=2)
    monkeypatch.setattr(rw, "_conda_run_cmd", lambda *a: pytest.fail("must refuse before launching"))
    with pytest.raises(ValueError, match=match):
        rw._stage_run_arm(cfg)


def test_stage_prepare_writes_only_our_arm_and_merges_the_stats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preparing the new arm must not touch the recorded arms: it writes `{our_arm}_trajectory_
    kinematic.npz` and its retarget config, sets up only arms whose directory lacks a reference, and
    updates `prepare_stats.json` per arm (sampled-point and primitive statistics) without dropping
    entries for arms it does not recompute."""
    demo = tmp_path / "demo"
    demo.mkdir()
    (demo / "scene.xml").write_text("<mujoco/>")
    (demo / "assets").mkdir()
    shipped = demo / "trajectory_kinematic.npz"
    KinematicTrajectory(qpos=np.zeros((4, 35)), qvel=np.zeros((4, 34)), frequency=50.0).save(shipped)
    cfg = _make_cfg(tmp_path, our_arm="ours_clearance", arms=["theirs", "ours", "ours_clearance"],
                    mesh_scale=1.0, demo_scene_xml=str(demo / "scene.xml"), demo_assets_dir=str(demo / "assets"),
                    demo_trajectory_kinematic_npz=str(shipped), demo_keypoints_npz=str(shipped),
                    mano_right_pkl="unused")
    theirs_dir = processed_data_dir(rw._arm_output_root(cfg, "theirs"), "sharpa", "right", "whisking_c4", 0)
    theirs_dir.mkdir(parents=True)
    KinematicTrajectory(qpos=np.ones((4, 35)), qvel=np.zeros((4, 34)), frequency=50.0).save(
        theirs_dir / "trajectory_kinematic.npz")
    stats_path = Path(cfg.isolation.output_root) / "prepare_stats.json"
    stats_path.write_text(json.dumps({"ours": {"thumb_median_mm": 0.077}, "other": {"kept": True}}))

    monkeypatch.setattr(rw, "_load_scene_and_mesh", lambda cfg: (None, None, None, None))
    monkeypatch.setattr(rw, "baseline_world_inputs", lambda *a: (np.zeros((3, 4, 4)), None, None, 50.0))
    monkeypatch.setattr(rw, "mano_finger_labels", lambda *a: None)
    monkeypatch.setattr(rw, "retarget_kinematic", lambda *a: KinematicTrajectory(
        qpos=np.full((3, 35), 2.0), qvel=np.zeros((3, 34)), frequency=50.0))
    monkeypatch.setattr(rw, "kinematic_reference_stats", lambda qpos, *a: KinematicStats(
        int(qpos.shape[0]), 1, 1.0, 0.0, 0.0, 0.0, 0.0, float(qpos[0, 0]), 0.0, 0.0))
    monkeypatch.setattr(rw, "robot_contact_frames", lambda qpos, *a, **k: np.ones(qpos.shape[0], dtype=bool))
    monkeypatch.setattr(rw, "primitive_clearance_stats",
                        lambda qpos, *a, **k: SimpleNamespace(as_dict=lambda: {"n_frames": int(qpos.shape[0])}))
    rw._stage_prepare(cfg)

    out = Path(cfg.isolation.output_root)
    with np.load(out / "ours_clearance_trajectory_kinematic.npz") as ours:
        assert ours["qpos"].shape == (3, 35)
    assert json.loads((out / "ours_clearance_retarget_config.json").read_text())["contact_margin"] == 0.003
    with np.load(theirs_dir / "trajectory_kinematic.npz") as theirs:       # recorded arm untouched
        assert float(theirs["qpos"][0, 0]) == 1.0
    stats = json.loads(stats_path.read_text())
    assert stats["other"] == {"kept": True}
    assert stats["ours"] == {"thumb_median_mm": 0.077}                     # no reference on disk: kept
    assert stats["theirs"]["thumb_median_mm"] == 1.0 and stats["theirs"]["primitive"] == {"n_frames": 4}
    assert stats["ours_clearance"]["thumb_median_mm"] == 2.0
    assert stats["ours_clearance"]["primitive"] == {"n_frames": 3}


def test_stage_report_writes_a_failed_row_for_an_arm_that_ended_during_warmup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An arm that terminates inside the welded warmup has an empty metric window. That is exactly
    the run whose report matters, so the stage must write its row (n_steps=0, NaN means, success
    False, no physics-penetration stats computed on zero frames) instead of aborting."""
    cfg = _make_cfg(tmp_path, mesh_scale=1.0)
    _write_arm_artifacts(cfg, "theirs", np.full(40, 0.05), [2] * 6, legacy_meta=False)
    _write_arm_artifacts(cfg, "ours", np.zeros(0), [2, 4], legacy_meta=False)
    monkeypatch.setattr(rw, "_load_scene_and_mesh", lambda cfg: (None, None, None, None))
    seen_frames: list[int] = []

    def _fake_stats(qpos: np.ndarray, *args: object) -> KinematicStats:
        seen_frames.append(int(qpos.shape[0]))
        return KinematicStats(int(qpos.shape[0]), 0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    monkeypatch.setattr(rw, "kinematic_reference_stats", _fake_stats)
    rw._stage_report(cfg)

    rows = {r["arm"]: r for r in json.loads((Path(cfg.isolation.output_root) / "report.json").read_text())}
    ours = rows["ours"]
    assert seen_frames == [40]
    assert ours["tracking"]["n_steps"] == 0 and ours["physics_stats_frames"] == 0
    assert ours["success"] is False
    assert np.isnan(ours["obj_pos_err_mean"]) and np.isnan(ours["obj_quat_err_max"])
    assert np.isnan(ours["physics_any_penetration_frac"])
    assert ours["time_base_steps"]["metric_steps"] == 0
    assert rows["theirs"]["success"] is True


def _fake_stats(qpos: np.ndarray, *args: object) -> KinematicStats:
    return KinematicStats(int(qpos.shape[0]), 0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def test_stage_report_fails_an_arm_that_terminated_after_warmup_with_a_low_mean_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The addendum's success needs the whole clip with no early termination. "ours" stops at executed
    step 40 of max_sim_steps 60 after 20 post-warmup steps at 0.02 m: its window means pass, the arm must
    not. "theirs" executes the whole clip and succeeds."""
    cfg = _make_cfg(tmp_path, mesh_scale=1.0)
    _write_arm_artifacts(cfg, "theirs", np.full(40, 0.05), [2] * 6, legacy_meta=False)
    _write_arm_artifacts(cfg, "ours", np.full(20, 0.02), [2] * 4, legacy_meta=False)
    monkeypatch.setattr(rw, "_load_scene_and_mesh", lambda cfg: (None, None, None, None))
    monkeypatch.setattr(rw, "kinematic_reference_stats", _fake_stats)
    rw._stage_report(cfg)

    rows = {r["arm"]: r for r in json.loads((Path(cfg.isolation.output_root) / "report.json").read_text())}
    ours, theirs = rows["ours"], rows["theirs"]
    assert ours["obj_pos_err_mean"] == pytest.approx(0.02) and ours["window_means_success"] is True
    assert ours["completed_clip"] is False and ours["status"] == "terminated_early"
    assert ours["success"] is False
    assert theirs["completed_clip"] is True and theirs["status"] == "completed" and theirs["success"] is True


def test_stage_report_writes_a_no_output_row_for_an_arm_the_guard_killed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A killed physics run leaves only the scene, the reference and the baseline's config.yaml. The report
    must still write every arm's row: the killed arm's as a failed no-output row with the same keys, and
    the MPC split computed from the arms that ran."""
    cfg = _make_cfg(tmp_path, mesh_scale=1.0, arms=["theirs", "ours", "ours_clearance"])
    _write_arm_artifacts(cfg, "theirs", np.full(40, 0.05), [2] * 6, legacy_meta=False)
    _write_arm_artifacts(cfg, "ours", np.r_[np.full(10, 0.02), np.full(10, 0.3)], [2, 4, 4, 4], legacy_meta=False)
    killed = processed_data_dir(rw._arm_output_root(cfg, "ours_clearance"), "sharpa", "right", "whisking_c4", 0)
    killed.mkdir(parents=True)
    (killed / "config.yaml").write_text("max_sim_steps: 0\n")
    monkeypatch.setattr(rw, "_load_scene_and_mesh", lambda cfg: (None, None, None, None))
    monkeypatch.setattr(rw, "kinematic_reference_stats", _fake_stats)
    rw._stage_report(cfg)

    rows = json.loads((Path(cfg.isolation.output_root) / "report.json").read_text())
    assert [r["arm"] for r in rows] == ["theirs", "ours", "ours_clearance"]
    clearance = rows[2]
    assert clearance["status"] == "no_output" and clearance["success"] is False
    assert clearance["missing_artifacts"] == ["arm_meta.json", "isolation_errors.npz", "trajectory_mjwp.npz"]
    assert np.isnan(clearance["obj_pos_err_mean"])
    assert set(clearance) == set(rows[0]) == set(rows[1])
    assert rows[0]["missing_artifacts"] == [] and rows[1]["mpc_iterations"]["split_chunk"] == 3


def test_load_arm_raises_type_error_on_a_non_mapping_legacy_config(tmp_path: Path) -> None:
    """The legacy `config.yaml` fallback must fail with an explicit TypeError (not an assert, which
    `python -O` strips) when the baseline's saved config is not a mapping."""
    cfg = _make_cfg(tmp_path)
    _write_arm_artifacts(cfg, "ours", np.full(20, 0.02), [2] * 4, legacy_meta=True)
    data_dir = processed_data_dir(rw._arm_output_root(cfg, "ours"), "sharpa", "right", "whisking_c4", 0)
    (data_dir / "config.yaml").write_text("- not\n- a mapping\n")
    with pytest.raises(TypeError, match="config.yaml"):
        rw._load_arm(cfg, "ours")


@pytest.mark.parametrize(("stage", "expects_budget"), [("dry_run", False), ("run_arm", False), ("report", True)])
def test_main_skips_the_cuda_context_for_stages_that_only_launch_the_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str, expects_budget: bool,
) -> None:
    """`dry_run`/`run_arm` only launch the baseline subprocess; `apply_gpu_budget` would still open a
    ~0.5 GB CUDA context in the orchestrator for the whole physics run, which counts against the
    shared-host GPU guard. Stages that run our own torch code keep the budget."""
    cfg = _make_cfg(tmp_path, stage=stage)
    calls: list[str] = []
    monkeypatch.setattr(rw, "apply_cpu_budget", lambda: calls.append("cpu"))
    monkeypatch.setattr(rw, "apply_gpu_budget", lambda: calls.append("gpu"))
    for name in ("_stage_dry_run", "_stage_run_arm", "_stage_report"):
        monkeypatch.setattr(rw, name, lambda cfg, _name=name: calls.append(_name))
    rw.main.__wrapped__(cfg)
    assert ("gpu" in calls) is expects_budget and "cpu" in calls and calls[-1] == f"_stage_{stage}"



def _prepare_setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, our_arm: str = "ours_clearance_10fps_inputs",
                   **overrides: object) -> DictConfig:
    """`test_stage_prepare_writes_only_our_arm_and_merges_the_stats`'s setup for a 10-frame clip and the C3c arm."""
    demo = tmp_path / "demo"
    demo.mkdir()
    (demo / "scene.xml").write_text("<mujoco/>")
    (demo / "assets").mkdir()
    shipped = demo / "trajectory_kinematic.npz"
    KinematicTrajectory(qpos=np.zeros((10, 35)), qvel=np.zeros((10, 34)), frequency=50.0).save(shipped)
    cfg = _make_cfg(tmp_path, our_arm=our_arm, arms=["theirs", our_arm],
                    mesh_scale=1.0, demo_scene_xml=str(demo / "scene.xml"), demo_assets_dir=str(demo / "assets"),
                    demo_trajectory_kinematic_npz=str(shipped), demo_keypoints_npz=str(shipped),
                    mano_right_pkl="unused", **overrides)
    theirs_dir = processed_data_dir(rw._arm_output_root(cfg, "theirs"), "sharpa", "right", "whisking_c4", 0)
    theirs_dir.mkdir(parents=True)
    KinematicTrajectory(qpos=np.ones((10, 35)), qvel=np.zeros((10, 34)), frequency=50.0).save(
        theirs_dir / "trajectory_kinematic.npz")
    monkeypatch.setattr(rw, "_load_scene_and_mesh", lambda cfg: (None, None, None, None))
    monkeypatch.setattr(rw, "mano_finger_labels", lambda *a: None)
    monkeypatch.setattr(rw, "kinematic_reference_stats", lambda qpos, *a: KinematicStats(
        int(qpos.shape[0]), 1, 1.0, 0.0, 0.0, 0.0, 0.0, float(qpos[0, 0]), 0.0, 0.0))
    monkeypatch.setattr(rw, "robot_contact_frames", lambda qpos, *a, **k: np.ones(qpos.shape[0], dtype=bool))
    monkeypatch.setattr(rw, "primitive_clearance_stats",
                        lambda qpos, *a, **k: SimpleNamespace(as_dict=lambda: {"n_frames": int(qpos.shape[0])}))
    return cfg


def test_stage_prepare_engine_rate_upsamples_the_inputs_and_solves_at_full_rate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`reference_rate: engine` samples the inputs at `engine_stride`, upsamples them (object and wrist
    SE(3) by spline, hand vertices lerped in the wrist frame) and solves the reference at full rate. The raw rows
    between keys are off the key line, so only upsampled inputs reproduce it."""
    import torch

    cfg = _prepare_setup(tmp_path, monkeypatch, reference_rate="engine", engine_stride=3)
    f = torch.arange(10, dtype=torch.float64)
    off_key = (torch.arange(10) % 3 != 0).to(torch.float64)
    obj = torch.eye(4, dtype=torch.float64).repeat(10, 1, 1)
    obj[:, 0, 3] = 0.01 * f + 0.5 * off_key
    wrist = torch.eye(4, dtype=torch.float64).repeat(10, 1, 1)
    wrist[:, 1, 3] = 0.02 * f + 0.3 * off_key
    verts = torch.zeros(10, 778, 3, dtype=torch.float64)
    verts[:, :, 2] = (0.1 * f + 0.7 * off_key)[:, None]
    monkeypatch.setattr(rw, "baseline_world_inputs", lambda *a: (obj, wrist, verts, 50.0))
    calls: list[tuple[int, float, np.ndarray, np.ndarray, np.ndarray]] = []

    def _fake_retarget(*a: object) -> KinematicTrajectory:
        obj_in, wrist_in, verts_in, freq = (np.asarray(a[0]), np.asarray(a[1]), np.asarray(a[3]), float(a[9]))  # type: ignore[arg-type]
        calls.append((obj_in.shape[0], freq, obj_in, wrist_in, verts_in))
        qpos = np.zeros((obj_in.shape[0], 35))
        qpos[:, :28] = 0.1 * np.arange(obj_in.shape[0])[:, None]
        qpos[:, 31] = 1.0
        return KinematicTrajectory(qpos=qpos, qvel=np.zeros((obj_in.shape[0], 34)), frequency=freq)

    monkeypatch.setattr(rw, "retarget_kinematic", _fake_retarget)
    rw._stage_prepare(cfg)

    assert [(c[0], c[1]) for c in calls] == [(10, 50.0)]
    _, _, obj_in, wrist_in, verts_in = calls[0]
    frames = np.arange(10, dtype=np.float64)
    assert np.abs(obj_in[:, 0, 3] - 0.01 * frames).max() <= 1e-12
    assert np.abs(obj_in[:, :3, :3] - np.eye(3)).max() <= 1e-12
    assert np.abs(wrist_in[:, 1, 3] - 0.02 * frames).max() <= 1e-12
    assert np.abs(verts_in[:, :, 2] - 0.1 * frames[:, None]).max() <= 1e-12
    assert np.abs(verts_in[:, :, :2]).max() <= 1e-12
    out = Path(cfg.isolation.output_root)
    with np.load(out / "ours_clearance_10fps_inputs_trajectory_kinematic.npz") as ours:
        assert ours["qpos"].shape == (10, 35) and ours["qvel"].shape == (10, 34) and float(ours["frequency"]) == 50.0
        assert np.abs(ours["qpos"][[0, 3, 6, 9], 0] - [0.0, 0.3, 0.6, 0.9]).max() <= 1e-12
    meta = json.loads((out / "ours_clearance_10fps_inputs_retarget_config.json").read_text())
    assert meta["reference_rate"] == "engine" and meta["engine_stride"] == 3 and meta["key_frames"] == [0, 3, 6, 9]
    assert isinstance(meta["solve_seconds"], float) and isinstance(meta["upsample_seconds"], float)
    assert meta["solve_seconds"] >= 0.0 and meta["upsample_seconds"] >= 0.0
    assert abs(meta["solve_seconds"] + meta["upsample_seconds"] - meta["wall_seconds"]) <= 1e-9
    stats = json.loads((out / "prepare_stats.json").read_text())
    assert stats["ours_clearance_10fps_inputs"]["primitive"] == {"n_frames": 10}


def test_stage_prepare_full_rate_records_no_upsampling(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _prepare_setup(tmp_path, monkeypatch, our_arm="ours_clearance_full", reference_rate="full")
    monkeypatch.setattr(rw, "baseline_world_inputs", lambda *a: (np.zeros((10, 4, 4)), None, None, 50.0))
    seen: list[int] = []
    monkeypatch.setattr(rw, "retarget_kinematic", lambda *a: seen.append(a[0].shape[0]) or KinematicTrajectory(
        qpos=np.zeros((10, 35)), qvel=np.zeros((10, 34)), frequency=50.0))
    rw._stage_prepare(cfg)
    meta = json.loads((Path(cfg.isolation.output_root) / "ours_clearance_full_retarget_config.json").read_text())
    assert seen == [10]
    assert meta["reference_rate"] == "full" and meta["engine_stride"] is None and meta["key_frames"] is None
    assert meta["upsample_seconds"] == 0.0 and meta["solve_seconds"] == meta["wall_seconds"]


def test_stage_prepare_rejects_an_unknown_reference_rate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _prepare_setup(tmp_path, monkeypatch, reference_rate="30hz")
    monkeypatch.setattr(rw, "_load_scene_and_mesh", lambda cfg: pytest.fail("must reject before loading the scene"))
    monkeypatch.setattr(rw, "retarget_kinematic", lambda *a: pytest.fail("must reject before solving"))
    with pytest.raises(ValueError, match="reference_rate"):
        rw._stage_prepare(cfg)


def test_config_lists_the_c3c_rate_composition_arms() -> None:
    from omegaconf import OmegaConf

    conf = OmegaConf.load(Path(__file__).resolve().parents[2] / "run" / "conf" / "config.yaml")
    assert list(conf.isolation.arms) == ["theirs", "ours", "ours_clearance", "ours_clearance_full",
                                         "ours_clearance_10fps_inputs"]
    assert conf.isolation.our_arm == "ours_clearance"
    assert conf.isolation.reference_rate == "full" and conf.isolation.engine_stride == 3
