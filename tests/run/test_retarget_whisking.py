"""Tests for `run/pipeline/retarget_whisking.py`: the baseline-npz format check for
`fastdo.retarget.isolation.setup_arm_directory`, and argument plumbing for the `_stage_run_arm`
orchestration -- with a fake baseline command standing in for `_baseline_physics_driver.py` +
the `retargeting` conda env, so that these tests never run the real baseline.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from omegaconf import DictConfig, OmegaConf

from fastdo.retarget.isolation import KinematicStats, processed_data_dir, setup_arm_directory
from fastdo.retarget.kinematic import KinematicTrajectory

_MODULE_PATH = Path(__file__).resolve().parents[2] / "run" / "pipeline" / "retarget_whisking.py"


def _load_module() -> ModuleType:
    """`run/pipeline/` is not a package (no `__init__.py`); load the script by path instead."""
    spec = importlib.util.spec_from_file_location("retarget_whisking", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rw = _load_module()


def _make_cfg(tmp_path: Path, **isolation_overrides: object) -> DictConfig:
    baseline_root = tmp_path / "baseline"
    baseline_root.mkdir(parents=True, exist_ok=True)
    isolation = {
        "baseline_root": str(baseline_root),
        "output_root": str(tmp_path / "outputs" / "isolation"),
        "temp_dir": str(tmp_path / "temp"),
        "task_name": "whisking_c4",
        "robot_type": "sharpa",
        "embodiment_type": "right",
        "data_id": 0,
        "seed": 0,
        "n_per_link_pen": 24,
        "conda_bin": "/nonexistent/conda",
        "conda_env": "retargeting",
        "num_samples": None,
        "num_perturb_samples": None,
        "max_sim_steps": 0,
        "gpu_mem_cap_mib": 11000.0,
        "dry_run_seconds": 1,
        "dry_run_poll_seconds": 1,
        "stage": "run_arm",
        "arm": "theirs",
        "our_arm": "ours",
        "arms": ["theirs", "ours"],
        "match_recorded_arm": None,
    }
    isolation.update(isolation_overrides)
    return OmegaConf.create({"isolation": isolation})


# --- npz format / arm-directory layout -------------------------------------------------------


def test_processed_data_dir_matches_baseline_layout_convention(tmp_path: Path) -> None:
    """Mirrors `retargeting.utils.io.get_processed_data_dir`'s exact 4-segment layout, which
    `optimize_physics.py`'s own `process_config` relies on to find `scene.xml`/the reference npz."""
    d = processed_data_dir(tmp_path, "sharpa", "right", "whisking_c4", 0)
    assert d == tmp_path / "sharpa" / "right" / "whisking_c4" / "0"


def test_setup_arm_directory_copies_scene_and_npz_and_symlinks_assets(tmp_path: Path) -> None:
    demo_scene_xml = tmp_path / "demo" / "scene.xml"
    demo_scene_xml.parent.mkdir(parents=True)
    demo_scene_xml.write_text("<mujoco/>")

    demo_assets_dir = tmp_path / "demo" / "assets"
    demo_assets_dir.mkdir()
    (demo_assets_dir / "marker.txt").write_text("asset")

    traj = KinematicTrajectory(
        qpos=np.random.default_rng(0).normal(size=(5, 35)),
        qvel=np.random.default_rng(1).normal(size=(5, 34)),
        frequency=50.0,
    )
    src_npz = tmp_path / "demo" / "trajectory_kinematic.npz"
    traj.save(src_npz)

    output_root = tmp_path / "outputs" / "isolation" / "ours"
    layout = setup_arm_directory(
        output_root=output_root, demo_scene_xml=demo_scene_xml, demo_assets_dir=demo_assets_dir,
        trajectory_kinematic_src=src_npz, task="whisking_c4", robot_type="sharpa",
        embodiment_type="right", data_id=0,
    )

    assert layout.data_dir == output_root / "sharpa" / "right" / "whisking_c4" / "0"
    assert (layout.data_dir / "scene.xml").read_text() == "<mujoco/>"

    # The baseline's own trajectory_kinematic.npz format: qpos (T,35) float64, qvel (T,34)
    # float64, scalar frequency -- copied verbatim, byte-for-byte round-trippable.
    with np.load(layout.data_dir / "trajectory_kinematic.npz") as data:
        assert set(data.keys()) == {"qpos", "qvel", "frequency"}
        assert data["qpos"].shape == (5, 35)
        assert data["qvel"].shape == (5, 34)
        assert data["qpos"].dtype == np.float64
        assert data["qvel"].dtype == np.float64
        assert float(data["frequency"]) == 50.0
        np.testing.assert_allclose(data["qpos"], traj.qpos)

    assert layout.assets_link.is_symlink()
    assert layout.assets_link.resolve() == demo_assets_dir.resolve()
    assert (layout.assets_link / "marker.txt").read_text() == "asset"

    task_info = json.loads(layout.task_info_json.read_text())
    assert task_info == {
        "task": "whisking_c4", "dataset_name": "do_as_i_do", "robot_type": "sharpa",
        "embodiment_type": "right", "data_id": 0,
    }
    assert layout.task_info_json.parent == layout.data_dir.parent


def test_setup_arm_directory_is_idempotent_on_the_assets_symlink(tmp_path: Path) -> None:
    """A second call (e.g. re-running `prepare` for both arms against a shared `assets/` symlink
    target under the same output_root) must not raise `FileExistsError`."""
    demo_scene_xml = tmp_path / "demo" / "scene.xml"
    demo_scene_xml.parent.mkdir(parents=True)
    demo_scene_xml.write_text("<mujoco/>")
    demo_assets_dir = tmp_path / "demo" / "assets"
    demo_assets_dir.mkdir()
    src_npz = tmp_path / "demo" / "trajectory_kinematic.npz"
    KinematicTrajectory(qpos=np.zeros((1, 35)), qvel=np.zeros((1, 34)), frequency=50.0).save(src_npz)

    output_root = tmp_path / "outputs" / "isolation" / "theirs"
    for _ in range(2):
        setup_arm_directory(output_root=output_root, demo_scene_xml=demo_scene_xml,
                            demo_assets_dir=demo_assets_dir, trajectory_kinematic_src=src_npz,
                            task="whisking_c4", robot_type="sharpa", embodiment_type="right", data_id=0)


# --- argument plumbing -------------------------------------------------------------------------


def test_driver_args_omits_none_overrides_and_points_at_the_right_arm_dir(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)  # num_samples/num_perturb_samples both None
    args = rw._driver_args(cfg, arm="theirs", max_sim_steps=0)

    assert "--num-samples" not in args
    assert "--num-perturb-samples" not in args
    expected_root = str(rw._arm_output_root(cfg, "theirs"))
    assert args[args.index("--output-root-dir") + 1] == expected_root
    assert args[args.index("--task") + 1] == "whisking_c4"
    assert args[args.index("--data-id") + 1] == "0"
    expected_data_dir = (rw._arm_output_root(cfg, "theirs") / "sharpa" / "right" / "whisking_c4" / "0")
    assert args[args.index("--errors-out") + 1] == str(expected_data_dir / "isolation_errors.npz")
    assert args[args.index("--meta-out") + 1] == str(expected_data_dir / "arm_meta.json")


def test_driver_args_includes_env_count_overrides_when_set(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path, num_samples=256, num_perturb_samples=2)
    args = rw._driver_args(cfg, arm="ours", max_sim_steps=100)

    assert args[args.index("--num-samples") + 1] == "256"
    assert args[args.index("--num-perturb-samples") + 1] == "2"
    assert args[args.index("--max-sim-steps") + 1] == "100"
    assert args[args.index("--output-root-dir") + 1] == str(rw._arm_output_root(cfg, "ours"))


def test_conda_run_cmd_wraps_with_nice_and_the_configured_conda_env(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path, conda_bin="/opt/conda/bin/conda", conda_env="retargeting")
    cmd = rw._conda_run_cmd(cfg, "--foo", "bar")

    assert cmd[:3] == ["nice", "-n", "15"]
    assert cmd[3] == "/opt/conda/bin/conda"
    assert cmd[4:8] == ["run", "-n", "retargeting", "--no-capture-output"]
    assert cmd[8] == "python"
    assert cmd[-2:] == ["--foo", "bar"]


# --- run_arm orchestration end-to-end, with a fake baseline command ----------------------------

_FAKE_DRIVER = '''
import argparse, json, sys
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--baseline-root", required=True)
parser.add_argument("--output-root-dir", required=True)
parser.add_argument("--task", required=True)
parser.add_argument("--data-id", type=int, default=0)
parser.add_argument("--robot-type", default="sharpa")
parser.add_argument("--embodiment-type", default="right")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--num-samples", type=int, default=None)
parser.add_argument("--num-perturb-samples", type=int, default=None)
parser.add_argument("--max-sim-steps", type=int, default=0)
parser.add_argument("--errors-out", required=True)
parser.add_argument("--meta-out", required=True)
parser.add_argument("--fail", action="store_true")
args = parser.parse_args()

if args.fail:
    sys.exit(1)

errors_out = Path(args.errors_out)
errors_out.parent.mkdir(parents=True, exist_ok=True)
np.savez(errors_out, obj_pos_err=0.01, obj_quat_err=0.02,
         pos_err_frames=np.array([0.01, 0.02]), quat_err_frames=np.array([0.02, 0.03]))

meta_out = Path(args.meta_out)
meta_out.parent.mkdir(parents=True, exist_ok=True)
# Mirrors the real driver's own meta["output_dir"]: retargeting.config.process_config resolves
# output_root_dir/{robot_type}/{embodiment_type}/{task}/{data_id}, not the raw --output-root-dir.
resolved_output_dir = str(Path(args.output_root_dir) / args.robot_type / args.embodiment_type
                          / args.task / str(args.data_id))
meta_out.write_text(json.dumps({
    "task": args.task, "wall_seconds": 0.01, "num_samples": args.num_samples,
    "num_perturb_samples": args.num_perturb_samples, "warmup_steps": 0,
    "output_dir": resolved_output_dir,
}))
print("=== fake driver received: task={} output_root_dir={} ===".format(args.task, args.output_root_dir))
'''


@pytest.fixture
def fake_driver(tmp_path: Path) -> Path:
    path = tmp_path / "fake_driver.py"
    path.write_text(_FAKE_DRIVER)
    return path


def test_stage_run_arm_plumbs_args_through_to_a_fake_baseline_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_driver: Path,
) -> None:
    """No real baseline execution: `_conda_run_cmd` is monkeypatched to invoke `fake_driver.py`
    directly with `sys.executable` (bypassing `nice`/`conda run`/the retargeting env entirely),
    so this exercises exactly the argument-plumbing contract `_baseline_physics_driver.py` must
    honor, without depending on the `retargeting` conda env being present."""
    cfg = _make_cfg(tmp_path)

    def _fake_conda_run_cmd(cfg: DictConfig, *args: str) -> list[str]:
        return [sys.executable, str(fake_driver), *args]

    monkeypatch.setattr(rw, "_conda_run_cmd", _fake_conda_run_cmd)
    rw._stage_run_arm(cfg)

    data_dir = rw._arm_output_root(cfg, "theirs") / "sharpa" / "right" / "whisking_c4" / "0"
    meta = json.loads((data_dir / "arm_meta.json").read_text())
    assert meta["task"] == "whisking_c4"
    assert meta["output_dir"] == str(data_dir)
    with np.load(data_dir / "isolation_errors.npz") as errs:
        assert float(errs["obj_pos_err"]) == pytest.approx(0.01)
        assert float(errs["obj_quat_err"]) == pytest.approx(0.02)

    log_path = Path(cfg.isolation.temp_dir) / "arm_theirs.log"
    assert "fake driver received" in log_path.read_text()


def test_stage_run_arm_raises_on_nonzero_exit_from_the_baseline_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_driver: Path,
) -> None:
    cfg = _make_cfg(tmp_path)

    def _fake_conda_run_cmd(cfg: DictConfig, *args: str) -> list[str]:
        return [sys.executable, str(fake_driver), *args, "--fail"]

    monkeypatch.setattr(rw, "_conda_run_cmd", _fake_conda_run_cmd)
    with pytest.raises(RuntimeError, match="theirs"):
        rw._stage_run_arm(cfg)


def test_stage_run_arm_rejects_an_unknown_arm(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path, arm="both")
    with pytest.raises(ValueError, match="isolation.arm"):
        rw._stage_run_arm(cfg)


# --- report stage: one post-warmup time base, logger not print ---------------------------------


def _write_arm_artifacts(cfg: DictConfig, arm: str, post_pos_err: np.ndarray, opt_steps: list[int],
                         legacy_meta: bool) -> None:
    """Synthetic stand-ins for one arm's driver/baseline outputs: 20-step warmup, 10-step chunks, an
    8-frame reference resampled 5x (a 40-step clip, so max_sim_steps = 60)."""
    warmup, chunk, max_it = 20, 10, 4
    data_dir = processed_data_dir(rw._arm_output_root(cfg, arm), "sharpa", "right", "whisking_c4", 0)
    data_dir.mkdir(parents=True)
    pos = np.concatenate([np.zeros(warmup), post_pos_err])
    np.savez(data_dir / "isolation_errors.npz", obj_pos_err=pos.mean(), obj_quat_err=(pos / 2).mean(),
             pos_err_frames=pos, quat_err_frames=pos / 2)
    n_chunks = pos.size // chunk
    np.savez(data_dir / "trajectory_mjwp.npz", qpos=np.zeros((n_chunks, chunk, 35)),
             opt_steps=np.array(opt_steps).reshape(-1, 1),
             improvement=np.full((n_chunks, max_it), 0.5, dtype=np.float32),
             rew_max=np.zeros((n_chunks, max_it), dtype=np.float32))
    settings = {"max_sim_steps": warmup + 40, "ctrl_steps": chunk, "max_num_iterations": max_it,
                "improvement_threshold": 0.01, "improvement_check_steps": 2, "ref_dt": 0.025,
                "sim_dt": 0.005}
    meta: dict[str, object] = {"task": "whisking_c4", "wall_seconds": 12.5, "warmup_steps": warmup}
    if legacy_meta:  # runs predating these meta fields: the baseline's own saved config.yaml
        OmegaConf.save(OmegaConf.create(settings), data_dir / "config.yaml")
    else:
        meta.update(settings)
    (data_dir / "arm_meta.json").write_text(json.dumps(meta))
    KinematicTrajectory(qpos=np.zeros((8, 35)), qvel=np.zeros((8, 34)), frequency=50.0).save(
        data_dir / "trajectory_kinematic.npz")


def test_stage_report_puts_errors_and_penetration_on_the_post_warmup_time_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """"ours" terminates early: its whole-trace mean (0.08 m, warmup included) would pass the 0.1 m
    criterion; on the post-warmup base (0.16 m) it must fail. "theirs" executes a whole extra chunk
    past max_sim_steps on the held last frame (0.5 m there): the window must stop at max_sim_steps.
    Penetration stats must see exactly the frames the tracking errors cover, and the table must go
    through the logger, not stdout."""
    cfg = _make_cfg(tmp_path, mesh_scale=1.0)
    _write_arm_artifacts(cfg, "theirs", np.r_[np.full(40, 0.05), np.full(10, 0.5)], [2] * 7,
                         legacy_meta=False)
    _write_arm_artifacts(cfg, "ours", np.r_[np.full(10, 0.02), np.full(10, 0.3)], [2, 4, 4, 4],
                         legacy_meta=True)
    monkeypatch.setattr(rw, "_load_scene_and_mesh", lambda cfg: (None, None, None, None))
    seen_frames: list[int] = []

    def _fake_stats(qpos: np.ndarray, *args: object) -> KinematicStats:
        seen_frames.append(int(qpos.shape[0]))
        return KinematicStats(int(qpos.shape[0]), 0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    monkeypatch.setattr(rw, "kinematic_reference_stats", _fake_stats)
    caplog.set_level(logging.INFO)
    rw._stage_report(cfg)

    rows = {r["arm"]: r for r in json.loads((Path(cfg.isolation.output_root) / "report.json").read_text())}
    ours, theirs = rows["ours"], rows["theirs"]
    assert ours["time_base"] == theirs["time_base"] == "post_warmup_to_max_sim_steps"
    assert ours["obj_pos_err_mean"] == pytest.approx(0.16)
    assert ours["baseline_log_obj_pos_err"] == pytest.approx(0.08)
    assert ours["success"] is False
    assert ours["baseline_log_success"] is True  # the whole-trace verdict, kept as its own flag
    assert theirs["obj_pos_err_mean"] == pytest.approx(0.05) and theirs["obj_pos_err_max"] == pytest.approx(0.05)
    assert theirs["success"] is True
    assert seen_frames == [40, 20]
    assert ours["physics_stats_frames"] == ours["tracking"]["n_steps"] == 20
    assert theirs["physics_stats_frames"] == theirs["tracking"]["n_steps"] == 40
    assert theirs["tracking"]["end_step"] == 60

    assert ours["time_base_steps"] == {
        "max_sim_steps": 60, "warmup_steps": 20, "raw_reference_frames": 8, "executed_steps": 40,
        "chunk_steps": 10, "clip_steps": 40, "run_steps": 40, "capped": False, "metric_end_step": 40,
        "metric_steps": 20, "overshoot_steps": 0, "post_warmup_executed_steps": 20,
        "clip_fraction_executed": 0.5, "completed_clip": False,
    }
    assert ours["status"] == "terminated_early" and theirs["status"] == "completed"
    assert theirs["completed_clip"] is True and theirs["time_base_steps"]["completed_clip"] is True
    assert theirs["time_base_steps"]["overshoot_steps"] == 10
    assert theirs["time_base_steps"]["clip_fraction_executed"] == 1.0
    # Common split at the earliest post-warmup crossing of 0.1 m (ours, step 30 -> chunk 3).
    assert ours["mpc_iterations"]["split_chunk"] == theirs["mpc_iterations"]["split_chunk"] == 3
    assert ours["mpc_iterations"]["mean_before"] == pytest.approx(10 / 3)
    assert ours["mpc_iterations"]["mean_after"] == pytest.approx(4.0)
    assert ours["mpc_iterations"]["n_budget_exhausted"] == 3

    assert "=== Physics isolation study: whisking demo ===" in caplog.text
    assert "overshoot=10" in caplog.text
    assert "Physics isolation study" not in capsys.readouterr().out
