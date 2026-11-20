"""Tests for `run/pipeline/_baseline_physics_driver.py`'s run metadata.

The driver runs inside the `retargeting` conda env and is never executed against the real baseline
here; only its pure `_run_meta` helper is exercised, with a stand-in for the baseline's resolved
`Config` (after `process_config` and `optimize_physics.main` have mutated it in place).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np

_DRIVER_PATH = Path(__file__).resolve().parents[2] / "run" / "pipeline" / "_baseline_physics_driver.py"


def _load_driver() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_baseline_physics_driver", _DRIVER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


drv = _load_driver()


def _resolved_config(tmp_path: Path, raw_frames: int) -> SimpleNamespace:
    ref = tmp_path / "trajectory_kinematic.npz"
    np.savez(ref, qpos=np.zeros((raw_frames, 35)), qvel=np.zeros((raw_frames, 34)),
             frequency=np.float64(50.0))
    return SimpleNamespace(
        warmup_steps=np.int64(600), max_sim_steps=1566, ctrl_steps=100, max_num_iterations=32,
        improvement_threshold=np.float64(0.01), improvement_check_steps=2, data_path=str(ref),
        output_dir=str(tmp_path), ref_dt=0.0333, sim_dt=np.float64(0.005),
    )


def test_run_meta_records_each_arms_own_step_accounting(tmp_path: Path) -> None:
    """Everything the report needs to put both arms side by side (resolved `max_sim_steps`, warmup,
    raw reference frame count, executed steps, MPC iteration budget) lands in arm_meta.json."""
    config = _resolved_config(tmp_path, raw_frames=138)
    errors = {"obj_pos_err": 0.1, "obj_quat_err": 0.2, "pos_err_frames": np.zeros(1100),
              "quat_err_frames": np.zeros(1100, dtype=np.float32)}

    meta = drv._run_meta(task="whisking_c4", wall_s=2763.0, overrides={"num_perturb_samples": 2},
                         config=config, errors=errors)

    assert meta == {
        "task": "whisking_c4", "wall_seconds": 2763.0, "num_samples": "default(1024)",
        "num_perturb_samples": 2, "has_errors": True, "output_dir": str(tmp_path),
        "warmup_steps": 600, "max_sim_steps": 1566, "ctrl_steps": 100, "max_num_iterations": 32,
        "improvement_threshold": 0.01, "improvement_check_steps": 2, "raw_reference_frames": 138,
        "executed_steps": 1100, "ref_dt": 0.0333, "sim_dt": 0.005,
    }
    assert json.loads(json.dumps(meta)) == meta


def test_run_meta_without_errors_reports_zero_executed_steps(tmp_path: Path) -> None:
    config = _resolved_config(tmp_path, raw_frames=135)
    meta = drv._run_meta(task="t", wall_s=1.0, overrides={}, config=config, errors=None)
    assert meta["has_errors"] is False
    assert meta["executed_steps"] == 0
    assert meta["raw_reference_frames"] == 135
    assert meta["num_perturb_samples"] == "default(4)"


# ---- generic `--override key=value` -------------------------------------------------------

import pytest


def test_parse_overrides_types_values_with_yaml_scalars() -> None:
    out = drv.parse_overrides(["horizon=1.6", "max_num_iterations=12", "use_torch_compile=true",
                               "pos_noise_scale=0", "viewer=viser", "pair_margin_range=[-0.01, 0.01]"])
    assert out == {"horizon": 1.6, "max_num_iterations": 12, "use_torch_compile": True,
                   "pos_noise_scale": 0, "viewer": "viser", "pair_margin_range": [-0.01, 0.01]}
    assert isinstance(out["horizon"], float) and isinstance(out["max_num_iterations"], int)


def test_parse_overrides_empty_and_none() -> None:
    assert drv.parse_overrides(None) == {}
    assert drv.parse_overrides([]) == {}


def test_parse_overrides_value_may_contain_equals() -> None:
    assert drv.parse_overrides(["viewer=a=b"]) == {"viewer": "a=b"}


@pytest.mark.parametrize("bad", ["horizon", "=1.0", " =1", "horizon="])
def test_parse_overrides_refuses_malformed(bad: str) -> None:
    with pytest.raises(ValueError):
        drv.parse_overrides([bad])


def test_parse_overrides_refuses_duplicates() -> None:
    with pytest.raises(ValueError, match="twice"):
        drv.parse_overrides(["horizon=1.6", "horizon=3.0"])


@pytest.mark.parametrize("key", ["task", "output_root_dir", "seed", "num_samples", "num_perturb_samples",
                                 "max_sim_steps", "noise_scale", "force"])
def test_parse_overrides_refuses_driver_managed_keys(key: str) -> None:
    """Keys with a dedicated flag (or that the driver pins) cannot be set twice two ways."""
    with pytest.raises(ValueError, match="driver-managed"):
        drv.parse_overrides([f"{key}=1"])


def test_check_override_keys_refuses_unknown_fields() -> None:
    """`filter_config_fields` silently DROPS unknown keys, so a typo would run the default config."""
    drv.check_override_keys({"horizon": 1.6}, allowed={"horizon", "seed"})
    with pytest.raises(ValueError, match="horizn"):
        drv.check_override_keys({"horizn": 1.6}, allowed={"horizon", "seed"})


def test_run_meta_records_cli_overrides_and_resolved_values(tmp_path: Path) -> None:
    config = _resolved_config(tmp_path, raw_frames=100)
    config.horizon = 1.6
    meta = drv._run_meta(task="t", wall_s=1.0, overrides={}, config=config, errors=None,
                         cli_overrides={"horizon": 1.6})
    assert meta["cli_overrides"] == {"horizon": 1.6}
    assert meta["resolved_overrides"] == {"horizon": 1.6}
    assert json.loads(json.dumps(meta)) == meta


def test_run_meta_without_cli_overrides_is_unchanged(tmp_path: Path) -> None:
    config = _resolved_config(tmp_path, raw_frames=100)
    meta = drv._run_meta(task="t", wall_s=1.0, overrides={}, config=config, errors=None)
    assert "cli_overrides" not in meta and "resolved_overrides" not in meta


# --- the opt-in simulator-stack switch -------------------------------------------------------------------


def _site(tmp_path: Path, extra: tuple[str, ...] = (), drop: tuple[str, ...] = ()) -> Path:
    site = tmp_path / "site"
    for m in ("mujoco", "mujoco_warp", "warp"):
        if m not in drop:
            (site / m).mkdir(parents=True)
            (site / m / "__init__.py").write_text("")
    for d in ("mujoco-3.14.0.dist-info", "mujoco_warp-3.14.0.dist-info", "warp_lang-1.17.0.dist-info", *extra):
        (site / d).mkdir(parents=True, exist_ok=True)
    return site


def test_sim_stack_default_is_installed_and_changes_nothing() -> None:
    assert drv.DEFAULT_SIM_STACK == "installed"
    path = ["a"]
    assert drv.activate_sim_stack("installed", environ={}, path=path, modules={}) is None
    assert path == ["a"]


def test_sim_stack_mjwarp314_puts_the_site_first(tmp_path: Path) -> None:
    site = _site(tmp_path)
    path = ["a", "b"]
    out = drv.activate_sim_stack("mjwarp314", environ={drv.SIM_STACK_SITE_ENV: str(site)}, path=path, modules={})
    assert out == str(site) and path == [str(site), "a", "b"]


def test_sim_stack_refuses_missing_env_unknown_stack_and_late_selection(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match=drv.SIM_STACK_SITE_ENV):
        drv.activate_sim_stack("mjwarp314", environ={}, path=[], modules={})
    with pytest.raises(ValueError):
        drv.activate_sim_stack("mjx", environ={}, path=[], modules={})
    site = _site(tmp_path)
    with pytest.raises(RuntimeError, match="already imported"):
        drv.activate_sim_stack("mjwarp314", environ={drv.SIM_STACK_SITE_ENV: str(site)}, path=[],
                               modules={"warp": object()})


def test_sim_stack_refuses_a_site_that_would_shadow_other_packages(tmp_path: Path) -> None:
    site = _site(tmp_path, extra=("torch", "numpy-2.5.3.dist-info"))
    with pytest.raises(RuntimeError, match="more than the simulator stack"):
        drv.activate_sim_stack("mjwarp314", environ={drv.SIM_STACK_SITE_ENV: str(site)}, path=[], modules={})


def test_sim_stack_refuses_an_incomplete_site(tmp_path: Path) -> None:
    site = _site(tmp_path, drop=("mujoco_warp",))
    with pytest.raises(RuntimeError, match="missing packages"):
        drv.activate_sim_stack("mjwarp314", environ={drv.SIM_STACK_SITE_ENV: str(site)}, path=[], modules={})


def test_check_sim_stack_refuses_shadowed_versions() -> None:
    drv.check_sim_stack("installed", {"mujoco": "3.4.0"})
    drv.check_sim_stack("mjwarp314", {"mujoco": "3.14.0", "mujoco_warp": "3.14.0", "warp": "1.17.0"})
    with pytest.raises(RuntimeError, match="differ"):
        drv.check_sim_stack("mjwarp314", {"mujoco": "3.4.0", "mujoco_warp": "0.0.1", "warp": "1.10.1"})


def test_sim_stack_needs_the_fast_executor(tmp_path: Path) -> None:
    args = ["--baseline-root", str(tmp_path), "--output-root-dir", str(tmp_path), "--task", "t", "--errors-out",
            str(tmp_path / "e.npz"), "--meta-out", str(tmp_path / "m.json"), "--executor", "reference",
            "--sim-stack", "mjwarp314"]
    with pytest.raises(SystemExit):
        drv.main(args)
