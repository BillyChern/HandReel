"""The entry points with no `FASTDO_*` variable set: a clear error naming the variable, never a write under /opt.

With no variable set, every host path resolves under the placeholder layout `/opt/fastdo` (`fastdo.hostpaths`). These
tests unset every `FASTDO_*` variable, import the entry points and run their path resolution: each must stop with a
message that names the variable to set (`HostPathNotSet`, or the script's own usage error) before it creates or reads
anything there. They skip on a host where `/opt/fastdo` really exists, since that layout is then in use.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from fastdo import hostpaths
from fastdo.amortised import gpu_lock

REPO = Path(__file__).resolve().parents[1]
PLACEHOLDER = Path(hostpaths.HOST_ROOT)

pytestmark = pytest.mark.skipif(PLACEHOLDER.exists(), reason=f"{PLACEHOLDER} exists on this host: the layout is in use")


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod                  # dataclasses resolve their module while being created
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(autouse=True)
def _no_global_hydra() -> None:
    """`import sam2` (in an earlier test module) initialises GlobalHydra, and `initialize_config_dir` refuses to run
    inside it; these tests compose the config themselves, so they start from a clear GlobalHydra."""
    from hydra.core.global_hydra import GlobalHydra

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()


@pytest.fixture
def no_fastdo_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[pytest.MonkeyPatch]:
    for var in [v for v in os.environ if v.startswith("FASTDO_")]:
        monkeypatch.delenv(var)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    yield monkeypatch
    assert not PLACEHOLDER.exists(), f"a test created {PLACEHOLDER}"


def _config(*overrides: str) -> DictConfig:
    with initialize_config_dir(config_dir=str(REPO / "run" / "conf"), version_base=None):
        return compose(config_name="config", overrides=[f"paths.root={REPO}", *overrides])


def test_require_refuses_only_placeholder_paths_that_are_missing(no_fastdo_env: pytest.MonkeyPatch,
                                                                 tmp_path: Path) -> None:
    assert hostpaths.require(tmp_path / "absent", "X") == tmp_path / "absent"        # outside /opt/fastdo: unchecked
    assert hostpaths.require(tmp_path / "new", "X", create=True) == tmp_path / "new"
    for create in (False, True):
        with pytest.raises(hostpaths.HostPathNotSet, match=r"set FASTDO_EXT_ROOT .*Host paths") as e:
            hostpaths.require(hostpaths.ext_root() / "x", "FASTDO_EXT_ROOT", create=create)
        assert str(PLACEHOLDER) in str(e.value)
    assert issubclass(hostpaths.HostPathNotSet, RuntimeError)


def test_every_default_is_the_placeholder_and_is_refused(no_fastdo_env: pytest.MonkeyPatch) -> None:
    resolved = {"FASTDO_EXT_ROOT": hostpaths.ext_root(), "FASTDO_PRIMARY_ROOT": hostpaths.primary_root(),
                "FASTDO_SCRATCH_ROOT": hostpaths.scratch_root(),
                "FASTDO_UNIT_REGISTRY": hostpaths.unit_registry_dir(), "FASTDO_MANO_DIR": hostpaths.mano_dir(),
                "FASTDO_MANO_LEFT_DIR": hostpaths.mano_left_dir(), "FASTDO_HANDS23_ROOT": hostpaths.hands23_root(),
                "FASTDO_EGO4D_LISTS": hostpaths.ego4d_lists_dir(), "FASTDO_V2D_TRACK3_ROOT": hostpaths.v2d_track3_root()}
    for var, path in resolved.items():
        assert path == PLACEHOLDER or PLACEHOLDER in path.parents, var
        with pytest.raises(hostpaths.HostPathNotSet, match=var):
            hostpaths.require(path, var, create=True)
    assert hostpaths.gpu_tools_dir() is None                          # the shared GPU lock is opt-in


def test_gpu_lock_is_not_created_under_the_placeholder(no_fastdo_env: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with pytest.raises(hostpaths.HostPathNotSet, match="FASTDO_GPU_TOOLS_DIR"):
        gpu_lock.acquire(PLACEHOLDER / "tools" / "gpu.lock", 0.0)
    grant = gpu_lock.acquire(tmp_path / "tools" / "gpu.lock", 0.0)       # an explicit directory still works
    assert grant.source == "own"
    os.close(grant.fd)


_UNIT_PROBE = """
import json, sys
from pathlib import Path
from fastdo import hostpaths
from fastdo.amortised import admission, unit

tmp = Path(sys.argv[1])
u = unit.UnitConfig(name="n", work="m:f", own_cap_mib=1, ckpt_path="c", status_path="s")
out = {"tools": str(admission.TOOLS), "lock_path": u.lock_path, "state_path": u.state_path}
try:
    unit._atomic_write(Path(u.registry_dir) / "n.json", "{}")
    out["registry"] = "no error"
except hostpaths.HostPathNotSet as e:
    out["registry"] = str(e)
# With no shared tools directory a unit runs its work with no lock and no admission check.
run = unit.UnitConfig(name="toy", work="fastdo.amortised.unit:toy_counter", own_cap_mib=64,
                      ckpt_path=str(tmp / "ckpt.json"), status_path=str(tmp / "status.json"),
                      registry_dir=str(tmp / "registry"), work_args={"n_steps": 3, "seed": 0},
                      fakes={"nvml_mib": 1})
out["rc"] = unit.run_unit(run)
status = json.loads((tmp / "status.json").read_text())
out["lock_source"], out["checks"], out["phase"] = status["lock_source"], status["checks"], status["phase"]
print(json.dumps(out))
"""


def test_amortised_units_run_without_a_shared_lock_when_none_is_set(no_fastdo_env: pytest.MonkeyPatch,
                                                                    tmp_path: Path) -> None:
    # admission and unit fix their paths at import, so they are imported in a fresh interpreter with no variable set.
    env = {k: v for k, v in os.environ.items() if not k.startswith("FASTDO_")}
    out = json.loads(subprocess.run([sys.executable, "-c", _UNIT_PROBE, str(tmp_path)], env=env, capture_output=True,
                                    text=True, check=True).stdout)
    assert (out["tools"], out["lock_path"], out["state_path"]) == ("None", "", "")
    assert "FASTDO_UNIT_REGISTRY" in out["registry"]                   # the registry default is still refused
    assert out["rc"] == 0 and out["phase"] == "finished"
    assert out["lock_source"] == "none" and out["checks"] == []          # no lock taken, no admission checked


def test_stage_paths_name_every_unset_variable(no_fastdo_env: pytest.MonkeyPatch) -> None:
    stages = ("decode", "mask", "shape", "intrinsics", "hands", "geometry", "tracks", "solve", "retarget")
    with pytest.raises(hostpaths.HostPathNotSet) as e:
        hostpaths.check_stage_paths(_config("prompt_stage.enabled=true", "retarget_stage.enabled=true"), stages)
    msg = str(e.value)
    for key, variables in (pair for keys in hostpaths.STAGE_HOST_KEYS.values() for pair in keys):
        assert f"{key}: " in msg and variables in msg, key
    with pytest.raises(hostpaths.HostPathNotSet) as e:
        hostpaths.check_stage_paths(_config("retarget_stage.enabled=true", "retarget_stage.side=right"), stages)
    assert "mano_pkl_path.right" in str(e.value) and "mano_pkl_path.left" not in str(e.value)
    hostpaths.check_stage_paths(_config(), ("decode", "mask"))                    # no host path needed


def test_stage_paths_pass_once_the_variables_are_set(no_fastdo_env: pytest.MonkeyPatch, tmp_path: Path) -> None:
    no_fastdo_env.setenv("FASTDO_EXT_ROOT", str(tmp_path))
    hostpaths.check_stage_paths(_config("prompt_stage.enabled=true", "retarget_stage.enabled=true",
                                        f"isolation.baseline_root={tmp_path}"), ("hands", "retarget"))


def test_bench_throughput_stops_before_writing_anything(no_fastdo_env: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from run.pipeline import bench_throughput as bench
    from tests.run.test_bench_throughput import _cfg, _saved_rule

    cfg = _cfg(tmp_path, "bench.doh_root=null", "bench.whisking_copies=1",
               f"qc_stage.rule_path={_saved_rule(tmp_path / 'rule.json')}")
    with pytest.raises(hostpaths.HostPathNotSet, match=r"hands\.hawor_python: .*FASTDO_CONDA_BASE"):
        bench.run_bench(cfg, REPO)
    assert not (tmp_path / "bench").exists() and not (tmp_path / "engine").exists()


def test_run_example_names_the_variables_before_any_download(no_fastdo_env: pytest.MonkeyPatch,
                                                            tmp_path: Path) -> None:
    import fastdo.bench.gpu_budget as budget

    example = _load("run_example_unset", REPO / "release" / "example" / "run_example.py")

    def no_fetch(url: str, dest: Path) -> None:
        raise AssertionError(f"downloaded {url}")

    argv = ["--out", str(tmp_path / "out"), "--stages", "decode,mask,shape,intrinsics,hands,geometry,solve"]
    with pytest.raises(SystemExit, match="FASTDO_GPU_MEM_GB"):                   # nothing set at all
        example.main(argv, fetch=no_fetch)
    no_fastdo_env.setenv("FASTDO_GPU_MEM_GB", "32")
    no_fastdo_env.setattr(budget, "apply_cpu_budget", lambda: None)              # keep this process's budgets
    no_fastdo_env.setattr(budget, "apply_gpu_budget", lambda: None)
    with pytest.raises(SystemExit, match=r"hands\.hawor_python: .*FASTDO_CONDA_BASE"):
        example.main(argv, fetch=no_fetch)
    assert not (tmp_path / "out").exists()


def test_fetch_weights_will_not_download_into_the_placeholder(no_fastdo_env: pytest.MonkeyPatch,
                                                              capsys: pytest.CaptureFixture[str]) -> None:
    fw = _load("fetch_weights_unset", REPO / "scripts" / "fetch_weights.py")

    def boom(*_a: object, **_k: object) -> None:
        raise AssertionError("a transport was called")

    with pytest.raises(SystemExit) as e:
        fw.main(["--yes", "--only", "hands23"], transports=fw.Transports(url=boom, hf=boom, git_zip=boom))
    assert e.value.code == 2 and "FASTDO_HANDS23_ROOT" in capsys.readouterr().err
    assert fw.main([], transports=fw.Transports(url=boom, hf=boom, git_zip=boom)) == 0    # the table still prints


def test_build_hoi4d_hands_names_its_variables(no_fastdo_env: pytest.MonkeyPatch, tmp_path: Path,
                                               capsys: pytest.CaptureFixture[str]) -> None:
    from run.pipeline import build_hoi4d_hands as hoi4d

    split = tmp_path / "split.json"
    split.write_text(json.dumps({"windows": [{"clip_id": "c0", "sequence": "s0", "pool": "dev", "shift_px": [0, 0]}]}))
    with pytest.raises(SystemExit) as e:
        hoi4d.main([str(split), "--out-root", str(tmp_path / "out")])
    err = capsys.readouterr().err
    assert e.value.code == 2 and "FASTDO_HAWOR_PYTHON" in err and "FASTDO_MANO_DIR" in err


def test_isolation_paths_follow_the_primary_root_and_are_refused_unset(no_fastdo_env: pytest.MonkeyPatch,
                                                                       tmp_path: Path) -> None:
    items = [("isolation", *k) for k in hostpaths.ISOLATION_HOST_KEYS]
    with pytest.raises(hostpaths.HostPathNotSet) as e:
        hostpaths.check_config_paths(_config(), items)
    for key, variables, _ in hostpaths.ISOLATION_HOST_KEYS:
        assert f"{key}: " in str(e.value) and variables in str(e.value), key
    no_fastdo_env.setenv("FASTDO_PRIMARY_ROOT", str(tmp_path))
    cfg = _config()
    assert str(cfg.isolation.baseline_root).startswith(str(tmp_path))
    assert str(cfg.isolation.temp_dir).startswith(str(tmp_path))         # scratch follows the primary root
    hostpaths.check_config_paths(cfg, items)


def test_build_retarget_scenes_names_its_variables(no_fastdo_env: pytest.MonkeyPatch,
                                                   capsys: pytest.CaptureFixture[str]) -> None:
    scenes = _load("build_retarget_scenes_unset", REPO / "run" / "pipeline" / "build_retarget_scenes.py")
    no_fastdo_env.setattr(sys, "argv", ["build_retarget_scenes.py", "--out-root", "/nonexistent-out"])
    with pytest.raises(SystemExit) as e:
        scenes.main()
    err = capsys.readouterr().err
    assert e.value.code == 2 and "FASTDO_PRIMARY_ROOT" in err and "FASTDO_CONDA_BASE" in err


def test_v2d_track3_default_root_names_its_variable(no_fastdo_env: pytest.MonkeyPatch) -> None:
    from fastdo.bench import v2d_track3

    with pytest.raises(hostpaths.HostPathNotSet, match="FASTDO_V2D_TRACK3_ROOT"):
        v2d_track3.episode_indices(hostpaths.v2d_track3_root())


def test_ego4d_lists_and_manifest_name_their_variables(no_fastdo_env: pytest.MonkeyPatch) -> None:
    from fastdo.io.datasets import ego4d_partition

    with pytest.raises(hostpaths.HostPathNotSet, match="FASTDO_EGO4D_LISTS"):
        ego4d_partition.x_pool()
    with pytest.raises(hostpaths.HostPathNotSet, match="FASTDO_PRIMARY_ROOT"):
        ego4d_partition.load_participants({"file": str(hostpaths.primary_root() / "data/ego4d/manifest.csv"),
                                           "sha256": "0"})


def test_unitree_ros_lookup_names_its_variable(no_fastdo_env: pytest.MonkeyPatch) -> None:
    from fastdo.robot import g1_model

    if (REPO / "third_party" / "unitree_ros" / g1_model.UNITREE_ROS_XML).is_file():
        pytest.skip("this checkout has its own unitree_ros clone")
    # fixed at import (possibly under an earlier override): what a fresh process with no variable set computes
    no_fastdo_env.setattr(g1_model, "_MAIN_CHECKOUT", hostpaths.primary_root() / "third_party" / "unitree_ros")
    with pytest.raises(FileNotFoundError, match="FASTDO_UNITREE_ROS"):
        g1_model.resolve_unitree_ros()


def test_setting_the_gpu_tools_dir_turns_the_shared_lock_on(no_fastdo_env: pytest.MonkeyPatch, tmp_path: Path) -> None:
    env = {**{k: v for k, v in os.environ.items() if not k.startswith("FASTDO_")}, "FASTDO_GPU_TOOLS_DIR": str(tmp_path)}
    probe = ("import json; from fastdo.amortised import unit; u = unit.UnitConfig(name='n', work='m:f', "
             "own_cap_mib=1, ckpt_path='c', status_path='s'); print(json.dumps([u.lock_path, u.state_path]))")
    out = json.loads(subprocess.run([sys.executable, "-c", probe], env=env, capture_output=True, text=True,
                                    check=True).stdout)
    assert out == [str(tmp_path / "gpu.lock"), str(tmp_path / "gpu_priority.json")]
