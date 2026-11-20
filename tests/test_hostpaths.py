"""Host paths moved from literals to env-var keys resolve to the pinned literals with no variable set.

Each changed config key and code constant is pinned to the exact literal it replaced (known answer), with every
override variable unset; the overrides are then checked to reach the config and the code alike."""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from fastdo import hostpaths

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "run" / "conf" / "config.yaml"
OLD_RESOLVED = REPO / "tests" / "fixtures" / "release" / "config_defaults_resolved.json"
VARS = ("FASTDO_EXT_ROOT", "FASTDO_PRIMARY_ROOT", "FASTDO_SCRATCH_ROOT", "FASTDO_GPU_TOOLS_DIR", "FASTDO_UNIT_REGISTRY",
        "FASTDO_MANO_DIR", "FASTDO_MANO_LEFT_DIR", "FASTDO_HANDS23_ROOT", "FASTDO_HANDS23_PYTHON",
        "FASTDO_HANDS23_WEIGHTS", "FASTDO_CONDA_BASE")

# The literals these keys held before they became env-var keys.
OLD_CONFIG = {
    "paths.conda_base": "/opt/fastdo/miniforge3",
    "isolation.mano_dir": "/opt/fastdo/HaWoR/_DATA/data/mano",
    "isolation.mano_left_dir": "/opt/fastdo/HaWoR/_DATA/data_left/mano_left",
    "isolation.mano_right_pkl": "/opt/fastdo/HaWoR/_DATA/data/mano/MANO_RIGHT.pkl",
    "isolation.mano_left_pkl":
        "/opt/fastdo/HaWoR/_DATA/data_left/mano_left/MANO_LEFT.pkl",
    "isolation.conda_bin": "/opt/fastdo/miniforge3/bin/conda",
    "prompt_stage.python": "/opt/fastdo/handreel/temp/hands23/env_hands23/bin/python",
    "prompt_stage.weights": "/opt/fastdo/handreel/temp/hands23/weights/model_hands23.pth",
    "retarget_stage.mano_pkl_path.right":
        "/opt/fastdo/HaWoR/_DATA/data/mano/MANO_RIGHT.pkl",
    "retarget_stage.mano_pkl_path.left":
        "/opt/fastdo/HaWoR/_DATA/data_left/mano_left/MANO_LEFT.pkl",
    "hands.hawor_python": "/opt/fastdo/miniforge3/envs/hawor/bin/python",
}
OLD_REGISTRY = "/opt/fastdo/handreel/temp/units"
OLD_GATE_REFIT = "/opt/fastdo/handreel/temp/hands23"


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for v in VARS:
        monkeypatch.delenv(v, raising=False)
    return monkeypatch


def _cfg():
    return OmegaConf.load(CONFIG)


def _gate_refit_expr():
    """`hands23_detect.GATE_REFIT` (and its `_env_dir` helper), evaluated without importing the module (it needs
    torch)."""
    src = (REPO / "src/fastdo/prompt/hands23_detect.py").read_text()
    body = ast.parse(src).body
    helper = next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == "_env_dir")
    node = next(n for n in body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "GATE_REFIT")
    ns = {"os": os, "Path": Path}
    exec(compile(ast.Module([helper], []), "hands23_detect.py", "exec"), ns)  # noqa: S102 (our own source)
    code = compile(ast.Expression(node.value), "hands23_detect.py", "eval")
    return lambda: eval(code, ns)


@pytest.mark.parametrize("key", sorted(OLD_CONFIG))
def test_config_key_resolves_to_the_old_literal(clean_env: pytest.MonkeyPatch, key: str) -> None:
    assert OmegaConf.select(_cfg(), key, throw_on_missing=True) == OLD_CONFIG[key]


def test_no_config_value_changed_anywhere(clean_env: pytest.MonkeyPatch) -> None:
    """Whole-config known answer: every resolved leaf outside the keys added since equals the old config's.

    The added keys are named and popped here: the new `paths.*` keys, and `policy_scenes.pin` and `policy_scenes.cpu`
    (the scene driver's one-CPU launch), which are also pinned to their defaults true and null. The old config
    (`run/conf/config.yaml` before the change, sha256 25f15cde...) is vendored RESOLVED (FASTDO_ROOT and every FASTDO_*
    path variable unset) as a fixture, so this runs, and fails rather than skips, without git history."""
    clean_env.delenv("FASTDO_ROOT", raising=False)
    old_c = json.loads(OLD_RESOLVED.read_text())
    new_c = json.loads(json.dumps(OmegaConf.to_container(_cfg(), resolve=True)))
    for k in ("ext", "primary", "scratch", "mano_dir", "mano_left_dir", "hands23_root"):
        new_c["paths"].pop(k)
    assert (new_c["policy_scenes"].pop("pin"), new_c["policy_scenes"].pop("cpu")) == (True, None)
    # The two notes files moved from a development notes file (not shipped) to outputs/whisking/; the fixture's old
    # values are redacted to the new defaults.
    assert (new_c["whisking"].pop("notes_path"), new_c["hands_parity"].pop("notes_path")) == (
        "./outputs/whisking/notes.md", "./outputs/whisking/hands_parity_notes.md")
    assert (old_c["whisking"].pop("notes_path"), old_c["hands_parity"].pop("notes_path")) == (
        "./outputs/whisking/notes.md", "./outputs/whisking/hands_parity_notes.md")
    # The gate's DexYCB exclusion list named two evaluation files of the development host (not shipped); the release
    # default is empty and the user supplies their own (the fixture's old value is redacted to the same empty list).
    assert (new_c["gate_data"].pop("dexycb_exclude_files"), old_c["gate_data"].pop("dexycb_exclude_files")) == ([], [])
    assert new_c == old_c


def test_prompt_stage_config_defaults_and_from_config(clean_env: pytest.MonkeyPatch) -> None:
    from fastdo.engine.workers_prompt import PromptStageConfig

    for c in (PromptStageConfig(), PromptStageConfig.from_mapping(_cfg().prompt_stage)):
        assert c.python == OLD_CONFIG["prompt_stage.python"]
        assert c.weights == OLD_CONFIG["prompt_stage.weights"]


_MODULE_CONSTANTS = """
import json
from fastdo.amortised import admission, unit
u = unit.UnitConfig(name="n", work="m:f", own_cap_mib=1, ckpt_path="c", status_path="s")
print(json.dumps([str(admission.TOOLS), str(admission.STATE_PATH), str(admission.LOCK_PATH), str(admission.PEER_HOLD),
                  str(unit.REGISTRY), [str(u.registry_dir), str(u.state_path), str(u.lock_path)]]))
"""


def test_code_constants_are_the_old_literals(clean_env: pytest.MonkeyPatch) -> None:
    # admission and unit fix these constants at import, so they are read in a fresh interpreter with every override
    # unset: an earlier import in this session under an override (FASTDO_GPU_TOOLS_DIR, as the README sets it for the
    # test run) must not decide the result.
    env = {k: v for k, v in os.environ.items() if k not in VARS}
    out = subprocess.run([sys.executable, "-c", _MODULE_CONSTANTS], env=env, capture_output=True, text=True,
                         check=True).stdout
    tools, state, lock, peer, registry, ucfg = json.loads(out)
    assert (tools, state, lock, peer) == ("None",) * 4              # the shared GPU lock is opt-in
    assert registry == OLD_REGISTRY
    assert ucfg == [OLD_REGISTRY, "", ""]
    assert str(_gate_refit_expr()()) == OLD_GATE_REFIT
    assert str(hostpaths.mano_dir()) == OLD_CONFIG["isolation.mano_dir"]
    assert str(hostpaths.mano_left_dir()) == OLD_CONFIG["isolation.mano_left_dir"]


@pytest.mark.parametrize("var", ["FASTDO_EXT_ROOT", "FASTDO_PRIMARY_ROOT", "FASTDO_SCRATCH_ROOT",
                                 "FASTDO_HANDS23_ROOT"])
def test_overrides_reach_config_code_and_the_detector_alike(clean_env: pytest.MonkeyPatch, tmp_path: Path,
                                                            var: str) -> None:
    clean_env.setenv(var, str(tmp_path / "x"))
    cfg = _cfg()
    assert cfg.paths.hands23_root == str(hostpaths.hands23_root())
    assert _gate_refit_expr()() == hostpaths.hands23_root()
    assert str(hostpaths.hands23_root()).startswith(str(tmp_path / "x"))
    assert cfg.prompt_stage.python == str(hostpaths.hands23_root() / "env_hands23/bin/python")
    if var != "FASTDO_HANDS23_ROOT":
        assert cfg.paths.scratch == str(hostpaths.scratch_root())
        assert hostpaths.gpu_tools_dir() is None                      # roots never set the GPU lock


def test_mano_and_single_file_overrides(clean_env: pytest.MonkeyPatch, tmp_path: Path) -> None:
    clean_env.setenv("FASTDO_MANO_DIR", str(tmp_path / "m"))
    clean_env.setenv("FASTDO_MANO_LEFT_DIR", str(tmp_path / "l"))
    clean_env.setenv("FASTDO_HANDS23_WEIGHTS", str(tmp_path / "w.pth"))
    cfg = _cfg()
    assert cfg.retarget_stage.mano_pkl_path.right == str(tmp_path / "m/MANO_RIGHT.pkl")
    assert cfg.isolation.mano_left_pkl == str(tmp_path / "l/MANO_LEFT.pkl")
    assert hostpaths.mano_dir() == tmp_path / "m"
    assert cfg.prompt_stage.weights == str(tmp_path / "w.pth")
    from fastdo.engine.workers_prompt import PromptStageConfig

    assert PromptStageConfig().weights == str(tmp_path / "w.pth")


def test_an_empty_override_fails_loudly(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("FASTDO_UNIT_REGISTRY", "")
    with pytest.raises(ValueError, match="FASTDO_UNIT_REGISTRY is set but empty"):
        hostpaths.unit_registry_dir()


@pytest.mark.parametrize("var", ["FASTDO_EXT_ROOT", "FASTDO_PRIMARY_ROOT", "FASTDO_SCRATCH_ROOT"])
def test_moving_a_root_never_turns_the_gpu_lock_on(clean_env: pytest.MonkeyPatch, tmp_path: Path, var: str) -> None:
    clean_env.setenv(var, str(tmp_path / "elsewhere"))
    assert hostpaths.gpu_tools_dir() is None
    assert str(hostpaths.scratch_root()).startswith(str(tmp_path))     # the root itself did move


def test_the_gpu_lock_is_opt_in(clean_env: pytest.MonkeyPatch, tmp_path: Path) -> None:
    assert hostpaths.gpu_tools_dir() is None
    clean_env.setenv("FASTDO_GPU_TOOLS_DIR", str(tmp_path / "tools"))
    assert hostpaths.gpu_tools_dir() == tmp_path / "tools"
    clean_env.setenv("FASTDO_GPU_TOOLS_DIR", "")
    with pytest.raises(ValueError, match="FASTDO_GPU_TOOLS_DIR is set but empty"):
        hostpaths.gpu_tools_dir()


@pytest.mark.parametrize("var", ["FASTDO_EXT_ROOT", "FASTDO_PRIMARY_ROOT", "FASTDO_SCRATCH_ROOT",
                                 "FASTDO_HANDS23_ROOT"])
def test_an_empty_variable_fails_in_the_detector_as_in_hostpaths(clean_env: pytest.MonkeyPatch, var: str) -> None:
    clean_env.setenv(var, "")
    with pytest.raises(ValueError, match=f"{var} is set but empty"):
        hostpaths.hands23_root()
    with pytest.raises(ValueError, match=f"{var} is set but empty"):
        _gate_refit_expr()()
