"""GlobalHydra state around the sam2 imports and loads of `fastdo.shape.masks_sam2`.

`release/example/run_example.py` is not a Hydra app. Its box-to-mask step calls `load_sam2_image_predictor`, which
imports two sam2 modules in turn. When the second import restored "no GlobalHydra" with Hydra's own
`restore_gh_from_backup(None)`, which deletes the singleton unconditionally, the example stopped with
`KeyError: <class 'hydra.core.global_hydra.GlobalHydra'>` before the engine started. The CPU `--stub` run never
reaches that import.

`_restore_global_hydra` has three cases, each pinned here: GlobalHydra absent before and after; absent before and
created by the guarded step; present before. Stub packages stand in for sam2 (one does what `sam2/__init__` does, one
does nothing), so these tests run without sam2. The last two tests run the real loader on CPU with the real sam2
package and stop at the checkpoint download; they skip only when sam2 is not installed.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from hydra.core.singleton import Singleton
from hydra.initialize import restore_gh_from_backup

from fastdo.model_pins import HF_REVISIONS
from fastdo.shape import masks_sam2

SAM2_MISSING = "sam2 is not installed (the frontend extra installs it: uv sync --extra dev --extra frontend)"

# What sam2/__init__ does on import: initialise Hydra's global config module unless something already owns it.
SAM2_LIKE_INIT = (
    "from hydra import initialize_config_module\n"
    "from hydra.core.global_hydra import GlobalHydra\n"
    "if not GlobalHydra.instance().is_initialized():\n"
    "    initialize_config_module('stub_lib_cfg', version_base='1.2')\n"
)


def _write_pkg(root: Path, name: str, init: str, modules: tuple[str, ...] = ("first", "second")) -> None:
    pkg = root / name
    pkg.mkdir()
    (pkg / "__init__.py").write_text(init)
    for module in modules:
        (pkg / f"{module}.py").write_text("")


def _write_cfg(root: Path, name: str, key: str) -> None:
    """A config module `name` holding `app.yaml` with `{key}: 1`."""
    pkg = root / name
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "app.yaml").write_text(f"{key}: 1\n")


@pytest.fixture
def stubs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Stub packages on sys.path, GlobalHydra removed for the test and put back afterwards."""
    _write_pkg(tmp_path, "stub_sam2_like", SAM2_LIKE_INIT)
    _write_pkg(tmp_path, "stub_plain", "")
    _write_cfg(tmp_path, "stub_lib_cfg", "lib_key")
    _write_cfg(tmp_path, "stub_app_cfg", "app_key")
    monkeypatch.syspath_prepend(str(tmp_path))
    saved = Singleton._instances.pop(GlobalHydra, None)
    try:
        yield tmp_path
    finally:
        Singleton._instances.pop(GlobalHydra, None)
        if saved is not None:
            Singleton._instances[GlobalHydra] = saved
        for name in [m for m in sys.modules if m.split(".")[0].startswith("stub_")]:
            sys.modules.pop(name)


def _caller_app(initialised: bool) -> None:
    """GlobalHydra present before the guarded step: initialised with the caller's config module, or created only."""
    if initialised:
        initialize_config_module("stub_app_cfg", version_base="1.2")
    else:
        GlobalHydra.instance()


def _assert_caller_state(initialised: bool) -> None:
    assert GlobalHydra in Singleton._instances
    assert GlobalHydra.instance().is_initialized() is initialised
    if initialised:
        assert compose(config_name="app").app_key == 1      # the caller's search path is back, not the library's


def test_hydras_own_helper_raises_on_the_absent_absent_case(stubs: Path) -> None:
    """The root cause, kept as a pin: Hydra's helper deletes a key that is not there."""
    assert GlobalHydra not in Singleton._instances
    with pytest.raises(KeyError):
        restore_gh_from_backup(None)


def test_absent_before_and_after_deletes_nothing(stubs: Path) -> None:
    masks_sam2._restore_global_hydra(None)                  # direct: no KeyError
    assert GlobalHydra not in Singleton._instances
    masks_sam2._import_sam2("stub_plain.first")             # an import that never touches Hydra
    masks_sam2._import_sam2("stub_plain.second")
    assert GlobalHydra not in Singleton._instances


def test_absent_before_and_created_by_the_import_is_removed(stubs: Path) -> None:
    masks_sam2._import_sam2("stub_sam2_like.first")         # runs the package __init__, which creates GlobalHydra
    assert GlobalHydra not in Singleton._instances
    masks_sam2._import_sam2("stub_sam2_like.second")        # no __init__ this time: KeyError before the fix
    assert GlobalHydra not in Singleton._instances


@pytest.mark.parametrize("initialised", [True, False], ids=["initialised", "created-only"])
def test_present_before_is_put_back(stubs: Path, initialised: bool) -> None:
    _caller_app(initialised)
    masks_sam2._import_sam2("stub_sam2_like.first")         # created-only: the __init__ initialises it in place
    _assert_caller_state(initialised)
    masks_sam2._import_sam2("stub_sam2_like.second")
    _assert_caller_state(initialised)


@pytest.mark.parametrize("before", ["absent", "initialised", "created-only"])
def test_the_config_swap_restores_each_case(stubs: Path, monkeypatch: pytest.MonkeyPatch, before: str) -> None:
    """`_under_sam2_hydra`, the swap every sam2 build runs under, restores the same three cases, also when the build
    raises. The library's config module is the stub's, so this needs no sam2."""
    real = initialize_config_module
    monkeypatch.setattr("hydra.initialize_config_module",
                        lambda config_module, version_base=None: real("stub_lib_cfg", version_base=version_base))
    if before != "absent":
        _caller_app(before == "initialised")

    def check() -> None:
        if before == "absent":
            assert GlobalHydra not in Singleton._instances
        else:
            _assert_caller_state(before == "initialised")

    assert masks_sam2._under_sam2_hydra(lambda: compose(config_name="app").lib_key) == 1
    check()

    def failing() -> Any:
        raise RuntimeError("build failed")

    with pytest.raises(RuntimeError, match="build failed"):
        masks_sam2._under_sam2_hydra(failing)
    check()


class _CheckpointReached(Exception):
    """Raised in place of the checkpoint download: the loader got past both sam2 imports and the config swap."""


def _drop_sam2_modules() -> None:
    for name in [m for m in sys.modules if m == "sam2" or m.startswith("sam2.")]:
        sys.modules.pop(name)                               # a genuinely first import, as in a fresh process


def _stop_at_download(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def fake_download(**kwargs: Any) -> str:
        calls.append(kwargs)
        raise _CheckpointReached

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    return calls


@pytest.mark.skipif(importlib.util.find_spec("sam2") is None, reason=SAM2_MISSING)
def test_the_real_image_predictor_loader_reaches_the_checkpoint_outside_a_hydra_app(
        stubs: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The example's path on CPU, with the real sam2: `load_sam2_image_predictor` imports the predictor module, then
    `sam2.build_sam` through `_import_sam2`, then builds under the config swap and asks for the pinned checkpoint.
    Before the fix the second import raised KeyError; now the call gets to the download, and Hydra is left absent."""
    _drop_sam2_modules()
    calls = _stop_at_download(monkeypatch)
    with pytest.raises(_CheckpointReached):
        masks_sam2.load_sam2_image_predictor(device="cpu")
    assert "sam2.build_sam" in sys.modules                  # the build module was really imported, by the loader
    assert calls and calls[0]["repo_id"] == masks_sam2.MODEL_ID
    assert calls[0]["revision"] == HF_REVISIONS[masks_sam2.MODEL_ID]
    assert GlobalHydra not in Singleton._instances


@pytest.mark.skipif(importlib.util.find_spec("sam2") is None, reason=SAM2_MISSING)
def test_the_real_image_predictor_loader_inside_a_hydra_app_puts_the_apps_state_back(
        stubs: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The same inside a Hydra app (every run/pipeline script): sam2/__init__ skips its initialisation, the build
    still finds sam2's configs, and the app's search path is back afterwards."""
    _drop_sam2_modules()
    _caller_app(initialised=True)
    _stop_at_download(monkeypatch)
    with pytest.raises(_CheckpointReached):
        masks_sam2.load_sam2_image_predictor(device="cpu")
    assert "sam2.build_sam" in sys.modules
    _assert_caller_state(initialised=True)
