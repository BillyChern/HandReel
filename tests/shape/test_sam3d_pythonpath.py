"""`ShapeConfig.sam3d_pythonpath`: extra PYTHONPATH entries for the SAM 3D worker process.

The worker is a fake interpreter (a POSIX shell script) that records the PYTHONPATH it was started with and answers
the ready line, so no SAM 3D code, weights or GPU are involved.
"""

from __future__ import annotations

import logging
import os
import stat
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from omegaconf import OmegaConf

from fastdo.shape import ShapeConfig, sam3d_backend
from fastdo.shape.sam3d_backend import SAM3D_PYTHONPATH_ENV, start_sam3d_worker

_REPO = Path(__file__).resolve().parents[2]
_UNSET = "<unset>"


def _snapshot(tmp_path: Path) -> Callable[..., str]:
    snapshot = tmp_path / "snapshot"
    (snapshot / "checkpoints").mkdir(parents=True, exist_ok=True)
    (snapshot / "checkpoints" / "pipeline.yaml").write_text("_target_: x\n")
    return lambda *args, **kwargs: str(snapshot)


def _fake_interpreter(tmp_path: Path) -> tuple[Path, Path]:
    """A worker 'python' that writes its PYTHONPATH (or `_UNSET`) to a file, reports ready and exits on shutdown."""
    record = tmp_path / "pythonpath.txt"
    script = tmp_path / "fake_python.sh"
    script.write_text("#!/bin/sh\n"
                      f"printf '%s' \"${{PYTHONPATH-{_UNSET}}}\" > '{record}'\n"
                      "echo '{\"ready\": true, \"load_seconds\": 0.0, \"spconv_kernels\": \"pinned\"}'\n"
                      "read line\n"
                      "exit 0\n")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return script, record


def _worker_pythonpath(tmp_path: Path, **fields: Any) -> str:
    python, record = _fake_interpreter(tmp_path)
    cfg = ShapeConfig(backend="sam3d", sam3d_python=str(python), device="cpu", **fields)
    client = start_sam3d_worker(cfg, snapshot_fn=_snapshot(tmp_path), startup_timeout_s=60.0)
    assert client.close(timeout_s=30.0) == 0
    return record.read_text()


def test_entries_are_prepended_in_order_before_the_existing_pythonpath(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    existing = os.pathsep.join(["/existing/a", "/existing/b"])
    monkeypatch.setenv("PYTHONPATH", existing)
    monkeypatch.setenv(SAM3D_PYTHONPATH_ENV, "/ignored/by/an/explicit/config")
    monkeypatch.setenv("FASTDO_TEST_SECRET_TOKEN", "not-a-real-secret-value")
    with caplog.at_level(logging.INFO, logger=sam3d_backend.__name__):
        seen = _worker_pythonpath(tmp_path, sam3d_pythonpath=("/pylib/flash", "/pylib/dgr"))
    assert seen == os.pathsep.join(["/pylib/flash", "/pylib/dgr", "/existing/a", "/existing/b"])
    assert os.environ["PYTHONPATH"] == existing                       # the parent's environment is untouched
    assert "/pylib/flash" in caplog.text and "/pylib/dgr" in caplog.text
    assert "not-a-real-secret-value" not in caplog.text and "/existing/a" not in caplog.text   # never the environment


def test_entries_without_an_existing_pythonpath(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for value in (None, ""):                                          # unset, and set but empty (no stray separator)
        if value is None:
            monkeypatch.delenv("PYTHONPATH", raising=False)
        else:
            monkeypatch.setenv("PYTHONPATH", value)
        assert _worker_pythonpath(tmp_path, sam3d_pythonpath=("/pylib/flash",)) == "/pylib/flash"


def test_none_reads_the_environment_variable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTHONPATH", "/existing")
    monkeypatch.setenv(SAM3D_PYTHONPATH_ENV, os.pathsep.join(["/env/flash", "", "/env/dgr"]))   # empty parts skipped
    assert ShapeConfig().sam3d_pythonpath is None
    assert _worker_pythonpath(tmp_path) == os.pathsep.join(["/env/flash", "/env/dgr", "/existing"])
    assert sam3d_backend.resolve_sam3d_pythonpath(ShapeConfig(sam3d_pythonpath=())) == ()   # explicit: no entries


@pytest.mark.parametrize("env_value", [None, ""])
def test_no_entries_leaves_the_worker_environment_inherited(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                           env_value: str | None) -> None:
    if env_value is None:
        monkeypatch.delenv(SAM3D_PYTHONPATH_ENV, raising=False)
    else:
        monkeypatch.setenv(SAM3D_PYTHONPATH_ENV, env_value)
    started: list[tuple[list[str], Any]] = []

    class _Recorder:
        def __init__(self, command: list[str], env: Any = None, startup_timeout_s: float = 0.0,
                     spconv_kernels: str | None = None) -> None:
            started.append((list(command), env))

    monkeypatch.setattr(sam3d_backend, "Sam3dWorkerClient", _Recorder)
    python, _record = _fake_interpreter(tmp_path)
    for cfg in (ShapeConfig(backend="sam3d", sam3d_python=str(python)),
                ShapeConfig(backend="sam3d", sam3d_python=str(python), sam3d_pythonpath=())):
        start_sam3d_worker(cfg, snapshot_fn=_snapshot(tmp_path))
    assert [env for _command, env in started] == [None, None]

    monkeypatch.undo()                                                # the real client: PYTHONPATH stays unset
    monkeypatch.delenv(SAM3D_PYTHONPATH_ENV, raising=False)
    monkeypatch.delenv("PYTHONPATH", raising=False)
    assert _worker_pythonpath(tmp_path) == _UNSET


@pytest.mark.parametrize("value", ["/a:/b", ("",), ("/a", ""), ("/a", 3), (Path("/a"),), 5])
def test_invalid_values_raise(value: object) -> None:
    with pytest.raises(ValueError, match="sam3d_pythonpath"):
        ShapeConfig(sam3d_pythonpath=value)  # type: ignore[arg-type]


def test_lists_become_tuples_and_the_hydra_default_is_null() -> None:
    assert ShapeConfig(sam3d_pythonpath=["/a", "/b"]).sam3d_pythonpath == ("/a", "/b")  # type: ignore[arg-type]
    node = OmegaConf.load(_REPO / "run" / "conf" / "config.yaml").bench.shape
    assert "sam3d_pythonpath" in node and node.sam3d_pythonpath is None
    shape = OmegaConf.to_container(node, resolve=True)
    assert isinstance(shape, dict)
    assert ShapeConfig(**{str(k): v for k, v in shape.items()}).sam3d_pythonpath is None
