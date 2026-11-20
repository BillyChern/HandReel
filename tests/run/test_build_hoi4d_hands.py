"""CPU tests for run/pipeline/build_hoi4d_hands.py: portable defaults, held-out refusal and the child exit code.

The HaWoR driver is never run: `--hawor-python` points at a stub interpreter that ignores its arguments and exits
with a chosen code, so the tests check only the wrapper's own logic.
"""
from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from run.pipeline import build_hoi4d_hands as B

ENV_VARS = ("FASTDO_ROOT", "FASTDO_HAWOR_PYTHON", "FASTDO_HAWOR_ROOT", "FASTDO_MANO_DIR")


def _split(tmp_path: Path, pools: list[str]) -> Path:
    windows = [{"clip_id": f"c{i}", "sequence": f"seq{i}", "pool": p, "shift_px": [0, 0]} for i, p in enumerate(pools)]
    f = tmp_path / "split.json"
    f.write_text(json.dumps({"windows": windows}))
    return f


def _stub(tmp_path: Path, rc: int) -> Path:
    f = tmp_path / f"stub_{rc}.sh"
    f.write_text(f"#!/bin/sh\necho stub\nexit {rc}\n")
    f.chmod(f.stat().st_mode | stat.S_IXUSR)
    return f


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


@pytest.mark.parametrize("pool", ["final_test", "FINAL_TEST", " Final_Test "])
def test_final_test_refused_in_any_case(tmp_path: Path, clean_env: pytest.MonkeyPatch, pool: str) -> None:
    with pytest.raises(SystemExit, match="REFUSING"):
        B.main([str(_split(tmp_path, ["dev"])), "--pool", pool])


@pytest.mark.parametrize("rc, expected", [(0, 0), (3, 1)])
def test_child_failure_sets_exit_code(tmp_path: Path, clean_env: pytest.MonkeyPatch, rc: int, expected: int) -> None:
    argv = [str(_split(tmp_path, ["dev", "dev"])), "--root", str(tmp_path), "--out-root", str(tmp_path / "out"),
            "--hawor-python", str(_stub(tmp_path, rc)), "--mano-right-pkl", str(tmp_path / "MANO_RIGHT.pkl")]
    assert B.main(argv) == expected


def test_missing_interpreter_is_an_error_only_when_a_window_needs_it(tmp_path: Path,
                                                                     clean_env: pytest.MonkeyPatch) -> None:
    split = _split(tmp_path, ["dev"])
    with pytest.raises(SystemExit) as e:
        B.main([str(split), "--root", str(tmp_path), "--out-root", str(tmp_path / "out")])
    assert e.value.code == 2
    cached = B.hands_npz(tmp_path / "out", "seq0")
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"")
    assert B.main([str(split), "--root", str(tmp_path), "--out-root", str(tmp_path / "out")]) == 0


def test_empty_env_values_mean_unset(tmp_path: Path, clean_env: pytest.MonkeyPatch,
                                     capsys: pytest.CaptureFixture[str]) -> None:
    for var in ENV_VARS:
        clean_env.setenv(var, "")
    # An empty FASTDO_ROOT falls back to this checkout (never the current directory); empty interpreter and MANO
    # values count as unset, so a window that needs the driver stops with a usage error instead of running "".
    with pytest.raises(SystemExit) as e:
        B.main([str(_split(tmp_path, ["dev"])), "--out-root", str(tmp_path / "out")])
    assert e.value.code == 2
    assert "set --hawor-python" in capsys.readouterr().err


def test_root_default_is_the_checkout_when_env_is_empty(tmp_path: Path, clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("FASTDO_ROOT", "")
    seen: dict[str, list[str]] = {}

    def fake_run(cmd: list[str], **_kw: object) -> object:
        seen["cmd"] = cmd
        return type("R", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    clean_env.setattr(B.subprocess, "run", fake_run)
    argv = [str(_split(tmp_path, ["dev"])), "--out-root", str(tmp_path / "out"),
            "--hawor-python", "py", "--mano-right-pkl", "m.pkl"]
    assert B.main(argv) == 0
    cmd = seen["cmd"]
    assert cmd[cmd.index("--hoi4d-root") + 1] == str(B.REPO / "data/hoi4d")
    assert cmd[cmd.index("--hawor-root") + 1] == str(B.REPO / "third_party/do-as-i-do/reconstruction/modules/HaWoR")


def test_empty_cli_path_is_rejected(tmp_path: Path, clean_env: pytest.MonkeyPatch) -> None:
    with pytest.raises(SystemExit) as e:
        B.main([str(_split(tmp_path, ["dev"])), "--hawor-python", ""])
    assert e.value.code == 2
