"""Run every audit test in its own temporary directory: MuJoCo writes MUJOCO_LOG.TXT into the working directory when
it warns (as tests/physics and tests/policy do per test)."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _cwd_in_tmp_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
