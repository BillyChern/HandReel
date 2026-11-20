"""The worked example (`python -m fastdo.audit.example`) runs on CPU and gives the documented report."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fastdo.audit import example


def test_worked_example(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert example.main([str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "Z0 base rate: 1/3" in out
    assert "hand_absent: passes on 1/2 tested clips (Z0 1/1, Z1 0/1)" in out
    doc = json.loads((tmp_path / "report.json").read_text())
    by = {r["clip"]: r for r in doc["runs"]}
    assert by["pressed_still"]["z0"] and by["pressed_still"]["contact_removed_null"]["status"] == "pass"
    assert by["pressed_still"]["contact_removed_null"]["verified_n0"] is True
    assert 7.0 < by["pressed_still"]["grip_census"]["grip_over_weight_p50"] < 8.0
    assert by["carried"]["reference_passes"] and not by["carried"]["z0"]
    assert by["carried"]["contact_removed_null"]["status"] == "fail"
    assert 0.9 < by["carried"]["grip_census"]["grip_over_weight_p50"] < 1.1
    assert by["kinematic_only"]["contact_removed_null"]["status"] == "not_run"
    assert by["kinematic_only"]["z0"] is False


def test_help_and_bad_arguments_write_nothing(tmp_path, monkeypatch, capsys):
    """`--help` prints usage and exits 0; an option-like argument is refused; neither creates a directory."""
    from fastdo.audit import example

    monkeypatch.chdir(tmp_path)
    assert example.main(["--help"]) == 0
    assert "usage:" in capsys.readouterr().out
    assert example.main(["--json", "x"]) == 2
    assert list(tmp_path.iterdir()) == []
