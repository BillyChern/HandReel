"""`python -m fastdo.audit` end to end on the synthetic scene: one run file per run, a JSON report, exit codes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fastdo.audit import example
from fastdo.audit.__main__ import main
from tests.audit import _scene as S


def _runs(tmp: Path) -> list[Path]:
    """The worked example's two simulated runs: `pressed_still` (Z0) and `carried` (Z1)."""
    return [p for p in example.write_example(tmp) if p.stem != "kinematic_only"]


def _kinematic(tmp: Path, name: str, moving: bool) -> Path:
    n, W = 120, 20
    st = S.press(S.model())
    ref = np.repeat(S.object_rows(st.qpos[None]), n, 0)
    if moving:
        ref[W:, 2] += np.linspace(0.0, 0.5, n - W)
    p = tmp / f"{name}.npz"
    np.savez(p, reference=ref, warmup_rows=np.int64(W), required_rows=np.int64(n))
    return p


@pytest.mark.parametrize("removal", ["displace", "xml"])
def test_cli_report(tmp_path: Path, removal: str, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "report.json"
    runs = _runs(tmp_path)
    before = sorted(p.name for p in tmp_path.iterdir())
    assert main([str(p) for p in runs] + ["--json", str(out), "--removal", removal]) == 0
    doc = json.loads(out.read_text())
    by = {r["clip"]: r for r in doc["runs"]}
    assert by["pressed_still"]["z0"] and not by["carried"]["z0"]
    assert by["pressed_still"]["contact_removed_null"]["status"] == "pass"
    assert by["pressed_still"]["contact_removed_null"]["verified_n0"] is True
    assert by["carried"]["contact_removed_null"]["status"] == "fail"
    assert all(r["positive_control"]["passed"] for r in doc["runs"])
    assert by["carried"]["positive_control"]["max_dev"] == 0.0           # replay equals the stored executed rows
    assert by["pressed_still"]["grip_census"]["grip_over_weight_p50"] > 1.0
    assert doc["summary"]["z0_base_rate"] == "1/2" and doc["summary"]["problems"] == []
    assert doc["summary"]["grip_census"]["runs"] == 2
    if removal == "xml":
        assert all(r["pairs_removed"] == 1 for r in doc["inputs"]["runs"])
    assert "Z0 base rate: 1/2" in capsys.readouterr().out
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(before + ["report.json"])


def test_cli_kinematic_only(tmp_path: Path) -> None:
    out = tmp_path / "r.json"
    assert main([str(_kinematic(tmp_path, "k", False)), "--certificate", "C289", "--json", str(out)]) == 0
    doc = json.loads(out.read_text())
    r = doc["runs"][0]
    assert r["certificate"] == "C289[obj+M_i+informative]" and r["certificate_complete"] is False
    assert r["contact_removed_null"]["status"] == "not_run" and r["grip_census"]["status"] == "not_run"
    assert r["motionless_object_null"]["passed"] is False         # C_289: a motionless reference is not informative
    assert doc["summary"]["reference_rejected_by_design"] == ["k"] and doc["summary"]["reference_rejected"] == []


def test_cli_threshold_config(tmp_path: Path) -> None:
    cfg = tmp_path / "loose.json"
    cfg.write_text(json.dumps({"preset": "CD", "pos_m": 1.0, "rot_rad": 3.2}))
    out = tmp_path / "r.json"
    assert main([str(_kinematic(tmp_path, "lifted", True)), "--certificate", str(cfg), "--json", str(out)]) == 0
    assert json.loads(out.read_text())["runs"][0]["z0"] is True     # 0.5 m lift passes a 1 m position threshold
