"""Tests for `run/robot/export_g1_package.py`: the CLI writes a package for a synthetic clip and refuses
bad input."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from tests.robot import synthetic

_PATH = Path(__file__).resolve().parents[2] / "run" / "robot" / "export_g1_package.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("export_g1_package", _PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_writes_a_package(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from fastdo.robot.g1_model import load_g1_upper, resolve_unitree_ros
    try:
        resolve_unitree_ros()
    except FileNotFoundError:
        pytest.skip("official unitree_ros clone not available")
    cli = _load()
    clip = synthetic.write_clip(load_g1_upper(), tmp_path / "clip")
    out = tmp_path / "pkg"
    rc = cli.main([str(clip), "--out", str(out), "--object-xy", *map(str, synthetic.OBJECT_XY),
                   "--table-z", str(synthetic.TABLE_Z), "--arm-vmax", "0.5"])
    assert rc == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["package"] == str(out) and printed["success"] is False
    assert printed["execute_allowed"] is False and printed["gate_failures"]       # the box is never touched
    meta = json.loads((out / "package.json").read_text())
    assert meta["limits"]["arm_vmax"] == [0.5] * 7 and meta["table_z"] == pytest.approx(synthetic.TABLE_Z)


def test_cli_refuses_bad_input(tmp_path: Path) -> None:
    cli = _load()
    assert cli.main([str(tmp_path), "--out", str(tmp_path / "pkg")]) == 2
    args = cli.parse_args([str(tmp_path), "--out", "x", "--object-xy", "0.3", "-0.1", "--yaw-offset-deg", "90"])
    cfg = cli.build_config(args)
    assert cfg.table_xy == (0.3, -0.1) and cfg.yaw_offset == pytest.approx(1.5707963, abs=1e-6)


def test_cli_camera_pitch_flag(tmp_path: Path) -> None:
    cli = _load()
    assert cli.build_config(cli.parse_args([str(tmp_path), "--out", "x"])).camera_pitch_deg is None
    cfg = cli.build_config(cli.parse_args([str(tmp_path), "--out", "x", "--camera-pitch-deg", "45"]))
    assert cfg.camera_pitch_deg == 45.0
    (tmp_path / "clip_record.npz").write_bytes(b"")
    (tmp_path / "mesh.obj").write_text("")
    assert cli.main([str(tmp_path), "--out", str(tmp_path / "pkg"), "--camera-pitch-deg", "95"]) == 2


def test_cli_search_and_adaptation_flags(tmp_path: Path) -> None:
    cli = _load()
    args = cli.parse_args([str(tmp_path), "--out", "x", "--first-row", "10", "--no-trim", "--table-z-range", "0.72",
                           "0.86", "--object-xy", "0.3", "-0.2", "--sym-axis", "2", "--sym-deg", "90"])
    cfg = cli.build_config(args)
    assert cfg.first_row == 10 and cfg.trim_to_contact is False and cfg.search.table_z_range == (0.72, 0.86)
    assert cfg.sym_axis == 2 and cfg.sym_deg == 90.0
    default = cli.build_config(cli.parse_args([str(tmp_path), "--out", "x"]))
    assert default.table_xy is None and default.trim_to_contact and default.first_row == 0
    (tmp_path / "clip_record.npz").write_bytes(b"")
    (tmp_path / "mesh.obj").write_text("")
    assert cli.main([str(tmp_path), "--out", str(tmp_path / "pkg"), "--table-z-range", "0.9", "0.7"]) == 2


def test_cli_accepts_a_directory_without_a_record(tmp_path: Path) -> None:
    """A directory with the solve, hand and intrinsics artifacts passes the input check (the record is rebuilt);
    without `hands.npz` it is refused."""
    cli = _load()
    (tmp_path / "mesh.obj").write_text("")
    (tmp_path / "solve.npz").write_bytes(b"")
    (tmp_path / "intrinsics.npz").write_bytes(b"")
    assert cli.main([str(tmp_path), "--out", str(tmp_path / "pkg")]) == 2          # no hands.npz
    (tmp_path / "hands.npz").write_bytes(b"")
    with pytest.raises((ValueError, EOFError, OSError)):                           # past the check: the loader reads
        cli.main([str(tmp_path), "--out", str(tmp_path / "pkg"), "--table-z-range", "0.7", "0.9"])


def test_cli_home_q_adds_a_checked_home(tmp_path: Path) -> None:
    cli = _load()
    q = [0.1] * 17
    cfg = cli.build_config(cli.parse_args([str(tmp_path), "--out", "x", "--home-q", *map(str, q)]))
    assert [n for n, _ in cfg.transfer.homes] == ["zeros", "arms_hanging", "measured"]
    assert cfg.transfer.homes[-1][1] == tuple(q)
    assert [n for n, _ in cli.build_config(cli.parse_args([str(tmp_path), "--out", "x"])).transfer.homes] == \
        ["zeros", "arms_hanging"]
    (tmp_path / "clip_record.npz").write_bytes(b"")
    (tmp_path / "mesh.obj").write_text("")
    assert cli.main([str(tmp_path), "--out", str(tmp_path / "pkg"), "--home-q", *(["nan"] * 17)]) == 2


# --- --speed-scale simulates the run at that scale and records it; --home-q waist -----

def test_cli_speed_scale_is_simulated_and_recorded(tmp_path: Path) -> None:
    """`--speed-scale 0.5`: `transfers` records 0.5, the simulated playback of every run takes twice the physics steps
    of the 1.0 export (mutation: the scale recorded but not simulated), the drills and tolerance runs too, and the
    package states it (README, re-export command); out-of-range values exit 2."""
    from fastdo.robot.g1_model import load_g1_upper, resolve_unitree_ros
    try:
        resolve_unitree_ros()
    except FileNotFoundError:
        pytest.skip("official unitree_ros clone not available")
    cli = _load()
    clip = synthetic.write_clip(load_g1_upper(), tmp_path / "clip")
    explicit = ["--object-xy", *map(str, synthetic.OBJECT_XY), "--table-z", str(synthetic.TABLE_Z), "--arm-vmax", "0.5"]
    for bad in ("0", "1.5", "-0.5"):
        assert cli.main([str(clip), "--out", str(tmp_path / f"bad{bad}"), *explicit, "--speed-scale", bad]) == 2
    assert cli.main([str(clip), "--out", str(tmp_path / "half"), *explicit, "--speed-scale", "0.5"]) == 0
    assert cli.main([str(clip), "--out", str(tmp_path / "full"), *explicit]) == 0
    half, full = (json.loads((tmp_path / n / "sim_stats.json").read_text())["transfers"] for n in ("half", "full"))
    assert half["speed_scale"] == 0.5 and full["speed_scale"] == 1.0

    def runs(tr: dict) -> list[dict]:
        rec = next(r for r in tr["homes"] if r["index"] == 0)
        return [rec["run"], *rec["drills"], *rec["table_tolerance_runs"], *rec["placement_runs"], *rec["state_runs"]]

    def play_steps(tr: dict) -> list[int]:
        return [r["phases"]["play"]["steps"] for r in runs(tr) if "play" in r["phases"]]
    # the operator-stop drills too, and every run records the scale it was simulated at
    assert len(play_steps(half)) == len(runs(half)) - 1                   # all but the drill stopped in TO_HOLD
    for h, f in zip(play_steps(half), play_steps(full), strict=True):
        assert h == pytest.approx(2 * f, rel=0.02), (h, f)
    assert {r["run_options"]["speed_scale"] for r in runs(half)} == {0.5}
    assert {r["run_options"]["speed_scale"] for r in runs(full)} == {1.0}
    meta = json.loads((tmp_path / "half" / "package.json").read_text())
    assert meta["sim_gate"]["simulated_run_options"]["speed_scale"] == 0.5
    assert "--speed-scale 0.5" in meta["reexport_command"]
    assert "--speed-scale" not in json.loads((tmp_path / "full" / "package.json").read_text())["reexport_command"]
    readme = (tmp_path / "half" / "README.md").read_text()
    assert "Playback speed scale 0.5" in readme and "bring-up package" in readme


@pytest.mark.parametrize(("joint", "value"), [(15, "0.02"), (14, "0.0101"), (15, "-0.0101"), (16, "0.0101")])
def test_cli_refuses_a_home_q_waist_off_zero(tmp_path: Path, caplog: pytest.LogCaptureFixture, joint: int,
                                             value: str) -> None:
    """A measured home whose waist (roll 14, pitch 15, yaw 16) is more than 0.01 rad from 0 can never be started from
    (the run script refuses it), so the export refuses it (exit 2) before any work; each waist joint alone
    (the yaw was once left unchecked)."""
    cli = _load()
    (tmp_path / "clip_record.npz").write_bytes(b"")
    (tmp_path / "mesh.obj").write_text("")
    home = ["0"] * 17
    home[joint] = value
    caplog.set_level("ERROR")
    assert cli.main([str(tmp_path), "--out", str(tmp_path / "pkg"), "--home-q", *home]) == 2
    assert "waist" in caplog.text and not (tmp_path / "pkg").exists()
