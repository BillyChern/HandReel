# Copyright 2026 The alongray authors
# SPDX-License-Identifier: Apache-2.0
"""The CLI scores two .npz files and agrees with the Python API."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import alongray
from alongray import score
from alongray.cli import main


def _traj(n: int = 20, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.array([0.05, -0.03, 0.55])
    return base + 0.08 * rng.standard_normal((n, 3)) * np.array([1.0, 1.0, 0.5])


def _files(tmp_path: Path) -> tuple[Path, Path, dict]:
    rng = np.random.default_rng(1)
    obj = _traj(15, 1)
    joints = _traj(15, 2)[:, None, :] + 0.03 * rng.standard_normal((15, 21, 3))
    pred_obj = 1.08 * obj + rng.normal(0, 0.005, obj.shape)
    pred_joints = 0.95 * joints
    k = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
    pred, gt = tmp_path / "pred.npz", tmp_path / "gt.npz"
    np.savez(pred, object=pred_obj, hand=pred_joints)
    np.savez(gt, object=obj, hand=joints, K=k)
    ref = score(pred_obj, obj, pred_joints, joints, k).as_dict()
    return pred, gt, ref


def test_cli_prints_the_api_result(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pred, gt, ref = _files(tmp_path)
    out_file = tmp_path / "s.json"
    assert main([str(pred), str(gt), "--out", str(out_file)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed == json.loads(out_file.read_text())
    for body in ("object", "hand"):
        for key in ("e_par", "gauge_part", "off_orbit_part", "lam", "z_rms", "reproj_rms_px"):
            assert printed[body][key] == ref[body][key]
    assert printed["hand"]["aligned"]["pa_mpjpe"] == ref["hand"]["aligned"]["pa_mpjpe"]


def test_cli_intrinsics_flag_and_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pred, gt, _ = _files(tmp_path)
    assert main([str(pred), str(gt), "--intrinsics", "1200,1200,320,240"]) == 0
    a = json.loads(capsys.readouterr().out)
    assert main([str(pred), str(gt)]) == 0
    b = json.loads(capsys.readouterr().out)
    assert a["object"]["reproj_rms_px"] == pytest.approx(2.0 * b["object"]["reproj_rms_px"], rel=1e-12)
    only_obj = tmp_path / "only.npz"
    np.savez(only_obj, object=np.ones((15, 3)))
    assert main([str(only_obj), str(gt)]) == 2                      # 'hand' in one file only
    assert main([str(tmp_path / "missing.npz"), str(gt)]) == 2
    assert "error" in capsys.readouterr().err


def test_module_entry_point_runs(tmp_path: Path) -> None:
    pred, gt, ref = _files(tmp_path)
    src = str(Path(alongray.__file__).resolve().parents[1])         # the alongray under test, installed or not
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([src, os.environ.get("PYTHONPATH", "")]))
    r = subprocess.run([sys.executable, "-m", "alongray", str(pred), str(gt)], capture_output=True, text=True,
                       check=False, env=env)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["object"]["e_par"] == ref["object"]["e_par"]


def test_cli_hand_root(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pred, gt, _ = _files(tmp_path)
    p, g = np.load(pred), np.load(gt)
    ref3 = score(p["object"], g["object"], p["hand"], g["hand"], g["K"], hand_root=3).as_dict()
    assert main([str(pred), str(gt), "--hand-root", "3"]) == 0
    out3 = json.loads(capsys.readouterr().out)
    assert main([str(pred), str(gt)]) == 0
    out0 = json.loads(capsys.readouterr().out)
    assert out3["hand"]["lam"] == ref3["hand"]["lam"] != out0["hand"]["lam"]
    assert out3["hand"]["aligned"]["mpjpe_root_aligned"] == ref3["hand"]["aligned"]["mpjpe_root_aligned"]
    assert main([str(pred), str(gt), "--hand-root", "99"]) == 2


@pytest.mark.filterwarnings("ignore:a predicted reference point is exactly")
@pytest.mark.parametrize("case", ["gt_zero", "negative_lam", "pred_zero", "npy_file", "object_joints", "bad_K"])
def test_cli_degenerate_input_exits_2(tmp_path: Path, case: str, capsys: pytest.CaptureFixture[str]) -> None:
    obj = _traj(15, 1)
    pred, gt = tmp_path / "pred.npz", tmp_path / "gt.npz"
    args: list[str] = []
    if case == "gt_zero":
        np.savez(pred, object=obj)
        np.savez(gt, object=np.zeros_like(obj))
    elif case in ("negative_lam", "pred_zero"):
        np.savez(pred, object=-obj if case == "negative_lam" else np.zeros_like(obj))
        np.savez(gt, object=obj)
    elif case == "npy_file":
        pred = tmp_path / "pred.npy"
        np.save(pred, obj)
        np.savez(gt, object=obj)
    elif case == "object_joints":
        np.savez(pred, object=np.repeat(obj[:, None], 4, 1))
        np.savez(gt, object=np.repeat(obj[:, None], 4, 1))
    else:
        np.savez(pred, object=obj)
        np.savez(gt, object=obj)
        args = ["--intrinsics", "600,600,320"]
    assert main([str(pred), str(gt), *args]) == 2
    assert "alongray: error:" in capsys.readouterr().err
