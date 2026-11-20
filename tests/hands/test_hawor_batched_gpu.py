"""GPU parity, speed and invariance tests of the batched HaWoR hand stage (D3b) on the whisking demo.

Skipped unless CUDA, the hawor python, `$FASTDO_HAWOR_ROOT/weights/hawor/checkpoints/hawor.ckpt`, the whisking video
and the reference npz all exist. `paths.baseline` is taken from `$FASTDO_HAWOR_ROOT` (three levels up), so the test
runs from a worktree. The driver inherits `FASTDO_GPU_MEM_GB`; HaWoR peaked at 5.76-5.92 GB, so the GPU run sets
the variable to 7 or more:

    FASTDO_HAWOR_ROOT=/opt/fastdo/handreel/third_party/do-as-i-do/reconstruction/modules/HaWoR \\
        FASTDO_GPU_MEM_GB=12 uv run pytest tests/hands/test_hawor_batched_gpu.py -q
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
import torch

from fastdo.hands.hawor_npz import load_hawor_npz
from tests.hands.hawor_env import HAWOR_PYTHON, hawor_root

REPO = Path(__file__).resolve().parents[2]
SIDES = ("left", "right")


def _compose() -> Any:
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from hydra.initialize import get_gh_backup, restore_gh_from_backup

    root = hawor_root()
    overrides = [] if root is None else [f"paths.baseline={root.parents[2]}"]
    backup = get_gh_backup()
    try:
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=str(REPO / "run" / "conf"), version_base=None):
            return compose(config_name="config", overrides=overrides)
    finally:
        restore_gh_from_backup(backup)


def _skip_reason() -> str:
    if not torch.cuda.is_available():
        return "CUDA not available"
    if not HAWOR_PYTHON.is_file():
        return "hawor python not available"
    root = hawor_root()
    if root is None or not (root / "weights" / "hawor" / "checkpoints" / "hawor.ckpt").is_file():
        return "$FASTDO_HAWOR_ROOT/weights/hawor/checkpoints/hawor.ckpt not available"
    cfg = _compose()
    for key in ("video", "reference_npz"):
        if not Path(cfg.hands_parity[key]).is_file():
            return f"hands_parity.{key} not available"
    return ""


_REASON = _skip_reason()
pytestmark = pytest.mark.skipif(bool(_REASON), reason=_REASON or "requirements present")


def _script() -> ModuleType:
    path = REPO / "run" / "pipeline" / "hands_whisking_parity.py"
    spec = importlib.util.spec_from_file_location("hands_whisking_parity", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


@pytest.fixture(scope="module")
def cfg() -> Any:
    return _compose()


@pytest.fixture(scope="module")
def script() -> ModuleType:
    return _script()


@pytest.fixture(scope="module")
def parity(cfg: Any, script: ModuleType, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    return script.run_parity_calls(cfg, tmp_path_factory.mktemp("hands_parity"))


def _ranges(cfg: Any) -> list[list[int]]:
    return [list(r) for r in cfg.hands_parity.both_detected_ranges]


def test_driver_self_check_matches_hawor(tmp_path: Path) -> None:
    from fastdo.hands.hawor_batched import DRIVER_SCRIPT, driver_env

    response = tmp_path / "self_check.json"
    proc = subprocess.run([str(HAWOR_PYTHON), str(DRIVER_SCRIPT), "--self-check", "--response", str(response)],
                          cwd=hawor_root(), env=driver_env(os.environ), capture_output=True, text=True, timeout=1800,
                          check=False)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    result = json.loads(response.read_text())["self_check"]
    assert result["max_abs_rotmat"] <= 1e-5
    assert result["max_abs_betas"] <= 1e-4
    assert result["max_abs_trans_m"] <= 1e-5


def test_whisking_parity_vs_hawor(cfg: Any, script: ModuleType, parity: dict[str, Any]) -> None:
    metrics = script.parity_metrics(parity["single_npz"], Path(cfg.hands_parity.reference_npz), _ranges(cfg))

    assert metrics["n_eval_rows"] == 44
    assert metrics["both"]["mean_mm"] <= 10.0
    assert metrics["left"]["mean_mm"] <= 10.0 and metrics["right"]["mean_mm"] <= 10.0
    assert metrics["both"]["p95_mm"] <= 30.0
    assert metrics["left"]["root_rel_mean_mm"] <= 4.0 and metrics["right"]["root_rel_mean_mm"] <= 4.0
    for side in SIDES:
        load_hawor_npz(parity["single_npz"], side).validate(46, f"{side}_hand")


@pytest.mark.timing
def test_whisking_46_frames_within_3s(parity: dict[str, Any]) -> None:
    report = parity["single_report"]
    assert report.compute_seconds <= 3.0
    assert report.clips[0].seconds["total"] <= 3.0


def test_multi_clip_call_matches_single(parity: dict[str, Any]) -> None:
    single = _load(parity["single_npz"])
    for path in parity["copies_npz"]:
        copy = _load(path)
        for side in SIDES:
            np.testing.assert_array_equal(copy[f"{side}_box"], single[f"{side}_box"])
            np.testing.assert_array_equal(copy[f"{side}_valid"], single[f"{side}_valid"])
            np.testing.assert_array_equal(copy[f"{side}_infilled"], single[f"{side}_infilled"])
            rows = single[f"{side}_valid"]
            diff = copy[f"{side}_joints"][rows].astype(np.float64) - single[f"{side}_joints"][rows]
            assert float(np.abs(diff).max()) * 1000.0 <= 0.01
    assert parity["copies_report"].peak_cuda_gb <= 7.0


def test_gap_fill_on_whisking_gap(cfg: Any, script: ModuleType, parity: dict[str, Any]) -> None:
    single = _load(parity["single_npz"])

    assert single["frame_idx"][single["left_infilled"]].tolist() == [123, 126]
    assert single["left_valid"].all() and single["right_valid"].all()
    rows = script.gap_rows(parity["single_npz"], Path(cfg.hands_parity.reference_npz), [123, 126])
    assert rows["mean_mm"] <= 40.0
    assert all(np.isfinite(v).all() for v in single.values() if v.dtype.kind == "f")


def test_sampled_temporal_ablation_is_worse(cfg: Any, script: ModuleType, parity: dict[str, Any]) -> None:
    reference = Path(cfg.hands_parity.reference_npz)
    single = script.parity_metrics(parity["single_npz"], reference, _ranges(cfg))
    sampled = script.parity_metrics(parity["sampled_npz"], reference, _ranges(cfg))
    assert sampled["both"]["mean_mm"] > single["both"]["mean_mm"] + 5.0


def test_hawor_faces_match_mano_topology(cfg: Any, parity: dict[str, Any]) -> None:
    ours, reference = _load(parity["single_npz"]), _load(Path(cfg.hands_parity.reference_npz))
    right = ours["right_faces"]

    np.testing.assert_array_equal(right[:1538], reference["right_faces"][:1538])
    assert set(right[1538:].ravel().tolist()) == set(reference["right_faces"][1538:].ravel().tolist())
    edges = [(int(f[i]), int(f[(i + 1) % 3])) for f in right for i in range(3)]
    assert len(edges) == len(set(edges)) and all((j, i) in set(edges) for i, j in edges)
    np.testing.assert_array_equal(ours["left_faces"], right[:, [0, 2, 1]])


def test_engine_hands_stage_matches_parity_call(cfg: Any, parity: dict[str, Any], tmp_path: Path) -> None:
    from fastdo.engine.hands_batch import run_hands_stage
    from fastdo.engine.manifest import ClipManifest
    from fastdo.engine.orchestrator import EngineConfig
    from fastdo.engine.workers import make_decode_worker, make_intrinsics_worker
    from fastdo.hands.hawor_batched import hands_config_from_cfg

    video = Path(cfg.hands_parity.video)
    root = tmp_path / "artifacts"
    clip_dir = root / "whisking"
    manifest = ClipManifest.create(["whisking"], [str(video)], [46])
    make_decode_worker({"whisking": video}, stride=3, backend="ffmpeg")("whisking", clip_dir)
    manifest.mark("whisking", "decode", "done", 0.0)
    (clip_dir / "camera.json").write_text(json.dumps({"fx": 999.0292358398438, "fy": 999.0292167663574,
                                                      "cx": 640.0, "cy": 360.0}))
    make_intrinsics_worker()("whisking", clip_dir)                   # the given calibration: MoGe is never loaded
    manifest.mark("whisking", "intrinsics", "done", 0.0)

    run_hands_stage(manifest, EngineConfig(artifact_root=root), hands_config_from_cfg(cfg), tmp_path / "jobs")

    assert manifest.status("whisking", "hands").state == "done"
    hands, single = _load(clip_dir / "hands.npz"), _load(parity["single_npz"])
    np.testing.assert_array_equal(hands["frame_idx"], single["frame_idx"])
    for side in SIDES:
        np.testing.assert_array_equal(hands[f"{side}_valid"], single[f"{side}_valid"])
        rows = single[f"{side}_valid"]
        diff = hands[f"{side}_joints"][rows].astype(np.float64) - single[f"{side}_joints"][rows]
        assert float(np.abs(diff).max()) * 1000.0 <= 0.01
