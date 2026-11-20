"""Model loads under `run_manifest_batched`: once per worker (per slot, per hands batch), never per clip."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from fastdo.engine.batched_runner import BatchedEngineConfig, StageBatchConfig, run_manifest_batched
from fastdo.engine.batched_stages import (
    ConcurrentClipStage,
    FrameBatchedStage,
    HandsBatchStage,
    SolveBatchStage,
)
from fastdo.engine.manifest import ClipManifest
from fastdo.engine.orchestrator import EngineConfig
from fastdo.engine.workers import (
    make_encode_worker,
    make_geometry_worker,
    make_hands_worker,
    make_intrinsics_worker,
    make_mask_worker,
    make_shape_worker,
    make_tracks_worker,
)
from fastdo.engine.workers_shape import SAMPLES_FILE
from fastdo.solver.api import SolverConfig
from tests.engine.fakes import FakeDinov2, FakeMoGe, FakeOnlineTracker, write_geometry_scene
from tests.engine.fakes_batched import (
    FakeHandsDriver,
    FakeSolveMany,
    finish_fake_solve,
    prepare_fake_solve,
    write_clip_inputs,
)
from tests.engine.test_hands_batch import HANDS_CFG
from tests.shape.test_masks_sam2 import _FakeSam2Predictor
from tests.shape.test_shape_worker import _CubeBackend

IDS = [f"c{i}" for i in range(5)]
LOADERS = {"encoder": "fastdo.encoder.dinov2.load_dinov2", "mask": "fastdo.shape.masks_sam2.load_sam2_predictor",
           "tracks": "fastdo.tracks.cotracker._load_cotracker_model",
           "intrinsics": "fastdo.geometry.moge_model._load_model", "geometry": "fastdo.geometry.moge_model._load_model",
           "shape": "fastdo.engine.workers_shape._load_backend"}
OUTPUT = {"encoder": "encoder.npz", "geometry": "geometry.npz", "mask": "mask.npz", "tracks": "tracks.npz",
          "intrinsics": "intrinsics.npz", "shape": SAMPLES_FILE, "hands": "hands.npz", "solve": "solve.npz"}


def _run(root: Path, stage: str, impl: Any, max_batch: int, wave: int | None = None) -> None:
    manifest = ClipManifest.create(IDS, IDS, [5] * len(IDS))
    cfg = BatchedEngineConfig(EngineConfig(artifact_root=root, stages=(stage,)),
                              {stage: StageBatchConfig(max_batch=max_batch)}, max_clips_per_wave=wave, run_id="t0")
    run_manifest_batched(manifest, cfg, {stage: impl})
    assert {c: manifest.status(c, stage).error for c in IDS if manifest.status(c, stage).state != "done"} == {}
    assert all((root / c / OUTPUT[stage]).is_file() for c in IDS)


def _inputs(root: Path, stage: str) -> list[Any]:
    if stage == "geometry":
        return [write_geometry_scene(root / c, tag_offset=10 * seed) for seed, c in enumerate(IDS)]
    for seed, clip_id in enumerate(IDS):
        write_clip_inputs(root, clip_id, 5, seed=seed, with_mask=stage == "shape")
        if stage == "intrinsics":
            (root / clip_id / "intrinsics.npz").unlink()
    return []


@pytest.mark.parametrize("stage", ["encoder", "geometry", "mask", "tracks", "intrinsics", "shape", "hands", "solve"])
def test_model_loads_once_per_worker_for_n_clips(stage: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    if stage == "mask":
        pytest.importorskip("PIL")
    monkeypatch.setattr("fastdo.bench.gpu_budget.apply_gpu_budget", lambda *a, **k: None)
    root = tmp_path / "run"
    scenes = _inputs(root, stage)
    loads: list[Any] = []
    dino = FakeDinov2()
    fakes: dict[str, Any] = {"encoder": lambda: dino, "mask": lambda: _FakeSam2Predictor({}, {}),
                             "tracks": FakeOnlineTracker, "intrinsics": FakeMoGe, "shape": _CubeBackend,
                             "geometry": lambda: FakeMoGe({k: v for s in scenes for k, v in s.depth.items()})}
    if stage in LOADERS:
        def _load(*_args: Any, **_kwargs: Any) -> Any:
            loads.append(fakes[stage]())
            return loads[-1]
        monkeypatch.setattr(LOADERS[stage], _load)

    if stage == "encoder":
        _run(root, stage, FrameBatchedStage(make_encode_worker(device="cpu", crop_size=14, max_batch=4)), 3)
        assert len(loads) == 1 and dino.batch_sizes == [4, 4, 4, 3, 4, 4, 2]
    elif stage == "geometry":
        _run(root, stage, FrameBatchedStage(make_geometry_worker(device="cpu")), 3)
        assert len(loads) == 1
    elif stage in ("mask", "tracks"):
        factory = (lambda: make_mask_worker(device="cpu")) if stage == "mask" else (
            lambda: make_tracks_worker(n_queries=4, device="cpu"))
        _run(root, stage, ConcurrentClipStage(factory), 2)
        assert len(loads) == 2
        if stage == "mask":
            loads.clear()
            _inputs(tmp_path / "one", stage)
            _run(tmp_path / "one", stage, ConcurrentClipStage(factory), 1)
            assert len(loads) == 1
    elif stage == "intrinsics":
        _run(root, stage, make_intrinsics_worker(device="cpu"), 1)
        assert len(loads) == 1
    elif stage == "shape":
        _run(root, stage, make_shape_worker(sdf_resolution=8, n_samples=64), 1)
        assert len(loads) == 1
    elif stage == "hands":
        for wave, calls in ((None, 1), (2, 3)):
            driver = FakeHandsDriver()
            monkeypatch.setattr("fastdo.engine.hands_batch.run_hands_batched", driver)
            run_root = tmp_path / f"hands_{wave}"
            _inputs(run_root, stage)
            stage_impl = HandsBatchStage(HANDS_CFG, run_root / ".hands_jobs", run_root / ".gpu.lock",
                                         make_hands_worker(None))
            _run(run_root, stage, stage_impl, 8, wave=wave)
            assert len(driver.calls) == calls and sum(len(c) for c in driver.calls) == 5
    else:
        fake = FakeSolveMany()
        _run(root, stage, SolveBatchStage(prepare_fake_solve, finish_fake_solve, SolverConfig(), fake), 8)
        assert len(fake.calls) == 1 and len(fake.calls[0][0]) == 5
