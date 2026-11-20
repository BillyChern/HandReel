"""Worker lifecycle: every stage worker applies the per-process CUDA cap before its first allocation, a
factory allocates and loads nothing, and a model loads once per worker, not once per clip."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import Tensor

from fastdo.engine import workers_frames
from fastdo.engine.orchestrator import StageFn
from fastdo.engine.worker_support import WorkerContext
from fastdo.engine.workers import (
    make_decode_worker,
    make_encode_worker,
    make_geometry_worker,
    make_hands_worker,
    make_intrinsics_worker,
    make_mask_worker,
    make_tracks_worker,
    write_mask_prompt,
)
from tests.engine.fakes import (
    FakeDinov2,
    FakeMoGe,
    FakeOnlineTracker,
    make_hawor_npz,
    random_frames,
    write_decode,
    write_geometry_scene,
)
from tests.shape.test_masks_sam2 import _FakeSam2Predictor

_MODEL_LOADERS = {
    "encoder": "fastdo.encoder.dinov2.load_dinov2",
    "mask": "fastdo.shape.masks_sam2.load_sam2_predictor",
    "tracks": "fastdo.tracks.cotracker._load_cotracker_model",
    "intrinsics": "fastdo.geometry.moge_model._load_model",
    "geometry": "fastdo.geometry.moge_model._load_model",
}
_OUTPUT = {"encoder": "encoder.npz", "mask": "mask.npz", "tracks": "tracks.npz", "intrinsics": "intrinsics.npz",
           "geometry": "geometry.npz"}
_FACTORIES: dict[str, Callable[[], StageFn]] = {
    "encoder": lambda: make_encode_worker(device="cpu", crop_size=14),
    "mask": lambda: make_mask_worker(device="cpu"),
    "tracks": lambda: make_tracks_worker(n_queries=4, device="cpu"),
    "intrinsics": lambda: make_intrinsics_worker(device="cpu"),
    "geometry": lambda: make_geometry_worker(device="cpu"),
}


def _write_clip(stage: str, artifact_dir: Path, seed: int) -> Any:
    """Writes one clip's inputs for `stage` and returns the fake model that clip needs."""
    if stage == "geometry":
        return write_geometry_scene(artifact_dir, tag_offset=10 * seed)
    write_decode(artifact_dir, random_frames(4, h=8, w=8, seed=seed))
    prompt = np.zeros((8, 8), bool)
    prompt[2:6, 2:6] = True
    write_mask_prompt(artifact_dir, 1, prompt)
    fakes: dict[str, Any] = {"encoder": FakeDinov2(), "mask": _FakeSam2Predictor({}, {}),
                             "tracks": FakeOnlineTracker(), "intrinsics": FakeMoGe()}
    return fakes[stage]


@pytest.mark.parametrize("stage", sorted(_MODEL_LOADERS))
def test_model_workers_cap_gpu_memory_first_and_load_their_model_once(
        stage: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("PIL")                                      # the mask stage stages JPEGs for SAM 2
    dirs = [tmp_path / "clip0", tmp_path / "clip1"]
    fakes = [_write_clip(stage, d, seed) for seed, d in enumerate(dirs)]
    fake = FakeMoGe(fakes[0].depth | fakes[1].depth) if stage == "geometry" else fakes[0]
    events: list[str] = []

    def _load(*_args: Any, **_kwargs: Any) -> Any:
        events.append("load")
        return fake

    monkeypatch.setattr("fastdo.bench.gpu_budget.apply_gpu_budget", lambda *a, **k: events.append("budget"))
    monkeypatch.setattr(_MODEL_LOADERS[stage], _load)

    worker = _FACTORIES[stage]()
    assert events == []                                             # factories allocate and load nothing
    worker("clip0", dirs[0])
    worker("clip1", dirs[1])

    assert events == ["budget", "load"]                             # cap first, one load across both clips
    assert all((d / _OUTPUT[stage]).is_file() for d in dirs)


@pytest.mark.parametrize("stage", ["decode", "hands"])
def test_io_workers_cap_gpu_memory_before_their_first_allocation(
        stage: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    def _record(*_args: Any, **_kwargs: Any) -> None:
        events.append("read")

    def _fake_decode(*_args: Any, **_kwargs: Any) -> tuple[Tensor, list[int]]:
        events.append("read")
        return torch.zeros(2, 8, 8, 3, dtype=torch.uint8), [0, 3]

    monkeypatch.setattr("fastdo.bench.gpu_budget.apply_gpu_budget", lambda *a, **k: events.append("budget"))
    monkeypatch.setattr(workers_frames, "decode_frames", _fake_decode)
    monkeypatch.setattr(workers_frames, "probe_fps", lambda _path: 30.0)
    monkeypatch.setattr("fastdo.engine.workers_hands.write_hands_artifact", _record)
    dirs = [tmp_path / "clip0", tmp_path / "clip1"]
    for d in dirs:
        write_decode(d, random_frames(2, h=8, w=8))
        make_hawor_npz(d / "all_hand_meshes.npz", t=2)
    worker = make_decode_worker({d.name: tmp_path / "x.mp4" for d in dirs}) if stage == "decode" \
        else make_hands_worker()

    worker("clip0", dirs[0])
    worker("clip1", dirs[1])

    assert events == ["budget", "read", "read"]


def test_model_loads_are_serialized_across_worker_contexts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Concurrent slots of `run_clips_concurrently` load in pool threads; their loaders (sam2's Hydra swap,
    torch.hub's sys.path edits) touch process-wide state, so two contexts never load at the same time."""
    n_contexts = 3
    inside, max_inside = [0], [0]
    guard = threading.Lock()
    barrier = threading.Barrier(n_contexts, timeout=30)
    monkeypatch.setattr("fastdo.bench.gpu_budget.apply_gpu_budget", lambda *a, **k: barrier.wait())

    def _loader(tag: int) -> Callable[[], int]:
        def load() -> int:
            with guard:
                inside[0] += 1
                max_inside[0] = max(max_inside[0], inside[0])
            time.sleep(0.05)
            with guard:
                inside[0] -= 1
            return tag
        return load

    contexts = [WorkerContext(loader=_loader(tag)) for tag in range(n_contexts)]
    models: dict[int, Any] = {}

    def _use(tag: int) -> None:
        models[tag] = contexts[tag].model()                         # passes the barrier together, then loads

    threads = [threading.Thread(target=_use, args=(tag,)) for tag in range(n_contexts)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    assert models == {tag: tag for tag in range(n_contexts)}
    assert max_inside[0] == 1
