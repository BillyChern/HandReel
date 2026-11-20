"""Tests for the D3 frame and video stage workers: decode, encoder, mask, tracks (fakes only; no weights, no GPU)."""

from __future__ import annotations

import shutil
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from fastdo.encoder.dinov2 import encode_crops
from fastdo.engine.batching import run_clips_concurrently
from fastdo.engine.workers import (
    center_crops,
    make_decode_worker,
    make_encode_worker,
    make_mask_worker,
    make_tracks_worker,
    write_mask_prompt,
)
from fastdo.shape.masks_sam2 import propagate_masks_sam2
from fastdo.tracks.cotracker import sample_query_points_in_mask, track_points_cotracker
from tests.engine.fakes import FakeDinov2, FakeOnlineTracker, random_frames, write_decode
from tests.shape.test_masks_sam2 import _FakeSam2Predictor

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available")


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


# --- decode -------------------------------------------------------------------------------------------


@needs_ffmpeg
def test_decode_worker_pinned_keys_uncompressed(tmp_path: Path) -> None:
    video = tmp_path / "testsrc.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=30", "-frames:v", "12",
                    "-pix_fmt", "yuv420p", str(video)], check=True)
    clip = tmp_path / "clip0"

    make_decode_worker({"clip0": video}, stride=3)("clip0", clip)

    path = clip / "decode.npz"
    with zipfile.ZipFile(path) as archive:
        assert all(info.compress_type == zipfile.ZIP_STORED for info in archive.infolist())
    data = _load(path)
    assert set(data) == {"frames", "frame_idx", "fps", "stride"}
    assert data["stride"].shape == () and data["stride"].dtype == np.int64 and int(data["stride"]) == 3
    assert data["frames"].shape == (4, 48, 64, 3) and data["frames"].dtype == np.uint8
    assert data["frame_idx"].tolist() == [0, 3, 6, 9] and data["frame_idx"].dtype == np.int64
    assert data["fps"].shape == () and data["fps"].dtype == np.float64
    assert float(data["fps"]) == pytest.approx(30.0 / 3, abs=1e-9)


def test_decode_worker_decodes_only_a_window(tmp_path: Path) -> None:
    """S2's P4: one source file yields many 10 s windows, each decoding only its own range."""
    video = tmp_path / "testsrc.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=30", "-frames:v", "60",
                    "-g", "10", "-pix_fmt", "yuv420p", str(video)], check=True)
    whole, win = tmp_path / "whole", tmp_path / "win"

    make_decode_worker({"whole": video}, stride=3)("whole", whole)
    make_decode_worker({"win": (video, 30, 45)}, stride=3)("win", win)

    a, b = _load(whole / "decode.npz"), _load(win / "decode.npz")
    assert b["frame_idx"].tolist() == [30, 33, 36, 39, 42]          # source numbering, not 0-based
    assert float(b["fps"]) == pytest.approx(float(a["fps"]), abs=1e-9)
    rows = [a["frame_idx"].tolist().index(i) for i in b["frame_idx"].tolist()]
    assert np.array_equal(b["frames"], a["frames"][rows])           # identical pixels to the full decode


def test_decode_worker_rejects_a_window_outside_the_video(tmp_path: Path) -> None:
    video = tmp_path / "testsrc.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=30", "-frames:v", "12",
                    "-pix_fmt", "yuv420p", str(video)], check=True)
    with pytest.raises(ValueError, match="range"):
        make_decode_worker({"clip0": (video, 5, 99)})("clip0", tmp_path / "c")


def test_decode_worker_rejects_an_unregistered_clip(tmp_path: Path) -> None:
    with pytest.raises(KeyError, match="clip9"):
        make_decode_worker({"clip0": tmp_path / "a.mp4"})("clip9", tmp_path)


# --- encoder ------------------------------------------------------------------------------------------


def test_encode_worker_writes_expected_shape(tmp_path: Path) -> None:
    frames = random_frames(5, h=24, w=32)
    write_decode(tmp_path, frames)

    make_encode_worker(device="cpu", crop_size=14, model=FakeDinov2())("clip0", tmp_path)

    data = _load(tmp_path / "encoder.npz")
    assert set(data) == {"tokens", "frame_idx", "crop"}
    assert str(data["crop"]) == "center"
    assert data["tokens"].shape == (5, 2 * FakeDinov2.hidden) and data["tokens"].dtype == np.float32
    assert data["frame_idx"].tolist() == [0, 3, 6, 9, 12]
    expected = encode_crops(FakeDinov2(), center_crops(torch.from_numpy(frames), 14))
    np.testing.assert_allclose(data["tokens"], expected.numpy(), rtol=0, atol=1e-6)


def test_center_crops_takes_the_centred_square() -> None:
    frames = torch.zeros(2, 4, 8, 3, dtype=torch.uint8)
    frames[:, :, 2:6] = 200                                          # the centred 4x4 square of an 8-wide frame
    crops = center_crops(frames, 4)
    assert crops.shape == (2, 3, 4, 4) and crops.dtype == torch.uint8
    assert bool((crops == 200).all())


def test_encode_worker_concatenates_clips_into_budgeted_forward_passes(tmp_path: Path) -> None:
    counts = {"a": 3, "b": 2, "c": 4}
    all_frames = {name: random_frames(n, seed=i) for i, (name, n) in enumerate(counts.items())}
    for name, frames in all_frames.items():
        write_decode(tmp_path / name, frames)
    model = FakeDinov2()
    worker = make_encode_worker(device="cpu", crop_size=14, max_batch=4, gpu_budget_gb=1.0, crop_cost_gb=0.3,
                                model=model)

    worker.run_many([(name, tmp_path / name) for name in counts])

    # 1.0 GiB / 0.3 GiB per crop -> 3 crops per forward pass, crossing clip boundaries (a | a a b | b c c | c c)
    assert model.batch_sizes == [3, 3, 3]
    for name, frames in all_frames.items():
        tokens = _load(tmp_path / name / "encoder.npz")["tokens"]
        expected = encode_crops(FakeDinov2(), center_crops(torch.from_numpy(frames), 14))
        np.testing.assert_allclose(tokens, expected.numpy(), rtol=0, atol=1e-6)


# --- mask ---------------------------------------------------------------------------------------------


def test_mask_worker_matches_masks_sam2_contract(tmp_path: Path) -> None:
    pytest.importorskip("PIL")                                       # SAM 2 is staged through JPEG frames
    n, h, w, keyframe_index = 6, 4, 4, 3
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 256, size=(n, h, w, 3), dtype=np.uint8)
    keyframe_mask = rng.random((h, w)) > 0.5
    forward = {4: rng.random((h, w)) > 0.5, 5: rng.random((h, w)) > 0.5}
    backward = {2: rng.random((h, w)) > 0.5, 1: rng.random((h, w)) > 0.5, 0: rng.random((h, w)) > 0.5}
    write_decode(tmp_path, frames)
    write_mask_prompt(tmp_path, keyframe_index, keyframe_mask)

    make_mask_worker(device="cpu", predictor=_FakeSam2Predictor(forward, backward))("clip0", tmp_path)

    expected = propagate_masks_sam2(torch.from_numpy(frames), keyframe_index, torch.from_numpy(keyframe_mask),
                                    device="cpu", predictor=_FakeSam2Predictor(forward, backward))
    data = _load(tmp_path / "mask.npz")
    assert set(data) == {"masks", "frame_idx", "keyframe_index"}
    assert data["masks"].dtype == np.bool_
    np.testing.assert_array_equal(data["masks"], expected.numpy())
    assert data["frame_idx"].tolist() == [0, 3, 6, 9, 12, 15] and int(data["keyframe_index"]) == keyframe_index


def test_mask_worker_loads_the_requested_variant_and_rejects_unknown_ones(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def _load_predictor(device: str = "cuda", model_id: str = "") -> Any:
        seen.append(model_id)
        return _FakeSam2Predictor({}, {})

    monkeypatch.setattr("fastdo.shape.masks_sam2.load_sam2_predictor", _load_predictor)
    with pytest.raises(ValueError, match="variant"):
        make_mask_worker(variant="huge")                             # fails at construction, before any load
    assert seen == []


def test_mask_worker_runs_clip_batched_through_run_clips_concurrently(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("PIL")
    predictors: list[_FakeSam2Predictor] = []
    model_ids: list[str] = []

    def _load_predictor(device: str = "cuda", model_id: str = "") -> Any:
        predictors.append(_FakeSam2Predictor({}, {}))
        model_ids.append(model_id)
        return predictors[-1]

    monkeypatch.setattr("fastdo.shape.masks_sam2.load_sam2_predictor", _load_predictor)
    prompt = np.zeros((8, 8), bool)
    prompt[2:6, 2:6] = True
    clips = []
    for i in range(3):
        write_decode(tmp_path / f"c{i}", random_frames(4, h=8, w=8, seed=i))
        write_mask_prompt(tmp_path / f"c{i}", 0, prompt)
        clips.append((f"c{i}", tmp_path / f"c{i}"))

    runs = run_clips_concurrently(lambda: make_mask_worker(variant="tiny", device="cpu"), clips,
                                  max_concurrent=2, gpu_budget_gb=12.0, cost_fn=lambda _: 6.0)

    assert [run.error for run in runs] == ["", "", ""]
    assert len(predictors) == 2 and model_ids == ["facebook/sam2.1-hiera-tiny"] * 2   # one per slot
    assert sum(len(p.init_state_calls) for p in predictors) == 3
    assert all((tmp_path / c / "mask.npz").is_file() for c, _ in clips)


def test_concurrent_mask_slots_load_sam2_without_racing_on_hydra(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two `run_clips_concurrently` slots start together (a barrier in the CUDA cap) and each loads SAM 2 through
    the real `load_sam2_predictor`, whose patched build composes sam2's config: no clip fails on GlobalHydra."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from hydra.initialize import get_gh_backup, restore_gh_from_backup

    session_backup = get_gh_backup()                                 # before `import sam2` initializes Hydra
    pytest.importorskip("sam2")
    pytest.importorskip("PIL")
    from sam2 import build_sam

    inside, max_inside, composed = [0], [0], list[str]()
    guard = threading.Lock()
    barrier = threading.Barrier(2, timeout=30)

    def _fake_build(model_id: str, device: str = "cuda") -> _FakeSam2Predictor:
        with guard:
            inside[0] += 1
            max_inside[0] = max(max_inside[0], inside[0])
        time.sleep(0.05)
        composed.append(str(compose(config_name="configs/sam2.1/sam2.1_hiera_t.yaml").model._target_))
        time.sleep(0.05)
        with guard:
            inside[0] -= 1
        return _FakeSam2Predictor({}, {})

    monkeypatch.setattr(build_sam, "build_sam2_video_predictor_hf", _fake_build)
    monkeypatch.setattr("fastdo.bench.gpu_budget.apply_gpu_budget", lambda *a, **k: barrier.wait())
    prompt = np.zeros((8, 8), bool)
    prompt[2:6, 2:6] = True
    clips = []
    for i in range(2):
        write_decode(tmp_path / f"c{i}", random_frames(4, h=8, w=8, seed=i))
        write_mask_prompt(tmp_path / f"c{i}", 0, prompt)
        clips.append((f"c{i}", tmp_path / f"c{i}"))
    (tmp_path / "app.yaml").write_text("app_key: 1\n")
    try:
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=str(tmp_path), version_base=None):
            runs = run_clips_concurrently(lambda: make_mask_worker(variant="tiny", device="cpu"), clips,
                                          max_concurrent=2, gpu_budget_gb=12.0, cost_fn=lambda _: 6.0)
            assert compose(config_name="app").app_key == 1           # the app's Hydra state is back
    finally:
        restore_gh_from_backup(session_backup)

    assert [run.error for run in runs] == ["", ""]
    assert max_inside[0] == 1 and len(composed) == 2
    assert all((tmp_path / c / "mask.npz").is_file() for c, _ in clips)


def test_mask_and_tracks_workers_require_a_mask_prompt(tmp_path: Path) -> None:
    write_decode(tmp_path, random_frames(3))
    with pytest.raises(FileNotFoundError, match="mask_prompt"):
        make_mask_worker(device="cpu", predictor=_FakeSam2Predictor({}, {}))("clip0", tmp_path)
    with pytest.raises(FileNotFoundError, match="mask_prompt"):
        make_tracks_worker(device="cpu", model=FakeOnlineTracker())("clip0", tmp_path)
    with pytest.raises(ValueError, match="empty"):
        write_mask_prompt(tmp_path, 0, np.zeros((4, 4), bool))


# --- tracks -------------------------------------------------------------------------------------------


def test_tracks_worker_tracks_queries_sampled_in_the_prompt_mask(tmp_path: Path) -> None:
    frames = random_frames(5, h=8, w=8)
    write_decode(tmp_path, frames)
    prompt = np.zeros((8, 8), bool)
    prompt[2:6, 3:7] = True
    write_mask_prompt(tmp_path, 2, prompt)

    make_tracks_worker(n_queries=6, seed=0, device="cpu", model=FakeOnlineTracker())("clip0", tmp_path)

    query_uv = sample_query_points_in_mask(torch.from_numpy(prompt), 6, generator=torch.Generator().manual_seed(0))
    tracks, vis = track_points_cotracker(torch.from_numpy(frames), 2, query_uv, device="cpu",
                                         model=FakeOnlineTracker())
    data = _load(tmp_path / "tracks.npz")
    assert set(data) == {"tracks", "vis", "query_uv", "frame_idx", "keyframe_index"}
    assert data["tracks"].shape == (5, 6, 2) and data["tracks"].dtype == np.float32
    assert data["vis"].shape == (5, 6) and data["vis"].dtype == np.bool_
    np.testing.assert_array_equal(data["query_uv"], query_uv.numpy())
    np.testing.assert_array_equal(data["tracks"], tracks.numpy())
    np.testing.assert_array_equal(data["vis"], vis.numpy())
    assert data["frame_idx"].tolist() == [0, 3, 6, 9, 12] and int(data["keyframe_index"]) == 2


def test_decode_worker_passes_local_probe_through(tmp_path, monkeypatch) -> None:
    """The flag must REACH `decode_window`, or the whole O(window) path is unreachable from the engine.

    A default-off flag that no caller can set is the failure this project has paid for repeatedly: the
    code exists, the tests pass, and nothing in production can ever turn it on. So assert the argument
    arrives, for both values, rather than that the worker merely runs.
    """
    import fastdo.engine.workers_frames as wf

    seen: list[bool] = []

    def fake_decode_window(source, start, stop, stride=1, *, local_probe=False, return_times=False,
                           packet_table=False, scale_height=None):
        seen.append(local_probe)
        assert scale_height is None, "this fake is the unscaled path"
        idx = [start, start + stride]
        frames = torch.zeros((2, 4, 4, 3), dtype=torch.uint8)
        times = [i / 30.0 for i in idx]
        return (frames, idx, times) if return_times else (frames, idx)

    monkeypatch.setattr(wf, "decode_window", fake_decode_window)
    monkeypatch.setattr(wf, "probe_fps", lambda _p: 30.0)

    src = {"c": (tmp_path / "v.mp4", 10, 20)}
    for flag in (False, True):
        wf.make_decode_worker(src, stride=1, local_probe=flag)("c", tmp_path)
    assert seen == [False, True]


def test_timebase_mismatch_flags_a_one_frame_shift_and_tolerates_a_start_offset() -> None:
    """The timebase check must fire on a one-frame decode shift and stay quiet on a correct decode.

    Worked from the real numbers: four of eight Ego4D videos start at 21.029 ms, not 0. At 30 fps the
    bound is 1/fps = 33.3 ms, so a CORRECT decode of those files sits 21.0 ms off its
    nominal `n/fps` and passes, while a decode shifted by one frame sits 21.0 + 33.3 = 54.3 ms off and is
    flagged. That margin is the whole value of the check, and a test that only exercised a zero-offset
    file would never touch it.
    """
    from fastdo.engine.workers_frames import timebase_mismatch

    fps, t0 = 30.0, 0.021029
    idx = [37, 40, 43]

    correct = [t0 + i / fps for i in idx]
    assert not timebase_mismatch(idx, correct, fps)

    shifted = [t0 + (i + 1) / fps for i in idx]
    assert timebase_mismatch(idx, shifted, fps)

    # With the offset supplied, the correct decode is exact and the shift is still caught.
    assert not timebase_mismatch(idx, correct, fps, start_time=t0)
    assert timebase_mismatch(idx, shifted, fps, start_time=t0)


def test_timebase_mismatch_refuses_a_nonpositive_rate() -> None:
    """A row cannot be placed in time without a rate; 0 would silently divide-by-zero into inf."""
    from fastdo.engine.workers_frames import timebase_mismatch

    with pytest.raises(ValueError, match="fps must be positive"):
        timebase_mismatch([0], [0.0], 0.0)


def test_decode_worker_records_pts_and_the_timebase_flag(tmp_path, monkeypatch) -> None:
    """The timebase check requires the pts ON DISK; deriving them later cannot audit the decode that wrote them."""
    import numpy as np

    import fastdo.engine.workers_frames as wf

    def fake_decode_window(source, start, stop, stride=1, *, local_probe=False, return_times=False,
                           packet_table=False, scale_height=None):
        idx = [start, start + stride]
        frames = torch.zeros((2, 4, 4, 3), dtype=torch.uint8)
        times = [0.021029 + i / 30.0 for i in idx]
        return (frames, idx, times) if return_times else (frames, idx)

    monkeypatch.setattr(wf, "decode_window", fake_decode_window)
    monkeypatch.setattr(wf, "probe_fps", lambda _p: 30.0)

    wf.make_decode_worker({"c": (tmp_path / "v.mp4", 10, 20)}, stride=1)("c", tmp_path)

    with np.load(tmp_path / wf.DECODE_FILE) as data:
        assert "pts" in data.files, "the decoder must record each row's best-effort pts"
        assert data["pts"].shape == (2,)
        assert int(data["timebase_mismatch"]) == 0


def test_timebase_flags_separates_unrecorded_from_clean(tmp_path) -> None:
    """A decode that never measured its pts has not PASSED the check; it did not take it.

    Folding those into `clean` would let a corpus decoded before the flag existed -- or by the whole-file
    path, where the check is undefined -- report a clean bill of health it never earned.
    """
    import numpy as np

    from fastdo.engine.workers_frames import DECODE_FILE, timebase_flags

    def write(clip_id: str, **extra) -> None:
        d = tmp_path / clip_id
        d.mkdir()
        np.savez(d / DECODE_FILE, frame_idx=np.array([0, 1]), fps=np.array(30.0), **extra)

    write("clean", timebase_mismatch=np.array(0))
    write("flagged", timebase_mismatch=np.array(1))
    write("old")                                     # decoded before the flag existed

    got = timebase_flags(tmp_path, ["clean", "flagged", "old", "missing"])
    assert got == {"flagged": 1, "clean": 1, "unrecorded": 2}


@needs_ffmpeg
def test_the_flag_written_by_a_real_decode_reaches_the_labels(tmp_path: Path) -> None:
    """The reachability check, on a real file rather than a constructed `.npz`.

    The timebase check spans two modules: the decode worker records the pts and sets
    the flag, and `workers_qc.decode_timestamps` labels a flagged window from them. Each half has unit
    tests over hand-built archives, which is exactly the shape a clause can have while never joining up
    in production (the pattern this project has paid for three times). So this runs the REAL windowed
    decode worker and hands its output to the REAL label function.

    A CFR `testsrc` is a clean window: the flag must be 0, and the measured pts must agree with the
    arithmetic to the last bit. That is the assertion -- if a clean file disagreed, the flag would be
    firing on well-formed input and every S2 window would silently switch to pts labelling.
    """
    from fastdo.engine.workers_qc import decode_timestamps

    source = tmp_path / "v.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
                    "-i", "testsrc=size=64x48:rate=30:duration=4", "-pix_fmt", "yuv420p", str(source)],
                   check=True)
    artifact_dir = tmp_path / "clip"
    artifact_dir.mkdir()
    make_decode_worker({"clip": (source, 30, 60)}, stride=3)("clip", artifact_dir)

    with np.load(artifact_dir / "decode.npz") as data:
        assert {"pts", "timebase_mismatch"} <= set(data.files)         # the window path records both
        assert int(data["timebase_mismatch"]) == 0                     # a CFR file is clean
        recorded = np.asarray(data["pts"], dtype=np.float64)

    frame_idx, timestamps = decode_timestamps(artifact_dir)
    assert frame_idx.tolist() == list(range(30, 60, 3))
    assert np.allclose(timestamps, frame_idx / 30.0, rtol=0.0, atol=1e-9)
    assert np.allclose(recorded, timestamps, rtol=0.0, atol=1e-9)      # measured and derived agree


def test_the_decode_window_fakes_accept_the_real_signature() -> None:
    """Twice now, adding a keyword to `decode_window` has broken these fakes at runtime rather than here.

    `return_times` did it first; `scale_height` did it again, and both times the break surfaced as a
    TypeError in an unrelated-looking test on the full suite. A fake that takes `**kwargs` would not fail
    -- it would silently swallow the new argument and keep passing while testing nothing. So this
    compares the fakes against the REAL signature directly: every keyword-only parameter of
    `decode_window` must be one each fake declares by name.
    """
    import inspect

    from fastdo.io.video import decode_window as real

    wanted = {name for name, p in inspect.signature(real).parameters.items()
              if p.kind is inspect.Parameter.KEYWORD_ONLY}
    assert wanted, "decode_window has no keyword-only parameters; this guard is checking nothing"

    module = sys.modules[__name__]
    fakes = [obj for name, obj in vars(module).items() if name.startswith("test_")]
    declared_anywhere = set()
    for test_fn in fakes:
        for const in test_fn.__code__.co_consts:
            if getattr(const, "co_name", None) == "fake_decode_window":
                params = set(const.co_varnames[:const.co_argcount + const.co_kwonlyargcount])
                assert wanted <= params, (
                    f"fake_decode_window in {test_fn.__name__} is missing {sorted(wanted - params)}")
                declared_anywhere |= params
    assert declared_anywhere, "no fake_decode_window was found; this guard has gone stale"
