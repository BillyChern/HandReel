"""Tests for `fastdo.engine.hands_batch.run_hands_stage` with a monkeypatched `run_hands_batched` (no driver, no GPU)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from fastdo.engine.hands_batch import HANDS_OUTPUT, run_hands_stage
from fastdo.engine.manifest import ClipManifest
from fastdo.engine.orchestrator import EngineConfig
from fastdo.geometry.intrinsics import camera_matrix, write_intrinsics
from fastdo.hands.hawor_batched import HandsCallReport, HandsClipInput, HandsClipResult, HandsConfig
from tests.engine.fakes import make_hawor_npz, random_frames, write_decode
from tests.hands.fake_hawor_driver import clip_result, write_response, write_valid_hand_npz

HANDS_CFG = HandsConfig(hawor_python=Path("/fake/python"), hawor_root=Path("/fake/HaWoR"))


def _setup(tmp_path: Path, ids: Sequence[str], fx: float = 500.0) -> tuple[ClipManifest, EngineConfig]:
    manifest = ClipManifest.create(list(ids), [f"{c}.mp4" for c in ids], [4] * len(ids))
    root = tmp_path / "artifacts"
    for clip_id in ids:
        write_decode(root / clip_id, random_frames(4, h=8, w=8))
        write_intrinsics(root / clip_id, camera_matrix(fx, fx, 4.0, 4.0), "given")
        manifest.mark(clip_id, "decode", "done", 0.1)
        manifest.mark(clip_id, "intrinsics", "done", 0.1)
    return manifest, EngineConfig(artifact_root=root)


class _FakeBatched:
    """Writes a valid npz for every clip not in `fail` and reports it done with total 1.0 s."""

    def __init__(self, fail: Sequence[str] = ()) -> None:
        self.fail = set(fail)
        self.calls: list[list[HandsClipInput]] = []
        self.job_dirs: list[Path] = []

    def __call__(self, clips: Sequence[HandsClipInput], cfg: HandsConfig, job_dir: Path) -> HandsCallReport:
        self.calls.append(list(clips))
        self.job_dirs.append(Path(job_dir))
        results = []
        for clip in clips:
            if clip.clip_id in self.fail:
                results.append(HandsClipResult(clip.clip_id, "failed", f"boom {clip.clip_id}", 0, {}))
                continue
            with np.load(clip.frames_path, allow_pickle=False) as data:
                frame_idx = data["frame_idx"]
            write_valid_hand_npz(clip.out_npz, frame_idx)
            results.append(HandsClipResult(clip.clip_id, "done", "", len(frame_idx), {"total": 1.0}))
        return HandsCallReport(1 if self.fail else 0, 6.0, 1.5, 2.0, 5.0, tuple(results))


def test_run_hands_stage_one_call_and_marks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ids = [f"c{i}" for i in range(5)]
    manifest, engine_cfg = _setup(tmp_path, ids)
    fake = _FakeBatched(fail=["c3"])
    monkeypatch.setattr("fastdo.engine.hands_batch.run_hands_batched", fake)

    report = run_hands_stage(manifest, engine_cfg, HANDS_CFG, tmp_path / "jobs")

    assert report is not None and len(fake.calls) == 1 and [c.clip_id for c in fake.calls[0]] == ids
    assert fake.job_dirs == [tmp_path / "jobs" / "call_0000"]
    for clip_id in ("c0", "c1", "c2", "c4"):
        status = manifest.status(clip_id, "hands")
        assert status.state == "done" and status.seconds == pytest.approx(1.0 + 7.5 / 5, abs=1e-9)
        assert (engine_cfg.artifact_root / clip_id / "hands.npz").is_file()
    failed = manifest.status("c3", "hands")
    assert failed.state == "failed" and "boom c3" in failed.error
    assert not (engine_cfg.artifact_root / "c3" / "hands.npz").exists()


def test_run_hands_stage_respects_max_clips_per_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest, engine_cfg = _setup(tmp_path, [f"c{i}" for i in range(5)])
    fake = _FakeBatched()
    monkeypatch.setattr("fastdo.engine.hands_batch.run_hands_batched", fake)
    engine_cfg = EngineConfig(artifact_root=engine_cfg.artifact_root, manifest_path=tmp_path / "manifest.parquet")

    from dataclasses import replace

    run_hands_stage(manifest, engine_cfg, replace(HANDS_CFG, max_clips_per_call=2), tmp_path / "jobs")

    assert [len(call) for call in fake.calls] == [2, 2, 1]
    assert [d.name for d in fake.job_dirs] == ["call_0000", "call_0001", "call_0002"]
    assert ClipManifest.load(tmp_path / "manifest.parquet").status("c4", "hands").state == "done"


def test_run_hands_stage_skips_ineligible_clips(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest, engine_cfg = _setup(tmp_path, ["ok", "precomputed", "no_intrinsics", "no_decode", "already"])
    make_hawor_npz(engine_cfg.artifact_root / "precomputed" / "all_hand_meshes.npz", t=4)
    manifest.mark("no_intrinsics", "intrinsics", "failed", 0.0, error="x")
    manifest.mark("no_decode", "decode", "pending", 0.0)
    manifest.mark("already", "hands", "done", 1.0)
    fake = _FakeBatched()
    monkeypatch.setattr("fastdo.engine.hands_batch.run_hands_batched", fake)

    run_hands_stage(manifest, engine_cfg, HANDS_CFG, tmp_path / "jobs")

    assert [[c.clip_id for c in call] for call in fake.calls] == [["ok"]]
    assert manifest.status("precomputed", "hands").state == "pending"


def test_run_hands_stage_returns_none_without_eligible_clips(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest, engine_cfg = _setup(tmp_path, ["a"])
    manifest.mark("a", "hands", "done", 1.0)
    fake = _FakeBatched()
    monkeypatch.setattr("fastdo.engine.hands_batch.run_hands_batched", fake)

    assert run_hands_stage(manifest, engine_cfg, HANDS_CFG, tmp_path / "jobs") is None
    assert fake.calls == []


def test_run_hands_stage_uses_intrinsics_focal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest, engine_cfg = _setup(tmp_path, ["clip"], fx=812.5)
    fake = _FakeBatched()
    monkeypatch.setattr("fastdo.engine.hands_batch.run_hands_batched", fake)

    run_hands_stage(manifest, engine_cfg, HANDS_CFG, tmp_path / "jobs")

    (clip,) = fake.calls[0]
    artifact_dir = engine_cfg.artifact_root / "clip"
    assert HANDS_OUTPUT == "hands_hawor.npz"
    assert clip.focal_px == 812.5
    assert clip.frames_path == artifact_dir / "decode.npz" and clip.out_npz == artifact_dir / "hands_hawor.npz"


def test_run_hands_stage_salvages_partial_response(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ids = ["c0", "c1", "c2", "c3"]
    manifest, engine_cfg = _setup(tmp_path, ids)

    def _partial(clips: Sequence[HandsClipInput], cfg: HandsConfig, job_dir: Path) -> HandsCallReport:
        job_dir.mkdir(parents=True, exist_ok=True)
        done = []
        for clip in clips[:2]:
            with np.load(clip.frames_path, allow_pickle=False) as data:
                write_valid_hand_npz(clip.out_npz, data["frame_idx"])
            done.append(clip_result(clip.clip_id, "done", "", 4, total=2.0))
        write_response(job_dir / "response.json", "", done, load_seconds=4.0, warmup_seconds=0.0)
        raise TimeoutError("HaWoR driver exceeded 200 s; see driver.log")

    monkeypatch.setattr("fastdo.engine.hands_batch.run_hands_batched", _partial)

    report = run_hands_stage(manifest, engine_cfg, HANDS_CFG, tmp_path / "jobs")

    assert report is not None and [c.status for c in report.clips] == ["done", "done", "failed", "failed"]
    for clip_id in ("c0", "c1"):
        status = manifest.status(clip_id, "hands")
        assert status.state == "done" and status.seconds == pytest.approx(2.0 + 4.0 / 4, abs=1e-9)
        assert (engine_cfg.artifact_root / clip_id / "hands.npz").is_file()
    for clip_id in ("c2", "c3"):
        status = manifest.status(clip_id, "hands")
        assert status.state == "failed" and "exceeded 200 s" in status.error


def test_run_hands_stage_marks_whole_call_failed_without_response(tmp_path: Path,
                                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    manifest, engine_cfg = _setup(tmp_path, ["c0", "c1", "c2"])
    calls: list[int] = []

    def _crash(clips: Sequence[HandsClipInput], cfg: HandsConfig, job_dir: Path) -> HandsCallReport:
        calls.append(len(clips))
        raise RuntimeError("HaWoR driver exited with code -9: killed")

    monkeypatch.setattr("fastdo.engine.hands_batch.run_hands_batched", _crash)
    from dataclasses import replace

    run_hands_stage(manifest, engine_cfg, replace(HANDS_CFG, max_clips_per_call=2), tmp_path / "jobs")

    assert calls == [2, 1]                                           # the stage moves on to the next call
    assert all(manifest.status(c, "hands").state == "failed" for c in ("c0", "c1", "c2"))
    assert "code -9" in manifest.status("c2", "hands").error
