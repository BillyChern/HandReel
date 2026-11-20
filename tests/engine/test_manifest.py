"""Tests for `fastdo.engine.manifest.ClipManifest`."""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import pytest

from fastdo.engine.manifest import STAGE_NAMES, ClipManifest


def _create(n: int = 5) -> ClipManifest:
    clip_ids = [f"clip{i}" for i in range(n)]
    sources = [f"source{i}.mp4" for i in range(n)]
    n_frames = [10 * (i + 1) for i in range(n)]
    return ClipManifest.create(clip_ids, sources, n_frames)


def test_manifest_create_and_roundtrip(tmp_path: Path) -> None:
    manifest = _create(5)
    path = tmp_path / "manifest.parquet"
    manifest.save(path)

    loaded = ClipManifest.load(path)
    assert loaded.clip_ids() == [f"clip{i}" for i in range(5)]
    for i in range(5):
        clip_id = f"clip{i}"
        assert loaded.n_frames(clip_id) == 10 * (i + 1)
        for stage in STAGE_NAMES:
            status = loaded.status(clip_id, stage)
            assert status.state == "pending"
            assert status.seconds == 0.0
            assert status.error == ""


def test_manifest_mark_and_pending() -> None:
    manifest = _create(5)
    for i in range(3):
        manifest.mark(f"clip{i}", "decode", "done", seconds=1.5 * (i + 1))
    assert sorted(manifest.pending("decode")) == ["clip3", "clip4"]
    # Marked clips report their recorded seconds and a "done" state.
    assert manifest.status("clip0", "decode").state == "done"
    assert manifest.status("clip0", "decode").seconds == pytest.approx(1.5)


def test_manifest_atomic_save(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = _create(3)
    path = tmp_path / "manifest.parquet"
    manifest.save(path)

    # Mutate in memory, then simulate a crash mid-write on the *next* save.
    manifest.mark("clip0", "decode", "done", seconds=2.0)

    def _raise_replace(*_args: object, **_kwargs: object) -> None:
        raise OSError("simulated crash mid-write")

    monkeypatch.setattr("os.replace", _raise_replace)
    with pytest.raises(OSError):
        manifest.save(path)

    reloaded = ClipManifest.load(path)
    assert reloaded.status("clip0", "decode").state == "pending"
    # The `.tmp` sibling must not be left behind as debris after a failed save.
    assert not path.with_suffix(".tmp").exists()


def test_manifest_stage_seconds_done_rows_only() -> None:
    manifest = _create(3)
    manifest.mark("clip0", "decode", "done", seconds=3.0)
    manifest.mark("clip1", "decode", "failed", seconds=1.0, error="boom")
    assert manifest.stage_seconds("decode") == [3.0]


def test_manifest_stage_names_includes_custom_stages_sorted() -> None:
    manifest = _create(2)
    manifest.mark("clip0", "zzz_custom", "done", seconds=1.0)
    manifest.mark("clip0", "aaa_custom", "done", seconds=1.0)
    names = manifest.stage_names()
    assert names == sorted(names)
    assert set(STAGE_NAMES).issubset(set(names))
    assert {"aaa_custom", "zzz_custom"}.issubset(set(names))


def test_manifest_save_column_order_is_deterministic(tmp_path: Path) -> None:
    """Marking stages in a different order per clip must not change the saved column order."""
    manifest = _create(2)
    manifest.mark("clip0", "encoder", "done", seconds=1.0)
    manifest.mark("clip0", "decode", "done", seconds=1.0)
    manifest.mark("clip1", "decode", "done", seconds=1.0)
    manifest.mark("clip1", "encoder", "done", seconds=1.0)

    path = tmp_path / "manifest.parquet"
    manifest.save(path)
    columns = pq.read_table(path).column_names

    # The fixed prefix always comes first, then per-stage blocks in sorted-stage order (batched runs add batch_id).
    assert columns[:3] == ["clip_id", "source", "n_frames"]
    stage_cols = columns[3:]
    expected_stage_cols: list[str] = []
    for stage in sorted(STAGE_NAMES):
        expected_stage_cols.extend([f"{stage}_state", f"{stage}_seconds", f"{stage}_error", f"{stage}_batch_id"])
    assert stage_cols == expected_stage_cols


def test_manifest_batch_id_roundtrip_and_legacy_default(tmp_path: Path) -> None:
    from fastdo.engine.manifest import ClipStatus
    from fastdo.engine.orchestrator import EngineConfig, run_clip

    manifest = _create(2)
    manifest.mark("clip0", "encoder", "done", 1.5, batch_id="t0-encoder-00000")
    run_clip("clip1", EngineConfig(artifact_root=tmp_path / "art", stages=("decode",)),
             {"decode": lambda clip_id, artifact_dir: None}, manifest)
    manifest.save(tmp_path / "m.parquet")
    loaded = ClipManifest.load(tmp_path / "m.parquet")
    status = loaded.status("clip0", "encoder")
    assert (status.batch_id, status.seconds) == ("t0-encoder-00000", 1.5)
    assert loaded.status("clip1", "decode").state == "done" and loaded.status("clip1", "decode").batch_id == ""
    _write_d1_manifest(tmp_path / "d1.parquet")
    legacy = ClipManifest.load(tmp_path / "d1.parquet")
    assert all(legacy.status(c, s).batch_id == "" for c in legacy.clip_ids() for s in STAGE_NAMES)
    assert ClipStatus("clip0", "intrinsics", "pending", 0.0, "") == legacy.status("clip0", "intrinsics")


# --- D3 amendments: the intrinsics stage and the new stage order (2026-09-13) --------------------

_D1_STAGE_NAMES = ("decode", "encoder", "gate", "mask", "shape", "geometry", "hands", "tracks", "solve", "qc",
                   "retarget")


def test_stage_order_intrinsics_hands_geometry() -> None:
    from fastdo.engine.orchestrator import STAGE_GPU

    order = list(STAGE_NAMES)
    assert order.index("shape") < order.index("intrinsics") < order.index("hands") < order.index("geometry")
    assert STAGE_GPU["intrinsics"] is True
    # STAGE_GPU also declares the OPT-IN stages (prompt, contact, driftqc), so it is a superset. It used to be
    # required to equal STAGE_NAMES, which is why `prompt` had no entry and ran unlocked; see
    # test_orchestrator.test_a_stage_that_never_declared_its_gpu_use_is_locked.
    assert set(STAGE_GPU) >= set(STAGE_NAMES)


def _write_d1_manifest(path: Path) -> None:
    """A parquet manifest as D1 saved it: sorted stage triples of the old 11-stage order, no intrinsics."""
    import pyarrow as pa

    rows = []
    for i, clip_id in enumerate(["clip0", "clip1"]):
        row: dict[str, object] = {"clip_id": clip_id, "source": f"s{i}.mp4", "n_frames": 46}
        for stage in sorted(_D1_STAGE_NAMES):
            done = clip_id == "clip0" and stage in ("decode", "encoder")
            row[f"{stage}_state"] = "done" if done else "pending"
            row[f"{stage}_seconds"] = 1.0 if done else 0.0
            row[f"{stage}_error"] = ""
        rows.append(row)
    pq.write_table(pa.Table.from_pylist(rows), path)


def test_d1_manifest_loads_and_resumes_under_new_stage_order(tmp_path: Path) -> None:
    from fastdo.engine.manifest import ClipStatus
    from fastdo.engine.orchestrator import EngineConfig, run_manifest

    path = tmp_path / "d1.parquet"
    _write_d1_manifest(path)
    manifest = ClipManifest.load(path)
    assert manifest.status("clip0", "decode").state == "done"
    assert manifest.status("clip0", "intrinsics") == ClipStatus("clip0", "intrinsics", "pending", 0.0, "")
    assert manifest.pending("intrinsics") == ["clip0", "clip1"]

    calls: list[tuple[str, str]] = []

    def _worker(stage: str):
        def run(clip_id: str, _artifact_dir: Path) -> None:
            calls.append((clip_id, stage))
        return run

    resumed = tmp_path / "resumed.parquet"
    cfg = EngineConfig(artifact_root=tmp_path / "artifacts", manifest_path=resumed)
    run_manifest(manifest, cfg, {stage: _worker(stage) for stage in STAGE_NAMES})

    assert [s for c, s in calls if c == "clip0"] == [s for s in STAGE_NAMES if s not in ("decode", "encoder")]
    assert [s for c, s in calls if c == "clip1"] == list(STAGE_NAMES)
    reloaded = ClipManifest.load(resumed)
    assert all(reloaded.status(c, s).state == "done" for c in ("clip0", "clip1") for s in STAGE_NAMES)
