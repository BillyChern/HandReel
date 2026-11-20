"""The engine's `shape` stage worker: keyframe, mesh preparation, SDF and samples artifacts (D4; fakes only)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
import trimesh

from fastdo.engine.workers import make_shape_worker
from fastdo.engine.workers_shape import (
    MESH_FILE,
    SAMPLES_FILE,
    SDF_FILE,
    SHAPE_META_FILE,
    read_mesh_samples,
    read_sdf_grid,
    read_shape_meta,
)
from fastdo.retarget.primitives import as_phi_fn
from fastdo.shape import ShapeConfig, ShapeResult
from fastdo.solver.sdf import sdf_lookup
from tests.engine.fakes import (
    forbid_full_frames,
    full_load_decode_rows,
    random_frames,
    write_decode,
)

_SIDE = 0.3


class _CubeBackend:
    """Returns a cube of side 0.3 away from the origin and records what it was asked to reconstruct."""

    def __init__(self) -> None:
        box = trimesh.creation.box(extents=(_SIDE, _SIDE, _SIDE))
        self.vertices = np.asarray(box.vertices) + np.array([0.1, -0.2, 0.5])
        self.faces = np.asarray(box.faces)
        self.calls: list[tuple[np.ndarray, np.ndarray]] = []

    def reconstruct(self, image: np.ndarray, mask: np.ndarray) -> ShapeResult:
        self.calls.append((image.copy(), mask.copy()))
        return ShapeResult(self.vertices, self.faces, None, {"decode": 0.25})


def _write_clip(artifact_dir: Path, keyframe_index: int, n: int = 4, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    frames = random_frames(n, h=16, w=20, seed=seed)
    write_decode(artifact_dir, frames)
    masks = np.zeros((n, 16, 20), dtype=bool)
    for row in range(n):
        masks[row, 2 + row:8 + row, 3:12] = True
    np.savez(artifact_dir / "mask.npz", masks=masks, frame_idx=np.arange(n, dtype=np.int64) * 3,
             keyframe_index=np.array(keyframe_index, dtype=np.int64))
    return frames, masks


def test_shape_worker_pipeline_end_to_end(tmp_path: Path) -> None:
    clip = tmp_path / "clip0"
    frames, masks = _write_clip(clip, keyframe_index=2)
    backend = _CubeBackend()

    make_shape_worker(backend=backend, sdf_resolution=32, n_samples=512, samples_seed=3)("clip0", clip)

    ((image, mask),) = backend.calls                              # the mask stage's keyframe, once
    np.testing.assert_array_equal(image, frames[2])
    np.testing.assert_array_equal(mask, masks[2])
    assert all((clip / name).is_file() for name in (MESH_FILE, SDF_FILE, SAMPLES_FILE, SHAPE_META_FILE))

    sdf, samples = read_sdf_grid(clip), read_mesh_samples(clip)
    phi, _ = sdf_lookup(sdf, samples.points[None].double())
    assert float(phi.abs().max()) <= sdf.step                      # phi ~ 0 on the cube surface within one cell
    centre_phi, _ = sdf_lookup(sdf, torch.zeros(1, 1, 3, dtype=torch.float64))
    half_side = 0.5 / np.sqrt(3.0)                                 # unit diagonal -> side 1/sqrt(3)
    assert abs(float(centre_phi) + half_side) <= sdf.step          # negative inside
    phi_retarget, _ = as_phi_fn(sdf)(samples.points[None].double())   # the retargeting stage reads the same grid
    assert torch.equal(phi, phi_retarget)

    assert samples.points.shape == samples.normals.shape == (512, 3)
    assert samples.points.dtype == samples.normals.dtype == torch.float32
    assert float((samples.points.abs().max(dim=1).values - half_side).abs().max()) < 1e-5
    assert bool(((samples.normals * samples.points).sum(dim=1) > 0).all())    # outward

    mesh = cast("trimesh.Trimesh", trimesh.load(clip / MESH_FILE, process=False, force="mesh"))
    lo, hi = np.asarray(mesh.vertices).min(0), np.asarray(mesh.vertices).max(0)
    assert abs(float(np.linalg.norm(hi - lo)) - 1.0) < 1e-6 and np.allclose((lo + hi) / 2.0, 0.0, atol=1e-7)

    meta = read_shape_meta(clip)
    assert meta["keyframe_index"] == 2 and meta["frame_idx"] == 6
    assert meta["normalization_scale"] == pytest.approx(1.0 / (_SIDE * np.sqrt(3.0)))
    assert meta["normalization_center"] == pytest.approx([0.1, -0.2, 0.5])
    assert meta["watertight"] is True and meta["texture_path"] is None
    assert meta["phase_seconds"] == {"decode": 0.25}
    assert (meta["sparse_structure_steps"], meta["structured_latent_steps"], meta["solver"]) == (25, 25, "euler")
    assert (meta["flow_seed"], meta["samples_seed"], meta["sdf_padding"], meta["sdf_resolution"], meta["n_samples"]) == (
        42, 3, 0.15, 32, 512)                                      # enough to rebuild sdf.npz and samples.npz
    assert "seed" not in meta                                     # the ambiguous key is gone


def test_shape_worker_reads_only_the_keyframe_row_and_writes_what_the_full_read_wrote(tmp_path: Path) -> None:
    """The keyframe row is read alone; the backend sees the same bytes and every artifact is unchanged."""
    arms: dict[str, tuple[_CubeBackend, Path]] = {}
    for arm in ("before", "rows"):
        clip = tmp_path / arm
        frames, _ = _write_clip(clip, keyframe_index=3, n=5, seed=11)
        backend = _CubeBackend()
        with pytest.MonkeyPatch.context() as mp:
            if arm == "before":
                mp.setattr("fastdo.engine.workers_shape.read_decode_rows", full_load_decode_rows)
            else:
                forbid_full_frames(mp)
            make_shape_worker(backend=backend, sdf_resolution=16, n_samples=64)("clip0", clip)
        arms[arm] = (backend, clip)

    (before_image, before_mask), = arms["before"][0].calls
    (image, mask), = arms["rows"][0].calls
    assert image.dtype == before_image.dtype and image.shape == before_image.shape
    assert image.tobytes() == before_image.tobytes() == frames[3].tobytes()
    np.testing.assert_array_equal(mask, before_mask)
    for name in (MESH_FILE, SDF_FILE, SAMPLES_FILE):
        assert (arms["rows"][1] / name).read_bytes() == (arms["before"][1] / name).read_bytes(), name
    assert read_shape_meta(arms["rows"][1])["frame_idx"] == read_shape_meta(arms["before"][1])["frame_idx"] == 9


def test_shape_worker_still_bounds_the_keyframe_by_the_decode_rows(tmp_path: Path) -> None:
    clip = tmp_path / "clip0"
    write_decode(clip, random_frames(3, h=8, w=8))
    np.savez(clip / "mask.npz", masks=np.ones((5, 8, 8), bool), frame_idx=np.arange(5) * 3,
             keyframe_index=np.array(3))
    with pytest.raises(ValueError, match="keyframe_index 3 is outside the 3 decode rows"):
        make_shape_worker(backend=_CubeBackend(), sdf_resolution=16, n_samples=16)("clip0", clip)


def test_shape_worker_caps_gpu_memory_first_and_loads_its_backend_once(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    backend = _CubeBackend()
    backend.load = lambda: events.append("load")                   # type: ignore[attr-defined]

    def _make(cfg: ShapeConfig) -> Any:
        events.append(f"make:{cfg.backend}")
        return backend

    monkeypatch.setattr("fastdo.bench.gpu_budget.apply_gpu_budget", lambda *a, **k: events.append("budget"))
    monkeypatch.setattr("fastdo.engine.workers_shape.make_shape_backend", _make)
    dirs = [tmp_path / "clip0", tmp_path / "clip1"]
    for seed, d in enumerate(dirs):
        _write_clip(d, keyframe_index=seed, seed=seed)

    worker = make_shape_worker(ShapeConfig(backend="sam3d"), sdf_resolution=16, n_samples=64)
    assert events == []                                            # the factory allocates and loads nothing
    worker("clip0", dirs[0])
    worker("clip1", dirs[1])

    assert events == ["budget", "make:sam3d", "load"]              # cap first, one backend for both clips
    assert len(backend.calls) == 2
    assert all((d / SDF_FILE).is_file() for d in dirs)


def test_shape_worker_needs_the_mask_artifact_and_a_non_empty_keyframe_mask(tmp_path: Path) -> None:
    clip = tmp_path / "clip0"
    write_decode(clip, random_frames(2, h=8, w=8))
    worker = make_shape_worker(backend=_CubeBackend(), sdf_resolution=16, n_samples=16)
    with pytest.raises(FileNotFoundError, match="'mask'"):
        worker("clip0", clip)
    np.savez(clip / "mask.npz", masks=np.zeros((2, 8, 8), bool), frame_idx=np.array([0, 3]),
             keyframe_index=np.array(1))
    with pytest.raises(ValueError, match="empty"):
        worker("clip0", clip)


def test_shape_json_records_deterministic_only_when_true(tmp_path: Path) -> None:
    """A default run keeps today's shape.json key set, so roots written before the cache still compare."""
    for deterministic in (False, True):
        clip = tmp_path / f"clip_{deterministic}"
        _write_clip(clip, keyframe_index=1)
        make_shape_worker(ShapeConfig(deterministic=deterministic), backend=_CubeBackend(), sdf_resolution=16,
                          n_samples=64)("clip", clip)
        meta = read_shape_meta(clip)
        assert ("deterministic" in meta) is deterministic
        assert meta.get("deterministic", False) is deterministic
