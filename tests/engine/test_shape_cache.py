"""The per-clip shape cache; fakes unless marked GPU."""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import trimesh
from omegaconf import OmegaConf

from fastdo.bench import bench_batched, bench_stages
from fastdo.engine import shape_cache
from fastdo.engine.shape_cache import SHAPE_CODE_FILES, lookup, store
from fastdo.engine.workers_shape import (
    MESH_FILE,
    SAMPLES_FILE,
    SDF_FILE,
    SHAPE_META_FILE,
    make_shape_worker,
)
from fastdo.shape import ShapeConfig, ShapeResult
from fastdo.shape.sam3d_backend import Sam3dBackend, sam3d_checkpoint_identity, sam3d_env_identity
from tests.engine.fakes import random_frames, write_decode

ARTIFACTS = (MESH_FILE, SDF_FILE, SAMPLES_FILE)
PREP: dict[str, Any] = {"sdf_resolution": 16, "sdf_padding": 0.15, "n_samples": 64, "samples_seed": 0}
BASE_CFG = ShapeConfig(backend="sam3d", deterministic=False, spconv_kernel_pin=False)   # the key set before the cache
TODAY_KEYS = {"clip_id", "keyframe_index", "frame_idx", "backend", "backend_class", "sparse_structure_steps",
              "structured_latent_steps", "solver", "flow_seed", "mesh_postprocess", "normalization_scale",
              "normalization_center", "watertight", "n_vertices", "n_faces", "sdf_resolution", "sdf_padding",
              "n_samples", "samples_seed", "texture_path", "phase_seconds"}


class _Fake:
    """A cube backend with cache identities that counts its `reconstruct` calls."""

    def __init__(self, identity: str = "ckpt:a", env: str = "env:a") -> None:
        box = trimesh.creation.box(extents=(0.3, 0.2, 0.4))
        self.vertices, self.faces = np.asarray(box.vertices) + 0.1, np.asarray(box.faces)
        self.identity, self.env, self.calls = identity, env, 0

    def reconstruct(self, image: np.ndarray, mask: np.ndarray) -> ShapeResult:
        self.calls += 1
        return ShapeResult(self.vertices, self.faces, None, {"decode": 0.125 * self.calls})

    def cache_identity(self) -> str:
        return self.identity

    def cache_env_identity(self) -> str:
        return self.env


def _clip(d: Path, keyframe_index: int = 1, image_fn: Any = None, mask_fn: Any = None) -> None:
    frames = random_frames(3, h=10, w=12, seed=7)
    masks = np.zeros((3, 10, 12), dtype=bool)
    masks[:, 2:8, 3:9] = True
    if image_fn is not None:
        frames = image_fn(frames)
    if mask_fn is not None:
        masks = mask_fn(masks)
    write_decode(d, frames)
    np.savez(d / "mask.npz", masks=masks, frame_idx=np.arange(3, dtype=np.int64) * 3,
             keyframe_index=np.array(keyframe_index, dtype=np.int64))


def _run(d: Path, cache: Path | None, backend: Any, cfg: ShapeConfig = BASE_CFG, clip_id: str = "clip",
         **prep: Any) -> dict[str, Any]:
    make_shape_worker(cfg, backend=backend, cache_dir=cache, **(PREP | prep))(clip_id, d)
    meta: dict[str, Any] = json.loads((d / SHAPE_META_FILE).read_text())
    return meta


def _entry(cache: Path, key: str) -> Path:
    return cache / key[:2] / key


def test_entry_files_are_the_worker_artifacts() -> None:
    assert set(shape_cache.ENTRY_FILES) == {MESH_FILE, SDF_FILE, SAMPLES_FILE, SHAPE_META_FILE}


def test_cache_disabled_is_unchanged(tmp_path: Path) -> None:                       # test 15 (with Check M4)
    clip = tmp_path / "clip"
    _clip(clip)
    meta = _run(clip, None, _Fake())
    assert set(meta) == TODAY_KEYS
    assert all(p.is_relative_to(clip) for p in tmp_path.rglob("*") if p != clip)


def test_miss_then_hit_byte_identical(tmp_path: Path) -> None:                      # test 16
    cache, a, b, fake = tmp_path / "cache", tmp_path / "a", tmp_path / "b", _Fake()
    _clip(a), _clip(b)
    miss = _run(a, cache, fake)
    assert fake.calls == 1 and miss["cache_hit"] is False and miss["cache_seconds"] >= 0
    hit = _run(b, cache, fake)
    assert fake.calls == 1 and hit["cache_hit"] is True and hit["cache_key"] == miss["cache_key"]
    entry = _entry(cache, miss["cache_key"])
    for name in ARTIFACTS:
        assert (a / name).read_bytes() == (b / name).read_bytes() == (entry / name).read_bytes()
    for meta in (miss, hit):
        meta.pop("cache_hit"), meta.pop("cache_seconds")
    assert miss == hit
    assert list((a / SHAPE_META_FILE).read_text().splitlines())[1].strip().startswith('"clip_id"')
    assert not list(a.glob("*.tmp")) and not list(b.glob("*.tmp"))


def test_hit_writes_the_current_keyframe_metadata(tmp_path: Path) -> None:          # Check H3
    cache, a, b = tmp_path / "cache", tmp_path / "a", tmp_path / "b"
    _clip(a, keyframe_index=2)
    _clip(b, keyframe_index=0, image_fn=lambda f: f[[2, 0, 1]], mask_fn=lambda m: m[[2, 0, 1]])
    fake = _Fake()
    first = _run(a, cache, fake, clip_id="a")
    second = _run(b, cache, fake, clip_id="b")
    assert fake.calls == 1 and second["cache_hit"] is True
    assert (first["keyframe_index"], first["frame_idx"]) == (2, 6)
    assert (second["clip_id"], second["keyframe_index"], second["frame_idx"]) == ("b", 0, 0)
    stored = json.loads((_entry(cache, first["cache_key"]) / SHAPE_META_FILE).read_text())
    assert not {"clip_id", "keyframe_index", "frame_idx", "cache_hit", "cache_key", "cache_seconds"} & set(stored)


def test_hit_does_not_load_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:   # test 17
    cache, a, b = tmp_path / "cache", tmp_path / "a", tmp_path / "b"
    _clip(a), _clip(b)
    _run(a, cache, _Fake())

    class _Unloadable(_Fake):
        def load(self) -> None:
            raise AssertionError("the backend must not load on a hit")

        def reconstruct(self, image: np.ndarray, mask: np.ndarray) -> ShapeResult:
            raise AssertionError("the backend must not run on a hit")

    monkeypatch.setattr("fastdo.engine.workers_shape.make_shape_backend", lambda cfg: _Unloadable())
    make_shape_worker(BASE_CFG, cache_dir=cache, **PREP)("clip", b)
    assert json.loads((b / SHAPE_META_FILE).read_text())["cache_hit"] is True


def _swap_image(f: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(f.reshape(f.shape[0], f.shape[2], f.shape[1], 3))


def _code_copy(tmp_path: Path, append: bool) -> Path:
    root = tmp_path / f"code_{append}"
    for rel in SHAPE_CODE_FILES:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(shape_cache.SHAPE_CODE_ROOT / rel, root / rel)
    if append:
        with (root / SHAPE_CODE_FILES[0]).open("ab") as f:
            f.write(b"#")
    return root


VARIANTS = ["image_byte", "mask_pixel", "image_shape", "seed", "deterministic", "slat_steps", "solver",
            "mesh_postprocess", "sam3d_repo", "n_samples", "samples_seed", "sdf_resolution", "sdf_padding",
            "cache_identity", "env", "code_version", "code_file"]


@pytest.mark.parametrize("variant", VARIANTS)
def test_key_changes_with_each_part(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variant: str) -> None:  # 18
    cache, fake = tmp_path / "cache", _Fake()
    base_dir = tmp_path / "base"
    _clip(base_dir)
    base = _run(base_dir, cache, fake)
    assert fake.calls == 1
    kw: dict[str, Any] = {}
    image_fn = mask_fn = None
    cfg, backend = BASE_CFG, fake
    if variant == "image_byte":
        def image_fn(f: np.ndarray) -> np.ndarray:
            f = f.copy()
            f[1, 0, 0, 0] = (int(f[1, 0, 0, 0]) + 1) % 256
            return f
    elif variant == "mask_pixel":
        def mask_fn(m: np.ndarray) -> np.ndarray:
            m = m.copy()
            m[1, 0, 0] = ~m[1, 0, 0]
            return m
    elif variant == "image_shape":
        image_fn = _swap_image
        frames = random_frames(3, h=10, w=12, seed=7)
        assert _swap_image(frames).shape == (3, 12, 10, 3) and _swap_image(frames).tobytes() == frames.tobytes()
    elif variant in ("seed", "deterministic", "slat_steps", "solver", "mesh_postprocess", "sam3d_repo"):
        field, value = {"seed": ("seed", 43), "deterministic": ("deterministic", True),
                        "slat_steps": ("structured_latent_steps", 24), "solver": ("solver", "midpoint"),
                        "mesh_postprocess": ("mesh_postprocess", False), "sam3d_repo": ("sam3d_repo", "x/y")}[variant]
        cfg = dataclasses.replace(BASE_CFG, **{field: value})
    elif variant in PREP:
        kw = {variant: {"sdf_resolution": 17, "sdf_padding": 0.2, "n_samples": 65, "samples_seed": 1}[variant]}
    elif variant == "cache_identity":
        backend.identity = "ckpt:b"
    elif variant == "env":
        backend.env = "env:b"
    elif variant == "code_version":
        monkeypatch.setattr(shape_cache, "SHAPE_CODE_VERSION", shape_cache.SHAPE_CODE_VERSION + 1)
    elif variant == "code_file":
        monkeypatch.setattr(shape_cache, "SHAPE_CODE_ROOT", _code_copy(tmp_path, append=False))
        assert _run(base_dir, cache, fake)["cache_key"] == base["cache_key"]      # same bytes elsewhere: same key
        monkeypatch.setattr(shape_cache, "SHAPE_CODE_ROOT", _code_copy(tmp_path, append=True))
    changed_dir = tmp_path / "changed"
    _clip(changed_dir, image_fn=image_fn, mask_fn=mask_fn)
    calls = fake.calls
    changed = _run(changed_dir, cache, backend, cfg=cfg, **kw)
    assert changed["cache_key"] != base["cache_key"]
    assert changed["cache_hit"] is False and fake.calls == calls + 1


def test_key_is_stable_and_ignores_cache_location(tmp_path: Path) -> None:          # test 19
    keys = set()
    for i in range(2):
        d = tmp_path / f"clip{i}"
        _clip(d)
        keys.add(_run(d, tmp_path / f"cache{i}", _Fake(), clip_id=f"id{i}")["cache_key"])
    assert len(keys) == 1


def test_keyframe_dtypes_are_asserted(tmp_path: Path) -> None:                       # Check M5
    d = tmp_path / "clip"
    _clip(d)
    with np.load(d / "decode.npz") as data:
        np.savez(d / "decode.npz", frames=data["frames"].astype(np.int16), frame_idx=data["frame_idx"])
    with pytest.raises(ValueError, match="uint8"):
        _run(d, tmp_path / "cache", _Fake())


def test_atomic_store_leaves_no_partial_entry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # 20
    cache, fake, real_rename, raised = tmp_path / "cache", _Fake(), os.rename, []

    def flaky(src: Any, dst: Any) -> None:
        if not raised:
            raised.append(dst)
            raise OSError("simulated crash at rename")
        real_rename(src, dst)

    monkeypatch.setattr(os, "rename", flaky)
    a = tmp_path / "a"
    _clip(a)
    key = _run(a, cache, fake)["cache_key"]
    assert raised and not _entry(cache, key).exists()
    assert all(p.is_relative_to(cache / ".tmp") for p in cache.rglob("*") if p.is_file())
    assert not any((cache / ".tmp").iterdir())                        # store removed its own tmp dir
    b = tmp_path / "b"
    _clip(b)
    assert _run(b, cache, fake)["cache_hit"] is False and fake.calls == 2
    assert lookup(cache, key) is not None and _run(a, cache, fake)["cache_hit"] is True


def test_concurrent_store_keeps_first(tmp_path: Path) -> None:                       # test 21
    cache, key = tmp_path / "cache", "ab" + "0" * 62
    first = {name: name.encode() for name in shape_cache.ENTRY_FILES}
    store(cache, key, {"schema": "x"}, first)
    store(cache, key, {"schema": "x"}, {name: b"other" for name in shape_cache.ENTRY_FILES})
    entry = lookup(cache, key)
    assert entry is not None and entry.files == first
    assert not list((cache / ".tmp").iterdir())


def test_corrupt_entry_is_quarantined_and_recomputed(tmp_path: Path) -> None:        # test 22
    cache, fake, a, b = tmp_path / "cache", _Fake(), tmp_path / "a", tmp_path / "b"
    _clip(a), _clip(b)
    key = _run(a, cache, fake)["cache_key"]
    path = _entry(cache, key) / SAMPLES_FILE
    data = bytearray(path.read_bytes())
    data[-1] ^= 0xFF
    path.write_bytes(bytes(data))
    meta = _run(b, cache, fake)
    assert meta["cache_hit"] is False and fake.calls == 2
    assert len(list((cache / ".corrupt").iterdir())) == 1 and lookup(cache, key) is not None
    manifest = _entry(cache, key) / "manifest.json"                 # an extra listed file is corrupt too (Check L3)
    doc = json.loads(manifest.read_text())
    doc["files"]["extra.txt"] = doc["files"][MESH_FILE]
    manifest.write_text(json.dumps(doc))
    assert lookup(cache, key) is None and len(list((cache / ".corrupt").iterdir())) == 2


def test_backend_without_identity_refuses_cache(tmp_path: Path) -> None:             # test 23
    class _Bare:
        def reconstruct(self, image: np.ndarray, mask: np.ndarray) -> ShapeResult:
            raise AssertionError

    d = tmp_path / "clip"
    _clip(d)
    with pytest.raises(ValueError, match="does not support the shape cache"):
        _run(d, tmp_path / "cache", _Bare())


def test_sam3d_cache_identity_from_hf_layout(tmp_path: Path) -> None:                # test 24
    snap = tmp_path / "models--x" / "snapshots" / "c0ffee"
    (snap / "checkpoints").mkdir(parents=True)
    (tmp_path / "models--x" / "blobs").mkdir()
    for name, blob in (("x.ckpt", "h1"), ("pipeline.yaml", "h0")):
        (tmp_path / "models--x" / "blobs" / blob).write_text(name)
        (snap / "checkpoints" / name).symlink_to(f"../../../blobs/{blob}")
    assert sam3d_checkpoint_identity("x", lambda **kw: str(snap)) == "sam3d:c0ffee:pipeline.yaml=h0,x.ckpt=h1"
    backend = Sam3dBackend(BASE_CFG, snapshot_fn=lambda **kw: str(snap))
    assert backend.cache_identity() == "sam3d:c0ffee:pipeline.yaml=h0,x.ckpt=h1"
    (snap / "checkpoints" / "x.ckpt").unlink()
    (snap / "checkpoints" / "x.ckpt").write_text("regular")
    with pytest.raises(ValueError, match="symlink"):
        sam3d_checkpoint_identity("x", lambda **kw: str(snap))


def test_sam3d_env_identity_probes_the_resolved_interpreter(tmp_path: Path) -> None:  # Check H1
    pkg = tmp_path / "pylib" / "sam3d_objects"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("raise RuntimeError('the probe must not import the package')\n")
    cfg = dataclasses.replace(BASE_CFG, sam3d_python=sys.executable, sam3d_pythonpath=(str(tmp_path / "pylib"),))
    first = json.loads(sam3d_env_identity(cfg))
    assert first["interpreter"] == os.path.realpath(sys.executable) and first["pythonpath"] == [str(pkg.parent)]
    assert first["probe"]["torch_imported"] is False and first["probe"]["sam3d_objects_sha256"]
    assert "numpy" in first["probe"]["distributions"]
    assert set(first["probe"]["env"]) >= {"SPCONV_ALGO", "ATTN_BACKEND", "SPARSE_ATTN_BACKEND"}
    (pkg / "sub.py").write_text("x = 1\n")
    assert json.loads(sam3d_env_identity(cfg))["probe"]["sam3d_objects_sha256"] != first["probe"][
        "sam3d_objects_sha256"]
    assert Sam3dBackend(cfg, env_probe=lambda c: "probe:" + c.backend).cache_env_identity() == "probe:sam3d"
    with pytest.raises(ValueError, match="sam3d_objects"):
        sam3d_env_identity(dataclasses.replace(cfg, sam3d_pythonpath=(str(tmp_path),)))
    with pytest.raises(ValueError, match="interpreter"):
        sam3d_env_identity(dataclasses.replace(cfg, sam3d_python=str(tmp_path / "missing_python")))


def test_bench_shape_cache_dir_wiring(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # test 25
    seen: list[dict[str, Any]] = []
    monkeypatch.setattr(bench_stages, "make_shape_worker", lambda cfg, **kw: seen.append(kw) or (lambda c, d: None))
    for value in ("cache/shape", None):
        cfg = OmegaConf.create({"bench": {"shape": {"backend": "sam3d"}, "shape_cache_dir": value}})
        bench_stages._shape_stage(cfg, tmp_path, [])
    assert seen == [{"cache_dir": (tmp_path / "cache/shape").resolve()}, {"cache_dir": None}]
    conf = OmegaConf.load(Path(__file__).parents[2] / "run" / "conf" / "config.yaml")
    assert conf.bench.shape_cache_dir is None
    assert {"cache_hit", "cache_seconds"} <= bench_batched.VOLATILE_KEYS and "cache_key" not in bench_batched.VOLATILE_KEYS


@pytest.mark.skipif(not torch.cuda.is_available(), reason="SAM 3D Objects needs CUDA (and > 17 GiB of allocator memory)")
def test_real_shape_cache_on_d14_clip(tmp_path: Path) -> None:                         # test 26 (GPU)
    if not os.environ.get("FASTDO_SAM3D_PYTHON"):
        pytest.skip("FASTDO_SAM3D_PYTHON (the SAM 3D environment's interpreter) is not set")
    if float(os.environ.get("FASTDO_GPU_MEM_GB", "12")) < 19:
        pytest.skip("SAM 3D peaks at 18.67 GiB allocated; run with FASTDO_GPU_MEM_GB >= 19")
    source, cache_root = os.environ.get("FASTDO_ENGINE_DET_D14_CLIP"), os.environ.get("FASTDO_ENGINE_DET_CACHE")
    if not source or not cache_root:
        pytest.skip("FASTDO_ENGINE_DET_D14_CLIP (a batched-run artifact dir) and FASTDO_ENGINE_DET_CACHE are not set")
    cache = Path(cache_root) / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    dirs = [tmp_path / "miss", tmp_path / "hit"]
    for d in dirs:
        d.mkdir()
        for name in ("decode.npz", "mask.npz"):
            shutil.copyfile(Path(source) / name, d / name)
    cfg = ShapeConfig(backend="sam3d")
    backend = Sam3dBackend(cfg)
    record: dict[str, Any] = {"source": source, "cache": str(cache), "cfg": dataclasses.asdict(cfg)}
    try:
        start = time.perf_counter()
        make_shape_worker(cfg, backend=backend, cache_dir=cache)("d14", dirs[0])
        record["miss_wall_s"] = time.perf_counter() - start
    finally:
        record["worker_exit"] = backend.close()

    def _refuse() -> Any:
        raise AssertionError("the backend must not load on a hit")

    unloaded = Sam3dBackend(cfg, loader=_refuse)
    start = time.perf_counter()
    make_shape_worker(cfg, backend=unloaded, cache_dir=cache)("d14", dirs[1])
    record["hit_wall_s"] = time.perf_counter() - start
    metas = [json.loads((d / SHAPE_META_FILE).read_text()) for d in dirs]
    record |= {"miss_meta": metas[0], "hit_meta": metas[1], "loaded_on_hit": unloaded.loaded,
               "identical": {name: (dirs[0] / name).read_bytes() == (dirs[1] / name).read_bytes() for name in ARTIFACTS}}
    sys.stderr.write("F1_CACHE_RECORD " + json.dumps(record) + "\n")
    assert metas[0]["cache_hit"] is False and metas[1]["cache_hit"] is True and not unloaded.loaded
    assert all(record["identical"].values())
    stripped = [{k: v for k, v in m.items() if k not in ("cache_hit", "cache_seconds")} for m in metas]
    assert stripped[0] == stripped[1]


def test_hands_outputs_enter_no_shape_cache_key() -> None:
    """The camera shear changes `hands.npz`. The shape stage runs before `hands`; its key hashes the
    keyframe image and mask, the backend identities, the config and `SHAPE_CODE_FILES` only, none of which is a hands
    artifact or hands code, so no shape cache key needs a bump."""
    import inspect

    assert list(inspect.signature(shape_cache.ShapeCacheKeyer.key).parameters) == ["self", "image", "mask"]
    assert not [rel for rel in SHAPE_CODE_FILES if rel.startswith("hands/") or "hands" in Path(rel).name]
