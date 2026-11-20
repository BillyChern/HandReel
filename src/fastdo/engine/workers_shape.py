"""Shape stage worker: one geometry-only reconstruction per clip, then the canonical mesh, its SDF grid and samples.

`shape` runs after `mask` (`fastdo.engine.manifest.STAGE_NAMES`). It reconstructs the object on the clip's keyframe:
decode row `keyframe_index` of `mask.npz`, the row the mask prompt sits on,
with that row's mask. The mesh then goes through `normalize_mesh`, `check_watertight`,
`mesh_sdf_grid` and `canonical_samples`.

Artifacts (tmp file then `os.replace`; the npz files are uncompressed and load without pickle):
- `mesh.obj`: the canonical-unit mesh, with its bounding-box centre at the origin and a unit bounding-box diagonal.
  Every other artifact, the solver's `scale` and the retargeting `mesh_scale` (metres per canonical unit) use this
  frame.
- `sdf.npz`: `values` (G,G,G) float64 indexed [ix,iy,iz], `origin` (3,) float64 and `step` 0-d float64, i.e.
  `fastdo.solver.sdf.SdfGrid` field for field (`read_sdf_grid`); negative inside.
- `samples.npz`: `points` (M,3) float32 and outward unit `normals` (M,3) float32, i.e. `fastdo.solver.api.MeshSamples`
  (`read_mesh_samples`). The retargeting stage passes them as `surface` and `surface_normals`.
- `shape.json`: the keyframe (`keyframe_index`, `frame_idx`); the backend and its flow settings (`backend`,
  `backend_class`, the step counts, `solver`, `flow_seed`, `mesh_postprocess`); `normalization_scale` and
  `normalization_center` (canonical = (raw - center) * scale); `watertight` and the mesh size; the mesh preparation
  that produced `sdf.npz` and `samples.npz` (`sdf_resolution`, `sdf_padding`, `n_samples`, `samples_seed`);
  `texture_path` (always null) and the backend's `phase_seconds`; `deterministic: true` only when
  `ShapeConfig.deterministic` is set (default runs keep the original key set); likewise
  `spconv_kernels: "pinned"` only when `ShapeConfig.spconv_kernel_pin` is set.

Shape cache (`fastdo.engine.shape_cache`), only with `cache_dir`: the key covers the keyframe bytes, the
config, the mesh preparation, the backend's checkpoint and environment identities and the shape code. A hit copies the
entry's `mesh.obj`, `sdf.npz` and `samples.npz` and writes `shape.json` from the stored payload plus this clip's
`clip_id`, `keyframe_index` and `frame_idx`, without loading the backend; a miss computes and stores. Either way
`shape.json` gains `cache_hit`, `cache_key` and `cache_seconds` (lookup plus copy or store), and `phase_seconds` stays
the computing run's. Without `cache_dir` nothing changes.

The worker holds one backend for its lifetime. `make_shape_backend(cfg)` and the backend's `load()` run once, on the
first clip and after the CUDA cap. Every later clip reuses the loaded pipeline, which for SAM 3D is the
running worker process. Prewarm (`fastdo.engine.prewarm`): without a cache the worker is a `PrewarmableStageFn` whose
`prewarm()` performs that same load ahead of the first clip (the runner calls it only when `bench.batched.prewarm`
lists `shape`); the first clip then waits for it and uses the loaded backend, so nothing else changes.

`make_template_shape_worker` runs this same worker with a `TemplateMesh` backend: a fixed mesh file per clip (the
GT YCB mesh of a DexYCB calibration clip; never a default). `shape.json` then records
`backend: "template"` and the normalization that maps the metric template to the canonical frame.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import trimesh

from fastdo.engine.npz_rows import npz_array_shape
from fastdo.engine.orchestrator import StageFn
from fastdo.engine.prewarm import PrewarmableStageFn, PrewarmSlot
from fastdo.engine.shape_cache import UNCACHED_META_KEYS, ShapeCacheKeyer, lookup, restore, store
from fastdo.engine.worker_support import (
    DECODE_FILE,
    WorkerContext,
    read_decode_rows,
    require_artifact,
    save_npz,
)
from fastdo.engine.workers_video import MASK_FILE
from fastdo.shape.backend import (
    ShapeBackend,
    ShapeConfig,
    ShapeResult,
    make_shape_backend,
    non_default_sampler_options,
    non_default_shape_options,
    shape_options,
)
from fastdo.shape.mesh_utils import canonical_samples, check_watertight, normalize_mesh
from fastdo.solver.api import MeshSamples
from fastdo.solver.sdf import SdfGrid, mesh_sdf_grid

__all__ = [
    "MESH_FILE",
    "SAMPLES_FILE",
    "SDF_FILE",
    "SHAPE_META_FILE",
    "TEMPLATE_BACKEND",
    "TemplateMesh",
    "make_shape_worker",
    "make_template_shape_worker",
    "read_mesh_samples",
    "read_sdf_grid",
    "read_shape_meta",
]

logger = logging.getLogger(__name__)

MESH_FILE = "mesh.obj"
SDF_FILE = "sdf.npz"
SAMPLES_FILE = "samples.npz"
SHAPE_META_FILE = "shape.json"
TEMPLATE_BACKEND = "template"    # shape.json `backend` of `make_template_shape_worker` (a metric template mesh)
# What the backend reports about a fast post-process, copied into shape.json beside the options.
_POSTPROCESS_STAT_KEYS = ("decimator", "faces_raw", "faces_decimated", "faces_removed_mincut", "removed",
                          "area_ratio")
_QC_OUTCOME_KEY = "qc"          # the request's one outcome, never per pass, so it is never prefixed


def _postprocess_stats(stats: Mapping[str, Any]) -> dict[str, Any]:
    """shape.json's flat post-process statistics: the SHIPPED mesh's under the plain names, plus, after a wall-QC
    fallback, the discarded primary pass's under `primary_*`.

    The worker snapshots the primary's statistics before the fallback runs and nests the fallback's under
    `fallback`. Copying only the flat keys therefore described the mesh that was thrown away, while `mesh.obj`,
    `n_faces` and `n_vertices` in the same record came from the mesh that was kept, and the shipped mesh's own
    decimator and area ratio never reached shape.json at all - which the topology check reads.
    """
    record = {key: stats[key] for key in _POSTPROCESS_STAT_KEYS if key in stats}
    fallback = stats.get("fallback")
    if isinstance(fallback, Mapping):
        record = {f"primary_{key}": value for key, value in record.items()}
        record.update({key: fallback[key] for key in _POSTPROCESS_STAT_KEYS if key in fallback})
    if _QC_OUTCOME_KEY in stats:
        record[_QC_OUTCOME_KEY] = stats[_QC_OUTCOME_KEY]
    return record


def read_sdf_grid(artifact_dir: Path) -> SdfGrid:
    """The clip's canonical-unit `SdfGrid` from `<artifact_dir>/sdf.npz`."""
    path = require_artifact(Path(artifact_dir) / SDF_FILE, "shape")
    with np.load(path, allow_pickle=False) as data:
        return SdfGrid(values=torch.from_numpy(np.asarray(data["values"], dtype=np.float64)),
                       origin=torch.from_numpy(np.asarray(data["origin"], dtype=np.float64)), step=float(data["step"]))


def read_mesh_samples(artifact_dir: Path) -> MeshSamples:
    """The clip's canonical surface samples and outward normals from `<artifact_dir>/samples.npz`."""
    path = require_artifact(Path(artifact_dir) / SAMPLES_FILE, "shape")
    with np.load(path, allow_pickle=False) as data:
        return MeshSamples(points=torch.from_numpy(np.asarray(data["points"], dtype=np.float32)),
                           normals=torch.from_numpy(np.asarray(data["normals"], dtype=np.float32)))


def read_shape_meta(artifact_dir: Path) -> dict[str, Any]:
    """`<artifact_dir>/shape.json` as a dict."""
    meta: dict[str, Any] = json.loads(require_artifact(Path(artifact_dir) / SHAPE_META_FILE, "shape").read_text())
    return meta


def _read_keyframe(artifact_dir: Path) -> tuple[int, int, np.ndarray, np.ndarray]:
    """`(keyframe_index, source frame index, image (H,W,3) uint8, mask (H,W) bool)` of the clip's keyframe."""
    decode_path = require_artifact(artifact_dir / DECODE_FILE, "decode")
    n_frames = npz_array_shape(decode_path, "frames")[0]          # only the keyframe row is read
    path = require_artifact(artifact_dir / MASK_FILE, "mask")
    with np.load(path, allow_pickle=False) as data:
        keyframe_index, masks = int(data["keyframe_index"]), data["masks"]
        if not (0 <= keyframe_index < min(len(masks), n_frames)):
            raise ValueError(f"{path}: keyframe_index {keyframe_index} is outside the {n_frames} decode rows")
        mask = np.asarray(masks[keyframe_index], dtype=bool)
    if not mask.any():
        raise ValueError(f"{path}: the keyframe mask (row {keyframe_index}) is empty; nothing to reconstruct")
    frames, frame_idx = read_decode_rows(decode_path, [keyframe_index])
    return keyframe_index, int(frame_idx[keyframe_index]), frames[0], mask


def _write_mesh(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    tmp = path.with_name(path.name + ".tmp")
    try:
        trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(str(tmp), file_type="obj")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


def _load_backend(cfg: ShapeConfig, built: ShapeBackend | None = None) -> ShapeBackend:
    backend = built if built is not None else make_shape_backend(cfg)
    load = getattr(backend, "load", None)
    if callable(load):
        load()
    return backend


def make_shape_worker(cfg: ShapeConfig | None = None, backend: ShapeBackend | None = None, sdf_resolution: int = 64,
                      sdf_padding: float = 0.15, n_samples: int = 2048, samples_seed: int = 0,
                      cache_dir: Path | None = None) -> StageFn:
    """The `shape` stage worker (see the module docstring); nothing loads until its first clip.

    Args:
        cfg: Backend selection and the geometry-only flow settings (`ShapeConfig()`: TRELLIS, 25 + 25 Euler steps).
        backend: An already-built backend to use instead of `make_shape_backend(cfg)` (tests inject fakes).
        sdf_resolution: Grid samples per axis of `mesh_sdf_grid`.
        sdf_padding: `mesh_sdf_grid`'s margin, as a fraction of the object's extent.
        n_samples: Surface samples in `samples.npz` (the DexYCB evaluation's solver uses 2048).
        samples_seed: Seed of `canonical_samples` (the flow seed is `cfg.seed`).
        cache_dir: Shape cache directory (module docstring); None disables the cache.

    Raises:
        ValueError: On `sdf_resolution < 2` or `n_samples < 1`, at construction; with `cache_dir`, on the first clip
            if the backend has no cache identities or its environment probe fails.
    """
    cfg = cfg if cfg is not None else ShapeConfig()
    if sdf_resolution < 2 or n_samples < 1:
        raise ValueError(f"sdf_resolution must be >= 2 and n_samples >= 1, got {sdf_resolution} and {n_samples}")
    prep: dict[str, Any] = {"sdf_resolution": sdf_resolution, "sdf_padding": sdf_padding, "n_samples": n_samples,
                            "samples_seed": samples_seed}
    cache = None if cache_dir is None else Path(cache_dir)
    state: dict[str, Any] = {}              # with a cache: the unloaded backend and its keyer, built on the first clip

    def _keyer() -> ShapeCacheKeyer:
        if "keyer" not in state:
            state["backend"] = backend if backend is not None else make_shape_backend(cfg)
            state["keyer"] = ShapeCacheKeyer(cfg, state["backend"], prep)
        keyer: ShapeCacheKeyer = state["keyer"]
        return keyer

    # prewarm (`fastdo.engine.prewarm`): without a cache the backend can be loaded ahead of the first clip by
    # `prewarm()` on a runner thread; the first clip then takes that backend instead of loading its own.
    slot = PrewarmSlot("shape") if cache is None else None

    def load_model() -> ShapeBackend:
        prewarmed = slot.take() if slot is not None else None
        return prewarmed if prewarmed is not None else _load_backend(cfg)

    loader = load_model if cache is None else (lambda: _load_backend(cfg, state.get("backend")))
    ctx = WorkerContext(loader=loader, model=backend)

    def compute(clip_id: str, artifact_dir: Path, keyframe_index: int, frame_idx: int, image: np.ndarray,
                mask: np.ndarray) -> dict[str, Any]:
        model = ctx.model()
        result = model.reconstruct(image, mask)
        vertices, faces, scale = normalize_mesh(result.vertices, result.faces)
        raw = np.asarray(result.vertices, dtype=np.float64)
        center = (raw.min(axis=0) + raw.max(axis=0)) / 2.0
        vertices, faces, watertight = check_watertight(vertices, faces)
        sdf = mesh_sdf_grid(vertices, faces, resolution=sdf_resolution, padding=sdf_padding)
        points, normals = canonical_samples(vertices, faces, n_samples, samples_seed)

        _write_mesh(artifact_dir / MESH_FILE, vertices, faces)
        save_npz(artifact_dir / SDF_FILE, values=sdf.values.numpy(), origin=sdf.origin.numpy(),
                 step=np.array(sdf.step, dtype=np.float64))
        save_npz(artifact_dir / SAMPLES_FILE, points=points.numpy(), normals=normals.numpy())
        meta: dict[str, Any] = {
            "clip_id": clip_id, "keyframe_index": keyframe_index, "frame_idx": frame_idx, "backend": cfg.backend,
            "backend_class": type(model).__name__, "sparse_structure_steps": cfg.sparse_structure_steps,
            "structured_latent_steps": cfg.structured_latent_steps, "solver": cfg.solver, "flow_seed": cfg.seed,
            "mesh_postprocess": cfg.mesh_postprocess, "normalization_scale": scale,
            "normalization_center": center.tolist(), "watertight": watertight, "n_vertices": len(vertices),
            "n_faces": len(faces), **prep, "texture_path": result.texture_path,
            "phase_seconds": dict(result.phase_seconds)}
        if cfg.deterministic:
            meta["deterministic"] = True
        if cfg.spconv_kernel_pin:
            meta["spconv_kernels"] = "pinned"
        meta.update(non_default_sampler_options(cfg))    # recorded only when set (today's shape.json unchanged)
        if non_default_shape_options(cfg):        # nothing is recorded while every option is default
            meta.update(shape_options(cfg))
            stats = getattr(model, "last_postprocess_stats", None) or {}
            meta.update(_postprocess_stats(stats))
        logger.info("shape of %r on decode row %d: %d faces (watertight=%s), normalization scale %.4f", clip_id,
                    keyframe_index, len(faces), watertight, scale)
        return meta

    def shape_worker(clip_id: str, artifact_dir: Path) -> None:
        ctx.budget()
        if slot is not None:
            slot.wait()             # outside the model-load lock: a prewarm in progress finishes first
        artifact_dir = Path(artifact_dir)
        keyframe_index, frame_idx, image, mask = _read_keyframe(artifact_dir)
        if cache is None:
            _write_json(artifact_dir / SHAPE_META_FILE,
                        compute(clip_id, artifact_dir, keyframe_index, frame_idx, image, mask))
            return
        start = time.perf_counter()
        key, parts = _keyer().key(image, mask)
        clip_meta = {"clip_id": clip_id, "keyframe_index": keyframe_index, "frame_idx": frame_idx}
        entry = lookup(cache, key)
        if entry is not None:
            restore(entry.files, artifact_dir, (MESH_FILE, SDF_FILE, SAMPLES_FILE))
            stored = json.loads(entry.files[SHAPE_META_FILE])
            meta = clip_meta | stored | {"cache_hit": True, "cache_key": key,
                                         "cache_seconds": time.perf_counter() - start}
            _write_json(artifact_dir / SHAPE_META_FILE, meta)
            logger.info("shape of %r: cache hit %s", clip_id, key[:12])
            return
        lookup_seconds = time.perf_counter() - start
        meta = compute(clip_id, artifact_dir, keyframe_index, frame_idx, image, mask)
        start = time.perf_counter()
        stored = {k: v for k, v in meta.items() if k not in UNCACHED_META_KEYS}
        files = {name: (artifact_dir / name).read_bytes() for name in (MESH_FILE, SDF_FILE, SAMPLES_FILE)}
        files[SHAPE_META_FILE] = json.dumps(stored, indent=2).encode()
        try:
            store(cache, key, parts, files)
        except OSError as exc:              # the artifacts are complete; only the cache entry is lost
            logger.warning("shape of %r: could not store cache entry %s (%s)", clip_id, key[:12], exc)
        meta |= {"cache_hit": False, "cache_key": key, "cache_seconds": lookup_seconds + time.perf_counter() - start}
        _write_json(artifact_dir / SHAPE_META_FILE, meta)

    if slot is None:
        return shape_worker

    def prewarm() -> None:
        """Loads the backend now, exactly as the first clip would (`_load_backend`), without the model-load lock."""
        slot.prewarm(lambda: _load_backend(cfg, backend))

    return PrewarmableStageFn(shape_worker, prewarm)


class TemplateMesh:
    """A `ShapeBackend` returning one fixed mesh file (float64 vertices, int64 faces); `image` and `mask` are ignored."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def reconstruct(self, image: np.ndarray, mask: np.ndarray) -> ShapeResult:
        start = time.perf_counter()
        mesh = cast("trimesh.Trimesh", trimesh.load(str(self.path), process=False, force="mesh"))
        return ShapeResult(vertices=np.asarray(mesh.vertices, dtype=np.float64),
                           faces=np.asarray(mesh.faces, dtype=np.int64),
                           phase_seconds={"load": time.perf_counter() - start})

    def cache_identity(self) -> str:
        """The shape cache's checkpoint part: `"template:" + sha256` of the mesh file."""
        return "template:" + hashlib.sha256(self.path.read_bytes()).hexdigest()

    def cache_env_identity(self) -> str:
        """The shape cache's environment part: a template mesh has no model environment."""
        return "template"


def make_template_shape_worker(templates: Mapping[str, Path], sdf_resolution: int = 64, sdf_padding: float = 0.15,
                               n_samples: int = 2048, samples_seed: int = 0) -> StageFn:
    """The `shape` stage on a fixed template mesh per clip: `make_shape_worker(ShapeConfig(backend="template",
    mesh_postprocess=False), backend=TemplateMesh(templates[clip_id]), ...)`, so the normalization, watertight check,
    SDF, samples and `shape.json` are the shape worker's own code. Calibration clips only (module docstring).

    Raises:
        ValueError: On `sdf_resolution < 2` or `n_samples < 1`, at construction.
        KeyError: For a clip without a template.
    """
    if sdf_resolution < 2 or n_samples < 1:
        raise ValueError(f"sdf_resolution must be >= 2 and n_samples >= 1, got {sdf_resolution} and {n_samples}")
    table = {str(clip_id): Path(path) for clip_id, path in templates.items()}
    cfg = ShapeConfig(backend=TEMPLATE_BACKEND, mesh_postprocess=False)

    def template_shape_worker(clip_id: str, artifact_dir: Path) -> None:
        if clip_id not in table:
            raise KeyError(f"no template mesh registered for clip {clip_id!r}")
        worker = make_shape_worker(cfg, backend=TemplateMesh(table[clip_id]), sdf_resolution=sdf_resolution,
                                   sdf_padding=sdf_padding, n_samples=n_samples, samples_seed=samples_seed)
        worker(clip_id, artifact_dir)

    return template_shape_worker
