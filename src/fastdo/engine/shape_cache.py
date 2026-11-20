"""Per-clip shape cache: content-keyed `shape` stage artifacts, stored atomically.

Key (`ShapeCacheKeyer`): sha256 over length-framed parts, in this order (`name\\0 + 8-byte little-endian length +
payload` each):
- `schema`: `SHAPE_CACHE_SCHEMA`.
- `image`, `mask`: canonical JSON `{"dtype", "shape"}` (the actual shape), `\\0`, then the raw bytes. The keyframe must
  be uint8 and its mask bool (never cast, so distinct frames cannot collide).
- `shape_config`: canonical JSON of `dataclasses.asdict(cfg)` without the raw `sam3d_python` and `sam3d_pythonpath`
  (their resolved values are in `env`), and without any fast-path option or sampler option still at its default
  (`SHAPE_OPTION_DEFAULTS`, `SAMPLER_OPTION_DEFAULTS`), so a default request keeps the key it had before those options
  existed. Set, each one enters the key exactly as `spconv_kernel_pin` does.
- `mesh_prep`: canonical JSON of `sdf_resolution`, `sdf_padding`, `n_samples` and `samples_seed`.
- `checkpoint`: the backend's `cache_identity()` (weights and pipeline config).
- `env`: canonical JSON of the backend's `cache_env_identity()` (for SAM 3D: the resolved interpreter, its PYTHONPATH,
  the sam3d_objects sources, the installed distributions and the GPU; Check H1) and this process's numpy, torch and
  trimesh versions.
- `code`: `SHAPE_CODE_VERSION` and the bytes of `SHAPE_CODE_FILES` under `SHAPE_CODE_ROOT`.
A backend without both identity methods refuses the cache (`ValueError`). Identities are computed once per keyer.

Store (`<cache_dir>/<key[:2]>/<key>/`): `mesh.obj`, `sdf.npz`, `samples.npz`, `shape.json` and `manifest.json`
(`schema`, `key`, part digests, per-file sha256 and size, `created_utc`). `store` writes a private directory under
`.tmp/` (every file fsynced, manifest last) and renames it into place; a concurrent writer's entry wins. `lookup`
accepts an entry only if its manifest names the key and exactly `ENTRY_FILES` with matching sizes and sha256; any
other entry moves to `.corrupt/` and counts as a miss. Nothing is deleted in place.
"""

from __future__ import annotations

import dataclasses
import errno
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import shutil
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from fastdo.shape.backend import SAMPLER_OPTION_DEFAULTS, SHAPE_OPTION_DEFAULTS, ShapeConfig

__all__ = [
    "ENTRY_FILES",
    "MANIFEST_FILE",
    "SHAPE_CACHE_SCHEMA",
    "SHAPE_CODE_FILES",
    "SHAPE_CODE_ROOT",
    "SHAPE_CODE_VERSION",
    "UNCACHED_META_KEYS",
    "CacheEntry",
    "ShapeCacheKeyer",
    "canonical_json",
    "lookup",
    "restore",
    "store",
]

logger = logging.getLogger(__name__)

SHAPE_CACHE_SCHEMA = "fastdo-shape-cache/1"
SHAPE_CODE_VERSION = 1          # bump by hand for a semantic change outside SHAPE_CODE_FILES
SHAPE_CODE_ROOT = Path(__file__).resolve().parents[1]      # the fastdo package directory
SHAPE_CODE_FILES: tuple[str, ...] = tuple(sorted((
    "engine/shape_cache.py", "engine/worker_support.py", "engine/workers_shape.py", "shape/backend.py",
    "shape/mesh_utils.py", "shape/sam3d_backend.py", "shape/sam3d_determinism.py", "shape/sam3d_worker.py", "solver/sdf.py",
    "shape/sam3d_spconv_pin.py", "shape/fast_postprocess.py", "shape/wall_qc.py", "shape/sam3d_cuda_graph.py",
    "shape/sam3d_cuda_graph_slat.py", "shape/sam3d_multistep.py")))
ENTRY_FILES: tuple[str, ...] = ("mesh.obj", "samples.npz", "sdf.npz", "shape.json")
MANIFEST_FILE = "manifest.json"
# shape.json keys that describe the clip or the lookup, never the cached shape (Check H3).
UNCACHED_META_KEYS: tuple[str, ...] = ("clip_id", "keyframe_index", "frame_idx", "cache_hit", "cache_key",
                                       "cache_seconds")
_CFG_ENV_FIELDS = ("sam3d_python", "sam3d_pythonpath")


def _at_option_default(name: str, value: Any) -> bool:
    """Whether `name` is a fast-path option or a sampler option still at its default (then it is
    left out of the key)."""
    for defaults in (SHAPE_OPTION_DEFAULTS, SAMPLER_OPTION_DEFAULTS):
        if name in defaults and value == defaults[name]:
            return True
    return False


def canonical_json(obj: Any) -> bytes:
    """Sorted-key, compact, ASCII JSON; NaN refused."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()


def _array_part(array: np.ndarray, dtype: type, what: str) -> bytes:
    if array.dtype != dtype:
        raise ValueError(f"the shape cache needs a {np.dtype(dtype).name} keyframe {what}, got {array.dtype}")
    header = canonical_json({"dtype": np.dtype(dtype).name, "shape": list(array.shape)})
    return header + b"\0" + np.ascontiguousarray(array).tobytes()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _frame(h: Any, name: str, payload: bytes) -> None:
    h.update(name.encode() + b"\0" + len(payload).to_bytes(8, "little") + payload)


def _code_identity() -> str:
    h = hashlib.sha256()
    for rel in sorted(SHAPE_CODE_FILES):
        _frame(h, rel, (SHAPE_CODE_ROOT / rel).read_bytes())
    return f"{SHAPE_CODE_VERSION}:{h.hexdigest()}"


def _engine_env() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for name in ("numpy", "torch", "trimesh"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


class ShapeCacheKeyer:
    """The cache key of one worker: backend identities and code hash once, then one key per keyframe.

    Args:
        cfg: The worker's shape settings.
        backend: The (unloaded) backend; needs `cache_identity()` and `cache_env_identity()`.
        mesh_prep: `sdf_resolution`, `sdf_padding`, `n_samples` and `samples_seed`.
    """

    def __init__(self, cfg: ShapeConfig, backend: Any, mesh_prep: Mapping[str, Any]) -> None:
        self.cfg, self.backend, self.mesh_prep = cfg, backend, dict(mesh_prep)
        self._static: list[tuple[str, bytes]] | None = None

    def _static_parts(self) -> list[tuple[str, bytes]]:
        if self._static is None:
            identity, env = (getattr(self.backend, name, None) for name in ("cache_identity", "cache_env_identity"))
            if not callable(identity) or not callable(env):
                raise ValueError(f"backend {type(self.backend).__name__} does not support the shape cache (it needs "
                                 "cache_identity() and cache_env_identity())")
            config = {k: v for k, v in dataclasses.asdict(self.cfg).items()
                      if k not in _CFG_ENV_FIELDS and not _at_option_default(k, v)}
            self._static = [
                ("shape_config", canonical_json(config)), ("mesh_prep", canonical_json(self.mesh_prep)),
                ("checkpoint", str(identity()).encode()),
                ("env", canonical_json({"backend": str(env()), "engine": _engine_env()})),
                ("code", _code_identity().encode())]
        return self._static

    def key(self, image: np.ndarray, mask: np.ndarray) -> tuple[str, dict[str, str]]:
        """`(key, {part name: sha256 of its payload})`.

        Raises:
            ValueError: On a non-uint8 image, a non-bool mask, or a backend without cache identities.
        """
        parts = [("schema", SHAPE_CACHE_SCHEMA.encode()), ("image", _array_part(image, np.uint8, "image")),
                 ("mask", _array_part(mask, np.bool_, "mask")), *self._static_parts()]
        h = hashlib.sha256()
        for name, payload in parts:
            _frame(h, name, payload)
        return h.hexdigest(), {name: _sha256(payload) for name, payload in parts}


@dataclasses.dataclass(frozen=True)
class CacheEntry:
    """A validated entry: its directory and the verified bytes of every `ENTRY_FILES` file."""

    key: str
    path: Path
    files: dict[str, bytes]


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _quarantine(cache_dir: Path, entry: Path, key: str, reason: str) -> None:
    logger.warning("shape cache entry %s is invalid (%s); moving it to .corrupt/ and recomputing", entry, reason)
    target = Path(cache_dir) / ".corrupt" / f"{key}.{uuid.uuid4().hex}"
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        os.rename(entry, target)
    except OSError as exc:               # ENOENT: a concurrent reader moved it first; else a logged miss
        if exc.errno != errno.ENOENT:
            logger.warning("could not quarantine shape cache entry %s (%s); treating it as a miss", entry, exc)


def lookup(cache_dir: Path, key: str) -> CacheEntry | None:
    """The valid entry for `key`, or None (absent, or invalid and quarantined)."""
    entry = Path(cache_dir) / key[:2] / key
    if not entry.is_dir():
        return None
    try:
        manifest = json.loads((entry / MANIFEST_FILE).read_text())
        listed = manifest["files"]
        if manifest.get("schema") != SHAPE_CACHE_SCHEMA or manifest.get("key") != key or sorted(listed) != sorted(
                ENTRY_FILES):
            raise ValueError(f"manifest schema, key or file set mismatch (files {sorted(listed)})")
        files = {name: (entry / name).read_bytes() for name in ENTRY_FILES}
        for name, data in files.items():
            if len(data) != listed[name]["bytes"] or _sha256(data) != listed[name]["sha256"]:
                raise ValueError(f"{name} does not match its manifest")
    except FileNotFoundError as exc:
        if not entry.is_dir():           # quarantined concurrently: a plain miss (Check L3)
            return None
        _quarantine(cache_dir, entry, key, f"missing {Path(str(exc.filename)).name}")
        return None
    except (OSError, ValueError, KeyError, TypeError) as exc:
        _quarantine(cache_dir, entry, key, f"{type(exc).__name__}: {exc}")
        return None
    return CacheEntry(key=key, path=entry, files=files)


def store(cache_dir: Path, key: str, parts: Mapping[str, str], files: Mapping[str, bytes]) -> Path:
    """Stores `files` (exactly `ENTRY_FILES`) under `key` atomically and returns the entry directory; an entry that
    already exists (a concurrent writer) is kept and `files` are discarded.

    Raises:
        ValueError: If `files` is not exactly `ENTRY_FILES`.
        OSError: If writing or renaming fails (the private tmp directory is removed first).
    """
    if sorted(files) != sorted(ENTRY_FILES):
        raise ValueError(f"a shape cache entry holds exactly {ENTRY_FILES}, got {sorted(files)}")
    cache_dir = Path(cache_dir)
    entry = cache_dir / key[:2] / key
    tmp = cache_dir / ".tmp" / f"{key}.{os.getpid()}.{uuid.uuid4().hex}"
    tmp.mkdir(parents=True)
    try:
        manifest = {"schema": SHAPE_CACHE_SCHEMA, "key": key, "parts": dict(parts),
                    "files": {name: {"sha256": _sha256(data), "bytes": len(data)} for name, data in files.items()},
                    "created_utc": datetime.now(UTC).isoformat()}
        for name, data in [*files.items(), (MANIFEST_FILE, json.dumps(manifest, indent=2).encode())]:
            with (tmp / name).open("wb") as f:
                f.write(data)
                os.fsync(f.fileno())
        _fsync_dir(tmp)
        entry.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.rename(tmp, entry)
        except OSError as exc:
            if exc.errno not in (errno.ENOTEMPTY, errno.EEXIST):
                raise
            logger.info("shape cache entry %s already exists (concurrent writer); keeping it", entry)
            shutil.rmtree(tmp, ignore_errors=True)
            return entry
        _fsync_dir(entry.parent)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return entry


def restore(files: Mapping[str, bytes], artifact_dir: Path, names: Sequence[str]) -> None:
    """Writes `files[name]` for each of `names`, in order, to `<artifact_dir>/<name>` via `<name>.tmp` and
    `os.replace`."""
    for name in names:
        target = Path(artifact_dir) / name
        tmp = target.with_name(name + ".tmp")
        try:
            tmp.write_bytes(files[name])
            os.replace(tmp, target)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
