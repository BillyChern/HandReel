"""Shared plumbing for the engine's stage workers: artifact npz I/O, row matching and worker lifecycle.

Artifacts are uncompressed `np.savez` archives written to a tmp sibling and moved into place with
`os.replace`, so a killed worker never leaves a truncated `<stage>.npz` that a resumed run would trust; they
load without pickle.

`WorkerContext` is the state every worker closure holds. It applies the per-process CUDA cap
(`fastdo.bench.gpu_budget.apply_gpu_budget`) once, before the worker's first allocation, and loads the
worker's model once, on first use, then reuses it for every later clip. A factory that builds a
context allocates and loads nothing. Model loads are serialized across every context in the process
(`_LOAD_LOCK`): clip-batched stages load one model per `run_clips_concurrently` slot inside pool threads, and
the loaders touch process-wide state (sam2 swaps Hydra's GlobalHydra; `torch.hub.load` edits `sys.path` and
writes its cache directory). Serial loads also keep two load-time memory peaks from stacking on the GPU.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from fastdo.bench import gpu_budget
from fastdo.engine.npz_rows import read_npz_rows

__all__ = [
    "DECODE_FILE",
    "SOLVE_FILE",
    "BatchedWorker",
    "WorkerContext",
    "load_npz",
    "read_decode",
    "read_decode_rows",
    "read_frame_idx",
    "require_artifact",
    "rows_by_frame_index",
    "save_npz",
]

DECODE_FILE = "decode.npz"
SOLVE_FILE = "solve.npz"   # here, not in workers_qc: the contact stage reads it too, and importing
                           # workers_qc from workers_contact would make a cycle (qc consumes contact.npz)
_LOAD_LOCK = threading.Lock()  # one model load at a time, process-wide (see the module docstring)


def save_npz(path: Path, **arrays: np.ndarray) -> None:
    """Writes `arrays` to `path` with uncompressed `np.savez`, via a tmp sibling and `os.replace`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with tmp.open("wb") as f:
            np.savez(f, **arrays)  # type: ignore[arg-type]  # numpy's stub types `allow_pickle` into **kwds
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def require_artifact(path: Path, stage: str) -> Path:
    """`path` if it is a file, else `FileNotFoundError` naming the stage that should have written it."""
    if not Path(path).is_file():
        raise FileNotFoundError(f"{path} not found; run the {stage!r} stage for this clip first")
    return Path(path)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    """Every array of an npz archive, loaded eagerly and without pickle."""
    with np.load(Path(path), allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def read_decode(artifact_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """`(frames (N,H,W,3) uint8, frame_idx (N,) int64)` from `<artifact_dir>/decode.npz`."""
    path = require_artifact(Path(artifact_dir) / DECODE_FILE, "decode")
    with np.load(path, allow_pickle=False) as data:
        return np.asarray(data["frames"]), np.asarray(data["frame_idx"], dtype=np.int64)


def read_decode_rows(path: Path, rows: Sequence[int] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`(frames[rows] (len(rows),H,W,3) uint8, frame_idx (N,) int64)` of the `decode.npz` at `path`, reading only
    those rows of `frames`: byte-identical to `read_decode`'s `frames[rows]`, for stages that need a few
    rows of a 0.6-1.5 GB window. `rows` follows `fastdo.engine.npz_rows.read_npz_rows`."""
    frames = read_npz_rows(Path(path), "frames", rows)
    with np.load(Path(path), allow_pickle=False) as data:
        return frames, np.asarray(data["frame_idx"], dtype=np.int64)


def read_frame_idx(artifact_dir: Path) -> np.ndarray:
    """`(N,)` int64 source frame indices from `<artifact_dir>/decode.npz`, without reading the frames."""
    path = require_artifact(Path(artifact_dir) / DECODE_FILE, "decode")
    with np.load(path, allow_pickle=False) as data:
        return np.asarray(data["frame_idx"], dtype=np.int64)


def rows_by_frame_index(source_idx: np.ndarray, wanted: np.ndarray, source: str) -> np.ndarray:
    """Row positions in `source_idx` holding each entry of `wanted`, in `wanted`'s order.

    Args:
        source_idx: `(T,)` source frame index of every row of some artifact.
        wanted: `(N,)` frame indices to select.
        source: Name of the artifact, for error messages.

    Raises:
        ValueError: If `source_idx` repeats a frame index, or lacks an entry of `wanted`; the message names
            the first missing frame index.
    """
    source_idx = np.asarray(source_idx, dtype=np.int64).reshape(-1)
    wanted = np.asarray(wanted, dtype=np.int64).reshape(-1)
    if np.unique(source_idx).size != source_idx.size:
        raise ValueError(f"{source} repeats a frame index; its rows cannot be matched unambiguously")
    order = np.argsort(source_idx, kind="stable")
    sorted_idx = source_idx[order]
    pos = np.minimum(np.searchsorted(sorted_idx, wanted), max(source_idx.size - 1, 0))
    found = sorted_idx[pos] == wanted if source_idx.size else np.zeros(wanted.shape, dtype=bool)
    if not found.all():
        missing = wanted[~found]
        raise ValueError(f"{source} has no row for frame index {int(missing[0])} "
                         f"({missing.size} of {wanted.size} requested frame indices missing)")
    return order[pos]


class WorkerContext:
    """One worker's lifecycle: the CUDA cap applied once before anything else, and its model loaded once.

    Args:
        loader: Zero-argument callable returning the worker's model; called at most once, on the first
            `model()` call. `None` for workers without a model.
        model: An already-built model (tests inject fakes); `loader` is then never called.
    """

    def __init__(self, loader: Callable[[], Any] | None = None, model: Any = None) -> None:
        self._loader = loader
        self._model = model
        self._budget_applied = False
        self._lock = threading.Lock()

    def budget(self) -> None:
        """Applies the per-process CUDA cap if this worker has not done so yet."""
        with self._lock:
            self._apply_budget_locked()

    def model(self) -> Any:
        """The worker's model, loaded on first call (after the CUDA cap, one load process-wide at a time) and reused.

        Raises:
            RuntimeError: If the context has neither a model nor a loader.
        """
        with self._lock:
            self._apply_budget_locked()
            if self._model is None:
                if self._loader is None:
                    raise RuntimeError("this worker has no model and no model loader")
                with _LOAD_LOCK:
                    self._model = self._loader()
            return self._model

    def _apply_budget_locked(self) -> None:
        if not self._budget_applied:
            gpu_budget.apply_gpu_budget()
            self._budget_applied = True


class BatchedWorker:
    """A frame-batched stage worker: a `StageFn` whose `run_many` shares forward passes across clips.

    Calling the worker as a `StageFn` runs the one-clip case. `run_many` succeeds or fails as a unit: per-clip
    failure isolation belongs to the caller (the orchestrator runs one clip at a time).
    """

    def __call__(self, clip_id: str, artifact_dir: Path) -> None:
        self.run_many([(clip_id, Path(artifact_dir))])

    def run_many(self, clips: Sequence[tuple[str, Path]]) -> None:
        """Processes `(clip_id, artifact_dir)` pairs, writing each clip's artifact."""
        raise NotImplementedError
