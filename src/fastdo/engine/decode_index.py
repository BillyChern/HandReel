"""The persisted decode index: Gate A's evidence once the decoded pixels are gone.

`decode.npz` holds the window's frame indices and its PIXELS in one file. Two separate rules delete it:
the Ego4D licence requires decoded-pixel caches to be removed after processing, and the shard-boundary
prune removes it. Either way Gate A criterion 2 -- "the recorded `frame_idx` equals
`range(start, stop, s_v)`" -- loses the file it reads.

So the indices are written out first, into `decode_index_audit.json` beside the artifact root, and the
gate falls back to them. That fallback existed, correct and tested, for some time before this module:
nothing wrote the file, the reader treats a missing file as "no entry" rather than an error, and the test
supplied its input as a dict literal. This module is the writer, kept here rather than in the study
script so that the pipeline can call it too -- a writer only a study can reach is a writer a run will not
have used.

`contiguous_stride` is the field that makes the fallback as strong as the pixels. The pixel path compares
`frame_idx` to `arange(start, stop, stride)` ELEMENTWISE; the recorded path sees `n`, `first`, `last` and
`stride`, four scalars a non-uniform run can satisfy exactly ([0, 1, 8, 9] against [0, 3, 6, 9]).
Asserting every gap equals decode's OWN stride member -- not `diff[0]`, so a uniformly wrong run cannot
certify itself -- is what makes the two paths answer the same question.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["DECODE_FILE", "DECODE_INDEX_FILE", "build_index_audit", "index_entry", "load_index_audit",
           "write_index_audit"]

logger = logging.getLogger(__name__)

DECODE_FILE = "decode.npz"
DECODE_INDEX_FILE = "decode_index_audit.json"


def index_entry(artifact_dir: Path) -> dict[str, Any]:
    """The `decode.npz` fields Gate A's recorded path reads, or `{"error": ...}`.

    Reads `frame_idx` and `stride` only -- never `frames`. The Ego4D licence forbids third-party access
    to its pixels, and `save_npz` writes an uncompressed archive, so reading two members does not touch
    the third.
    """
    path = Path(artifact_dir) / DECODE_FILE
    if not path.is_file():
        return {"error": "no decode.npz"}
    try:
        with np.load(path) as data:
            if "frame_idx" not in data:
                return {"error": "decode.npz has no frame_idx"}
            idx = np.asarray(data["frame_idx"], dtype=np.int64)
            stride = int(data["stride"]) if "stride" in data else None
    except Exception as exc:  # noqa: BLE001 - one unreadable window must not abort the sweep
        return {"error": f"{type(exc).__name__}: {exc}"}
    gaps = np.diff(idx)
    contiguous = bool(idx.size <= 1 or (gaps == (stride if stride is not None else gaps[0])).all())
    return {"n": int(idx.size), "first": int(idx[0]) if idx.size else None,
            "last": int(idx[-1]) if idx.size else None, "stride": stride,
            "contiguous_stride": contiguous}


def build_index_audit(artifact_root: Path, clip_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
    """`clip_id -> index_entry(...)` for every clip that has an artifact directory.

    Call this BEFORE anything deletes `decode.npz`; it cannot be reconstructed afterwards. A clip with no
    artifact directory is omitted entirely (nothing was decoded); a clip whose decode is unreadable gets
    an `error` entry, which `load_index_audit` drops on the way back in -- so a damaged window reads as
    absent rather than as clean, and `engine.prune` refuses to prune it.
    """
    root = Path(artifact_root)
    return {clip_id: index_entry(root / clip_id) for clip_id in clip_ids if (root / clip_id).is_dir()}


def write_index_audit(artifact_root: Path, clip_ids: Sequence[str], path: Path | None = None) -> Path:
    """Persist `build_index_audit` to `path` (default: `DECODE_INDEX_FILE` beside `artifact_root`)."""
    path = Path(path) if path is not None else Path(artifact_root).parent / DECODE_INDEX_FILE
    audit = build_index_audit(artifact_root, clip_ids)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(audit, indent=1, sort_keys=True))
    usable = sum(1 for entry in audit.values() if "error" not in entry)
    logger.info("wrote %s: %d of %d windows have a usable decode index", path.name, usable, len(audit))
    return path


def load_index_audit(path: Path) -> dict[str, dict[str, Any]]:
    """A persisted `DECODE_INDEX_FILE`, or `{}` when absent. Entries carrying `error` are dropped."""
    path = Path(path)
    if not path.is_file():
        return {}
    loaded: Mapping[str, Any] = json.loads(path.read_text())
    return {k: v for k, v in loaded.items() if isinstance(v, dict) and "error" not in v}
