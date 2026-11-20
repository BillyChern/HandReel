"""Shard pruning: retire the heavy per-window artifacts, keep what they prove.

The pruning rule:

    **After a shard's qc:** delete decode.npz, mask.npz and geometry.npz, after writing their sha256 and
    the per-row mask area to `pruned.json`.
    **Kept:** manifest.parquet, profile.json, gate.json, solve.npz, solve.json, qc.json, clip_record.npz,
    hands.npz, shape files, tracks.npz.

Three things here are not obvious and are each pinned by a test:

1. **Pruning deletes Gate A's evidence.** `decode.npz` carries criterion 2's `frame_idx`, so a shard pruned
   before the index audit is written is a shard whose window discipline can never be checked again --
   silently, because the gate's fallback reads a missing file as an absent entry rather than an error.
   `prune_clip` therefore REFUSES without a recorded index entry for that clip. The ordering is a code
   invariant here rather than a note in a runbook, because the failure is invisible at the call site.

2. **The record is written and flushed BEFORE the first unlink.** A crash between the two loses the
   artifacts and their provenance together; in this order it can only lose disk space.

3. **Nothing here runs automatically.** The functions are callable, and no stage calls them. Deletion is
   irreversible and the trigger is a shard boundary the operator owns, so wiring it to a pipeline is a
   separate decision from having the mechanism -- and `dry_run=True` is the default.

The rule's 60 GB cap is stated against a 720p window (~1.5 GB), and whether the run is 720p at all is
open. So `shard_cap_report` takes the per-window size as an argument and measures the
live bytes on disk instead of assuming either.
"""
from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["DECODE_FILE", "PRUNABLE_FILES", "PRUNED_FILE", "SHARD_LIVE_BYTES_CAP", "WINDOWS_PER_SHARD",
           "PrunePrevented", "clip_prune_record", "live_bytes", "mask_row_areas", "needs_index_entry",
           "prune_clip", "prune_shard", "shard_cap_report"]

logger = logging.getLogger(__name__)

PRUNED_FILE = "pruned.json"
#: The one prunable file that is also Gate A's evidence -- the reason the index guard exists at all.
DECODE_FILE = "decode.npz"
#: The rule's three heavy artifacts, in the order it lists them.
PRUNABLE_FILES: tuple[str, ...] = (DECODE_FILE, "mask.npz", "geometry.npz")
#: The rule: "a shard of 40 windows, at most 60 GB live". The byte figure is decimal GB, as the rule's sizes are.
WINDOWS_PER_SHARD = 40
SHARD_LIVE_BYTES_CAP = 60 * 1000**3


class PrunePrevented(RuntimeError):
    """A prune was refused because deleting would have destroyed evidence nothing else holds.

    Raised, not returned: a caller that ignores a returned "did nothing" report deletes the next clip
    anyway, and the artifacts are gone by the time anyone reads the report.
    """


def needs_index_entry(artifact_dir: Path, files: Sequence[str] = PRUNABLE_FILES) -> bool:
    """Whether pruning `artifact_dir` would destroy Gate A's evidence, and so needs a persisted index.

    True exactly when this prune would delete an existing `decode.npz`. The guard binds to that file
    rather than to the clip, because the thing being protected is criterion 2's `frame_idx` and nothing
    else: a window that never decoded has no evidence to lose, and refusing it would block a shard over a
    clip that is not in it. Deleting `mask.npz` beside an absent decode tells Gate A nothing it did not
    already know.
    """
    return DECODE_FILE in files and (Path(artifact_dir) / DECODE_FILE).is_file()


def _usable(entry: Mapping[str, Any] | None) -> bool:
    """An index entry that actually certifies the window: present, non-empty, and not an error record."""
    return bool(entry) and "error" not in entry


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def mask_row_areas(mask_path: Path) -> dict[str, Any]:
    """Per-row mask area from `mask.npz`: `{"frame_idx": [...], "area_px": [...]}`, or `{"error": ...}`.

    "Per-row" is per decoded row, not per frame of the source video: `mask.npz` holds `masks` (N,H,W) bool
    beside `frame_idx` (N,), and the area is the true-pixel count of each row. This is the one summary
    the rule preserves through the deletion, so it is computed from the array rather than copied from any
    earlier stage's summary of it.
    """
    mask_path = Path(mask_path)
    if not mask_path.is_file():
        return {"error": "no mask.npz"}
    try:
        with np.load(mask_path) as z:
            if "masks" not in z:
                return {"error": "mask.npz has no masks"}
            masks = np.asarray(z["masks"])
            frame_idx = np.asarray(z["frame_idx"], dtype=np.int64) if "frame_idx" in z else None
        areas = masks.reshape(masks.shape[0], -1).sum(axis=1, dtype=np.int64) if masks.ndim == 3 else None
    except Exception as exc:  # noqa: BLE001 - a damaged mask must not block the record of the rest
        return {"error": f"{type(exc).__name__}: {exc}"}
    if areas is None:
        return {"error": f"masks has ndim {masks.ndim}, expected 3"}
    out: dict[str, Any] = {"area_px": [int(a) for a in areas]}
    if frame_idx is not None:
        out["frame_idx"] = [int(i) for i in frame_idx]
    return out


def clip_prune_record(artifact_dir: Path, files: Sequence[str] = PRUNABLE_FILES) -> dict[str, Any]:
    """What the pruning rule requires be written down before the deletion: sha256, size, and the per-row mask area.

    Files that are absent are recorded as absent rather than skipped, so a later reader can tell "this
    window had no geometry" from "this record was written by an older tool".
    """
    artifact_dir = Path(artifact_dir)
    record: dict[str, Any] = {"files": {}, "mask_area": mask_row_areas(artifact_dir / "mask.npz")}
    for name in files:
        path = artifact_dir / name
        if path.is_file():
            record["files"][name] = {"sha256": _sha256(path), "bytes": path.stat().st_size}
        else:
            record["files"][name] = {"absent": True}
    return record


def prune_clip(artifact_dir: Path, *, index_entry: Mapping[str, Any] | None,
               files: Sequence[str] = PRUNABLE_FILES, dry_run: bool = True) -> dict[str, Any]:
    """Write `pruned.json` for one window, then delete `files`. Returns the record that was written.

    Args:
        artifact_dir: the window's artifact directory.
        index_entry: this clip's entry from `build_index_audit`, already
            persisted. `None`, empty, or an entry carrying `error` refuses the prune whenever
            `decode.npz` is present -- see point 1 of the module docstring. Passed in rather than read
            here so that this module stays a leaf.
        dry_run: when true (the default) the record is returned and NOTHING is written or deleted.

    Raises:
        PrunePrevented: if `index_entry` does not certify that Gate A can still be checked afterwards.
    """
    artifact_dir = Path(artifact_dir)
    if needs_index_entry(artifact_dir, files) and not _usable(index_entry):
        why = ("is an error entry ({!r})".format(index_entry["error"])
               if index_entry and "error" in index_entry else "is missing")
        raise PrunePrevented(
            f"{artifact_dir.name}: the persisted decode index {why}; pruning would delete Gate A "
            f"criterion 2's only evidence. Write the index audit first "
            f"(s2_gate_a.write_index_audit).")
    record = clip_prune_record(artifact_dir, files)
    # None once the guard allows it: a window with no decode.npz has no index to record.
    record["index_entry"] = dict(index_entry) if index_entry else None
    if dry_run:
        record["dry_run"] = True
        return record
    path = artifact_dir / PRUNED_FILE
    path.write_text(json.dumps(record, indent=1, sort_keys=True))      # (2) record first, then delete
    removed = []
    for name in files:
        target = artifact_dir / name
        if target.is_file():
            target.unlink()
            removed.append(name)
    logger.info("pruned %s: removed %s", artifact_dir.name, ", ".join(removed) or "nothing")
    record["removed"] = removed
    return record


def prune_shard(artifact_root: Path, clip_ids: Sequence[str], index_audit: Mapping[str, Mapping[str, Any]],
                *, files: Sequence[str] = PRUNABLE_FILES, dry_run: bool = True) -> dict[str, Any]:
    """`prune_clip` over a shard, refusing the WHOLE shard if any clip lacks its index entry.

    All-or-nothing on purpose: a partial prune leaves a shard in a state no one can describe, and the
    check is cheap compared with the deletion it guards.
    """
    root = Path(artifact_root)
    missing = [c for c in clip_ids
               if needs_index_entry(root / c, files) and not _usable(index_audit.get(c))]
    if missing:
        raise PrunePrevented(f"{len(missing)} of {len(clip_ids)} clips have no usable persisted decode "
                             f"index (first: {missing[0]!r}); write the index audit before pruning.")
    records = {c: prune_clip(root / c, index_entry=index_audit.get(c), files=files, dry_run=dry_run)
               for c in clip_ids if (root / c).is_dir()}
    freed = sum(f.get("bytes", 0) for r in records.values() for f in r["files"].values())
    return {"clips": records, "freed_bytes": int(freed), "dry_run": dry_run}


def live_bytes(artifact_root: Path, clip_ids: Sequence[str],
               files: Sequence[str] = PRUNABLE_FILES) -> int:
    """Bytes of `files` currently on disk across `clip_ids` -- measured, not estimated from a resolution."""
    root = Path(artifact_root)
    return int(sum((root / c / n).stat().st_size
                   for c in clip_ids for n in files if (root / c / n).is_file()))


def shard_cap_report(artifact_root: Path, clip_ids: Sequence[str], *,
                     windows_per_shard: int = WINDOWS_PER_SHARD,
                     cap_bytes: int = SHARD_LIVE_BYTES_CAP) -> dict[str, Any]:
    """Whether the live heavy artifacts respect the rule's cap: `<= windows_per_shard` windows, `<= cap_bytes`.

    Both bounds are arguments because the rule derives its 60 GB from a 720p window, and whether the run is
    720p is open -- hard-coding a figure whose premise is unresolved would make this read as settled. The
    byte figure is measured off the disk, so it stays honest whatever the resolution.
    """
    n, used = len(clip_ids), live_bytes(artifact_root, clip_ids)
    reasons = []
    if n > windows_per_shard:
        reasons.append(f"{n} windows exceeds the shard size {windows_per_shard}")
    if used > cap_bytes:
        reasons.append(f"{used / 1000**3:.1f} GB live exceeds the cap {cap_bytes / 1000**3:.1f} GB")
    return {"ok": not reasons, "reasons": reasons, "windows": n, "live_bytes": used,
            "cap_bytes": int(cap_bytes), "windows_per_shard": int(windows_per_shard)}
