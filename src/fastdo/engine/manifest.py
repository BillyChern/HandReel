"""Parquet-backed clip manifest: the single source of truth for per-clip, per-stage state.

Each row is one clip; every pipeline stage contributes a `{stage}_state` / `{stage}_seconds` /
`{stage}_error` / `{stage}_batch_id` column block. `batch_id` names the `batches.jsonl` record of the batched
runner's final attempt (`fastdo.engine.batched_runner`); it is `""` after `create`, for `run_clip` marks and on
legacy manifests saved without the column. This is the durable, cross-clip aggregation surface the orchestrator
(`fastdo.engine.orchestrator`) and the clips-per-GPU-hour report read from -- stage workers
themselves never touch it, only the orchestrator does (via `.mark`).
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from fastdo.engine.refusal import is_refusal

__all__ = ["STAGE_NAMES", "ClipManifest", "ClipStatus"]

# `intrinsics` precedes `hands` and `geometry`: the geometry stage's hand-anchored metric
# scale needs both the camera matrix and the hand joints. Manifests saved under the older 11-stage order
# still load: a stage with no column in a row reads as "pending" (`status`, `pending`, `_normalized_rows`).
STAGE_NAMES: tuple[str, ...] = (
    "decode",
    "encoder",
    "gate",
    "mask",
    "shape",
    "intrinsics",
    "hands",
    "geometry",
    "tracks",
    "solve",
    "qc",
    "retarget",
)

_VALID_STATES = frozenset({"pending", "done", "failed"})


@dataclass(frozen=True)
class ClipStatus:
    """One clip's recorded state for one stage."""

    clip_id: str
    stage: str
    state: str
    seconds: float
    error: str
    batch_id: str = ""


def _stage_columns(stage: str) -> tuple[str, str, str, str]:
    return f"{stage}_state", f"{stage}_seconds", f"{stage}_error", f"{stage}_batch_id"


class ClipManifest:
    """Parquet-backed table of per-clip, per-stage processing state.

    Rows are kept in memory as plain dicts (`clip_id -> row`); pyarrow is only used at the
    `save`/`load` boundary. Any stage name may be marked, not just `STAGE_NAMES` -- the engine
    orchestrator is exercised in tests with fake stage names ("a", "b", ...), and this manifest
    must support that without change.
    """

    def __init__(self, rows: dict[str, dict[str, Any]]) -> None:
        self._rows = rows

    @classmethod
    def create(
        cls,
        clip_ids: Sequence[str],
        sources: Sequence[str],
        n_frames: Sequence[int],
    ) -> ClipManifest:
        """Build a manifest with every `STAGE_NAMES` stage `"pending"` for every clip."""
        if not (len(clip_ids) == len(sources) == len(n_frames)):
            raise ValueError("clip_ids, sources, and n_frames must have equal length")
        rows: dict[str, dict[str, Any]] = {}
        for clip_id, source, n in zip(clip_ids, sources, n_frames, strict=True):
            row: dict[str, Any] = {"clip_id": clip_id, "source": source, "n_frames": int(n)}
            for stage in STAGE_NAMES:
                state_col, seconds_col, error_col, batch_col = _stage_columns(stage)
                row[state_col] = "pending"
                row[seconds_col] = 0.0
                row[error_col] = ""
                row[batch_col] = ""
            rows[clip_id] = row
        return cls(rows)

    @classmethod
    def load(cls, path: Path) -> ClipManifest:
        """Load a manifest previously written by `.save`."""
        table = pq.read_table(path)
        rows: dict[str, dict[str, Any]] = {}
        for record in table.to_pylist():
            rows[record["clip_id"]] = record
        return cls(rows)

    def clip_ids(self) -> list[str]:
        """All clip ids, in the manifest's own (insertion) order."""
        return list(self._rows.keys())

    def n_frames(self, clip_id: str) -> int:
        return int(self._rows[clip_id]["n_frames"])

    def mark(self, clip_id: str, stage: str, state: str, seconds: float, error: str = "",
             batch_id: str = "") -> None:
        """Record `stage`'s outcome for `clip_id`. `state` must be `"pending"`/`"done"`/`"failed"`."""
        if state not in _VALID_STATES:
            raise ValueError(f"invalid state {state!r}, expected one of {sorted(_VALID_STATES)}")
        if clip_id not in self._rows:
            raise KeyError(f"unknown clip_id {clip_id!r}")
        state_col, seconds_col, error_col, batch_col = _stage_columns(stage)
        row = self._rows[clip_id]
        row[state_col] = state
        row[seconds_col] = float(seconds)
        row[error_col] = error
        row[batch_col] = batch_id

    def status(self, clip_id: str, stage: str) -> ClipStatus:
        """Current `ClipStatus` for `clip_id` on `stage` (defaults to `"pending"`/0.0/""/"" if unmarked)."""
        state_col, seconds_col, error_col, batch_col = _stage_columns(stage)
        row = self._rows[clip_id]
        return ClipStatus(
            clip_id=clip_id,
            stage=stage,
            state=row.get(state_col, "pending"),
            seconds=float(row.get(seconds_col, 0.0)),
            error=row.get(error_col, ""),
            batch_id=row.get(batch_col) or "",
        )

    def pending(self, stage: str) -> list[str]:
        """Clip ids not yet `"done"` on `stage` (unmarked and `"failed"` rows both count as pending,
        so a previously failed stage is retried on the next `run_manifest` pass)."""
        state_col = _stage_columns(stage)[0]
        return [cid for cid, row in self._rows.items() if row.get(state_col, "pending") != "done"]

    def refused(self, stages: Sequence[str]) -> set[str]:
        """Clip ids a stage DECLINED on the merits, rather than failed on.

        A refusal finishes a clip: no later stage runs, and re-running the refusing stage buys an identical refusal,
        because the verdict is a deterministic function of artifacts already on disk. `pending` cannot tell the two
        apart -- it sees only that the row is not `done` -- so on every resume the refused clips were re-run. Measured
        from one run's own manifest: **13 refusals, every one at `prompt`, 182.7 s in total** (median 11.9 s each),
        re-spent on every resume to obtain 13 identical refusals.

        Any refused stage finishes the CLIP, so this is a per-clip question and not a per-stage one:
        a clip refused at `prompt` still has `gate` and everything after it sitting at `pending`, and
        filtering one stage at a time would let those re-admit it.
        """
        return {
            clip_id
            for clip_id in self._rows
            if any(is_refusal(self.status(clip_id, stage).error) for stage in stages)
        }

    def stage_seconds(self, stage: str) -> list[float]:
        """Seconds recorded for `stage` on rows that reached `"done"` (failed/pending excluded)."""
        state_col, seconds_col = _stage_columns(stage)[:2]
        return [
            float(row[seconds_col])
            for row in self._rows.values()
            if row.get(state_col, "pending") == "done"
        ]

    def stage_names(self) -> list[str]:
        """Every stage name ever created or marked, sorted for deterministic ordering.

        Always a superset of `STAGE_NAMES` (pre-populated by `.create`); custom stage names
        introduced only via `.mark` (e.g. fake stages in tests) are included too.
        """
        known_stages: set[str] = set(STAGE_NAMES)
        for row in self._rows.values():
            for key in row:
                if key.endswith("_state"):
                    known_stages.add(key[: -len("_state")])
        return sorted(known_stages)

    def save(self, path: Path) -> None:
        """Write atomically: build the table, write it to a `.tmp` sibling, then `os.replace`.

        A crash between those two steps leaves the previous `path` untouched, and the `.tmp`
        sibling itself is removed on failure rather than left behind as debris.
        """
        rows = self._normalized_rows()
        table = pa.Table.from_pylist(rows)
        tmp_path = path.with_suffix(".tmp")
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            pq.write_table(table, tmp_path)
            os.replace(tmp_path, path)
        except OSError:
            tmp_path.unlink(missing_ok=True)
            raise

    def _normalized_rows(self) -> list[dict[str, Any]]:
        """Every row rebuilt in one fixed, sorted column order, so pyarrow sees one deterministic
        schema regardless of the (history-dependent) order each row's stages were marked in."""
        known_stages = self.stage_names()
        normalized: list[dict[str, Any]] = []
        for row in self._rows.values():
            full: dict[str, Any] = {
                "clip_id": row["clip_id"],
                "source": row["source"],
                "n_frames": row["n_frames"],
            }
            for stage in known_stages:
                state_col, seconds_col, error_col, batch_col = _stage_columns(stage)
                full[state_col] = row.get(state_col, "pending")
                full[seconds_col] = row.get(seconds_col, 0.0)
                full[error_col] = row.get(error_col, "")
                full[batch_col] = row.get(batch_col) or ""
            normalized.append(full)
        return normalized
