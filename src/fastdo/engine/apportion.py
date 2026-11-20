"""Frame-proportional apportionment of batch wall time, the append-only batch log, and its invariant check.

A batch of the batched runner (`fastdo.engine.batched_runner`) is timed once. Its wall time is split over its clips in
proportion to the frames (decode rows) each processed, in integer nanoseconds with largest remainders, so the shares
sum to the wall exactly. A clip's stage seconds are the sum of its shares over every attempt it joined (a failed batch
plus its single retry) divided by 1e9. Every attempt is appended to `batches.jsonl` before the manifest is marked, so
`check_apportionment` can re-derive each manifest seconds value from the log.

A `LedgerBatchRecord` is a `BatchRecord` with one more key, `ledger_wait_ns`: the pipelined runner writes it for
an attempt its memory ledger gated, and only then, so a run without the ledger writes the same lines as before.
`read_batch_log` returns that subclass for a line holding the key and a plain `BatchRecord` otherwise.

An `AdmissionBatchRecord` is a `BatchRecord` with the optional wait keys `ledger_wait_ns`, `card_wait_ns` and
`oom_backoff_ns` (`fastdo.engine.card_admission`), in that order after the base keys; a key whose value is None is not
written. The runners write it only while `card_admission` or `oom_retry` is on, and `read_batch_log` returns it for a
line holding `card_wait_ns` or `oom_backoff_ns`.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, BinaryIO

from fastdo.engine.manifest import ClipManifest

__all__ = ["WAIT_KEYS", "AdmissionBatchRecord", "ApportionmentCheck", "BatchRecord", "LedgerBatchRecord",
           "append_batch_records", "apportion_ns", "check_apportionment", "read_batch_log"]

logger = logging.getLogger(__name__)


def apportion_ns(wall_ns: int, weights: Sequence[int]) -> list[int]:
    """Largest-remainder split of `wall_ns` in proportion to `weights` (all-zero weights count as 1 each).

    Floors `wall_ns * w_i // W`, then gives the remaining nanoseconds one each to the largest remainders
    `wall_ns * w_i % W`, ties to the lower index. The result sums to exactly `wall_ns`.

    Raises:
        ValueError: On a negative `wall_ns` or weight, or empty `weights`.
    """
    if not weights:
        raise ValueError("weights must not be empty")
    if wall_ns < 0 or any(w < 0 for w in weights):
        raise ValueError(f"wall_ns and weights must be >= 0, got {wall_ns} and {list(weights)}")
    w = [int(x) for x in weights] if any(weights) else [1] * len(weights)
    total = sum(w)
    shares = [wall_ns * x // total for x in w]
    remainders = [wall_ns * x % total for x in w]
    for i in sorted(range(len(w)), key=lambda i: (-remainders[i], i))[:wall_ns - sum(shares)]:
        shares[i] += 1
    return shares


@dataclass(frozen=True)
class BatchRecord:
    """One attempt of the batched runner: `errors[i]` is `""` on success, `retry_of` is `""` on a first attempt."""

    batch_id: str
    run_id: str
    stage: str
    clip_ids: tuple[str, ...]
    weights: tuple[int, ...]
    wall_ns: int
    share_ns: tuple[int, ...]
    errors: tuple[str, ...]
    retry_of: str
    batched: bool
    peak_gpu_mem_gb: float
    started_utc: str


@dataclass(frozen=True)
class LedgerBatchRecord(BatchRecord):
    """A `BatchRecord` of an attempt the memory ledger gated: `ledger_wait_ns` is the `perf_counter_ns` the
    attempt waited for its declared cost to fit, after its batch id was assigned and before its clock started (so
    `wall_ns` excludes it). Serialised as the base record's keys, in order, then `ledger_wait_ns`."""

    ledger_wait_ns: int

    @classmethod
    def of(cls, record: BatchRecord, ledger_wait_ns: int) -> LedgerBatchRecord:
        return cls(**{f.name: getattr(record, f.name) for f in fields(BatchRecord)}, ledger_wait_ns=ledger_wait_ns)


@dataclass(frozen=True)
class AdmissionBatchRecord(BatchRecord):
    """A `BatchRecord` with the wait keys (module docstring), each `perf_counter_ns` spent before the attempt's
    clock started, so `wall_ns` excludes all three: `ledger_wait_ns` as in `LedgerBatchRecord`, `card_wait_ns` waiting
    for the card's free memory (`card_admission`), `oom_backoff_ns` sleeping after the clip's previous attempt OOMed
    (`oom_retry`). None means "not recorded" and is not serialised."""

    ledger_wait_ns: int | None = None
    card_wait_ns: int | None = None
    oom_backoff_ns: int | None = None

    @classmethod
    def with_waits(cls, record: BatchRecord, **waits: int | None) -> AdmissionBatchRecord:
        """`record` with the given wait keys set (a None leaves the record's own value); keeps the others it has."""
        unknown = sorted(set(waits) - set(WAIT_KEYS))
        if unknown:
            raise TypeError(f"unknown wait keys {unknown}; valid: {list(WAIT_KEYS)}")
        own: dict[str, Any] = {f.name: getattr(record, f.name) for f in fields(BatchRecord)}
        own.update({name: getattr(record, name, None) for name in WAIT_KEYS})
        own.update({name: value for name, value in waits.items() if value is not None})
        return cls(**own)


WAIT_KEYS = ("ledger_wait_ns", "card_wait_ns", "oom_backoff_ns")
_ADMISSION_KEYS = frozenset({"card_wait_ns", "oom_backoff_ns"})
_TUPLE_FIELDS = ("clip_ids", "weights", "share_ns", "errors")


def _payload(record: BatchRecord) -> dict[str, Any]:
    payload = asdict(record)
    if isinstance(record, AdmissionBatchRecord):
        for name in WAIT_KEYS:
            if payload[name] is None:
                del payload[name]
    return payload


def append_batch_records(path: Path, records: Sequence[BatchRecord]) -> None:
    """Appends one JSON object per record per line, then `flush` and `os.fsync`.

    A torn last line left by a killed writer (no trailing newline) is truncated first, with a WARNING, so the
    appended records never merge into it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab+") as f:
        size = f.seek(0, os.SEEK_END)
        if size:
            f.seek(size - 1)
            if f.read(1) != b"\n":
                keep = _last_line_start(f, size)
                logger.warning("%s ends in a torn line of %d bytes; truncating it before appending", path, size - keep)
                f.truncate(keep)
                f.seek(0, os.SEEK_END)
        for record in records:
            f.write((json.dumps(_payload(record)) + "\n").encode())
        f.flush()
        os.fsync(f.fileno())


def _last_line_start(f: BinaryIO, size: int, chunk: int = 4096) -> int:
    """Offset just past the last newline of the first `size` bytes of `f` (0 when there is none), read backwards."""
    end = size
    while end > 0:
        start = max(0, end - chunk)
        f.seek(start)
        if (found := f.read(end - start).rfind(b"\n")) >= 0:
            return start + found + 1
        end = start
    return 0


def _record(payload: dict[str, Any]) -> BatchRecord:
    for name in _TUPLE_FIELDS:
        payload[name] = tuple(payload[name])
    if _ADMISSION_KEYS & payload.keys():
        return AdmissionBatchRecord(**payload)
    return (LedgerBatchRecord if "ledger_wait_ns" in payload else BatchRecord)(**payload)


def read_batch_log(path: Path) -> list[BatchRecord]:
    """Every record of `path` (`[]` for a missing file).

    Raises:
        ValueError: On a malformed line that is not the torn last line (the last line with no trailing newline is
            skipped with a WARNING instead), or on a duplicate `batch_id`.
    """
    path = Path(path)
    if not path.is_file():
        return []
    text = path.read_text()
    lines = text.split("\n")
    records: list[BatchRecord] = []
    for number, line in enumerate(lines, start=1):
        if not line:
            continue
        try:
            records.append(_record(json.loads(line)))
        except (json.JSONDecodeError, TypeError, KeyError) as exc:
            if number == len(lines):                      # no newline after it: a write the process never finished
                logger.warning("%s: skipping the torn last line %d (%s)", path, number, exc)
                continue
            raise ValueError(f"{path}: malformed batch record on line {number}: {exc}") from exc
    return list(_index_by_id(records, str(path)).values())


def _index_by_id(records: Sequence[BatchRecord], source: str) -> dict[str, BatchRecord]:
    """`batch_id -> record`; raises `ValueError` on a duplicate id (a log silently keeping one would misattribute)."""
    by_id: dict[str, BatchRecord] = {}
    for record in records:
        if record.batch_id in by_id:
            raise ValueError(f"{source}: duplicate batch_id {record.batch_id!r}")
        by_id[record.batch_id] = record
    return by_id


@dataclass(frozen=True)
class ApportionmentCheck:
    """Per stage: the clips' attributed nanoseconds, the wall of the records they reference, and the difference."""

    stage_total_ns: dict[str, int]
    batch_wall_ns: dict[str, int]
    unattributed_ns: dict[str, int]


def check_apportionment(manifest: ClipManifest, records: Sequence[BatchRecord]) -> ApportionmentCheck:
    """Re-derives every batched manifest seconds value from `records` (clips with `batch_id == ""` are skipped).

    For each clip and stage it follows `retry_of` back to the first attempt: every id must exist, every record's
    shares must sum to its wall, and the manifest seconds must equal the clip's shares over the chain / 1e9 exactly.
    `unattributed_ns` is the wall of the referenced records that no clip's final chain carries (a killed retry).

    Raises:
        ValueError: Naming the batch or clip whose identity fails, a duplicate `batch_id` or a `retry_of` cycle, or
            the stage whose clips carry more nanoseconds than its records' walls (`unattributed_ns` < 0).
    """
    by_id = _index_by_id(records, "batch records")
    totals: dict[str, int] = {}
    referenced: dict[str, set[str]] = {}
    for stage in manifest.stage_names():
        for clip_id in manifest.clip_ids():
            status = manifest.status(clip_id, stage)
            if not status.batch_id:
                continue
            total_ns, batch_id, chain = 0, status.batch_id, set[str]()
            while batch_id:
                if batch_id in chain:
                    raise ValueError(f"clip {clip_id!r} stage {stage!r}: retry_of cycle at batch {batch_id!r}")
                chain.add(batch_id)
                record = by_id.get(batch_id)
                if record is None:
                    raise ValueError(f"clip {clip_id!r} stage {stage!r} references unknown batch {batch_id!r}")
                if sum(record.share_ns) != record.wall_ns or len(record.share_ns) != len(record.clip_ids):
                    raise ValueError(f"batch {batch_id!r}: shares {record.share_ns} do not sum to {record.wall_ns}")
                if clip_id not in record.clip_ids:
                    raise ValueError(f"batch {batch_id!r} does not hold clip {clip_id!r}")
                total_ns += record.share_ns[record.clip_ids.index(clip_id)]
                referenced.setdefault(stage, set()).add(batch_id)
                batch_id = record.retry_of
            if status.seconds != total_ns / 1e9:
                raise ValueError(f"clip {clip_id!r} stage {stage!r}: manifest seconds {status.seconds!r} != "
                                 f"{total_ns} ns / 1e9 from its batches")
            totals[stage] = totals.get(stage, 0) + total_ns
    walls = {stage: sum(by_id[i].wall_ns for i in ids) for stage, ids in referenced.items()}
    if negative := {stage: walls[stage] - totals[stage] for stage in walls if walls[stage] < totals[stage]}:
        raise ValueError(f"unattributed_ns < 0 for stage(s) {negative}: clips carry more ns than their batches' walls")
    return ApportionmentCheck(stage_total_ns=totals, batch_wall_ns=walls,
                              unattributed_ns={stage: walls[stage] - totals[stage] for stage in walls})
