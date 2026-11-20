"""Tests for `fastdo.engine.apportion`: largest-remainder shares, the batch log, the invariant check."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Self

import pytest

from fastdo.engine.apportion import (
    BatchRecord,
    append_batch_records,
    apportion_ns,
    check_apportionment,
    read_batch_log,
)
from fastdo.engine.manifest import ClipManifest


def test_apportion_ns_largest_remainder() -> None:
    cases = {(1_000_000_001, (46, 46, 92)): [250_000_000, 250_000_000, 500_000_001], (7, (1, 2)): [2, 5],
             (10, (0, 0, 0)): [4, 3, 3], (0, (3, 1)): [0, 0], (5, (7,)): [5]}
    for (wall, weights), expected in cases.items():
        shares = apportion_ns(wall, list(weights))
        assert shares == expected and sum(shares) == wall
    for bad_wall, bad_weights in ((-1, [1]), (5, []), (5, [1, -1])):
        with pytest.raises(ValueError):
            apportion_ns(bad_wall, bad_weights)


def _record(batch_id: str, clips: tuple[str, ...], weights: tuple[int, ...], wall: int, shares: tuple[int, ...],
            errors: tuple[str, ...] | None = None, retry_of: str = "") -> BatchRecord:
    return BatchRecord(batch_id=batch_id, run_id=batch_id.rsplit("-", 2)[0], stage="a", clip_ids=clips,
                       weights=weights, wall_ns=wall, share_ns=shares, errors=errors or ("",) * len(clips),
                       retry_of=retry_of, batched=True, peak_gpu_mem_gb=0.5, started_utc="2026-09-14T16:00:00Z")


def test_batch_log_roundtrip_and_torn_last_line(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    log = tmp_path / "batches.jsonl"
    assert read_batch_log(log) == []
    records = [_record("r-a-00000", ("c0", "c1"), (4, 7), 11, (4, 7)),
               _record("r-a-00001", ("c1",), (7,), 9, (9,), errors=("x",), retry_of="r-a-00000")]
    append_batch_records(log, records[:1])
    append_batch_records(log, records[1:])
    assert read_batch_log(log) == records

    with log.open("a") as f:
        f.write('{"batch_id": "x')
    with caplog.at_level(logging.WARNING):
        assert read_batch_log(log) == records
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1

    with log.open("a") as f:
        f.write(log.read_text().splitlines()[0] + "\n")
    with pytest.raises(ValueError):
        read_batch_log(log)


def test_check_apportionment_exact_and_unattributed() -> None:
    manifest = ClipManifest.create(["c0", "c1", "c2"], ["s0", "s1", "s2"], [4, 7, 5])
    first = _record("r-a-00000", ("c0", "c1", "c2"), (4, 7, 5), 16_000, (4_000, 7_000, 5_000),
                    errors=("", "x", ""))
    retry = _record("r-a-00001", ("c1",), (7,), 9_000, (9_000,), retry_of="r-a-00000")
    manifest.mark("c0", "a", "done", 4_000 / 1e9, batch_id="r-a-00000")
    manifest.mark("c2", "a", "done", 5_000 / 1e9, batch_id="r-a-00000")
    manifest.mark("c1", "a", "done", (7_000 + 9_000) / 1e9, batch_id="r-a-00001")

    check = check_apportionment(manifest, [first, retry])
    assert (check.stage_total_ns, check.batch_wall_ns, check.unattributed_ns) == ({"a": 25_000}, {"a": 25_000},
                                                                                  {"a": 0})

    fresh = _record("r2-a-00000", ("c1",), (7,), 9_000, (9_000,))
    manifest.mark("c1", "a", "done", 9_000 / 1e9, batch_id="r2-a-00000")
    check = check_apportionment(manifest, [first, fresh])
    assert (check.stage_total_ns, check.batch_wall_ns, check.unattributed_ns) == ({"a": 18_000}, {"a": 25_000},
                                                                                  {"a": 7_000})

    with pytest.raises(ValueError, match="r-a-00000"):
        check_apportionment(manifest, [_record("r-a-00000", ("c0", "c1", "c2"), (4, 7, 5), 16_000,
                                               (4_000, 6_999, 5_000)), fresh])
    manifest.mark("c0", "a", "done", 4_001 / 1e9, batch_id="r-a-00000")
    with pytest.raises(ValueError, match="c0"):
        check_apportionment(manifest, [first, fresh])
    manifest.mark("c0", "a", "done", 4_000 / 1e9, batch_id="r-a-00009")
    with pytest.raises(ValueError, match="r-a-00009"):
        check_apportionment(manifest, [first, fresh])


class _ReadCounter:
    """Delegates to a real file object and counts the bytes `read` returns."""

    def __init__(self, f, counts: list[int]) -> None:  # type: ignore[no-untyped-def]
        self._f, self._counts = f, counts

    def read(self, n: int = -1) -> bytes:
        data = self._f.read(n)
        self._counts.append(len(data))
        return bytes(data)

    def __getattr__(self, name: str):  # type: ignore[no-untyped-def]
        return getattr(self._f, name)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self._f.close()


def test_append_reads_only_the_tail_of_the_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    counts: list[int] = []
    real_open = Path.open
    monkeypatch.setattr(Path, "open", lambda self, *a, **k: _ReadCounter(real_open(self, *a, **k), counts))
    record = _record("r-a-00000", ("c0",), (4,), 11, (11,))
    line = (json.dumps(asdict(record)) + "\n").encode()
    log, prefix = tmp_path / "batches.jsonl", b"x" * 99 + b"\n"
    log.write_bytes(prefix * 500)

    append_batch_records(log, [record])                  # clean log: one byte read, not 50 kB
    assert sum(counts) == 1                              # counted before `read_bytes`, which also goes through `open`
    assert log.read_bytes() == prefix * 500 + line

    counts.clear()
    log.write_bytes(prefix * 500 + b"y" * 5000)          # torn line longer than one backward chunk
    append_batch_records(log, [record])
    assert sum(counts) < 10_000 and log.read_bytes() == prefix * 500 + line
    log.write_bytes(b"z" * 10)                           # a torn line and nothing else
    append_batch_records(log, [record])
    assert log.read_bytes() == line


def test_duplicate_batch_ids_negative_unattributed_and_cycles_raise(tmp_path: Path) -> None:
    log = tmp_path / "batches.jsonl"
    record = _record("r-a-00000", ("c0", "c1"), (4, 7), 10, (-5, 15))
    append_batch_records(log, [record, record])
    with pytest.raises(ValueError, match="duplicate batch_id 'r-a-00000'"):
        read_batch_log(log)
    manifest = ClipManifest.create(["c0", "c1"], ["s0", "s1"], [4, 7])
    manifest.mark("c1", "a", "done", 15 / 1e9, batch_id="r-a-00000")
    with pytest.raises(ValueError, match="duplicate batch_id"):
        check_apportionment(manifest, [record, record])
    with pytest.raises(ValueError, match="unattributed_ns < 0 for stage.*'a': -5"):
        check_apportionment(manifest, [record])
    looped = _record("r-a-00000", ("c1",), (7,), 15, (15,), retry_of="r-a-00000")
    with pytest.raises(ValueError, match="retry_of cycle"):
        check_apportionment(manifest, [looped])
