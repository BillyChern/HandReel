"""Tests for `fastdo.bench.yield_report`: yield and clips per GPU-hour from a finished manifest."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import pytest

from fastdo.bench import ProfileReport, StageRecord, YieldReport, compute_yield_report
from fastdo.bench.yield_report import (
    CAVEAT_KEY,
    GATE_UNBUILT,
    QC_FILE,
    YIELD_CAVEAT,
    clip_rate_agreement,
    clip_total_seconds,
    read_qc_verdicts,
    read_yield_report,
    write_yield_report,
)
from fastdo.engine.manifest import ClipManifest
from fastdo.engine.orchestrator import clips_per_gpu_hour_report

_CORE_FIELDS = ["n_clips", "n_gate_accepted", "n_qc_accepted", "clips_per_gpu_hour", "yield_fraction",
                "per_stage_seconds"]


def _manifest(n: int) -> ClipManifest:
    ids = [f"clip_{i:02d}" for i in range(n)]
    return ClipManifest.create(ids, ["whisking"] * n, [138] * n)


def test_compute_yield_report_arithmetic() -> None:
    manifest = _manifest(10)
    for i, clip_id in enumerate(manifest.clip_ids()):
        manifest.mark(clip_id, "decode", "done", 1.0)
        if i < 6:
            manifest.mark(clip_id, "gate", "done", 0.5)
            manifest.mark(clip_id, "mask", "done", 2.0)
            manifest.mark(clip_id, "qc", "done" if i < 3 else "failed", 0.5, error="" if i < 3 else "qc error")
        else:
            manifest.mark(clip_id, "gate", "failed", 0.5, error="rejected by the gate")

    report = compute_yield_report(manifest)

    total = 10 * 1.0 + 10 * 0.5 + 6 * 2.0 + 6 * 0.5
    assert report.n_clips == 10
    assert report.n_gate_accepted == 6
    assert report.n_qc_accepted == 3
    assert report.yield_fraction == 0.3
    assert report.per_stage_seconds["decode"] == 10.0
    assert report.per_stage_seconds["gate"] == 5.0
    assert report.per_stage_seconds["mask"] == 12.0
    assert report.per_stage_seconds["qc"] == 3.0          # a failed stage's seconds are spent compute too
    assert report.per_stage_seconds["solve"] == 0.0
    assert sum(report.per_stage_seconds.values()) == total
    # clips per GPU-hour = 3600 / (GPU seconds per clip); for one clip this is 3600 / total, as in ProfileReport.
    assert report.clips_per_gpu_hour == 3600.0 / (total / 10) == 1200.0


def test_orchestrator_report_matches_the_yield_report() -> None:
    """`clips_per_gpu_hour_report` writes the rate as `compute_yield_report` does, so the two agree
    bit for bit on the same manifest (failed-stage seconds included, n = every clip)."""
    manifest = _manifest(10)
    for i, clip_id in enumerate(manifest.clip_ids()):
        manifest.mark(clip_id, "gate", "done", 0.25)
        manifest.mark(clip_id, "decode", "done", 1.0)
        manifest.mark(clip_id, "shape", "done", 1.5)
        manifest.mark(clip_id, "qc", "done" if i < 6 else "failed", 0.25, error="" if i < 6 else "qc error")

    report = compute_yield_report(manifest)

    assert sum(report.per_stage_seconds.values()) == 30.0
    assert clips_per_gpu_hour_report(manifest)["overall"] == report.clips_per_gpu_hour == 1200.0


def test_compute_yield_report_handles_all_rejected() -> None:
    manifest = _manifest(4)
    for clip_id in manifest.clip_ids():
        manifest.mark(clip_id, "decode", "done", 1.0)
        manifest.mark(clip_id, "gate", "failed", 0.25, error="rejected by the gate")

    report = compute_yield_report(manifest)

    assert (report.n_gate_accepted, report.n_qc_accepted, report.yield_fraction) == (0, 0, 0.0)
    assert report.clips_per_gpu_hour == 3600.0 / 1.25
    empty = compute_yield_report(ClipManifest.create([], [], []))
    assert (empty.n_clips, empty.yield_fraction) == (0, 0.0)
    assert math.isinf(empty.clips_per_gpu_hour)
    untouched = compute_yield_report(_manifest(2))
    assert untouched.yield_fraction == 0.0 and math.isinf(untouched.clips_per_gpu_hour)


def test_unbuilt_gate_is_accept_all_only_when_flagged() -> None:
    manifest = _manifest(3)
    for clip_id in manifest.clip_ids():
        for stage in ("decode", "mask", "qc"):
            manifest.mark(clip_id, stage, "done", 1.0)

    assert compute_yield_report(manifest).n_gate_accepted == 0      # a built gate that never ran accepts nothing
    flagged = compute_yield_report(manifest, gate_built=False)
    assert (flagged.n_gate_accepted, flagged.n_qc_accepted, flagged.yield_fraction) == (3, 3, 1.0)
    assert GATE_UNBUILT == "unbuilt"


def test_qc_verdicts_count_acceptance_not_stage_completion() -> None:
    manifest = _manifest(4)
    for clip_id in manifest.clip_ids():
        manifest.mark(clip_id, "gate", "done", 0.1)
        manifest.mark(clip_id, "qc", "done", 0.1)      # the qc worker marks "done" for rejected clips too
    verdicts = {"clip_00": True, "clip_01": False, "clip_02": True}   # clip_03 has no qc.json

    assert compute_yield_report(manifest, qc_verdicts=verdicts).n_qc_accepted == 2
    manifest.mark("clip_00", "qc", "failed", 0.1, error="stale verdict")
    report = compute_yield_report(manifest, qc_verdicts=verdicts)
    assert report.n_qc_accepted == 1 and report.yield_fraction == 0.25


def test_read_qc_verdicts(tmp_path: Path) -> None:
    from fastdo.engine.workers_qc import QC_FILE as ENGINE_QC_FILE

    assert QC_FILE == ENGINE_QC_FILE
    for clip_id, accepted in (("a", True), ("b", False)):
        (tmp_path / clip_id).mkdir()
        (tmp_path / clip_id / QC_FILE).write_text(json.dumps({"clip_id": clip_id, "accepted": accepted}))
    (tmp_path / "c").mkdir()

    assert read_qc_verdicts(tmp_path, ["a", "b", "c", "missing"]) == {"a": True, "b": False}


def test_write_yield_report_keeps_core_fields_and_provenance(tmp_path: Path) -> None:
    path = tmp_path / "bench" / "yield_report.json"
    report = YieldReport(n_clips=3, n_gate_accepted=3, n_qc_accepted=1, clips_per_gpu_hour=float("inf"),
                         yield_fraction=1 / 3, per_stage_seconds={"decode": 1.5})
    extra = {"gate_status": GATE_UNBUILT, "rates": {"a/b~c": float("-inf"), "ok": 2.0}, "seq": [1.0, float("nan")]}

    write_yield_report(path, report, extra)

    text = path.read_text()
    payload = json.loads(text, parse_constant=lambda token: pytest.fail(f"non-standard JSON token {token}"))
    # the caveat trails every report (see test_every_yield_report_carries_its_own_caveat), so the
    # non_finite_values sentinel is now second-to-last rather than last
    assert list(payload)[:6] == _CORE_FIELDS
    assert list(payload)[-2:] == ["non_finite_values", CAVEAT_KEY]
    assert payload["gate_status"] == "unbuilt"
    assert payload["clips_per_gpu_hour"] is None and payload["per_stage_seconds"] == {"decode": 1.5}
    assert payload["non_finite_values"] == {"/clips_per_gpu_hour": "inf", "/rates/a~1b~0c": "-inf", "/seq/1": "nan"}
    restored = read_yield_report(path)
    assert math.isinf(restored["clips_per_gpu_hour"]) and restored["rates"]["a/b~c"] == float("-inf")
    assert math.isnan(restored["seq"][1]) and restored["rates"]["ok"] == 2.0
    assert not path.with_name(path.name + ".tmp").exists()
    with pytest.raises(ValueError, match="overwrite"):
        write_yield_report(path, report, {"n_clips": 7})
    with pytest.raises(ValueError, match="overwrite"):
        write_yield_report(path, report, {"non_finite_values": {}})


def test_clip_rate_agreement_between_profile_and_manifest(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    manifest = _manifest(2)
    seconds = {"decode": 0.125, "mask": 1.7, "tracks": 0.3}
    for clip_id in manifest.clip_ids():
        for stage, value in seconds.items():
            manifest.mark(clip_id, stage, "done", value)
    with caplog.at_level(logging.WARNING):
        assert clip_rate_agreement(manifest, tmp_path) == (None, 0)     # nothing compared is not agreement
    assert any("compared nothing" in r.getMessage() for r in caplog.records)
    profile = ProfileReport(clip_id="clip_00", system="engine", n_frames=138, fps=10.0,
                            stages=[StageRecord(name, value) for name, value in seconds.items()])
    profile.to_json(tmp_path / "clip_00" / "profile.json")

    total = clip_total_seconds(manifest, "clip_00")
    assert total == profile.total_seconds()
    assert clip_rate_agreement(manifest, tmp_path) == (0.0, 1)       # clip_01 has no profile.json and is skipped
    manifest.mark("clip_00", "mask", "done", 1.0)
    diff, compared = clip_rate_agreement(manifest, tmp_path)
    assert diff is not None and diff > 1.0 and compared == 1


def test_every_yield_report_carries_its_own_caveat(tmp_path: Path) -> None:
    """`yield_fraction` is the number a reader -- or a paper -- will quote, and it means "passed the
    configured screen", not "is usable". The 2026-09-20 finding is that the shipped screen's out-of-fold AUC
    is 0.4904, the 46th percentile of five pure-noise features, so the distinction is not pedantic.

    The caveat is emitted by the writer rather than documented, so no caller can omit it.
    """
    report = YieldReport(n_clips=4, n_gate_accepted=4, n_qc_accepted=2, clips_per_gpu_hour=15.2,
                         yield_fraction=0.5, per_stage_seconds={"decode": 1.0})
    path = tmp_path / "yield_report.json"
    write_yield_report(path, report)

    payload = json.loads(path.read_text())
    assert payload[CAVEAT_KEY] == YIELD_CAVEAT
    assert "not a usable-data rate" in payload[CAVEAT_KEY].lower().replace("  ", " ")
    assert "out of fold" in payload[CAVEAT_KEY].lower()

    # and a caller cannot quietly overwrite it
    with pytest.raises(ValueError, match="reserved"):
        write_yield_report(path, report, {CAVEAT_KEY: "looks fine to me"})


def test_a_stage_that_failed_before_qc_keeps_an_earlier_acceptance_out_of_the_yield() -> None:
    """A resumed run can leave qc "done" from the previous attempt beside a stage that fails now (for example
    a drift screen added on resume). Such a clip is not accepted; a failure AFTER qc (retarget) does not undo qc."""
    manifest = _manifest(4)
    for clip_id in manifest.clip_ids():
        manifest.mark(clip_id, "gate", "done", 0.1)
        manifest.mark(clip_id, "qc", "done", 0.1)
    manifest.mark("clip_00", "driftqc", "failed", 0.1, error="rejected by the drift screen")   # an opt-in stage
    manifest.mark("clip_01", "solve", "failed", 0.1, error="solve error")
    manifest.mark("clip_02", "retarget", "failed", 0.1, error="retarget error")
    verdicts = dict.fromkeys(manifest.clip_ids(), True)

    assert compute_yield_report(manifest).n_qc_accepted == 2
    report = compute_yield_report(manifest, qc_verdicts=verdicts)
    assert (report.n_qc_accepted, report.yield_fraction) == (2, 0.5)
