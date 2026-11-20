import time
from pathlib import Path

import pytest

from fastdo.bench.profiler import Profiler, ProfileReport, StageRecord


def test_stage_context_records_wall_clock() -> None:
    prof = Profiler(clip_id="c0", system="test", n_frames=10, fps=10.0, sync_cuda=False)
    with prof.stage("a", env="none"):
        time.sleep(0.02)
    rep = prof.report()
    assert rep.stages[0].name == "a"
    assert rep.stages[0].seconds >= 0.02
    assert rep.stages[0].meta == {"env": "none"}


def test_add_and_totals() -> None:
    prof = Profiler(clip_id="c0", system="test", n_frames=300, fps=30.0, sync_cuda=False)
    prof.add("x", 1800.0)
    prof.add("y", 1800.0)
    rep = prof.report()
    assert rep.total_seconds() == pytest.approx(3600.0)
    assert rep.gpu_hours() == pytest.approx(1.0)
    assert rep.clips_per_gpu_hour() == pytest.approx(1.0)


def test_json_roundtrip(tmp_path: Path) -> None:
    rep = ProfileReport(
        clip_id="c1", system="baseline", n_frames=5, fps=1.0,
        stages=[StageRecord("s", 2.5, 0.0, {"k": "v"})],
    )
    p = tmp_path / "r.json"
    rep.to_json(p)
    back = ProfileReport.from_json(p)
    assert back == rep


def test_empty_report_has_infinite_throughput() -> None:
    rep = ProfileReport(clip_id="c", system="s", n_frames=0, fps=0.0, stages=[])
    assert rep.total_seconds() == 0.0
    assert rep.clips_per_gpu_hour() == float("inf")


def test_stage_default_sync_cuda_records_stage() -> None:
    prof = Profiler("c", "test", 1, 1.0)  # default sync_cuda=True
    with prof.stage("a"):
        time.sleep(0.01)
    rep = prof.report()
    assert len(rep.stages) == 1
    assert rep.stages[0].name == "a"
    assert rep.stages[0].seconds >= 0.01
    assert rep.stages[0].peak_gpu_mem_gb >= 0.0
