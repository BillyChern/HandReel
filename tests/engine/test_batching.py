"""Tests for `fastdo.engine.batching` (D3): budget-aware grouping and clip-concurrent stage runs."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from fastdo.engine.batching import dynamic_batches, run_clips_concurrently
from fastdo.engine.orchestrator import StageFn


def test_dynamic_batches_respects_budget() -> None:
    items = list(range(7))

    batches = dynamic_batches(items, max_batch=8, gpu_budget_gb=12.0, cost_fn=lambda _: 5.0)

    assert all(len(batch) <= 2 for batch in batches)
    assert [item for batch in batches for item in batch] == items          # every item once, in order
    assert [len(batch) for batch in batches] == [2, 2, 2, 1]


def test_dynamic_batches_shrinks_to_one() -> None:
    items = ["a", "b", "c"]

    batches = dynamic_batches(items, max_batch=4, gpu_budget_gb=12.0, cost_fn=lambda _: 20.0)

    assert batches == [["a"], ["b"], ["c"]]


def test_dynamic_batches_caps_group_size_and_mixes_costs() -> None:
    costs = {"s1": 1.0, "big": 10.0, "s2": 1.0, "s3": 1.0, "s4": 1.0}

    batches = dynamic_batches(list(costs), max_batch=3, gpu_budget_gb=11.0, cost_fn=costs.__getitem__)

    assert batches == [["s1", "big"], ["s2", "s3", "s4"]]
    assert dynamic_batches([], max_batch=2, gpu_budget_gb=1.0) == []


def test_dynamic_batches_cap_binds_with_items_left_over() -> None:
    assert dynamic_batches(list(range(5)), max_batch=2, gpu_budget_gb=100.0) == [[0, 1], [2, 3], [4]]


def test_dynamic_batches_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError, match="max_batch"):
        dynamic_batches([1], max_batch=0, gpu_budget_gb=1.0)
    with pytest.raises(ValueError, match="cost"):
        dynamic_batches([1], max_batch=1, gpu_budget_gb=1.0, cost_fn=lambda _: float("nan"))


def _slot_factory(made: list[int], seen: dict[str, int], barrier: threading.Barrier | None) -> StageFn:
    slot = len(made)
    made.append(slot)

    def run(clip_id: str, _artifact_dir: Path) -> None:
        seen[clip_id] = slot
        if barrier is not None:
            barrier.wait()                    # only passes when both clips of a group run at the same time
        if clip_id == "c3":
            raise RuntimeError("boom")

    return run


def test_run_clips_concurrently_runs_budgeted_groups_in_parallel_on_reused_slots(tmp_path: Path) -> None:
    made: list[int] = []
    seen: dict[str, int] = {}
    barrier = threading.Barrier(2, timeout=30)
    clips = [(f"c{i}", tmp_path / f"c{i}") for i in range(4)]

    runs = run_clips_concurrently(lambda: _slot_factory(made, seen, barrier), clips, max_concurrent=2,
                                  gpu_budget_gb=12.0, cost_fn=lambda _: 6.0)

    assert [run.clip_id for run in runs] == ["c0", "c1", "c2", "c3"]
    assert made == [0, 1]                                                   # two predictor slots, reused
    assert [seen[f"c{i}"] for i in range(4)] == [0, 1, 0, 1]
    assert [run.error for run in runs] == ["", "", "", "boom"]              # isolated per clip
    assert all(run.seconds >= 0.0 for run in runs)


def test_run_clips_concurrently_serializes_clips_that_exceed_the_budget(tmp_path: Path) -> None:
    made: list[int] = []
    seen: dict[str, int] = {}
    clips = [(f"c{i}", tmp_path / f"c{i}") for i in range(3)]

    runs = run_clips_concurrently(lambda: _slot_factory(made, seen, None), clips, max_concurrent=4,
                                  gpu_budget_gb=12.0, cost_fn=lambda _: 20.0)

    assert made == [0] and [seen[c] for c, _ in clips] == [0, 0, 0]
    assert [run.error for run in runs] == ["", "", ""]
