"""Tests for `fastdo.engine.batched_stages` and `SlotPool`: slot reuse, the hands and solve adapters."""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

import pytest
import torch

import fastdo.solver
from fastdo.engine.batched_stages import (
    HandsBatchStage,
    SolveBatchStage,
    SolveManyFn,
    resolve_solve_many,
)
from fastdo.engine.batching import SlotPool
from fastdo.engine.workers_hands import make_hands_worker
from fastdo.solver.api import SolverConfig
from fastdo.solver.memory import BatchPlan
from tests.engine.fakes import make_hawor_npz
from tests.engine.fakes_batched import (
    FakeHandsDriver,
    FakeSolveMany,
    SlotWorkerFactory,
    finish_fake_solve,
    prepare_fake_solve,
    solve_one,
    write_clip_inputs,
)
from tests.engine.test_hands_batch import HANDS_CFG


def _clips(root: Path, rows: list[int]) -> list[tuple[str, Path]]:
    return [(f"c{i}", write_clip_inputs(root, f"c{i}", n, seed=i)) for i, n in enumerate(rows)]


def test_slot_pool_reuses_slots_across_groups(tmp_path: Path) -> None:
    clips = _clips(tmp_path, [4, 4, 4, 4, 4])
    factory = SlotWorkerFactory("mask")
    pool = SlotPool(factory)

    runs = [run for group in ([0, 1], [2], [3, 4]) for run in pool.run_group([clips[i] for i in group])]

    assert factory.made == 2 and pool.n_slots == 2
    assert [run.clip_id for run in runs] == [c for c, _ in clips] and all(run.error == "" for run in runs)
    assert {c: factory.slots[c] for c in ("c0", "c2", "c3")} == dict.fromkeys(("c0", "c2", "c3"), 0)
    assert factory.slots["c1"] == factory.slots["c4"] == 1


def test_slot_pool_records_errors_and_reraises_base_exceptions(tmp_path: Path) -> None:
    clips = _clips(tmp_path, [4, 4])
    pool = SlotPool(SlotWorkerFactory("mask", fail_clips=("c1",)))
    assert [run.error for run in pool.run_group(clips)] == ["", "fake mask failure c1"]
    killer = SlotPool(SlotWorkerFactory("mask", kill_clips=("c0",)))
    killer.run_group(clips[:1])
    with pytest.raises(SystemExit):
        killer.run_group(clips)
    assert (tmp_path / "c1" / "mask.npz").is_file()                 # the batch-mate's thread finished first


def test_hands_batch_stage_one_driver_call_with_precomputed_clips(tmp_path: Path,
                                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    def stage_run(driver: FakeHandsDriver, root: Path) -> tuple[dict[str, str], list[tuple[str, Path]]]:
        clips = _clips(root, [4, 4, 4, 4])
        make_hawor_npz(root / "c1" / "all_hand_meshes.npz", t=12)
        monkeypatch.setattr("fastdo.engine.hands_batch.run_hands_batched", driver)
        stage = HandsBatchStage(HANDS_CFG, root / "jobs", root / ".gpu.lock", make_hands_worker(None))
        return dict(stage.run_batch(clips, "t0-hands-00000").errors), clips

    driver = FakeHandsDriver()
    errors, clips = stage_run(driver, tmp_path / "ok")
    assert driver.calls == [["c0", "c2", "c3"]] and driver.job_dirs == [tmp_path / "ok" / "jobs" / "t0-hands-00000"]
    assert errors == {} and all((d / "hands.npz").is_file() for _, d in clips)

    errors, _ = stage_run(FakeHandsDriver(fail=("c2",)), tmp_path / "fail")
    assert errors == {"c2": "boom c2"}

    errors, clips = stage_run(FakeHandsDriver(timeout_after_done=("c0",)), tmp_path / "timeout")
    assert (clips[0][1] / "hands.npz").is_file() and (clips[1][1] / "hands.npz").is_file()
    assert sorted(errors) == ["c2", "c3"] and all("TimeoutError" in error for error in errors.values())


def test_solve_batch_stage_one_call_and_t20_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                     caplog: pytest.LogCaptureFixture) -> None:
    clips = _clips(tmp_path, [4, 7, 5])
    fake = FakeSolveMany()
    finished: dict[str, torch.Tensor] = {}

    def finish(clip_id: str, artifact_dir: Path, call: object, solution: fastdo.solver.TrajectorySolution) -> None:
        finished[clip_id] = solution.poses
        finish_fake_solve(clip_id, artifact_dir, call, solution)  # type: ignore[arg-type]

    outcome = SolveBatchStage(prepare_fake_solve, finish, SolverConfig(), fake).run_batch(clips, "t0-solve-00000")

    assert outcome.errors == {} and outcome.batched is True and len(fake.calls) == 1
    calls = [prepare_fake_solve(c, d) for c, d in clips]
    assert all(torch.equal(m.points, c.mesh.points) for m, c in zip(fake.calls[0][0], calls, strict=True))
    assert [p.shape[0] for p in fake.calls[0][2]] == [4, 7, 5]
    for (clip_id, _), call in zip(clips, calls, strict=True):
        assert torch.equal(finished[clip_id], solve_one(call.init_poses, call.init_scale).poses)

    def prepare(clip_id: str, artifact_dir: Path):  # type: ignore[no-untyped-def]
        if clip_id == "c1":
            raise ValueError("no tracks for c1")
        return prepare_fake_solve(clip_id, artifact_dir)

    fake = FakeSolveMany()
    outcome = SolveBatchStage(prepare, finish_fake_solve, SolverConfig(), fake).run_batch(clips, "t0-solve-00001")
    assert outcome.errors == {"c1": "no tracks for c1"} and [p.shape[0] for p in fake.calls[0][2]] == [4, 5]

    seen: list[tuple[int, float]] = []

    def one(mesh, obs, init_poses, init_scale, cfg):  # type: ignore[no-untyped-def]
        seen.append((int(init_poses.shape[0]), init_scale))
        return solve_one(init_poses, init_scale)

    monkeypatch.delattr(fastdo.solver, "solve_trajectories", raising=False)
    monkeypatch.setattr(fastdo.solver, "solve_trajectory", one)
    with caplog.at_level(logging.WARNING):
        solve_many, batched = resolve_solve_many()
        solutions = solve_many([c.mesh for c in calls], [c.obs for c in calls], [c.init_poses for c in calls],
                               [c.init_scale for c in calls], SolverConfig())
    assert batched is False and len(solutions) == 3
    assert seen == [(int(c.init_poses.shape[0]), c.init_scale) for c in calls]
    assert any("solve_trajectories unavailable: solving clips one at a time" in r.getMessage() for r in caplog.records)
    monkeypatch.setattr(fastdo.solver, "solve_trajectories", fake, raising=False)
    assert resolve_solve_many() == (fake, True)


def test_hands_batch_stage_owns_one_warm_driver_when_opted_in(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Off by default (no warm kwarg at all); on, every group's call goes to the stage's one warm driver."""
    import fastdo.engine.batched_stages as bs

    closed: list[object] = []

    class _Warm:
        def __init__(self, cfg: object, log_dir: Path) -> None:
            self.log_dir = log_dir

        def close(self) -> None:
            closed.append(self)

    monkeypatch.setattr(bs, "WarmHandsDriver", _Warm)
    driver = FakeHandsDriver()
    monkeypatch.setattr("fastdo.engine.hands_batch.run_hands_batched", driver)
    cold = HandsBatchStage(HANDS_CFG, tmp_path / "jobs", tmp_path / ".gpu.lock", make_hands_worker(None))
    cold.run_batch(_clips(tmp_path / "a", [4]), "t0-hands-00000")
    cold.close()
    assert cold.warm is None and driver.warm == [None] and closed == []

    stage = HandsBatchStage(replace(HANDS_CFG, warm_driver=True), tmp_path / "jobs", tmp_path / ".gpu.lock",
                            make_hands_worker(None))
    for i, root in enumerate(("b", "c")):
        stage.run_batch(_clips(tmp_path / root, [4, 4]), f"t0-hands-{i + 1:05d}")
    stage.close()
    assert isinstance(stage.warm, _Warm) and stage.warm.log_dir == tmp_path / "jobs"
    assert driver.warm[1:] == [stage.warm, stage.warm] and closed == [stage.warm]


def test_hands_batch_stage_missing_intrinsics_matches_single_stream(tmp_path: Path,
                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    clips = _clips(tmp_path, [4, 4])
    (clips[1][1] / "intrinsics.npz").unlink()
    driver = FakeHandsDriver()
    monkeypatch.setattr("fastdo.engine.hands_batch.run_hands_batched", driver)
    outcome = HandsBatchStage(HANDS_CFG, tmp_path / "jobs", tmp_path / ".gpu.lock", make_hands_worker(None)).run_batch(
        clips, "t0-hands-00000")
    with pytest.raises(ValueError) as single:
        make_hands_worker(HANDS_CFG)("c1", clips[1][1])
    assert driver.calls == [["c0"]] and dict(outcome.errors) == {"c1": str(single.value)}


def test_solve_batch_stage_logs_executed_sub_batches(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    clips = _clips(tmp_path, [4, 7, 5])
    pair, single = BatchPlan((0, 1), 1, 0), BatchPlan((2,), 1, 0)

    def planned(plans: list[object]) -> SolveManyFn:
        def solve_many(*args):  # type: ignore[no-untyped-def]
            return [replace(s, plan=p) for s, p in zip(FakeSolveMany()(*args), plans, strict=True)]
        return solve_many

    with caplog.at_level(logging.INFO, logger="fastdo.engine.batched_stages"):
        outcome = SolveBatchStage(prepare_fake_solve, finish_fake_solve, SolverConfig(),
                                  planned([pair, pair, single])).run_batch(clips, "t0-solve-00000")
    assert outcome.errors == {} and any("executed sub-batches [2, 1]" in r.getMessage() for r in caplog.records)

    outcome = SolveBatchStage(prepare_fake_solve, finish_fake_solve, SolverConfig(),   # an unreadable plan only logs
                              planned([object()] * 3)).run_batch(clips, "t0-solve-00001")
    assert outcome.errors == {} and outcome.batched is True
    assert any("could not read the solve sub-batch plans" in r.getMessage() for r in caplog.records)
