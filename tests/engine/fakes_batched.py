"""Deterministic fakes for the batched-orchestration tests: clip inputs, batched and slot workers, solve and hands.

Nothing here loads a model, touches the GPU or downloads anything.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Collection, Sequence
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from fastdo.engine.batched_stages import BatchOutcome
from fastdo.engine.orchestrator import StageFn
from fastdo.engine.worker_support import BatchedWorker, read_decode, read_frame_idx, save_npz
from fastdo.engine.workers_solve import SolveCall
from fastdo.engine.workers_video import write_mask_prompt
from fastdo.geometry.intrinsics import camera_matrix, write_intrinsics
from fastdo.hands.hawor_batched import HandsCallReport, HandsClipInput, HandsClipResult, HandsConfig
from fastdo.solver.api import MeshSamples, SolverConfig, TrajectoryObservations, TrajectorySolution
from tests.engine.fakes import random_frames, write_decode
from tests.engine.fakes_solve import fake_solution
from tests.hands.fake_hawor_driver import clip_result, write_response, write_valid_hand_npz

H, W = 24, 32
PASS_ROWS = 4


def prompt_mask() -> np.ndarray:
    """A centred 12x16 object mask on the 24x32 frame."""
    mask = np.zeros((H, W), dtype=bool)
    mask[6:18, 8:24] = True
    return mask


def write_clip_inputs(root: Path, clip_id: str, rows: int, seed: int, with_mask: bool = False) -> Path:
    """`decode.npz` (stride 3), `mask_prompt.npz` on row 1, `intrinsics.npz` and, with `with_mask`, `mask.npz`."""
    artifact_dir = Path(root) / clip_id
    write_decode(artifact_dir, random_frames(rows, H, W, seed), stride=3)
    write_mask_prompt(artifact_dir, 1, prompt_mask())
    write_intrinsics(artifact_dir, camera_matrix(30.0, 30.0, 16.0, 12.0), "given")
    if with_mask:
        save_npz(artifact_dir / "mask.npz", masks=np.repeat(prompt_mask()[None], rows, axis=0),
                 frame_idx=read_frame_idx(artifact_dir), keyframe_index=np.array(1, dtype=np.int64))
    return artifact_dir


class RowMeanWorker(BatchedWorker):
    """Writes `<stage>.npz` `value` (per-row channel means) through passes of at most 4 rows across clips."""

    def __init__(self, stage: str = "geometry", sleep_s: float = 0.0,
                 fail_when: Callable[[Sequence[str]], bool] | None = None,
                 kill_when: Callable[[Sequence[str]], bool] | None = None) -> None:
        self.stage, self.sleep_s, self.fail_when, self.kill_when = stage, sleep_s, fail_when, kill_when
        self.passes: list[list[str]] = []
        self.calls: list[list[str]] = []

    def run_many(self, clips: Sequence[tuple[str, Path]]) -> None:
        ids = [clip_id for clip_id, _ in clips]
        self.calls.append(ids)
        if self.kill_when is not None and self.kill_when(ids):
            raise SystemExit("killed batch")
        if self.fail_when is not None and self.fail_when(ids):
            raise RuntimeError("poison batch")
        if self.sleep_s:
            time.sleep(self.sleep_s)
        decoded = [read_decode(Path(artifact_dir)) for _, artifact_dir in clips]
        items = [(i, row) for i, (frames, _) in enumerate(decoded) for row in range(len(frames))]
        values: list[list[np.ndarray]] = [[] for _ in clips]
        for start in range(0, len(items), PASS_ROWS):
            chunk = items[start:start + PASS_ROWS]
            self.passes.append([ids[i] for i, _ in chunk])
            for i, row in chunk:
                values[i].append(decoded[i][0][row].reshape(-1, 3).astype(np.float64).mean(axis=0))
        for (_, artifact_dir), rows, (_, frame_idx) in zip(clips, values, decoded, strict=True):
            save_npz(Path(artifact_dir) / f"{self.stage}.npz", value=np.stack(rows).astype(np.float32),
                     frame_idx=frame_idx)


class SlotWorkerFactory:
    """`make_worker` for `ConcurrentClipStage`: each call builds a new slot `StageFn` (see the module docstring)."""

    def __init__(self, stage: str, sleep_s: float = 0.0, fail_clips: Collection[str] = (),
                 kill_clips: Collection[str] = (), meet: Collection[str] = ()) -> None:
        self.stage, self.sleep_s = stage, sleep_s
        self.fail_clips, self.kill_clips, self.meet = set(fail_clips), set(kill_clips), set(meet)
        self.barrier = threading.Barrier(len(self.meet), timeout=5) if self.meet else None
        self.made = 0
        self.calls: dict[str, int] = {}
        self.slots: dict[str, int] = {}
        self.durations_ns: dict[str, int] = {}
        self._lock = threading.Lock()

    def __call__(self) -> StageFn:
        slot = self.made
        self.made += 1

        def worker(clip_id: str, artifact_dir: Path) -> None:
            with self._lock:
                self.calls[clip_id] = self.calls.get(clip_id, 0) + 1
                n_call = self.calls[clip_id]
            if clip_id in self.meet and self.barrier is not None:
                self.barrier.wait()
            start = time.perf_counter_ns()
            if clip_id in self.fail_clips or (clip_id in self.kill_clips and n_call == 1):
                raise RuntimeError(f"fake {self.stage} failure {clip_id}")
            if clip_id in self.kill_clips:
                raise SystemExit(f"killed {clip_id}")
            if self.sleep_s:
                time.sleep(self.sleep_s)
            frames, frame_idx = read_decode(Path(artifact_dir))
            save_npz(Path(artifact_dir) / f"{self.stage}.npz", value=frames.max(axis=(1, 2, 3)).astype(np.uint8),
                     frame_idx=frame_idx)
            with self._lock:
                self.slots[clip_id] = slot
                self.durations_ns[clip_id] = time.perf_counter_ns() - start

        return worker


def fail_for(clip_ids: Collection[str], stage: str, worker: StageFn) -> StageFn:
    """`worker`, except that the listed clips raise `RuntimeError(f"fake {stage} failure {clip_id}")`."""
    failing = set(clip_ids)

    def run(clip_id: str, artifact_dir: Path) -> None:
        if clip_id in failing:
            raise RuntimeError(f"fake {stage} failure {clip_id}")
        worker(clip_id, artifact_dir)

    return run


class SleepBatchStage:
    """A `BatchedStage` that sleeps once per batch and writes nothing."""

    def __init__(self, sleep_s: float) -> None:
        self.sleep_s = sleep_s
        self.batches: list[list[str]] = []

    def run_batch(self, clips: Sequence[tuple[str, Path]], batch_id: str) -> BatchOutcome:
        self.batches.append([clip_id for clip_id, _ in clips])
        time.sleep(self.sleep_s)
        return BatchOutcome()


class FakeSolveMany:
    """`fake_solve_many`: pure per clip (`init_poses` translated by `init_scale * 0.01`); records each call."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[MeshSamples], list[TrajectoryObservations], list[Tensor], list[float]]] = []

    def __call__(self, meshes: list[MeshSamples], obs: list[TrajectoryObservations], init_poses: list[Tensor],
                 init_scales: list[float], config: SolverConfig) -> list[TrajectorySolution]:
        self.calls.append((list(meshes), list(obs), list(init_poses), list(init_scales)))
        return [solve_one(poses, scale) for poses, scale in zip(init_poses, init_scales, strict=True)]


def solve_one(init_poses: Tensor, init_scale: float) -> TrajectorySolution:
    """The per-clip fake solve behind `FakeSolveMany`."""
    poses = init_poses.clone()
    poses[:, :3, 3] += init_scale * 0.01
    return fake_solution(poses, init_scale)


def prepare_fake_solve(clip_id: str, artifact_dir: Path) -> SolveCall:
    """A `SolveCall` with T = decode rows; the initial translations are the rows' mean intensities."""
    frames, frame_idx = read_decode(Path(artifact_dir))
    t = int(frames.shape[0])
    poses = torch.eye(4, dtype=torch.float64).repeat(t, 1, 1)
    poses[:, 0, 3] = torch.from_numpy(frames.reshape(t, -1).mean(axis=1) / 255.0)
    mesh = MeshSamples(points=torch.zeros(8, 3), normals=torch.zeros(8, 3))
    obs = TrajectoryObservations(intrinsics=torch.eye(3), masks=torch.zeros(t, 2, 2, dtype=torch.bool),
                                 hand_masks=torch.zeros(t, 2, 2, dtype=torch.bool), pointmaps=torch.zeros(t, 2, 2, 3),
                                 tracks=torch.zeros(t, 1, 2), track_vis=torch.zeros(t, 1, dtype=torch.bool),
                                 anchors=torch.zeros(1, 3))
    return SolveCall(mesh=mesh, obs=obs, init_poses=poses, init_scale=1.0 + t / 10.0, context={"frame_idx": frame_idx})


def finish_fake_solve(clip_id: str, artifact_dir: Path, call: SolveCall, solution: TrajectorySolution) -> None:
    save_npz(Path(artifact_dir) / "solve.npz", poses=solution.poses.numpy())


class FakeHandsDriver:
    """A stand-in for `run_hands_batched` (the `_FakeBatched` pattern of `tests/engine/test_hands_batch.py`)."""

    def __init__(self, fail: Collection[str] = (), timeout_after_done: Collection[str] = ()) -> None:
        self.fail, self.timeout_after_done = set(fail), list(timeout_after_done)
        self.calls: list[list[str]] = []
        self.job_dirs: list[Path] = []
        self.warm: list[object] = []

    def __call__(self, clips: Sequence[HandsClipInput], cfg: HandsConfig, job_dir: Path,
                 **kwargs: object) -> HandsCallReport:
        self.calls.append([clip.clip_id for clip in clips])
        self.job_dirs.append(Path(job_dir))
        self.warm.append(kwargs.get("warm"))
        if self.timeout_after_done:
            Path(job_dir).mkdir(parents=True, exist_ok=True)
            done = []
            for clip in clips:
                if clip.clip_id in self.timeout_after_done:
                    frame_idx = read_frame_idx(clip.frames_path.parent)
                    write_valid_hand_npz(clip.out_npz, frame_idx)
                    done.append(clip_result(clip.clip_id, "done", "", len(frame_idx)))
            write_response(Path(job_dir) / "response.json", "", done)
            raise TimeoutError("HaWoR driver exceeded its timeout; see driver.log")
        results = []
        for clip in clips:
            if clip.clip_id in self.fail:
                results.append(HandsClipResult(clip.clip_id, "failed", f"boom {clip.clip_id}", 0, {}))
                continue
            frame_idx = read_frame_idx(clip.frames_path.parent)
            write_valid_hand_npz(clip.out_npz, frame_idx)
            results.append(HandsClipResult(clip.clip_id, "done", "", len(frame_idx), {"total": 1.0}))
        return HandsCallReport(1 if self.fail else 0, 6.0, 1.5, 2.0, 5.0, tuple(results))
