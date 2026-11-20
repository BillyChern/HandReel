"""M5: the training loop's resume, assist schedule, clip sampler and RSI.

Known answer: a run killed (SIGKILL) after a checkpoint and resumed reproduces the uninterrupted run bit for bit
(learner, replay, success store and episode log: `Trainer.digest`), in fresh single-threaded processes; so does a run
resumed from an OLDER checkpoint while the store already holds a success stored after it (a kill between a stored
success and the next checkpoint): the same digest and the same stats, the offer reported "stored", not "exists".
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from fastdo.amortised.schedule import AssistConfig, AssistSchedule, ClipSampler
from tests.amortised import m5_toy

TREE = Path(__file__).resolve().parents[2]


def _env() -> dict[str, str]:
    return {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": f"{TREE / 'src'}:{TREE}",
            "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}


def _cli(out: Path, steps: int, *extra: str) -> list[str]:
    return [sys.executable, "-m", "tests.amortised.m5_toy", str(out), str(steps), *extra]


N = 70


@pytest.fixture(scope="module")
def ref_run(tmp_path_factory) -> tuple[dict, Path]:
    """The uninterrupted N-step run: its printed summary and its output directory."""
    out = tmp_path_factory.mktemp("m5ref") / "ref"
    ref = subprocess.run(_cli(out, N), cwd=TREE, env=_env(), capture_output=True, text=True, timeout=900, check=True)
    return json.loads(ref.stdout.strip().splitlines()[-1]), out


def test_resume_after_a_kill_reproduces(tmp_path, ref_run) -> None:
    want = ref_run[0]
    out = tmp_path / "killed"
    proc = subprocess.Popen(_cli(out, N, "--checkpoint-every", "10"), cwd=TREE, env=_env(),
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t0 = time.monotonic()
    try:
        while time.monotonic() - t0 < 600:
            ck = sorted((out / "ckpt").glob("step_*")) if (out / "ckpt").is_dir() else []
            if ck and int(ck[-1].name.split("_")[1]) >= 30:
                break
            if proc.poll() is not None:
                break
            time.sleep(0.05)
    finally:
        if proc.poll() is None:
            os.kill(proc.pid, signal.SIGKILL)
        proc.wait()
    assert proc.returncode == -signal.SIGKILL, "the run finished before it could be killed"
    res = subprocess.run(_cli(out, N, "--resume", "--checkpoint-every", "10"), cwd=TREE, env=_env(),
                         capture_output=True, text=True, timeout=900, check=True)
    got = json.loads(res.stdout.strip().splitlines()[-1])
    assert got["resumed_from"] is not None and 30 <= got["resumed_from"] < N
    assert got["step"] == want["step"] == N
    assert got["digest"] == want["digest"], (got, want)


def test_resume_from_an_older_checkpoint_with_a_later_success_stored(tmp_path, ref_run) -> None:
    """Deterministic: run to N with checkpoints every 10, drop the newest (step N), resume from N - 10 while the store
    still holds every success of steps N-9..N. Precondition (asserted): the uninterrupted run stored a success in that
    window, so a re-offer of identical bytes is exercised."""
    want, ref_out = ref_run
    log = [json.loads(x) for x in (ref_out / "log.jsonl").read_text().splitlines()]
    assert any(r["offer"] == "stored" and r["step"] >= N - 10 for r in log), "no success after the older checkpoint"
    out = tmp_path / "older"
    subprocess.run(_cli(out, N, "--checkpoint-every", "10"), cwd=TREE, env=_env(), capture_output=True, text=True,
                   timeout=900, check=True)
    newest = sorted((out / "ckpt").glob("step_*"))
    assert [p.name for p in newest] == [f"step_{N - 10:09d}", f"step_{N:09d}"]
    shutil.rmtree(newest[-1])
    res = subprocess.run(_cli(out, N, "--resume", "--checkpoint-every", "10"), cwd=TREE, env=_env(),
                         capture_output=True, text=True, timeout=900, check=True)
    got = json.loads(res.stdout.strip().splitlines()[-1])
    assert got["resumed_from"] == N - 10
    assert got["offers"] == want["offers"] and "exists" not in got["offers"], (got["offers"], want["offers"])
    assert {k: v for k, v in got.items() if k != "resumed_from"} == {k: v for k, v in want.items()
                                                                      if k != "resumed_from"}
    assert (out / "log.jsonl").read_text() == (ref_out / "log.jsonl").read_text()


def test_assist_schedule_decays_on_half_passing_blocks_and_keeps_one_world_in_four_unassisted() -> None:
    a = AssistSchedule(AssistConfig(window=16), 2)
    assert [a.world_alpha(w, 1) for w in range(8)] == [0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]
    for k in range(16):
        a.record(0, 1.0, passed=k < 8)            # 8/16 = 50% pass -> decay
        a.record(1, 1.0, passed=k < 7)            # 7/16 -> no decay
        a.record(0, 0.0, passed=False)            # unassisted episodes never count
    assert a.level.tolist() == [0.9, 1.0]
    for _ in range(16 * 10):
        a.record(0, float(a.level[0]) or 1e-3, passed=True)
    assert a.level[0] == 0.0
    off = AssistSchedule(AssistConfig(enabled=False), 1)
    assert all(off.world_alpha(w, 0) == 0.0 for w in range(8))


def test_clip_sampler_weights_follow_failures_and_certification() -> None:
    s = ClipSampler(3, eps=0.05, window=16)
    np.testing.assert_allclose(s.weights(), [1.05, 1.05, 1.05])
    for k in range(20):
        s.record(0, failed=True)
        s.record(1, failed=k % 2 == 0)
    s.certified[2] = True
    np.testing.assert_allclose(s.weights(), [2.05, 1.55, 0.05])
    rng = np.random.default_rng(0)
    draws = np.bincount([s.draw(rng) for _ in range(4000)], minlength=3) / 4000
    np.testing.assert_allclose(draws, s.weights() / s.weights().sum(), atol=0.03)


def test_rsi_episodes_start_at_their_row(tmp_path) -> None:
    from fastdo.amortised.trainer import Trainer

    t = Trainer(m5_toy.config(max_steps=30, rsi_prob=1.0, assist=False), m5_toy.inputs(), tmp_path)
    t.start()
    for i, e in enumerate(t.eps):
        assert e.start_row >= t.inp.specs[e.clip].W
        assert int(t.rec["row"][i]) == e.start_row == int(t.env.start_row[i])
