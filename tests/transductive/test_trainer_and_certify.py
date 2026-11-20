"""Seeded, resumable training (exact on the CPU backend), the success store, certification artifacts and the
clip cache round trip (toy scene)."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from fastdo.policy.ppo import PpoConfig
from fastdo.retarget.isolation_artifacts import arm_time_base, load_arm_artifacts
from fastdo.transductive.certify import reexecute, write_run_dir
from fastdo.transductive.clip import load_clip, save_clip
from fastdo.transductive.config import EnvConfig
from fastdo.transductive.env import TransductiveEnv
from fastdo.transductive.trainer import TrainConfig, train
from tests.transductive.toy import toy_clip

PPO = PpoConfig(n_steps=8, n_epochs=2, n_minibatches=2, hidden=(16, 16))


def _env() -> TransductiveEnv:
    return TransductiveEnv(EnvConfig(rows_per_step=10), {"a": toy_clip("a"), "b": toy_clip("b", ref_shift=0.01)},
                           ["a", "b"], worlds_per_clip=2)


def _params(path: Path) -> list[torch.Tensor]:
    from fastdo.policy.checkpoint import load_checkpoint

    return list(load_checkpoint(path).model.state_dict().values())


def test_resume_reproduces_the_uninterrupted_run(tmp_path: Path) -> None:
    straight = TrainConfig(seed=7, n_updates=4, checkpoint_every=2, ppo=PPO, init_log_std=-1.0)
    train(_env(), tmp_path / "s", straight)
    train(_env(), tmp_path / "r", TrainConfig(seed=7, n_updates=2, checkpoint_every=2, ppo=PPO, init_log_std=-1.0))
    train(_env(), tmp_path / "r", straight, resume=True)          # continues from update 2 in a fresh env
    a, b = _params(tmp_path / "s/checkpoints/update_000004.pt"), _params(tmp_path / "r/checkpoints/update_000004.pt")
    assert all(torch.equal(x, y) for x, y in zip(a, b, strict=True))
    log_s = [json.loads(x) for x in (tmp_path / "s/log.jsonl").read_text().splitlines()]
    log_r = [json.loads(x) for x in (tmp_path / "r/log.jsonl").read_text().splitlines()]
    assert [x["env_rows"] for x in log_s] == [x["env_rows"] for x in log_r]
    train(_env(), tmp_path / "o", TrainConfig(seed=8, n_updates=4, checkpoint_every=2, ppo=PPO, init_log_std=-1.0))
    c = _params(tmp_path / "o/checkpoints/update_000004.pt")
    assert not all(torch.equal(x, y) for x, y in zip(a, c, strict=True))


def test_success_store_caps_and_records_env_rows(tmp_path: Path) -> None:
    env = TransductiveEnv(EnvConfig(rows_per_step=10), {"a": toy_clip("a")}, ["a"], worlds_per_clip=2)
    train(env, tmp_path, TrainConfig(seed=0, n_updates=6, checkpoint_every=6, store_per_clip=3, ppo=PPO,
                                     init_log_std=-5.0))                # near-deterministic: the warm start succeeds
    files = sorted((tmp_path / "successes" / "a").glob("*.npz"))
    assert len(files) == 3
    idx = [json.loads(x) for x in (tmp_path / "successes/index.jsonl").read_text().splitlines()]
    assert [r["k"] for r in idx] == [0, 1, 2] and all(r["env_rows_clip"] >= 80 for r in idx)
    with np.load(files[0]) as z:
        assert z["ctrl"].dtype == np.float32 and z["ctrl"].shape == (80, 7)


def test_reexecution_reproduces_the_training_rollout_and_writes_scorer_artifacts(tmp_path: Path) -> None:
    sp = toy_clip(press=True)
    env = TransductiveEnv(EnvConfig(rows_per_step=10, futility=False), {"a": sp}, ["a"], worlds_per_clip=1)
    env.reset()
    gen = torch.Generator().manual_seed(1)
    while True:
        _, _, done, _ = env.step(0.05 * torch.randn(1, 7, generator=gen))
        if bool(done[0]):
            break
    succ = env.pop_successes()
    assert succ, "the small-residual toy episode should pass"
    exe = reexecute(sp, succ[0]["ctrl"])
    assert exe.end == "clip_end" and exe.rows == sp.T
    pdir = tmp_path / "run"
    meta = {"warmup_steps": sp.W, "max_sim_steps": sp.max_sim_steps, "ctrl_steps": sp.ctrl_steps,
            "max_num_iterations": 32, "improvement_threshold": 0.01, "improvement_check_steps": 2,
            "raw_reference_frames": sp.max_sim_steps - sp.W, "ref_dt": 0.005, "sim_dt": 0.005}
    out = write_run_dir(pdir, sp, exe, meta, {"test": True})
    assert out["whole_pos_m"] == pytest.approx(succ[0]["whole_pos_m"], abs=1e-12)
    a = load_arm_artifacts(pdir)
    tb = arm_time_base(a)
    assert tb.executed_steps == sp.T and tb.completed_clip and tb.warmup_steps == sp.W
    np.testing.assert_array_equal(a.qpos_exec, exe.qpos)
    (tmp_path / "evil").mkdir()
    (tmp_path / "evil" / "trajectory_mjwp.npz").symlink_to(pdir / "trajectory_mjwp.npz")
    with pytest.raises(RuntimeError, match="symlink"):
        write_run_dir(tmp_path / "evil", sp, exe, meta, {})


def test_clip_cache_round_trip(tmp_path: Path) -> None:
    sp = toy_clip(press=True)
    save_clip(sp, tmp_path)
    back = load_clip(tmp_path, sp.task)
    raw = []
    for m in (sp.ex.model, back.ex.model):
        b = np.zeros(mujoco.mj_sizeModel(m), np.uint8)
        mujoco.mj_saveModel(m, None, b)
        raw.append(b)
    assert np.array_equal(*raw)
    for name in ("qpos_ref", "qvel_ref", "w_dq", "hand_geom", "gate_in_hand"):
        np.testing.assert_array_equal(getattr(back.rw, name), getattr(sp.rw, name))
    np.testing.assert_array_equal(back.ctrl_ref, sp.ctrl_ref)
    assert (back.T, back.W, back.ex.weld_ids, back.rw.pen_scale) == (sp.T, sp.W, sp.ex.weld_ids, sp.rw.pen_scale)


def test_resume_after_a_crash_does_not_duplicate_successes_or_log(tmp_path: Path) -> None:
    def cfg(n: int) -> TrainConfig:
        return TrainConfig(seed=3, n_updates=n, checkpoint_every=2, store_per_clip=50, ppo=PPO, init_log_std=-5.0)

    def env() -> TransductiveEnv:
        return TransductiveEnv(EnvConfig(rows_per_step=10), {"a": toy_clip("a")}, ["a"], worlds_per_clip=2)

    train(env(), tmp_path / "s", cfg(3))
    train(env(), tmp_path / "r", cfg(3))
    for f in ("update_000003.pt", "state_000003.npz"):              # "crash" after update 3, before its checkpoint
        (tmp_path / "r/checkpoints" / f).unlink()
    train(env(), tmp_path / "r", cfg(3), resume=True)               # replays update 3 from the update-2 checkpoint
    for name in ("successes/index.jsonl", "log.jsonl"):
        a = [json.loads(x) for x in (tmp_path / "s" / name).read_text().splitlines()]
        b = [json.loads(x) for x in (tmp_path / "r" / name).read_text().splitlines()]
        keys = ("task", "k", "update", "env_rows_clip") if name.startswith("succ") else ("update", "env_rows")
        assert [[x[k] for k in keys] for x in a] == [[x[k] for k in keys] for x in b], name
    assert len(list((tmp_path / "r/successes/a").glob("*.npz"))) == len(list((tmp_path / "s/successes/a").glob("*.npz")))


def test_budget_stop_writes_a_checkpoint(tmp_path: Path) -> None:
    env = TransductiveEnv(EnvConfig(rows_per_step=10), {"a": toy_clip("a")}, ["a"], worlds_per_clip=2)
    train(env, tmp_path, TrainConfig(seed=0, n_updates=50, max_env_rows=400, checkpoint_every=10, ppo=PPO))
    assert (tmp_path / "checkpoints/update_000003.pt").is_file()          # 3 x 2 worlds x 8 steps x 10 rows >= 400
