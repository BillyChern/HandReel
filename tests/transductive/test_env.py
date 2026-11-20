"""The transductive environment's policy path, episode rules, success store hand-off and determinism (toy)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fastdo.physics.fd_problem import rollout, step_costs
from fastdo.policy.protocol import TrackingEnv
from fastdo.transductive.config import EnvConfig
from fastdo.transductive.env import TransductiveEnv
from fastdo.transductive.group import END_CODES, WorldGroup
from fastdo.transductive.obs import delta_scale
from fastdo.transductive.sim_cpu import CpuWorlds
from tests.transductive.toy import toy_clip


def _run_episode(env: TransductiveEnv, actions) -> list[dict]:
    infos = []
    while True:
        _, _, done, info = env.step(actions(env))
        infos.append(info)
        if bool(done[0]):
            return infos


def test_protocol_and_zero_residual_is_the_warm_start() -> None:
    sp = toy_clip()
    env = TransductiveEnv(EnvConfig(rows_per_step=10), {"a": sp}, ["a"], worlds_per_clip=2)
    assert isinstance(env, TrackingEnv)
    env.reset()
    infos = _run_episode(env, lambda e: torch.zeros(e.n_worlds, e.act_dim))
    raw = sum(float(i["raw_reward"][0]) for i in infos)
    ro = rollout(sp.ex, sp.rw, sp.ctrl_ref[:sp.T], want_grad=False)
    sc = step_costs(sp.ex, sp.rw, ro)
    mult = np.where(np.arange(1, sp.T + 1) < sp.W, sp.rw.warmup_mult, 1.0)
    fd = -(mult * (sc.a + sp.rw.vel_scale * sc.b + sp.rw.pen_scale * sc.h ** 2 + sp.rw.ped_scale * sc.ped)).sum()
    assert raw == pytest.approx(fd, abs=1e-12)
    last = infos[-1]
    assert last["end"][0] == END_CODES["clip_end"] and last["primary"][0] and last["ep_rows"][0] == sp.T
    succ = env.pop_successes()
    assert len(succ) == 2 and {s["world"] for s in succ} == {0, 1}
    np.testing.assert_array_equal(succ[0]["ctrl"], sp.ctrl_ref[:sp.T].astype(np.float32))
    assert succ[0]["env_rows_clip"] == 2 * sp.T and env.rows_per_clip[0] == 2 * sp.T


def test_residual_ramp_and_scaling() -> None:
    sp = toy_clip()
    cfg = EnvConfig(rows_per_step=4, hold="foh")
    g = WorldGroup(cfg, CpuWorlds([sp], [0]), [0], 1)
    g.reset_all()
    a = np.full((1, sp.n_robot), 0.5)
    g.step(a)
    target = delta_scale(cfg, sp.n_robot) * np.tanh(0.5)
    np.testing.assert_allclose(g.buf[0][:4] - sp.ctrl_ref[:4].astype(np.float32),
                               (np.arange(1, 5) / 4)[:, None] * target[None], atol=1e-6)
    np.testing.assert_array_equal(g.delta[0], target)
    zoh = WorldGroup(EnvConfig(rows_per_step=4, hold="zoh"), CpuWorlds([sp], [0]), [0], 1)
    zoh.reset_all()
    zoh.step(a)
    np.testing.assert_allclose(zoh.buf[0][:4] - sp.ctrl_ref[:4].astype(np.float32), np.repeat(target[None], 4, 0),
                               atol=1e-6)


def test_results_do_not_depend_on_worker_count() -> None:
    specs = {"a": toy_clip("a"), "b": toy_clip("b", ref_shift=0.02)}
    rng = np.random.default_rng(3)
    acts = [torch.as_tensor(rng.normal(0, 0.5, (6, 7)), dtype=torch.float32) for _ in range(12)]
    outs = []
    for workers in (0, 3):
        env = TransductiveEnv(EnvConfig(rows_per_step=10), specs, ["a", "b"], worlds_per_clip=3, n_workers=workers)
        try:
            seq = [env.reset()]
            for a in acts:
                o, r, d, _ = env.step(a)
                seq += [o, r, d.float()]
            outs.append(seq)
        finally:
            env.close()
    for x, y in zip(*outs, strict=True):
        assert torch.equal(x, y)


def test_snapshot_restore_continues_exactly() -> None:
    specs = {"a": toy_clip("a", press=True)}
    rng = np.random.default_rng(5)
    acts = [torch.as_tensor(rng.normal(0, 0.5, (2, 7)), dtype=torch.float32) for _ in range(6)]
    env = TransductiveEnv(EnvConfig(rows_per_step=10), specs, ["a"], worlds_per_clip=2)
    env.reset()
    for a in acts[:3]:
        env.step(a)
    snap = env.snapshot()
    ref = [env.step(a) for a in acts[3:]]
    env2 = TransductiveEnv(EnvConfig(rows_per_step=10), specs, ["a"], worlds_per_clip=2, n_workers=2)
    try:
        env2.restore(snap)
        got = [env2.step(a) for a in acts[3:]]
    finally:
        env2.close()
    for (o1, r1, d1, _), (o2, r2, d2, _) in zip(ref, got, strict=True):
        assert torch.equal(o1, o2) and torch.equal(r1, r2) and torch.equal(d1, d2)


def test_futility_ends_hopeless_episodes_and_never_a_success() -> None:
    bad = toy_clip(ref_shift=0.0)
    bad.rw.qpos_ref[:, bad.ex.obj_qadr] += 0.5            # object reference 0.5 m away: PRIMARY impossible early
    env = TransductiveEnv(EnvConfig(rows_per_step=10, futility=True), {"a": bad}, ["a"], worlds_per_clip=1)
    env.reset()
    infos = _run_episode(env, lambda e: torch.zeros(e.n_worlds, e.act_dim))
    assert infos[-1]["end"][0] == END_CODES["futile"] and infos[-1]["ep_rows"][0] < bad.T
    good = toy_clip()
    env = TransductiveEnv(EnvConfig(rows_per_step=10, futility=True), {"a": good}, ["a"], worlds_per_clip=1)
    env.reset()
    assert _run_episode(env, lambda e: torch.zeros(e.n_worlds, e.act_dim))[-1]["primary"][0]


class _DropAtEnd(CpuWorlds):
    """The executor terminating in the clip's last block (object 'dropped' at row T)."""

    def step_rows(self, i, ctrls):
        blk = super().step_rows(i, ctrls)
        if int(self.row[i]) >= self.spec(i).T:
            blk.end = "drop"
        return blk


def test_termination_in_the_last_block_still_completes_the_clip() -> None:
    sp = toy_clip()
    g = WorldGroup(EnvConfig(rows_per_step=10), _DropAtEnd([sp], [0]), [0], 1)
    g.reset_all()
    for _ in range(sp.T // 10):
        _, _, done, info, succ = g.step(np.zeros((1, sp.n_robot)))
    assert done[0] and info["end"][0] == END_CODES["drop"] and info["primary"][0] and len(succ) == 1


def test_explicit_controls_running_out_is_incomplete() -> None:
    sp = toy_clip()
    g = WorldGroup(EnvConfig(rows_per_step=10), CpuWorlds([sp], [0]), [0], 1)
    g.reset_all()
    g.step_controls([sp.ctrl_ref[:10]])
    _, _, done, info, succ = g.step_controls([np.zeros((0, sp.nu))])
    assert done[0] and info["end"][0] == END_CODES["incomplete"] and not info["primary"][0] and not succ


def test_assist_is_training_only_and_promotes_per_clip() -> None:
    sp = toy_clip()
    cfg = EnvConfig(rows_per_step=10, assist=True, assist_start=0.5, assist_step=0.25, assist_window=2,
                    assist_free_every=2)
    env = TransductiveEnv(cfg, {"a": sp}, ["a"], worlds_per_clip=4)
    np.testing.assert_array_equal(env.next_alpha(), [0.0, 0.5, 0.0, 0.5])
    env.reset()
    infos = _run_episode(env, lambda e: torch.zeros(e.n_worlds, e.act_dim))
    last = infos[-1]
    assert list(last["alpha"]) == [0.0, 0.5, 0.0, 0.5]
    assert list(last["primary"]) == [True, False, True, False]            # assisted episodes never count
    assert list(last["assisted_primary"]) == [False, True, False, True]
    assert [s["world"] for s in env.pop_successes()] == [0, 2]
    assert env.assist_alpha[0] == 0.25                                     # 2/2 assisted passes -> promoted
    np.testing.assert_array_equal(env.next_alpha(), [0.0, 0.25, 0.0, 0.25])


def test_unassisted_worlds_are_the_executor_exactly() -> None:
    sp = toy_clip(press=True)
    rng = np.random.default_rng(9)
    acts = [torch.as_tensor(rng.normal(0, 0.5, (2, 7)), dtype=torch.float32) for _ in range(8)]
    outs = []
    for assist in (False, True):
        env = TransductiveEnv(EnvConfig(rows_per_step=10, assist=assist, assist_free_every=2), {"a": sp}, ["a"],
                              worlds_per_clip=2)
        env.reset()
        outs.append([env.step(a) for a in acts])
    for (o1, r1, _, _), (o2, r2, _, _) in zip(*outs, strict=True):
        assert torch.equal(o1[0], o2[0]) and torch.equal(r1[0], r2[0])          # world 0 is unassisted
    assert not torch.equal(outs[0][-1][0][1], outs[1][-1][0][1])                 # world 1 felt the assist
