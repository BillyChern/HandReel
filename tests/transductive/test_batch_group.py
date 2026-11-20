"""The batched torch episode logic (GPU path) reproduces the per-world numpy `WorldGroup` (CPU path)."""

from __future__ import annotations

import mujoco
import numpy as np
import torch

from fastdo.physics.fd_problem import quat_sub
from fastdo.transductive.batch_env import BatchedEnv
from fastdo.transductive.config import EnvConfig
from fastdo.transductive.env import TransductiveEnv
from fastdo.transductive.reward import contact_terms
from fastdo.transductive.sim_cpu import CpuWorlds
from fastdo.transductive.torch_terms import PoolRef, contact_terms_t, quat_sub_t
from tests.transductive.toy import toy_clip


def _specs() -> dict:
    bad = toy_clip("c", ref_shift=0.0)
    bad.rw.qpos_ref[:, bad.ex.obj_qadr] += 0.3                 # futile episodes
    return {"a": toy_clip("a", press=True), "b": toy_clip("b", ref_shift=0.02), "c": bad}


def test_batched_group_matches_world_group() -> None:
    specs = _specs()
    cfg = EnvConfig(rows_per_step=10, assist=True, assist_free_every=2, assist_window=2, assist_start=0.5)
    cpu = TransductiveEnv(cfg, specs, ["a", "b", "c"], worlds_per_clip=3)
    bat = BatchedEnv(cfg, specs, ["a", "b", "c"], worlds_per_clip=3, backend="cpu-pool", device="cpu")
    rng = np.random.default_rng(11)
    o1, o2 = cpu.reset(), bat.reset()
    np.testing.assert_allclose(o1.numpy(), o2.numpy(), atol=1e-5)
    n_succ = 0
    for _ in range(30):
        a = torch.as_tensor(rng.normal(0.0, 0.3, (9, 7)), dtype=torch.float32)
        o1, r1, d1, i1 = cpu.step(a)
        o2, r2, d2, i2 = bat.step(a)
        np.testing.assert_allclose(o1.numpy(), o2.numpy(), atol=1e-5)
        np.testing.assert_allclose(r1.numpy(), r2.numpy(), rtol=1e-6, atol=1e-7)
        assert torch.equal(d1, d2)
        for k in ("end", "primary", "strict", "rows", "ep_rows", "assisted_primary"):
            np.testing.assert_array_equal(i1[k], i2[k], err_msg=k)
        np.testing.assert_allclose(i1["raw_reward"], i2["raw_reward"], rtol=1e-9, atol=1e-12)
        s1, s2 = cpu.pop_successes(), bat.pop_successes()
        assert [(s["task"], s["world"], s["env_rows_clip"]) for s in s1] == \
            [(s["task"], s["world"], s["env_rows_clip"]) for s in s2]
        for x, y in zip(s1, s2, strict=True):
            np.testing.assert_array_equal(x["ctrl"], y["ctrl"])
        n_succ += len(s1)
    for k, v in cpu.counters().items():
        np.testing.assert_array_equal(v, bat.counters()[k], err_msg=k)
    assert n_succ > 0 and cpu.ends[2, 4] > 0                         # successes and futile ends were exercised
    assert cpu.assist_alpha[0] < 0.5                                  # and the curriculum moved


def test_quat_sub_t_matches_numpy() -> None:
    rng = np.random.default_rng(0)
    qa, qb = rng.normal(size=(64, 4)), rng.normal(size=(64, 4))
    qa /= np.linalg.norm(qa, axis=1, keepdims=True)
    qb /= np.linalg.norm(qb, axis=1, keepdims=True)
    qb[0] = qa[0]                                                     # the zero-rotation branch
    np.testing.assert_allclose(quat_sub_t(torch.as_tensor(qa), torch.as_tensor(qb)).numpy(), quat_sub(qa, qb),
                               atol=1e-12)


def test_contact_terms_t_matches_numpy_on_a_flat_buffer() -> None:
    sp = toy_clip(press=True)
    sim = CpuWorlds([sp, sp], [0, 1])
    sim.step_rows(0, sp.ctrl_ref[:50])
    sim.step_rows(1, sp.ctrl_ref[:30])
    dist, geom, wid = [], [], []
    for w, d in enumerate(sim.data):
        dist += list(np.asarray(d.contact.dist).reshape(-1))
        geom += list(np.asarray(d.contact.geom).reshape(-1, 2))
        wid += [w] * int(d.ncon)
    n = len(dist)
    pad = 7                                                           # garbage beyond nacon must not leak
    dist_t = torch.as_tensor(np.array(dist + [-1.0] * pad))
    geom_t = torch.as_tensor(np.array(geom + [[1, 2]] * pad, np.int64).reshape(-1, 2))
    wid_t = torch.as_tensor(np.array(wid + [0] * pad))
    ref = PoolRef.build(sp, "cpu", torch.float64)
    pen, ho, ped = contact_terms_t(ref, dist_t, geom_t, wid_t, torch.tensor([n]), 2)
    for w, d in enumerate(sim.data):
        assert (float(pen[w]), bool(ho[w]), bool(ped[w])) == contact_terms(d, sp.rw)
    assert float(pen.max()) > 0 and mujoco.mj_version() > 0


def test_batched_drop_at_block_end_matches_world_group() -> None:
    sp = toy_clip("d", ctrl_steps=20)
    sp.terminate_z = 0.15                      # the object rests at z ~0.2 until pushed: drop only when it falls
    sp2 = toy_clip("e", ctrl_steps=20)
    sp2.terminate_z = 10.0                     # every state is "below": the executor stops at the first block end
    specs = {"d": sp, "e": sp2}
    cfg = EnvConfig(rows_per_step=10, futility=False)
    cpu = TransductiveEnv(cfg, specs, ["d", "e"], worlds_per_clip=2)
    bat = BatchedEnv(cfg, specs, ["d", "e"], worlds_per_clip=2, backend="cpu-pool", device="cpu")
    cpu.reset(), bat.reset()
    rng = np.random.default_rng(2)
    for _ in range(12):
        a = torch.as_tensor(rng.normal(0.0, 0.5, (4, 7)), dtype=torch.float32)
        _, r1, d1, i1 = cpu.step(a)
        _, r2, d2, i2 = bat.step(a)
        assert torch.equal(d1, d2)
        np.testing.assert_array_equal(i1["ep_rows"], i2["ep_rows"])
        np.testing.assert_array_equal(i1["end"], i2["end"])
        np.testing.assert_allclose(r1.numpy(), r2.numpy(), rtol=1e-6, atol=1e-7)
    assert cpu.ends[1, 2] >= 2                  # clip e dropped at its block ends (rows 20), in both paths


class _InvalidAt(CpuWorlds):
    """World 0 of each episode turns invalid after executing row 13 (mid policy step)."""

    def step_rows(self, i, ctrls):
        blk = super().step_rows(i, ctrls)
        if i == 0 and (blk.rows == 13).any():
            n = int(np.flatnonzero(blk.rows == 13)[0]) + 1
            for k in ("rows", "qpos", "qvel", "pen", "ho", "ped", "ctrl"):
                setattr(blk, k, getattr(blk, k)[:n])
            blk.end = "invalid"
            self.row[i] = 14
        return blk


def test_batched_invalid_mid_step_matches_world_group() -> None:
    from fastdo.transductive.batch_group import BatchGroup, CpuPool, PoolsBackend
    from fastdo.transductive.group import END_CODES, WorldGroup

    sp = toy_clip("f", press=True)
    cfg = EnvConfig(rows_per_step=10, futility=False)
    wg = WorldGroup(cfg, _InvalidAt([sp], [0, 0]), [0], 1)
    bg = BatchGroup(cfg, PoolsBackend([CpuPool(sp, 2, cfg.assist_wn, sim_cls=_InvalidAt)], [0]), [sp], [0, 0], [0],
                    1, "cpu", torch.float64)
    o1, o2 = wg.reset_all(), bg.reset_all()
    np.testing.assert_allclose(o1, o2.numpy(), atol=1e-5)
    rng = np.random.default_rng(4)
    ends = []
    for _ in range(6):
        a = rng.normal(0.0, 0.3, (2, 7))
        o1, r1, d1, i1, _ = wg.step(a)
        o2, r2, d2, i2, _ = bg.step(torch.as_tensor(a))
        np.testing.assert_allclose(o1, o2.numpy(), atol=1e-5)
        np.testing.assert_allclose(r1, r2.numpy(), rtol=1e-6, atol=1e-7)
        np.testing.assert_array_equal(d1, d2.numpy())
        for k in ("end", "rows", "ep_rows"):
            np.testing.assert_array_equal(i1[k], i2[k], err_msg=k)
        ends.append(int(i1["end"][0]))
    assert END_CODES["invalid"] in ends


def test_float32_pools_report_exact_assist_levels() -> None:
    """The curriculum compares an episode's level with its clip's level exactly; float32 pools must not round it."""
    from fastdo.transductive.batch_group import BatchGroup, CpuPool, PoolsBackend

    sp = toy_clip("g")
    cfg = EnvConfig(rows_per_step=10, assist=True)
    bg = BatchGroup(cfg, PoolsBackend([CpuPool(sp, 2, cfg.assist_wn)], [0]), [sp], [0, 0], [0], 1, "cpu",
                    torch.float32)
    bg.reset_all(torch.tensor([0.0, 0.9], dtype=torch.float64))
    _, _, _, info, _ = bg.step(torch.zeros(2, 7), torch.tensor([0.0, 0.9], dtype=torch.float64))
    assert info["alpha"][1] == 0.9 and info["alpha"].dtype == np.float64


def _compare_envs(cpu: TransductiveEnv, bat: BatchedEnv, steps: int, seed: int, n: int) -> None:
    rng = np.random.default_rng(seed)
    np.testing.assert_allclose(cpu.reset().numpy(), bat.reset().numpy(), atol=1e-5)
    for _ in range(steps):
        a = torch.as_tensor(rng.normal(0.0, 0.3, (n, 7)), dtype=torch.float32)
        o1, r1, d1, i1 = cpu.step(a)
        o2, r2, d2, i2 = bat.step(a)
        np.testing.assert_allclose(o1.numpy(), o2.numpy(), atol=1e-5)
        np.testing.assert_allclose(r1.numpy(), r2.numpy(), rtol=1e-6, atol=1e-7)
        assert torch.equal(d1, d2)
        for k in ("end", "primary", "strict", "rows", "ep_rows", "clip"):
            np.testing.assert_array_equal(i1[k], i2[k], err_msg=k)
        s1, s2 = cpu.pop_successes(), bat.pop_successes()
        assert [(s["task"], s["world"]) for s in s1] == [(s["task"], s["world"]) for s in s2]
    for k, v in cpu.counters().items():
        np.testing.assert_array_equal(v, bat.counters()[k], err_msg=k)


def test_rsi_and_early_termination_match_between_paths_and_backends() -> None:
    """RSI episodes (random reference rows, weld settle, early termination) agree between the numpy path and the
    batched path on BOTH backends (per-clip pools and the union-style CPU backend), and never produce successes."""
    specs = _specs()
    cfg = EnvConfig(rows_per_step=10, rsi_frac=0.6, rsi_settle=5, rsi_min_rows=20, seed=7, assist=True,
                    assist_track=False, assist_free_every=2, assist_start=0.5)
    for backend in ("cpu-pool", "cpu-batch"):
        cpu = TransductiveEnv(cfg, specs, ["a", "b", "c"], worlds_per_clip=3, n_workers=2)
        try:
            bat = BatchedEnv(cfg, specs, ["a", "b", "c"], worlds_per_clip=3, backend=backend, device="cpu")
            _compare_envs(cpu, bat, 30, 5, 9)
        finally:
            cpu.close()
        assert cpu.ends[:, 6].sum() > 0                           # early terminations happened


def test_rsi_start_state_is_the_reference_with_a_settle_weld() -> None:
    sp = toy_clip("r", T=80, W=20, ref_shift=0.0)
    sim = CpuWorlds([sp], [0])
    sim.reset(0, 40, settle=5)
    a0 = sp.ex.obj_qadr
    np.testing.assert_array_equal(sim.qpos(0), sp.rw.qpos_ref[40])
    obj0 = sim.qpos(0)[a0:a0 + 7].copy()
    blk = sim.step_rows(0, sp.ctrl_ref[40:45])
    np.testing.assert_allclose(blk.qpos[-1, a0:a0 + 3], obj0[:3], atol=1e-3)   # held by the weld during the settle
    assert int(sim.weld_until[0]) == 45
    fresh = CpuWorlds([sp], [0])                                     # the shared model is left with the executor's
    np.testing.assert_array_equal(fresh.step_rows(0, sp.ctrl_ref[:30]).qpos,   # own weld target
                                  CpuWorlds([sp], [0]).step_rows(0, sp.ctrl_ref[:30]).qpos)
    st = sim.get_state(0)
    other = CpuWorlds([sp], [0])
    other.set_state(0, st)
    np.testing.assert_array_equal(other.step_rows(0, sp.ctrl_ref[45:60]).qpos, sim.step_rows(0, sp.ctrl_ref[45:60]).qpos)


def test_union_style_backend_rebinds_worlds_to_other_clips() -> None:
    from fastdo.transductive.batch_group import BatchGroup, CpuBatchBackend

    specs = [toy_clip("a"), toy_clip("b", ref_shift=0.02)]
    cfg = EnvConfig(rows_per_step=10)
    be = CpuBatchBackend(specs, [0, 0], cfg.assist_wn)
    g = BatchGroup(cfg, be, specs, [0, 0], [0, 1], 2, "cpu", torch.float64)
    g.reset_all()
    g.next_clip[1] = 1                                               # world 1's next episode runs clip b
    for _ in range(specs[0].T // 10):
        _, _, done, info, _ = g.step(torch.zeros(2, 7))
    assert bool(done.all()) and list(info["clip"]) == [0, 0]
    assert list(g.clip.numpy()) == [0, 1] and int(be.sim.world_clip[1]) == 1
    _, _, _, info, _ = g.step(torch.zeros(2, 7))
    assert list(info["clip"]) == [0, 1]
