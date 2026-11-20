"""The vectorized reward/tracking terms and the CPU backend reproduce the D4' executor and reward (toy scene)."""

from __future__ import annotations

import mujoco
import numpy as np

from fastdo.physics.fd_problem import _contact_pen, quat_sub, rollout, step_costs
from fastdo.transductive.reward import contact_terms, row_costs, row_tracking
from fastdo.transductive.sim_cpu import CpuWorlds
from tests.transductive.toy import toy_clip


def _weighted_fd(sp, ro) -> np.ndarray:
    sc = step_costs(sp.ex, sp.rw, ro)
    rw = sp.rw
    c = sc.a + rw.vel_scale * sc.b + rw.pen_scale * sc.h ** 2 + rw.ped_scale * sc.ped
    mult = np.where(np.arange(1, ro.qpos.shape[0] + 1) < sp.W, rw.warmup_mult, 1.0)
    return -(mult * c)


def test_backend_matches_fd_rollout_bitwise_and_reward_matches_step_costs() -> None:
    sp = toy_clip(ref_shift=0.03, press=True)
    rng = np.random.default_rng(0)
    u = sp.ctrl_ref[:sp.T] + rng.normal(0.0, 0.01, (sp.T, sp.nu))
    ro = rollout(sp.ex, sp.rw, u, want_grad=False)
    sim = CpuWorlds([sp], [0])
    blocks = [sim.step_rows(0, u[s:s + 7]) for s in range(0, sp.T, 7)]       # blocks not aligned with anything
    qpos = np.concatenate([b.qpos for b in blocks])
    assert np.array_equal(qpos, ro.qpos)
    assert np.array_equal(np.concatenate([b.pen for b in blocks]), ro.pen)
    assert np.array_equal(np.concatenate([b.ho for b in blocks]), ro.ho_touch)
    rows = np.concatenate([b.rows for b in blocks])
    rc = row_costs(sp.rw, sp.ex.obj_qadr, sp.W, rows, qpos, np.concatenate([b.qvel for b in blocks]),
                   np.concatenate([b.pen for b in blocks]), np.concatenate([b.ho for b in blocks]),
                   np.concatenate([b.ped for b in blocks]))
    np.testing.assert_allclose(rc.weighted, _weighted_fd(sp, ro), rtol=0, atol=1e-12)
    assert (rc.h > 0).any(), "the pressing toy must exercise the penetration hinge"
    shuffled = rng.permutation(rows.shape[0])                                   # samples are independent
    rc2 = row_costs(sp.rw, sp.ex.obj_qadr, sp.W, rows[shuffled], qpos[shuffled],
                    np.concatenate([b.qvel for b in blocks])[shuffled], ro.pen[shuffled], ro.ho_touch[shuffled],
                    ro.ped_touch[shuffled])
    np.testing.assert_array_equal(rc2.weighted, rc.weighted[shuffled])


def test_contact_terms_match_fd_contact_pen() -> None:
    sp = toy_clip(press=True)
    m, d = sp.ex.model, mujoco.MjData(sp.ex.model)
    sim = CpuWorlds([sp], [0])
    blk = sim.step_rows(0, sp.ctrl_ref[:sp.T])
    for q in blk.qpos[::5]:
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        best, _, ho, ped = _contact_pen(m, d, sp.rw, False)
        assert contact_terms(d, sp.rw) == (best, ho, ped)


def test_tracking_errors_are_do_as_i_do_frames() -> None:
    sp = toy_clip(ref_shift=0.02)
    blk = CpuWorlds([sp], [0]).step_rows(0, sp.ctrl_ref[:sp.T])
    pos, rot = row_tracking(sp.rw, sp.ex.obj_qadr, blk.rows, blk.qpos)
    a0 = sp.ex.obj_qadr
    ref = sp.rw.qpos_ref[:sp.T].astype(np.float32).astype(np.float64)        # state after row s vs reference row s
    np.testing.assert_array_equal(pos, np.linalg.norm(blk.qpos[:, a0:a0 + 3] - ref[:, a0:a0 + 3], axis=1))
    np.testing.assert_array_equal(rot, np.linalg.norm(quat_sub(blk.qpos[:, a0 + 3:a0 + 7], ref[:, a0 + 3:a0 + 7]),
                                                      axis=1))


def test_drop_termination_only_at_block_ends() -> None:
    sp = toy_clip(ctrl_steps=20)
    sp.terminate_z = 10.0                       # every state is "below": the executor stops at the first block end
    sim = CpuWorlds([sp], [0])
    b1 = sim.step_rows(0, sp.ctrl_ref[:13])
    assert b1.end is None and b1.rows.shape[0] == 13
    b2 = sim.step_rows(0, sp.ctrl_ref[13:40])
    assert b2.end == "drop" and b2.rows[-1] == 19 and int(sim.row[0]) == 20


def test_state_snapshot_is_exact() -> None:
    sp = toy_clip(press=True)
    u = sp.ctrl_ref[:sp.T]
    sim = CpuWorlds([sp, sp], [0, 1])
    sim.step_rows(0, u[:30])
    snap = sim.get_state(0)
    after = sim.step_rows(0, u[30:60]).qpos
    sim.set_state(1, snap)
    assert int(sim.row[1]) == 30
    np.testing.assert_array_equal(sim.step_rows(1, u[30:60]).qpos, after)
    sim.reset(0)
    fresh = CpuWorlds([sp], [0])
    np.testing.assert_array_equal(sim.step_rows(0, u[:10]).qpos, fresh.step_rows(0, u[:10]).qpos)
