"""D4' unit tests: knot basis, reward known answers, contact-normal Jacobian, chained-FD Jacobian vs rollout
finite differences, adjointness, LM descent on a toy, never-worse acceptance, opt-in config, parallel FD."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fd_toy import toy_problem  # noqa: E402

import fastdo.physics.fd_gauss_newton as G  # noqa: E402
from fastdo.physics.fd_linearize import FDPool, fd_steps_per_row, linearize  # noqa: E402
from fastdo.physics.fd_problem import (Rollout, _contact_pen, dq_obj_rot, quat_sub, rollout,  # noqa: E402
                                       step_costs)
from fastdo.physics.planner_config import D4PrimeConfig, PlannerConfig, from_dict  # noqa: E402

CFG = D4PrimeConfig(enabled=True, knot_steps=10, workers=0, max_iters=4, cg_max_iters=60)


def test_knot_basis_matches_torch_interpolate_and_adjoint():
    torch = pytest.importorskip("torch")
    F = torch.nn.functional
    rng = np.random.default_rng(0)
    for T, n in ((80, 10), (1500, 40), (2700, 40), (37, 40)):
        b = G.KnotBasis(T, n)
        th = rng.normal(size=(b.K, 3))
        ref = F.interpolate(torch.tensor(th.T[None]), size=b.K * n, mode="linear", align_corners=True)[0].T.numpy()
        np.testing.assert_allclose(b.apply(th), ref[:T], atol=1e-12)
        g = rng.normal(size=(T, 3))
        assert np.vdot(b.apply(th), g) == pytest.approx(np.vdot(th, b.adjoint(g)), rel=1e-12)


def test_quat_sub_known_answers():
    from scipy.spatial.transform import Rotation as R

    rng = np.random.default_rng(1)
    qa = R.random(50, random_state=2)
    qb = R.random(50, random_state=3)
    wxyz = lambda r: np.roll(r.as_quat(), 1, axis=-1)  # noqa: E731
    got = quat_sub(wxyz(qa), wxyz(qb))
    want = (qb.inv() * qa).as_rotvec()
    np.testing.assert_allclose(got, want, atol=1e-9)
    assert np.abs(quat_sub(wxyz(qa), wxyz(qa))).max() < 1e-12
    # DO AS I DO's sign convention: angle 2*atan2(|v|, w) > pi is wrapped by -2 pi (w < 0 representation)
    q = np.array([-np.cos(0.2), np.sin(0.2), 0.0, 0.0])
    np.testing.assert_allclose(quat_sub(q, np.array([1.0, 0, 0, 0])), [-0.4, 0, 0], atol=1e-12)
    # Jacobian of the local-frame tangent: central FD of the closed form vs a direct FD here
    q0 = wxyz(qa[:4])
    qr = wxyz(qb[:4])
    Jq = dq_obj_rot(q0, qr)
    for i in range(3):
        d = np.zeros(3)
        d[i] = 1e-7
        qp = wxyz(qa[:4] * R.from_rotvec(np.tile(d, (4, 1))))
        np.testing.assert_allclose((quat_sub(qp, qr) - quat_sub(q0, qr)) / 1e-7, Jq[:, :, i], atol=1e-5)
    assert rng is not None


def test_step_costs_known_answer():
    ex, rw, u0 = toy_problem(T=30, W=10)
    ro = rollout(ex, rw, u0)
    T = 30
    ro.qpos = rw.qpos_ref[1:T + 1].copy()
    ro.qvel = rw.qvel_ref[1:T + 1].copy()
    ro.pen = np.zeros(T)
    ro.qpos[5, 0] += 0.02                     # base pos x: weight 0.1, in the warmup (x 0.1)
    ro.qpos[15, 7] += 0.03                    # object x: weight 1.0
    ro.qvel[20, 3] += 2.0                     # velocity: scale 1e-4
    ro.pen[25] = 0.009                        # 4 mm beyond the 5 mm margin: 3000 * 0.004^2
    ro.qpos[T - 1, 8] += 0.01                 # terminal row x10
    sc = step_costs(ex, rw, ro)
    want = 0.1 * 0.1 * 0.02 + 1.0 * 0.03 + 1e-4 * 2.0 + 3000 * 0.004 ** 2 + 10 * 0.01
    assert sc.total == pytest.approx(want, rel=1e-9)
    assert sc.mult[:9].tolist() == [0.1] * 9 and sc.mult[9] == 1.0 and sc.mult[-1] == 10.0


def test_pedestal_mismatch_branches():
    ex, rw, u0 = toy_problem(T=30, W=10)
    ro = rollout(ex, rw, u0)
    rw.gate_in_hand[:] = False
    rw.gate_near_ped[:] = False
    rw.gate_in_hand[12:15] = True             # in-hand branch at sim steps 12..14 -> rows 11..13
    rw.gate_near_ped[20:22] = True            # rest branch at sim steps 20, 21 -> rows 19, 20
    rw.gate_in_hand[5] = True                 # warmup: never penalised
    ro.ho_touch[:] = False
    ro.ho_touch[12] = True
    ro.ped_touch[:] = False
    sc = step_costs(ex, rw, ro)
    assert np.flatnonzero(sc.ped).tolist() == [11, 13, 19, 20]


def test_contact_normal_jacobian_matches_fd():
    ex, rw, _ = toy_problem(T=10, W=5)
    m = ex.model
    d = mujoco.MjData(m)
    d.qpos[:] = ex.q0
    d.qpos[2] = -0.078                         # palm into the box top by ~8 mm, tilted: one deepest corner
    d.qpos[3], d.qpos[4] = 0.05, 0.08
    d.qpos[6] = 0.2
    mujoco.mj_forward(m, d)
    pen, grad, _, _ = _contact_pen(m, d, rw, True)
    assert pen > rw.pen_margin and grad is not None
    num = np.zeros(m.nv)
    for i in range(m.nv):
        vals = []
        for sgn in (1.0, -1.0):
            q = d.qpos.copy()
            v = np.zeros(m.nv)
            v[i] = sgn * 1e-6
            mujoco.mj_integratePos(m, q, v, 1.0)
            d2 = mujoco.MjData(m)
            d2.qpos[:] = q
            mujoco.mj_forward(m, d2)
            vals.append(_contact_pen(m, d2, rw, False)[0])
        num[i] = (vals[0] - vals[1]) / 2e-6
    np.testing.assert_allclose(grad, num, atol=2e-3 * max(1.0, np.abs(num).max()))


def _frozen_residual(gm, ex, rw, z, u0, basis, sigma):
    ro = rollout(ex, rw, u0 + basis.apply(z * sigma))
    sc = step_costs(ex, rw, ro)
    cq, cv, cp = gm.c
    return (cq[:, None] * rw.w_dq[None] * sc.dq, cv[:, None] * sc.dv, cp * sc.h)


def test_chained_fd_jacobian_matches_rollout_differences_and_is_adjoint():
    ex, rw, u0 = toy_problem(T=60, W=20, ref_shift=0.01)
    basis = G.KnotBasis(60, CFG.knot_steps)
    sigma = G.knot_sigma(ex.model.nu, CFG)
    ro = rollout(ex, rw, u0)
    sc = step_costs(ex, rw, ro)
    with FDPool(ex, 0) as pool:
        lin = linearize(pool, ro, eps=1e-6, centered=True)
    gm = G.GNModel(ex, rw, ro, sc, lin, basis, sigma, CFG)
    rng = np.random.default_rng(4)
    z0 = np.zeros((basis.K, ex.model.nu))
    errs = []
    for _ in range(3):
        v = rng.normal(size=z0.shape)
        jv = gm.jv(v)
        h = 1e-4
        rp = _frozen_residual(gm, ex, rw, z0 + h * v, u0, basis, sigma)
        rm = _frozen_residual(gm, ex, rw, z0 - h * v, u0, basis, sigma)
        num = np.concatenate([((a - b) / (2 * h)).ravel() for a, b in zip(rp, rm)])
        ana = np.concatenate([a.ravel() for a in jv])
        errs.append(np.linalg.norm(ana - num) / np.linalg.norm(num))
        w = tuple(rng.normal(size=a.shape) for a in jv)
        assert gm.dot(jv, w) == pytest.approx(float(np.vdot(v, gm.jtw(w))), rel=1e-9)
    assert max(errs) < 1e-2, errs


def test_gradient_equals_objective_gradient_with_penetration():
    """J^T r must be grad J_smooth at z_k (floors inactive): the penetration term enters the GN model in full."""
    ex, rw, u0 = toy_problem(T=40, W=30, ref_shift=0.01, press=True)
    rw.ped_scale = 0.0
    cfg = D4PrimeConfig(enabled=True, knot_steps=10, workers=0, a_floor=1e-12, b_floor=1e-12)
    basis = G.KnotBasis(40, cfg.knot_steps)
    sigma = G.knot_sigma(ex.model.nu, cfg)
    ro = rollout(ex, rw, u0)
    sc = step_costs(ex, rw, ro)
    assert (sc.h > 0).sum() >= 5
    with FDPool(ex, 0) as pool:
        lin = linearize(pool, ro, eps=1e-6, centered=True)
    gm = G.GNModel(ex, rw, ro, sc, lin, basis, sigma, cfg)
    g = gm.jtw(gm.r)
    cq, cv, cp = gm.c
    q_k = 0.5 * (gm.dot(gm.r, gm.r) + float(np.sum(sc.mult * (np.maximum(sc.a, 1e-12) + rw.vel_scale * np.maximum(sc.b, 1e-12)))))
    assert q_k == pytest.approx(sc.total, rel=1e-9)        # the majoriser touches J at z_k
    rng = np.random.default_rng(7)
    for _ in range(3):
        v = rng.normal(size=g.shape)
        h = 1e-5
        jp = step_costs(ex, rw, rollout(ex, rw, u0 + basis.apply(h * v * sigma))).total
        jm = step_costs(ex, rw, rollout(ex, rw, u0 - basis.apply(h * v * sigma))).total
        assert float(np.vdot(g, v)) == pytest.approx((jp - jm) / (2 * h), rel=5e-2)   # the half-weight bug: 36-79% off


def test_lm_reduces_cost_on_toy():
    ex, rw, u0 = toy_problem(T=60, W=20, ref_shift=0.015)
    with FDPool(ex, 0) as pool:
        res = G.optimize(ex, rw, u0, CFG, pool)
    Js = [res.J0] + [r.J for r in res.history if r.accepted]
    assert len(Js) >= 3
    assert all(b < a for a, b in zip(Js, Js[1:]))
    assert res.costs.total < 0.7 * res.J0
    assert all(r.fd_steps == 60 * fd_steps_per_row(ex.model, False) for r in res.history)


def test_never_worse_acceptance(monkeypatch):
    ex, rw, u0 = toy_problem(T=40, W=10, ref_shift=0.01)
    real = G.rollout
    calls = {"n": 0}

    def worse(ex_, rw_, ctrls, *a, **k):
        calls["n"] += 1
        ro = real(ex_, rw_, ctrls, *a, **k)
        if calls["n"] > 1:                     # every trial after the nominal: blow the object away
            ro.qpos = ro.qpos.copy()
            ro.qpos[:, 7] += 1.0
        return ro

    monkeypatch.setattr(G, "rollout", worse)
    with FDPool(ex, 0) as pool:
        res = G.optimize(ex, rw, u0, D4PrimeConfig(enabled=True, knot_steps=10, workers=0, max_trials=3), pool)
    assert res.stop == "stalled"
    assert np.all(res.z == 0.0)
    assert res.costs.total == res.J0
    rec = res.history[0]
    assert not rec.accepted and rec.trials == 3


def test_opt_in_and_config_round_trip():
    ex, rw, u0 = toy_problem(T=20, W=5)
    with FDPool(ex, 0) as pool, pytest.raises(RuntimeError):
        G.optimize(ex, rw, u0, D4PrimeConfig(), pool)
    assert PlannerConfig().d4p.enabled is False and PlannerConfig().planner == "ref"
    p = PlannerConfig(planner="d4p", d4p=D4PrimeConfig(enabled=True, max_iters=7))
    assert from_dict(json.loads(json.dumps(p.to_dict()))) == p


def test_parallel_linearization_equals_serial():
    ex, rw, u0 = toy_problem(T=40, W=10)
    ro = rollout(ex, rw, u0)
    with FDPool(ex, 0) as p0:
        a = linearize(p0, ro)
    with FDPool(ex, 2) as p2:
        b = linearize(p2, ro)
    assert np.array_equal(a.A, b.A) and np.array_equal(a.B, b.B)


def test_rollout_flags_instability(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)          # MuJoCo writes MUJOCO_LOG.TXT into the working directory on a warning
    ex, rw, u0 = toy_problem(T=20, W=5)
    bad = u0.copy()
    bad[8:, 3] = 1e9
    ro = rollout(ex, rw, bad)
    assert not ro.finite or ro.terminated_at is not None
    assert not np.isfinite(step_costs(ex, rw, ro).total)
    assert isinstance(ro, Rollout)
