"""T0' grasp pre-load: QP known answers, an analytic two-finger pinch, and a MuJoCo mutation test (the toy hand
drops the box under the plain kinematic warm start and holds it under T0')."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from fastdo.physics.dense_qp import solve_qp
from fastdo.physics.grasp_preload import (GraspContact, PreloadConfig, actuator_gains, closure_offset, grasp_forces,
                                          object_wrench_demand, preload_controls, pyramid_edges)


def test_qp_matches_kkt_on_equality_problem():
    rng = np.random.default_rng(0)
    L = rng.standard_normal((6, 6))
    H = L @ L.T + np.eye(6)
    c = rng.standard_normal(6)
    A = rng.standard_normal((2, 6))
    b = rng.standard_normal(2)
    K = np.block([[H, A.T], [A, np.zeros((2, 2))]])
    x_ref = np.linalg.solve(K, np.concatenate([-c, b]))[:6]
    r = solve_qp(H, c, A, b)
    assert r.status == "optimal"
    np.testing.assert_allclose(r.x, x_ref, atol=1e-8)


def test_qp_box_projection_known_answer():
    # min 1/2 |x - p|^2 s.t. 0 <= x <= 1  ->  clip(p, 0, 1)
    p = np.array([-0.5, 0.3, 1.7, 0.99])
    G = np.vstack([np.eye(4), -np.eye(4)])
    h = np.concatenate([np.zeros(4), -np.ones(4)])
    r = solve_qp(np.eye(4), -p, G=G, h=h, tol=1e-12)
    assert r.status == "optimal"
    np.testing.assert_allclose(r.x, np.clip(p, 0, 1), atol=1e-8)


def _pinch(mu: float):
    a = 0.03
    cs = [GraspContact(0, 1, np.array([-a, 0, 0]), np.array([-a, 0, 0]), np.array([1.0, 0, 0]), 0.0),
          GraspContact(1, 2, np.array([a, 0, 0]), np.array([a, 0, 0]), np.array([-1.0, 0, 0]), 0.0)]
    return cs


@pytest.mark.parametrize("mu", [0.5, 1.0, 1.5])
def test_two_finger_pinch_minimal_squeeze_is_mg_over_2mu(mu):
    m, g = 0.2, 9.81
    w = object_wrench_demand(m, np.full(3, 1e-4), np.eye(3), np.zeros(3), np.zeros(3), np.zeros(3), np.array([0, 0, -g]))
    np.testing.assert_allclose(w, [0, 0, m * g, 0, 0, 0])
    cfg = PreloadConfig(s_mu=1.0, s_w=1.0, f_min_n=0.0, eps=1e-9, lam=1e6)
    out = grasp_forces(_pinch(mu), np.zeros(3), w, mu, cfg)
    np.testing.assert_allclose(out["fn"], [m * g / (2 * mu)] * 2, rtol=2e-3)
    assert out["residual"] < 1e-4
    # every contact force inside its (linearised) cone and the pair balances the weight
    np.testing.assert_allclose(out["f"].sum(0), [0, 0, m * g], atol=1e-4)


def test_pinch_safety_factors_scale_the_squeeze():
    m, g, mu = 0.2, 9.81, 1.0
    w = np.array([0, 0, m * g, 0, 0, 0])
    out = grasp_forces(_pinch(mu), np.zeros(3), w, mu, PreloadConfig(s_mu=2.0, s_w=1.5, f_min_n=0.0, lam=1e6))
    np.testing.assert_allclose(out["fn"], [1.5 * m * g / (2 * mu / 2.0)] * 2, rtol=2e-3)


def test_one_sided_contact_cannot_balance_and_reports_the_residual():
    m, g = 0.2, 9.81
    out = grasp_forces(_pinch(1.0)[:1], np.zeros(3), np.array([0, 0, m * g, 0, 0, 0]), 1.0,
                       PreloadConfig(s_mu=1.0, s_w=1.0, f_min_n=0.0))
    assert out["residual"] > 0.5


def test_pyramid_edges_lie_on_the_cone():
    n = np.array([0.0, 0.6, 0.8])
    E = pyramid_edges(n, 0.7)
    for e in E:
        tang = e - (e @ n) * n
        assert e @ n == pytest.approx(1.0)
        assert np.linalg.norm(tang) == pytest.approx(0.7)


def test_closure_offset_minimal_norm_known_answer():
    cfg = PreloadConfig(p0_m=0.0005, pmax_m=0.002)
    # one contact, moving joint 0 by 1 moves the contact 0.05 m along the normal: needed overlap 3 mm gap + 0.5 mm
    jn = np.array([[0.05, 0.0]])
    rho = 1e5
    dq, st = closure_offset(jn, np.array([0.003]), cfg, rho=rho)
    assert st == "optimal"
    # soft band: min x^2 + rho (0.0035 - 0.05 x)^2  ->  x = rho 0.05 0.0035 / (1 + rho 0.05^2)
    np.testing.assert_allclose(dq, [rho * 0.05 * 0.0035 / (1 + rho * 0.05 ** 2), 0.0], rtol=1e-6, atol=1e-9)
    # a stiff band reproduces the hard-constraint minimal norm 0.0035 / 0.05
    dq_h, _ = closure_offset(jn, np.array([0.003]), cfg, rho=1e10)
    np.testing.assert_allclose(dq_h, [0.0035 / 0.05, 0.0], rtol=1e-5, atol=1e-9)
    # two joints sharing the move: minimal norm splits along jn
    jn2 = np.array([[0.03, 0.04]])
    dq2, _ = closure_offset(jn2, np.array([0.003]), cfg, rho=1e10)
    np.testing.assert_allclose(dq2, jn2[0] * 0.0035 / (jn2[0] @ jn2[0]), rtol=1e-5, atol=1e-9)


TOY = """
<mujoco>
  <option timestep="0.002" gravity="0 0 -9.81" cone="pyramidal"/>
  <worldbody>
    <body name="right_palm" pos="0 0 0.3">
      <joint name="right_pos_x" type="slide" axis="1 0 0" armature="1"/>
      <joint name="right_pos_y" type="slide" axis="0 1 0" armature="1"/>
      <joint name="right_pos_z" type="slide" axis="0 0 1" armature="1"/>
      <joint name="right_rot_x" type="hinge" axis="1 0 0" armature="1"/>
      <joint name="right_rot_y" type="hinge" axis="0 1 0" armature="1"/>
      <joint name="right_rot_z" type="hinge" axis="0 0 1" armature="1"/>
      <geom name="palm_visual" type="box" size="0.01 0.01 0.01" mass="0.5" contype="0" conaffinity="0"/>
      <body name="right_thumb_DP" pos="-0.05 0 0">
        <joint name="right_thumb_IP" type="slide" axis="1 0 0" range="-0.05 0.05" armature="0.1"/>
        <geom name="collision_hand_right_thumb_0" type="box" size="0.01 0.02 0.02" mass="0.05" group="3"
              contype="0" conaffinity="0"/>
      </body>
      <body name="right_index_DP" pos="0.05 0 0">
        <joint name="right_index_DIP" type="slide" axis="-1 0 0" range="-0.05 0.05" armature="0.1"/>
        <geom name="collision_hand_right_index_0" type="box" size="0.01 0.02 0.02" mass="0.05" group="3"
              contype="0" conaffinity="0"/>
      </body>
    </body>
    <body name="right_object" pos="0 0 0.3">
      <freejoint/>
      <geom name="right_object_0" type="box" size="0.03 0.02 0.02" mass="0.2" group="3" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
  <contact>
    <pair geom1="collision_hand_right_thumb_0" geom2="right_object_0" condim="3" friction="1 1 0.005 0.0001 0.0001"/>
    <pair geom1="collision_hand_right_index_0" geom2="right_object_0" condim="3" friction="1 1 0.005 0.0001 0.0001"/>
  </contact>
  <actuator>
    <position joint="right_pos_x" kp="1000" kv="100" ctrlrange="-5 5"/>
    <position joint="right_pos_y" kp="1000" kv="100" ctrlrange="-5 5"/>
    <position joint="right_pos_z" kp="1000" kv="100" ctrlrange="-5 5"/>
    <position joint="right_rot_x" kp="1000" kv="100" ctrlrange="-6 6"/>
    <position joint="right_rot_y" kp="1000" kv="100" ctrlrange="-6 6"/>
    <position joint="right_rot_z" kp="1000" kv="100" ctrlrange="-6 6"/>
    <position joint="right_thumb_IP" kp="300" kv="30" ctrlrange="-0.05 0.05"/>
    <position joint="right_index_DIP" kp="300" kv="30" ctrlrange="-0.05 0.05"/>
  </actuator>
</mujoco>
"""


def _toy_reference(model: mujoco.MjModel, T: int) -> tuple[np.ndarray, np.ndarray]:
    """Static reference: hand at rest, finger pads 0.2 mm off the box faces, box in place."""
    q = np.zeros(model.nq)
    q[6] = q[7] = 0.0098                                    # pads (half-width 0.01) at x = -0.0302 / +0.0302
    q[8:11] = [0.0, 0.0, 0.3]
    q[11:15] = [1, 0, 0, 0]
    qref = np.repeat(q[None], T, axis=0)
    return qref, qref[:, :model.nu].copy()


W_TOY = 100                                                 # executor-like warmup: zero gravity, box welded in place


def _hold(model: mujoco.MjModel, qref: np.ndarray, ctrl: np.ndarray) -> float:
    """Executor semantics on the toy: the box is pinned (qpos/qvel reset) and gravity is off before W, released at W."""
    d = mujoco.MjData(model)
    d.qpos[:] = qref[0]
    mujoco.mj_forward(model, d)
    g = model.opt.gravity.copy()
    for t in range(ctrl.shape[0]):
        model.opt.gravity[:] = 0.0 if t < W_TOY else g
        if t < W_TOY:
            d.qpos[8:15] = qref[0, 8:15]
            d.qvel[8:14] = 0.0
        d.ctrl[:] = ctrl[t]
        mujoco.mj_step(model, d)
    model.opt.gravity[:] = g
    return float(qref[0, 10] - d.qpos[10])                 # how far the box fell (m)


def test_actuator_gains_read_position_actuators():
    m = mujoco.MjModel.from_xml_string(TOY)
    dof, kp, kv = actuator_gains(m)
    np.testing.assert_array_equal(dof, np.arange(8))
    np.testing.assert_allclose(kp, [1000] * 6 + [300] * 2)
    np.testing.assert_allclose(kv, [100] * 6 + [30] * 2)


def test_all_terms_off_is_the_identity():
    m = mujoco.MjModel.from_xml_string(TOY)
    qref, cref = _toy_reference(m, 50)
    cfg = PreloadConfig(dynamics_ff=False, closure=False, forces=False)
    out = preload_controls(m, qref, cref, 0, m.opt.gravity.copy(), "right", cfg)
    np.testing.assert_array_equal(out["ctrl"], cref)


def test_mutation_kinematic_warm_start_drops_and_preload_holds():
    m = mujoco.MjModel.from_xml_string(TOY)
    T = 600                                                 # 0.2 s warmup + 1 s held
    qref, cref = _toy_reference(m, T)
    drop_t0 = _hold(m, qref, cref)                          # mutation: no squeeze -> the box slides out
    assert drop_t0 > 0.05
    cfg = PreloadConfig(ramp_steps=50, smooth=1, stride=10)
    out = preload_controls(m, qref, cref, W_TOY, m.opt.gravity.copy(), "right", cfg)
    assert (out["n_contacts"] == 2).all()
    # both fingers close inward (thumb +x, index along its -x axis = +q) by the pre-load
    assert (out["ctrl"][W_TOY:, 6] > cref[W_TOY:, 6]).all() and (out["ctrl"][W_TOY:, 7] > cref[W_TOY:, 7]).all()
    drop_t0p = _hold(m, qref, out["ctrl"])
    assert drop_t0p < 0.005
    # the force term alone (no gap closure): finger offset = f_n / kp exactly (slide joints along the normals), and
    # f_n is at least the analytic minimal squeeze s_w m g / (2 mu / s_mu). (GJK reports a face-face contact at a pad
    # CORNER, off the COM, so balancing the torque costs more than the centred-pinch minimum: conservative.)
    cfg_f = PreloadConfig(ramp_steps=50, smooth=1, stride=10, closure=False, dynamics_ff=False, f_min_n=0.0, lam=1e6)
    out_f = preload_controls(m, qref, cref, W_TOY, m.opt.gravity.copy(), "right", cfg_f)
    fn_min = 1.5 * 0.2 * 9.81 / (2 * 1.0 / 2.0)
    k = len(out_f["rows"]) - 1
    fn_each = out_f["fn_sum"][k] / 2
    np.testing.assert_allclose(out_f["offset_force"][k, 6:8], [fn_each / 300] * 2, rtol=1e-6)
    assert fn_each >= fn_min * (1 - 2e-3)
    assert out_f["residual"][k] < 0.05          # soft wrench equality (lam): within 5% of the 1.5x demand
