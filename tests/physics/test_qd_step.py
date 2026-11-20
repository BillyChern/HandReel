"""Known-answer tests of the quasi-dynamic convex contact step (`fastdo.physics.qd`).

KA1 rest: a box on a table with no hand contact stays at rest (gravity balanced by the table); without the table it
    falls at the quasi-dynamic speed h g (no momentum).
KA2 push: a fingertip pushing a box with force F = kp delta (fingertip blocked by the box) leaves it at rest iff
    F <= mu m g, and slides it otherwise (thresholds at +-1%).
KA3 pinch: two antipodal fingertips squeezing with F each hold a free box against gravity iff mu >= mu*, with mu*
    computed here from MuJoCo's contact frames: the polyhedral cone's vertical capacity per contact is
    mu F max_k d_k . z, torque balance about the grasp axis needs equal shares, so mu* = m g / (2 F min_i h_i).
KA4 gradient: implicit-differentiation d v / d w (w = commanded hand velocity) matches central finite differences at
    a non-degenerate sliding configuration.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qd_scenes import HALF, KP, MASS, R_TIP, pinch_scene, push_scene

from fastdo.physics.qd import (
    QDConfig,
    active_set,
    build_model,
    ctrl_from_command_velocity,
    dv_dcmd_vel,
    replay,
    step,
)

G = 9.81
H = 0.005
STILL = 1e-8          # m/s: "at rest"
MOVING = 1e-6         # m/s: "moves"


def _box_speed(v: np.ndarray) -> float:
    return float(np.linalg.norm(v[:6]))


def test_ka1_box_on_table_stays_at_rest():
    m = push_scene(tip_x=-0.2)                        # fingertip 14 cm away: no hand contact
    qm = build_model(m, QDConfig(tol=1e-12))
    q = m.qpos0.copy()
    r = step(qm, q, np.zeros(3))
    assert r.sol.status == "optimal"
    hand_rows = [i for i, gg in enumerate(r.contacts.geom) if "tip" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM,
                                                                                         int(gg[0])) or "")]
    assert not hand_rows and r.contacts.dist.size >= 3      # the table supports the box, nothing else touches it
    assert _box_speed(r.v) < STILL
    ro = replay(qm, q, np.zeros((200, 3)), obj_z_adr=2)
    assert np.abs(ro.qpos[:, :7] - q[:7]).max() < 1e-7       # 1 s of steps: no drift


def test_ka1_free_box_falls_at_quasi_dynamic_speed():
    m = push_scene(floor=False, tip_x=-0.2)
    qm = build_model(m, QDConfig(tol=1e-12))
    r = step(qm, m.qpos0.copy(), np.zeros(3))
    np.testing.assert_allclose(r.v[:6], [0, 0, -H * G, 0, 0, 0], atol=1e-9)


@pytest.mark.parametrize("ratio, slides", [(0.99, False), (1.01, True), (0.5, False), (2.0, True)])
def test_ka2_push_slides_iff_force_exceeds_mu_n(ratio: float, slides: bool):
    mu = 0.5
    m = push_scene(mu_floor=mu)
    qm = build_model(m, QDConfig(tol=1e-12))
    q = m.qpos0.copy()
    delta = ratio * mu * MASS * G / KP                 # commanded overlap: blocked force kp delta = ratio mu m g
    r = step(qm, q, np.array([delta, 0.0, 0.0]))
    assert r.sol.status == "optimal"
    if slides:
        assert r.v[0] > MOVING
        assert r.v[0] == pytest.approx(r.v[6], rel=1e-6)      # the fingertip stays on the face it pushes
    else:
        assert _box_speed(r.v) < STILL
    gap = (r.q_next[0] - HALF) - (r.q_next[7] - (HALF + R_TIP) + R_TIP)   # box -x face minus fingertip front
    assert gap > -1e-9                                         # no penetration after the step


def _pinch_threshold(n_edges: int, force: float) -> float:
    m = pinch_scene(1.0, margin=0.005)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    assert d.ncon == 2
    th = 2.0 * np.pi * np.arange(n_edges) / n_edges           # the registered edges, computed independently here
    E = np.stack([np.cos(th), np.sin(th)], axis=1)
    h = []
    for i in range(d.ncon):
        fr = np.asarray(d.contact[i].frame).reshape(3, 3)
        dk = E[:, 0, None] * fr[1] + E[:, 1, None] * fr[2]
        h.append(float(np.max(dk @ np.array([0.0, 0.0, 1.0]))))
    return MASS * G / (2.0 * force * min(h))


@pytest.mark.parametrize("n_edges", [4, 8])
@pytest.mark.parametrize("ratio, holds", [(1.01, True), (0.99, False), (1.5, True), (0.5, False)])
def test_ka3_pinch_holds_iff_mu_exceeds_threshold(n_edges: int, ratio: float, holds: bool):
    delta = 0.005
    mu_star = _pinch_threshold(n_edges, KP * delta)
    assert mu_star == pytest.approx(MASS * G / (2 * KP * delta), rel=0.1)   # near the Coulomb (SOC) value
    m = pinch_scene(ratio * mu_star)
    qm = build_model(m, QDConfig(tol=1e-12, n_edges=n_edges))
    r = step(qm, m.qpos0.copy(), np.array([delta, -delta]))
    assert r.sol.status == "optimal"
    if holds:
        assert _box_speed(r.v) < STILL
    else:
        assert r.v[2] < -MOVING


def test_ka4_implicit_gradient_matches_finite_differences():
    m = push_scene(mu_floor=0.5, mu_tip=0.2)
    qm = build_model(m, QDConfig(tol=1e-13))
    q = m.qpos0.copy()
    w = np.array([1.5 * 0.5 * MASS * G / KP / H, 0.3, 0.2])    # sliding push plus lateral fingertip motion

    def solve(wv: np.ndarray):
        return step(qm, q, ctrl_from_command_velocity(qm, q, wv))

    r = solve(w)
    s = r.qp.G @ r.v - r.qp.g
    assert np.min(np.maximum(r.z, s)) > 1e-4                   # strictly complementary: the KA is well posed
    assert active_set(r).sum() > 0 and r.v[0] > MOVING
    J = dv_dcmd_vel(qm, r)
    eps = 1e-5
    Jfd = np.zeros_like(J)
    for j in range(3):
        e = np.zeros(3)
        e[j] = eps
        Jfd[:, j] = (solve(w + e).v - solve(w - e).v) / (2 * eps)
    assert np.abs(Jfd[:6]).max() > 1e-2                        # the object really responds to the command
    np.testing.assert_allclose(J, Jfd, atol=1e-6 * np.abs(Jfd).max(), rtol=0)


def test_ka4_stuck_box_has_zero_gradient():
    m = push_scene(mu_floor=0.5)
    qm = build_model(m, QDConfig(tol=1e-13))
    q = m.qpos0.copy()
    w = np.array([0.5 * 0.5 * MASS * G / KP / H, 0.0, 0.0])
    r = step(qm, q, ctrl_from_command_velocity(qm, q, w))
    assert np.abs(dv_dcmd_vel(qm, r)[:6]).max() < 1e-8


def test_build_model_rejects_unsupported_structure():
    xml = """<mujoco><worldbody><body><joint type="hinge"/><geom size="0.1"/></body></worldbody></mujoco>"""
    with pytest.raises(ValueError, match="neither actuated nor free"):
        build_model(mujoco.MjModel.from_xml_string(xml))
