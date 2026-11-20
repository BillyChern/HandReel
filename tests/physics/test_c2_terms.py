"""C2 tier-0 terms on a toy two-finger hand holding a box (MuJoCo CPU)."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from fastdo.physics.c2_terms import (
    contacts_at,
    finger_of,
    frames_to_steps,
    grasp_force_nnls,
    hand_info,
    squeeze_frame,
    squeeze_trajectory,
)
from fastdo.physics.planner_config import SqueezeConfig

INERT = '<inertial pos="0 0 0" mass="{m}" diaginertia="1e-4 1e-4 1e-4"/>'
XML = f"""
<mujoco>
  <worldbody>
    <body name="right_base_tx">{INERT.format(m=0.1)}<joint name="right_pos_x" type="slide" axis="1 0 0"/>
     <body name="right_base_ty">{INERT.format(m=0.1)}<joint name="right_pos_y" type="slide" axis="0 1 0"/>
      <body name="right_base_tz">{INERT.format(m=0.1)}<joint name="right_pos_z" type="slide" axis="0 0 1"/>
       <body name="right_base_roll">{INERT.format(m=0.1)}<joint name="right_rot_x" axis="1 0 0"/>
        <body name="right_base_pitch">{INERT.format(m=0.1)}<joint name="right_rot_y" axis="0 1 0"/>
         <body name="right_base_yaw">{INERT.format(m=0.1)}<joint name="right_rot_z" axis="0 0 1"/>
          <body name="right_hand_C_MC">
           <geom name="palm" type="box" size="0.06 0.02 0.005" mass="0.5" group="3"/>
           <body name="right_index_PP" pos="0.05 0 0">
            <joint name="right_index_J" axis="0 1 0" range="-1 1"/>
            <geom name="idx" type="capsule" fromto="0 0 0 0 0 -0.04" size="0.008" mass="0.04" group="3"/>
           </body>
           <body name="right_thumb_PP" pos="-0.05 0 0">
            <joint name="right_thumb_J" axis="0 1 0" range="-1 1"/>
            <geom name="thb" type="capsule" fromto="0 0 0 0 0 -0.04" size="0.008" mass="0.04" group="3"/>
           </body>
          </body>
         </body>
        </body>
       </body>
      </body>
     </body>
    </body>
    <body name="right_object" pos="0 0 -0.03">
     <freejoint/>
     <geom name="obj" type="box" size="0.038 0.02 0.02" mass="0.2" group="3"/>
    </body>
  </worldbody>
  <contact>
    <pair geom1="idx" geom2="obj" friction="1.5 1.5 0.15 0 0"/>
    <pair geom1="thb" geom2="obj" friction="1.5 1.5 0.15 0 0"/>
    <pair geom1="palm" geom2="obj" friction="1.5 1.5 0.15 0 0"/>
  </contact>
  <actuator>
    <position joint="right_pos_x" kp="1000"/><position joint="right_pos_y" kp="1000"/>
    <position joint="right_pos_z" kp="1000"/><position joint="right_rot_x" kp="1000"/>
    <position joint="right_rot_y" kp="1000"/><position joint="right_rot_z" kp="1000"/>
    <position joint="right_index_J" kp="300"/><position joint="right_thumb_J" kp="300"/>
  </actuator>
</mujoco>
"""


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(XML)


def test_finger_of() -> None:
    assert finger_of("right_index_MP") == 1
    assert finger_of("right_thumb_DP") == 0
    assert finger_of("right_hand_C_MC") is None


def test_hand_info_gravity_and_pairs(model: mujoco.MjModel) -> None:
    info = hand_info(model, "right")
    tz = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_base_tz")
    assert info.z_ctrl == 2 and info.kp_z == pytest.approx(1000.0)
    assert info.m_hand == pytest.approx(float(model.body_subtreemass[tz]))
    assert info.m_obj == pytest.approx(0.2)
    assert len(info.pairs) == 3
    assert sorted(v for v in info.hand_geom_finger.values() if v is not None) == [0, 1]
    assert info.finger_ctrl.tolist() == [6, 7]


def _data(model: mujoco.MjModel, idx_q: float = 0.0, thb_q: float = 0.0, obj_x: float = 0.0) -> mujoco.MjData:
    d = mujoco.MjData(model)
    d.qpos[6], d.qpos[7] = idx_q, thb_q
    d.qpos[8] = obj_x
    d.qpos[10] = -0.03
    d.qpos[11] = 1.0
    mujoco.mj_forward(model, d)
    return d


def test_contacts_gap_and_normal_into_object(model: mujoco.MjModel) -> None:
    info = hand_info(model, "right")
    cs = {c.finger: c for c in contacts_at(model, _data(model), info, 0.005)}
    assert set(cs) == {0, 1}                       # palm excluded
    assert cs[1].gap == pytest.approx(0.004, abs=5e-4)
    assert cs[1].normal[0] == pytest.approx(-1.0, abs=1e-3)   # index (+x side) pushes toward -x
    assert cs[0].normal[0] == pytest.approx(1.0, abs=1e-3)
    pen = {c.finger: c for c in contacts_at(model, _data(model, obj_x=0.007), info, 0.005)}
    assert pen[1].gap == pytest.approx(-0.003, abs=5e-4)       # penetrating: negative gap, normal still inward
    assert pen[1].normal[0] == pytest.approx(-1.0, abs=1e-3)


def test_grasp_force_nnls_balances_weight(model: mujoco.MjModel) -> None:
    info = hand_info(model, "right")
    d = _data(model)
    cs = contacts_at(model, d, info, 0.005)
    w = np.array([0.0, 0.0, 0.2 * 9.81])
    f_n, res = grasp_force_nnls(cs, d.xipos[info.obj_body].copy(), w, mu=1.05)
    assert res < 0.05
    assert f_n.min() > 0.0
    assert f_n[0] == pytest.approx(f_n[1], rel=0.2)
    assert f_n.sum() * 1.05 >= 0.2 * 9.81 * 0.95


def test_squeeze_moves_fingers_into_object(model: mujoco.MjModel) -> None:
    info = hand_info(model, "right")
    d = _data(model)
    cs = contacts_at(model, d, info, 0.005)
    cfg = SqueezeConfig()
    gap_q, fing_q = squeeze_frame(model, d, info, cs, np.array([1.0, 1.0]), cfg)
    # closing the 4 mm gap: index rotates +y (tip toward -x), thumb -y (tip toward +x)
    assert gap_q[0] > 0 and gap_q[1] < 0
    assert fing_q[1, 0] > 0 and fing_q[0, 1] < 0
    assert fing_q[2:].sum() == 0.0
    d2 = _data(model, idx_q=float(gap_q[0] + fing_q[1, 0]), thb_q=float(gap_q[1] + fing_q[0, 1]))
    after = {c.finger: c.gap for c in contacts_at(model, d2, info, 0.01)}
    assert after[1] < 0 and after[0] < 0                 # both now penetrate (the commanded squeeze)
    assert after[1] > -cfg.pen_max_m - 0.002


def test_squeeze_trajectory_shapes(model: mujoco.MjModel) -> None:
    info = hand_info(model, "right")
    q = np.tile(_data(model).qpos.copy(), (6, 1))
    q[:, 10] += np.linspace(0, 0.05, 6)                    # object lifted (moves with the hand in reference)
    q[:, 2] += np.linspace(0, 0.05, 6)
    out = squeeze_trajectory(model, q, info, SqueezeConfig(), 1 / 30, 1.05)
    assert out["gap"].shape == (6, 2) and out["finger"].shape == (6, 5, 2)
    assert out["held"].tolist() == [1.0] * 6
    assert (out["n_contacts"] == 2).all()


def test_frames_to_steps_ramp_interp_hold() -> None:
    per = np.array([[1.0], [3.0], [5.0]])
    out = frames_to_steps(per, warmup_steps=10, steps_per_frame=2, total_steps=20, ramp_steps=4, smooth_frames=1)
    assert out.shape == (20, 1)
    assert out[5, 0] == 0.0 and out[9, 0] == pytest.approx(0.75)    # ramp 6..10 toward frame 0
    assert out[10, 0] == pytest.approx(1.0)
    assert out[15, 0] == pytest.approx(5.0) and out[19, 0] == pytest.approx(5.0)   # last frame at 10+5, then hold
    assert out[11, 0] == pytest.approx(1.8)
