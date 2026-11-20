"""Toy single-hand scene for the D4' tests: a 7-dof position-servoed "hand" (wrist slides + hinges + one finger)
and a free box with a warmup weld, named like DO AS I DO's scenes (collision_hand_*, right_object*, right_pedestal_*)."""

from __future__ import annotations

import mujoco
import numpy as np

from fastdo.physics.fd_problem import RewardSpec, build_executor

XML = """
<mujoco>
  <option timestep="0.005" integrator="implicitfast" iterations="100" ls_iterations="200"/>
  <default><joint armature="0.01" damping="2"/></default>
  <worldbody>
    <geom name="right_pedestal_0" type="cylinder" size="0.05 0.05" pos="0.3 0 0.05"/>
    <geom name="table" type="box" size="TABLE TABLE 0.01" pos="0 0 0.17" friction="0.3"/>
    <body name="right_palm" pos="0 0 0.3">
      <joint name="right_pos_x" type="slide" axis="1 0 0"/>
      <joint name="right_pos_y" type="slide" axis="0 1 0"/>
      <joint name="right_pos_z" type="slide" axis="0 0 1"/>
      <joint name="right_rot_x" type="hinge" axis="1 0 0"/>
      <joint name="right_rot_y" type="hinge" axis="0 1 0"/>
      <joint name="right_rot_z" type="hinge" axis="0 0 1"/>
      <geom name="collision_hand_palm" type="box" size=".04 .04 .01" mass="0.5"/>
      <body name="right_finger" pos="0.05 0 -0.02">
        <joint name="right_f1" type="hinge" axis="0 1 0" range="-1 1"/>
        <geom name="collision_hand_f1" type="capsule" fromto="0 0 0 0 0 -0.05" size=".01" mass="0.05"/>
      </body>
    </body>
    <body name="right_object" pos="0.0 0 0.2">
      <freejoint/>
      <geom name="right_object_0" type="box" size=".02 .02 .02" mass="0.1"/>
    </body>
  </worldbody>
  <equality><weld name="right_object_weld" body1="right_object" active="false"/></equality>
  <actuator>
    <position joint="right_pos_x" kp="300"/><position joint="right_pos_y" kp="300"/>
    <position joint="right_pos_z" kp="300"/><position joint="right_rot_x" kp="5"/>
    <position joint="right_rot_y" kp="5"/><position joint="right_rot_z" kp="5"/>
    <position joint="right_f1" kp="2"/>
  </actuator>
</mujoco>
"""


def toy_model(table_half: float = 0.2) -> mujoco.MjModel:
    """The toy scene; a small table (`table_half` 0.025) leaves the finger free so the palm can reach the object."""
    m = mujoco.MjModel.from_xml_string(XML.replace("TABLE", repr(float(table_half))))
    for b in range(m.nbody):
        if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) == "right_palm":
            m.body_gravcomp[b] = 0.0
    return m


def _names(m: mujoco.MjModel) -> list[str]:
    return [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(m.ngeom)]


def toy_problem(T: int = 80, W: int = 20, ref_shift: float = 0.0, pen_scale: float = 3000.0, press: bool = False):
    """(executor, reward spec, warm-start controls). The reference is the warm start's own rollout, with the wrist
    x reference shifted by `ref_shift` after the release (so the optimum moves the controls). The object rests on a
    table under the hand, without hand contact (a smooth regime for derivative checks)."""
    m = toy_model(0.025 if press else 0.2)
    q0 = np.zeros(m.nq)
    q0[7:10] = [0.0, 0.0, 0.2]
    q0[10] = 1.0
    ctrl0 = np.zeros(m.nu)
    ex = build_executor(m, q0, ctrl0, W, np.array([0.0, 0.0, -9.81]), [0], T)
    t = np.arange(T) / T
    u0 = np.zeros((T, m.nu))
    u0[:, 0] = 0.02 * np.sin(2 * np.pi * t)
    u0[:, 2] = -0.08 + 0.01 * t
    u0[:, 6] = 0.3 * t
    if press:
        u0[:, 2] = -0.09
    names = _names(ex.model)
    hand = np.array(["collision_hand_" in n for n in names])
    obj = np.array(["object" in n for n in names])
    from fastdo.physics.fd_problem import rollout

    rw0 = RewardSpec(qpos_ref=np.zeros((T + 2, m.nq)), qvel_ref=np.zeros((T + 2, m.nv)), w_dq=np.ones(m.nv),
                     vel_scale=1e-4, pen_scale=pen_scale, pen_margin=0.005, terminal_scale=10.0, warmup_mult=0.1,
                     hand_geom=hand, object_geom=obj)
    base = rollout(ex, rw0, u0)
    qref = np.concatenate([ex.q0[None], base.qpos, base.qpos[-1:]], axis=0)
    qref[W + 1:, 0] += ref_shift
    vref = np.concatenate([np.zeros((1, m.nv)), base.qvel, base.qvel[-1:]], axis=0)
    w = np.full(m.nv, 0.01)
    w[:3], w[3:6] = 0.1, 0.03
    w[-6:-3], w[-3:] = 1.0, 0.3
    rw = RewardSpec(qpos_ref=qref, qvel_ref=vref, w_dq=w, vel_scale=1e-4, pen_scale=pen_scale, pen_margin=0.005,
                    terminal_scale=10.0, warmup_mult=0.1, hand_geom=hand, object_geom=obj, ped_scale=0.5,
                    side_object_geom=np.array([n.startswith("right_object") for n in names]),
                    pedestal_geom=np.array([n.startswith("right_pedestal_") for n in names]),
                    gate_in_hand=np.zeros(T + 2, bool), gate_near_ped=np.zeros(T + 2, bool))
    return ex, rw, u0
