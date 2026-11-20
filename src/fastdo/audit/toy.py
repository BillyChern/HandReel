# Copyright 2026 The HandReel authors. SPDX-License-Identifier: Apache-2.0
"""A synthetic MuJoCo scene with closed-form contact answers, for the worked example and the known-answer tests.

World: a fixed start pedestal (top at z = 0.1 m), a free cube object (`right_object`, mass 0.2 kg, half size
0.025 m) and a gravity-compensated hand plate (`collision_hand_palm`) on a vertical slide joint driven by a position
servo (kp 300). Every geom has contype = conaffinity = 0, so contact arises only through the explicit `<pair>`s, as
in DO AS I DO's scenes (a scene census found 0 default-collision geoms on every clip).

States, all settled by stepping the scene:
- REST: cube on the pedestal, plate 0.2 m above it. No hand-object contact: grip 0.
- PRESS: cube on the pedestal, servo target below the cube's top, so the plate presses down. At equilibrium the
  hand-object normal force equals the servo's force |actuator_force| (the plate is gravity-compensated); the
  pedestal carries that force plus the weight.
- CARRY: no pedestal under the cube (`CARRY_XML`); the cube rests on the plate. Hand-object normal force = the weight.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "CARRY_XML",
    "HALF",
    "MASS",
    "PED_TOP",
    "TOP",
    "WEIGHT",
    "XML",
    "G",
    "State",
    "carry",
    "carry_model",
    "masks",
    "model",
    "object_rows",
    "press",
    "rest",
    "rows",
    "settle",
]

MASS, HALF, PED_TOP, G = 0.2, 0.025, 0.1, 9.81
_PEDESTAL_PAIR = '<pair name="object_pedestal" geom1="right_object_cube" geom2="right_pedestal_start" condim="3"/>'
XML = f"""<mujoco model="audit_toy">
  <option timestep="0.005" gravity="0 0 -{G}"/>
  <default><geom contype="0" conaffinity="0"/></default>
  <worldbody>
    <geom name="right_pedestal_start" type="box" size="0.1 0.1 0.05" pos="0 0 0.05"/>
    <body name="right_hand" pos="0 0 0" gravcomp="1">
      <joint name="hand_z" type="slide" axis="0 0 1" damping="20"/>
      <geom name="collision_hand_palm" type="box" size="0.05 0.05 0.01" mass="1"/>
    </body>
    <body name="right_object" pos="0 0 {PED_TOP + HALF}">
      <freejoint name="object_free"/>
      <geom name="right_object_cube" type="box" size="{HALF} {HALF} {HALF}" mass="{MASS}"/>
    </body>
  </worldbody>
  <actuator><position name="hand_z_servo" joint="hand_z" kp="300"/></actuator>
  <contact>
    <pair name="hand_object" geom1="collision_hand_palm" geom2="right_object_cube" condim="3"/>
    {_PEDESTAL_PAIR}
    <pair name="hand_pedestal" geom1="collision_hand_palm" geom2="right_pedestal_start" condim="3"/>
  </contact>
</mujoco>
"""
CARRY_XML = XML.replace(_PEDESTAL_PAIR, "")      # the cube can rest only on the plate
WEIGHT = MASS * G
TOP = PED_TOP + 2 * HALF + 0.01          # plate centre when its lower face touches the cube's top


def model(xml: str = XML) -> Any:
    import mujoco

    return mujoco.MjModel.from_xml_string(xml)


def masks(m: Any) -> tuple[np.ndarray, np.ndarray]:
    """(hand, object) geom masks of the toy scene."""
    import mujoco

    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(m.ngeom)]
    return (np.array(["collision_hand_" in n for n in names]), np.array(["object" in n for n in names]))


@dataclass(frozen=True)
class State:
    qpos: np.ndarray
    qvel: np.ndarray
    ctrl: np.ndarray
    actuator_force: float


def _run(m: Any, qpos: np.ndarray, target: float, steps: int) -> State:
    import mujoco

    d = mujoco.MjData(m)
    d.qpos[:] = qpos
    d.ctrl[0] = target
    for _ in range(steps):
        mujoco.mj_step(m, d)
    return State(d.qpos.copy(), d.qvel.copy(), d.ctrl.copy(), float(abs(d.actuator_force[0])))


def settle(m: Any, target: float, q_hand: float, steps: int = 4000) -> State:
    q = np.array(m.qpos0, np.float64)
    q[0] = q_hand
    return _run(m, q, target, steps)


def rest(m: Any) -> State:
    return settle(m, TOP + 0.2, TOP + 0.2)


def press(m: Any, depth: float = 0.01) -> State:
    return settle(m, TOP - depth, TOP)


def carry_model() -> Any:
    """The scene without the pedestal pair."""
    return model(CARRY_XML)


def carry(m: Any, height: float = 0.3) -> State:
    """The cube settled on the plate, plate servoed at `height` (in `carry_model()`)."""
    q = np.array(m.qpos0, np.float64)
    q[0] = height
    q[1:4] = (0.0, 0.0, height + 0.01 + HALF)
    q[4:8] = (1.0, 0.0, 0.0, 0.0)
    return _run(m, q, height, 4000)


def rows(states: list[State]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (np.stack([s.qpos for s in states]), np.stack([s.qvel for s in states]),
            np.stack([s.ctrl for s in states]))


def object_rows(qpos: np.ndarray) -> np.ndarray:
    """The object's (x, y, z, qw, qx, qy, qz) columns of toy-scene qpos rows."""
    return qpos[:, 1:8]
