"""MJCF known-answer scenes for the quasi-dynamic step tests."""

from __future__ import annotations

import mujoco

KP, KV = 1000.0, 10.0
MASS = 1.0
HALF = 0.05
R_TIP = 0.01


def _pair(g1: str, g2: str, mu: float) -> str:
    return f'<pair geom1="{g1}" geom2="{g2}" condim="3" friction="{mu} {mu} 0.005 0.0001 0.0001"/>'


def push_scene(mu_floor: float = 0.5, mu_tip: float = 0.2, floor: bool = True, tip_x: float = -(HALF + R_TIP)) -> mujoco.MjModel:
    """A box (1 kg, 10 cm) resting on a plane; a 3-dof slide fingertip (gravity-compensated) on its -x face."""
    pairs = (_pair("floor", "box", mu_floor) if floor else "") + _pair("tip", "box", mu_tip)
    acts = "".join(f'<position joint="f{a}" kp="{KP}" kv="{KV}"/>' for a in "xyz")
    xml = f"""<mujoco><option timestep="0.005" gravity="0 0 -9.81"/><worldbody>
      <geom name="floor" type="plane" size="1 1 0.1" contype="0" conaffinity="0"/>
      <body name="box" pos="0 0 {HALF}"><freejoint/>
        <geom name="box" type="box" size="{HALF} {HALF} {HALF}" mass="{MASS}" contype="0" conaffinity="0"/></body>
      <body name="finger" pos="{tip_x} 0 {HALF}" gravcomp="1">
        <joint name="fx" type="slide" axis="1 0 0"/><joint name="fy" type="slide" axis="0 1 0"/>
        <joint name="fz" type="slide" axis="0 0 1"/>
        <geom name="tip" type="sphere" size="{R_TIP}" mass="0.01" contype="0" conaffinity="0"/></body>
    </worldbody><contact>{pairs}</contact><actuator>{acts}</actuator></mujoco>"""
    return mujoco.MjModel.from_xml_string(xml)


def pinch_scene(mu: float, margin: float = 0.0) -> mujoco.MjModel:
    """A free box (1 kg) between two gravity-compensated 1-dof x-slide fingertips touching its +-x faces; no floor."""
    x = HALF + R_TIP
    pr = "".join(f'<pair geom1="tip{s}" geom2="box" condim="3" margin="{margin}" '
                 f'friction="{mu} {mu} 0.005 0.0001 0.0001"/>' for s in "LR")
    xml = f"""<mujoco><option timestep="0.005" gravity="0 0 -9.81"/><worldbody>
      <body name="box" pos="0 0 0"><freejoint/>
        <geom name="box" type="box" size="{HALF} {HALF} {HALF}" mass="{MASS}" contype="0" conaffinity="0"/></body>
      <body name="fL" pos="{-x} 0 0" gravcomp="1"><joint name="jL" type="slide" axis="1 0 0"/>
        <geom name="tipL" type="sphere" size="{R_TIP}" mass="0.01" contype="0" conaffinity="0"/></body>
      <body name="fR" pos="{x} 0 0" gravcomp="1"><joint name="jR" type="slide" axis="1 0 0"/>
        <geom name="tipR" type="sphere" size="{R_TIP}" mass="0.01" contype="0" conaffinity="0"/></body>
    </worldbody><contact>{pr}</contact><actuator>
      <position joint="jL" kp="{KP}" kv="{KV}"/><position joint="jR" kp="{KP}" kv="{KV}"/></actuator></mujoco>"""
    return mujoco.MjModel.from_xml_string(xml)
