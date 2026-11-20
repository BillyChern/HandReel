# Copyright 2026 The HandReel authors. SPDX-License-Identifier: Apache-2.0
"""The MuJoCo adapter on a synthetic scene: a cube on a pedestal, a one-dof "hand" that pushes it.

Known answers:
- the warm-up weld holds the cube at its start pose (zero error on the welded rows);
- a push of 0.4 m is a Z1 clip: the static, hand-absent and plan-then-freeze nulls all fail;
- a 5 mm nudge is a Z0 clip: the same three nulls pass, and hand-absent is VERIFIED (no hand force after release);
- without moving the hand away, the pair edit ("pairs" mode) leaks hand-object force, through the edited pair
  and through a hand geom outside the pair set: the audit marks that run `invalid` rather than counting it;
  "displace" mode removes the hand from every collision path and the null is verified.
"""

from __future__ import annotations

import numpy as np
import pytest

import nullaudit as na

mujoco = pytest.importorskip("mujoco")
from nullaudit.mujoco_adapter import MujocoExecutor  # noqa: E402

W, T = 100, 600
XML = """
<mujoco>
  <option timestep="0.005" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <geom name="pedestal" type="box" pos="0 0 0.05" size="0.1 0.1 0.05"/>
    <body name="hand" pos="-0.06 0 0.13">
      <joint name="hx" type="slide" axis="1 0 0" damping="5"/>
      <geom name="palm" type="sphere" size="0.02" contype="0" conaffinity="0" mass="1"/>
      {extra}
    </body>
    <body name="obj" pos="0 0 0.13">
      <freejoint/>
      <geom name="cube" type="box" size="0.03 0.03 0.03" mass="0.1"/>
    </body>
  </worldbody>
  <contact><pair geom1="palm" geom2="cube"/></contact>
  <actuator><position name="hx_pos" joint="hx" kp="300"/></actuator>
  <equality><weld body1="obj"/></equality>
</mujoco>
"""
FINGER = '<geom name="finger" type="box" pos="0 0 0" size="0.012 0.012 0.012" contype="1" conaffinity="1"/>'


def _executor(extra: str = "", **kw) -> MujocoExecutor:
    m = mujoco.MjModel.from_xml_string(XML.format(extra=extra))
    hand = np.asarray(m.geom_bodyid == mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand"))
    return MujocoExecutor(model=m, q0=m.qpos0.copy(), hand_geom=hand, warmup_rows=W, weld_eq_ids=(0,), **kw)


def _push(dist: float) -> np.ndarray:
    """Hand target: 0 through the warm-up, then a ramp to contact (the palm touches at +0.01) plus `dist`."""
    u = np.zeros((T, 1))
    u[W:, 0] = np.minimum(np.linspace(0.0, 0.01 + dist, T - W) * 2.0, 0.01 + dist)
    return u


def _clip(ex: MujocoExecutor, dist: float):
    u = _push(dist)
    ref = ex(u)
    assert ref.status == "clip_end" and len(ref.object_traj) == T
    return u, ref


def test_warmup_weld_holds_start_pose() -> None:
    ex = _executor()
    _, ro = _clip(ex, 0.4)
    q0 = ex.q0[ex.obj_qadr:ex.obj_qadr + 7]
    assert np.abs(ro.object_traj[:W] - q0).max() < 1e-6
    assert (ro.hand_object_force[W:] > 0).any()                     # the push touches the cube


def test_push_is_z1_and_every_null_fails() -> None:
    ex = _executor()
    u, ro = _clip(ex, 0.4)
    test = na.DoAsIDoObjectTest(ro.object_traj, required_rows=T)
    a = na.audit_clip(ro.object_traj, test, simulator=ex, controls=u, release_row=W,
                      absent_controls=na.move_hand_away(u, column=0, offset=-1.0))
    assert a.reference_passes and not a.z0
    for k in ("static_at_start", "hand_absent", "plan_then_freeze"):
        assert a.outcomes[k].status == "fail", k
    assert a.outcomes["hand_absent"].verified_n0 is True


def test_nudge_is_z0_and_the_nulls_pass() -> None:
    ex = _executor()
    u, ro = _clip(ex, 0.005)
    a = na.audit_clip(ro.object_traj, na.DoAsIDoObjectTest(ro.object_traj, required_rows=T), simulator=ex,
                      controls=u, release_row=W, absent_controls=na.move_hand_away(u, column=0, offset=-1.0))
    assert a.z0
    for k in ("static_at_start", "hand_absent", "plan_then_freeze"):
        assert a.outcomes[k].status == "pass", k
    assert a.outcomes["hand_absent"].verified_n0 is True


@pytest.mark.parametrize("extra", ["", FINGER])
def test_leak_is_invalid_and_displace_is_verified(extra: str) -> None:
    """Without moving the hand, the pair edit leaks through the unpaired finger's default collision on every engine
    (invalid). Through the edited pair alone the outcome depends on the engine: MuJoCo 3.13.x still produces force
    (invalid, pinned), 3.4.0 honours the 10 m gap (a verified fail). On another engine the ledger decides: never an
    unverified or leaking pass. Displacement is a verified fail everywhere."""
    u = _push(0.4)
    for mode in ("pairs", "displace"):
        ex = _executor(extra, absent_mode=mode)
        ro = ex(u)
        a = na.audit_clip(ro.object_traj, na.DoAsIDoObjectTest(ro.object_traj, required_rows=T), simulator=ex,
                          controls=u, release_row=W, nulls=("hand_absent",))
        o = a.outcomes["hand_absent"]
        if mode == "displace" or (extra == "" and mujoco.__version__ == "3.4.0"):
            assert o.status == "fail" and o.verified_n0 is True, (mode, extra)
        elif extra or mujoco.__version__.startswith("3.13."):
            assert o.status == "invalid" and o.verified_n0 is False, (mode, extra, mujoco.__version__)
        else:
            assert o.status in ("invalid", "fail") and o.verified_n0 is (o.status == "fail"), (mode, o)


def test_drop_stops_the_run() -> None:
    ex = _executor(terminate_z=0.08)
    ro = ex(_push(0.4))
    assert ro.status == "drop" and len(ro.object_traj) < T
    assert not na.DoAsIDoObjectTest(np.zeros((T, 7)) + [0, 0, 0, 1, 0, 0, 0], required_rows=T)(ro.object_traj).full


def _palm_only(extra: str, **kw) -> MujocoExecutor:
    m = mujoco.MjModel.from_xml_string(XML.format(extra=extra))
    palm = np.zeros(m.ngeom, bool)
    palm[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "palm")] = True
    return MujocoExecutor(model=m, q0=m.qpos0.copy(), hand_geom=palm, warmup_rows=W, weld_eq_ids=(0,), **kw)


def test_ledger_sees_a_hand_geom_outside_the_mask() -> None:
    """The finger is on the hand body but not in the hand mask. Displacing the masked palm leaves the finger in
    place, and it pushes the cube; the ledger must see that force (the run is invalid, not a verified null)."""
    u = _push(0.4)
    ex = _palm_only(FINGER, absent_mode="displace")
    ro = ex(u)
    a = na.audit_clip(ro.object_traj, na.DoAsIDoObjectTest(ro.object_traj, required_rows=T), simulator=ex,
                      controls=u, release_row=W, nulls=("hand_absent",))
    o = a.outcomes["hand_absent"]
    assert o.status == "invalid" and o.verified_n0 is False, o
    assert (ex.last_other_force[W:] > 0).any() and not (ex.last_hand_force[W:] > 0).any()


def test_ledger_control_full_hand_mask_is_verified() -> None:
    """Green control: with the whole hand body in the mask the same displaced run is a verified null."""
    u = _push(0.4)
    ex = _executor(FINGER, absent_mode="displace")
    ro = ex(u)
    o = na.audit_clip(ro.object_traj, na.DoAsIDoObjectTest(ro.object_traj, required_rows=T), simulator=ex,
                      controls=u, release_row=W, nulls=("hand_absent",)).outcomes["hand_absent"]
    assert o.status == "fail" and o.verified_n0 is True


def test_last_rows_equal_a_manual_loop() -> None:
    """last_qpos, last_qvel and last_ctrl are the post-step states and applied controls of a plain mj_step loop."""
    ex = _executor()
    u = _push(0.4)
    ex(u)
    m = ex.model
    d = mujoco.MjData(m)
    d.qpos[:] = ex.q0
    qp, qv = [], []
    for s in range(T):
        warm = s < W
        m.opt.gravity[:] = 0.0 if warm else ex.gravity
        d.eq_active[0] = 1 if warm else 0
        d.ctrl[:] = np.float32(u[s, 0])
        mujoco.mj_step(m, d)
        qp.append(d.qpos.copy())
        qv.append(d.qvel.copy())
    m.opt.gravity[:] = ex.gravity
    assert np.array_equal(ex.last_qpos, np.array(qp)) and np.array_equal(ex.last_qvel, np.array(qv))
    assert np.array_equal(ex.last_ctrl[:, 0], u[:, 0].astype(np.float32).astype(np.float64))


def test_a_run_restores_the_models_gravity() -> None:
    ex = _executor(gravity=np.array([0.0, 0.0, -1.0]))
    before = np.array(ex.model.opt.gravity)
    ex(_push(0.4))
    assert np.array_equal(ex.model.opt.gravity, before)


def test_a_diverging_servo_is_invalid() -> None:
    """Ten rows after release the servo target jumps 1e5 m away from the cube: the hand reaches ~1e5 m/s within a
    step, far below MuJoCo's own 1e10 bound, so only the |qvel| guard ends the run. A wide bound runs it to the end
    (green control: the guard, not the scene, decides)."""
    u = _push(0.4)
    u[W + 10:, 0] = -1e5
    ex = _executor()
    ro = ex(u)
    assert ro.status == "invalid" and len(ro.object_traj) == W + 11
    assert ex.last_invalid is not None and ex.last_invalid.startswith("|qvel|"), ex.last_invalid
    wide = _executor(qvel_max=1e12)
    assert wide(u).status == "clip_end" and wide.last_invalid is None


@pytest.mark.parametrize("warning", ["mjWARN_CONTACTFULL", "mjWARN_CNSTRFULL", "mjWARN_BADQACC"])
def test_a_full_contact_or_constraint_buffer_is_invalid(warning: str) -> None:
    """Contacts or constraints dropped for lack of memory make the step's state invalid, as a bad acceleration
    does (the warnings cannot be provoked reliably on a small scene, so the counter is set directly)."""
    ex = _executor()
    d = mujoco.MjData(ex.model)
    mujoco.mj_forward(ex.model, d)
    assert ex._invalid(d) is None
    d.warning[getattr(mujoco.mjtWarning, warning)].number = 1
    assert ex._invalid(d) == f"MuJoCo warning {warning}"


def test_a_bad_control_is_invalid_and_the_applied_control_is_recorded() -> None:
    """MuJoCo zeroes a huge control and steps on (BADCTRL): the run is invalid, and `last_ctrl` holds the control
    MuJoCo applied (0), not the stored one."""
    u = _push(0.4)
    u[W + 5, 0] = 1e12
    ex = _executor()
    ro = ex(u)
    assert ro.status == "invalid" and ex.last_invalid == "MuJoCo warning mjWARN_BADCTRL"
    assert ex.last_ctrl.shape == (W + 6, 1) and ex.last_ctrl[-1, 0] == 0.0
    assert np.array_equal(ex.last_ctrl[:-1, 0], u[:W + 5, 0].astype(np.float32).astype(np.float64))


def test_executor_settings_are_checked() -> None:
    m = mujoco.MjModel.from_xml_string(XML.format(extra=""))
    hand = np.asarray(m.geom_bodyid == mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand"))
    for kw in ({"weld_eq_ids": (1,)}, {"weld_eq_ids": (-1,)}, {"q0": m.qpos0[:-1]}, {"ctrl0": np.zeros(2)},
               {"gravity": np.zeros(2)}, {"ctrl_steps": 0}):
        args = {"model": m, "q0": m.qpos0.copy(), "hand_geom": hand, "warmup_rows": W, **kw}
        with pytest.raises(ValueError):
            MujocoExecutor(**args)
    with pytest.raises(ValueError, match="controls"):
        _executor()(np.zeros((T, 2)))
