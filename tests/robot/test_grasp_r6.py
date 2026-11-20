"""Known answers of the grasp synthesis and closure: (a) the synthesised approach gives zero penetration
before contact, (b) grasp closure lifts a sphere the open-loop grasp leaves behind, and never exceeds a joint limit.

The scene: the right Dex3 hand at a fixed grasp pose `_G` around a sphere whose radius is chosen so that the hand
clears it by 0.5 mm (the fingertips "on the surface", no normal force: the provisional lab-clip failure). The sphere
rests on a table under it. Rows: an approach onto the grasp (task rows 0..C0-1), a hold (C0..C0+4), a lift of 8 cm.
"""

from __future__ import annotations

import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from fastdo.robot import g1_constants as C
from fastdo.robot.approach import (
    ApproachConfig,
    HandClearance,
    open_hand,
    plan_approach,
    smoothstep,
)
from fastdo.robot.arm_ik import IkCollider, site_pose, solve_arm_ik
from fastdo.robot.dex3_retarget import dex3_tip_positions
from fastdo.robot.grasp_closure import ClosureConfig, close_grasp
from fastdo.robot.replay import (
    SceneConfig,
    SuccessCriteria,
    build_ik_model,
    build_replay_scene,
    replay,
)
from fastdo.robot.trajectory import (
    ExportConfig,
    TrajectoryLimits,
    build_trajectory,
    check_trajectory,
)

_Q0 = np.array([-0.6, -0.3, 0.0, 0.3, 0.0, 0.0, 0.0])
_G = np.array([0.0, -0.2, -0.4, 0.3, 0.3, 0.3, 0.3])
_DT, _C0, _HOLD, _LIFT = 0.1, 6, 5, 15
_DZ = 0.08
_GAP = 0.0005
_CENTRE = (0.38, -0.12, 0.90)                          # sphere centre, robot frame (a reachable grasp, probe)
_H = 0.015                                             # the sphere centre sits this far above the fingertip plane


def _tip_plane(g1) -> tuple[np.ndarray, np.ndarray]:
    """`(circumcentre of the three Dex3 fingertips at _G, unit normal of their plane pointing to the palm side)` in
    the Dex3 base frame."""
    tips = dex3_tip_positions(g1, _G[None])[0][1:]
    cc = _circumcentre(tips)
    n = np.cross(tips[1] - tips[0], tips[2] - tips[0])
    n /= np.linalg.norm(n)
    return cc, (n if (-cc) @ n > 0 else -n)


def _circumcentre(p: np.ndarray) -> np.ndarray:
    a, b, c = p
    ab, ac = b - a, c - a
    n = np.cross(ab, ac)
    return a + (np.cross(n, ab) * np.dot(ac, ac) + np.cross(ac, n) * np.dot(ab, ab)) / (2.0 * np.dot(n, n))


@pytest.fixture(scope="module")
def sphere(g1):
    """`(mesh, radius, centre in the Dex3 base frame, clearance)`: the hand at `_G` clears the sphere by `_GAP`; the
    centre is `_H` above the fingertip plane, so the tips sit on the LOWER half (a cradle, which closure can lift)."""
    cc, nrm = _tip_plane(g1)
    cc = cc + _H * nrm
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    base_obj = np.eye(4)
    base_obj[:3, 3] = -cc
    r = float(np.linalg.norm(dex3_tip_positions(g1, _G[None])[0][1] - cc))
    for _ in range(3):                                   # clearance is linear in r with slope -1 near the tips
        clr = HandClearance(mesh.vertices, mesh.faces, r)
        r += clr.clearance(base_obj, _G) - _GAP
    clr = HandClearance(mesh.vertices, mesh.faces, r)
    assert abs(clr.clearance(base_obj, _G) - _GAP) < 2e-4
    return mesh, r, cc, clr


@pytest.fixture(scope="module")
def task(g1, sphere):
    """The synthesised approach, hold and lift as joint rows, the scene and the reference (module docstring)."""
    mesh, r, cc, clr = sphere
    n = _C0 + _HOLD + _LIFT
    times = np.arange(n) * _DT
    xr = site_pose(g1, _Q0)[0]                           # the base turned so the tip-plane normal points up
    u = -(xr[:3, :3] @ _tip_plane(g1)[1])
    axis = np.cross(u, [0.0, 0.0, -1.0])
    turn = Rotation.from_rotvec(axis / np.linalg.norm(axis) * np.arccos(np.clip(-u[2], -1.0, 1.0))).as_matrix()
    centre = np.asarray(_CENTRE)
    x0 = np.eye(4)
    x0[:3, :3] = turn @ xr[:3, :3]
    x0[:3, 3] = centre - x0[:3, :3] @ cc
    start = np.eye(4)
    start[:3, 3] = centre
    base_obj = np.repeat((np.linalg.inv(start) @ x0)[None], n, axis=0)
    hand = np.tile(_G, (n, 1))
    plan = plan_approach(times, _C0, base_obj, hand, np.arange(_C0, n), np.zeros(3), clr, ApproachConfig())
    assert plan.ok, plan.reason
    lift = np.zeros(n)
    lift[_C0 + _HOLD:] = _DZ * smoothstep(np.arange(1, _LIFT + 1) / _LIFT)
    targets = np.repeat(x0[None], n, axis=0)
    targets[: _C0 + 1] = np.einsum("ij,tjk->tik", start, plan.base_obj)
    targets[:, 2, 3] += lift
    hand[: _C0 + 1] = plan.hand_q
    cfg = SceneConfig(table_z=float(centre[2] - r))
    ik_g1 = build_ik_model(cfg)
    pre = solve_arm_ik(ik_g1, np.repeat(x0[None], 30, axis=0), q_init=_Q0, collider=IkCollider(ik_g1),
                       hand_q=np.tile(_G, (30, 1)))
    ik = solve_arm_ik(ik_g1, targets, q_init=pre.q[-1], collider=IkCollider(ik_g1), hand_q=hand)
    assert ik.pos_err.max() < 0.01, ik.pos_err.max()
    scene = build_replay_scene(mesh.vertices, mesh.faces, r, start, cfg)
    ref = np.repeat(start[None], n, axis=0)
    ref[:, 2, 3] += lift
    return {"times": times, "q": ik.q, "hand": hand, "scene": scene, "ref": ref, "start": start, "plan": plan,
            "span": (_C0, n - 1), "base_obj": base_obj, "clr": clr}


def _replay(g1, t: dict, hand: np.ndarray):
    traj = build_trajectory(g1, t["times"], t["q"], hand)
    m, logs = replay(t["scene"], traj, t["ref"], t["start"], SuccessCriteria())
    return traj, m, logs


def test_a_the_planned_approach_never_penetrates_before_contact(g1, task) -> None:
    """(a) known answer: every planned approach sample before contact clears the sphere, the start clears it by the
    required 10 mm, AND the simulation shows zero robot-object penetration before the contact row's time."""
    plan, clr = task["plan"], task["clr"]
    assert plan.penetrating_samples == 0 and plan.min_clearance_before_contact_m > 0.0
    assert plan.start_clearance_m >= 0.010
    assert clr.clearance(plan.base_obj[0], plan.hand_q[0]) >= 0.010
    traj, _, logs = _replay(g1, task, task["hand"])
    t_contact = traj.hold_s + traj.time_scale * float(task["times"][_C0] - task["times"][0])
    before = logs["t"] < t_contact - 0.05
    assert before.sum() > 50
    assert float(np.asarray(logs["object_depth"])[before].max()) == 0.0


def test_a_the_pipeline_plans_the_approach_on_a_short_lead(g1, tmp_path) -> None:
    """(a) in `prepare_task`: the index tip touches the box from row 3 (a 0.3-s lead, like the DexYCB clips). With
    `approach_mode="palm"` the rows are padded to a >= 0.5-s approach, every row before contact clears the box, the
    start clears it by the required margin, and the reach search keeps those rows (`approach_synthesised`). The
    default vertical mode is unchanged (no plan)."""
    from fastdo.robot.kinematic_path import KinematicPathConfig, load_reconstruction
    from fastdo.robot.kinematic_task import prepare_task
    from tests.robot import synthetic

    clip = synthetic.write_clip(g1, tmp_path / "short", synthetic.contact_object_path(g1, 3, 0.0))
    recon = load_reconstruction(clip)
    assert prepare_task(recon, KinematicPathConfig(table_xy=synthetic.OBJECT_XY), g1).approach is None
    prep = prepare_task(recon, KinematicPathConfig(table_xy=synthetic.OBJECT_XY, approach_mode="palm"), g1)
    rec, task = prep.approach, prep.task
    assert rec is not None and rec["ok"] and rec["penetrating_samples"] == 0 and task.approach_synthesised
    c0 = rec["c0_task_row"]
    assert task.times[c0] - task.times[0] >= 0.5 - 1e-9 and (prep.source_rows[:c0] < 0).any()
    clr = HandClearance(recon.mesh_vertices, recon.mesh_faces, recon.record.object_scale)
    base_obj = np.linalg.solve(task.object_world, task.wrist_world)
    before = [clr.clearance(base_obj[t], task.hand_q[t]) for t in range(c0)]
    assert min(before) > 0.0
    assert before[0] >= rec["required_start_clearance_m"] - 1e-9 >= 0.010


def test_b_the_open_loop_grasp_leaves_the_sphere(g1, task) -> None:
    """(b) known answer, without closure: the fingertips sit on the sphere and the lift does not carry it."""
    _, m, _ = _replay(g1, task, task["hand"])
    assert m.object_displacement < 0.5 * _DZ
    assert not m.success


@pytest.fixture(scope="module")
def closed(g1, task):
    rows, rec = close_grasp(g1, task["scene"], task["times"], task["q"], task["hand"], task["span"], task["ref"],
                            task["start"], np.ones(len(task["times"]), bool), SuccessCriteria(), ExportConfig(),
                            TrajectoryLimits())
    return rows, rec


def test_b_closure_lifts_the_sphere(g1, task, closed) -> None:
    """(b) known answer, with closure: the same lift carries the sphere at least half way, and it does not fall."""
    rows, rec = closed
    _, m, _ = _replay(g1, task, rows)
    assert m.object_displacement >= 0.5 * _DZ, (m.object_displacement, rec)
    assert not m.object_fell
    assert rec["reached"]


def test_b_closure_never_exceeds_a_joint_limit(g1, task, closed) -> None:
    """(b) known answer: the biased rows and their trajectory stay inside the command limits, the replay never
    leaves the MJCF range, even with an absurd bias cap."""
    lim = np.asarray(C.RIGHT_HAND_COMMAND_LIMITS)
    for rows in (closed[0], close_grasp(g1, task["scene"], task["times"], task["q"], task["hand"], task["span"],
                                        task["ref"], task["start"], np.ones(len(task["times"]), bool),
                                        SuccessCriteria(), ExportConfig(), TrajectoryLimits(),
                                        ClosureConfig(max_bias=5.0, iters=1))[0]):
        assert (rows >= lim[:, 0] - 1e-9).all() and (rows <= lim[:, 1] + 1e-9).all()
        traj, m, _ = _replay(g1, task, rows)
        assert not check_trajectory(traj, TrajectoryLimits())
        assert m.limit_violation_ticks == 0


def test_open_hand_is_at_the_extension_limits() -> None:
    lim = np.asarray(C.RIGHT_HAND_COMMAND_LIMITS)
    o = open_hand(_G)
    assert o[0] == _G[0]
    assert o[3] == lim[3, 0] and o[4] == lim[4, 0] and o[1] == lim[1, 1] and o[2] == lim[2, 1]


def test_b_the_bias_stops_inside_the_limits_and_never_opens() -> None:
    """(b): a biased joint stops `limit_margin` inside its command limit (on a provisional lab clip, at the knot margin the
    simulated thumb_2 went 0.023 rad past its MJCF limit), and the stop never moves a joint AGAINST the bias (a joint
    already past the margin keeps its retargeted value)."""
    from fastdo.robot.grasp_closure import _biased

    lim = TrajectoryLimits().hand_bounds()
    q = np.tile(_G, (3, 1))
    q[2, 4] = lim[4, 1] - 0.02                           # index_1 already inside the 0.05 margin
    bias = np.zeros((3, 7))
    bias[1:, [1, 2]] = -5.0                              # thumb flexes negative
    bias[1:, [3, 4, 5, 6]] = 5.0
    out = _biased(q, bias, TrajectoryLimits(), ExportConfig(), 0.05)
    assert np.allclose(out[0], _G)
    assert out[1, 2] == pytest.approx(lim[2, 0] + 0.05) and out[1, 4] == pytest.approx(lim[4, 1] - 0.05)
    assert out[2, 4] == pytest.approx(q[2, 4])
    assert out[1, 0] == _G[0]
