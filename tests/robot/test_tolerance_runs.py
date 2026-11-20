"""`fastdo.robot.tolerance_runs` (object placement corners and envelope, the run's object criteria,
the measured-state starts) and the replay changes they rest on (the settle known answer of the sim settle gate, the
vectorised contact scan).

Known answers:
- settle: a 6 cm box resting on the table settles < 2 mm / < 0.5 deg; the same box started with its
  centre 1 cm past the table's far edge tips over (> 5 mm, > 2 deg), and the gate refuses it;
- envelope: a 6 cm box, yaw about its own vertical: sqrt(2) x 1 cm + 2 x 4.243 cm x sin(2.5 deg) = 17.84 mm, and the
  largest actual vertex displacement over the 8 corners stays below it."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
import trimesh

from fastdo.robot import g1_sdk_replay as S
from fastdo.robot.replay import SceneConfig, _contacts, build_replay_scene, replay
from fastdo.robot.tolerance_runs import (
    object_criteria,
    object_envelope_m,
    object_vertices_world,
    placement_pose,
    state_starts,
)
from fastdo.robot.trajectory import RobotTrajectory, gravity_torque
from tests.robot import synthetic

_BOX = trimesh.creation.box(extents=[1.0, 1.0, 1.0])


def _static_traj(g1) -> RobotTrajectory:
    q = synthetic.arm_path()[0]
    at, ht = np.arange(251) / 250.0, np.arange(101) / 100.0
    tau = gravity_torque(g1, q[None], synthetic.HAND_Q[None])[0]
    return RobotTrajectory(arm_t=at, arm_q=np.tile(q, (len(at), 1)), arm_dq=np.zeros((len(at), 7)),
                           arm_tau_ff=np.tile(tau, (len(at), 1)), hand_t=ht,
                           hand_q=np.tile(synthetic.HAND_Q, (len(ht), 1)), hand_dq=np.zeros((len(ht), 7)),
                           time_scale=1.0, hold_s=0.0, source_t=np.array([0.0, 1.0]))


def _box_at(x: float, y: float, table_z: float = 0.80, lift: float = 0.0005) -> tuple:
    start = np.eye(4)
    start[:3, 3] = [x, y, table_z + 0.03 + lift]
    return build_replay_scene(_BOX.vertices, _BOX.faces, 0.06, start, SceneConfig(table_z=table_z)), start


@pytest.mark.parametrize(("x", "rests"), [(0.70, True), (0.79, False)])
def test_settle_known_answer(g1, x: float, rests: bool) -> None:
    """The far table edge is at x = 0.78 m: a box centred at 0.70 rests; centred at 0.79 (its centre of mass 1 cm past
    the edge) it tips off. The robot holds the synthetic hold pose far from it (y = 0.30 m)."""
    scene, start = _box_at(x, 0.30)
    traj = _static_traj(g1)
    metrics, _ = replay(scene, traj, np.stack([start, start]), start)
    stats = {"replay": metrics.to_dict()}
    fails = [f for f in S._replay_failures(stats) if "settling" in f]
    if rests:
        assert metrics.settle_drift < 0.002 and metrics.settle_rot_deg < 0.5 and fails == []
    else:
        assert metrics.settle_drift > 0.005 and metrics.settle_rot_deg > 2.0
        assert len(fails) == 2, fails


def _loop_contacts(scene, d) -> tuple[float, float, bool, float]:
    """The former per-contact loop of `replay._contacts` (reference for the vectorised version)."""
    self_d = table_d = obj_d = 0.0
    touch = False
    rob, bodies = scene.robot_geoms, scene.g1.model.geom_bodyid
    for c in d.contact[: d.ncon]:
        depth = -float(c.dist)
        g_1, g_2 = int(c.geom[0]), int(c.geom[1])
        if (rob[g_1] and bodies[g_2] == scene.object_body) or (rob[g_2] and bodies[g_1] == scene.object_body):
            touch = True
            obj_d = max(obj_d, depth)
        if rob[g_1] and rob[g_2]:
            self_d = max(self_d, depth)
        elif (rob[g_1] and g_2 == scene.table_geom) or (rob[g_2] and g_1 == scene.table_geom):
            table_d = max(table_d, depth)
    return self_d, table_d, touch, obj_d


def test_vectorised_contacts_equal_the_loop(g1) -> None:
    """On 200 random right-arm / hand configurations around a box on the table (many with arm-table, hand-box and
    self contacts), the vectorised scan returns exactly the loop's values."""
    scene, _ = _box_at(0.35, -0.20)
    m = scene.g1.model
    d = mujoco.MjData(m)
    rng = np.random.default_rng(0)
    rlo, rhi = g1.right_arm_range.T
    hlo, hhi = g1.right_hand_range.T
    seen = {"self": 0, "table": 0, "object": 0}
    for _ in range(200):
        d.qpos[:] = scene.g1.default_qpos()
        d.qpos[g1.right_arm_qpos] = rng.uniform(rlo, rhi)
        d.qpos[g1.right_hand_qpos] = rng.uniform(hlo, hhi)
        mujoco.mj_forward(m, d)
        got, want = _contacts(scene, d), _loop_contacts(scene, d)
        assert got == want and all(type(a) is type(b) for a, b in zip(got, want, strict=True))
        seen["self"] += want[0] > 0
        seen["table"] += want[1] > 0
        seen["object"] += want[2]
    assert min(seen.values()) > 0, seen


def test_placement_pose_turns_about_the_object_vertical() -> None:
    start = np.eye(4)
    start[:3, :3] = trimesh.transformations.rotation_matrix(0.3, [1, 0, 0])[:3, :3]
    start[:3, 3] = [0.4, -0.1, 0.83]
    p = placement_pose(start, 0.01, -0.01, 5.0)
    assert np.allclose(p[:3, 3], [0.41, -0.11, 0.83])
    rel = p[:3, :3] @ start[:3, :3].T
    assert np.allclose(rel[2], [0.0, 0.0, 1.0]) and np.degrees(np.arctan2(rel[1, 0], rel[0, 0])) == pytest.approx(5.0)


def test_object_envelope_bounds_every_corner(g1) -> None:
    scene, start = _box_at(0.40, -0.05)
    env = object_envelope_m(scene, start, 0.01, 5.0)
    assert env == pytest.approx(np.sqrt(2) * 0.01 + 2 * 0.03 * np.sqrt(2) * np.sin(np.radians(2.5)), abs=1e-9)
    v0 = object_vertices_world(scene, start)
    assert np.allclose(np.sort(v0[:, 2]), np.sort(v0[:, 2])) and np.ptp(v0, axis=0) == pytest.approx([0.06] * 3)
    worst = max(np.linalg.norm(object_vertices_world(scene, placement_pose(start, 0.01 * sx, 0.01 * sy, 5.0 * sa))
                               - v0, axis=1).max() for sx, sy, sa in S.PLACEMENT_CASES.values())
    assert 0.9 * env < worst <= env + 1e-12
    assert object_envelope_m(scene, start, 0.0, 0.0) == 0.0


def test_object_criteria_read_the_playback() -> None:
    ref0, ref1 = np.eye(4), np.eye(4)
    ref1[:3, 3] = [0.0, 0.10, 0.0]
    sim = {"object": {"play_start_m": [0.0, 0.0, 0.83], "play_end_m": [0.0, 0.08, 0.83], "run_end_m": [0.0, 0.08, 0.83]}}
    crit = object_criteria(sim, ref0, ref1, 0.80)
    assert crit["reference_displacement_m"] == pytest.approx(0.10) and crit["displacement_m"] == pytest.approx(0.08)
    assert crit["final_error_m"] == pytest.approx(np.hypot(0.02, 0.83)) and crit["fell"] is False
    sim["object"]["run_end_m"] = [0.0, 0.08, 0.70]                       # fell off after the playback
    assert object_criteria(sim, ref0, ref1, 0.80)["fell"] is True
    none = object_criteria({"object": {"play_start_m": None, "play_end_m": None, "run_end_m": [0, 0, 0]}}, ref0, ref1,
                           0.8)
    assert none["displacement_m"] is None and S._object_failures("x", none)


def test_state_starts_cover_the_tolerance_edges() -> None:
    home = np.arange(17) * 0.01
    arm_rng = np.tile([-3.0, 3.0], (14, 1))
    arm_rng[3] = [-3.0, home[3] + 0.05]                                   # joint 3 only 0.05 rad from its limit
    hand_rng = np.tile([-2.0, 2.0], (7, 1))
    hand_rng[2] = [-2.0, 0.0]                                             # thumb 2 at its limit at 0
    out = state_starts(home, np.zeros(7), 0.08, arm_rng, hand_rng, 4, 0.999, 2)
    assert [o["name"] for o in out] == S.state_run_names(4) and len(out) == 34
    by = {o["name"]: o for o in out}
    assert by["arm_axis_05+"]["arm_offset"][5] == pytest.approx(0.07992) and by["arm_axis_05+"]["clipped"] == []
    assert np.allclose(by["arm_axis_05+"]["start_q"][14:], home[14:])
    assert by["arm_axis_03+"]["arm_offset"][3] == pytest.approx(0.05) and by["arm_axis_03+"]["clipped"] == ["arm 3"]
    assert np.allclose(np.abs(by["arm_corner_1"]["arm_offset"])[[0, 1, 2, 4]], 0.07992)
    assert by["hand+"]["hand_offset"][2] == 0.0 and by["hand+"]["clipped"] == ["hand 2"]
    assert by["hand-"]["hand_offset"][2] == pytest.approx(-0.0999) and by["hand-"]["clipped"] == []
    again = state_starts(home, np.zeros(7), 0.08, arm_rng, hand_rng, 4, 0.999, 2)
    assert all(np.array_equal(a["arm_offset"], b["arm_offset"]) for a, b in zip(out, again, strict=True))  # seeded
