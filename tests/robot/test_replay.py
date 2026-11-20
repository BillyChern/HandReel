"""Tests for `fastdo.robot.replay`: the object mesh lands where the pose says, and the success criteria on a known
push. Known answer: a hand swept 28 cm in +y across a 10 cm box pushes it ~13 cm, and the replay calls a 10 cm
reference a success. Mutations: the same physical run against a reference that does not move, and a sweep that
misses the box, both fail for the right reason. Each safety metric is also shown to FIRE on a known physical case
(arm into the torso, a table through the hand, a command past a joint limit) next to a clean control."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from fastdo.robot.arm_ik import IkCollider, site_pose, solve_arm_ik
from fastdo.robot.dex3_retarget import dex3_tip_positions
from fastdo.robot.replay import SceneConfig, build_ik_model, build_replay_scene, replay
from fastdo.robot.trajectory import RobotTrajectory, build_trajectory, gravity_torque

_BOX = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
_HAND = np.array([0.0, 0.0, -0.3, 0.3, 0.3, 0.3, 0.3])
_N = 30


def test_object_mesh_sits_at_its_pose(g1) -> None:
    off = trimesh.creation.box(extents=[1.0, 0.6, 0.4])
    off.apply_translation([0.3, 0.1, -0.2])                 # off-centre: MuJoCo re-centres meshes internally
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler("z", 30, degrees=True).as_matrix()
    pose[:3, 3] = [0.3, -0.2, 0.9]
    scene = build_replay_scene(off.vertices, off.faces, 0.1, pose)
    m = scene.g1.model
    d = mujoco.MjData(m)
    d.qpos[:] = scene.g1.default_qpos()
    mujoco.mj_forward(m, d)
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "object_geom")
    mid = m.geom_dataid[gid]
    verts = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
    world = verts @ d.geom_xmat[gid].reshape(3, 3).T + d.geom_xpos[gid]
    expected = (0.1 * off.vertices) @ pose[:3, :3].T + pose[:3, 3]
    assert cKDTree(expected).query(world)[0].max() < 1e-6
    assert m.body_mass[scene.object_body] == pytest.approx(0.2)


@pytest.fixture(scope="module")
def sweep(g1):
    """A right-hand sweep along +y at 6 cm above the table, from the collision-aware IK."""
    cfg = SceneConfig(table_z=0.80)
    ik_g1 = build_ik_model(cfg)
    q0 = np.array([-0.6, -0.3, 0.0, 0.3, 0.0, 0.0, 0.0])
    targets = np.repeat(site_pose(g1, q0), _N, axis=0)
    targets[:, 0, 3], targets[:, 1, 3], targets[:, 2, 3] = 0.36, np.linspace(-0.30, -0.02, _N), 0.86
    ik = solve_arm_ik(ik_g1, targets, q_init=q0, collider=IkCollider(ik_g1))
    assert ik.pos_err.max() < 0.08
    traj = build_trajectory(g1, np.arange(_N) * 0.1, ik.q, np.tile(_HAND, (_N, 1)))
    poses = site_pose(g1, ik.q)
    tips = np.einsum("tij,vj->tvi", poses[:, :3, :3], dex3_tip_positions(g1, _HAND)[0]) + poses[:, None, :3, 3]
    return cfg, traj, float(tips[:, :, 0].mean())


def _run(sweep, y0: float, ref_dy: float):
    cfg, traj, x = sweep
    start = np.eye(4)
    start[:3, 3] = [x, y0, cfg.table_z + 0.05 + 0.001]
    scene = build_replay_scene(_BOX.vertices, _BOX.faces, 0.10, start, cfg)
    ref = np.repeat(start[None], _N, axis=0)
    ref[:, 1, 3] += np.linspace(0.0, ref_dy, _N)
    return replay(scene, traj, ref, start)


def test_push_is_a_success(sweep) -> None:
    metrics, logs = _run(sweep, -0.12, 0.10)
    assert metrics.success, metrics.failed_criteria
    assert metrics.object_displacement > 0.08 and metrics.object_final_error < 0.05
    assert metrics.arm_rms < 0.05 and metrics.self_collision_steps == 0 and metrics.limit_violation_ticks == 0
    assert logs["arm_q"].shape[1] == 7 and len(logs["t"]) == len(logs["object_qpos"])


def test_reference_that_does_not_move_fails(sweep) -> None:
    """Mutation of the reference: the same physical push against a static reference is not a success."""
    metrics, _ = _run(sweep, -0.12, 0.0)
    assert not metrics.success and "reference_does_not_move_object" in metrics.failed_criteria


def test_missed_object_fails(sweep) -> None:
    """The box placed beyond the end of the sweep is never touched: the object must move."""
    metrics, _ = _run(sweep, 0.25, 0.10)
    assert metrics.object_displacement < 0.01
    assert not metrics.success and "object_not_moved" in metrics.failed_criteria


def _hold(g1, q_arm: np.ndarray, duration: float = 1.0) -> RobotTrajectory:
    """A constant command (arm `q_arm`, hand `_HAND`) with its exact gravity feed-forward, built WITHOUT
    `build_trajectory` (which would clip it into the command limits)."""
    at, ht = np.arange(round(duration * 250) + 1) / 250.0, np.arange(round(duration * 100) + 1) / 100.0
    tau = gravity_torque(g1, q_arm[None], _HAND[None])[0]
    qa, qh = np.tile(q_arm, (len(at), 1)), np.tile(_HAND, (len(ht), 1))
    return RobotTrajectory(arm_t=at, arm_q=qa, arm_dq=np.zeros_like(qa), arm_tau_ff=np.tile(tau, (len(at), 1)),
                           hand_t=ht, hand_q=qh, hand_dq=np.zeros_like(qh), time_scale=1.0, hold_s=0.0,
                           source_t=np.array([0.0, duration]))


def _hold_metrics(g1, q_arm: np.ndarray, cfg: SceneConfig, unlimit: str | None = None):
    """Replay `_hold` with a box parked far from the arm and a static reference. `unlimit`: a joint whose MJCF limit
    constraint is switched off in the replay model (its `jnt_range`, which the metric reads, is kept)."""
    start = np.eye(4)
    start[:3, 3] = [0.6, 0.3, cfg.table_z + 0.051]
    scene = build_replay_scene(_BOX.vertices, _BOX.faces, 0.10, start, cfg)
    if unlimit is not None:
        scene.g1.model.jnt_limited[mujoco.mj_name2id(scene.g1.model, mujoco.mjtObj.mjOBJ_JOINT, unlimit)] = 0
    metrics, logs = replay(scene, _hold(g1, q_arm), np.repeat(start[None], 2, axis=0), start)
    return metrics, logs


_REACH = np.array([-0.6, -0.3, 0.0, 0.3, 0.0, 0.0, 0.0])    # right_dex3_base at z = 1.007 m, hand forward


def test_static_hold_is_clean(g1) -> None:
    """Control: an in-range, collision-free pose held with its exact gravity feed-forward tracks to ~0 and trips only
    the static-reference criterion."""
    metrics, _ = _hold_metrics(g1, np.zeros(7), SceneConfig())
    assert metrics.arm_rms < 1e-3 and metrics.limit_violation_ticks == 0
    assert metrics.self_collision_steps == 0 and metrics.table_max_depth == 0.0
    assert metrics.failed_criteria == ("reference_does_not_move_object",)


def test_arm_into_the_torso_is_a_self_collision(g1) -> None:
    """Known answer: shoulder roll +1.5 rad swings the right upper arm into the torso (measured 28 mm deep)."""
    metrics, _ = _hold_metrics(g1, np.array([0.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0]), SceneConfig())
    assert metrics.self_collision_max_depth > 0.01 and metrics.self_collision_steps > 100
    assert "self_collision" in metrics.failed_criteria and not metrics.success


def test_table_through_the_hand_is_a_penetration(g1) -> None:
    """Known answer: the same reach pose over a table at 0.80 m is clear (control); a table top at 1.00 m cuts
    through the hand (base at 1.007 m), and the penetration criterion fires."""
    clear, _ = _hold_metrics(g1, _REACH, SceneConfig(table_z=0.80))
    assert clear.table_max_depth == 0.0 and "table_penetration" not in clear.failed_criteria
    hit, _ = _hold_metrics(g1, _REACH, SceneConfig(table_z=1.00))
    assert hit.table_max_depth > 0.01 and hit.table_penetration_steps > 100
    assert "table_penetration" in hit.failed_criteria


def test_command_past_a_limit_is_a_limit_violation(g1) -> None:
    """Known answer: the elbow commanded 0.3 rad past its MJCF upper limit (2.0944) is counted as a violation on
    every tick once it leaves the 0.01 rad band; with the limit constraint switched off it reaches the command
    (2.394 rad), with it on the soft limit still yields > 0.01 rad. An in-range command (control) has none."""
    q = np.array([-0.3, -0.3, 0.0, 2.0944 + 0.3, 0.0, 0.0, 0.0])
    free, logs = _hold_metrics(g1, q, SceneConfig(), unlimit="right_elbow_joint")
    assert logs["arm_q"][:, 3].max() == pytest.approx(2.3944, abs=5e-3)
    assert free.limit_violation_ticks > 200 and "joint_limits" in free.failed_criteria
    soft, logs = _hold_metrics(g1, q, SceneConfig())
    assert logs["arm_q"][:, 3].max() > 2.0944 + 0.01 and "joint_limits" in soft.failed_criteria
    ok, _ = _hold_metrics(g1, np.array([-0.3, -0.3, 0.0, 2.0, 0.0, 0.0, 0.0]), SceneConfig())
    assert ok.limit_violation_ticks == 0
