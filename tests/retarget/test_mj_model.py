"""Tests for `fastdo.retarget.mj_model.RobotScene`: layout, FK, point Jacobians, surface sampling.

Uses the baseline's shipped whisking demo scene (`third_party/do-as-i-do`, read-only) as the real
fixture; that submodule is never modified.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from fastdo.retarget.mj_model import RobotScene

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENE_XML = (REPO_ROOT / "third_party" / "do-as-i-do" / "retargeting" / "outputs" / "sharpa"
             / "right" / "whisking" / "0" / "scene.xml")
TRAJECTORY_NPZ = SCENE_XML.parent / "trajectory_kinematic.npz"

pytestmark = pytest.mark.skipif(not SCENE_XML.exists(), reason="baseline submodule scene.xml not checked out")


@pytest.fixture(scope="module")
def scene() -> RobotScene:
    return RobotScene.from_xml(SCENE_XML)


@pytest.fixture(scope="module")
def trajectory() -> dict[str, np.ndarray]:
    with np.load(TRAJECTORY_NPZ) as data:
        return {"qpos": data["qpos"], "qvel": data["qvel"], "frequency": data["frequency"]}


def test_joint_layout_covers_nq_with_expected_sizes(scene: RobotScene) -> None:
    layout = scene.joint_layout()
    assert set(layout) == {"wrist", "hand", "object", "wrist_dof", "hand_dof"}

    assert layout["wrist"] == slice(0, 6)
    assert layout["hand"] == slice(6, 28)
    assert layout["object"] == slice(28, 35)
    assert layout["wrist_dof"] == slice(0, 6)
    assert layout["hand_dof"] == slice(6, 28)

    assert scene.model.nq == 35
    assert scene.model.nv == 34
    assert scene.nv == 34
    assert scene.robot_dof == slice(0, 28)
    covered = set(range(layout["wrist"].start, layout["wrist"].stop))
    covered |= set(range(layout["hand"].start, layout["hand"].stop))
    covered |= set(range(layout["object"].start, layout["object"].stop))
    assert covered == set(range(35))


def test_fk_reproduces_object_and_wrist_pose_from_shipped_trajectory(
    scene: RobotScene, trajectory: dict[str, np.ndarray]
) -> None:
    qpos = trajectory["qpos"]
    layout = scene.joint_layout()

    # The body that owns the *last* wrist dof is the wrist's terminal frame; find it structurally
    # (not by a hardcoded body name) from the compiled model, matching mj_model's own derivation.
    last_wrist_dof = layout["wrist_dof"].stop - 1
    wrist_jid = next(j for j in range(scene.model.njnt) if int(scene.model.jnt_dofadr[j]) == last_wrist_dof)
    wrist_body = mujoco.mj_id2name(scene.model, mujoco.mjtObj.mjOBJ_BODY, int(scene.model.jnt_bodyid[wrist_jid]))
    object_jid = next(j for j in range(scene.model.njnt) if scene.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE)
    object_body = mujoco.mj_id2name(scene.model, mujoco.mjtObj.mjOBJ_BODY, int(scene.model.jnt_bodyid[object_jid]))

    for k in (0, qpos.shape[0] // 2, qpos.shape[0] - 1):
        q = qpos[k]
        poses = scene.fk(q[None, :])

        # Object: the free-joint qpos *is* the body's world pose directly.
        obj_qpos = q[layout["object"]]
        obj_pose = poses[object_body][0]
        assert np.abs(obj_pose[:3, 3] - obj_qpos[:3]).max() < 1e-9
        obj_quat_expected = obj_qpos[3:7] / np.linalg.norm(obj_qpos[3:7])
        obj_quat_actual = Rotation.from_matrix(obj_pose[:3, :3]).as_quat()  # xyzw
        obj_quat_actual = np.array([obj_quat_actual[3], *obj_quat_actual[:3]])  # -> wxyz
        sign_agnostic_err = min(np.abs(obj_quat_actual - obj_quat_expected).max(),
                                 np.abs(obj_quat_actual + obj_quat_expected).max())
        assert sign_agnostic_err < 1e-6

        # Wrist: translate-then-intrinsic-XYZ-Euler chain, reconstructed independently with scipy.
        wrist_qpos = q[layout["wrist"]]
        wrist_pose = poses[wrist_body][0]
        assert np.abs(wrist_pose[:3, 3] - wrist_qpos[:3]).max() < 1e-9
        expected_R = Rotation.from_euler("XYZ", wrist_qpos[3:6]).as_matrix()
        assert np.abs(wrist_pose[:3, :3] - expected_R).max() < 1e-9


def _random_valid_q(scene: RobotScene, seed: int) -> np.ndarray:
    """A second, non-degenerate configuration: wrist+hand dofs uniform within their joint ranges,
    object position near the origin and a fresh normalised random quaternion. This is *not* meant
    to replace testing against the real trajectory frame below -- a uniform-in-range q can still
    land near a joint limit or in an otherwise atypical pose that finite differences handle fine
    but that wouldn't by itself catch a bug only visible on realistic, non-degenerate hand poses
    (e.g. a sign error that cancels out at q=0). The shipped trajectory frame is what actually
    exercises a realistic hand-object contact configuration; this q only broadens dof coverage.
    """
    rng = np.random.default_rng(seed)
    q = np.zeros(scene.model.nq, dtype=np.float64)
    layout = scene.joint_layout()
    for j in range(scene.model.njnt):
        if scene.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        lo, hi = scene.model.jnt_range[j]
        adr = int(scene.model.jnt_qposadr[j])
        q[adr] = rng.uniform(lo, hi) if hi > lo else 0.0
    obj = layout["object"]
    q[obj.start:obj.start + 3] = rng.uniform(-0.05, 0.05, size=3)
    quat = rng.normal(size=4)
    q[obj.start + 3:obj.stop] = quat / np.linalg.norm(quat)
    return q


def test_point_jacobians_match_central_finite_differences(scene: RobotScene) -> None:
    with np.load(TRAJECTORY_NPZ) as data:
        q_real = data["qpos"][0].copy()   # a real, non-degenerate hand-object contact pose
    q_random = _random_valid_q(scene, seed=0)

    hand_body_names = ["right_hand_C_MC", "right_thumb_DP", "right_index_MP", "right_pinky_PP"]
    body_ids = np.array(
        [mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, name) for name in hand_body_names]
    )
    rng = np.random.default_rng(0)
    local_pts = rng.uniform(-0.01, 0.01, size=(len(body_ids), 3))

    for q0 in (q_real, q_random):
        analytic = scene.point_jacobians(q0, body_ids, local_pts)[0]   # (T=1,...) -> (U,3,28)

        eps = 1e-6
        nv = scene.model.nv
        fd_full = np.empty((len(body_ids), 3, nv), dtype=np.float64)
        for i in range(nv):
            dv = np.zeros(nv)
            dv[i] = eps
            q_plus = q0.copy()
            mujoco.mj_integratePos(scene.model, q_plus, dv, 1.0)
            wp_plus = scene.world_points(q_plus, body_ids, local_pts)

            dv[i] = -eps
            q_minus = q0.copy()
            mujoco.mj_integratePos(scene.model, q_minus, dv, 1.0)
            wp_minus = scene.world_points(q_minus, body_ids, local_pts)

            fd_full[:, :, i] = (wp_plus - wp_minus) / (2.0 * eps)

        # The object free joint's dofs (robot_dof.stop:nv) must be structurally zero for points
        # rigidly attached to the hand subtree -- verify this *before* trusting the sliced compare.
        object_dof_err = np.abs(fd_full[:, :, scene.robot_dof.stop:nv]).max()
        assert object_dof_err < 1e-9, f"object dofs not structurally zero: {object_dof_err}"

        fd = fd_full[:, :, scene.robot_dof]
        max_err = np.abs(analytic - fd).max()
        assert max_err < 1e-6, f"max FD error {max_err}"


def test_point_kinematics_shares_one_forward_pass(scene: RobotScene, trajectory: dict[str, np.ndarray]) -> None:
    qpos = trajectory["qpos"][:3]
    points, body_ids, _ = scene.surface_points(n_per_link=2, seed=0)
    n_dof = scene.robot_dof.stop - scene.robot_dof.start

    world, jac = scene.point_kinematics(qpos, body_ids, points, jacobian=True)
    assert world.shape == (3, len(body_ids), 3)
    assert jac is not None
    assert jac.shape == (3, len(body_ids), 3, n_dof)

    world_only, jac_none = scene.point_kinematics(qpos, body_ids, points, jacobian=False)
    assert jac_none is None
    assert np.array_equal(world_only, world)

    # point_jacobians/world_points are built on point_kinematics: same q must give identical output.
    batched_jac = scene.point_jacobians(qpos, body_ids, points)
    assert batched_jac.shape == (3, len(body_ids), 3, n_dof)
    assert np.array_equal(batched_jac, jac)
    for t in range(3):
        assert np.array_equal(scene.world_points(qpos[t], body_ids, points), world[t])
        single_jac = scene.point_jacobians(qpos[t], body_ids, points)   # (nq,) -> T=1 via atleast_2d
        assert single_jac.shape == (1, len(body_ids), 3, n_dof)
        assert np.array_equal(single_jac[0], jac[t])


def test_surface_points_cover_many_links_and_lie_on_their_geometry(scene: RobotScene) -> None:
    n_per_link = 30
    points, body_ids, body_names = scene.surface_points(n_per_link=n_per_link, seed=42)

    unique_bodies = sorted(set(body_ids.tolist()))
    assert len(unique_bodies) >= 10
    assert points.shape[0] > 100
    assert points.shape[0] == n_per_link * len(unique_bodies)
    assert len(body_names) == len(unique_bodies)
    assert body_names == [mujoco.mj_id2name(scene.model, mujoco.mjtObj.mjOBJ_BODY, b) for b in unique_bodies]

    # Independently rebuild each body's collision geometry (not via mj_model's private helper) and
    # check every sampled point lies within 2 mm of that body's own surface, in the body frame
    # (samples land exactly on the mesh surface by construction; 2 mm is generous slack for
    # primitive tessellation faceting on the smallest finger-link radii, ~7-9 mm).
    for b in unique_bodies:
        geom_ids = [g for g in range(scene.model.ngeom)
                    if scene.model.geom_bodyid[g] == b and scene.model.geom_group[g] == 3]
        assert geom_ids, f"body {b} has no collision geoms"
        meshes = []
        for g in geom_ids:
            gtype = scene.model.geom_type[g]
            size = scene.model.geom_size[g]
            if gtype == mujoco.mjtGeom.mjGEOM_BOX:
                m = trimesh.creation.box(extents=2.0 * size)
            elif gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
                m = trimesh.creation.cylinder(radius=float(size[0]), height=2.0 * float(size[1]))
            elif gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
                m = trimesh.creation.capsule(radius=float(size[0]), height=2.0 * float(size[1]))
            else:
                raise AssertionError(f"unexpected geom type {gtype} in fixture scene")
            transform = np.eye(4)
            transform[:3, :3] = Rotation.from_quat(
                [*scene.model.geom_quat[g][1:], scene.model.geom_quat[g][0]]  # wxyz -> xyzw
            ).as_matrix()
            transform[:3, 3] = scene.model.geom_pos[g]
            m.apply_transform(transform)
            meshes.append(m)
        combined = meshes[0] if len(meshes) == 1 else trimesh.util.concatenate(meshes)

        body_pts = points[body_ids == b]
        _, dist, _ = trimesh.proximity.closest_point(combined, body_pts)
        assert dist.max() < 0.002, f"body {b} max distance {dist.max()}"
