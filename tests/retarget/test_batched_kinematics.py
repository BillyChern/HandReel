"""Tests for `fastdo.retarget.batched_kinematics`: one `mj_kinematics` per frame,
Jacobian columns from joint axes and anchors on the static ancestor mask, canonical-frame Jacobians, and the
closed-form wrist chain. Real-scene tests use the whisking scene; the MJCF tests need no data.
"""

from __future__ import annotations

import time
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from fastdo.retarget.batched_kinematics import (
    BatchedKinematics,
    canonical_points,
    wrist_chain_is_closed_form,
    wrist_pose_closed_form,
    wrist_qpos0,
)
from fastdo.retarget.kinematic_terms import (
    wrist_prior_term,
    wrist_prior_term_closed_form,
    wrist_terminal_body,
)
from fastdo.retarget.mj_model import RobotScene
from fastdo.retarget.primitives import hand_capsule_covers
from fastdo.retarget.transport import robot_canonical_points
from tests.retarget.whisking_data import build_scene, scene_missing

DT = torch.float64
_BUILD_S: dict[str, float] = {}

_MJCF = """<mujoco model="wrist_chain">
  <worldbody>
    <body name="right_base_tx">
      <joint name="right_pos_x" type="slide" axis="1 0 0" range="-5 5"/>
      <geom type="sphere" size="0.004" contype="0" conaffinity="0"/>
      <body name="right_base_ty">
        <joint name="right_pos_y" type="slide" axis="0 1 0" ref="{pos_y_ref}" range="-5 5"/>
        <geom type="sphere" size="0.004" contype="0" conaffinity="0"/>
        <body name="right_base_tz">
          <joint name="right_pos_z" type="slide" axis="0 0 1" range="-5 5"/>
          <geom type="sphere" size="0.004" contype="0" conaffinity="0"/>
          <body name="right_base_roll" pos="{roll_pos}">
            <joint name="right_rot_x" type="hinge" axis="{rot_x_axis}" ref="{rot_x_ref}" range="-6.28 6.28"/>
            <geom type="sphere" size="0.004" contype="0" conaffinity="0"/>
            <body name="right_base_pitch">
              <joint name="right_rot_y" type="hinge" axis="0 1 0" range="-6.28 6.28"/>
              <geom type="sphere" size="0.004" contype="0" conaffinity="0"/>
              <body name="right_base_yaw">
                <joint name="right_rot_z" type="hinge" axis="0 0 1" range="-6.28 6.28"/>
                <geom type="sphere" size="0.004" contype="0" conaffinity="0"/>
                <body name="right_index" pos="0.05 0 0">
                  <joint name="right_index_MCP_FE" type="hinge" axis="0 1 0" range="-1 1"/>
                  <geom type="capsule" size="0.008 0.02" group="3"/>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
    <body name="object" pos="0.3 0 0">
      <freejoint name="object_joint"/>
      <geom type="sphere" size="0.03"/>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture(scope="module")
def scene() -> RobotScene:
    t0 = time.perf_counter()
    built = build_scene()
    _BUILD_S["scene"] = time.perf_counter() - t0
    return built


def _real_data(scene: RobotScene) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`q (7, 35)` with every robot dof uniform in its range (`default_rng(0)`), and the 96 OT samples plus the
    48 cover endpoints (144 points)."""
    model = scene.model
    q = np.zeros((7, 35))
    q[:, 28:35] = [0.1, -0.2, 0.3, 1.0, 0.0, 0.0, 0.0]
    rng = np.random.default_rng(0)
    for d in range(28):
        j = int(model.dof_jntid[d])
        lo, hi = model.jnt_range[j]
        q[:, model.jnt_qposadr[j]] = rng.uniform(lo, hi, 7)
    pts, ids, _ = scene.surface_points(6, seed=0)
    covers = hand_capsule_covers(scene)
    body_ids = np.concatenate([ids, covers.body_ids, covers.body_ids])
    local = np.concatenate([pts, covers.local_a, covers.local_b])
    return q, body_ids, local


def _identity_blocks(n: int) -> torch.Tensor:
    return torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], dtype=DT).expand(n, 7)


@scene_missing
def test_world_points_and_jacobians_match_point_kinematics(scene: RobotScene) -> None:
    q, body_ids, local = _real_data(scene)
    bk = BatchedKinematics.from_scene(scene)
    assert bk.n_dof == 28
    batch = bk.forward(q)
    world = bk.world_points(batch, body_ids, local)
    world_ref, jac_ref = scene.point_kinematics(q, body_ids, local)
    assert jac_ref is not None
    assert float((world - torch.from_numpy(world_ref)).abs().max()) <= 1e-12
    obj_block = _identity_blocks(7)
    jac = bk.canonical_jacobian(batch, body_ids, canonical_points(world, obj_block, 1.0), obj_block, 1.0)
    assert jac.shape == (7, 144, 3, 28)
    assert float((jac - torch.from_numpy(jac_ref)).abs().max()) <= 1e-12


@scene_missing
def test_canonical_jacobian_matches_robot_canonical_points(scene: RobotScene) -> None:
    q, body_ids, local = _real_data(scene)
    mesh_scale = 0.18086324855404615
    trans = np.random.default_rng(3).uniform(-0.5, 0.5, (7, 3))
    quat = Rotation.random(7, random_state=3).as_quat()[:, [3, 0, 1, 2]]
    q[:, 28:31], q[:, 31:35] = trans, quat
    obj_block = torch.from_numpy(q[:, 28:35])
    bk = BatchedKinematics.from_scene(scene)
    batch = bk.forward(q)
    canon = canonical_points(bk.world_points(batch, body_ids, local), obj_block, mesh_scale)
    jac = bk.canonical_jacobian(batch, body_ids, canon, obj_block, mesh_scale)
    world_ref, jac_ref = scene.point_kinematics(q, body_ids, local)
    assert jac_ref is not None
    canon_ref, canon_jac_ref = robot_canonical_points(torch.from_numpy(q), torch.from_numpy(world_ref),
                                                      torch.from_numpy(jac_ref), mesh_scale)
    assert float((canon - canon_ref).abs().max()) <= 1e-12
    assert float((jac - canon_jac_ref).abs().max()) <= 1e-11


@scene_missing
def test_jacobian_is_exactly_zero_off_the_ancestor_mask(scene: RobotScene) -> None:
    """Off the 1264 static (point, ancestor dof) pairs every column is exactly 0. On them, the columns that exceed
    1e-12 on every frame are exactly those of MuJoCo's `mj_jac` (`point_kinematics`): 1248 on this data. The other 16
    are geometric zeros: 14 cover endpoints on their joint's anchor (exactly 0 in `mj_jac`) and 2 palm-box cover
    endpoints on the `right_rot_z` axis (<= 3.9e-16). The spec's 1250 counted columns > 0, not > 1e-12."""
    q, body_ids, local = _real_data(scene)
    bk = BatchedKinematics.from_scene(scene)
    pu, pd = bk.ancestor_pairs(body_ids)
    assert pu.dtype == np.int64 and pd.dtype == np.int64 and len(pu) == 1264
    obj_block = torch.from_numpy(q[:, 28:35])
    batch = bk.forward(q)
    jac = bk.canonical_jacobian(batch, body_ids, canonical_points(bk.world_points(batch, body_ids, local),
                                                                  obj_block, 1.0), obj_block, 1.0)
    columns = jac.permute(0, 1, 3, 2)                                            # (T, U, 28, 3)
    mask = torch.zeros(len(body_ids), 28, dtype=torch.bool)
    mask[torch.from_numpy(pu), torch.from_numpy(pd)] = True
    off = columns[:, ~mask]
    assert torch.equal(off, torch.zeros_like(off))
    on = columns[:, torch.from_numpy(pu), torch.from_numpy(pd)].abs().amax(dim=-1)   # (T, 1264)
    _, jac_ref = scene.point_kinematics(q, body_ids, local)
    assert jac_ref is not None
    on_ref = torch.from_numpy(jac_ref).permute(0, 1, 3, 2)[:, torch.from_numpy(pu), torch.from_numpy(pd)].abs().amax(-1)
    populated = (on > 1e-12).all(dim=0)
    assert torch.equal(populated, (on_ref > 1e-12).all(dim=0))
    assert int(populated.sum()) == 1248
    assert int((on_ref == 0).all(dim=0).sum()) == 14


@scene_missing
def test_body_pose_matches_fk(scene: RobotScene) -> None:
    q, _, _ = _real_data(scene)
    name = wrist_terminal_body(scene)
    body = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, name)
    bk = BatchedKinematics.from_scene(scene)
    pose = bk.body_pose(bk.forward(q), body)
    assert pose.shape == (7, 4, 4)
    assert float((pose - torch.from_numpy(scene.fk(q)[name])).abs().max()) <= 1e-12


@scene_missing
def test_forward_leaves_scene_data_untouched(scene: RobotScene) -> None:
    q, _, _ = _real_data(scene)
    before = scene.data.qpos.copy()
    BatchedKinematics.from_scene(scene).forward(q)
    assert np.array_equal(scene.data.qpos, before)


@scene_missing
def test_wrist_chain_closed_form_matches_mujoco(scene: RobotScene) -> None:
    assert wrist_chain_is_closed_form(scene)
    rng = np.random.default_rng(1)
    q = np.zeros((50, 35))
    q[:, 31] = 1.0
    q[:, 3:6] = rng.uniform(-np.pi, np.pi, (50, 3))
    q[:, 0:3] = rng.uniform(-0.5, 0.5, (50, 3))
    pose = wrist_pose_closed_form(torch.from_numpy(q[:, :6]), wrist_qpos0(scene))
    assert float((pose - torch.from_numpy(scene.fk(q)[wrist_terminal_body(scene)])).abs().max()) <= 1e-12


def _write_variant(tmp_path: Path, name: str, roll_pos: str = "0 0 0", rot_x_axis: str = "1 0 0",
                   rot_x_ref: str = "0", pos_y_ref: str = "0") -> Path:
    path = tmp_path / f"{name}.xml"
    path.write_text(_MJCF.format(roll_pos=roll_pos, rot_x_axis=rot_x_axis, rot_x_ref=rot_x_ref, pos_y_ref=pos_y_ref))
    return path


def test_wrist_chain_detector_on_minimal_mjcf(tmp_path: Path) -> None:
    variant_a = RobotScene.from_xml(_write_variant(tmp_path, "a"))
    assert variant_a.robot_dof == slice(0, 7) and wrist_terminal_body(variant_a) == "right_base_yaw"
    assert wrist_chain_is_closed_form(variant_a)
    rng = np.random.default_rng(2)
    q = np.zeros((10, variant_a.model.nq))
    q[:, 0:3] = rng.uniform(-0.5, 0.5, (10, 3))
    q[:, 3:6] = rng.uniform(-np.pi, np.pi, (10, 3))
    q[:, 7:14] = [0.3, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    pose = wrist_pose_closed_form(torch.from_numpy(q[:, :6]), wrist_qpos0(variant_a))
    assert float((pose - torch.from_numpy(variant_a.fk(q)["right_base_yaw"])).abs().max()) <= 1e-12

    assert not wrist_chain_is_closed_form(RobotScene.from_xml(_write_variant(tmp_path, "b", roll_pos="0 0 0.01")))
    assert not wrist_chain_is_closed_form(RobotScene.from_xml(_write_variant(tmp_path, "c", rot_x_axis="0 1 0")))


def test_closed_form_honours_joint_reference_offsets(tmp_path: Path) -> None:
    """MuJoCo moves a hinge or slide by `qpos - qpos0`, and a joint `ref` compiles to `qpos0` (hinge refs in degrees
    under the default compiler). A variant, ref 0.3 on `right_rot_x`, passes the detector while the
    closed form without the offset is 4.9e-3 off `fk`; given `wrist_qpos0` the pose and the closed-form prior (its
    residual and all 13 difference configurations) match `fk` and the `scene.fk` term, also for a slide ref."""
    rng = np.random.default_rng(4)
    for name, kwargs, expected in (("ref_rot_x", {"rot_x_ref": "0.3"}, [0.0, 0.0, 0.0, np.radians(0.3), 0.0, 0.0]),
                                   ("ref_pos_y", {"pos_y_ref": "0.2"}, [0.0, 0.2, 0.0, 0.0, 0.0, 0.0])):
        variant = RobotScene.from_xml(_write_variant(tmp_path, name, **kwargs))
        assert wrist_chain_is_closed_form(variant), name
        qpos0 = wrist_qpos0(variant)
        assert qpos0.dtype == DT and qpos0.shape == (6,)
        assert float((qpos0 - torch.tensor(expected, dtype=DT)).abs().max()) <= 1e-15, name
        q = np.zeros((10, variant.model.nq))
        q[:, 0:3] = rng.uniform(-0.5, 0.5, (10, 3))
        q[:, 3:6] = rng.uniform(-np.pi, np.pi, (10, 3))
        q[:, 7:14] = [0.3, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        q_wrist = torch.from_numpy(q[:, :6])
        fk = torch.from_numpy(variant.fk(q)["right_base_yaw"])
        assert float((wrist_pose_closed_form(q_wrist, qpos0) - fk).abs().max()) <= 1e-10, name
        assert float((wrist_pose_closed_form(q_wrist, torch.zeros(6, dtype=DT)) - fk).abs().max()) > 1e-3, name

        q_target = q.copy()
        q_target[:, :6] += 0.1 * rng.normal(size=(10, 6))
        targets = torch.from_numpy(variant.fk(q_target)["right_base_yaw"])
        ref = wrist_prior_term(torch.from_numpy(q), variant, "right_base_yaw", targets, 100.0, 1e-5)
        got = wrist_prior_term_closed_form(q_wrist, targets, 100.0, 1e-5, True, qpos0=qpos0)
        assert float((got.residual - ref.residual).abs().max()) <= 1e-10, name
        assert float((got.jacobian - ref.jacobian).abs().max()) <= 1e-8, name
        trial = wrist_prior_term_closed_form(q_wrist, targets, 100.0, 1e-5, False, qpos0=qpos0)
        assert float((trial.residual - ref.residual).abs().max()) <= 1e-10, name


def test_from_scene_rejects_a_non_hinge_slide_robot_dof(tmp_path: Path) -> None:
    scene = RobotScene.from_xml(_write_variant(tmp_path, "a"))
    joint = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_JOINT, "right_index_MCP_FE")
    scene.model.jnt_type[joint] = mujoco.mjtJoint.mjJNT_BALL
    with pytest.raises(ValueError, match="neither hinge nor slide"):
        BatchedKinematics.from_scene(scene)
