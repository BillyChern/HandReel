"""Options of the fast kinematic reference, on the real Sharpa Wave scene (CPU MuJoCo, synthetic inputs).

The fingertip targets are the robot's own tip sites at a configuration `q_bad` that breaks three joint limits (index PIP
below 0, middle MCP_AA and ring DIP above their ranges); the wrist target is `q_bad`'s wrist pose; the object sits 2 m
away (no contact, no clearance activity), so the fingertip prior is what moves the fingers.

- `joint_limits="penalty"` with a weak `lam_lim` follows the targets out of the box on at least one dof (the
  sensitivity/mutation check of the feasibility assertion); `joint_limits="projected"` keeps every limited dof inside `jnt_range` EXACTLY on every
  frame, and pins at least one joint AT its limit (a projection, not a barrier).
- The fingertip prior reaches in-range targets to sub-millimetre accuracy, while `lam_tip=0` (the default) leaves the
  fingers at their mid-range initialization.
- Masked frames: NaN hand vertices on a frame with `frame_valid` False never reach the solve.
- `RetargetConfig` refuses an unknown `joint_limits` or a negative `lam_tip`; the legacy path refuses `lam_tip`.
"""

from __future__ import annotations

import os
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch
import trimesh

from fastdo.retarget.kinematic import MANO_TIP_VERTICES, RetargetConfig, retarget_kinematic
from fastdo.retarget.kinematic_fast import TIP_FINGERS, tip_sites
from fastdo.retarget.kinematic_terms import robot_joint_limits, wrist_terminal_body
from fastdo.retarget.mj_model import RobotScene
from fastdo.solver.sdf import mesh_sdf_grid

_REPO = Path(__file__).resolve().parents[2]
_REL = Path("retargeting/outputs/sharpa/right/whisking/0/scene.xml")
_CANDIDATES = [_REPO / "third_party/do-as-i-do" / _REL,
               *([Path(os.environ["FASTDO_BASELINE_ROOT"]) / _REL] if os.environ.get("FASTDO_BASELINE_ROOT") else []),
               Path("/opt/fastdo/handreel/third_party/do-as-i-do") / _REL]
SCENE_XML = next((p for p in _CANDIDATES if p.is_file()), None)
pytestmark = pytest.mark.skipif(SCENE_XML is None, reason="Sharpa right scene (DO AS I DO whisking scene.xml) missing")

T = 4
FAST = {"warm_start_max_iter": 4, "joint_max_iter": 12}
# joint index within the 28 robot dofs (6 wrist + 22 hand; tests/retarget scene layout): value outside jnt_range
BREAKS = {13: -0.4, 16: 0.6, 22: 1.8}          # index_PIP < 0, middle_MCP_AA > 0.349, ring_DIP > 1.396


@pytest.fixture(scope="module")
def scene() -> RobotScene:
    assert SCENE_XML is not None
    return RobotScene.from_xml(SCENE_XML)


def _fk(scene: RobotScene, q28: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`(tips (5,3), wrist-terminal body pose (4,4))` at robot dofs `q28` (object block at identity)."""
    model, data = scene.model, mujoco.MjData(scene.model)
    data.qpos[:28] = q28
    data.qpos[31] = 1.0
    mujoco.mj_kinematics(model, data)
    body, local = tip_sites(scene)
    tips = np.stack([data.xpos[b] + data.xmat[b].reshape(3, 3) @ p for b, p in zip(body, local, strict=True)])
    wb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, wrist_terminal_body(scene))
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = data.xmat[wb].reshape(3, 3), data.xpos[wb]
    return tips, pose


def _problem(scene: RobotScene, q28: np.ndarray):
    tips, wrist = _fk(scene, q28)
    verts = np.full((T, 778, 3), 0.3)
    verts[:, list(MANO_TIP_VERTICES)] = tips
    obj = np.tile(np.eye(4), (T, 1, 1))
    obj[:, :3, 3] = [2.0, 2.0, 2.0]
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.5)
    sdf = mesh_sdf_grid(np.asarray(sphere.vertices), np.asarray(sphere.faces, dtype=np.int64), resolution=16,
                        padding=0.1)
    surface = torch.from_numpy(np.asarray(sphere.vertices, dtype=np.float64))
    return (torch.from_numpy(obj), torch.from_numpy(np.tile(wrist, (T, 1, 1))), 0.1, torch.from_numpy(verts),
            torch.zeros(778, dtype=torch.int8), scene, sdf, surface, surface / surface.norm(dim=1, keepdim=True), 30.0)


def _q_bad(scene: RobotScene) -> np.ndarray:
    q_min, q_max, _ = (t.numpy() for t in robot_joint_limits(scene.model, scene.robot_dof))
    q = np.zeros(28)
    q[:3] = [0.0, 0.0, 0.3]
    q[6:] = 0.5 * (q_min[6:] + q_max[6:])
    for j, v in BREAKS.items():
        q[j] = v
    return q


def _violation(scene: RobotScene, qpos: np.ndarray) -> np.ndarray:
    q_min, q_max, limited = (t.numpy() for t in robot_joint_limits(scene.model, scene.robot_dof))
    q = qpos[:, :28]
    return ((np.clip(q - q_max, 0, None) + np.clip(q_min - q, 0, None)) * limited).max(axis=0)


def test_projected_limits_hold_exactly_where_the_penalty_does_not(scene: RobotScene) -> None:
    args = _problem(scene, _q_bad(scene))
    penalty = retarget_kinematic(*args, RetargetConfig(lam_tip=1e5, lam_lim=1.0, **FAST))
    projected = retarget_kinematic(*args, RetargetConfig(lam_tip=1e5, lam_lim=1.0, joint_limits="projected", **FAST))
    broken = _violation(scene, penalty.qpos)
    assert broken.max() > 0.05                                               # sensitivity: the check can fail
    assert _violation(scene, projected.qpos).max() == 0.0                    # exactly inside, every dof, every frame
    q_min, q_max, _ = (t.numpy() for t in robot_joint_limits(scene.model, scene.robot_dof))
    at_bound = (projected.qpos[:, :28] == q_min) | (projected.qpos[:, :28] == q_max)
    assert at_bound[:, broken > 0.05].all()                                  # held AT the limit, not pushed inside
    assert np.isfinite(projected.qvel).all()


def test_fingertip_prior_reaches_in_range_targets(scene: RobotScene) -> None:
    q = _q_bad(scene)
    for j in BREAKS:
        q[j] = 0.2                                                           # inside every range
    q[[11, 15, 19, 24]] = 1.0                                                # flex the MCPs away from mid-range
    tips, _ = _fk(scene, q)
    tipped = retarget_kinematic(*_problem(scene, q), RetargetConfig(lam_tip=1e5, joint_limits="projected", **FAST))
    free = retarget_kinematic(*_problem(scene, q), RetargetConfig(**FAST))
    err_tip = np.linalg.norm(np.stack([_fk(scene, row[:28])[0] for row in tipped.qpos]) - tips, axis=2)
    err_free = np.linalg.norm(np.stack([_fk(scene, row[:28])[0] for row in free.qpos]) - tips, axis=2)
    assert err_tip.max() < 1e-3 and err_free.max() > 1e-2
    assert len(TIP_FINGERS) == len(MANO_TIP_VERTICES) == 5


def test_masked_frame_vertices_never_reach_the_solve(scene: RobotScene) -> None:
    obj, wrist, scale, verts, labels, sc, sdf, surface, normals, freq = _problem(scene, _q_bad(scene))
    valid = torch.tensor([True, False, True, True])
    clean = retarget_kinematic(obj, wrist, scale, verts, labels, sc, sdf, surface, normals, freq,
                               RetargetConfig(lam_tip=1e4, joint_limits="projected", **FAST), frame_valid=valid)
    poisoned = verts.clone()
    poisoned[1] = float("nan")
    out = retarget_kinematic(obj, wrist, scale, poisoned, labels, sc, sdf, surface, normals, freq,
                             RetargetConfig(lam_tip=1e4, joint_limits="projected", **FAST), frame_valid=valid)
    assert np.array_equal(out.qpos, clean.qpos) and np.isfinite(out.qpos).all()


def test_config_refuses_bad_values_and_the_legacy_path_refuses_the_tip_prior(scene: RobotScene) -> None:
    with pytest.raises(ValueError, match="joint_limits"):
        RetargetConfig(joint_limits="barrier")
    with pytest.raises(ValueError, match="lam_tip"):
        RetargetConfig(lam_tip=-1.0)
    assert RetargetConfig().joint_limits == "penalty" and RetargetConfig().lam_tip == 0.0     # defaults: off
    with pytest.raises(NotImplementedError, match="lam_tip"):
        retarget_kinematic(*_problem(scene, _q_bad(scene)), RetargetConfig(lam_tip=1.0, fast_path=False))
