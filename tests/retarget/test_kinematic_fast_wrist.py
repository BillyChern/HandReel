"""Tests for `FastTermBuilder`'s wrist prior on wrist chains other than the shipped
whisking scene's, each against the legacy `scene.fk` term `wrist_prior_term`:

- nonzero joint reference offsets: a joint `ref` compiles to `model.qpos0` and MuJoCo moves a hinge or slide by
  `qpos - qpos0`, so the closed form must take the wrist joints' `qpos0`;
- a chain the closed-form detector rejects, where the batched `mj_kinematics` fallback runs.

Each test edits the compiled model of a fresh whisking `RobotScene` in memory (MuJoCo reads `qpos0` and `body_pos`
on every `mj_kinematics`) and builds the terms on the first `N_FRAMES` whisking frames.
"""

from __future__ import annotations

import logging
from typing import Any

import mujoco
import numpy as np
import pytest
import torch
from torch import Tensor

from fastdo.retarget.batched_kinematics import wrist_qpos0
from fastdo.retarget.contact import human_contact_measure
from fastdo.retarget.kinematic import (
    RetargetConfig,
    _human_atom_subset,
    _obj_qpos_block,
    _wrist_init_batch,
)
from fastdo.retarget.kinematic_fast import FastTermBuilder
from fastdo.retarget.kinematic_terms import (
    wrist_prior_term,
    wrist_prior_term_closed_form,
    wrist_terminal_body,
)
from fastdo.retarget.mj_model import RobotScene
from fastdo.solver.sdf import SdfGrid
from tests.retarget.whisking_data import (
    build_scene,
    build_whisk_sdf_and_surface,
    load_whisking_inputs,
    whisking_data_missing,
)

DT = torch.float64
N_FRAMES = 8
logger = logging.getLogger(__name__)
SdfSurface = tuple[SdfGrid, Tensor, Tensor]


@pytest.fixture(scope="module")
def whisk_sdf_and_surface() -> SdfSurface:
    return build_whisk_sdf_and_surface()


def _builder_and_state(scene: RobotScene, sdf_surface: SdfSurface) -> tuple[FastTermBuilder, Tensor, Tensor, Tensor]:
    """`FastTermBuilder` on the first `N_FRAMES` whisking frames, built as `retarget_kinematic_fast` builds it; a
    wrist state 0.05 (m or rad) off the prior's Euler init with the fingers mid-range (`default_rng(7)`); the
    object qpos block; the wrist targets."""
    cfg = RetargetConfig()
    sdf, surface, normals = sdf_surface
    inputs: dict[str, Any] = load_whisking_inputs(np.arange(N_FRAMES))
    obj_block = _obj_qpos_block(inputs["obj_poses"])
    target = inputs["wrist_poses"].to(DT)
    subset = _human_atom_subset(int(inputs["hand_labels"].shape[0]), cfg.n_human_atoms, cfg.seed)
    human = human_contact_measure(inputs["hand_verts"][:, subset], inputs["obj_poses"], inputs["mesh_scale"],
                                  inputs["hand_labels"][subset], sdf, surface, sigma=cfg.sigma, tau=cfg.tau)
    builder = FastTermBuilder(scene, sdf, surface, normals, human, obj_block, target, inputs["mesh_scale"], cfg)
    noise = torch.from_numpy(np.random.default_rng(7).standard_normal((N_FRAMES, 6)))
    x = torch.zeros(N_FRAMES, builder.n_dof, dtype=DT)
    x[:, :6] = _wrist_init_batch(target) + 0.05 * noise
    x[:, 6:] = (builder.q_min[6:] + builder.q_max[6:]) / 2.0
    return builder, x, obj_block, target


def _assert_prior_matches_the_fk_term(label: str, scene: RobotScene, builder: FastTermBuilder, x: Tensor,
                                      obj_block: Tensor, target: Tensor) -> None:
    """The builder's refresh prior (residual <= 1e-12, central-difference Jacobian <= 1e-8, hand columns exactly 0)
    and trial prior (residual, zero Jacobian) against `wrist_prior_term` on the full qpos."""
    cfg = RetargetConfig()
    ref = wrist_prior_term(torch.cat([x, obj_block], dim=1), scene, wrist_terminal_body(scene), target, cfg.lam_wrist,
                           cfg.fd_eps_wrist, need_jacobian=True)
    batch = builder._geometry(x)[0]
    got = builder._wrist_prior(x, batch, refresh=True)
    trial = builder._wrist_prior(x, batch, refresh=False)
    residual_gap = float((got.residual - ref.residual).abs().max())
    jacobian_gap = float((got.jacobian - ref.jacobian).abs().max())
    logger.info("%s: closed form %s, residual gap %.3g, Jacobian gap %.3g (max |J| %.3g)", label,
                builder._closed_form, residual_gap, jacobian_gap, float(ref.jacobian.abs().max()))
    assert float(ref.jacobian[:, :, :6].abs().max()) > 1.0
    assert residual_gap <= 1e-12
    assert jacobian_gap <= 1e-8
    assert torch.equal(got.jacobian[:, :, 6:], torch.zeros_like(got.jacobian[:, :, 6:]))
    assert float((trial.residual - ref.residual).abs().max()) <= 1e-12
    assert trial.jacobian.shape == ref.jacobian.shape and not trial.jacobian.any()


@whisking_data_missing
def test_closed_form_wrist_prior_honours_joint_reference_offsets(whisk_sdf_and_surface: SdfSurface) -> None:
    """Offsets on a slide (`right_pos_y`, 0.02 m) and a hinge (`right_rot_x`, 0.3 rad): the detector still accepts
    the chain (reference offsets keep its structure), and the builder's closed-form prior equals the `scene.fk`
    term, so the builder passes the wrist joints' `qpos0` (reviewer's C3c finding: a ref passed the detector while
    the closed form ignored it)."""
    scene = build_scene()
    model = scene.model
    for joint, offset in (("right_pos_y", 0.02), ("right_rot_x", 0.3)):
        model.qpos0[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)]] = offset
    builder, x, obj_block, target = _builder_and_state(scene, whisk_sdf_and_surface)
    assert builder._closed_form
    _assert_prior_matches_the_fk_term("qpos0 offsets", scene, builder, x, obj_block, target)


@whisking_data_missing
def test_fallback_wrist_prior_matches_the_fk_term(whisk_sdf_and_surface: SdfSurface) -> None:
    """A 1 cm `body_pos` offset of `right_base_pitch` breaks the closed-form structure: the detector rejects the
    chain, the builder takes the batched `mj_kinematics` fallback with its own central difference, and that prior
    equals the `scene.fk` term within the closed-form test's tolerances (residual 1e-12, Jacobian 1e-8). The closed
    form is more than 1e-3 off on this chain, so a builder that took it here would fail."""
    scene = build_scene()
    pitch = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, "right_base_pitch")
    scene.model.body_pos[pitch] = [0.0, 0.0, 0.01]
    builder, x, obj_block, target = _builder_and_state(scene, whisk_sdf_and_surface)
    assert not builder._closed_form
    cfg = RetargetConfig()
    closed = wrist_prior_term_closed_form(x[:, :6], target, cfg.lam_wrist, cfg.fd_eps_wrist, False,
                                          qpos0=wrist_qpos0(scene))
    fk_term = wrist_prior_term(torch.cat([x, obj_block], dim=1), scene, wrist_terminal_body(scene), target,
                               cfg.lam_wrist, cfg.fd_eps_wrist, need_jacobian=False)
    assert float((closed.residual - fk_term.residual).abs().max()) > 1e-3
    _assert_prior_matches_the_fk_term("fallback", scene, builder, x, obj_block, target)
