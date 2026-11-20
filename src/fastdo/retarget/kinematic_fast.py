"""Fast path of `fastdo.retarget.kinematic.retarget_kinematic`: the legacy path's objective, schedule and
warm starts, evaluated with exact levers only, so the solution keeps parity with the legacy path (qpos within
1e-12 on the first 20 whisking frames and equal per-stage LM iterations).

Levers: L1 one `mj_kinematics` per frame with Jacobian columns from joint axes and anchors on the static ancestor
pairs, and no Jacobian at all in trial evaluations (`fastdo.retarget.batched_kinematics`); L2 the closed-form wrist
prior (the wrist joints' `qpos0` subtracted as in MuJoCo) with its 13 finite-difference configurations in one
batched call, else a batched `mj_kinematics` fallback; L3 + L9 only atoms with band weight > 0
enter Sinkhorn (`contact_transport_term_inband`, uniform-b Sinkhorn with constants merged once); L4 single-thread
BLAS around `solveh_banded`; L5 cached banded-storage indices; L6 the human contact measure on the 64-atom subset
first; L7 value-only SDF lookups in the capsule-axis search; L8 an accepted step's refresh reuses the geometry of
the trial that evaluated the very same tensor (same `t*` warm start: only refreshes write it); L10 canonical-frame
Jacobians computed directly; L11 one concatenated `BlockTerm`; L12 no refresh after the step that ends an LM stage
(its plan, Jacobians and cost are never read; the next stage's `t*` is the reused trial search's).

Nothing else changes: no warm-started duals, no reduced `n_sinkhorn`, no schedule change, no kNN columns, no GPU.

Masked frames (hand rows the hands stage marks invalid must not create contacts or enter the
retargeting cost). `frame_valid (T,)` False marks a frame whose hand input is not evidence. Such a frame's hand
vertices and wrist target are first replaced by those of the nearest valid frame (the earlier one on a tie), so
nothing of them reaches the solve, the wrist initialization included; then its human band weights are zeroed (no
in-band atom, hence no transport residual and no band mass) and its wrist-prior rows get weight 0. The frame keeps the
object-clearance, joint-limit and smoothness terms, so the reference passes through it smoothly and without
penetrating. `frame_valid=None` or all True runs the unmasked code path unchanged.

Joint-limit and fingertip options of `RetargetConfig` (both off by default; off leaves every tensor of the solve
as before):
`joint_limits="projected"` passes the limited dofs' box to every `lm_optimize_blocks` stage (a projected LM, so the
reference never leaves the Sharpa joint ranges); `lam_tip > 0` appends the five fingertip sites to the kinematic
points and adds `sqrt(lam_tip) * (p_tip - p_human_tip)` (metres, 15 rows per frame, weight 0 on masked frames) to
the per-frame block (`FastTermBuilder`, `tip_sites`).
"""

from __future__ import annotations

import dataclasses
import logging
import math
import time
from typing import TYPE_CHECKING

import mujoco
import numpy as np
import torch
from torch import Tensor

from fastdo.geometry.lie import se3_inv, se3_log
from fastdo.retarget.batched_kinematics import (
    BatchedKinematics,
    KinematicsBatch,
    canonical_points,
    wrist_chain_is_closed_form,
    wrist_qpos0,
)
from fastdo.retarget.contact import ContactMeasure, human_contact_measure
from fastdo.retarget.kinematic import (
    MANO_TIP_VERTICES,
    KinematicTrajectory,
    _geometric_schedule,
    _human_atom_subset,
    _obj_qpos_block,
    _wrist_init_batch,
    joint_bounds,
)
from fastdo.retarget.kinematic_terms import (
    contact_band_term,
    joint_limit_term,
    primitive_penetration_term,
    robot_joint_limits,
    smoothness_term,
    wrist_prior_term_closed_form,
    wrist_terminal_body,
)
from fastdo.retarget.labels import robot_link_labels
from fastdo.retarget.mj_model import RobotScene
from fastdo.retarget.primitives import (
    CapsuleDistance,
    as_phi_fn,
    as_values_fn,
    capsule_min_distance,
    hand_capsule_covers,
)
from fastdo.retarget.qvel import kinematic_qvel
from fastdo.retarget.transport import (
    canonical_diameter,
    contact_transport_term_inband,
    inband_measure,
)
from fastdo.solver.lm_blocks import BlockLMConfig, BlockTerm, SmoothTerm, lm_optimize_blocks
from fastdo.solver.sdf import SdfGrid

if TYPE_CHECKING:
    from fastdo.retarget.kinematic import RetargetConfig

logger = logging.getLogger(__name__)

__all__ = ["TIP_FINGERS", "FastTermBuilder", "retarget_kinematic_fast", "tip_sites"]

_DT = torch.float64
TIP_FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def tip_sites(scene: RobotScene) -> tuple[np.ndarray, np.ndarray]:
    """`(body_ids (5,), local positions (5,3))` of the scene's `<side>_<finger>_tip` sites in `TIP_FINGERS` order, the
    side read from its `<side>_palm` site (DO AS I DO's IK targets).

    Raises:
        ValueError: If the scene has no `right_palm`/`left_palm` site or lacks a fingertip site.
    """
    model = scene.model
    sides = [s for s in ("right", "left") if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{s}_palm") >= 0]
    if len(sides) != 1:
        raise ValueError(f"expected exactly one of right_palm / left_palm sites in the scene, found {sides}")
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{sides[0]}_{f}_tip") for f in TIP_FINGERS]
    if min(ids) < 0:
        raise ValueError(f"the scene lacks a {sides[0]}_<finger>_tip site ({dict(zip(TIP_FINGERS, ids, strict=True))})")
    return (np.asarray(model.site_bodyid[ids], dtype=np.int64),
            np.asarray(model.site_pos[ids], dtype=np.float64).copy())


class FastTermBuilder:
    """Evaluates the legacy path's terms [transport, primitive, joint limits, wrist prior, band] as one `BlockTerm`,
    plus per-dof smoothness, for `lm_optimize_blocks`. `frame_weight (T,)`, when given, scales the wrist prior's row
    weights per frame (0 on masked frames; module docstring)."""

    def __init__(self, scene: RobotScene, sdf: SdfGrid, surface: Tensor, surface_normals: Tensor,
                 human: ContactMeasure, obj_block: Tensor, target_wrist: Tensor, scale: float,
                 cfg: RetargetConfig, frame_weight: Tensor | None = None, tip_targets: Tensor | None = None) -> None:
        self._cfg, self._sdf, self._scale = cfg, sdf, scale
        self._obj_block, self._target_wrist = obj_block, target_wrist
        self._frame_weight = frame_weight
        covers = hand_capsule_covers(scene, cfg.box_face_tol)
        local_pts, sample_ids, _ = scene.surface_points(cfg.n_per_link, seed=cfg.seed)
        self._covers, self._sample_ids = covers, sample_ids
        self._U, self._P = int(sample_ids.shape[0]), covers.n
        body_ids = [sample_ids, covers.body_ids, covers.body_ids]
        local = [local_pts, covers.local_a, covers.local_b]
        self._tip_canon: Tensor | None = None
        if cfg.lam_tip > 0.0:
            if tip_targets is None or tuple(tip_targets.shape) != (obj_block.shape[0], len(TIP_FINGERS), 3):
                raise ValueError(f"lam_tip > 0 needs tip_targets ({obj_block.shape[0]}, {len(TIP_FINGERS)}, 3), got "
                                 f"{None if tip_targets is None else tuple(tip_targets.shape)}")
            tip_body, tip_local = tip_sites(scene)
            body_ids.append(tip_body)
            local.append(tip_local)
            self._tip_canon = canonical_points(tip_targets.to(_DT), obj_block, scale)       # (T,5,3)
        self._body_ids = np.concatenate(body_ids)
        self._local = np.concatenate(local)
        self._radius = torch.from_numpy(covers.radius) / scale
        self._robot_labels = robot_link_labels(scene.model, sample_ids)
        self.q_min, self.q_max, self._limited = robot_joint_limits(scene.model, scene.robot_dof)
        self.n_dof = scene.robot_dof.stop - scene.robot_dof.start
        self._lam_smooth = torch.full((self.n_dof,), cfg.lam_q, dtype=_DT)
        slides = scene.layout["wrist_dof"].start - scene.robot_dof.start
        self._lam_smooth[slides:slides + 3] = cfg.lam_q_slide
        self.diameter = canonical_diameter(surface.to(_DT))
        self._measure = inband_measure(human, surface_normals)
        self._kin = BatchedKinematics.from_scene(scene)
        self._closed_form = wrist_chain_is_closed_form(scene)
        self._wrist_qpos0 = wrist_qpos0(scene)
        self._wrist_body = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, wrist_terminal_body(scene))
        self._phi_fn, self._values_fn = as_phi_fn(sdf), as_values_fn(sdf)
        self._t_star: Tensor | None = None
        self._cached_x: Tensor | None = None
        self._cached: tuple[KinematicsBatch, Tensor, CapsuleDistance] | None = None

    @property
    def limited(self) -> Tensor:
        """`(n,)` 1.0 where the robot dof is range-limited (`robot_joint_limits`)."""
        return self._limited

    def _geometry(self, x: Tensor) -> tuple[KinematicsBatch, Tensor, CapsuleDistance]:
        """Kinematics, canonical points of all 144 points and the cover search at `x` (reused for the same tensor)."""
        if self._cached is not None and self._cached_x is x:
            return self._cached
        cfg, U, P = self._cfg, self._U, self._P
        batch = self._kin.forward(torch.cat([x, self._obj_block], dim=1).numpy())
        canon = canonical_points(self._kin.world_points(batch, self._body_ids, self._local), self._obj_block,
                                 self._scale)
        dist = capsule_min_distance(self._phi_fn, canon[:, U:U + P], canon[:, U + P:U + 2 * P], self._radius,
                                    t_init=self._t_star, n_samples=cfg.n_axis_samples, n_golden=cfg.n_golden,
                                    values_fn=self._values_fn)
        self._cached_x, self._cached = x, (batch, canon, dist)
        return self._cached

    def _wrist_prior(self, x: Tensor, batch: KinematicsBatch, refresh: bool) -> BlockTerm:
        term = self._wrist_prior_term(x, batch, refresh)
        if self._frame_weight is None:
            return term
        return BlockTerm(residual=term.residual, jacobian=term.jacobian,
                         weight=term.weight * self._frame_weight[:, None])

    def _wrist_prior_term(self, x: Tensor, batch: KinematicsBatch, refresh: bool) -> BlockTerm:
        cfg = self._cfg
        if self._closed_form:
            return wrist_prior_term_closed_form(x[:, :6], self._target_wrist, cfg.lam_wrist, cfg.fd_eps_wrist,
                                                need_jacobian=refresh, n_dof=self.n_dof, qpos0=self._wrist_qpos0)
        scale, T = math.sqrt(cfg.lam_wrist), x.shape[0]

        def residual(b: KinematicsBatch) -> Tensor:
            return scale * se3_log(se3_inv(self._kin.body_pose(b, self._wrist_body)) @ self._target_wrist)

        r0 = residual(batch)
        if not refresh:
            return BlockTerm(residual=r0, jacobian=torch.zeros((), dtype=_DT).expand(T, 6, self.n_dof),
                             weight=torch.ones_like(r0))
        q = torch.cat([x, self._obj_block], dim=1).numpy()
        jac = torch.zeros(T, 6, self.n_dof, dtype=_DT)
        for i in range(6):
            qp, qm = q.copy(), q.copy()
            qp[:, i] += cfg.fd_eps_wrist
            qm[:, i] -= cfg.fd_eps_wrist
            jac[:, :, i] = (residual(self._kin.forward(qp)) - residual(self._kin.forward(qm))) / (2.0 * cfg.fd_eps_wrist)
        return BlockTerm(residual=r0, jacobian=jac, weight=torch.ones_like(r0))

    def evaluate(self, x: Tensor, eps: float, lam_pen: float, refresh: bool,
                 plan_state: dict[str, Tensor | None]) -> tuple[list[BlockTerm], SmoothTerm | None]:
        """A refresh (`refresh=True`) recomputes the OT plan and every Jacobian and writes `t*`; a trial reuses
        `plan_state["plan"]`, computes no Jacobian and leaves `t*` alone."""
        cfg, U, P, T = self._cfg, self._U, self._P, x.shape[0]
        batch, canon, dist = self._geometry(x)
        # The transport points' Jacobian only on frames with an in-band atom (the only frames whose
        # transport rows can carry mass); the covers' and tips' Jacobian on every frame. Same per-frame values.
        jac = jac_tr = None
        ne = self._measure.nonempty
        if refresh:
            jac = self._kin.canonical_jacobian(batch, self._body_ids[U:], canon[:, U:], self._obj_block, self._scale)
            if bool(ne.any()):
                jac_tr = self._kin.canonical_jacobian(batch, self._body_ids[:U], canon[ne, :U], self._obj_block[ne],
                                                      self._scale, frames=ne)
            else:
                jac_tr = torch.zeros(0, U, 3, self.n_dof, dtype=_DT)
        margin = cfg.contact_margin / self._scale
        transport, plan = contact_transport_term_inband(
            self._measure, canon[:, :U], jac_tr, self._robot_labels, eps,
            cfg.rho_h_scale * self.diameter, cfg.rho_r_scale * self.diameter, cfg.lambda_label, cfg.n_sinkhorn,
            None if refresh else plan_state.get("plan"), margin, self._sdf, self.n_dof, sinkhorn=cfg.sinkhorn,
            robot_jac_frames=ne if refresh else None)
        plan_state["plan"] = plan
        zero = torch.zeros((), dtype=_DT).expand(T, P, 3, self.n_dof)
        jac_a, jac_b = (jac[:, :P], jac[:, P:2 * P]) if jac is not None else (zero, zero)
        penetration, _ = primitive_penetration_term(canon[:, U:U + P], canon[:, U + P:U + 2 * P], jac_a, jac_b, self._radius,
                                                    self._sdf, lam_pen, cfg.clearance_min / self._scale,
                                                    need_jacobian=refresh, distance=dist)
        if refresh:
            self._t_star = dist.t_star
        band = contact_band_term(dist, jac_a, jac_b, self._covers.body_ids, self._sample_ids, plan.sum(dim=1) * float(U),
                                 cfg.lam_band, margin, need_jacobian=refresh)
        parts = [transport, penetration, joint_limit_term(x, self.q_min, self.q_max, self._limited, cfg.lam_lim),
                 self._wrist_prior(x, batch, refresh), band]
        if self._tip_canon is not None:
            parts.append(self._tip_term(canon, jac, T))
        residual = torch.cat([t.residual for t in parts], dim=1)
        weight = torch.cat([t.weight for t in parts], dim=1)
        jacobian = torch.cat([t.jacobian for t in parts], dim=1) if refresh else \
            torch.zeros((), dtype=_DT).expand(T, residual.shape[1], self.n_dof)
        smooth = smoothness_term(x, self._lam_smooth) if T > 1 else None
        return [BlockTerm(residual=residual, jacobian=jacobian, weight=weight)], smooth

    def _tip_term(self, canon: Tensor, jac: Tensor | None, T: int) -> BlockTerm:
        """`sqrt(lam_tip) * scale * (canonical tip - canonical target)` = the metric fingertip offset (the object frame
        is a rotation, so norms are kept), 15 rows per frame, weighted by `frame_weight` (0 on masked frames)."""
        assert self._tip_canon is not None
        k = 2 * self._P                                   # `jac` covers the points after the U transport samples
        s = math.sqrt(self._cfg.lam_tip) * self._scale
        residual = (s * (canon[:, self._U + k:] - self._tip_canon)).reshape(T, -1)
        jacobian = torch.zeros((), dtype=_DT).expand(T, residual.shape[1], self.n_dof) if jac is None else \
            (s * jac[:, k:]).reshape(T, -1, self.n_dof)
        weight = torch.ones_like(residual) if self._frame_weight is None else \
            self._frame_weight[:, None].expand_as(residual)
        return BlockTerm(residual=residual, jacobian=jacobian, weight=weight)

    def on_final_accept(self, x: Tensor) -> None:
        """L12: the stage ended on an accepted step at `x`; carry its (reused trial) search's `t*` forward."""
        if self._cached is None or self._cached_x is not x:
            raise ValueError("on_final_accept expects the tensor of the last evaluation")
        self._t_star = self._cached[2].t_star


def _frame_mask(frame_valid: Tensor | None, T: int) -> Tensor | None:
    """`frame_valid` as a `(T,)` bool tensor, or None when it is absent or all True (the unmasked path).

    Raises:
        ValueError: If `frame_valid` is not a `(T,)` bool tensor, or marks no frame valid.
    """
    if frame_valid is None:
        return None
    mask = torch.as_tensor(frame_valid)
    if mask.dtype != torch.bool or tuple(mask.shape) != (T,):
        raise ValueError(f"frame_valid must be a ({T},) bool tensor, got {tuple(mask.shape)} {mask.dtype}")
    if not bool(mask.any()):
        raise ValueError("frame_valid marks no frame valid; there is nothing to retarget")
    return None if bool(mask.all()) else mask


def _nearest_valid(mask: Tensor) -> Tensor:
    """`(T,)` int64 index of the nearest frame where `mask` is True (the earlier one on a tie)."""
    idx = torch.nonzero(mask).flatten()
    frames = torch.arange(mask.shape[0])
    after = torch.searchsorted(idx, frames).clamp(max=idx.numel() - 1)
    before = (after - 1).clamp(min=0)
    take_before = (frames - idx[before]).abs() <= (idx[after] - frames).abs()
    return torch.where(take_before, idx[before], idx[after])


def retarget_kinematic_fast(obj_poses: Tensor, wrist_poses: Tensor, scale: float, hand_verts: Tensor,
                            hand_labels: Tensor, scene: RobotScene, sdf: SdfGrid, surface: Tensor,
                            surface_normals: Tensor, frequency: float, cfg: RetargetConfig,
                            frame_valid: Tensor | None = None) -> KinematicTrajectory:
    """`retarget_kinematic`'s contract and legacy schedule on the fast evaluation; `frame_valid (T,)` bool masks the
    hand input of frames where it is False (module docstring)."""
    T = obj_poses.shape[0]
    for name, arr in (("wrist_poses", wrist_poses), ("hand_verts", hand_verts)):
        if arr.shape[0] != T:
            raise ValueError(f"{name} has {arr.shape[0]} frames, expected {T} (obj_poses's count)")
    mask = _frame_mask(frame_valid, T)
    t0 = time.perf_counter()
    if mask is not None:
        nearest = _nearest_valid(mask)
        wrist_poses, hand_verts = wrist_poses[nearest], hand_verts[nearest]
    obj_block = _obj_qpos_block(obj_poses)
    target_wrist = wrist_poses.to(_DT)
    subset = _human_atom_subset(int(hand_labels.shape[0]), cfg.n_human_atoms, cfg.seed)
    human = human_contact_measure(hand_verts[:, subset], obj_poses, scale, hand_labels[subset], sdf, surface,
                                  sigma=cfg.sigma, tau=cfg.tau)
    frame_weight = None if mask is None else mask.to(_DT)
    if frame_weight is not None:
        human = dataclasses.replace(human, weights=human.weights * frame_weight[:, None])
    tip_targets = hand_verts[:, list(MANO_TIP_VERTICES)].to(_DT) if cfg.lam_tip > 0.0 else None
    builder = FastTermBuilder(scene, sdf, surface, surface_normals, human, obj_block, target_wrist, scale, cfg,
                              frame_weight=frame_weight, tip_targets=tip_targets)
    bounds = joint_bounds(builder.q_min, builder.q_max, builder.limited) if cfg.joint_limits == "projected" else None
    eps_schedule = (cfg.eps_hi_scale * builder.diameter, cfg.eps_lo_scale * builder.diameter)
    lam_pen_schedule = _geometric_schedule(cfg.lam_pen_lo, cfg.lam_pen_hi, 2 * len(eps_schedule))

    x = torch.zeros(T, builder.n_dof, dtype=_DT)
    x[:, :6] = _wrist_init_batch(target_wrist)
    x[:, 6:] = (builder.q_min[6:] + builder.q_max[6:]) / 2.0
    stage = 0
    for max_iter in (cfg.warm_start_max_iter, cfg.joint_max_iter):
        for eps in eps_schedule:
            plan_state: dict[str, Tensor | None] = {}

            def evaluate(xx: Tensor, refresh: bool, _eps: float = eps, _lp: float = lam_pen_schedule[stage],
                         _ps: dict[str, Tensor | None] = plan_state) -> tuple[list[BlockTerm], SmoothTerm | None]:
                return builder.evaluate(xx, _eps, _lp, refresh, _ps)

            lm_cfg = BlockLMConfig(max_iter=max_iter, blas_threads=cfg.blas_threads, final_refresh=False)
            x = lm_optimize_blocks(x, evaluate, lm_cfg, on_final_accept=builder.on_final_accept, bounds=bounds).x
            stage += 1
    qpos = torch.cat([x, obj_block], dim=1)
    logger.info("retarget_kinematic (fast path): %.3fs (%d frames, %d masked)", time.perf_counter() - t0, T,
                0 if mask is None else int((~mask).sum()))
    return KinematicTrajectory(qpos=qpos.numpy(), qvel=kinematic_qvel(qpos, frequency).numpy(), frequency=frequency)
