"""Per-frame LM residual terms for `fastdo.retarget.kinematic`: penetration,
joint limits, wrist prior, and inter-frame smoothness -- everything except the contact-transport
term itself (`fastdo.retarget.transport.contact_transport_term`).

The objective's penetration term is `primitive_penetration_term`, a clearance band on every
hand collision primitive's analytic minimum signed distance (`fastdo.retarget.primitives`). The
sampled-point `penetration_term` is kept for diagnostics only; `retarget_kinematic` no longer uses it.

All residual/Jacobian pairs here are over the robot's own `n = RobotScene.robot_dof` width (28 for
the Sharpa Wave scene): the `q (T, 35)` passed in is always the *full* qpos (object block included,
held fixed per frame), matching `fastdo.retarget.contact`'s and `fastdo.retarget.transport`'s own
convention, but every `BlockTerm`/`SmoothTerm` this module returns is sized for the 28-dim state
`fastdo.solver.lm_blocks` optimizes over.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
import torch
from torch import Tensor

from fastdo.geometry.lie import se3_inv, se3_log
from fastdo.retarget.batched_kinematics import wrist_pose_closed_form
from fastdo.retarget.mj_model import RobotScene
from fastdo.retarget.primitives import (
    CapsuleDistance,
    PhiFn,
    as_phi_fn,
    as_values_fn,
    capsule_min_distance,
)
from fastdo.solver.lm_blocks import BlockTerm, SmoothTerm
from fastdo.solver.sdf import SdfGrid, sdf_lookup

__all__ = [
    "contact_band_term",
    "human_wrist_poses",
    "joint_limit_term",
    "penetration_term",
    "primitive_penetration_term",
    "robot_joint_limits",
    "smoothness_term",
    "wrist_prior_term",
    "wrist_prior_term_closed_form",
    "wrist_terminal_body",
]


def wrist_terminal_body(scene: RobotScene) -> str:
    """Name of the wrist-terminal MuJoCo body (`right_hand_C_MC` for the Sharpa scene).

    Derived structurally (the body attached to the last wrist-dof joint), the same technique
    `mj_model._hand_link_body_ids` uses internally -- duplicated here in a few lines rather than
    exposing it as a new `RobotScene` method, to avoid touching the already-reviewed `mj_model`.
    """
    layout = scene.joint_layout()
    last_wrist_dof = layout["wrist_dof"].stop - 1
    model = scene.model
    jid = next(j for j in range(model.njnt) if int(model.jnt_dofadr[j]) == last_wrist_dof)
    body_id = int(model.jnt_bodyid[jid])
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
    if name is None:
        raise ValueError(f"wrist-terminal body (id {body_id}) has no name in the compiled model")
    return name


def penetration_term(canonical_points: Tensor, canonical_jac: Tensor, sdf: SdfGrid, lam_pen: float) -> BlockTerm:
    """`sqrt(lam_pen) * relu(-phi(r_u))` on every robot surface point, all robot dofs.

    Args:
        canonical_points: `(T, U, 3)` canonical-frame robot points, e.g. from
            `fastdo.retarget.transport.robot_canonical_points`.
        canonical_jac: `(T, U, 3, n)` canonical-frame Jacobian of `canonical_points`.
        sdf: Canonical-unit-mesh signed-distance grid.
        lam_pen: Penetration penalty weight.
    """
    dtype = torch.float64
    phi, grad = sdf_lookup(sdf, canonical_points.to(dtype))              # (T,U), (T,U,3)
    active = (phi < 0.0).to(dtype)
    residual = math.sqrt(lam_pen) * torch.relu(-phi)                      # (T,U)
    dphi_dq = torch.einsum("tui,tuik->tuk", grad, canonical_jac.to(dtype))  # (T,U,n)
    jac = -math.sqrt(lam_pen) * active[..., None] * dphi_dq               # (T,U,n)
    return BlockTerm(residual=residual, jacobian=jac, weight=torch.ones_like(residual))


def primitive_penetration_term(end_a: Tensor, end_b: Tensor, jac_a: Tensor, jac_b: Tensor, radius: Tensor,
                               sdf: SdfGrid | PhiFn, lam_pen: float, clearance_min: float,
                               t_init: Tensor | None = None, n_samples: int = 16, n_golden: int = 3,
                               need_jacobian: bool = True, distance: CapsuleDistance | None = None,
                               ) -> tuple[BlockTerm, CapsuleDistance]:
    """`sqrt(lam_pen) * relu(clearance_min - d_min(q))` on every capsule cover, every frame.

    `d_min` is `fastdo.retarget.primitives.capsule_min_distance` (16 axis samples, golden-section
    refinement, warm start from `t_init`). Its Jacobian follows the envelope theorem with t* fixed:
    `d d_min / dq = grad phi(p*) . ((1 - t*) jac_a + t* jac_b)`, zero for an inactive capsule.

    Args:
        end_a, end_b: `(T, P, 3)` canonical-frame axis endpoints (`robot_canonical_points` of the covers'
            body-local endpoints).
        jac_a, jac_b: `(T, P, 3, n)` their canonical-frame Jacobians (ignored if `need_jacobian` is False).
        radius: `(P,)` cover radii, canonical units (metres / mesh_scale).
        sdf: Canonical object SDF grid, or an analytic `PhiFn`.
        lam_pen: Penalty weight.
        clearance_min: Required clearance, canonical units (0.001 m / mesh_scale in `retarget_kinematic`).
        t_init: `(T, P)` warm start, e.g. the previous accepted evaluation's `t_star`.
        need_jacobian: False for a cost-only LM trial evaluation (returns a zero Jacobian).
        distance: An already searched `CapsuleDistance` of these covers; the search is then
            skipped and `t_init`, `n_samples`, `n_golden` are not read.

    Returns:
        `(BlockTerm (T, P) / (T, P, n), CapsuleDistance)`; the caller keeps `.t_star` for the warm start.
    """
    dtype = torch.float64
    dist = distance if distance is not None else capsule_min_distance(
        as_phi_fn(sdf), end_a, end_b, radius, t_init=t_init, n_samples=n_samples, n_golden=n_golden,
        values_fn=as_values_fn(sdf))
    scale = math.sqrt(lam_pen)
    residual = scale * torch.relu(clearance_min - dist.d_min)                       # (T,P)
    active = (dist.d_min < clearance_min).to(dtype)
    jac = -scale * active[..., None] * _distance_jacobian(dist, jac_a, jac_b, need_jacobian)
    return BlockTerm(residual=residual, jacobian=jac, weight=torch.ones_like(residual)), dist


def _distance_jacobian(dist: CapsuleDistance, jac_a: Tensor, jac_b: Tensor, need_jacobian: bool) -> Tensor:
    """`d d_min / dq (T, P, n)` by the envelope theorem (t* and grad phi(p*) fixed); zeros for a trial."""
    dtype = torch.float64
    if not need_jacobian:
        return torch.zeros(*dist.d_min.shape, jac_a.shape[-1], dtype=dtype)
    t = dist.t_star[..., None, None]
    axis_jac = (1.0 - t) * jac_a.to(dtype) + t * jac_b.to(dtype)                      # (T,P,3,n)
    return torch.einsum("tpi,tpik->tpk", dist.grad, axis_jac)


def contact_band_term(dist: CapsuleDistance, jac_a: Tensor, jac_b: Tensor, cover_body_ids: np.ndarray,
                      sample_body_ids: np.ndarray, sample_mass: Tensor, lam_band: float, contact_margin: float,
                      need_jacobian: bool = True) -> BlockTerm:
    """Upper side of the clearance band: `sqrt(lam_band * W_b) * (d_min - contact_margin)`.

    Applied, per frame, to the closest cover of each body `b` (the body's clearance), weighted by the
    transport mass its robot samples carry, `W_b = sum_{u on b} U w_u` (fixed with the OT plan). Why
    it is needed: the transport targets sit `contact_margin` outside the surface, but the robot samples
    are spread over each link, and samples on a link's far side (and barycentres over thin wires) pull
    the link inward. Without this term every contact link settles on the 1 mm floor of
    `primitive_penetration_term` (thumb/index/middle median clearance ~1.0 mm on the whisking demo);
    with it the contact links sit inside the 2-5 mm band.

    Args:
        dist: The penetration term's `CapsuleDistance` (same evaluation, canonical units).
        jac_a, jac_b: `(T, P, 3, n)` canonical Jacobians of the covers' axis endpoints.
        cover_body_ids: `(P,)` body id of each cover; `sample_body_ids`: `(U,)` body id of each robot sample.
        sample_mass: `(T, U)` `U * w_u`, the transported mass per robot sample (plan column sums times U).
        lam_band: Weight; `contact_margin`: target clearance, canonical units.
    """
    dtype = torch.float64
    d = dist.d_min
    same_body = torch.as_tensor(cover_body_ids[:, None] == cover_body_ids[None, :])          # (P,P)
    body_min = torch.where(same_body[None], d[:, None, :], torch.full_like(d[:, None, :], math.inf)).min(dim=2).values
    closest = (d <= body_min).to(dtype)                                                     # (T,P)
    on_body = torch.as_tensor(sample_body_ids[None, :] == cover_body_ids[:, None], dtype=dtype)   # (P,U)
    body_mass = sample_mass.to(dtype) @ on_body.T                                           # (T,P)
    scale = torch.sqrt(lam_band * body_mass.clamp_min(0.0) * closest)
    residual = scale * (d - contact_margin)
    jac = scale[..., None] * _distance_jacobian(dist, jac_a, jac_b, need_jacobian)
    return BlockTerm(residual=residual, jacobian=jac, weight=torch.ones_like(residual))


def robot_joint_limits(model: mujoco.MjModel, robot_dof: slice) -> tuple[Tensor, Tensor, Tensor]:
    """`(q_min, q_max, limited)` (each `(n,)`) for the `n` dofs in `robot_dof`, from `model.jnt_range`.

    `limited` is 1.0 where the joint is actually range-limited (`model.jnt_limited`) and 0.0
    otherwise (an unlimited joint contributes no residual, rather than a spurious one from a
    meaningless `[0, 0]` range).
    """
    n = robot_dof.stop - robot_dof.start
    q_min = np.zeros(n, dtype=np.float64)
    q_max = np.zeros(n, dtype=np.float64)
    limited = np.zeros(n, dtype=np.float64)
    for d in range(robot_dof.start, robot_dof.stop):
        jid = next(j for j in range(model.njnt) if int(model.jnt_dofadr[j]) == d)
        lo, hi = model.jnt_range[jid]
        q_min[d - robot_dof.start] = lo
        q_max[d - robot_dof.start] = hi
        limited[d - robot_dof.start] = 1.0 if model.jnt_limited[jid] else 0.0
    return torch.from_numpy(q_min), torch.from_numpy(q_max), torch.from_numpy(limited)


def joint_limit_term(q_robot: Tensor, q_min: Tensor, q_max: Tensor, limited: Tensor, lam_lim: float) -> BlockTerm:
    """`sqrt(lam_lim) * (relu(q - q_max) + relu(q_min - q))` per robot dof.

    An unlimited dof (`limited[i] == 0`) is excluded via `weight`, not by zeroing the residual
    formula itself: zeroing only the Jacobian while leaving a nonzero residual would bias the cost
    with no gradient to correct it, which is worse than including the dof at all.

    Args:
        q_robot: `(T, n)` the robot-dof state (28 columns of `q`, i.e. the LM's own `x`).
        q_min, q_max, limited: `(n,)`, from `robot_joint_limits`.
    """
    dtype = torch.float64
    q_robot = q_robot.to(dtype)
    over_hi = (q_robot > q_max[None, :]).to(dtype)
    under_lo = (q_robot < q_min[None, :]).to(dtype)
    residual = math.sqrt(lam_lim) * (torch.relu(q_robot - q_max[None, :]) + torch.relu(q_min[None, :] - q_robot))
    diag = math.sqrt(lam_lim) * (over_hi - under_lo)                     # (T,n)
    jac = torch.diag_embed(diag)                                         # (T,n,n)
    weight = limited[None, :].expand_as(residual)
    return BlockTerm(residual=residual, jacobian=jac, weight=weight)


def smoothness_term(q_robot: Tensor, lam_q: float | Tensor) -> SmoothTerm:
    """`sqrt(lam_q) * (q_{k+1} - q_k)`, block-tridiagonal.

    `lam_q` is a scalar or a per-dof `(n,)` vector (the wrist slides are in metres and the
    hinges in radians, so one scalar cannot weight both sensibly)."""
    dtype = torch.float64
    q_robot = q_robot.to(dtype)
    T, n = q_robot.shape
    sqrt_lam = torch.as_tensor(lam_q, dtype=dtype).sqrt().expand(n)
    residual = sqrt_lam * (q_robot[1:] - q_robot[:-1])
    diag = torch.diag(sqrt_lam)
    jac_k = (-diag).expand(T - 1, n, n)
    jac_k1 = diag.expand(T - 1, n, n)
    return SmoothTerm(residual=residual, jac_k=jac_k, jac_k1=jac_k1, weight=torch.ones(T - 1, n, dtype=dtype))


def human_wrist_poses(joints: Tensor, is_right: bool = True) -> Tensor:
    """The baseline's MCP-based geometric wrist frame (`process_dataset.py::compute_wrist_rotation`).

    Args:
        joints: `(T, 21, 3)` MANO/OpenPose-order hand joints (0=wrist, 5=index MCP, 9=middle MCP,
            13=ring MCP), e.g. `fastdo.hands.hawor_npz.HandTrack.joints`, in whatever frame the
            caller's whole retargeting solve uses (this function does not change frames).
        is_right: Selects the index-MCP<->ring-MCP auxiliary axis direction (mirrors the baseline).

    Returns:
        `(T, 4, 4)` SE(3) poses: position = `joints[:, 0, :]` (the wrist joint itself); orientation
        `[x | y | z]` with `z`: middle MCP -> wrist, `y_aux`: index MCP <-> ring MCP, `x = y_aux
        x z`, `y = z x x`.

    Note:
        Deliberately does *not* reproduce the baseline's additional `_wrist_frame_offset` (a MANO-
        rotvec-based calibration to its own robot's `{side}_palm` site convention) -- this is a soft
        prior target for `wrist_prior_term`, not a hard site-matching constraint, and that offset is
        specific to the baseline's own IK site, not derivable from `joints` alone. See
        `fastdo.retarget.kinematic`'s module docstring for the full rationale.
    """
    def _norm(v: Tensor) -> Tensor:
        return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    z = _norm(joints[:, 9, :] - joints[:, 0, :])
    y_src = joints[:, 5, :] - joints[:, 13, :] if is_right else joints[:, 13, :] - joints[:, 5, :]
    y_aux = _norm(y_src)
    x = _norm(torch.cross(y_aux, z, dim=-1))
    y = _norm(torch.cross(z, x, dim=-1))
    R = torch.stack([x, y, z], dim=-1)
    T = joints.shape[0]
    pose = torch.zeros(T, 4, 4, dtype=joints.dtype)
    pose[:, :3, :3] = R
    pose[:, :3, 3] = joints[:, 0, :]
    pose[:, 3, 3] = 1.0
    return pose


def wrist_prior_term(q: Tensor, scene: RobotScene, wrist_body: str, target_poses: Tensor,
                     lam_wrist: float, fd_eps: float = 1e-5, need_jacobian: bool = True) -> BlockTerm:
    """`sqrt(lam_wrist) * Log(T_wrist(q)^-1 T_wrist_H)`, Jacobian by central FD.

    Central finite differences over the 6 wrist columns of `q` only (`scene.fk` gives the exact
    wrist-body pose for any `q` without needing to hand-derive the wrist's intrinsic-Euler chain);
    the other 22 (hand) columns are exactly zero, since the wrist body's pose does not depend on
    downstream hand joint angles.

    Args:
        q: `(T, 35)` full qpos (object block held fixed).
        scene: The `RobotScene` (its `.fk` is the only method used).
        wrist_body: Name of the wrist-terminal MuJoCo body (`right_hand_C_MC` for the Sharpa scene).
        target_poses: `(T, 4, 4)` human wrist target, e.g. from `human_wrist_poses`, in the same
            frame `scene.fk` reports body poses in.
        lam_wrist: Wrist-prior weight.
        fd_eps: Finite-difference step (qpos units: metres for the 3 slide dofs, radians for the 3
            hinge dofs).
        need_jacobian: If `False`, skips the 12 extra `scene.fk` calls the finite difference needs
            and returns a zero Jacobian -- for a trial (cost-only) LM evaluation, where the caller
            never reads `.jacobian` (see `fastdo.solver.lm_blocks.total_block_cost`), this is a
            meaningful chunk of `retarget_kinematic`'s wall time avoided for free.
    """
    dtype = torch.float64
    q_np = q.detach().cpu().numpy().astype(np.float64)
    T, n = q_np.shape[0], 28
    target = target_poses.to(dtype)

    def _residual(qq: np.ndarray) -> Tensor:
        fk = scene.fk(qq)
        T_wrist = torch.from_numpy(np.asarray(fk[wrist_body])).to(dtype)
        return math.sqrt(lam_wrist) * se3_log(se3_inv(T_wrist) @ target)

    r0 = _residual(q_np)
    jac = torch.zeros(T, 6, n, dtype=dtype)
    if not need_jacobian:
        return BlockTerm(residual=r0, jacobian=jac, weight=torch.ones_like(r0))
    for i in range(6):
        qp, qm = q_np.copy(), q_np.copy()
        qp[:, i] += fd_eps
        qm[:, i] -= fd_eps
        jac[:, :, i] = (_residual(qp) - _residual(qm)) / (2.0 * fd_eps)

    return BlockTerm(residual=r0, jacobian=jac, weight=torch.ones_like(r0))


def wrist_prior_term_closed_form(x_wrist: Tensor, target_poses: Tensor, lam_wrist: float, fd_eps: float,
                                 need_jacobian: bool, n_dof: int = 28, *, qpos0: Tensor) -> BlockTerm:
    """`wrist_prior_term` for a closed-form wrist chain (`batched_kinematics.
    wrist_chain_is_closed_form`): the wrist pose is `Trans(d0, d1, d2) Rx(d3) Ry(d4) Rz(d5)`, `d = x_wrist - qpos0`
    with `qpos0 (6,)` the wrist joints' reference offsets (`batched_kinematics.wrist_qpos0`).

    Residual `(T, 6)` = `sqrt(lam_wrist) * se3_log(se3_inv(pose(x_wrist)) @ target_poses)`. The Jacobian `(T, 6, n_dof)`
    is the same central difference with step `fd_eps` over the 6 wrist columns, its 13 configurations `[base, +e0,
    -e0, ..., +e5, -e5]` evaluated in one batched call; columns `6:` are exactly 0. Without `need_jacobian` the
    Jacobian is an expanded zero.
    """
    dtype = torch.float64
    x = x_wrist.to(dtype)
    T = x.shape[0]
    scale = math.sqrt(lam_wrist)
    target = target_poses.to(dtype)
    if not need_jacobian:
        r0 = scale * se3_log(se3_inv(wrist_pose_closed_form(x, qpos0)) @ target)
        return BlockTerm(residual=r0, jacobian=torch.zeros((), dtype=dtype).expand(T, 6, n_dof),
                         weight=torch.ones_like(r0))
    steps = torch.zeros(13, 6, dtype=dtype)
    for i in range(6):
        steps[1 + 2 * i, i] = fd_eps
        steps[2 + 2 * i, i] = -fd_eps
    configs = (x[:, None, :] + steps[None]).reshape(T * 13, 6)
    r = scale * se3_log(se3_inv(wrist_pose_closed_form(configs, qpos0)) @ target.repeat_interleave(13, dim=0))
    r = r.reshape(T, 13, 6)
    jac = torch.zeros(T, 6, n_dof, dtype=dtype)
    jac[:, :, :6] = ((r[:, 1::2] - r[:, 2::2]) / (2.0 * fd_eps)).transpose(1, 2)
    return BlockTerm(residual=r[:, 0], jacobian=jac, weight=torch.ones_like(r[:, 0]))
