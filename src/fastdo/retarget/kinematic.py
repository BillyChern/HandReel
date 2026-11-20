"""Contact-transport kinematic retargeting: `retarget_kinematic` and its I/O format.

Orchestrates `fastdo.retarget.transport.contact_transport_term` plus the penetration/joint-limit/
wrist-prior/smoothness terms in `fastdo.retarget.kinematic_terms` into one schedule:
initialize q from the wrist-prior target (hand dofs mid-range), a batched warm-start
sweep over all frames (no inter-frame coupling yet), then a joint block-tridiagonal LM over the
whole trajectory (`fastdo.solver.lm_blocks`) with the OT plan refreshed on every accepted step,
annealed over 2 eps levels.

Frame convention (see `fastdo.retarget.baseline_io`'s module docstring for the full derivation,
including a measured correction to the original plan):
`obj_poses`, `wrist_poses`, and `hand_verts` are all expressed in a **single**, shared frame -- the
baseline's own MuJoCo world frame, sourced from `fastdo.retarget.baseline_io.baseline_world_inputs`
(`qpos_obj_right`/`qpos_wrist_right`/`mano_verts_right` in `trajectory_keypoints.npz`, all mutually
consistent by construction: the baseline's own `process_dataset.py` shifts all three by the identical
per-clip `world_offset`). This both keeps `human_contact_measure`'s human/robot correspondence sound
(it needs `hand_verts`/`poses` in the same frame as each other) and places the robot's `q[28:35]`
correctly for the baseline's `optimize_physics.py`, which operates on the same compiled `scene.xml`.
(An earlier version of this module paired `hand_verts`/contact-measure poses from a *different*
camera-frame reconstruction with `qpos_obj_right` for `q[28:35]` -- measured to cause a severe
regression, since those two object-pose sources do not, in fact, share a frame; see
`fastdo.retarget.baseline_io`'s docstring for the evidence.)

Clearance band: the reference must be physics-feasible, so the objective's penetration
term is `kinematic_terms.primitive_penetration_term` -- `relu(clearance_min - d_min)` on the analytic
minimum signed distance of every hand collision primitive's capsule cover (`fastdo.retarget.
primitives`), with `clearance_min` = 1 mm, warm-started axis parameters, and `lam_pen` still annealed
low -> high (1e2 -> 1e6) -- plus `kinematic_terms.contact_band_term`, which pulls each contact link's
closest cover to `contact_margin` = 3 mm, where the transport targets also sit. Measured on the
whisking demo: without the band term the transport's inward pull leaves every contact link on the
1 mm floor; with it thumb/index/middle median clearances land in 2-5 mm. The longer schedule (10
warm-start rounds, 25 joint iterations) is what clears the last penetrating frames. The sampled-point
penetration term is no longer part of the objective (kept in `kinematic_terms` as a diagnostic).

The reference must also be trackable in time. With the earlier weights (lam_q = lam_wrist = 1) the stronger
clearance terms made it jitter: median finger step 0.39 rad per frame (shipped reference: 0.015),
wrist translation steps 50 mm (p95; shipped: 18 mm). `lam_q` = 1e3 on the hinges, `lam_q_slide` = 3e4
on the wrist slides (metres: a 1 cm step costs like a 0.06 rad one) and `lam_wrist` = 1e2 give finger
steps 0.018 / 0.042 rad (median / p95), wrist steps 0.076 rad / 35 mm (p95) and 0.056 rad median
distance from the wrist prior, with the clearance band intact.

One remaining deliberate simplification: `wrist_prior_term`'s target orientation (taken directly from
`wrist_poses`, i.e. the baseline's own IK site convention) is used as-is, rather than re-deriving
`fastdo.retarget.kinematic_terms.human_wrist_poses`'s MCP-based geometric frame from raw hand joints
-- that function remains available (and tested) for a future non-demo path that has no pre-existing
`trajectory_keypoints.npz` to read a wrist target from.

`RetargetConfig.fast_path` (the default) runs `fastdo.retarget.kinematic_fast.retarget_kinematic_fast`,
the same objective, schedule and warm starts evaluated with exact levers only (qpos parity with this module's path
within 1e-12 on the first 20 whisking frames, `tests/retarget/test_kinematic_fast.py`); `fast_path=False` runs the
legacy body below, kept as the parity reference.

A `hands` argument (`retarget_kinematic(obj_poses, scale, hands, scene, sdf, cfg)`) is replaced with explicit
tensors (`hand_verts`, `hand_labels`, plus the pose arrays above):
`fastdo.io.clip_record.HandTrack` (what `fastdo.hands.hawor_npz.load_hawor_npz` returns) has no
vertex field at all, and `human_contact_measure` needs full MANO vertices (778 per frame).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch import Tensor

from fastdo.retarget.contact import ContactMeasure, human_contact_measure
from fastdo.retarget.kinematic_terms import (
    contact_band_term,
    joint_limit_term,
    primitive_penetration_term,
    robot_joint_limits,
    smoothness_term,
    wrist_prior_term,
    wrist_terminal_body,
)
from fastdo.retarget.labels import robot_link_labels
from fastdo.retarget.mj_model import RobotScene
from fastdo.retarget.primitives import hand_capsule_covers
from fastdo.retarget.qvel import kinematic_qvel
from fastdo.retarget.transport import (
    canonical_diameter,
    contact_transport_term,
    robot_canonical_points,
)
from fastdo.solver.lm_blocks import BlockLMConfig, BlockTerm, SmoothTerm, lm_optimize_blocks
from fastdo.solver.sdf import SdfGrid

logger = logging.getLogger(__name__)

__all__ = ["JOINT_LIMIT_MODES", "MANO_TIP_VERTICES", "SINKHORN_MODES", "KinematicTrajectory", "RetargetConfig", "joint_bounds",
           "retarget_kinematic"]

JOINT_LIMIT_MODES = ("penalty", "projected")
SINKHORN_MODES = ("log", "scaling")
# MANO vertex ids of the thumb, index, middle, ring and pinky tips: the hands stage's joints 4/8/12/16/20 are exactly
# these vertices (checked on HaWoR's hands.npz, both sides), the targets DO AS I DO's IK tracks.
MANO_TIP_VERTICES = (744, 320, 443, 554, 671)


def joint_bounds(q_min: Tensor, q_max: Tensor, limited: Tensor) -> tuple[Tensor, Tensor]:
    """`(lower, upper)` `(n,)` box of the limited robot dofs for `lm_optimize_blocks(bounds=...)`; -inf/inf elsewhere."""
    on = limited > 0.5
    inf = torch.full_like(q_min, float("inf"))
    return torch.where(on, q_min, -inf), torch.where(on, q_max, inf)


@dataclass(frozen=True)
class RetargetConfig:
    """Weights and schedule for `retarget_kinematic`. Rho/eps are scaled by canonical-diameter^2.

    Lengths (`contact_margin`, `clearance_min`, `box_face_tol`) are metres; `n_axis_samples`/`n_golden`
    set the capsule-axis search of the primitive penetration term (16 and 3).

    `fast_path` selects `fastdo.retarget.kinematic_fast` (same objective and schedule, parity with the
    legacy path); `blas_threads` limits BLAS threads around the fast path's banded solve (None: no limit). The
    legacy path (`fast_path=False`) ignores `blas_threads`.

    Joint-limit and fingertip options (both off by default, so the default solve is unchanged):
    - `joint_limits`: `penalty` (the soft `joint_limit_term` alone, measured violating the Sharpa ranges on
      53% of DexYCB frames by up to 0.34 rad) or `projected` (the joint block LM runs as a projected LM on the box of
      the limited robot dofs' `jnt_range`, `lm_optimize_blocks(bounds=...)`: every iterate is feasible, and a joint
      may sit exactly AT its limit, e.g. a straight finger at PIP = 0, which a barrier would push away from).
    - `lam_tip` (1/m^2; 0 = off): the fingertip keypoint prior `sqrt(lam_tip) * (p_tip(q) - p_human_tip)` on the
      robot's five `<side>_<finger>_tip` sites against the MANO fingertip vertices `MANO_TIP_VERTICES` (the hands
      stage's joints 4/8/12/16/20 exactly), on valid frames only. Without it a finger that carries no transported
      contact mass is held only by smoothness near its mid-range initialization (diagnosis: our robot fingers
      touch the object where the human's do not). Fast path only.

    `sinkhorn` = `log` (default, the log-domain iteration) or `scaling` (the same `n_sinkhorn` iterations in
    the stabilized scaling domain, `fastdo.solver.ot.sinkhorn_unbalanced_uniform_b_scaling`: the transport plan
    matches the log domain to ~1e-10 relative, but it is not bit-identical, hence opt-in; the LM accept/reject path
    can amplify the round-off to qpos differences up to ~1.5e-6 (masked windows; 3.4e-10 on another 6
    clips, 1.8e-8 on a full Ego4D clip under the production config). Fast path only."""

    n_per_link: int = 6
    n_human_atoms: int = 64
    seed: int = 0
    sigma: float = 0.004
    tau: float = 0.008
    lambda_label: float = 0.05
    rho_h_scale: float = 1.0
    rho_r_scale: float = 1e-2
    eps_hi_scale: float = 5e-2
    eps_lo_scale: float = 5e-3
    n_sinkhorn: int = 25
    lam_pen_lo: float = 1e2
    lam_pen_hi: float = 1e6
    contact_margin: float = 0.003
    clearance_min: float = 0.001
    n_axis_samples: int = 16
    n_golden: int = 3
    box_face_tol: float = 5e-4
    lam_band: float = 1e3
    lam_lim: float = 1e3
    lam_wrist: float = 1e2
    lam_q: float = 1e3
    lam_q_slide: float = 3e4
    fd_eps_wrist: float = 1e-5
    warm_start_max_iter: int = 10
    joint_max_iter: int = 25
    fast_path: bool = True
    blas_threads: int | None = 1
    joint_limits: str = "penalty"
    lam_tip: float = 0.0
    sinkhorn: str = "log"

    def __post_init__(self) -> None:
        if self.blas_threads is not None and self.blas_threads < 1:
            raise ValueError("blas_threads must be None or >= 1")
        if self.joint_limits not in JOINT_LIMIT_MODES:
            raise ValueError(f"joint_limits must be one of {JOINT_LIMIT_MODES}, got {self.joint_limits!r}")
        if not (np.isfinite(self.lam_tip) and self.lam_tip >= 0.0):
            raise ValueError(f"lam_tip must be finite and >= 0, got {self.lam_tip}")
        if self.sinkhorn not in SINKHORN_MODES:
            raise ValueError(f"sinkhorn must be one of {SINKHORN_MODES}, got {self.sinkhorn!r}")


@dataclass(frozen=True)
class KinematicTrajectory:
    """`qpos (T,35)`, `qvel (T,34)`, `frequency` -- the baseline's `trajectory_kinematic.npz` format."""

    qpos: np.ndarray
    qvel: np.ndarray
    frequency: float

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, qpos=self.qpos.astype(np.float64), qvel=self.qvel.astype(np.float64),
                 frequency=np.asarray(self.frequency, dtype=np.float64))


def _obj_qpos_block(obj_poses: Tensor) -> Tensor:
    """`(T,4,4)` object poses -> `(T,7)` qpos block `[pos(3), quat_wxyz(4)]`."""
    dtype = torch.float64
    pos = obj_poses[:, :3, 3].to(dtype)
    rot = Rotation.from_matrix(obj_poses[:, :3, :3].detach().cpu().numpy().astype(np.float64))
    quat_xyzw = rot.as_quat()
    quat_wxyz = torch.from_numpy(quat_xyzw[:, [3, 0, 1, 2]]).to(dtype)
    return torch.cat([pos, quat_wxyz], dim=1)


def _wrist_init_batch(target_poses: Tensor) -> Tensor:
    """`(T,4,4)` wrist targets -> `(T,6)` qpos `[pos_x,pos_y,pos_z, rot_x,rot_y,rot_z]` per frame
    (intrinsic XYZ Euler, scipy's uppercase "XYZ": the hinge chain rot_x -> rot_y -> rot_z composes as
    R = Rx Ry Rz) -- batched so every frame gets its own wrist init, not just frame 0. This fixed
    an earlier version, which used extrinsic "xyz" and started 1.48 rad (median) off the prior target."""
    pos = target_poses[:, :3, 3].detach().cpu().numpy().astype(np.float64)
    euler = Rotation.from_matrix(target_poses[:, :3, :3].detach().cpu().numpy().astype(np.float64)).as_euler("XYZ")
    return torch.from_numpy(np.concatenate([pos, euler], axis=1))


def _geometric_schedule(lo: float, hi: float, n: int) -> list[float]:
    """`n` values geometrically interpolated from `lo` to `hi` (inclusive), e.g. for `lam_pen`
    annealing "contact first, cleanup second" across `n` outer-loop stages."""
    if n <= 1:
        return [hi]
    ratio = (hi / lo) ** (1.0 / (n - 1))
    return [lo * ratio ** i for i in range(n)]


def _slice_measure(human: ContactMeasure, frame_slice: slice) -> ContactMeasure:
    return ContactMeasure(points=human.points[frame_slice], raw_points=human.raw_points[frame_slice],
                          weights=human.weights[frame_slice], labels=human.labels,
                          sample_idx=human.sample_idx[frame_slice], sdf_values=human.sdf_values[frame_slice])


def _human_atom_subset(n_vertices: int, n_atoms: int, seed: int) -> Tensor:
    """Sorted int64 indices of the fixed `n_atoms` MANO-vertex subset (`arange` when `n_atoms >= n_vertices`)."""
    if n_atoms >= n_vertices:
        return torch.arange(n_vertices, dtype=torch.int64)
    idx = torch.randperm(n_vertices, generator=torch.Generator().manual_seed(seed))[:n_atoms]
    return idx.sort().values


def _subsample_human_measure(human: ContactMeasure, n_atoms: int, seed: int) -> ContactMeasure:
    """A fixed, reproducible random subset of `n_atoms` MANO vertices (out of 778).

    `sinkhorn_unbalanced`'s cost scales with `V * U` (`V` = human atoms, `U` = robot points) --
    most of the 778 MANO vertices carry zero band weight every frame (only the ones near the
    contact band ever do), so solving the full 778-atom OT problem is mostly wasted work.
    Profiling on the whisking demo showed `sinkhorn_unbalanced` dominating wall time; subsampling
    atoms once (a fixed vertex subset, same across all `T` frames) trades a small amount of
    contact-band resolution for a large, direct reduction in that cost. A no-op when `n_atoms >= V`.
    """
    v_total = human.labels.shape[0]
    if n_atoms >= v_total:
        return human
    idx = _human_atom_subset(v_total, n_atoms, seed)
    return ContactMeasure(points=human.points[:, idx], raw_points=human.raw_points[:, idx],
                          weights=human.weights[:, idx], labels=human.labels[idx],
                          sample_idx=human.sample_idx[:, idx], sdf_values=human.sdf_values[:, idx])


def retarget_kinematic(obj_poses: Tensor, wrist_poses: Tensor, scale: float, hand_verts: Tensor,
                       hand_labels: Tensor, scene: RobotScene, sdf: SdfGrid, surface: Tensor,
                       surface_normals: Tensor, frequency: float,
                       cfg: RetargetConfig, frame_valid: Tensor | None = None) -> KinematicTrajectory:
    """Contact-transport kinematic retargeting (single-world-frame inputs, outward contact margin; see the module
    docstring).

    Args:
        obj_poses: `(T,4,4)` object poses in the MuJoCo world frame (e.g.
            `fastdo.retarget.baseline_io.baseline_world_inputs`'s first return value) -- fixes
            `q[28:35]` every frame and feeds `human_contact_measure`.
        wrist_poses: `(T,4,4)` wrist-prior target, same frame as `obj_poses` (e.g.
            `baseline_world_inputs`'s second return value).
        scale: Metres-per-canonical-unit object scale (`mesh_scale`).
        hand_verts: `(T,778,3)` MANO vertices, same frame as `obj_poses` (e.g.
            `baseline_world_inputs`'s third return value -- `mano_verts_right`, *not*
            `all_hand_meshes.npz`'s raw vertices, which are in a different frame).
        hand_labels: `(778,)` int8 finger label per MANO vertex (`labels.mano_finger_labels`).
        scene: The robot `RobotScene`.
        sdf: Canonical-unit-mesh signed-distance grid.
        surface: `(M,3)` canonical object surface samples.
        surface_normals: `(M,3)` canonical-frame unit normals aligned with `surface` (e.g. from the
            same `trimesh.sample.sample_surface` call's face-index output via `mesh.face_normals`).
            Feeds `contact_transport_term`'s `contact_margin` offset.
        frequency: The clip's frequency in Hz (e.g. `baseline_world_inputs`'s fourth return value) --
            required, no default: silently assuming a rate the shipped data
            was not sampled at would produce a wrong `qvel` with no error to catch it.
        cfg: Weights and schedule.
        frame_valid: `(T,)` bool; False marks a frame whose hand input is not evidence: its hand
            vertices create no contact and its wrist target leaves the cost (`fastdo.retarget.kinematic_fast`'s
            module docstring). None: every frame valid. Fast path only.

    Returns:
        `KinematicTrajectory(qpos (T,35), qvel (T,34), frequency)`.
    """
    dtype = torch.float64
    T = obj_poses.shape[0]
    for name, arr in (("wrist_poses", wrist_poses), ("hand_verts", hand_verts)):
        if arr.shape[0] != T:
            raise ValueError(f"{name} has {arr.shape[0]} frames, expected {T} (obj_poses's count)")
    if cfg.fast_path:
        from fastdo.retarget.kinematic_fast import retarget_kinematic_fast

        return retarget_kinematic_fast(obj_poses, wrist_poses, scale, hand_verts, hand_labels, scene, sdf, surface,
                                       surface_normals, frequency, cfg, frame_valid=frame_valid)
    if frame_valid is not None and not bool(torch.as_tensor(frame_valid).all()):
        raise NotImplementedError("frame_valid masking runs on the fast path only (RetargetConfig.fast_path=True)")
    if cfg.lam_tip > 0.0:
        raise NotImplementedError("the fingertip prior (lam_tip > 0) runs on the fast path only")
    if cfg.sinkhorn != "log":
        raise NotImplementedError("the scaling-domain Sinkhorn (sinkhorn='scaling') runs on the fast path only")

    obj_block = _obj_qpos_block(obj_poses)                                        # (T,7), world frame
    target_wrist = wrist_poses.to(dtype)                                          # (T,4,4), world frame

    human = human_contact_measure(hand_verts, obj_poses, scale, hand_labels, sdf, surface,
                                  sigma=cfg.sigma, tau=cfg.tau)
    human = _subsample_human_measure(human, cfg.n_human_atoms, cfg.seed)

    local_pts, body_ids, _ = scene.surface_points(cfg.n_per_link, seed=cfg.seed)
    covers = hand_capsule_covers(scene, cfg.box_face_tol)
    cover_ids = np.concatenate([covers.body_ids, covers.body_ids])
    cover_ends = np.concatenate([covers.local_a, covers.local_b])                  # (2P,3), body-local
    cover_radius = torch.from_numpy(covers.radius) / scale                        # canonical units
    axis_state: dict[str, Tensor | None] = {"t_star": None}                       # warm start across iterations
    robot_labels = robot_link_labels(scene.model, body_ids)
    q_min, q_max, limited = robot_joint_limits(scene.model, scene.robot_dof)
    wrist_body = wrist_terminal_body(scene)
    lam_smooth = torch.full((scene.robot_dof.stop - scene.robot_dof.start,), cfg.lam_q, dtype=dtype)
    slides = scene.layout["wrist_dof"].start - scene.robot_dof.start                  # pos_x/y/z (metres)
    lam_smooth[slides:slides + 3] = cfg.lam_q_slide
    d2 = canonical_diameter(surface.to(dtype))

    def _canonical_fk(q: Tensor, q_np: np.ndarray, ids: np.ndarray, pts: np.ndarray, need_jacobian: bool,
                      ) -> tuple[Tensor, Tensor]:
        world_np, jac_np = scene.point_kinematics(q_np, ids, pts, jacobian=need_jacobian)
        world_jac = torch.from_numpy(jac_np) if jac_np is not None else torch.zeros(*world_np.shape, q.shape[1] - 7)
        return robot_canonical_points(q, torch.from_numpy(world_np), world_jac, scale)

    def build_terms(x: Tensor, frame_slice: slice, eps: float, lam_pen: float, refresh: bool,
                    plan_state: dict[str, Tensor | None], include_smoothness: bool,
                    ) -> tuple[list[BlockTerm], SmoothTerm | None]:
        # `refresh` doubles as "this is a full (post-accept/initial) evaluation, not a cost-only
        # trial" -- `fastdo.solver.lm_blocks.lm_optimize_blocks` only ever reads `.jacobian` from a
        # `refresh=True` evaluation (see `assemble_block_system`'s call site), so a `refresh=False`
        # (trial) call skips every expensive Jacobian computation below: both `point_kinematics`
        # calls' per-point `mj_jac` loops and `wrist_prior_term`'s 12 extra `scene.fk` calls, all
        # otherwise wasted on a residual-only cost check that gets thrown away on rejection.
        need_jacobian = refresh
        q = torch.cat([x, obj_block[frame_slice]], dim=1)
        q_np = q.detach().cpu().numpy()

        canon_pts, canon_jac = _canonical_fk(q, q_np, body_ids, local_pts, need_jacobian)
        transport, plan = contact_transport_term(
            _slice_measure(human, frame_slice), canon_pts, canon_jac, robot_labels,
            eps=eps, rho_h=cfg.rho_h_scale * d2, rho_r=cfg.rho_r_scale * d2, lambda_label=cfg.lambda_label,
            n_sinkhorn=cfg.n_sinkhorn, plan=None if refresh else plan_state.get("plan"),
            contact_margin=cfg.contact_margin / scale, surface_normals=surface_normals, sdf=sdf,
        )
        plan_state["plan"] = plan

        # Clearance band on every hand primitive (capsule covers), not on sampled points.
        # The axis search warm-starts from the last accepted evaluation's t*.
        ends, ends_jac = _canonical_fk(q, q_np, cover_ids, cover_ends, need_jacobian)
        P = covers.n
        penetration, dist = primitive_penetration_term(
            ends[:, :P], ends[:, P:], ends_jac[:, :P], ends_jac[:, P:], cover_radius, sdf, lam_pen,
            cfg.clearance_min / scale, t_init=axis_state["t_star"], n_samples=cfg.n_axis_samples,
            n_golden=cfg.n_golden, need_jacobian=need_jacobian)
        if refresh:
            axis_state["t_star"] = dist.t_star
        # Upper side of the band: each contact link's closest cover is pulled to contact_margin.
        sample_mass = plan.sum(dim=1) * float(body_ids.shape[0])                     # (T,U) = U * w_u
        band = contact_band_term(dist, ends_jac[:, :P], ends_jac[:, P:], covers.body_ids, body_ids, sample_mass,
                                 cfg.lam_band, cfg.contact_margin / scale, need_jacobian=need_jacobian)

        terms = [
            transport,
            penetration,
            joint_limit_term(x, q_min, q_max, limited, cfg.lam_lim),
            wrist_prior_term(q, scene, wrist_body, target_wrist[frame_slice], cfg.lam_wrist, cfg.fd_eps_wrist,
                             need_jacobian=need_jacobian),
            band,
        ]
        smooth = smoothness_term(x, lam_smooth) if include_smoothness and x.shape[0] > 1 else None
        return terms, smooth

    eps_schedule = (cfg.eps_hi_scale * d2, cfg.eps_lo_scale * d2)
    # 4 stages total (2 in the warm-start sweep, 2 in the joint pass): lam_pen anneals low -> high
    # across the *whole* schedule, "contact first, cleanup second".
    lam_pen_schedule = _geometric_schedule(cfg.lam_pen_lo, cfg.lam_pen_hi, 2 * len(eps_schedule))

    x = torch.zeros(T, 28, dtype=dtype)
    x[:, :6] = _wrist_init_batch(target_wrist)
    x[:, 6:] = (q_min[6:] + q_max[6:]) / 2.0
    bounds = joint_bounds(q_min, q_max, limited) if cfg.joint_limits == "projected" else None

    # Batched warm-start sweep: every round evaluates *all* T frames at once
    # (one (T,V,U) Sinkhorn call, not T separate (1,V,U) calls) with no inter-frame coupling yet
    # (block-diagonal only), each round warm-started from the previous round's accepted state --
    # replaces the original per-frame-sequential-Python-loop pass, which profiling showed paid T
    # times over for `sinkhorn_unbalanced`'s per-call overhead.
    t_warm_start = time.perf_counter()
    for stage, eps in enumerate(eps_schedule):
        lam_pen = lam_pen_schedule[stage]
        plan_state: dict[str, Tensor | None] = {}

        def evaluate(xx: Tensor, refresh: bool, _eps: float = eps, _lp: float = lam_pen,
                    _ps: dict[str, Tensor | None] = plan_state) -> tuple[list[BlockTerm], SmoothTerm | None]:
            return build_terms(xx, slice(0, T), _eps, _lp, refresh, _ps, include_smoothness=True)

        res = lm_optimize_blocks(x, evaluate, BlockLMConfig(max_iter=cfg.warm_start_max_iter), bounds=bounds)
        x = res.x
    t_warm_end = time.perf_counter()
    logger.info("batched warm-start sweep: %.3fs (%d frames, %d eps levels x %d rounds)",
               t_warm_end - t_warm_start, T, len(eps_schedule), cfg.warm_start_max_iter)

    t_joint_start = time.perf_counter()
    for i, eps in enumerate(eps_schedule):
        lam_pen = lam_pen_schedule[len(eps_schedule) + i]
        joint_plan_state: dict[str, Tensor | None] = {}

        def evaluate_joint(xx: Tensor, refresh: bool, _eps: float = eps, _lp: float = lam_pen,
                          _ps: dict[str, Tensor | None] = joint_plan_state,
                          ) -> tuple[list[BlockTerm], SmoothTerm | None]:
            return build_terms(xx, slice(0, T), _eps, _lp, refresh, _ps, include_smoothness=True)

        res = lm_optimize_blocks(x, evaluate_joint, BlockLMConfig(max_iter=cfg.joint_max_iter), bounds=bounds)
        x = res.x
    t_joint_end = time.perf_counter()
    logger.info("joint block LM pass: %.3fs (%d frames, %d dofs)", t_joint_end - t_joint_start, T, x.shape[1])
    logger.info("retarget_kinematic total: %.3fs", t_joint_end - t_warm_start)

    qpos = torch.cat([x, obj_block], dim=1)                                    # (T,35)
    return KinematicTrajectory(qpos=qpos.numpy(), qvel=kinematic_qvel(qpos, frequency).numpy(), frequency=frequency)
