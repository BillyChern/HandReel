"""Stages 1-3 of the kinematic robot path (`fastdo.robot.kinematic_path`, whose docstring describes them): the
kept rows of a reconstructed clip, the wrist targets, the retargeted hand and the placement-free `ReachTask` the
placement is searched on (`fastdo.robot.reach`). Also the path's config."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import trimesh

from fastdo.qc.export import retarget_row_valid
from fastdo.robot.approach import ApproachConfig, ApproachError, HandClearance, plan_approach
from fastdo.robot.arm_ik import ArmIkConfig
from fastdo.robot.dex3_retarget import (
    Dex3DexPilot,
    DexPilotConfig,
    dex3_tip_positions,
    dexpilot_vectors,
    mano_keypoints,
)
from fastdo.robot.g1_model import G1Upper
from fastdo.robot.grasp_closure import ClosureConfig
from fastdo.robot.kinematic_rows import (
    contact_anchor,
    hand_object_distance,
    spike_rows,
    static_window,
    task_span,
    wrist_targets_world,
)
from fastdo.robot.reach import ReachSearchConfig, ReachTask
from fastdo.robot.reach_eval import MarginConfig
from fastdo.robot.replay import SceneConfig, SuccessCriteria
from fastdo.robot.rest import stable_ups
from fastdo.robot.start_pose import (
    SidecarError,
    StartPoseOverride,
    SupportPlaneGravity,
    check_correction,
    file_sha256,
)
from fastdo.robot.symmetry import SymmetryAxis, rotational_symmetries
from fastdo.robot.trajectory import ExportConfig, TrajectoryLimits
from fastdo.robot.transfer import TABLE_TOL_M, TransferConfig
from fastdo.robot.world_frame import Reconstruction

__all__ = ["KinematicPathConfig", "PreparedTask", "prepare_task"]


@dataclass(frozen=True)
class KinematicPathConfig:
    table_z: float = 0.80                                # explicit placement only (the search picks its own)
    table_xy: tuple[float, float] | None = None          # None: placement search (stage 4, `fastdo.robot.reach`)
    yaw_offset: float = 0.0                              # rad, explicit placement only
    sym_axis: int | None = None                          # explicit placement only: grasp rotation about this
    sym_deg: float = 0.0                                 #   principal axis (`fastdo.robot.symmetry`)
    collision_d_safe: float = 0.01
    collision_weight: float = 10.0
    spike_m: float = 0.10
    spike_window: int = 5
    max_wrist_speed: float = 2.5
    contact_m: float = 0.015
    contact_vertex_stride: int = 4
    camera_pitch_deg: float | None = None                # None: up from the record's gravity_cam (or searched)
    first_row: int = 0                                   # rows before it are never used (misregistered start)
    trim_to_contact: bool = True                         # keep the human contact span +- lead / lag (stage 1b)
    contact_lead_s: float = 1.0
    contact_lag_s: float = 1.0
    approach_m: float = 0.08                             # vertical approach / retreat height (0: the human's path)
    approach_mode: str = "vertical"                      # "palm": the synthesised approach before the
                                                         #   first contact (`fastdo.robot.approach`); the retreat
                                                         #   after the last contact stays vertical
    approach: ApproachConfig = field(default_factory=ApproachConfig)
    grasp_closure: bool = False                          # `fastdo.robot.grasp_closure` (stage 6b)
    closure: ClosureConfig = field(default_factory=ClosureConfig)
    rest_s: float = 1.0                                  # contact onset this long after the first used row = the
                                                         #   object rests before it is touched (`ReachTask.rest_check`)
    static_object_m: float = 0.02                        # the object counts as static (pre-contact freeze, approach
    static_object_deg: float = 10.0                      #   / retreat rows) within this motion (`static_window`)
    use_symmetry: bool = True
    use_stable_ups: bool = True                          # resting-face up candidates (`fastdo.robot.rest`)
    up_face: int | None = None                           # explicit placement only: index into the stable ups
    symmetry_tol_m: float = 0.003
    table_margin: float = 0.02                           # object footprint must stay this far inside the table
    search: ReachSearchConfig = field(default_factory=ReachSearchConfig)
    dexpilot: DexPilotConfig = field(default_factory=DexPilotConfig)
    ik: ArmIkConfig = field(default_factory=ArmIkConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    limits: TrajectoryLimits = field(default_factory=TrajectoryLimits)
    scene: SceneConfig = field(default_factory=SceneConfig)
    criteria: SuccessCriteria = field(default_factory=SuccessCriteria)
    margin: MarginConfig = field(default_factory=MarginConfig)
    transfer: TransferConfig = field(default_factory=TransferConfig)   # stage 8: moves to the hold pose and home
    start_pose: StartPoseOverride | None = None          # start-pose sidecar (`fastdo.robot.start_pose`); None: c0 row
    support_plane: SupportPlaneGravity | None = None     # support_plane.json; accepted -> its gravity, no pitch search


def _to_world(c2w: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return np.einsum("tij,tvj->tvi", c2w[:, :3, :3], pts) + c2w[:, None, :3, 3]



@dataclass(frozen=True)
class PreparedTask:
    """Stages 1-3 of the module docstring for one clip: the kept rows and the placement-free task."""
    task: ReachTask
    row_range: tuple[int, int]       # first / last kept CLIP row
    spikes: np.ndarray
    ho_dist: np.ndarray
    contact: int | None
    hand_err: np.ndarray
    n_rows: int
    source_rows: np.ndarray          # (T,) clip row of each task row, -1 for a synthetic approach / retreat row
    object_motion: dict              # `StaticWindow.to_dict()`: measured object motion over the frozen rows
    notes: tuple[str, ...] = ()      # warnings (dropped rows and the displacement that made them dropped)
    start_pose: dict | None = None   # the applied start-pose sidecar and its correction (`_apply_start_pose`)
    gravity: dict | None = None      # where up came from (`_gravity`)
    gravity_source: str | None = None  # "support_plane" when an accepted plane gave up (else the record's)
    approach: dict | None = None     # the synthesised approach (`approach_mode` "palm")


def _check_sidecars(recon: Reconstruction, cfg: KinematicPathConfig) -> None:
    """The sidecars must belong to this window, its world frame and each other (`fastdo.robot.start_pose`); raises
    `SidecarError`."""
    sp, pl, name = cfg.start_pose, cfg.support_plane, recon.source_dir.name
    ego = recon.ego
    if pl is not None:
        if pl.window != name:
            raise SidecarError(f"{pl.path} is the support plane of window {pl.window}, not of {name}")
        plane_ego = pl.camera == "ego_motion"
        if not (isinstance(pl.camera, str) and (plane_ego or pl.camera.startswith("static"))) \
                or plane_ego != (ego is not None):
            raise SidecarError(f"{pl.path} was fitted in the camera world {pl.camera!r}, but the export's world is "
                               f"{'the ego-motion' if ego is not None else 'the static camera'}")
        if pl.accepted and (cfg.camera_pitch_deg is not None or cfg.up_face is not None):
            raise SidecarError("an accepted support plane gives up: --camera-pitch-deg / --up-face would override a "
                               "measurement with an assumption")
        if pl.accepted and sp is None:
            raise SidecarError(f"{pl.path} is accepted, but no start pose was given: a batch window's "
                               "start pose and its plane are taken together (--start-pose)")
    if sp is None:
        return
    if sp.window != name:
        raise SidecarError(f"{sp.path} is the start pose of window {sp.window}, not of {name}")
    if sp.world_camera == "ego_motion":
        if ego is None or file_sha256(ego.path) != sp.ego_motion_sha256:
            raise SidecarError(f"{sp.path} is in the world of the ego-motion {sp.ego_motion_sha256}; export with "
                               "--ego-motion and that file")
    elif ego is not None:
        raise SidecarError(f"{sp.path} is in the static camera's world; export without --ego-motion")
    if sp.kind == "snapped":
        if pl is None or pl.sha256 != sp.support_plane_sha256:
            raise SidecarError(f"{sp.path} was snapped on support plane {sp.support_plane_sha256}; pass that "
                               "support_plane.json")
        if not pl.accepted:
            raise SidecarError(f"{sp.path} was snapped on a plane the robot path refuses ({'; '.join(pl.refusal)})")
    elif pl is not None and pl.accepted:
        raise SidecarError(f"{sp.path} is not snapped although {pl.path} is accepted: re-run the start-pose screen")


def _apply_start_pose(cfg: KinematicPathConfig, frame_idx: np.ndarray, anchor: int, contact: int | None,
                      obj: np.ndarray, wrist: np.ndarray, joints: np.ndarray) -> dict | None:
    """Move every row from the contact onset on by `W = P inv(O_c0)` IN PLACE (module docstring of
    `fastdo.robot.start_pose`); the record of what was applied, or None without a sidecar."""
    sp = cfg.start_pose
    if sp is None:
        return None
    if contact is None:
        raise SidecarError("a start-pose sidecar replaces the contact-onset pose, but this clip has no contact onset")
    if int(frame_idx[anchor]) != sp.c0_frame_idx:
        raise SidecarError(f"{sp.path} replaces the pose at source frame {sp.c0_frame_idx}, but the export's contact "
                           f"onset is at source frame {int(frame_idx[anchor])}")
    w = sp.pose_world @ np.linalg.inv(obj[anchor])
    shift = float(np.linalg.norm(sp.pose_world[:3, 3] - obj[anchor, :3, 3]))   # the object frame's origin moves
    ang = float(np.degrees(np.arccos(np.clip((np.trace(w[:3, :3]) - 1.0) / 2.0, -1.0, 1.0))))
    check_correction(shift, ang, f"{sp.path}: the start pose (correction W at the export's contact onset)")
    obj[anchor:] = np.einsum("ij,tjk->tik", w, obj[anchor:])
    wrist[anchor:] = np.einsum("ij,tjk->tik", w, wrist[anchor:])
    joints[anchor:] = np.einsum("ij,tvj->tvi", w[:3, :3], joints[anchor:]) + w[:3, 3]
    return {**sp.to_dict(), "correction_world": w.tolist(), "correction_rot_deg": ang,
            "correction_shift_m": shift}


def _gravity(cfg: KinematicPathConfig, recon: Reconstruction, c2w0: np.ndarray) -> tuple[np.ndarray, dict, str]:
    """`(up in the video world, record, gravity source)`: an accepted support plane's `gravity_cam`, else the
    record's."""
    rec = recon.record
    pl = cfg.support_plane
    if pl is not None and pl.accepted:
        assert pl.gravity_cam is not None
        return c2w0[:3, :3] @ pl.gravity_cam, {"source": "support_plane", "support_plane": pl.to_dict()}, \
            "support_plane"
    out: dict = {"source": f"record (qc gravity_source={recon.gravity_source})"}
    if pl is not None:
        out["support_plane"] = pl.to_dict()
    return c2w0[:3, :3] @ rec.gravity_cam.astype(np.float64), out, recon.gravity_source


def prepare_task(recon: Reconstruction, cfg: KinematicPathConfig, g1: G1Upper) -> PreparedTask:
    """Stages 1-3 (video world, task span, wrist targets, hand retargeting) and the `ReachTask` the placement is
    searched on. Raises ValueError when the clip has no usable row."""
    rec = recon.record
    hand = rec.hand_right
    assert hand is not None
    _check_sidecars(recon, cfg)
    c2w_all = rec.cam_to_world.astype(np.float64)
    c2w0 = c2w_all[0]
    used_all = retarget_row_valid(rec, hand.valid)
    used_all[: cfg.first_row] = False
    obj_all = np.einsum("tij,tjk->tik", c2w_all, rec.object_pose.astype(np.float64))
    joints_all = _to_world(c2w_all, hand.joints.astype(np.float64))
    spikes_all = spike_rows(joints_all[:, 0], used_all, cfg.spike_window, cfg.spike_m, rec.timestamps,
                            cfg.max_wrist_speed)
    used_all = used_all & ~spikes_all
    if not used_all.any():
        raise ValueError(f"{rec.clip_id}: no row with a valid right hand and a supported object")
    verts_all = None if recon.right_vertices_cam is None else _to_world(c2w_all, recon.right_vertices_cam)
    hand_pts = joints_all if verts_all is None else verts_all
    stride = 1 if verts_all is None else cfg.contact_vertex_stride
    dist_all = hand_object_distance(hand_pts, obj_all, recon.mesh_vertices, recon.mesh_faces, rec.object_scale,
                                    used_all, stride)
    r0, r1, rest_check = task_span(dist_all, used_all, rec.timestamps, cfg.contact_m, cfg.contact_lead_s,
                                   cfg.contact_lag_s, cfg.rest_s, cfg.trim_to_contact)
    hull = trimesh.Trimesh(rec.object_scale * np.asarray(recon.mesh_vertices, dtype=np.float64),
                           np.asarray(recon.mesh_faces), process=False).convex_hull
    approach = cfg.trim_to_contact and cfg.approach_m > 0
    win = static_window(obj_all, np.asarray(hull.center_mass), used_all, dist_all, rec.timestamps, r0, r1,
                        cfg.contact_m, cfg.static_object_m, cfg.static_object_deg, approach)
    if cfg.approach_mode not in ("vertical", "palm"):
        raise ValueError(f"approach_mode must be 'vertical' or 'palm', got {cfg.approach_mode!r}")
    palm = approach and cfg.approach_mode == "palm"
    if palm:
        win = _pad_approach(win, used_all, dist_all, cfg.contact_m, cfg.approach.min_s)
    sl, synth = win.index, win.synthetic
    used, times = used_all[sl], win.times
    ho_dist = np.where(synth, np.nan, dist_all[sl])
    obj_world, joints_w = obj_all[sl], joints_all[sl]
    verts_w = None if verts_all is None else verts_all[sl]
    wrist_w = wrist_targets_world(joints_w)
    kp = mano_keypoints(joints_w, verts_w)
    kp_hand = np.einsum("tvi,tij->tvj", kp - joints_w[:, None, 0], wrist_w[:, :3, :3])
    vectors = dexpilot_vectors(kp_hand)
    anchor, contact = contact_anchor(ho_dist, used, cfg.contact_m)
    start_rec = _apply_start_pose(cfg, rec.frame_idx[sl], anchor, contact if contact is None else int(sl[contact]),
                                  obj_world, wrist_w, joints_w)
    up_record, gravity, gravity_source = _gravity(cfg, recon, c2w0)
    hits = np.flatnonzero(used & (ho_dist < cfg.contact_m))

    rate = 1.0 / float(np.median(np.diff(rec.timestamps)))
    hand_q = Dex3DexPilot(g1, cfg.dexpilot, rate_hz=rate).retarget_sequence(vectors, used, low_pass=True)
    # residual of the UNFILTERED solve (a fresh retargeter: same projection states and warm starts), so it measures
    # the retargeting, not the low-pass lag of the exported stream
    raw_q = Dex3DexPilot(g1, cfg.dexpilot, rate_hz=rate).retarget_sequence(vectors, used, low_pass=False)
    robot_vec = dexpilot_vectors(dex3_tip_positions(g1, raw_q))
    hand_err = np.where(used & ~synth, np.linalg.norm(robot_vec - vectors, axis=2).max(axis=1), np.nan)

    syms: tuple[SymmetryAxis, ...] = ()
    if cfg.use_symmetry:
        syms = tuple(rotational_symmetries(recon.mesh_vertices, recon.mesh_faces, rec.object_scale,
                                           cfg.symmetry_tol_m))
    ups = stable_ups(hull, obj_world[anchor, :3, :3], c2w0) if cfg.use_stable_ups else ()
    span = (int(hits[0]), int(hits[-1])) if hits.size and cfg.trim_to_contact and cfg.approach_m > 0 else None
    approach_rec = None
    if palm and span is not None:
        approach_rec = _palm_approach(recon, cfg, g1, hull, span[0], hits, times, obj_world, wrist_w, joints_w,
                                      hand_q)
    task = ReachTask(wrist_world=wrist_w, object_world=obj_world, joints_world=joints_w, anchor=anchor, rows=used,
                     times=times, hand_q=hand_q, mesh_vertices=recon.mesh_vertices, scale=rec.object_scale,
                     cam_to_world0=c2w0, up_record=up_record, rest_check=rest_check, symmetries=syms,
                     object_conf=np.asarray(rec.confidence, dtype=np.float64)[sl], contact_span=span,
                     approach_m=cfg.approach_m, hull_vertices=np.asarray(hull.vertices), hull_com=np.asarray(hull.center_mass),
                     stable_ups=ups, approach_synthesised=approach_rec is not None)
    return PreparedTask(task=task, row_range=(win.first, win.last), spikes=spikes_all[sl] & ~synth, ho_dist=ho_dist,
                        contact=contact, hand_err=hand_err, n_rows=rec.n_frames,
                        source_rows=np.where(synth, -1, sl).astype(np.int64), object_motion=win.to_dict(),
                        notes=win.warnings + _plane_notes(cfg), start_pose=start_rec, gravity=gravity,
                        gravity_source=gravity_source, approach=approach_rec)


def _pad_approach(win, used_all: np.ndarray, dist_all: np.ndarray, contact_m: float, min_s: float):
    """Prepend synthetic rows (copies of the first task row) until the rows before the first contact
    span `min_s`."""
    idx = np.asarray(win.index)
    hit = np.flatnonzero(used_all[idx] & (np.nan_to_num(dist_all[idx], nan=np.inf) < contact_m))
    if hit.size == 0:
        return win
    t = np.asarray(win.times, dtype=np.float64)
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.1
    lead = float(t[hit[0]] - t[0])
    extra = int(np.ceil((min_s - lead) / dt - 1e-9)) if lead < min_s - 1e-9 else 0
    if extra <= 0:
        return win
    return replace(win, index=np.concatenate([np.full(extra, idx[0]), idx]).astype(np.int64),
                   synthetic=np.concatenate([np.ones(extra, bool), np.asarray(win.synthetic)]),
                   times=np.concatenate([t[0] - dt * np.arange(extra, 0, -1), t]))


def _palm_approach(recon: Reconstruction, cfg: KinematicPathConfig, g1: G1Upper, hull: trimesh.Trimesh, c0: int,
                   hits: np.ndarray, times: np.ndarray, obj_world: np.ndarray, wrist_w: np.ndarray,
                   joints_w: np.ndarray, hand_q: np.ndarray) -> dict:
    """Plan the approach onto task row `c0` (`fastdo.robot.approach`) and write it IN PLACE into the
    rows before it (wrist, human joints moved rigidly with it, hand joints; up to the arrival row); the object is held
    at its pose. Raises
    `ApproachError` when no approach passes the collision check."""
    rec = recon.record
    base_obj = np.linalg.solve(obj_world, wrist_w)
    radius = float(np.linalg.norm(np.asarray(hull.vertices), axis=1).max())
    tc = cfg.transfer
    envelope = (np.sqrt(2.0) * tc.object_xy_tol + 2.0 * radius * np.sin(np.radians(tc.object_yaw_tol_deg) / 2.0)
                + max(tc.table_tol, TABLE_TOL_M))
    acfg = replace(cfg.approach, envelope_m=envelope)
    clr = HandClearance(recon.mesh_vertices, recon.mesh_faces, rec.object_scale, root=str(g1.root),
                        distmax=acfg.distmax)
    plan = plan_approach(times, c0, base_obj, hand_q, hits, np.asarray(hull.center_mass), clr, acfg)
    out = plan.to_dict() | {"c0_task_row": int(c0), "required_start_clearance_m": acfg.clear_m + envelope,
                            "envelope_m": envelope, "span_s": float(times[c0] - times[0])}
    if not plan.ok:
        raise ApproachError(f"{rec.clip_id}: synthesised approach refused: {plan.reason} ({out})")
    k = len(plan.base_obj)                                  # rows 0..arrival (c0, or c0-1 after a back-out)
    new_wrist = np.einsum("tij,tjk->tik", obj_world[:k], plan.base_obj)
    move = np.einsum("tij,tjk->tik", new_wrist, np.linalg.inv(wrist_w[:k]))
    joints_w[:k] = np.einsum("tij,tvj->tvi", move[:, :3, :3], joints_w[:k]) + move[:, None, :3, 3]
    wrist_w[:k] = new_wrist
    hand_q[:k] = plan.hand_q
    return out


def _plane_notes(cfg: KinematicPathConfig) -> tuple[str, ...]:
    pl = cfg.support_plane
    if pl is None or pl.accepted:
        return ()
    return (f"support plane REFUSED ({'; '.join(pl.refusal)}): up is not taken from it",)
