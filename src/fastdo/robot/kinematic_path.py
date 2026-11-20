"""The kinematic robot path: one reconstructed clip -> G1 right arm + Dex3-1 trajectory -> sim replay.

This IS the kinematic-baseline arm of the study: Unitree's DexPilot-style hand retargeting plus wrist IK of the same
video, with no physics in the loop. Every stage is a module of `fastdo.robot`, so the physics-refined arm can swap the
joint trajectories and reuse placement, export, replay and packaging.

Inputs (`load_reconstruction`, `fastdo.robot.world_frame`): an engine clip directory with `mesh.obj` (canonical object
mesh), `hands.npz` (`right_vertices`, camera frame, for the MANO tip vertices) and `clip_record.npz`
(`fastdo.io.clip_record.ClipRecord`) or the solve / hand / intrinsics artifacts it is rebuilt from; optionally the
window's camera ego-motion, which replaces the qc record's static camera (head motion is then not world motion).

Stages (`prepare_task` = 1-3, `run_kinematic_path` = all):
1. Video world: object poses, hand joints and vertices moved through `cam_to_world`; rows used = valid right hand AND
   supported object (`fastdo.qc.export.retarget_row_valid`, the retarget stage's own rule), not before `first_row`
   (a misregistered start), AND not a wrist SPIKE
   (`spike_rows`, masked like invalid rows and counted in the summary): rows of a short burst the wrist would have to
   enter or leave faster than `max_wrist_speed` (a smoke-test clip had 18-20 m/s hand-track jumps), then a row whose
   wrist is more than `spike_m` from the median wrist of the `spike_window` kept rows around it.
1b. Task span (`task_span`): with `trim_to_contact`, only the rows from `contact_lead_s` before the human's first
   hand-object contact to `contact_lag_s` after the last are kept (every per-row output covers exactly them,
   `row_range`); outside it the hand only comes and goes.
2. Wrist target: the Dex3 `base_link` frame of the human hand = position of MANO joint 0, rotation
   `mano_hand_frame` (Unitree's hand convention); keypoints in that frame -> the six DexPilot vectors.
3. Hand: `Dex3DexPilot.retarget_sequence` at the clip rate (rate-matched low-pass); the reported vector residual
   (`hand_vec_err`) is that of the unfiltered solve, so it measures the retargeting and not the low-pass lag.
4. Anchor and placement. The ANCHOR row is the contact onset: the first used row where the human hand (MANO
   vertices, every `contact_vertex_stride`-th; the 21 joints without `hands.npz`) is within `contact_m` of the object
   mesh (`hand_object_distance`); without such a row, the first used row. The object is placed on the table at its
   anchor pose (`place_on_table`), and is held STATIC before the anchor in the reference (`robot_frame_targets`).
   That the reconstructed object is static there is MEASURED, not assumed (`kinematic_rows.static_window`: hull
   centroid within `static_object_m` = 2 cm and rotation within `static_object_deg` = 10 deg of its contact-onset
   pose): rows up to the last one where it is farther are DROPPED, with a warning naming the displacement, and the
   robot does not reproduce that motion. On the early drafts the reconstructed object moved 35.3 cm (C3), 15.2 cm
   (C4) and 5.9 cm (C2) between the first kept row and the contact onset, 1.2 cm (C1) and 1.0 cm (C5) (frame-origin
   displacement, `pre_anchor_object_motion_m`). Before the anchor the wrist target keeps its pose
   RELATIVE to the object (object frame -> the placed start pose); from the anchor on, everything goes through the
   one rigid transform `T_rv`; both agree at the anchor. Anchoring at frame 0 instead failed the smoke-test clip: the
   reconstructed object travelled 0.61 m in the video world over the first 10 rows (1 s at 10 fps) while the human
   hand was still 13-30 cm from it, so the sim object sat 0.5-0.6 m from where the hand later grasped (closest
   robot-hand approach: 0.42 m).
   With `table_xy=None` the placement (object start, task yaw, table height, assumed head pitch when gravity was never
   estimated, grasp rotation about a symmetry axis) is SEARCHED for the largest minimum reach margin over the whole
   task (`fastdo.robot.reach`); before the first and after the last contact the wrist follows a vertical approach /
   retreat of `approach_m` instead of the human's path. Those rows need a static object too (measured as above;
   after the last contact against the last-contact pose); rows where it moved are dropped and their time refilled by
   SYNTHETIC rows (the first / last kept row repeated, `source_rows` -1) so the approach keeps its duration. Every
   such choice is listed in `adaptations`, with the measured object motion.
5. Arm: `solve_arm_ik` on every row (invalid rows hold the previous solution), collision-aware against the robot's
   own body and the table (`IkCollider` on `build_ik_model`, fingers at the stage-3 hand joints), joint speed capped
   at `ik.max_joint_speed` between rows; the per-row reach margin of that solution (`reach_eval.row_margins`, with the
   table-consistency rows of `reach.table_violation`) is reported.
6. Export: `build_trajectory` (250 / 100 Hz, smoothing, limits, gravity feed-forward).
7. Replay: `build_replay_scene` + `replay` against the placed reference object trajectory, read on the used rows whose
   object confidence is at least `search.object_conf_min` (rows past the clip end or unobserved are extrapolations;
   low-confidence rows are not trusted to say where the object is).
8. Transfers (`fastdo.robot.transfer.check_transfers`, with `transfer.enabled`): the run script's moves from each
   candidate HOME to the trajectory start and back are planned collision-free in the same scene (object at its start
   pose; at its simulated and its reference end pose for the way home), then the run script itself is simulated from
   every planned home, with operator-stop drills. The package carries the planned paths; the execution gate reads
   the simulation.
The row-level stages live in `fastdo.robot.kinematic_rows` (re-exported here).

Up direction: the record's `gravity_cam` (row 0), or, with `camera_pitch_deg`, a level camera pitched down by that
angle (`placement.pitched_up`, an operator assumption). The engine's QC stage records where `gravity_cam` and
`cam_to_world` came from (`qc.json` `gravity_source`, `camera_source`); with `gravity_source` "default" the up vector
is the camera's -y, wrong by the wearer's head pitch, and with `camera_source` "gravity_aligned_static" head motion is
world motion. Both are reported in `summary()["warnings"]` and the package README; on the smoke-test clip C1 (object
at a fixed (0.35, -0.20)) a pitch sweep of 0/30/45/60 deg moved the median wrist IK error from 0.21 to 0.11 m and the wrist at the contact onset
from 7.8 cm below to 2.9 cm above the table top. The placement search only considers object positions whose footprint lies
on the table (`placement.footprint_inside`, `table_margin`); an explicit `table_xy` that overhangs raises ValueError.
The placement search treats the head pitch as a free, REPORTED assumption only when gravity was never estimated.

Sidecars (`fastdo.robot.start_pose`; `KinematicPathConfig.start_pose` / `support_plane`, the export's
`--start-pose` / `--support-plane`): a start-pose sidecar replaces the contact-onset row's object pose (every
row from the onset on moved by one rigid correction, so the grasp is kept), and an ACCEPTED support plane (>= 2 rest
rows, normals within 3 deg) gives up from its `gravity_cam`, with gravity source "support_plane" and no
pitch / resting-face search. A record whose rest verdict is not True, or whose correction exceeds 8.6 cm / 12.2
deg, is refused (`start_pose.SidecarError`). Without sidecars nothing changes.
"""

from __future__ import annotations

from concurrent.futures import Executor
from dataclasses import dataclass, replace

import numpy as np

from fastdo.robot.arm_ik import ArmIkResult, IkCollider, solve_arm_ik
from fastdo.robot.g1_model import G1Upper, load_g1_upper
from fastdo.robot.grasp_closure import close_grasp
from fastdo.robot.kinematic_rows import (
    contact_anchor,
    hand_object_distance,
    path_warnings,
    robot_frame_targets,
    spike_rows,
    task_span,
    wrist_targets_world,
)
from fastdo.robot.kinematic_task import KinematicPathConfig, PreparedTask, prepare_task
from fastdo.robot.placement import Placement, footprint_inside
from fastdo.robot.reach import (
    PlacementChoice,
    ReachTask,
    SearchResult,
    choice_targets,
    physical_check,
    search_placement,
    table_depths,
    table_violation,
)
from fastdo.robot.reach_eval import RetryPool, ik_margins, row_margins
from fastdo.robot.replay import (
    ReplayMetrics,
    SceneConfig,
    build_ik_model,
    build_replay_scene,
    replay,
)
from fastdo.robot.rest import StableUp
from fastdo.robot.symmetry import SymmetryAxis
from fastdo.robot.trajectory import (
    RobotTrajectory,
    build_trajectory,
)
from fastdo.robot.transfer import TransferReport, check_transfers
from fastdo.robot.world_frame import Reconstruction, load_reconstruction

__all__ = ["KinematicPathConfig", "KinematicPathResult", "PreparedTask", "Reconstruction", "contact_anchor", "hand_object_distance",
           "load_reconstruction", "path_warnings", "prepare_task", "robot_frame_targets", "run_kinematic_path", "spike_rows",
           "task_span", "wrist_targets_world"]


@dataclass(frozen=True)
class KinematicPathResult:
    clip_id: str
    row_range: tuple[int, int]       # clip rows kept (first, last); every per-row array below covers exactly them
    used_rows: np.ndarray            # (T,) bool
    spike_rows: np.ndarray           # (T,) bool, masked as wrist spikes
    anchor_row: int                  # stage 4: contact onset, else the first used row (index into the kept rows)
    contact_row: int | None          # contact onset (None: the hand never came within contact_m)
    hand_object_dist: np.ndarray     # (T,) human hand to object mesh (m) on used rows, NaN elsewhere
    pre_anchor_object_motion: float  # m, max hull-centroid displacement from the anchor pose over the KEPT rows
                                     #   before it (<= static_object_m; `object_motion` has the dropped rows)
    rest_check: bool                 # the object rests >= rest_s before the human touches it
    choice: PlacementChoice
    placement: Placement
    symmetries: tuple[SymmetryAxis, ...]
    stable_ups: tuple[StableUp, ...]
    wrist_targets: np.ndarray        # (T,4,4) robot frame
    object_ref: np.ndarray           # (T,4,4) robot frame (metric object frame -> robot)
    ik: ArmIkResult
    margins: np.ndarray              # (T,) per-row reach margin of the exported IK (`reach_eval.row_margins`), NaN
                                     #      off the used rows
    hand_q: np.ndarray               # (T,7) hardware order
    hand_vec_err: np.ndarray         # (T,) max DexPilot vector residual (m) of the unfiltered solve on used
                                     #      rows, NaN elsewhere
    trajectory: RobotTrajectory
    metrics: ReplayMetrics
    logs: dict[str, np.ndarray]
    search: SearchResult | None
    camera_source: str
    gravity_source: str
    up_source: str                   # "record" or "camera_pitch_deg=<value>" (+ " (searched)")
    warnings: tuple[str, ...]
    adaptations: tuple[str, ...]
    ego: dict | None = None
    table_check: dict | None = None
    first_row: int = 0
    trim_to_contact: bool = True
    source_rows: np.ndarray | None = None   # (T,) clip row of each row, -1 = synthetic (None: row_range[0] + t)
    object_motion: dict | None = None       # `kinematic_rows.StaticWindow.to_dict()`
    transfers: TransferReport | None = None  # stage 8 (None: `transfer.enabled` False; no package can be written)
    table: dict | None = None                # the simulated table rectangle (`table_rectangle`)
    start_pose: dict | None = None           # the applied start-pose sidecar (`fastdo.robot.start_pose`), or None
    gravity: dict | None = None              # where up came from: an accepted support plane or the record
    approach: dict | None = None             # the synthesised approach (`approach_mode` "palm")
    closure: dict | None = None              # the grasp closure (`grasp_closure`)
    object_mass_kg: float | None = None      # the simulated object mass (0.2 kg until weighed)

    def clip_row(self, t: int) -> int:
        """Clip row of task row `t` (-1: a synthetic approach / retreat row)."""
        return int(self.source_rows[t]) if self.source_rows is not None else int(t + self.row_range[0])

    def summary(self) -> dict:
        used = self.used_rows
        m = self.margins[used]
        return {
            "clip_id": self.clip_id, "warnings": list(self.warnings), "adaptations": list(self.adaptations),
            "camera_source": self.camera_source, "gravity_source": self.gravity_source, "up_source": self.up_source,
            "ego_motion": self.ego, "row_range": list(self.row_range), "rows": len(used), "used_rows": int(used.sum()),
            "spike_rows": int(self.spike_rows.sum()),
            "anchor_row": self.anchor_row, "anchor_clip_row": self.clip_row(self.anchor_row),
            "contact_row": self.contact_row, "rest_check": self.rest_check,
            "pre_anchor_object_motion_m": self.pre_anchor_object_motion, "object_motion": self.object_motion,
            "synthetic_rows": 0 if self.source_rows is None else int((self.source_rows < 0).sum()),
            "human_hand_object_dist_m": _stats(self.hand_object_dist[used]),
            "choice": self.choice.to_dict(),
            "table_xy": list(self.placement.table_xy), "table_z": self.placement.table_z,
            "symmetry": [a.to_dict() for a in self.symmetries],
            "stable_ups": [u.to_dict() for u in self.stable_ups],
            "ik_pos_err_m": _stats(self.ik.pos_err[used]), "ik_rot_err_rad": _stats(self.ik.rot_err[used]),
            "reach_margin": {"min": float(m.min()), "mean": float(m.mean()), "infeasible_rows": int((m < 0).sum())},
            "wrist_path_extent_m": _extent(self.wrist_targets[used, :3, 3]),
            "table_check": self.table_check,
            "hand_vector_err_m": _stats(self.hand_vec_err[used]),
            "ik_clearance_m": _stats(self.ik.clearance[used]),
            "search": None if self.search is None else {"score": self.search.score, "final": self.search.final,
                                                        "evaluated": len(self.search.records)},
            "time_scale": self.trajectory.time_scale, "duration_s": self.trajectory.duration,
            "replay": self.metrics.to_dict(),
            "transfers": None if self.transfers is None else self.transfers.to_dict(),
            "table": self.table,
            "start_pose": self.start_pose, "gravity": self.gravity,
            "approach": self.approach, "grasp_closure": self.closure,
            "object_mass_kg": self.object_mass_kg,
            "object_mass_note": "ASSUMED, not weighed: S3 runs at this mass until the lab weighs the object, then the "
                                "package is re-simulated at the weighed mass",
        }


def table_rectangle(scene: SceneConfig) -> dict:
    """The table top the simulation and the run script's gate assume, robot frame (m): edges, top, thickness."""
    return {"near_edge_x_m": float(scene.table_x[0]), "far_edge_x_m": float(scene.table_x[1]),
            "right_edge_y_m": float(scene.table_y[0]), "left_edge_y_m": float(scene.table_y[1]),
            "top_z_m": float(scene.table_z), "thickness_m": float(scene.table_thickness)}


def _stats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0}
    return {"n": int(x.size), "median": float(np.median(x)), "p95": float(np.quantile(x, 0.95)), "max": float(x.max())}


def _extent(points: np.ndarray) -> float:
    """Largest distance between two of the `(N,3)` points (m)."""
    p = np.asarray(points, dtype=np.float64)
    return float(np.linalg.norm(p[:, None] - p[None], axis=2).max()) if len(p) else 0.0


def _adaptations(recon: Reconstruction, prep: PreparedTask, cfg: KinematicPathConfig,
                 choice: PlacementChoice, task: ReachTask, searched: bool) -> tuple[str, ...]:
    """Every choice that is not the human's recorded motion, in words (module docstring, "Adaptations")."""
    out = []
    n_rows, row_range, mo = prep.n_rows, prep.row_range, prep.object_motion
    if recon.ego is not None:
        out.append(f"world frame: the camera ego-motion (self-check ok, camera path {recon.ego.cam_path_m:.2f} m, "
                   f"{recon.ego.cam_rot_extent_deg:.0f} deg; per-row scale {recon.ego.scale_range[0]:.3f}-"
                   f"{recon.ego.scale_range[1]:.3f}, the object keeps one metric size)")
    if row_range != (0, n_rows - 1):
        why = []
        if cfg.first_row:
            why.append(f"rows before {cfg.first_row} misregistered")
        if cfg.trim_to_contact:
            why.append(f"human hand-object contact span +{cfg.contact_lead_s:g} s before / {cfg.contact_lag_s:g} s after")
        if mo["pre_contact_dropped_m"] or mo["pre_contact_dropped_deg"] or mo["post_contact_dropped_m"] \
                or mo["post_contact_dropped_deg"]:
            why.append(f"rows where the reconstructed object moved more than {mo['tolerance_m'] * 100:g} cm / "
                       f"{mo['tolerance_deg']:g} deg without the right hand on it (see the warnings)")
        out.append(f"rows kept: {row_range[0]}-{row_range[1]} of 0-{n_rows - 1} ({'; '.join(why)})")
    anchor_clip = int(prep.source_rows[task.anchor])
    video = prep.source_rows >= 0
    n_pre = int(video[:task.anchor].sum())                 # kept VIDEO rows before the onset (synthetic ones excluded)

    def motion(n: int, m: float, deg: float, where: str, synth: int) -> str:
        if n == 0:                                          # nothing measured: say so
            return f"no video rows are kept {where} ({synth} synthetic rows: nothing measured there)"
        return f"the reconstructed object moves at most {m * 100:.1f} cm / {deg:.1f} deg over the {n} kept video rows " \
               f"{where} (measured)"

    if task.anchor > 0:
        out.append(f"object reference held at its contact-onset pose (clip row {anchor_clip}) before it: "
                   + motion(n_pre, mo["pre_contact_kept_m"], mo["pre_contact_kept_deg"], "before it",
                            mo["synthetic_rows_before"]))
    if task.contact_span is not None and task.approach_m > 0:
        c0, c1 = (int(prep.source_rows[v]) for v in task.contact_span)
        n_post = int(video[task.contact_span[1] + 1:].sum())
        pads = (mo["synthetic_rows_before"], mo["synthetic_rows_after"])
        before = motion(n_pre, mo["pre_contact_kept_m"], mo["pre_contact_kept_deg"], "before the first contact",
                        pads[0])
        after = motion(n_post, mo["post_contact_kept_m"], mo["post_contact_kept_deg"], "after the last contact",
                       pads[1])
        out.append(f"approach / retreat: the human's hand path before the first contact (row {c0}) and after the last "
                   f"(row {c1}) is replaced by a vertical approach / retreat of {task.approach_m * 100:.0f} cm at the "
                   f"contact wrist orientation; the human hand is at least {cfg.contact_m * 100:g} cm from the object "
                   f"there (the contact threshold); {before}; {after} ({pads[0]} / {pads[1]} synthetic rows replace "
                   "video rows where it moved more)")
    sp = prep.start_pose
    if sp is not None:
        snap = sp.get("snap") or {}
        how_pose = (f"SNAPPED onto the accepted support plane (face {snap.get('face_index')}, turned "
                    f"{snap.get('rot_deg', float('nan')):.1f} deg, centroid moved {snap.get('disp_m', float('nan')) * 100:.1f} "
                    f"cm; caps {sp['caps']})") if sp["kind"] == "snapped" else "consolidated, not snapped"
        out.append(f"start pose from the sidecar ({sp['path']}, source {sp['source']}, {how_pose}): the object "
                   f"pose at the contact onset (source frame {sp['c0_frame_idx']}) is replaced by it, and every row from "
                   f"the onset on is moved by the same rigid correction (the object start {sp['correction_shift_m'] * 100:.1f} "
                   f"cm and {sp['correction_rot_deg']:.1f} deg from the onset row's pose) so the grasp keeps its "
                   "hand-object relation")
    how = "chosen by the placement search" if searched else "given explicitly"
    out.append(f"placement ({how}): object start at ({choice.table_xy[0]:.3f}, {choice.table_xy[1]:.3f}) m, whole task "
               f"yawed {choice.yaw_deg:+.1f} deg about the vertical, table top at {choice.table_z:.3f} m")
    if choice.up_face is not None:
        f = task.stable_ups[choice.up_face]
        out.append(f"ASSUMED up = the normal of the object's resting hull face {choice.up_face} ({how}; "
                   f"{np.round(f.up_object, 3).tolist()} in the mesh frame, landing probability {f.probability:.2f}, "
                   f"{f.prior_angle_deg:.0f} deg from the nearest head-pitch up): gravity was never estimated for this "
                   "clip, so 'up' is an assumption that makes the start pose a resting pose, not a measurement")
    if choice.pitch_deg is not None:
        out.append(f"ASSUMED head pitch {choice.pitch_deg:g} deg down ({how}): gravity was never estimated for this "
                   "clip, so 'up' is an assumption, not a measurement")
    if choice.sym_axis is not None and choice.sym_deg % 360.0:
        ax = task.symmetries[choice.sym_axis]
        mism = ax.mismatch_m.get(round(choice.sym_deg % 360.0), float("nan"))
        out.append(f"grasp rotated {choice.sym_deg:g} deg about the object's principal axis {ax.index} "
                   f"({np.round(ax.axis, 3).tolist()} in the mesh frame; surface mismatch {mism * 1000:.1f} mm p95)")
    return tuple(out)


def run_kinematic_path(recon: Reconstruction, cfg: KinematicPathConfig | None = None,
                       g1: G1Upper | None = None, pool: RetryPool | Executor | None = None) -> KinematicPathResult:
    """Stages 1-7 of the module docstring (`pool`: the placement search's IK process pool). Raises ValueError when
    the clip has no usable row."""
    cfg = KinematicPathConfig() if cfg is None else cfg
    g1 = load_g1_upper() if g1 is None else g1
    rec = recon.record
    prep = prepare_task(recon, cfg, g1)
    task = prep.task
    gravity_source = prep.gravity_source or recon.gravity_source
    syms, used, times, hand_q, anchor = task.symmetries, task.rows, task.times, task.hand_q, task.anchor
    r0, r1 = prep.row_range
    tx, ty = cfg.scene.table_x, cfg.scene.table_y
    search = None
    if cfg.table_xy is None:
        scfg = replace(cfg.search, table_margin=cfg.table_margin)
        if cfg.camera_pitch_deg is not None:
            scfg = replace(scfg, pitch_deg=(cfg.camera_pitch_deg,))
        allow_pitch = cfg.camera_pitch_deg is not None or gravity_source == "default"   # never with a plane
        search_task = replace(task, stable_ups=()) if cfg.camera_pitch_deg is not None else task
        search = search_placement(search_task, scfg, tx, ty, pool, allow_pitch=allow_pitch)
        choice = search.choice
    else:
        if cfg.sym_axis is not None and round(cfg.sym_deg % 360.0) not in syms[cfg.sym_axis].allowed_deg:
            raise ValueError(f"the object is not symmetric under {cfg.sym_deg:g} deg about axis {cfg.sym_axis}")
        if cfg.up_face is not None and not 0 <= cfg.up_face < len(task.stable_ups):
            raise ValueError(f"up_face {cfg.up_face}: the object has {len(task.stable_ups)} resting-face candidates")
        choice = PlacementChoice(table_xy=(float(cfg.table_xy[0]), float(cfg.table_xy[1])),
                                 yaw_deg=float(np.degrees(cfg.yaw_offset)), table_z=cfg.table_z,
                                 pitch_deg=cfg.camera_pitch_deg, sym_axis=cfg.sym_axis, sym_deg=cfg.sym_deg,
                                 up_face=cfg.up_face)
    placement, wrist_r, obj_r = choice_targets(task, choice)
    if not footprint_inside(placement.object_start, recon.mesh_vertices, rec.object_scale, tx, ty, cfg.table_margin):
        raise ValueError(f"object at table_xy {choice.table_xy} overhangs the table {tx} x {ty} "
                         f"(margin {cfg.table_margin} m)")
    scene_cfg = replace(cfg.scene, table_z=choice.table_z)
    ik_g1 = build_ik_model(scene_cfg, root=str(g1.root))
    collider = IkCollider(ik_g1, cfg.collision_d_safe, cfg.collision_weight)
    ik = solve_arm_ik(ik_g1, wrist_r, cfg=cfg.ik, valid=used, collider=collider, times=times, hand_q=hand_q)
    first = int(np.flatnonzero(used)[0])
    jl, sig = ik_margins(ik_g1, ik.q, cfg.ik.rot_weight)
    scfg = replace(cfg.search, table_margin=cfg.table_margin)
    tviol = table_violation(task, placement, wrist_r, obj_r, tx, ty, scfg)
    hand_d, obj_d = table_depths(task, placement, wrist_r, obj_r, tx, ty, scfg)
    margins = row_margins(np.nan_to_num(ik.pos_err), np.nan_to_num(ik.rot_err), jl, sig, ik.clearance, cfg.margin)
    margins = np.where(tviol > 0.0, np.minimum(margins, -np.maximum(tviol, 1.0)), margins)
    margins = np.where(used, margins, np.nan)
    table_chk = physical_check(task, placement, wrist_r, obj_r, tx, ty, scfg)
    table_chk["infeasible"] = [
        {"row": int(prep.source_rows[t]), "pos_err_m": round(float(ik.pos_err[t]), 4), "rot_err_rad": round(float(ik.rot_err[t]), 3),
         "clearance_m": round(float(ik.clearance[t]), 4), "hand_below_table_m": round(float(hand_d[t]), 3),
         "object_below_table_m": round(float(obj_d[t]), 3)} for t in np.flatnonzero(used & (margins < 0))]
    # rows before the first used row take its solution (solve_arm_ik holds the initial guess there)
    q_arm = ik.q.copy()
    q_arm[:first] = q_arm[first]
    scene = build_replay_scene(recon.mesh_vertices, recon.mesh_faces, rec.object_scale, placement.object_start,
                               scene_cfg, root=str(g1.root))
    conf = np.ones(len(used)) if task.object_conf is None else np.asarray(task.object_conf)
    trusted = used & (conf >= cfg.search.object_conf_min)
    ref_rows = trusted if trusted.any() else used
    closure = None
    hand_rows = hand_q
    if cfg.grasp_closure:                              # stage 6b
        if task.contact_span is None:
            raise ValueError(f"{rec.clip_id}: grasp closure needs a contact span (trim_to_contact, approach_m > 0)")
        hand_rows, closure = close_grasp(g1, scene, times, q_arm, hand_q, task.contact_span, obj_r,
                                         placement.object_start, ref_rows, cfg.criteria, cfg.export, cfg.limits,
                                         cfg.closure)
    traj = build_trajectory(g1, times, q_arm, hand_rows, cfg.export, cfg.limits)
    metrics, logs = replay(scene, traj, obj_r, placement.object_start, cfg.criteria, ref_rows=ref_rows)
    transfers = None
    if cfg.transfer.enabled:
        rows = np.flatnonzero(ref_rows)
        transfers = check_transfers(scene, traj, cfg.limits, g1, placement.object_start, obj_r[rows[-1]],
                                    logs["object_qpos"][-1], cfg.transfer, object_ref_start=obj_r[rows[0]])
    pre_motion = float(prep.object_motion["pre_contact_kept_m"])
    face = None if choice.up_face is None else task.stable_ups[choice.up_face]
    face_note = None if face is None else (f"face {choice.up_face}, landing probability {face.probability:.2f}, "
                                           f"{face.prior_angle_deg:.0f} deg from the nearest head-pitch up")
    warnings = path_warnings(recon.camera_source, gravity_source, choice.pitch_deg,
                             float(wrist_r[anchor, 2, 3] - choice.table_z), face_note) + prep.notes
    up_source = ("record" if choice.pitch_deg is None else f"camera_pitch_deg={choice.pitch_deg:g}") if face is None \
        else f"up_face={choice.up_face}"
    if gravity_source == "support_plane":
        up_source = "support_plane"
    if search is not None:
        tried = {(r["choice"]["pitch_deg"], r["choice"]["up_face"]) for r in search.records}
        if len(tried) > 1:
            up_source += (f" (searched over {sum(f is None for _, f in tried)} head pitches and "
                          f"{sum(f is not None for _, f in tried)} resting faces)")
    return KinematicPathResult(
        clip_id=rec.clip_id, row_range=(r0, r1), used_rows=used, spike_rows=prep.spikes, anchor_row=anchor,
        contact_row=prep.contact, hand_object_dist=prep.ho_dist, pre_anchor_object_motion=pre_motion,
        rest_check=task.rest_check,
        choice=choice, placement=placement, symmetries=syms, stable_ups=task.stable_ups, wrist_targets=wrist_r, object_ref=obj_r, ik=ik,
        margins=margins, hand_q=hand_q, hand_vec_err=prep.hand_err, trajectory=traj, metrics=metrics, logs=logs,
        search=search, camera_source=recon.camera_source, gravity_source=gravity_source, up_source=up_source,
        warnings=warnings, adaptations=_adaptations(recon, prep, cfg, choice, task, search is not None),
        ego=None if recon.ego is None else recon.ego.to_dict(), table_check=table_chk, first_row=cfg.first_row,
        trim_to_contact=cfg.trim_to_contact, source_rows=prep.source_rows, object_motion=prep.object_motion,
        transfers=transfers, table=table_rectangle(scene_cfg), start_pose=prep.start_pose, gravity=prep.gravity,
        approach=prep.approach, closure=closure, object_mass_kg=float(scene_cfg.object_mass))
