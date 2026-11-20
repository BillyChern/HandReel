"""The video world of a reconstructed clip for the robot path: records and camera ego-motion.

Two inputs the kinematic path (`fastdo.robot.kinematic_path`) did not have before:

1. `record_from_engine_dir`: an engine clip directory that kept `solve.npz`, `hands.npz`, `intrinsics.npz` and
   `mesh.obj` but no `clip_record.npz` (the "fresh" windows of the robot-path shortlist, C2/C3/C5) gets the record the
   qc stage would have written for it (`fastdo.engine.workers_qc`: the solve's poses, scale and confidence, both hand
   tracks with a side without a valid row exported as None, the static gravity-aligned camera from the default up, gate
   score NaN without `gate.json`). Timestamps are `frame_idx / source_fps`: these archives no longer hold `decode.npz`,
   and every such window is a 30 fps Ego4D window decoded at stride 3 (`frame_idx` steps of 3 = 10 fps rows), which
   the function checks.

2. `apply_ego_motion`: replaces the record's camera with the window's ego-motion (`ego_motion.npz` of
   `fastdo.geometry.ego_motion`, written by the geometry stage), when its self-check `ok` holds
   and every row is on the trajectory. Without it the qc record's camera is STATIC (`camera_source`
   "gravity_aligned_static"), so head motion becomes world motion of the hand and the object: the C1 window's
   camera travels 1.69 m and turns 42 deg in 10 s (its ego summary), and the static record moved the reference
   object 0.40 m between the contact onset and the clip end, against 0.24 m in the ego world.
   The ego-motion is a Sim(3) per row, X_w = s_t Q_t X_t + u_t (world = the reference row's camera, with its
   metric scale). The record keeps rigid poses, so the per-row scale s_t moves into the camera-frame quantities: hand
   joints and vertices, the hand translation and the object translation are multiplied by s_t, and `cam_to_world`
   becomes [Q_t | u_t]. Every hand point and the object centre then land exactly where the Sim(3) puts them. The object
   keeps ONE metric scale (the solve's, i.e. the reference row's), so its SIZE is off by s_t on row t; the
   `scale_range` in `EgoInfo` bounds that (C1: s_t in [0.993, 1.168]).
   Up is unchanged: `gravity_cam` (row 0's camera) composed with the new `cam_to_world[0]`; with the reference row 0
   that is the same vector as before.
   Opt-in (default off): `max_gap_rows > 0` first fills interior runs of at most that many off-trajectory
   rows on the SE(3) x R+ geodesic between their valid neighbours (`fastdo.geometry.ego_gaps`); a window is still
   refused if any row stays invalid, and `EgoInfo.filled_rows` counts the filled rows.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
import trimesh

from fastdo.geometry.ego_gaps import fill_gaps
from fastdo.io.clip_record import ClipRecord, HandTrack
from fastdo.qc.export import DEFAULT_GRAVITY_CAM, export_clip_record, gravity_aligned_cam_to_world
from fastdo.solver.api import TrajectorySolution

__all__ = ["EgoInfo", "Reconstruction", "apply_ego_motion", "ego_motion_info", "load_reconstruction",
           "record_from_engine_dir"]

_SIDES = ("left", "right")


@dataclass(frozen=True)
class EgoInfo:
    path: str
    ok: bool
    valid_rows: int
    rows: int
    reference_row: int
    scale_range: tuple[float, float]      # min / max of the per-row Sim(3) scale s_t
    cam_path_m: float
    cam_rot_extent_deg: float
    filled_rows: int = 0                  # rows filled by `ego_gaps.fill_gaps` (only with max_gap_rows > 0)

    def to_dict(self) -> dict:
        d = {"path": self.path, "ok": self.ok, "valid_rows": self.valid_rows, "rows": self.rows,
             "reference_row": self.reference_row, "scale_range": list(self.scale_range),
             "cam_path_m": self.cam_path_m, "cam_rot_extent_deg": self.cam_rot_extent_deg}
        if self.filled_rows:                  # absent by default: stored stats stay byte-identical
            d["filled_rows"] = self.filled_rows
        return d


def record_from_engine_dir(clip_dir: str | Path, source_fps: float = 30.0, stride: int = 3) -> ClipRecord:
    """The qc stage's `ClipRecord` of an engine clip directory without one (module docstring, item 1).

    Raises:
        FileNotFoundError: If `solve.npz`, `hands.npz`, `intrinsics.npz` or `mesh.obj` is missing.
        ValueError: If the hand rows are not the solve rows, or the rows are not `stride` source frames apart.
    """
    root = Path(clip_dir)
    for name in ("solve.npz", "hands.npz", "intrinsics.npz", "mesh.obj"):
        if not (root / name).is_file():
            raise FileNotFoundError(f"{root} has no {name}")
    with np.load(root / "solve.npz") as s:
        sol = {k: np.asarray(s[k]) for k in ("poses", "scale", "covariance", "per_frame_cost", "confidence",
                                            "iterations", "seconds", "frame_idx")}
    frame_idx = sol["frame_idx"].astype(np.int64)
    if frame_idx.size > 1 and not np.all(np.diff(frame_idx) == stride):
        raise ValueError(f"{root}: rows are not {stride} source frames apart: {np.unique(np.diff(frame_idx)).tolist()}")
    with np.load(root / "hands.npz") as h:
        if not np.array_equal(np.asarray(h["frame_idx"]), frame_idx):
            raise ValueError(f"{root}: hands.npz rows are not the solve rows")
        tracks: dict[str, HandTrack | None] = {}
        for side in _SIDES:
            tr = HandTrack(params=np.asarray(h[f"{side}_params"]), betas=np.asarray(h[f"{side}_betas"]),
                           trans=np.asarray(h[f"{side}_trans"]), joints=np.asarray(h[f"{side}_joints"]),
                           valid=np.asarray(h[f"{side}_valid"]).astype(bool))
            tracks[side] = tr if tr.valid.any() else None
    with np.load(root / "intrinsics.npz") as k:
        K = np.asarray(k["K"])
    gate = float("nan")
    if (root / "gate.json").is_file():
        gate = float(json.loads((root / "gate.json").read_text())["score"])
    solution = TrajectorySolution(poses=torch.from_numpy(sol["poses"]), scale=float(sol["scale"]),
                                  covariance=torch.from_numpy(sol["covariance"]),
                                  per_frame_cost=torch.from_numpy(sol["per_frame_cost"]),
                                  confidence=torch.from_numpy(sol["confidence"]), iterations=int(sol["iterations"]),
                                  seconds=float(sol["seconds"]))
    c2w = gravity_aligned_cam_to_world(DEFAULT_GRAVITY_CAM, frame_idx.size)
    clip_id = root.name
    return export_clip_record(clip_id, solution, str(root / "mesh.obj"), torch.from_numpy(K), c2w,
                              torch.from_numpy(DEFAULT_GRAVITY_CAM), tracks["left"], tracks["right"], gate, frame_idx,
                              frame_idx / float(source_fps))


def ego_motion_info(path: str | Path) -> EgoInfo:
    """The self-check of an `ego_motion.npz` (no conversion)."""
    with np.load(path, allow_pickle=False) as z:
        c2w = np.asarray(z["cam_to_world"], dtype=np.float64)
        valid = np.asarray(z["valid"], dtype=bool)
        summary = json.loads(str(z["summary"]))
        ref = int(z["reference_row"])
    s = np.cbrt(np.linalg.det(c2w[valid, :3, :3])) if valid.any() else np.array([np.nan])
    return EgoInfo(path=str(path), ok=bool(summary.get("ok", False)), valid_rows=int(valid.sum()), rows=len(valid),
                   reference_row=ref, scale_range=(float(np.min(s)), float(np.max(s))),
                   cam_path_m=float(summary.get("cam_path_m", np.nan)),
                   cam_rot_extent_deg=float(summary.get("cam_rot_extent_deg", np.nan)))


def _scaled_track(track: HandTrack | None, s: np.ndarray) -> HandTrack | None:
    if track is None:
        return None
    return replace(track, trans=(track.trans * s[:, None]).astype(np.float32),
                   joints=(track.joints * s[:, None, None]).astype(np.float32))


def apply_ego_motion(record: ClipRecord, right_vertices_cam: np.ndarray | None, ego_path: str | Path,
                     max_gap_rows: int = 0) -> tuple[ClipRecord, np.ndarray | None, EgoInfo]:
    """`(record, right vertices, info)` in the ego-motion world (module docstring, item 2; `max_gap_rows`: the
    opt-in gap fill, 0 = off).

    Raises:
        ValueError: If the ego-motion's rows are not the record's, its self-check failed, or a row is off the
            trajectory after the optional fill (a consumer must abstain then: `fastdo.geometry.ego_motion`).
    """
    info = ego_motion_info(ego_path)
    with np.load(ego_path, allow_pickle=False) as z:
        c2w = np.asarray(z["cam_to_world"], dtype=np.float64)
        frame_idx = np.asarray(z["frame_idx"], dtype=np.int64)
        valid = np.asarray(z["valid"], dtype=bool)
    if not np.array_equal(frame_idx, record.frame_idx):
        raise ValueError(f"{ego_path}: rows are not the record's rows")
    if not info.ok:
        raise ValueError(f"{ego_path}: the ego-motion self-check failed (ok=False); the static camera stays")
    if max_gap_rows > 0 and not valid.all():
        c2w, valid, filled = fill_gaps(c2w, valid, max_gap_rows)
        info = replace(info, valid_rows=int(valid.sum()), filled_rows=int(filled.sum()))
    if info.valid_rows != info.rows:
        raise ValueError(f"{ego_path}: {info.rows - info.valid_rows} rows are off the camera trajectory")
    s = np.cbrt(np.linalg.det(c2w[:, :3, :3]))
    rigid = c2w.copy()
    rigid[:, :3, :3] = c2w[:, :3, :3] / s[:, None, None]
    pose = np.asarray(record.object_pose, dtype=np.float64).copy()
    pose[:, :3, 3] *= s[:, None]
    out = replace(record, cam_to_world=rigid.astype(np.float32), object_pose=pose.astype(np.float32),
                  hand_left=_scaled_track(record.hand_left, s), hand_right=_scaled_track(record.hand_right, s))
    out.validate()
    verts = None if right_vertices_cam is None else np.asarray(right_vertices_cam, dtype=np.float64) * s[:, None, None]
    return out, verts, info


@dataclass(frozen=True)
class Reconstruction:
    record: ClipRecord
    right_vertices_cam: np.ndarray | None     # (T,778,3) or None (tips then come from the joints)
    mesh_vertices: np.ndarray                 # (V,3) canonical
    mesh_faces: np.ndarray                    # (F,3)
    source_dir: Path
    camera_source: str = "unknown"            # qc.json `camera_source` ("unknown" without a qc.json), or "ego_motion"
    gravity_source: str = "unknown"           # qc.json `gravity_source`
    record_source: str = "clip_record.npz"    # or "rebuilt" (`world_frame.record_from_engine_dir`)
    ego: EgoInfo | None = None                # the applied ego-motion (`world_frame.apply_ego_motion`)


def load_reconstruction(clip_dir: str | Path, ego_motion: str | Path | None = None,
                        ego_max_gap_rows: int = 0) -> Reconstruction:
    """Read an engine clip directory (module docstring). Without `clip_record.npz` the qc stage's record is rebuilt
    from the solve and hand artifacts (`world_frame.record_from_engine_dir`). With `ego_motion` (an `ego_motion.npz`
    of the same rows) the camera becomes the ego-motion (`world_frame.apply_ego_motion`; refused when its
    self-check failed; `ego_max_gap_rows`: its opt-in gap fill, 0 = off). Raises FileNotFoundError /
    ValueError on a bad directory."""
    root = Path(clip_dir)
    if (root / "clip_record.npz").is_file():
        record, record_source = ClipRecord.load(root / "clip_record.npz"), "clip_record.npz"
    else:
        record, record_source = record_from_engine_dir(root), "rebuilt"
    record.validate()
    if record.hand_right is None:
        raise ValueError(f"{root}: the clip has no right-hand track")
    verts = None
    if (root / "hands.npz").is_file():
        with np.load(root / "hands.npz") as h:
            if "right_vertices" in h:
                verts = np.asarray(h["right_vertices"], dtype=np.float64)
                if not np.array_equal(np.asarray(h["frame_idx"]), record.frame_idx):
                    raise ValueError(f"{root}: hands.npz rows are not the clip record's rows")
    mesh = trimesh.load(root / "mesh.obj", process=False, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"{root / 'mesh.obj'} is not a single triangle mesh")
    qc: dict = {}
    if (root / "qc.json").is_file():
        qc = json.loads((root / "qc.json").read_text())
    camera_source = str(qc.get("camera_source", "unknown"))
    ego = None
    if ego_motion is not None:
        record, verts, ego = apply_ego_motion(record, verts, ego_motion, ego_max_gap_rows)
        camera_source = "ego_motion"
    return Reconstruction(record=record, right_vertices_cam=verts, mesh_vertices=np.asarray(mesh.vertices),
                          mesh_faces=np.asarray(mesh.faces), source_dir=root, camera_source=camera_source,
                          gravity_source=str(qc.get("gravity_source", "unknown")), record_source=record_source,
                          ego=ego)
