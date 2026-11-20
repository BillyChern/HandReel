"""Per-window start-pose and gravity sidecars for the kinematic robot path.

The start-pose screen replaces the single contact-onset row's object
pose, whose ~cm / ~10 deg noise tilted the drafts' start poses, by a CONSOLIDATED rest pose (median position, chordal
L1 mean rotation over the rest rows before the onset), SNAPPED onto the estimated support plane when the window's
`support_plane.json` is accepted (capped: 2 cm, 15 deg). It writes the pose as `start_pose_a1.json`; the geometry
sidecar `support_plane.json` (`fastdo.geometry.support_plane_io`) carries `gravity_cam`, the plane
normal in row 0's camera (an UP vector, `ClipRecord.gravity_cam`'s convention).

`load_start_pose` / `load_support_plane` read and check them; `fastdo.robot.kinematic_task.prepare_task` applies them:
- **start pose**: the object pose at the contact-onset row becomes the sidecar pose `P`; every row from the onset on
  (object, wrist, hand joints) is moved by the one rigid world correction `W = P inv(O_c0)`, so the hand-object
  relation of the grasp is kept and the path stays continuous at the onset (before it the reference is held at `P` and
  the wrist keeps its pose relative to the per-row object, as without a sidecar). The rows, the contact onset and the
  object-motion measurements are those of the reconstruction (computed before `W`).
- **gravity**: when the plane is ACCEPTED (its own flags, AND at least `MIN_PLANE_ROWS` = 2 rest rows with the normal
  agreement sd <= `MAX_NORMAL_SD_DEG` = 3 deg), up = `cam_to_world[0] gravity_cam`, the gravity source is
  "support_plane" and the placement search does not search the head pitch or a resting face (`allow_pitch` False).
  A refused plane leaves the record's up (and the gravity warning) as without a sidecar.
Without sidecars nothing changes (tested byte for byte on the synthetic clip).

Every check that fails raises `SidecarError` (a ValueError; the export CLI exits 2 on it and only on it): a refused
screen window; a window whose rest verdict (`p7_ruled_pass`) is not True; a start pose farther from
the contact-onset pose than the noise bound (`MAX_CORRECTION_M` / `MAX_CORRECTION_DEG`: checked on the
record's own `start_vs_c0` here and on the correction the export computes in `kinematic_task`); a record or plane
without its `window`; a pose whose contact-onset frame or world frame (static camera vs the same `ego_motion.npz`, by
sha256) is not the export's; a plane whose camera is not the export's world; an accepted plane without a start pose; a
SNAPPED pose without its (accepted) support plane, or with another plane file than the one it was snapped on.

**Rest verdict.** The robot path does not recompute the rest check: it reads the start-pose record's `p7_ruled_pass`,
whatever reading produced it. Anything but `True` refuses; the record's `amendment` names the reading and the package
reports it.

**Correction bound.** `W = P inv(O_c0)` may move the onset pose by at most 8.6 cm (object-frame origin) and 12.2
deg: the 95th percentiles of a single row's pose-error norm under the calibrated iid noise model (30.6 mm / 4.37 deg
per axis, times the chi(3) 95% point 2.795). A larger difference is real motion or a tracker failure, not noise.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["MAX_CORRECTION_DEG", "MAX_CORRECTION_M", "MAX_NORMAL_SD_DEG", "MIN_PLANE_ROWS", "START_POSE_VERSION",
           "SidecarError", "StartPoseOverride", "SupportPlaneGravity", "check_correction", "file_sha256",
           "load_start_pose", "load_support_plane"]

START_POSE_VERSION = 1
MIN_PLANE_ROWS = 2              # a plane from one lite row is reported but REFUSED
MAX_NORMAL_SD_DEG = 3.0         # the plane recipe's normal agreement across rows (section 5a)
MAX_CORRECTION_M = 0.086        # 2.795 x 30.6 mm (95th pct of a row's position-error norm)
MAX_CORRECTION_DEG = 12.2       # 2.795 x 4.37 deg (95th pct of a row's rotation-error norm)


class SidecarError(ValueError):
    """A start-pose / support-plane sidecar check refused (module docstring); the export CLI exits 2."""


def check_correction(disp_m: object, rot_deg: object, what: str) -> None:
    """Refuse a start pose more than `MAX_CORRECTION_M` / `MAX_CORRECTION_DEG` from the onset pose (a
    missing or non-finite value refuses too); the message names the value."""
    num = (int, float)
    d = float(disp_m) if isinstance(disp_m, num) and not isinstance(disp_m, bool) else float("nan")
    a = float(rot_deg) if isinstance(rot_deg, num) and not isinstance(rot_deg, bool) else float("nan")
    if not (d <= MAX_CORRECTION_M and a <= MAX_CORRECTION_DEG):
        raise SidecarError(f"{what} is {d * 100:.1f} cm / {a:.1f} deg from the contact-onset pose; the export allows "
                           f"at most {MAX_CORRECTION_M * 100:.1f} cm and {MAX_CORRECTION_DEG:g} deg (the 95th "
                           "percentile of a single row's pose noise): more is real motion or a tracker failure")


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class SupportPlaneGravity:
    """`support_plane.json` as the robot path reads it."""
    path: str
    sha256: str
    window: str | None
    gravity_cam: np.ndarray | None   # (3,) unit up in row 0's camera, None when the fit had no row
    rows: int                        # rest rows the plane was fitted on
    normal_sd_deg: float | None
    accepted: bool                   # the file's own acceptance AND the row rule (>= 2 rows, sd <= 3 deg)
    refusal: tuple[str, ...]         # why not (the file's flags, then the row rule's)
    camera: str | None

    def to_dict(self) -> dict:
        return {"path": self.path, "sha256": self.sha256, "window": self.window, "accepted": self.accepted,
                "refusal": list(self.refusal), "rows": self.rows, "normal_sd_deg": self.normal_sd_deg,
                "gravity_cam": None if self.gravity_cam is None else self.gravity_cam.tolist(), "camera": self.camera,
                "rule": f"file accepted AND >= {MIN_PLANE_ROWS} rest rows AND normal sd <= {MAX_NORMAL_SD_DEG:g} deg"}


def load_support_plane(path: str | Path) -> SupportPlaneGravity:
    """Read `support_plane.json` (module docstring); raises `SidecarError` when it is not one."""
    p = Path(path)
    try:
        rec = json.loads(p.read_text())
    except (OSError, ValueError) as exc:
        raise SidecarError(f"{p}: not a readable support_plane.json ({exc})") from exc
    if not isinstance(rec, dict) or "accepted" not in rec or "rows" not in rec:
        raise SidecarError(f"{p}: not a support_plane.json (no 'accepted' / 'rows')")
    g = rec.get("gravity_cam")
    grav = None
    if g is not None:
        grav = np.asarray(g, dtype=np.float64)
        if grav.shape != (3,) or not np.isfinite(grav).all() or np.linalg.norm(grav) < 1e-9:
            raise SidecarError(f"{p}: gravity_cam must be a finite non-zero 3-vector")
        grav = grav / np.linalg.norm(grav)
    rows = len(rec.get("rows") or [])
    sd = rec.get("normal_sd_deg")
    sd = None if sd is None else float(sd)
    refusal = [f"file: {r}" for r in (rec.get("refusal") or [])]
    if rec.get("accepted") is not True and not refusal:
        refusal.append("file: not accepted")
    if rows < MIN_PLANE_ROWS:
        refusal.append(f"plane rows: {rows} rest row(s) < {MIN_PLANE_ROWS}")
    if sd is None or not sd <= MAX_NORMAL_SD_DEG:
        refusal.append(f"plane normals: normal agreement sd {sd} deg not <= {MAX_NORMAL_SD_DEG:g}")
    if grav is None:
        refusal.append("no gravity_cam")
    return SupportPlaneGravity(path=str(p), sha256=file_sha256(p), window=rec.get("window"), gravity_cam=grav,
                               rows=rows, normal_sd_deg=sd, accepted=not refusal, refusal=tuple(refusal),
                               camera=rec.get("camera"))


@dataclass(frozen=True)
class StartPoseOverride:
    """`start_pose_a1.json` as the robot path reads it (module docstring)."""
    path: str
    sha256: str
    window: str | None
    pose_world: np.ndarray           # (4,4) metric object frame -> the video world of `world`
    kind: str                        # "consolidated" | "snapped"
    source: str                      # the screen's start-pose source ("fallback": median + chordal L1 mean; "299")
    c0_frame_idx: int                # source frame of the contact-onset row the pose replaces
    world_camera: str                # "ego_motion" | "static"
    ego_motion_sha256: str | None
    snap: dict | None                # the screen's snap record (face, rotation, displacement) when snapped
    caps: dict                       # the screen's snap caps
    support_plane_sha256: str | None
    start_vs_c0: dict | None         # the screen's distance of the consolidated pose from the c0-row pose
    p7_ruled_pass: bool = False      # the record's ruled rest verdict (only True is loaded)
    amendment: str | None = None     # the record's amendment string: which rest-check reading produced the verdict

    def to_dict(self) -> dict:
        return {"path": self.path, "sha256": self.sha256, "window": self.window, "kind": self.kind,
                "source": self.source, "c0_frame_idx": self.c0_frame_idx, "world_camera": self.world_camera,
                "ego_motion_sha256": self.ego_motion_sha256, "snap": self.snap, "caps": self.caps,
                "support_plane_sha256": self.support_plane_sha256, "start_vs_c0": self.start_vs_c0,
                "p7_ruled_pass": self.p7_ruled_pass, "amendment": self.amendment,
                "pose_world": self.pose_world.tolist()}


def _rigid(m: object, path: Path) -> np.ndarray:
    a = np.asarray(m, dtype=np.float64)
    if a.shape != (4, 4) or not np.isfinite(a).all():
        raise SidecarError(f"{path}: pose_world must be a finite 4x4 matrix")
    r = a[:3, :3]
    if not (np.allclose(r.T @ r, np.eye(3), atol=1e-6) and abs(np.linalg.det(r) - 1.0) < 1e-6
            and np.allclose(a[3], [0.0, 0.0, 0.0, 1.0])):
        raise SidecarError(f"{path}: pose_world is not a rigid transform")
    return a


def load_start_pose(path: str | Path) -> StartPoseOverride:
    """Read `start_pose_a1.json` (module docstring); raises `SidecarError` on a refused or malformed record."""
    p = Path(path)
    try:
        rec = json.loads(p.read_text())
    except (OSError, ValueError) as exc:
        raise SidecarError(f"{p}: not a readable start_pose_a1.json ({exc})") from exc
    if not isinstance(rec, dict) or rec.get("kind") != "start_pose_a1" or rec.get("version") != START_POSE_VERSION:
        raise SidecarError(f"{p}: not a start_pose_a1.json of version {START_POSE_VERSION}")
    if rec.get("refused") is not False:
        raise SidecarError(f"{p}: the start-pose screen refused this window's start pose ({rec.get('refusal')}); it "
                           "cannot be exported")
    if not isinstance(rec.get("window"), str):
        raise SidecarError(f"{p}: the record names no window, so it cannot be matched to the export's")
    if rec.get("p7_ruled_pass") is not True:
        raise SidecarError(f"{p}: the record's ruled rest verdict p7_ruled_pass is {rec.get('p7_ruled_pass')!r}, not "
                           f"True ({rec.get('amendment')}): only a window whose object rests before contact is "
                           "exported")
    vs = rec.get("start_vs_c0")
    vs = vs if isinstance(vs, dict) else {}
    check_correction(vs.get("disp_m"), vs.get("rot_deg"), f"{p}: the record's start pose (start_vs_c0)")
    pose_kind = rec.get("pose")
    if pose_kind not in ("consolidated", "snapped"):
        raise SidecarError(f"{p}: pose must be 'consolidated' or 'snapped', got {pose_kind!r}")
    world = rec.get("world") or {}
    camera = world.get("camera")
    if camera not in ("ego_motion", "static"):
        raise SidecarError(f"{p}: world.camera must be 'ego_motion' or 'static', got {camera!r}")
    ego_sha = world.get("ego_motion_sha256")
    if (camera == "ego_motion") != isinstance(ego_sha, str):
        raise SidecarError(f"{p}: world.ego_motion_sha256 must be given exactly when the camera is the ego-motion")
    plane = rec.get("support_plane") or {}
    plane_sha = plane.get("sha256")
    if pose_kind == "snapped" and (plane.get("status") != "accepted" or not isinstance(plane_sha, str)):
        raise SidecarError(f"{p}: a snapped pose needs the accepted support plane it was snapped on (sha256)")
    c0 = rec.get("c0_frame_idx")
    if not isinstance(c0, int) or isinstance(c0, bool):
        raise SidecarError(f"{p}: c0_frame_idx must be an integer source frame")
    return StartPoseOverride(path=str(p), sha256=file_sha256(p), window=rec.get("window"),
                             pose_world=_rigid(rec.get("pose_world"), p), kind=pose_kind,
                             source=str(rec.get("start_pose_source")), c0_frame_idx=c0, world_camera=camera,
                             ego_motion_sha256=ego_sha, snap=rec.get("snap"), caps=dict(rec.get("caps") or {}),
                             support_plane_sha256=plane_sha if isinstance(plane_sha, str) else None,
                             start_vs_c0=rec.get("start_vs_c0"), p7_ruled_pass=True,
                             amendment=None if rec.get("amendment") is None else str(rec.get("amendment")))
