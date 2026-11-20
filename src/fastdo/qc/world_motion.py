"""WORLD-frame object motion features of a window: does the chosen object move relative to the static scene?

**Why.** Contact cannot tell "operating a fixed control" (driving: the hand rests on a lever, touch_frac
~1) from "manipulating a portable object", and no stored feature separates handling from non-handling windows on
Ego4D. A manipulated object MOVES relative to the static scene; a vehicle control, a shelf item looked at, a fixture
does not. Object poses are solved in each row's camera frame, so the camera trajectory is needed to ask it: that is
`ego_motion.npz` (`fastdo.geometry.ego_motion`, written by the opt-in `geometry_stage.ego_motion`).

**Composition.** Row t's object centroid in its camera frame is c_t = scale R_t m + t_t (m the template samples'
centroid, the solve's pose and scale). With the ego-motion's camera-to-world Sim(3) (s_t, Q_t, u_t) of that row,
c^w_t = s_t Q_t c_t + u_t, and the object's world rotation is Q_t R_t. The per-row metric scale s_t applies to the
object too: its depth came from the same pointmap. Hand joints are composed the same way.

**Rows used.** Rows on the ego trajectory (`valid`), with a finite pose and solve confidence > 0: ~30% of internet rows
have no pointmap and no track, and the solver extrapolates them, sometimes behind the camera.

**Features** (robust extents, so a few bad rows cannot make a window "move"):
  obj_world_disp_mm       p90 over used rows of |c^w_t - median_t c^w_t|
  obj_world_perp_disp_mm  the same, with each displacement's component along the row's viewing ray removed (the
                          solve's along-ray error is 4-10x its lateral error)
  obj_world_rot_deg       p90 rotation of the object's world orientation from the row nearest the median position
  obj_world_disp_rel      obj_world_disp_mm over the object's diameter
  obj_cam_disp_mm, obj_cam_rot_deg   the same statistics in the CAMERA frame (head motion included)
  obj_moving_frac         rows whose world displacement over +-`step_rows` exceeds `move_gate_mm`
  comotion                cosine of the object's and the nearest hand's world displacements over +-`step_rows`,
                          summed over rows where both exist (1 = they move together)
  hand_world_disp_mm      p90 displacement of the hand centroid (both sides pooled per row) in the world frame
  cam_path_mm, cam_rot_deg  the camera's own motion (context)
  used_frac               used rows / all rows; ego_ok the ego-motion self-check.
Nothing is fitted here; a screen threshold is a caller's choice.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

__all__ = [
    "WORLD_SCREEN_FEATURES",
    "WorldMotionFeatures",
    "WorldMotionScreenConfig",
    "world_motion_features",
    "world_screen_decision",
]

WORLD_SCREEN_FEATURES: tuple[str, ...] = ("obj_world_disp_mm", "obj_world_perp_disp_mm", "obj_world_rot_deg",
                                          "obj_moving_frac", "obj_world_disp_rel")


@dataclass(frozen=True)
class WorldMotionFeatures:
    """Window-level world-frame motion features (module docstring). NaN where no row is usable."""

    n_rows: int
    used_frac: float
    ego_ok: bool
    obj_world_disp_mm: float
    obj_world_perp_disp_mm: float
    obj_world_rot_deg: float
    obj_world_disp_rel: float
    obj_cam_disp_mm: float
    obj_cam_rot_deg: float
    obj_moving_frac: float
    comotion: float
    hand_world_disp_mm: float
    cam_path_mm: float
    cam_rot_deg: float


def _p90_disp(p: np.ndarray) -> float:
    return float(np.percentile(np.linalg.norm(p - np.median(p, axis=0), axis=1), 90)) * 1e3 if len(p) else math.nan


def _rot_p90(rots: np.ndarray, pos: np.ndarray) -> float:
    if not len(rots):
        return math.nan
    ref = rots[int(np.argmin(np.linalg.norm(pos - np.median(pos, axis=0), axis=1)))]
    cos = (np.einsum("tij,ij->t", rots, ref) - 1.0) / 2.0
    return float(np.percentile(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))), 90))


def _orthonormal(m: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(m)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    return r


def world_motion_features(poses: np.ndarray, scale: float, mesh_points: np.ndarray, confidence: np.ndarray,
                          cam_to_world: np.ndarray, ego_valid: np.ndarray, ego_ok: bool,
                          joints: Mapping[str, np.ndarray] | None = None,
                          joint_valid: Mapping[str, np.ndarray] | None = None,
                          step_rows: int = 5, move_gate_mm: float = 30.0) -> tuple[WorldMotionFeatures, dict]:
    """The window's `WorldMotionFeatures` and per-row arrays (`obj_world` (T,3) m, `used` (T,) bool).

    Args:
        poses: (T,4,4) solved object poses (camera frame), `scale` the solve's scale, `mesh_points` (M,3) template.
        confidence: (T,) the solve's per-row confidence (rows with 0 are not used).
        cam_to_world: (T,4,4) the ego-motion's Sim(3) per row; `ego_valid` (T,) bool; `ego_ok` its self-check.
        joints, joint_valid: side -> (T,J,3) camera-frame joints and (T,) validity (optional).
        step_rows: Half-width (rows) of the displacement used by `obj_moving_frac` and `comotion`.
        move_gate_mm: The displacement over +-`step_rows` that counts a row as moving.

    Raises:
        ValueError: On shape mismatches.
    """
    poses = np.asarray(poses, dtype=np.float64)
    c2w = np.asarray(cam_to_world, dtype=np.float64)
    t_rows = poses.shape[0]
    if poses.shape[1:] != (4, 4) or c2w.shape != (t_rows, 4, 4):
        raise ValueError(f"poses {poses.shape} and cam_to_world {c2w.shape} must be (T,4,4) with the same T")
    mesh = np.asarray(mesh_points, dtype=np.float64)
    conf = np.asarray(confidence, dtype=np.float64).reshape(-1)
    ego_valid = np.asarray(ego_valid, dtype=bool).reshape(-1)
    if conf.shape[0] != t_rows or ego_valid.shape[0] != t_rows:
        raise ValueError("confidence and ego_valid need one entry per row")
    centroid = mesh.mean(axis=0)
    diameter = abs(float(scale)) * float(np.linalg.norm(mesh.max(axis=0) - mesh.min(axis=0)))
    cam_c = scale * np.einsum("tij,j->ti", poses[:, :3, :3], centroid) + poses[:, :3, 3]
    used = (ego_valid & (conf > 0) & np.isfinite(cam_c).all(axis=1) & np.isfinite(c2w).all(axis=(1, 2))
            & (cam_c[:, 2] > 0))
    world_c = np.full((t_rows, 3), np.nan)
    world_r = np.full((t_rows, 3, 3), np.nan)
    perp = np.full((t_rows, 3), np.nan)
    for t in np.nonzero(used)[0]:
        sq = c2w[t, :3, :3]
        world_c[t] = sq @ cam_c[t] + c2w[t, :3, 3]
        q = _orthonormal(sq)
        world_r[t] = q @ _orthonormal(poses[t, :3, :3])
    idx = np.nonzero(used)[0]
    if len(idx):
        med = np.median(world_c[idx], axis=0)
        for t in idx:
            ray = c2w[t, :3, :3] @ (cam_c[t] / np.linalg.norm(cam_c[t]))
            ray = ray / np.linalg.norm(ray)
            d = world_c[t] - med
            perp[t] = d - (d @ ray) * ray
    hand_w = np.full((t_rows, 3), np.nan)
    if joints is not None and joint_valid is not None:
        for t in np.nonzero(ego_valid & np.isfinite(c2w).all(axis=(1, 2)))[0]:
            pts = [np.asarray(joints[s][t], float) for s in joints if bool(np.asarray(joint_valid[s])[t])]
            pts = [p[np.isfinite(p).all(axis=1)] for p in pts]
            pts = [p for p in pts if len(p)]
            if pts:
                hc = np.concatenate(pts).mean(axis=0)
                hand_w[t] = c2w[t, :3, :3] @ hc + c2w[t, :3, 3]
    moving, co_num, co_o, co_h = [], 0.0, 0.0, 0.0
    for t in idx:
        a, b = t - step_rows, t + step_rows
        if a < 0 or b >= t_rows or not (used[a] and used[b]):
            continue
        d_o = world_c[b] - world_c[a]
        moving.append(np.linalg.norm(d_o) * 1e3 > move_gate_mm)
        if np.isfinite(hand_w[a]).all() and np.isfinite(hand_w[b]).all():
            d_h = hand_w[b] - hand_w[a]
            co_num += float(d_o @ d_h)
            co_o += float(d_o @ d_o)
            co_h += float(d_h @ d_h)
    cam_rows = np.nonzero(ego_valid & np.isfinite(c2w).all(axis=(1, 2)))[0]
    cam_t = c2w[cam_rows, :3, 3]
    cam_path = float(np.linalg.norm(np.diff(cam_t, axis=0), axis=1).sum()) * 1e3 if len(cam_rows) > 1 else math.nan
    cam_rot = (_rot_p90(np.stack([_orthonormal(c2w[t, :3, :3]) for t in cam_rows]), cam_t)
               if len(cam_rows) else math.nan)
    hw = hand_w[np.isfinite(hand_w).all(axis=1)]
    disp = _p90_disp(world_c[idx])
    feats = WorldMotionFeatures(
        n_rows=int(t_rows), used_frac=float(used.mean()) if t_rows else 0.0, ego_ok=bool(ego_ok),
        obj_world_disp_mm=disp,
        obj_world_perp_disp_mm=float(np.percentile(np.linalg.norm(perp[idx], axis=1), 90)) * 1e3 if len(idx)
        else math.nan,
        obj_world_rot_deg=_rot_p90(world_r[idx], world_c[idx]),
        obj_world_disp_rel=disp / (diameter * 1e3) if diameter > 0 and math.isfinite(disp) else math.nan,
        obj_cam_disp_mm=_p90_disp(cam_c[idx]),
        obj_cam_rot_deg=_rot_p90(np.stack([_orthonormal(poses[t, :3, :3]) for t in idx]) if len(idx)
                                 else np.zeros((0, 3, 3)), cam_c[idx]),
        obj_moving_frac=float(np.mean(moving)) if moving else math.nan,
        comotion=co_num / math.sqrt(co_o * co_h) if co_o > 0 and co_h > 0 else math.nan,
        hand_world_disp_mm=_p90_disp(hw), cam_path_mm=cam_path, cam_rot_deg=cam_rot)
    return feats, {"obj_world": world_c, "used": used, "hand_world": hand_w}


@dataclass(frozen=True)
class WorldMotionScreenConfig:
    """`qc_stage.world_motion_screen`: an OPT-IN rejection of windows whose object does not move in the world.

    Attributes:
        enabled: Off by default; off leaves `qc.json` byte-identical (no key added, `ego_motion.npz` not read).
        feature: The `WorldMotionFeatures` field thresholded (`WORLD_SCREEN_FEATURES`).
        min_value: Keep iff `feature >= min_value`. **null refuses to build when enabled** (no default threshold).
        min_used_frac: Abstain (keep, flagged) when fewer rows than this are usable.
        abstain_on_bad_ego: Abstain (keep, flagged) when the ego-motion self-check failed.
        step_rows, move_gate_mm: As `world_motion_features`.
    """

    enabled: bool = False
    feature: str = "obj_world_disp_mm"
    min_value: float | None = None
    min_used_frac: float = 0.3
    abstain_on_bad_ego: bool = True
    step_rows: int = 5
    move_gate_mm: float = 30.0

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool) or not isinstance(self.abstain_on_bad_ego, bool):
            raise ValueError("world_motion_screen.enabled / abstain_on_bad_ego must be bools")  # noqa: TRY004
        if self.feature not in WORLD_SCREEN_FEATURES:
            raise ValueError(f"world_motion_screen.feature must be one of {WORLD_SCREEN_FEATURES}, got {self.feature!r}")
        if self.min_value is not None and not math.isfinite(float(self.min_value)):
            raise ValueError(f"world_motion_screen.min_value must be finite or null, got {self.min_value!r}")
        if not 0.0 <= float(self.min_used_frac) <= 1.0:
            raise ValueError(f"world_motion_screen.min_used_frac must be in [0, 1], got {self.min_used_frac!r}")
        if not isinstance(self.step_rows, int) or self.step_rows < 1 or not float(self.move_gate_mm) > 0:
            raise ValueError("world_motion_screen.step_rows must be an int >= 1 and move_gate_mm positive")
        if self.enabled and self.min_value is None:
            raise ValueError("world_motion_screen.enabled is true but min_value is null: set the pre-registered "
                             "threshold; there is no default")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object] | None) -> WorldMotionScreenConfig:
        """From a config mapping; an unknown key raises."""
        if not mapping:
            return cls()
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted({str(k) for k in mapping} - known)
        if unknown:
            raise ValueError(f"world_motion_screen has unknown keys {unknown}; known keys: {sorted(known)}")
        data = {str(k): mapping[k] for k in mapping}
        for key in ("min_value", "min_used_frac", "move_gate_mm"):
            if data.get(key) is not None:
                data[key] = float(data[key])  # type: ignore[arg-type]
        if "step_rows" in data:
            data["step_rows"] = int(data["step_rows"])  # type: ignore[arg-type]
        return cls(**data)  # type: ignore[arg-type]


def world_screen_decision(features: WorldMotionFeatures, cfg: WorldMotionScreenConfig) -> tuple[bool, str]:
    """`(keep, reason)`: reason is "pass", "fail", "abstain_bad_ego" or "abstain_few_rows" (abstentions keep)."""
    if cfg.min_value is None:
        raise ValueError("world_motion_screen.min_value is null")
    if cfg.abstain_on_bad_ego and not features.ego_ok:
        return True, "abstain_bad_ego"
    value = float(getattr(features, cfg.feature))
    if features.used_frac < cfg.min_used_frac or not math.isfinite(value):
        return True, "abstain_few_rows"
    return (True, "pass") if value >= float(cfg.min_value) else (False, "fail")
