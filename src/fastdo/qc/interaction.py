"""Window-level, label-free INTERACTION features: does the camera wearer's hand ever touch the chosen object?

**Why.** About 34 % of the narrated, prompt-ACCEPTED Ego4D windows of one production run show no object
handling by the wearer (driving, shelf browsing, adjusting the camera), that QC accepted 14 of those 15, and that the
per-solved headline counts them as solved. QC's five features measure fit quality, not whether an interaction
happened, so nothing in the pipeline asks the question this module asks.

**The per-frame signal is the contact ruler's `C_min`** (`fastdo.contact.contact_correction`'s `min_cost_mm`): slide the
solved object along the ray to its predicted centroid and take the smallest mean hand-joint-to-surface distance reached
anywhere on the slide. The slide makes it invariant to the along-ray placement error of the object (and, equivalently,
of the hand), which is the error that dominates monocular solves. A frame is TOUCHING when `C_min <= 56 mm`: the touch
gate fixed on DexYCB GT-contact frames only (subjects 03-06; 90.3 % of contact frames kept, 97.5 % of non-contact frames
rejected, AUC 0.984), which transferred to egocentric HOI4D at AUC 0.955 / 0.949.

**Differences from the contact stage, and why.** The contact stage uses ONE hand (`choose_side`) because it corrects
the object with that hand. An interaction can be done with either hand, so this module evaluates BOTH sides and takes
the per-frame minimum over the sides that are valid. The profile is the same `contact_profile` quantity, evaluated
with one KD-tree query for the whole slide grid instead of one per grid point (same distances; the mean over joints
may differ from `contact_profile` in the last ulp, which is why the two are compared with a tolerance in the tests,
never assumed bit-identical).

Nothing here is fitted: the gate is the contact ruler's, and a screen threshold is a caller's (config) choice.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from fastdo.contact import DEFAULT_GRID_MM

__all__ = [
    "DEFAULT_TOUCH_GATE_MM",
    "SCREEN_FEATURES",
    "InteractionFeatures",
    "InteractionScreenConfig",
    "frame_min_contact",
    "interaction_features",
    "longest_run",
    "screen_passes",
]

DEFAULT_TOUCH_GATE_MM = 56.0          # the touch gate (module docstring)
SIDES: tuple[str, ...] = ("left", "right")
SCREEN_FEATURES: tuple[str, ...] = ("touch_frac", "longest_touch_run_frac")


def frame_min_contact(poses: np.ndarray, scale: float, mesh_points: np.ndarray, joints: np.ndarray,
                      valid: np.ndarray, grid_mm: np.ndarray | None = None) -> np.ndarray:
    """Per-frame `C_min` in mm for ONE hand: `min_delta mean_j dist(joint_j - delta * ray, surface)`.

    Args:
        poses: `(T,4,4)` solved object poses (camera frame).
        scale: The solve's scale, applied to `mesh_points` as the solver applies it.
        mesh_points: `(M,3)` template surface samples in the object frame.
        joints: `(T,J,3)` hand joints in the camera frame; non-finite rows are ignored.
        valid: `(T,)` bool, whether that frame has a usable hand.
        grid_mm: Slide grid in mm; default `fastdo.contact.DEFAULT_GRID_MM` (-300..300 by 2).

    Returns:
        `(T,)` float64; NaN where the hand is invalid, has no finite joint, or the object centroid has no ray.

    Raises:
        ValueError: On shape mismatches.
    """
    poses = np.asarray(poses, dtype=float)
    mesh = np.asarray(mesh_points, dtype=float)
    joints = np.asarray(joints, dtype=float)
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"poses must be (T,4,4), got {poses.shape}")
    if mesh.ndim != 2 or mesh.shape[1] != 3:
        raise ValueError(f"mesh_points must be (M,3), got {mesh.shape}")
    t_rows = poses.shape[0]
    if joints.ndim != 3 or joints.shape[0] != t_rows or joints.shape[2] != 3 or valid.shape[0] != t_rows:
        raise ValueError(f"joints {joints.shape} and valid {valid.shape} must have {t_rows} rows of (J,3)")
    grid_m = (DEFAULT_GRID_MM if grid_mm is None else np.asarray(grid_mm, dtype=float)) / 1e3
    centroid = mesh.mean(axis=0)
    out = np.full(t_rows, np.nan, dtype=float)
    for t in range(t_rows):
        if not valid[t]:
            continue
        j = joints[t][np.isfinite(joints[t]).all(axis=1)]
        if j.size == 0:
            continue
        rot, trans = poses[t, :3, :3], poses[t, :3, 3]
        cp = scale * (rot @ centroid) + trans
        norm = float(np.linalg.norm(cp))
        if not np.isfinite(norm) or norm <= 0:            # a centroid at or behind the camera has no ray
            continue
        ray = cp / norm
        surface = scale * (mesh @ rot.T) + trans
        queries = j[None, :, :] - grid_m[:, None, None] * ray[None, None, :]       # (K,J,3)
        dist = cKDTree(surface).query(queries.reshape(-1, 3), k=1)[0].reshape(grid_m.size, j.shape[0])
        out[t] = float(dist.mean(axis=1).min()) * 1e3
    return out


def longest_run(flags: np.ndarray) -> int:
    """Length of the longest run of True in a 1-D bool array (0 for none)."""
    best = cur = 0
    for f in np.asarray(flags, dtype=bool).reshape(-1):
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


@dataclass(frozen=True)
class InteractionFeatures:
    """Window-level interaction features (module docstring). Fractions are over ALL `n_frames` rows unless named.

    Attributes:
        n_frames: Rows of the solve.
        hand_valid_frac: Rows where at least one side has a finite `C_min`.
        touch_frac: Rows with `C_min <= touch_gate_mm` on either hand (a row with no finite `C_min` is not touching).
        touch_frac_of_valid: The same count over `hand_valid_frac`'s rows (NaN when none).
        longest_touch_run: Longest run of consecutive touching rows.
        longest_touch_run_frac: `longest_touch_run / n_frames`.
        cmin_median_mm, cmin_p10_mm: Median and 10th percentile of the per-row either-hand `C_min` (NaN when none).
        touch_frac_left, touch_frac_right: `touch_frac` per side.
        obj_centroid_range_rel: Largest distance of the object's camera-frame centroid from its median position,
            divided by the object's diameter. CAMERA frame: head motion moves it too (there is no camera
            trajectory in the artifacts), so it is descriptive, not a world-motion measurement.
        obj_rotation_range_deg: Largest rotation angle of the object (camera frame) from its median-frame rotation;
            same caveat.
        touch_gate_mm: The gate used.
    """

    n_frames: int
    hand_valid_frac: float
    touch_frac: float
    touch_frac_of_valid: float
    longest_touch_run: int
    longest_touch_run_frac: float
    cmin_median_mm: float
    cmin_p10_mm: float
    touch_frac_left: float
    touch_frac_right: float
    obj_centroid_range_rel: float
    obj_rotation_range_deg: float
    touch_gate_mm: float


def _camera_motion(poses: np.ndarray, scale: float, mesh: np.ndarray) -> tuple[float, float]:
    centroid = mesh.mean(axis=0)
    diameter = abs(float(scale)) * float(np.linalg.norm(mesh.max(axis=0) - mesh.min(axis=0)))
    cps = scale * np.einsum("tij,j->ti", poses[:, :3, :3], centroid) + poses[:, :3, 3]
    ok = np.isfinite(cps).all(axis=1)
    if not ok.any() or not diameter > 0:
        return math.nan, math.nan
    med = np.median(cps[ok], axis=0)
    rng = float(np.linalg.norm(cps[ok] - med, axis=1).max()) / diameter
    rots = poses[ok, :3, :3]
    ref = rots[int(np.argmin(np.linalg.norm(cps[ok] - med, axis=1)))]
    cosang = (np.einsum("tij,ij->t", rots, ref) - 1.0) / 2.0          # trace(R_t ref^T) = sum_ij R_t[i,j] ref[i,j]
    ang = float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))).max())
    return rng, ang


def interaction_features(poses: np.ndarray, scale: float, mesh_points: np.ndarray,
                         joints: Mapping[str, np.ndarray], valid: Mapping[str, np.ndarray],
                         touch_gate_mm: float = DEFAULT_TOUCH_GATE_MM,
                         grid_mm: np.ndarray | None = None) -> tuple[InteractionFeatures, dict[str, np.ndarray]]:
    """The window's `InteractionFeatures` and the per-row arrays they came from.

    Args:
        poses, scale, mesh_points: The solve and the shape stage's samples (as `frame_min_contact`).
        joints: `side -> (T,J,3)` camera-frame joints, both sides of `hands.npz`.
        valid: `side -> (T,)` bool.
        touch_gate_mm: Per-row touching threshold on `C_min` (mm).
        grid_mm: Slide grid (mm).

    Returns:
        `(features, per_row)` with `per_row = {"cmin_left", "cmin_right", "cmin"}` (T,) float64 mm.

    Raises:
        ValueError: On a non-positive gate or shape mismatches.
    """
    if not (np.isfinite(touch_gate_mm) and touch_gate_mm > 0):
        raise ValueError(f"touch_gate_mm must be a positive finite number, got {touch_gate_mm!r}")
    poses = np.asarray(poses, dtype=float)
    mesh = np.asarray(mesh_points, dtype=float)
    per_row = {f"cmin_{s}": frame_min_contact(poses, scale, mesh, joints[s], valid[s], grid_mm) for s in SIDES}
    stack = np.stack([per_row[f"cmin_{s}"] for s in SIDES])
    finite = np.isfinite(stack).any(axis=0)
    cmin = np.full(poses.shape[0], np.nan)
    cmin[finite] = np.nanmin(stack[:, finite], axis=0)
    per_row["cmin"] = cmin
    n = int(poses.shape[0])
    touching = np.isfinite(cmin) & (cmin <= touch_gate_mm)
    run = longest_run(touching)
    n_valid = int(finite.sum())
    side_touch = {s: float(np.mean(np.isfinite(per_row[f"cmin_{s}"]) & (per_row[f"cmin_{s}"] <= touch_gate_mm)))
                  for s in SIDES}
    rng, ang = _camera_motion(poses, scale, mesh)
    feats = InteractionFeatures(
        n_frames=n, hand_valid_frac=n_valid / n, touch_frac=float(touching.sum()) / n,
        touch_frac_of_valid=float(touching.sum()) / n_valid if n_valid else math.nan,
        longest_touch_run=run, longest_touch_run_frac=run / n,
        cmin_median_mm=float(np.median(cmin[finite])) if n_valid else math.nan,
        cmin_p10_mm=float(np.percentile(cmin[finite], 10)) if n_valid else math.nan,
        touch_frac_left=side_touch["left"], touch_frac_right=side_touch["right"],
        obj_centroid_range_rel=rng, obj_rotation_range_deg=ang, touch_gate_mm=float(touch_gate_mm))
    return feats, per_row


@dataclass(frozen=True)
class InteractionScreenConfig:
    """`qc_stage.interaction_screen`: an OPT-IN rejection of windows where the hand never touches the object.

    Attributes:
        enabled: Off by default; off leaves the qc stage's output byte-identical (no key is added to `qc.json`).
        feature: Which `InteractionFeatures` field the rule thresholds (`SCREEN_FEATURES`).
        min_value: A window passes iff `feature >= min_value`. **null refuses to build when enabled**: there is
            no default threshold, as `qc_stage.rule_path` and `prompt_stage.min_score` have none.
        touch_gate_mm: Per-row touching threshold on `C_min` (the 56 mm touch gate).
    """

    enabled: bool = False
    feature: str = "touch_frac"
    min_value: float | None = None
    touch_gate_mm: float = DEFAULT_TOUCH_GATE_MM

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError(f"interaction_screen.enabled must be a bool, got {self.enabled!r}")  # noqa: TRY004
        if self.feature not in SCREEN_FEATURES:
            raise ValueError(f"interaction_screen.feature must be one of {SCREEN_FEATURES}, got {self.feature!r}")
        if not (np.isfinite(self.touch_gate_mm) and self.touch_gate_mm > 0):
            raise ValueError(f"interaction_screen.touch_gate_mm must be positive, got {self.touch_gate_mm!r}")
        if self.min_value is not None and not (np.isfinite(self.min_value) and 0.0 <= self.min_value <= 1.0):
            raise ValueError(f"interaction_screen.min_value must be in [0, 1] or null, got {self.min_value!r}")
        if self.enabled and self.min_value is None:
            raise ValueError("interaction_screen.enabled is true but min_value is null: set the pre-registered "
                             "threshold; there is no default")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object] | None) -> InteractionScreenConfig:
        """Builds from a config mapping; an unknown key raises rather than being ignored."""
        if not mapping:
            return cls()
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted({str(k) for k in mapping} - known)
        if unknown:
            raise ValueError(f"interaction_screen has unknown keys {unknown}; known keys: {sorted(known)}")
        data = dict(mapping)
        for key in ("min_value", "touch_gate_mm"):
            if data.get(key) is not None:
                data[key] = float(data[key])  # type: ignore[arg-type]
        return cls(**data)  # type: ignore[arg-type]


def screen_passes(features: InteractionFeatures, cfg: InteractionScreenConfig) -> bool:
    """Whether a window passes the screen: `getattr(features, cfg.feature) >= cfg.min_value`.

    Raises:
        ValueError: If `cfg.min_value` is None, or the features were computed with a different gate.
    """
    if cfg.min_value is None:
        raise ValueError("interaction_screen.min_value is null")
    if features.touch_gate_mm != cfg.touch_gate_mm:
        raise ValueError(f"features used touch gate {features.touch_gate_mm} mm, the screen {cfg.touch_gate_mm} mm")
    return bool(float(getattr(features, cfg.feature)) >= float(cfg.min_value))
