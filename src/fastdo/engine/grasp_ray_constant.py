"""Per-clip grasp-onset along-ray constant (opt-in `solve_stage.grasp_ray_constant`, default null = off).

**Why.** On an OakInk2 egocentric set, physics success separates on the hand->object gap along the camera ray at the
grasp. The PER-FRAME contact-ray smoother (`contact_ray_smoother`) closed that gap and the start gap closed (DO AS I DO
start distance 5.7 -> 1.5 mm), but its per-frame moves chased the hand and CANCELLED real world motion of the object
(reference displacement ratio to MoCap 0.79 -> 0.69), so the strict physics count did not rise. This module closes the
same gap with ONE scalar per clip, so the object's own trajectory (its image evidence) keeps its shape.

**Measurement (grasp onset).** Rows are gated by the contact-ray smoother's `C_min` (the 21-joint mean slide,
`<= touch_gate_mm`, as in that smoother). On a gated row, every MANO vertex of the prompted hand gets a SIGNED distance
to the object transformed by exp(l) (the `mode`'s transform below): the Euclidean distance to the nearest surface
sample, negative when the vertex lies behind that sample's outward normal (inside). For each l on `grid_log`, score(l) =
#vertices in CONTACT (|signed| <= `contact_eps_mm`) - #vertices PENETRATING (signed < -`contact_eps_mm`); l_t is the
maximiser (the middle of a tied run). It is not a measurement when the best score is below `patch_k` (the hand never
touches) or the maximiser is on the grid edge.
Why the MAXIMUM-CONTACT position (the known-answer check KA-G failed on the first rule): on real egocentric grasps the
hand touches the object over a BAND of along-ray positions (fingers on surfaces nearly parallel to the ray); the
per-frame smoother's "smallest sufficient move" (and its signed variant) returns the band's edge nearest the solved
depth, which is not equivariant: compose a known move k inside the band and the estimate does not move by -k (KA-G
failed on most windows). The band's maximum-contact point moves exactly with the object. The SIGN matters: an unsigned
contact count credits an object slicing through the hand.
The clip's constant is the MEDIAN of l_t over its first `onset_rows` gated rows (the grasp onset, where the hand has
just closed on the object), clipped to +-`max_log_shift`. Fewer than `min_gated_frames` gated rows
leaves the clip EXACTLY as solved (the same arrays: solve.npz byte-identical to the stage without the key).

**Action, one scalar c for the whole clip** (both modes move every row's centroid to exp(c) of its distance along its
own ray, so camera-frame displacements between rows are the solved ones times exp(c): motion is scaled, never
cancelled):
- `gauge`: the paper's gauge (Prop. 1): scale <- exp(c) scale and every translation <- exp(c) t. Every reprojection is
  EXACTLY preserved; the object's metric size follows its depth (the solve's free direction, fixed by the hand instead
  of the template's size prior).
- `shift`: every translation moves along the row's centroid ray, t <- t + (exp(c) - 1) c_t; size and rotation unchanged,
  so the silhouette scales by exp(-c) (the per-frame smoother's action, but one constant per clip).
The Laplace covariance's translation rows/columns are multiplied by exp(c) (the map's Jacobian in t), so qc's
translation-std feature sees the moved solve consistently. Rotation is never changed.

HaWoR's side-specific depth bias is NOT corrected: the physics endpoint needs the object where OUR hand is.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from fastdo.engine.contact_ray_smoother import (
    ContactRaySmootherConfig,
    _prompt_side,
    contact_depth_measurements,
)

__all__ = [
    "GRASP_KEY",
    "GRASP_MODES",
    "GraspRayConstantConfig",
    "GraspResult",
    "apply_log_constant",
    "grasp_constant_from_artifacts",
    "grasp_constant_from_mapping",
    "grasp_ray_constant",
    "onset_log_measurements",
]

GRASP_KEY = "grasp_ray_constant"
GRASP_MODES: tuple[str, ...] = ("gauge", "shift")
_HANDS: tuple[str, ...] = ("prompt", "auto", "left", "right")
_SIDES: tuple[str, ...] = ("left", "right")
_CMIN_GRID_MM = ContactRaySmootherConfig().grid()        # the smoother's C_min slide grid (-300..300 by 2 mm)


@dataclass(frozen=True)
class GraspRayConstantConfig:
    """`solve_stage.grasp_ray_constant` (module docstring). Every field must be spelled in the config block.

    Attributes:
        mode: "gauge" (scale and translations by one factor: image-exact) or "shift" (translations along the centroid
            rays at fixed size).
        touch_gate_mm: A row is gated (touching) when the smoother's `C_min <= touch_gate_mm`.
        onset_rows: The constant is the median over the first `onset_rows` gated rows.
        min_gated_frames: Fewer gated rows than this leaves the clip exactly as solved.
        max_log_shift: Cap on |c| (the depth anchor's band 0.2); at most 0.3 (the measurement grid's reach).
        patch_k: A row is measured only if its best contact-minus-penetration score reaches this many vertices.
        contact_eps_mm: A vertex is in contact when |signed surface distance| <= this, penetrating below -this (mm).
        hands: "prompt" (the prompt stage's side; "auto" when it has none), "auto", "left" or "right".
        grid_log: (lo, hi, step) of the log-factor grid; lo < 0 < hi; the grid holds 0 exactly.

    Raises:
        ValueError: On an unknown mode or hands, a non-positive or non-finite number, a non-integer count, or a bad grid.
    """

    mode: str = "gauge"
    touch_gate_mm: float = 56.0
    onset_rows: int = 10
    min_gated_frames: int = 3
    max_log_shift: float = 0.2
    patch_k: int = 20
    contact_eps_mm: float = 5.0
    hands: str = "prompt"
    grid_log: tuple[float, float, float] = (-0.3, 0.3, 0.004)

    def __post_init__(self) -> None:
        if self.mode not in GRASP_MODES:
            raise ValueError(f"solve_stage.{GRASP_KEY}.mode must be one of {GRASP_MODES}, got {self.mode!r}")
        for name in ("touch_gate_mm", "max_log_shift", "contact_eps_mm"):
            raw = getattr(self, name)
            value = math.nan if isinstance(raw, bool) else float(raw)
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"solve_stage.{GRASP_KEY}.{name} must be finite and > 0, got {raw!r}")
            object.__setattr__(self, name, value)
        for name in ("onset_rows", "min_gated_frames", "patch_k"):
            n = getattr(self, name)
            if isinstance(n, bool) or int(n) != n or int(n) < 1:
                raise ValueError(f"solve_stage.{GRASP_KEY}.{name} must be an integer >= 1, got {n!r}")
            object.__setattr__(self, name, int(n))
        if self.hands not in _HANDS:
            raise ValueError(f"solve_stage.{GRASP_KEY}.hands must be one of {_HANDS}, got {self.hands!r}")
        lo, hi, step = (float(v) for v in self.grid_log)
        if not (all(math.isfinite(v) for v in (lo, hi, step)) and lo < 0.0 < hi and step > 0.0):
            raise ValueError(f"solve_stage.{GRASP_KEY}.grid_log must be (lo < 0 < hi, step > 0), got {self.grid_log!r}")
        object.__setattr__(self, "grid_log", (lo, hi, step))
        if self.max_log_shift > min(-lo, hi):
            raise ValueError(f"solve_stage.{GRASP_KEY}.max_log_shift {self.max_log_shift} exceeds the grid's reach")

    def grid(self) -> np.ndarray:
        """The log-factor grid: integer multiples of `step` in [lo, hi] (0 is on it exactly)."""
        lo, hi, step = self.grid_log
        return np.arange(math.ceil(lo / step - 1e-9), math.floor(hi / step + 1e-9) + 1) * step


def grasp_constant_from_mapping(block: Any) -> GraspRayConstantConfig | None:
    """`solve_stage.grasp_ray_constant`: null (off), or a block spelling EVERY field; `grid_log` lists become tuples.

    Raises:
        ValueError: On a non-mapping, an unknown or a missing key, or anything the dataclass rejects.
    """
    if block is None:
        return None
    if not isinstance(block, Mapping):
        raise ValueError(f"solve_stage.{GRASP_KEY} must be null or a mapping, got {block!r}")  # noqa: TRY004
    data = {str(k): block[k] for k in block}
    known = [f.name for f in dataclasses.fields(GraspRayConstantConfig)]
    unknown = sorted(set(data) - set(known))
    if unknown:
        raise ValueError(f"solve_stage.{GRASP_KEY} has unknown keys {unknown}; known keys: {known}")
    missing = sorted(set(known) - set(data))
    if missing:
        raise ValueError(f"solve_stage.{GRASP_KEY} is missing {missing}: spell every field")
    data["grid_log"] = tuple(data["grid_log"])
    return GraspRayConstantConfig(**data)


def _centroids(poses: np.ndarray, scale: float, mesh: np.ndarray) -> np.ndarray:
    return scale * np.einsum("tij,j->ti", poses[:, :3, :3], mesh.mean(axis=0)) + poses[:, :3, 3]


def onset_log_measurements(poses: np.ndarray, scale: float, mesh_points: np.ndarray, mesh_normals: np.ndarray,
                           joints: np.ndarray, valid: np.ndarray, verts: np.ndarray, cfg: GraspRayConstantConfig
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per row: the contact-ray `C_min` (mm), the measured log factor l_t (NaN where not measured) and the gated mask
    (`C_min <= touch_gate_mm` and l_t measured). `mesh_normals` are the samples' outward unit normals (canonical)."""
    poses = np.asarray(poses, dtype=np.float64)
    mesh = np.asarray(mesh_points, dtype=np.float64)
    normals = np.asarray(mesh_normals, dtype=np.float64)
    verts = np.asarray(verts, dtype=np.float64)
    t_rows = poses.shape[0]
    if verts.ndim != 3 or verts.shape[0] != t_rows or verts.shape[2] != 3:
        raise ValueError(f"verts must be ({t_rows},V,3), got {verts.shape}")
    if normals.shape != mesh.shape:
        raise ValueError(f"mesh_normals {normals.shape} must match mesh_points {mesh.shape}")
    cmin, _, _ = contact_depth_measurements(poses, scale, mesh, joints, valid, _CMIN_GRID_MM)
    grid = cfg.grid()
    lam = np.exp(grid)
    cen = _centroids(poses, scale, mesh)
    ell = np.full(t_rows, np.nan)
    for t in np.flatnonzero(np.isfinite(cmin) & (cmin <= cfg.touch_gate_mm)):
        v = verts[t][np.isfinite(verts[t]).all(axis=1)]
        if v.shape[0] < cfg.patch_k:
            continue
        rot = poses[t, :3, :3]
        surf, nrm = scale * (mesh @ rot.T) + poses[t, :3, 3], normals @ rot.T
        tree = cKDTree(surf)
        if cfg.mode == "gauge":            # signed distance to exp(l) S  =  exp(l) * signed distance(exp(-l) v, S)
            q = v[None, :, :] / lam[:, None, None]
            factor = lam[:, None]
        else:                              # signed distance to S + (exp(l) - 1) c_t
            q = v[None, :, :] - (lam - 1.0)[:, None, None] * cen[t][None, None, :]
            factor = np.ones((grid.size, 1))
        dist, idx = tree.query(q.reshape(-1, 3), k=1)
        side = np.where(np.einsum("nj,nj->n", q.reshape(-1, 3) - surf[idx], nrm[idx]) < 0.0, -1.0, 1.0)
        signed = (side * dist).reshape(grid.size, v.shape[0]) * factor   # Euclidean, signed by the normal
        eps = cfg.contact_eps_mm / 1e3
        score = (np.abs(signed) <= eps).sum(axis=1) - (signed < -eps).sum(axis=1)   # contact minus penetration
        best = int(score.max())
        ties = np.flatnonzero(score == best)
        k = int(ties[len(ties) // 2])                  # the middle of a tied band (equivariant under a known move)
        if best >= cfg.patch_k and 0 < k < grid.size - 1:
            ell[t] = float(grid[k])
    with np.errstate(invalid="ignore"):
        gated = np.isfinite(cmin) & (cmin <= cfg.touch_gate_mm) & np.isfinite(ell)
    return cmin, ell, gated


def apply_log_constant(poses: np.ndarray, scale: float, mesh_points: np.ndarray, c: float, mode: str
                       ) -> tuple[np.ndarray, float]:
    """One log factor `c` on every row (module docstring): returns NEW `(poses, scale)` (float64)."""
    poses = np.asarray(poses, dtype=np.float64)
    out = poses.copy()
    lam = math.exp(c)
    if mode == "gauge":
        out[:, :3, 3] = lam * poses[:, :3, 3]
        return out, lam * float(scale)
    if mode == "shift":
        cen = _centroids(poses, scale, np.asarray(mesh_points, dtype=np.float64))
        out[:, :3, 3] = poses[:, :3, 3] + (lam - 1.0) * cen
        return out, float(scale)
    raise ValueError(f"mode must be one of {GRASP_MODES}, got {mode!r}")


@dataclass(frozen=True)
class GraspResult:
    """What `grasp_ray_constant` did.

    Attributes:
        poses: `(T,4,4)`; the INPUT array itself when inactive, else a new float64 array.
        scale: The clip's scale (the input when inactive or in `shift` mode).
        active: Whether the constant was applied.
        log_constant: The applied c (0.0 when inactive).
        record: JSON-ready summary (solve.json's block).
    """

    poses: np.ndarray
    scale: float
    active: bool
    log_constant: float
    record: dict[str, Any]


def grasp_ray_constant(poses: np.ndarray, scale: float, mesh_points: np.ndarray, mesh_normals: np.ndarray,
                       joints: np.ndarray, valid: np.ndarray, cfg: GraspRayConstantConfig, verts: np.ndarray
                       ) -> GraspResult:
    """The module docstring's constant on one hand (`joints` (T,21,3) gate it, `verts` (T,V,3) MANO vertices measure
    it against the samples `mesh_points` with outward `mesh_normals`)."""
    cmin, ell, gated = onset_log_measurements(poses, scale, mesh_points, mesh_normals, joints, valid, verts, cfg)
    rows = np.flatnonzero(gated)
    record: dict[str, Any] = {"config": dataclasses.asdict(cfg), "n_rows": int(np.asarray(poses).shape[0]),
                              "n_gated": int(rows.size), "n_hand_rows": int(np.isfinite(cmin).sum())}
    if rows.size < cfg.min_gated_frames:
        record.update(active=False, log_constant=0.0,
                      reason=f"{rows.size} gated rows < min_gated_frames {cfg.min_gated_frames}")
        return GraspResult(poses, float(scale), False, 0.0, record)
    onset = rows[: cfg.onset_rows]
    raw = float(np.median(ell[onset]))
    c = float(np.clip(raw, -cfg.max_log_shift, cfg.max_log_shift))
    new_poses, new_scale = apply_log_constant(poses, scale, mesh_points, c, cfg.mode)
    record.update(active=True, log_constant=c, log_constant_raw=raw, capped=bool(abs(raw) > cfg.max_log_shift),
                  onset_rows=[int(i) for i in onset], onset_ell=[round(float(v), 6) for v in ell[onset]],
                  ell_median_all_gated=float(np.median(ell[rows])), gated_rows=[int(i) for i in rows],
                  cmin_onset_mm=[round(float(v), 3) for v in cmin[onset]])
    return GraspResult(new_poses, new_scale, True, c, record)


def grasp_constant_from_artifacts(artifact_dir: Path, poses: np.ndarray, scale: float, mesh_points: np.ndarray,
                                  mesh_normals: np.ndarray, hands: Mapping[str, np.ndarray],
                                  cfg: GraspRayConstantConfig) -> GraspResult:
    """Chooses the hand (`cfg.hands`, like the smoother) and runs `grasp_ray_constant`; the record names the side.

    Args:
        artifact_dir: The clip's directory (for `prompt.json` when `cfg.hands` is "prompt").
        poses, scale, mesh_points, mesh_normals: The solve (camera frame) and the shape stage's `samples.npz` points
            and outward normals.
        hands: `hands.npz` contents on the solve's rows.
        cfg: The config.
    """
    from fastdo.engine.workers_contact import choose_side

    joints = {s: np.asarray(hands[f"{s}_joints"], dtype=np.float64) for s in _SIDES}
    valid = {s: np.asarray(hands[f"{s}_valid"]).astype(bool) for s in _SIDES}
    poses64 = np.asarray(poses, dtype=np.float64)
    mesh = np.asarray(mesh_points, dtype=np.float64)
    side, source = None, cfg.hands
    if cfg.hands == "prompt":
        side = _prompt_side(artifact_dir)
        source = "prompt" if side is not None else "auto(no prompt side)"
    elif cfg.hands in _SIDES:
        side = cfg.hands
    if side is None:
        side = choose_side(lambda t: scale * (mesh @ poses64[t, :3, :3].T) + poses64[t, :3, 3], joints, valid) or None
    if side is None:
        rec = {"config": dataclasses.asdict(cfg), "n_rows": int(poses64.shape[0]), "n_gated": 0, "n_hand_rows": 0,
               "active": False, "log_constant": 0.0, "reason": "no usable hand", "side": "", "side_source": source}
        return GraspResult(poses, float(scale), False, 0.0, rec)
    res = grasp_ray_constant(poses, scale, mesh, mesh_normals, joints[side], valid[side], cfg,
                             np.asarray(hands[f"{side}_vertices"], dtype=np.float64))
    res.record.update(side=side, side_source=source)
    return res
