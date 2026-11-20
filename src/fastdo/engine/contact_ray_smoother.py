"""Contact-gated along-ray log-depth smoother (opt-in `solve_stage.contact_ray_smoother`, default null = off).

**Why.** On an OakInk2 egocentric set, physics success separates on the
hand->object gap at the grasp (DO AS I DO's own frame-0 distance: 1.2 mm on successes vs 81 mm on failures). Contact is
OFF for egocentric video, so nothing in the solve ties the object's depth along the camera ray to the hand that
holds it; the object then wanders 30-100 mm along the ray away from the grasping hand, passes DO AS I DO's 10 cm in-hand
test (no pedestal) and falls once the warmup weld releases.

**Gate.** The label-free touch evidence `C_min` (`fastdo.qc.interaction.frame_min_contact`): slide the solved
object along the ray to its centroid and take the smallest mean distance from the 21 hand joints to its surface. A
row is GATED (touching) when `C_min <= touch_gate_mm` (56 mm: AUC 0.984 DexYCB, 0.955 egocentric HOI4D).

**Measurement: the hand's CONTACT PATCH, not the joint mean and not the wrist.** On a gated row, slide the object
along the same ray and find the SMALLEST move |delta| at which the hand's contact patch touches the surface: the mean
of the `patch_k` smallest MANO-vertex-to-surface distances is <= `contact_eps_mm` (if no slide reaches that, the slide
that minimises it). `y_t = log((r_t + delta_t) / r_t)`, with `r_t` the solved centroid distance. Why this point:
- the physics endpoint is contact: DO AS I DO's in-hand test and grasp are the MINIMUM hand-surface-vertex distance,
  and the patch (k vertices, robust to one noisy vertex) is that quantity made robust;
- the smallest sufficient move leaves an object the hand already touches where it is (delta = 0), and closes a gap
  by exactly the gap; the argmin of the 21-JOINT mean (the gate's own slide) does not: it pulls the object into the
  palm, through the joint cloud. A first, label-free engine pass showed it: it moved an egocentric clip whose start
  distance is 0.9 mm by the full cap;
- the wrist sits 2-9 cm from the grasp surface with an orientation-dependent along-ray offset, so it measures the
  grasp only through a per-grasp bias.
HaWoR's SIDE-SPECIFIC depth bias (right +15.8 mm far, left -9.4 mm near) is NOT corrected: the physics endpoint
needs the object where OUR hand is (retargeting consumes our hand), and a bias correction would re-open a 9-16 mm
hand-object gap; its cost shows up in the absolute object error instead.

**Model.** State x_t = the correction of the object's log depth along its ray (true minus solved). Random walk
`x_t = x_{t-1} + w_t`, `w_t ~ N(0, q)` with q the robust (MAD) variance of the SOLVED log-depth increments (floored at
`q_floor`); prior `x_0 ~ N(0, max_log_shift^2)`. Measurement on gated rows, `y_t = x_t + v_t`,
`sd(v_t) = (meas_floor_mm + C_min_t) / (1000 r_t)`: a hand on the surface (small `C_min`) is a sharp measurement, a hand
only near it a weak one (the
gate's confidence). A Rauch-Tung-Striebel smoother over the whole window propagates the gated rows' evidence to
the rows outside the gate (e.g. the start rows before the grasp).

**Action.** Row t's object moves ALONG ITS CENTROID RAY by `exp(clip(x_t, +-max_log_shift)) - 1` of its centroid
distance: `t_t <- t_t + (exp(dx_t) - 1) c_t`, rotation and the clip's scale unchanged. The centroid's image projection
is preserved EXACTLY. Because the object keeps ONE metric size (physics needs one mesh), the other points' projections
are not: the silhouette scales by `exp(-dx_t)` about the centroid's image. Exact image invariance holds only for the
paper's gauge (Prop. 1: size AND every translation scaled by one lambda), which is a per-clip global scale; a per-frame
depth move at fixed size is weakly but not zero-visible to the image terms (the solver's along-ray error). The
cap bounds that cost (`max_log_shift` 0.2 = the depth band: at most a 22 % silhouette size change).

A clip with fewer than `min_gated_frames` gated rows (or no usable hand) is left EXACTLY as solved: the same pose array
is returned, so solve.npz is byte-identical to the smoother-off stage.
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

__all__ = [
    "HAND_CHOICES",
    "SMOOTHER_KEY",
    "ContactRaySmootherConfig",
    "SmootherResult",
    "contact_depth_measurements",
    "rts_random_walk",
    "smooth_contact_ray",
    "smoother_from_artifacts",
    "smoother_from_mapping",
]

SMOOTHER_KEY = "contact_ray_smoother"
HAND_CHOICES: tuple[str, ...] = ("prompt", "auto", "left", "right")
_SIDES: tuple[str, ...] = ("left", "right")
_MAD_TO_SD = 1.4826


@dataclass(frozen=True)
class ContactRaySmootherConfig:
    """`solve_stage.contact_ray_smoother` (module docstring). Every field must be spelled in the config block.

    Attributes:
        touch_gate_mm: A row is gated (touching) when `C_min <= touch_gate_mm` (56 mm).
        max_log_shift: Cap on any row's along-ray move |log r_new / r_solved| (0.2 = the depth band); also the prior sd.
        meas_floor_mm: Measurement sd floor (mm) added to `C_min` (the per-joint noise of the hand estimate).
        min_gated_frames: Fewer gated rows than this leaves the clip exactly as solved.
        q_floor: Floor on the random-walk variance per row (log units squared).
        patch_k: Contact patch size: the k MANO vertices nearest the surface.
        contact_eps_mm: The patch touches when its mean surface distance is <= this (mm).
        hands: "prompt" (the prompt stage's chosen side: the hand that is retargeted; falls back to "auto" when
            prompt.json has no side), "auto" (`fastdo.engine.workers_contact.choose_side`), "left" or "right".
        grid_mm: (lo, hi, step) of the slide grid in mm (-300..300 by 2).

    Raises:
        ValueError: On a non-positive or non-finite number, an integer field that is not an integer >= 1, an unknown
            `hands`, or a grid that is not lo < 0 < hi with step > 0.
    """

    touch_gate_mm: float = 56.0
    max_log_shift: float = 0.2
    meas_floor_mm: float = 10.0
    min_gated_frames: int = 3
    q_floor: float = 1e-6
    patch_k: int = 20
    contact_eps_mm: float = 5.0
    hands: str = "prompt"
    grid_mm: tuple[float, float, float] = (-300.0, 300.0, 2.0)

    def __post_init__(self) -> None:
        for name in ("touch_gate_mm", "max_log_shift", "meas_floor_mm", "q_floor", "contact_eps_mm"):
            raw = getattr(self, name)
            value = math.nan if isinstance(raw, bool) else float(raw)
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"solve_stage.{SMOOTHER_KEY}.{name} must be finite and > 0, got {raw!r}")
            object.__setattr__(self, name, value)
        for name in ("min_gated_frames", "patch_k"):
            n = getattr(self, name)
            if isinstance(n, bool) or int(n) != n or int(n) < 1:
                raise ValueError(f"solve_stage.{SMOOTHER_KEY}.{name} must be an integer >= 1, got {n!r}")
            object.__setattr__(self, name, int(n))
        if self.hands not in HAND_CHOICES:
            raise ValueError(f"solve_stage.{SMOOTHER_KEY}.hands must be one of {HAND_CHOICES}, got {self.hands!r}")
        lo, hi, step = (float(v) for v in self.grid_mm)
        if not (all(math.isfinite(v) for v in (lo, hi, step)) and lo < 0.0 < hi and step > 0.0):
            raise ValueError(f"solve_stage.{SMOOTHER_KEY}.grid_mm must be (lo < 0 < hi, step > 0), got {self.grid_mm!r}")
        object.__setattr__(self, "grid_mm", (lo, hi, step))

    def grid(self) -> np.ndarray:
        """The slide grid in mm (`np.arange(lo, hi + step / 2, step)`, as `fastdo.contact.DEFAULT_GRID_MM`)."""
        lo, hi, step = self.grid_mm
        return np.arange(lo, hi + step / 2.0, step)


def smoother_from_mapping(block: Any) -> ContactRaySmootherConfig | None:
    """`solve_stage.contact_ray_smoother`: null (off), or a block spelling EVERY field (a registered arm never inherits
    a constant silently); `grid_mm` lists become tuples.

    Raises:
        ValueError: On a non-mapping, an unknown or a missing key, or anything the dataclass rejects.
    """
    if block is None:
        return None
    if not isinstance(block, Mapping):
        raise ValueError(f"solve_stage.{SMOOTHER_KEY} must be null or a mapping, got {block!r}")  # noqa: TRY004
    data = {str(k): block[k] for k in block}
    known = [f.name for f in dataclasses.fields(ContactRaySmootherConfig)]
    unknown = sorted(set(data) - set(known))
    if unknown:
        raise ValueError(f"solve_stage.{SMOOTHER_KEY} has unknown keys {unknown}; known keys: {known}")
    missing = sorted(set(known) - set(data))
    if missing:
        raise ValueError(f"solve_stage.{SMOOTHER_KEY} is missing {missing}: spell every field")
    data["grid_mm"] = tuple(data["grid_mm"])
    return ContactRaySmootherConfig(**data)


def contact_depth_measurements(poses: np.ndarray, scale: float, mesh_points: np.ndarray, joints: np.ndarray,
                               valid: np.ndarray, grid_mm: np.ndarray, verts: np.ndarray | None = None,
                               touch_gate_mm: float = math.inf, patch_k: int = 20,
                               contact_eps_mm: float = 5.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per row: `C_min` (mm), the measured along-ray move delta (mm, + = away from the camera) and the centroid
    distance r (m).

    `C_min` is `fastdo.qc.interaction.frame_min_contact`'s (21-joint mean, one KD-tree query for the whole slide grid).
    delta: with `verts` (MANO vertices, (T,V,3)) and `C_min <= touch_gate_mm`, the contact-patch move (module
    docstring); without `verts`, the argmin of the joint-mean slide (the gate's own profile; a diagnostic). Rows
    without a valid hand, a finite joint or a centroid ray get NaN (r is set whenever the ray exists); delta is NaN on
    rows the gate rejects.

    Raises:
        ValueError: On shape mismatches.
    """
    poses = np.asarray(poses, dtype=np.float64)
    mesh = np.asarray(mesh_points, dtype=np.float64)
    joints = np.asarray(joints, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    t_rows = poses.shape[0]
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or mesh.ndim != 2 or mesh.shape[1] != 3:
        raise ValueError(f"poses must be (T,4,4) and mesh_points (M,3), got {poses.shape} and {mesh.shape}")
    if joints.ndim != 3 or joints.shape[0] != t_rows or joints.shape[2] != 3 or valid.shape[0] != t_rows:
        raise ValueError(f"joints {joints.shape} and valid {valid.shape} must have {t_rows} rows of (J,3)")
    if verts is not None:
        verts = np.asarray(verts, dtype=np.float64)
        if verts.ndim != 3 or verts.shape[0] != t_rows or verts.shape[2] != 3:
            raise ValueError(f"verts must be ({t_rows},V,3), got {verts.shape}")
    grid = np.asarray(grid_mm, dtype=np.float64)
    grid_m = grid / 1e3
    centroid = mesh.mean(axis=0)
    cmin, delta, rad = (np.full(t_rows, np.nan) for _ in range(3))
    for t in range(t_rows):
        rot, trans = poses[t, :3, :3], poses[t, :3, 3]
        cp = scale * (rot @ centroid) + trans
        norm = float(np.linalg.norm(cp))
        if not np.isfinite(norm) or norm <= 0.0:
            continue
        rad[t] = norm
        if not valid[t]:
            continue
        j = joints[t][np.isfinite(joints[t]).all(axis=1)]
        if j.size == 0:
            continue
        ray = cp / norm
        tree = cKDTree(scale * (mesh @ rot.T) + trans)
        queries = j[None, :, :] - grid_m[:, None, None] * ray[None, None, :]
        prof = tree.query(queries.reshape(-1, 3), k=1)[0].reshape(grid_m.size, j.shape[0]).mean(axis=1)
        k = int(np.argmin(prof))
        cmin[t] = float(prof[k]) * 1e3
        if verts is None:
            delta[t] = float(grid[k])
            continue
        if not cmin[t] <= touch_gate_mm:
            continue
        v = verts[t][np.isfinite(verts[t]).all(axis=1)]
        if v.shape[0] < patch_k:
            continue
        q = v[None, :, :] - grid_m[:, None, None] * ray[None, None, :]
        dist = tree.query(q.reshape(-1, 3), k=1)[0].reshape(grid_m.size, v.shape[0])
        patch = np.partition(dist, patch_k - 1, axis=1)[:, :patch_k].mean(axis=1) * 1e3
        touching = np.flatnonzero(patch <= contact_eps_mm)
        if touching.size:
            delta[t] = float(grid[touching[np.argmin(np.abs(grid[touching]))]])
        else:
            delta[t] = float(grid[int(np.argmin(patch))])
    return cmin, delta, rad


def rts_random_walk(y: np.ndarray, r: np.ndarray, q: float, p0: float) -> tuple[np.ndarray, np.ndarray]:
    """Rauch-Tung-Striebel smoother of a 1-D random walk `x_t = x_{t-1} + N(0, q)`, `x_0 ~ N(0, p0)`.

    Args:
        y: `(T,)` measurements of x_t; NaN = no measurement on that row.
        r: `(T,)` measurement variances (ignored where `y` is NaN; must be > 0 elsewhere).
        q: Process variance per row (>= 0).
        p0: Prior variance of x_0 (> 0).

    Returns:
        `(mean, var)` of the smoothed posterior, each `(T,)`.
    """
    y = np.asarray(y, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    n = y.shape[0]
    xp, pp, xf, pf = (np.zeros(n) for _ in range(4))
    for t in range(n):
        xp[t], pp[t] = (0.0, p0) if t == 0 else (xf[t - 1], pf[t - 1] + q)
        if np.isfinite(y[t]):
            gain = pp[t] / (pp[t] + r[t])
            xf[t], pf[t] = xp[t] + gain * (y[t] - xp[t]), (1.0 - gain) * pp[t]
        else:
            xf[t], pf[t] = xp[t], pp[t]
    xs, ps = xf.copy(), pf.copy()
    for t in range(n - 2, -1, -1):
        c = pf[t] / pp[t + 1]
        xs[t] = xf[t] + c * (xs[t + 1] - xp[t + 1])
        ps[t] = pf[t] + c * c * (ps[t + 1] - pp[t + 1])
    return xs, ps


@dataclass(frozen=True)
class SmootherResult:
    """What `smooth_contact_ray` did.

    Attributes:
        poses: `(T,4,4)`; the INPUT array itself when inactive, else a new float64 array.
        active: Whether any row was moved.
        log_shift: `(T,)` applied along-ray log-depth correction (0 when inactive).
        record: JSON-ready summary (solve.json's block).
    """

    poses: np.ndarray
    active: bool
    log_shift: np.ndarray
    record: dict[str, Any]


def smooth_contact_ray(poses: np.ndarray, scale: float, mesh_points: np.ndarray, joints: np.ndarray,
                       valid: np.ndarray, cfg: ContactRaySmootherConfig, verts: np.ndarray | None = None
                       ) -> SmootherResult:
    """The module docstring's smoother on one hand (`joints` (T,21,3) gate it, `verts` (T,V,3) MANO vertices measure
    it; without `verts` the joint-mean slide argmin is the measurement, a diagnostic only). Rotation and scale are
    never changed."""
    poses_in = poses
    poses = np.asarray(poses, dtype=np.float64)
    t_rows = poses.shape[0]
    cmin, dstar, rad = contact_depth_measurements(poses, scale, mesh_points, joints, valid, cfg.grid(), verts,
                                                  cfg.touch_gate_mm, cfg.patch_k, cfg.contact_eps_mm)
    lo, hi, _ = cfg.grid_mm
    with np.errstate(invalid="ignore"):
        interior = (dstar > lo) & (dstar < hi)            # an argmin on the grid edge is not a measurement
        gated = (np.isfinite(cmin) & (cmin <= cfg.touch_gate_mm) & np.isfinite(dstar) & interior
                 & (rad + dstar / 1e3 > 0.0))
    n_gated = int(gated.sum())
    record: dict[str, Any] = {"config": dataclasses.asdict(cfg), "n_rows": t_rows, "n_gated": n_gated,
                              "n_hand_rows": int(np.isfinite(cmin).sum())}
    zero = np.zeros(t_rows)
    if n_gated < cfg.min_gated_frames:
        record.update(active=False, reason=f"{n_gated} gated rows < min_gated_frames {cfg.min_gated_frames}")
        return SmootherResult(poses_in, False, zero, record)
    has_ray = np.isfinite(rad)
    ell = np.log(rad[has_ray])
    inc = np.diff(ell)
    mad = float(np.median(np.abs(inc - np.median(inc)))) if inc.size else 0.0
    q = max((_MAD_TO_SD * mad) ** 2, cfg.q_floor)
    y = np.full(t_rows, np.nan)
    y[gated] = np.log1p(dstar[gated] / 1e3 / rad[gated])
    var = np.full(t_rows, np.inf)
    var[gated] = ((cfg.meas_floor_mm + cmin[gated]) / 1e3 / rad[gated]) ** 2
    rows = np.flatnonzero(has_ray)
    xs, ps = rts_random_walk(y[rows], var[rows], q, cfg.max_log_shift ** 2)
    shift = zero.copy()
    shift[rows] = np.clip(xs, -cfg.max_log_shift, cfg.max_log_shift)
    out = poses.copy()
    centroid = np.asarray(mesh_points, dtype=np.float64).mean(axis=0)
    cps = scale * np.einsum("tij,j->ti", poses[:, :3, :3], centroid) + poses[:, :3, 3]
    out[rows, :3, 3] = poses[rows, :3, 3] + np.expm1(shift[rows])[:, None] * cps[rows]
    n_capped = int((np.abs(xs) > cfg.max_log_shift).sum())
    record.update(active=True, q=q, n_capped=n_capped, meas_median=float(np.median(y[gated])),
                  shift_median=float(np.median(shift[rows])), shift_min=float(shift.min()),
                  shift_max=float(shift.max()), shift_first=float(shift[rows[0]]),
                  gated_rows=[int(i) for i in np.flatnonzero(gated)],
                  log_shift=[round(float(v), 6) for v in shift],
                  post_sd=[round(float(math.sqrt(max(v, 0.0))), 6) for v in ps])
    return SmootherResult(out, True, shift, record)


def _prompt_side(artifact_dir: Path) -> str | None:
    path = Path(artifact_dir) / "prompt.json"
    if not path.is_file():
        return None
    side = (json.loads(path.read_text()).get("choice") or {}).get("side")
    return side if side in _SIDES else None


def smoother_from_artifacts(artifact_dir: Path, poses: np.ndarray, scale: float, mesh_points: np.ndarray,
                            hands: Mapping[str, np.ndarray], cfg: ContactRaySmootherConfig) -> SmootherResult:
    """Chooses the hand (`cfg.hands`) and runs `smooth_contact_ray`; the record names the side and where it came from.

    Args:
        artifact_dir: The clip's directory (for `prompt.json` when `cfg.hands` is "prompt").
        poses, scale, mesh_points: The solve (camera frame) and the shape stage's `samples.npz` points.
        hands: `hands.npz` contents on the solve's rows (`{side}_joints` (T,21,3), `{side}_vertices` (T,778,3),
            `{side}_valid` (T,)).
        cfg: The smoother config.
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
               "active": False, "reason": "no usable hand", "side": "", "side_source": source}
        return SmootherResult(poses, False, np.zeros(poses64.shape[0]), rec)
    res = smooth_contact_ray(poses, scale, mesh, joints[side], valid[side], cfg,
                             np.asarray(hands[f"{side}_vertices"], dtype=np.float64))
    res.record.update(side=side, side_source=source)
    return res
