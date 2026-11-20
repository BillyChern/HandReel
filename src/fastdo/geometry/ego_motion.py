"""Camera EGO-MOTION of a window from its full-frame MoGe-2 pointmaps.

**Why.** About a third of narrated, accepted Ego4D windows show no object handling (driving,
shelf browsing), and nothing the pipeline stores separates them: the missing signal is WORLD-frame object motion,
i.e. whether the chosen object moves relative to the static scene. Object poses are solved in each frame's camera
frame, so head motion and object motion are confounded until the camera trajectory is known. This module recovers it.

**Model.** Each decode row t has a metric pointmap P_t (H,W,3) in its own camera frame, already multiplied by the
clip's hand-anchored scale. A static scene point seen at pixel u_i in row i and u_j in row j satisfies

    P_j[u_j] = s_ij R_ij P_i[u_i] + t_ij            (X_j = M_ij X_i, M_ij in Sim(3))

where (R_ij, t_ij) is the rigid camera motion and s_ij the ratio of the two rows' metric scales. **Sim(3), not SE(3):**
MoGe-2 predicts metric depth per frame, and its scale wanders by a few percent between frames of one video (measured
here: `scale` below, and its HOI4D validation). With SE(3) a relative depth-scale error e at distance z reads
as an along-ray translation of e*z (2 % at 3 m = 6 cm per frame), which would look exactly like object motion. The
per-row scale s_t is therefore estimated and carried: camera-to-world is X_w = s_t R_t X_t + t_t, the world frame is
the reference row's camera frame (and metric scale), and the same s_t is applied to anything else measured in row
t's camera frame -- the solved object and the hands, whose depths come from the same pointmap. SE(3) remains
available (`similarity=False`) as the ablation.

**Correspondences** (`fastdo.geometry.ego_tracks`): Shi-Tomasi corners on a downscaled grey frame, outside the object
mask and the hands' image regions, tracked by pyramidal Lucas-Kanade to rows i+k (k in `skips`) with a
forward-backward check. Each edge (i, i+k) yields pixel pairs, lifted through the two pointmaps.

**Robust registration per edge** (`register_similarity`). Residuals are normalised by distance, r = |Y - M X| / |Y|,
because monocular depth error grows with depth. (1) RANSAC over 3-point minimal Umeyama fits (fixed seed) with an
inlier gate r < `ransac_thresh_rel`; (2) IRLS with a Cauchy kernel w = 1 / (1 + (r/c)^2) whose M-step is the weighted
Umeyama closed form -- the exact minimiser of the weighted problem on Sim(3), i.e. each IRLS step solves the
linearised normal equations on the Lie algebra sim(3) exactly rather than by one Gauss-Newton step.

**Pose graph** (`solve_pose_graph`), on the connected component of good edges holding most rows: a spanning-tree
initialisation, then the standard decoupled Sim(3) averaging -- rotations by Gauss-Newton on so(3)
(e_ij = log(R_ij^T R_j^T R_i)), log-scales and translations by linear least squares given the rotations, each
IRLS-reweighted with a Cauchy kernel. Skip edges (k > 1) close short loops, which is what makes the graph more than
a chained odometry.

**Self-checks** (`EgoMotion.summary`): the fraction of rows on the trajectory; the inlier fraction and the median
normalised residual of the static set; LOOP consistency -- each skip edge's measured rotation against the rotation
composed from the k one-step edges it spans, before any optimisation; post-optimisation edge residuals; the scale
drift. `ok` is a fixed conjunction of these (`EgoMotionConfig`), and a consumer must abstain when it is False.

**Windowed smoothing** (`smooth_trajectory`, `smooth_half_rows`): the pose graph's row-to-row noise is independent
per one-step edge while head motion is smooth over a few rows at 10 fps; a +-3-row Gaussian local-linear window within each run of
trajectory rows halves the one-step relative pose error on HOI4D GT and leaves the absolute error unchanged.

**Failure modes.** (a) Dynamic background (other people, traffic): a minority is rejected by RANSAC/Cauchy; a majority
of moving pixels makes the estimate follow them (low inlier fraction, bad loops). (b) Vehicles: when the cabin fills
most of the frame the dominant rigid motion is the CABIN's, so "world" means the cabin -- for a screen asking whether
a hand-held object moves, a control fixed in the cabin then correctly reads as static. When the view is mostly
outside the windscreen, the road is the static set and a cabin control moves with the camera. (c) Textureless or dark
scenes, fast rotation and motion blur: too few tracks, edges fail, rows drop out, `ok` goes False. (d) The object or
hands not excluded (bad mask): they are few pixels and are rejected as outliers. (e) Depth-scale drift within a frame
(non-uniform) is not modelled.
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

__all__ = [
    "EGO_MOTION_FILE",
    "EdgeFit",
    "EgoMotion",
    "EgoMotionConfig",
    "load_ego_motion",
    "register_similarity",
    "rotation_angle_deg",
    "save_ego_motion",
    "sim3_matrix",
    "smooth_trajectory",
    "so3_exp",
    "so3_log",
    "solve_ego_motion",
    "solve_pose_graph",
    "umeyama",
]

EGO_MOTION_FILE = "ego_motion.npz"


@dataclass(frozen=True)
class EgoMotionConfig:
    """`geometry_stage.ego_motion`: the opt-in camera ego-motion estimate (module docstring). Defaults are a priori.

    Attributes:
        enabled: Off by default; off computes nothing and writes no file (geometry.npz is never changed either way).
        similarity: Sim(3) (per-row metric scale) when True, SE(3) when False.
        long_side: Long side (px) of the grey frames the tracker works on.
        max_corners, min_distance_px, corner_quality: Shi-Tomasi settings (working resolution).
        skips: Row offsets k of the tracked edges (i, i+k); 1 must be present.
        fb_max_px: Forward-backward error bound (working-resolution px).
        object_pad_frac: Object mask dilation, as a fraction of the image diagonal.
        hand_pad_frac: Hand box padding, as a fraction of the joints' box size (each side).
        ransac_iters, ransac_thresh_rel: RANSAC hypotheses and the normalised inlier gate.
        cauchy_c_rel: Cauchy scale of the per-edge IRLS (normalised residual).
        irls_iters: IRLS iterations per edge.
        min_inliers, min_inlier_frac: An edge is used only with at least this many / this fraction of inliers.
        rot_cauchy_deg, scale_cauchy, trans_cauchy_rel: Pose-graph Cauchy scales (rotation, log-scale, translation
            as a fraction of the edge's median scene distance).
        ok_min_row_frac, ok_max_loop_deg: `ok` requires this fraction of rows on the trajectory and a p90 loop
            rotation error at most this.
        seed: RANSAC seed. smooth_half_rows: Windowed smoothing of the pose-graph trajectory (`smooth_trajectory`): a
        local-linear fit with Gaussian weights of sigma `smooth_half_rows / 2` rows truncated at +-`smooth_half_rows`,
        within each contiguous run of trajectory rows. 0 = none. 3 was chosen on HOI4D development GT camera poses:
        one-step relative pose error 9.2 -> 4.4 mm and 0.43 -> 0.29 deg (medians of window medians), absolute error
        unchanged, the row-to-row second difference 14.0 -> 1.7 mm (GT 4.0 mm).
    """

    enabled: bool = False
    similarity: bool = True
    long_side: int = 640
    max_corners: int = 300
    min_distance_px: int = 10
    corner_quality: float = 0.01
    skips: tuple[int, ...] = (1, 3)
    fb_max_px: float = 1.0
    object_pad_frac: float = 0.02
    hand_pad_frac: float = 0.5
    ransac_iters: int = 128
    ransac_thresh_rel: float = 0.03
    cauchy_c_rel: float = 0.02
    irls_iters: int = 10
    min_inliers: int = 24
    min_inlier_frac: float = 0.3
    rot_cauchy_deg: float = 2.0
    scale_cauchy: float = 0.02
    trans_cauchy_rel: float = 0.01
    ok_min_row_frac: float = 0.8
    ok_max_loop_deg: float = 5.0
    seed: int = 252
    smooth_half_rows: int = 3

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool) or not isinstance(self.similarity, bool):
            raise ValueError("ego_motion.enabled and ego_motion.similarity must be bools")  # noqa: TRY004
        skips = tuple(int(k) for k in self.skips)
        if not skips or 1 not in skips or any(k < 1 for k in skips) or len(set(skips)) != len(skips):
            raise ValueError(f"ego_motion.skips must be distinct ints >= 1 including 1, got {self.skips!r}")
        object.__setattr__(self, "skips", tuple(sorted(skips)))
        if not isinstance(self.smooth_half_rows, int) or isinstance(self.smooth_half_rows, bool) \
                or self.smooth_half_rows < 0:
            raise ValueError(f"ego_motion.smooth_half_rows must be an int >= 0, got {self.smooth_half_rows!r}")
        for name in ("long_side", "max_corners", "min_distance_px", "ransac_iters", "irls_iters", "min_inliers"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"ego_motion.{name} must be an int >= 1, got {value!r}")
        for name in ("corner_quality", "fb_max_px", "ransac_thresh_rel", "cauchy_c_rel", "rot_cauchy_deg",
                     "scale_cauchy", "trans_cauchy_rel", "ok_max_loop_deg"):
            value = float(getattr(self, name))
            if not (math.isfinite(value) and value > 0):
                raise ValueError(f"ego_motion.{name} must be positive, got {value!r}")
        for name in ("object_pad_frac", "hand_pad_frac", "min_inlier_frac", "ok_min_row_frac"):
            value = float(getattr(self, name))
            if not (math.isfinite(value) and value >= 0):
                raise ValueError(f"ego_motion.{name} must be >= 0, got {value!r}")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> EgoMotionConfig:
        """From a config mapping; an unknown key raises rather than being ignored."""
        if not mapping:
            return cls()
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted({str(k) for k in mapping} - known)
        if unknown:
            raise ValueError(f"ego_motion has unknown keys {unknown}; known keys: {sorted(known)}")
        data = {str(k): mapping[k] for k in mapping}
        if "skips" in data:
            data["skips"] = tuple(int(k) for k in data["skips"])
        return cls(**data)

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), sort_keys=True)


# --- SO(3) / Sim(3) helpers -----------------------------------------------------------------------------------------


def _hat(w: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


def so3_exp(w: np.ndarray) -> np.ndarray:
    """Rodrigues: `exp(hat(w))` for one (3,) vector."""
    w = np.asarray(w, dtype=float)
    theta = float(np.linalg.norm(w))
    if theta < 1e-12:
        return np.eye(3) + _hat(w)
    k = _hat(w / theta)
    return np.eye(3) + math.sin(theta) * k + (1.0 - math.cos(theta)) * (k @ k)


def so3_log(r: np.ndarray) -> np.ndarray:
    """The (3,) rotation vector of a rotation matrix (angle in [0, pi])."""
    r = np.asarray(r, dtype=float)
    cos = float(np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0))
    theta = math.acos(cos)
    v = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    if theta < 1e-7:
        return 0.5 * v
    if math.pi - theta < 1e-4:                          # near pi: axis from the symmetric part
        m = (r + np.eye(3)) / 2.0
        axis = m[:, int(np.argmax(np.diag(m)))]
        axis = axis / np.linalg.norm(axis)
        if axis @ v < 0:
            axis = -axis
        return theta * axis
    return theta / (2.0 * math.sin(theta)) * v


def rotation_angle_deg(r: np.ndarray) -> float:
    """Rotation angle of a (3,3) rotation in degrees."""
    return math.degrees(math.acos(float(np.clip((np.trace(np.asarray(r, float)) - 1.0) / 2.0, -1.0, 1.0))))


def sim3_matrix(s: float, r: np.ndarray, t: np.ndarray) -> np.ndarray:
    """(4,4) `[[s R, t], [0, 1]]`."""
    m = np.eye(4)
    m[:3, :3] = float(s) * np.asarray(r, float)
    m[:3, 3] = np.asarray(t, float)
    return m


def umeyama(x: np.ndarray, y: np.ndarray, w: np.ndarray | None = None,
            similarity: bool = True) -> tuple[float, np.ndarray, np.ndarray]:
    """Weighted Umeyama: `(s, R, t)` minimising `sum_i w_i |y_i - (s R x_i + t)|^2` (s = 1 when not `similarity`).

    Raises:
        ValueError: On mismatched shapes or non-positive total weight.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.shape != y.shape or x.ndim != 2 or x.shape[1] != 3:
        raise ValueError(f"x and y must be matching (N,3), got {x.shape} and {y.shape}")
    w = np.ones(len(x)) if w is None else np.asarray(w, float)
    total = float(w.sum())
    if not total > 0:
        raise ValueError("umeyama needs a positive total weight")
    mx = (w[:, None] * x).sum(0) / total
    my = (w[:, None] * y).sum(0) / total
    xc, yc = x - mx, y - my
    cov = (w[:, None, None] * (yc[:, :, None] * xc[:, None, :])).sum(0) / total
    u, d, vt = np.linalg.svd(cov)
    sign = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[2, 2] = -1.0
    r = u @ sign @ vt
    if similarity:
        var = float((w * (xc ** 2).sum(1)).sum() / total)
        s = float(np.trace(np.diag(d) @ sign) / var) if var > 0 else 1.0
    else:
        s = 1.0
    t = my - s * r @ mx
    return s, r, t


def _batched_minimal(x: np.ndarray, y: np.ndarray, similarity: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unweighted Umeyama on B minimal samples at once: x, y (B,3,3) -> s (B,), R (B,3,3), t (B,3)."""
    mx, my = x.mean(1), y.mean(1)
    xc, yc = x - mx[:, None], y - my[:, None]
    cov = np.einsum("bni,bnj->bij", yc, xc) / 3.0
    u, d, vt = np.linalg.svd(cov)
    det = np.linalg.det(u) * np.linalg.det(vt)
    sign = np.ones((len(x), 3))
    sign[det < 0, 2] = -1.0
    r = np.einsum("bij,bj,bjk->bik", u, sign, vt)
    if similarity:
        var = (xc ** 2).sum((1, 2)) / 3.0
        s = np.where(var > 0, (d * sign).sum(1) / np.maximum(var, 1e-30), 1.0)
    else:
        s = np.ones(len(x))
    t = my - s[:, None] * np.einsum("bij,bj->bi", r, mx)
    return s, r, t


@dataclass(frozen=True)
class EdgeFit:
    """One edge's robust registration `X_j = s R X_i + t`.

    Attributes:
        ok: Enough inliers (`min_inliers`, `min_inlier_frac`) for the pose graph to use it.
        s, r, t: The fit (s = 1 under SE(3)).
        n_points, n_inliers: Correspondences in and normalised residual below `ransac_thresh_rel` at the end.
        resid_rel: Median normalised residual of the inliers.
        depth_m: Median distance of the inliers from camera j (sets the translation Cauchy scale).
    """

    ok: bool
    s: float
    r: np.ndarray
    t: np.ndarray
    n_points: int
    n_inliers: int
    resid_rel: float
    depth_m: float


def _failed(n: int) -> EdgeFit:
    return EdgeFit(False, math.nan, np.full((3, 3), np.nan), np.full(3, np.nan), n, 0, math.nan, math.nan)


def register_similarity(x: np.ndarray, y: np.ndarray, cfg: EgoMotionConfig,
                        rng: np.random.Generator) -> EdgeFit:
    """Robust `y ≈ s R x + t` (module docstring): RANSAC on 3-point Umeyama, then Cauchy IRLS."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    keep = np.isfinite(x).all(1) & np.isfinite(y).all(1)
    x, y = x[keep], y[keep]
    dist = np.linalg.norm(y, axis=1)
    keep = dist > 1e-3
    x, y, dist = x[keep], y[keep], dist[keep]
    n = len(x)
    if n < max(cfg.min_inliers, 3):
        return _failed(n)
    picks = np.stack([rng.choice(n, 3, replace=False) for _ in range(cfg.ransac_iters)])
    xs, ys = x[picks], y[picks]
    area = np.linalg.norm(np.cross(xs[:, 1] - xs[:, 0], xs[:, 2] - xs[:, 0]), axis=1)
    good = area > 1e-6
    if not good.any():
        return _failed(n)
    s_b, r_b, t_b = _batched_minimal(xs[good], ys[good], cfg.similarity)
    pred = s_b[:, None, None] * np.einsum("bij,nj->bni", r_b, x) + t_b[:, None, :]
    resid = np.linalg.norm(pred - y[None], axis=2) / dist[None]
    counts = (resid < cfg.ransac_thresh_rel).sum(1)
    best = int(np.argmax(counts))
    if counts[best] < 3:
        return _failed(n)
    inl = resid[best] < cfg.ransac_thresh_rel
    s, r, t = umeyama(x[inl], y[inl], similarity=cfg.similarity)
    for _ in range(cfg.irls_iters):
        res = np.linalg.norm(s * x @ r.T + t - y, axis=1) / dist
        w = 1.0 / (1.0 + (res / cfg.cauchy_c_rel) ** 2)
        s, r, t = umeyama(x, y, w, similarity=cfg.similarity)
    res = np.linalg.norm(s * x @ r.T + t - y, axis=1) / dist
    inl = res < cfg.ransac_thresh_rel
    if int(inl.sum()) >= 3:              # redescend: gross outliers get weight 0, not the Cauchy tail's small weight
        w = np.where(inl, 1.0 / (1.0 + (res / cfg.cauchy_c_rel) ** 2), 0.0)
        s, r, t = umeyama(x, y, w, similarity=cfg.similarity)
        res = np.linalg.norm(s * x @ r.T + t - y, axis=1) / dist
        inl = res < cfg.ransac_thresh_rel
    n_in = int(inl.sum())
    ok = n_in >= cfg.min_inliers and n_in >= cfg.min_inlier_frac * n and np.isfinite(s) and s > 0
    return EdgeFit(bool(ok), float(s), r, t, n, n_in, float(np.median(res[inl])) if n_in else math.nan,
                   float(np.median(dist[inl])) if n_in else math.nan)


# --- pose graph -----------------------------------------------------------------------------------------------------


def _components(n: int, edges: np.ndarray) -> list[list[int]]:
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j in edges:
        a, b = find(int(i)), find(int(j))
        if a != b:
            parent[max(a, b)] = min(a, b)
    groups: dict[int, list[int]] = {}
    for a in range(n):
        groups.setdefault(find(a), []).append(a)
    return sorted(groups.values(), key=lambda g: (-len(g), g[0]))


def _cauchy(r: np.ndarray, c: float) -> np.ndarray:
    return 1.0 / (1.0 + (np.asarray(r, float) / c) ** 2)


def solve_pose_graph(n_rows: int, edges: np.ndarray, fits: list[EdgeFit], cfg: EgoMotionConfig,
                     gn_iters: int = 6) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Camera-to-world Sim(3) per row from the good edges (module docstring).

    Returns:
        `(R (N,3,3), s (N,), t (N,3), valid (N,), reference_row)`; rows off the main component are NaN / invalid.
    """
    good = [e for e, f in enumerate(fits) if f.ok]
    rot = np.full((n_rows, 3, 3), np.nan)
    scale = np.full(n_rows, np.nan)
    trans = np.full((n_rows, 3), np.nan)
    valid = np.zeros(n_rows, bool)
    if not good:
        return rot, scale, trans, valid, -1
    comp = _components(n_rows, edges[good])[0]
    if len(comp) < 2:
        return rot, scale, trans, valid, -1
    nodes = sorted(comp)
    ref = nodes[0]
    in_comp = set(nodes)
    used = [e for e in good if int(edges[e, 0]) in in_comp]
    # spanning-tree init (BFS from ref, shortest skips first): T_i = T_j M_ij  <=>  T_j = T_i M_ij^{-1}
    rot[ref], scale[ref], trans[ref] = np.eye(3), 1.0, np.zeros(3)
    done = {ref}
    order = sorted(used, key=lambda e: (int(edges[e, 1] - edges[e, 0]), int(edges[e, 0])))
    while len(done) < len(nodes):
        grew = False
        for e in order:
            i, j = int(edges[e, 0]), int(edges[e, 1])
            f = fits[e]
            if i in done and j not in done:         # T_j = T_i M_ij^{-1}
                inv_r = f.r.T
                rot[j] = rot[i] @ inv_r
                scale[j] = scale[i] / f.s
                trans[j] = trans[i] - scale[i] / f.s * (rot[i] @ inv_r @ f.t)
                done.add(j)
                grew = True
            elif j in done and i not in done:       # T_i = T_j M_ij
                rot[i] = rot[j] @ f.r
                scale[i] = scale[j] * f.s
                trans[i] = scale[j] * (rot[j] @ f.t) + trans[j]
                done.add(i)
                grew = True
        if not grew:
            break
    unknown = [a for a in nodes if a != ref]
    col = {a: k for k, a in enumerate(unknown)}
    m = len(unknown)
    base_w = np.array([math.sqrt(max(fits[e].n_inliers, 1)) for e in used])
    # rotations: Gauss-Newton on so(3), e_ij = log(R_ij^T R_j^T R_i), d e / d(delta_i) = I, d/d(delta_j) = -(R_j^T R_i)^T
    c_rot = math.radians(cfg.rot_cauchy_deg)
    for _ in range(gn_iters):
        a_mat = np.zeros((3 * len(used), 3 * m))
        b_vec = np.zeros(3 * len(used))
        res_mag = np.zeros(len(used))
        for k, e in enumerate(used):
            i, j = int(edges[e, 0]), int(edges[e, 1])
            a_ij = rot[j].T @ rot[i]
            err = so3_log(fits[e].r.T @ a_ij)
            res_mag[k] = np.linalg.norm(err)
            b_vec[3 * k:3 * k + 3] = -err
            if i in col:
                a_mat[3 * k:3 * k + 3, 3 * col[i]:3 * col[i] + 3] = np.eye(3)
            if j in col:
                a_mat[3 * k:3 * k + 3, 3 * col[j]:3 * col[j] + 3] = -a_ij.T
        w = np.repeat(base_w * _cauchy(res_mag, c_rot), 3)
        sw = np.sqrt(w)
        delta = np.linalg.lstsq(a_mat * sw[:, None], b_vec * sw, rcond=None)[0]
        for a in unknown:
            rot[a] = rot[a] @ so3_exp(delta[3 * col[a]:3 * col[a] + 3])
        if np.abs(delta).max() < 1e-9:
            break
    # log-scales: lambda_i - lambda_j = log s_ij (IRLS)
    if cfg.similarity:
        lam = np.log(scale[nodes])
        lam_of = dict(zip(nodes, lam, strict=True))
        a_mat = np.zeros((len(used), m))
        b_vec = np.zeros(len(used))
        for k, e in enumerate(used):
            i, j = int(edges[e, 0]), int(edges[e, 1])
            b_vec[k] = math.log(fits[e].s)
            if i in col:
                a_mat[k, col[i]] += 1.0
            if j in col:
                a_mat[k, col[j]] -= 1.0
        w = base_w.copy()
        for _ in range(5):
            sw = np.sqrt(w)
            x = np.linalg.lstsq(a_mat * sw[:, None], b_vec * sw, rcond=None)[0]
            w = base_w * _cauchy(a_mat @ x - b_vec, cfg.scale_cauchy)
        for a in unknown:
            lam_of[a] = x[col[a]]
        for a in nodes:
            scale[a] = math.exp(lam_of[a])
    else:
        scale[nodes] = 1.0
    # translations: t_i - t_j = s_j R_j t_ij (IRLS), t_ref = 0
    a_mat = np.zeros((3 * len(used), 3 * m))
    b_vec = np.zeros(3 * len(used))
    c_t = np.array([cfg.trans_cauchy_rel * max(fits[e].depth_m, 0.1) for e in used])
    for k, e in enumerate(used):
        i, j = int(edges[e, 0]), int(edges[e, 1])
        b_vec[3 * k:3 * k + 3] = scale[j] * rot[j] @ fits[e].t
        if i in col:
            a_mat[3 * k:3 * k + 3, 3 * col[i]:3 * col[i] + 3] = np.eye(3)
        if j in col:
            a_mat[3 * k:3 * k + 3, 3 * col[j]:3 * col[j] + 3] = -np.eye(3)
    w = np.repeat(base_w, 3)
    for _ in range(5):
        sw = np.sqrt(w)
        x = np.linalg.lstsq(a_mat * sw[:, None], b_vec * sw, rcond=None)[0]
        res = np.linalg.norm((a_mat @ x - b_vec).reshape(-1, 3), axis=1)
        w = np.repeat(base_w * _cauchy(res / c_t, 1.0), 3)
    trans[ref] = 0.0
    for a in unknown:
        trans[a] = x[3 * col[a]:3 * col[a] + 3]
    valid[nodes] = True
    return rot, scale, trans, valid, ref


# --- the window's result --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class EgoMotion:
    """A window's camera trajectory and its self-checks.

    Attributes:
        cam_to_world: (N,4,4) float64 Sim(3), `X_w = cam_to_world[t] @ [X_t, 1]` (world = reference row's camera),
            after `smooth_trajectory` (`smooth_half_rows`; equal to `cam_to_world_raw` when 0).
        scale: (N,) the per-row s_t of the pose graph, before smoothing (1 under SE(3)); NaN off the trajectory.
        valid: (N,) rows on the trajectory.
        reference_row: The world frame's row (-1 when no trajectory).
        edges: (E,2) int rows (i, j).
        edge_ok, edge_n_points, edge_n_inliers, edge_resid_rel, edge_s: Per-edge registration outcome.
        edge_rel: (E,4,4) the measured M_ij (NaN when failed).
        loop_rot_deg: (E,) for skip edges, the angle between the measured R_ij and the product of the one-step
            edges it spans (NaN when any is missing, and for one-step edges).
        post_rot_deg, post_trans_rel: (E,) edge residuals after the pose graph (rotation; translation over the edge's
            median scene distance).
        summary: Window-level self-checks (see `summarise`, all on the unsmoothed graph), including `ok`.
        cam_to_world_raw: (N,4,4) the pose-graph trajectory before smoothing.
    """

    cam_to_world: np.ndarray
    scale: np.ndarray
    valid: np.ndarray
    reference_row: int
    edges: np.ndarray
    edge_ok: np.ndarray
    edge_n_points: np.ndarray
    edge_n_inliers: np.ndarray
    edge_resid_rel: np.ndarray
    edge_s: np.ndarray
    edge_rel: np.ndarray
    loop_rot_deg: np.ndarray
    post_rot_deg: np.ndarray
    post_trans_rel: np.ndarray
    summary: dict[str, Any]
    cam_to_world_raw: np.ndarray


def _loops(edges: np.ndarray, fits: list[EdgeFit]) -> np.ndarray:
    one = {(int(i), int(j)): f for (i, j), f in zip(edges, fits, strict=True) if j - i == 1 and f.ok}
    out = np.full(len(edges), np.nan)
    for e, ((i, j), f) in enumerate(zip(edges, fits, strict=True)):
        if j - i < 2 or not f.ok:
            continue
        comp = np.eye(3)
        for a in range(int(i), int(j)):
            if (a, a + 1) not in one:
                comp = None
                break
            comp = one[(a, a + 1)].r @ comp
        if comp is not None:
            out[e] = rotation_angle_deg(f.r.T @ comp)
    return out


def _nanpct(x: np.ndarray, q: float) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.percentile(x, q)) if x.size else math.nan


def summarise(n_rows: int, edges: np.ndarray, fits: list[EdgeFit], valid: np.ndarray, scale: np.ndarray,
              rot: np.ndarray, trans: np.ndarray, reference: int, loop: np.ndarray, post_rot: np.ndarray,
              post_trans: np.ndarray, cfg: EgoMotionConfig) -> dict[str, Any]:
    """Window-level self-checks; `ok` = row fraction >= `ok_min_row_frac` and loop p90 <= `ok_max_loop_deg`."""
    ok_edges = np.array([f.ok for f in fits], bool)
    n_pts = np.array([f.n_points for f in fits], float)
    n_in = np.array([f.n_inliers for f in fits], float)
    frac = np.where(n_pts > 0, n_in / np.maximum(n_pts, 1), np.nan)
    row_frac = float(valid.mean()) if n_rows else 0.0
    loop_p90 = _nanpct(loop, 90)
    ang = [rotation_angle_deg(rot[t]) for t in range(n_rows) if valid[t]]
    steps = [float(np.linalg.norm(trans[t + 1] - trans[t])) for t in range(n_rows - 1) if valid[t] and valid[t + 1]]
    logs = np.log(scale[valid]) if valid.any() else np.zeros(0)
    ok = bool(row_frac >= cfg.ok_min_row_frac and (math.isfinite(loop_p90) and loop_p90 <= cfg.ok_max_loop_deg))
    return {
        "ok": ok, "n_rows": int(n_rows), "reference_row": int(reference), "row_frac": row_frac,
        "n_edges": len(fits), "edge_ok_frac": float(ok_edges.mean()) if len(fits) else 0.0,
        "median_points": _nanpct(n_pts, 50), "median_inlier_frac": _nanpct(frac, 50),
        "static_resid_rel_median": _nanpct(np.array([f.resid_rel for f in fits]), 50),
        "loop_rot_deg_median": _nanpct(loop, 50), "loop_rot_deg_p90": loop_p90, "n_loops": int(np.isfinite(loop).sum()),
        "post_rot_deg_p90": _nanpct(post_rot, 90), "post_trans_rel_p90": _nanpct(post_trans, 90),
        "scale_log_absmax": float(np.abs(logs).max()) if logs.size else math.nan,
        "scale_log_sd": float(np.std(logs)) if logs.size else math.nan,
        "cam_rot_extent_deg": float(max(ang)) if ang else math.nan,
        "cam_path_m": float(sum(steps)) if steps else math.nan,
    }


def _local_linear_weights(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weights a with sum_u a_u y_u = the intercept at x = 0 of the w-weighted least-squares line through (x_u, y_u)."""
    s0, s1, s2 = w.sum(), (w * x).sum(), (w * x * x).sum()
    det = s0 * s2 - s1 * s1
    if det <= 1e-12 * max(s0 * s2, 1e-300):
        return w / s0
    return w * (s2 - s1 * x) / det


def smooth_trajectory(cam_to_world: np.ndarray, valid: np.ndarray, half_rows: int) -> np.ndarray:
    """Windowed smoothing of a Sim(3) trajectory (`EgoMotionConfig.smooth_half_rows`).

    Row t is replaced by a Gaussian-weighted (sigma `half_rows / 2`) LOCAL-LINEAR fit, over rows u within
    +-`half_rows` of t and in the same contiguous run of valid rows, of its log-scale, its camera centre and its
    rotation (in the tangent space at R_t: R_t exp(sum_u a_u log(R_t^T R_u))). Local-linear rather than a weighted
    mean so that a run's first and last rows, where the window is one-sided, keep a steady motion instead of lagging
    toward the run's inside (a window mean would shrink every displacement extent measured from them). Invalid rows
    stay NaN; a run of one or two rows is unchanged. Pose-graph noise is row-to-row (the one-step edges are measured
    independently), while head motion at 10 fps is smooth over a few rows.
    """
    c2w = np.asarray(cam_to_world, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    out = c2w.copy()
    if half_rows <= 0:
        return out
    n = len(c2w)
    run = np.full(n, -1)
    rid = -1
    for t in range(n):
        if valid[t]:
            if t == 0 or not valid[t - 1]:
                rid += 1
            run[t] = rid
    s = np.full(n, np.nan)
    rots: list[np.ndarray] = [np.eye(3)] * n
    for t in np.nonzero(valid)[0]:
        s[t] = float(np.cbrt(np.linalg.det(c2w[t, :3, :3])))
        rots[t] = c2w[t, :3, :3] / s[t]
    sigma = half_rows / 2.0
    for t in np.nonzero(valid)[0]:
        idx = np.array([u for u in range(max(0, t - half_rows), min(n, t + half_rows + 1)) if run[u] == run[t]])
        x = (idx - t).astype(np.float64)
        a = _local_linear_weights(x, np.exp(-0.5 * (x / sigma) ** 2))
        r_t = rots[t]
        rv = np.sum(a[:, None] * np.stack([so3_log(r_t.T @ rots[u]) for u in idx]), axis=0)
        out[t] = sim3_matrix(math.exp(float(np.sum(a * np.log(s[idx])))), r_t @ so3_exp(rv),
                             np.sum(a[:, None] * c2w[idx, :3, 3], axis=0))
    return out


def solve_ego_motion(n_rows: int, edges: np.ndarray, x_pts: list[np.ndarray], y_pts: list[np.ndarray],
                     cfg: EgoMotionConfig) -> EgoMotion:
    """The window's `EgoMotion` from per-edge lifted correspondences.

    Args:
        n_rows: Decode rows N.
        edges: (E,2) int (i, j), i < j < N.
        x_pts, y_pts: Per edge, (P_e,3) metric points in camera i and camera j (NaN rows allowed).
        cfg: The settings.
    """
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if len(x_pts) != len(edges) or len(y_pts) != len(edges):
        raise ValueError("one (x, y) point set per edge is required")
    if len(edges) and ((edges[:, 0] >= edges[:, 1]).any() or edges.min() < 0 or edges.max() >= n_rows):
        raise ValueError("edges must be (i, j) rows with 0 <= i < j < n_rows")
    rng = np.random.default_rng(cfg.seed)
    fits = [register_similarity(x, y, cfg, rng) for x, y in zip(x_pts, y_pts, strict=True)]
    rot, scale, trans, valid, ref = solve_pose_graph(n_rows, edges, fits, cfg)
    c2w = np.full((n_rows, 4, 4), np.nan)
    for t in range(n_rows):
        if valid[t]:
            c2w[t] = sim3_matrix(scale[t], rot[t], trans[t])
    post_rot = np.full(len(edges), np.nan)
    post_trans = np.full(len(edges), np.nan)
    rel = np.full((len(edges), 4, 4), np.nan)
    for e, ((i, j), f) in enumerate(zip(edges, fits, strict=True)):
        if not f.ok:
            continue
        rel[e] = sim3_matrix(f.s, f.r, f.t)
        if valid[i] and valid[j]:
            pred = np.linalg.inv(c2w[j]) @ c2w[i]          # M_ij implied by the trajectory
            post_rot[e] = rotation_angle_deg(f.r.T @ (pred[:3, :3] / np.cbrt(np.linalg.det(pred[:3, :3]))))
            post_trans[e] = float(np.linalg.norm(pred[:3, 3] - f.t)) / max(f.depth_m, 1e-3)
    loop = _loops(edges, fits)
    summary = summarise(n_rows, edges, fits, valid, scale, rot, trans, ref, loop, post_rot, post_trans, cfg)
    summary["smooth_half_rows"] = int(cfg.smooth_half_rows)
    raw = c2w
    c2w = smooth_trajectory(raw, valid, cfg.smooth_half_rows)
    return EgoMotion(c2w, scale, valid, ref, edges, np.array([f.ok for f in fits], bool),
                     np.array([f.n_points for f in fits], np.int64), np.array([f.n_inliers for f in fits], np.int64),
                     np.array([f.resid_rel for f in fits]), np.array([f.s for f in fits]), rel, loop, post_rot,
                     post_trans, summary, raw)


def save_ego_motion(path: Path, ego: EgoMotion, frame_idx: np.ndarray, cfg: EgoMotionConfig,
                    extra: Mapping[str, Any] | None = None) -> None:
    """Writes `ego_motion.npz` (tmp + replace, uncompressed); `summary`/`config` as JSON strings."""
    from fastdo.engine.worker_support import save_npz

    summary = dict(ego.summary) | dict(extra or {})
    save_npz(Path(path), cam_to_world=ego.cam_to_world, cam_to_world_raw=ego.cam_to_world_raw, scale=ego.scale, valid=ego.valid,
             reference_row=np.array(ego.reference_row, np.int64), frame_idx=np.asarray(frame_idx, np.int64),
             edges=ego.edges, edge_ok=ego.edge_ok, edge_n_points=ego.edge_n_points,
             edge_n_inliers=ego.edge_n_inliers, edge_resid_rel=ego.edge_resid_rel, edge_s=ego.edge_s,
             edge_rel=ego.edge_rel, loop_rot_deg=ego.loop_rot_deg, post_rot_deg=ego.post_rot_deg,
             post_trans_rel=ego.post_trans_rel, summary=np.array(json.dumps(summary, sort_keys=True)),
             config=np.array(cfg.to_json()))


def load_ego_motion(path: Path) -> dict[str, Any]:
    """`ego_motion.npz` as a dict of arrays plus parsed `summary` and `config`."""
    with np.load(Path(path), allow_pickle=False) as z:
        out: dict[str, Any] = {k: np.asarray(z[k]) for k in z.files}
    out["summary"] = json.loads(str(out["summary"]))
    out["config"] = json.loads(str(out["config"]))
    return out
