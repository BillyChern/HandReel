"""(a): the contact-faithful kinematic reference.

Goal: remove the relative hand-object ALONG-RAY gap at grasp by moving the HAND (never the object, which both arms
are scored against) along the camera ray, one offset per contact segment, under three gates.

Inputs per clip, at reference frame rate (frames f = 0 .. n-1), all in DO AS I DO's gravity-aligned, world-offset
scene frame (the frame of `trajectory_keypoints.npz` and of the clip's MuJoCo scene):
- the MANO vertices y (n, V, 3) of the retargeted side and its wrist positions (n, 3);
- the object pose (n, 7) = (p, wxyz) of the object body (the scene's free joint) and the object's signed-distance
  grid `GridSDF` in the BODY frame (128^3 nodes over the bounding cube plus 5 cm, trilinear; negative inside). On real
  clips the grid is the object's VISUAL mesh (the mesh M_i) signed by the generalised WINDING NUMBER
  (`GridSDF.from_mesh(sign="winding")`): the scanned/reconstructed meshes are open, so a pseudonormal sign put MoCap
  hands that touch the surface 6-8 cm "inside" (KA-noop, first WIP run), and the executor's convex collision parts
  (`GridSDF.from_convex_parts`, kept as a diagnostic) inflate the surface, so a touching hand reads as cm-deep
  penetration (KA-noop median |d| 15 mm, second WIP run);
- the camera centre c (3,) in the scene frame. Every raw dir is written in a camera frame (per-frame for egocentric
  video, the static `allocentric_top` camera for OakInk2 MoCap), so c is the image of the camera-frame origin: the ray
  of frame f is r_f = (wrist_f - c) / |wrist_f - c| (`camera_rays`).

Definitions:
- PATCH. Per frame, the `patch_k` = 20 MANO vertices of smallest SIGNED phi at d = 0 (`patch_select="signed"`): for
  a hand outside the object these are the contact patch of `fastdo.engine.contact_ray_smoother` (the k vertices
  nearest the surface); for a penetrating hand they are the DEEPEST vertices, so the penalty term
  sees them and KA-shift recovers an INWARD shift (selecting by |phi|, `patch_select="unsigned"`, picks the
  surface-crossing vertices of a penetrating hand: J is ~0 at d = 0 and a hand 30 or 80 mm inside the object is
  moved by only ~10 mm, failing KA-shift; it is kept as a labelled diagnostic).
  THE MoCap INTERPENETRATION FLOOR: MoCap MANO hands touching the visual mesh sit a
  few mm INSIDE it (TUNE MoCap: per-row patch depth max(0, -min_j phi) p50 1.2 mm, p95 3.10 mm, max 3.41 mm over
  611 rows of 5 clips). Under rev 1b (eps_pen = 2 mm, fit residual min_j phi) both terms of J(0) are non-zero on a
  touching MoCap hand, and along grazing rays the argmin moves it by cm (E_T KA-noop under rev 1b: median |d_raw|
  2.67 mm, max 38.0 mm -- FAILED). Rev 1c (PROPOSED, calibrated on TUNE only) gives J a contact
  BAND: with b = `fit_band` = `eps_pen` = the TUNE p95 rounded up to 0.1 mm (3.2 mm), the fit residual is the
  distance of min_j phi from the band [-b, 0] and the penalty starts at depth b, so a hand whose deepest patch vertex
  is within the MoCap floor has J = 0 and is not moved, while a hand 30/80 mm inside is still moved back to the band
  (KA-shift error <= b - gap; the tie rule below picks the band's edge nearest d = 0).
- CANDIDATE ROW. With v_j = phi_j grad_j / |grad_j| (the offset of patch vertex j from its closest surface point),
  a_j = r . v_j (along the ray) and l_j = |v_j - a_j r| (lateral): the row is a candidate iff some patch vertex has
  |a_j| <= d_max and l_j <= l_max. `lateral` reports min_j l_j over vertices with |a_j| <= d_max (inf if none).
- SEGMENT. A maximal run of candidate rows of >= `min_seg_frames` frames.
- OFFSET. d_k = argmin_d J_k(d), J_k(d) = sum_t [band(min_j phi_t(y_j + d r_t))^2 + lam_pen sum_j max(0, -phi - eps_pen)^2]
  over the segment's rows and patch vertices, band(m) = max(0, m) + max(0, -m - fit_band) (rev 1c; rev 1b is
  fit_band = 0, i.e. band(m) = |m|, `ContactRefConfig.rev1b()`). Search: a `grid_step` (1 mm) grid over [-2 d_max, 2 d_max] (wider than
  the cap, so an optimum beyond d_max is SEEN and capped rather than pinned at the cap), then golden-section
  refinement within one grid step. Among grid points within 1 mm RMS of the best (J <= J_min + |S| (1 mm)^2) the one
  of SMALLEST |d| is refined (the smoother's "smallest sufficient move": a hand on the surface stays; a thin object's
  far side is never chosen over its near side).
- GATES, in this order: (1) per-segment action cap |d_k| <= d_max, else the segment keeps d = 0 ("capped");
  (2) evidence: at d_k the patch's residual distance to the surface, e_t = min_j |phi_t(y_j + d_k r_t)|, is <= l_max
  on >= `evidence_frac` of the segment's rows, else d = 0 ("no_evidence"). DEVIATION: the specification
  names the LATERAL distance; e_t >= lateral, so this gate is stricter, and unlike the lateral distance it also
  refuses a segment whose rows need along-ray offsets that one d_k cannot reconcile (2 vs 9 cm); (3) windowed cap:
  with d(t) the field below on the executor grid and W the scoring window [W + g, T), if mean_W |d(t)| > c_w every
  offset is scaled by beta = c_w / mean_W |d(t)| (so the scaled mean is c_w).
- FIELD. d(f) = d_k inside segment k. Between two segments whose free interval g (frames) is <= 2 L (L = ramp_s in
  frames, so the two ramps would meet), d is the linear interpolation from d_k to d_{k+1} across the gap ("to the next
  value"): two equal offsets with a one-frame gap give a constant field. A longer gap and the clip's ends
  ramp linearly to 0 over min(L, the free interval at a clip end). Every step of the field is therefore at most
  max(|d_{k+1} - d_k| / (g + 1), |d_k| / (L' + 1)) (L' the ramp used), never a jump to 0 and back. `field_on_rows`
  samples it on the executor grid exactly as DO AS I DO's `load_data` samples the reference (the scorer's
  `reference_steps`: frame 0 held through the warmup, then a linspace of n k samples over [0, n - 1],
  k = round(ref_dt / sim_dt), the last value repeated).
- APPLY. h'_f = h_f + d(f) r_f for every hand keypoint (`shift_keypoints`); the object keys are untouched.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

__all__ = ["REV", "RULED", "ContactRefConfig", "ContactRefResult", "GridSDF", "Segment", "camera_center_exact", "camera_rays",
           "contact_reference", "field_on_rows", "find_segments", "offset_field", "quat_to_mat", "shift_keypoints",
           "window_scale"]

# Status of (a)'s J. Rev 1c (fit band + eps_pen from the TUNE MoCap floor) is PROPOSED: it passes KA-noop on TUNE
# but FAILS the E_T KA-noop check, so no consumer may use (a)'s field while RULED is False (the hook's `aref.json`
# records both; the clip hook refuses an unruled "aref" field).
REV = "1c-proposed"
RULED = False

_HAND_KEYS = ("qpos_wrist_{s}", "qpos_finger_{s}", "qpos_pip_{s}", "qpos_dip_{s}", "mano_verts_{s}")


@dataclass(frozen=True)
class ContactRefConfig:
    """(a)'s parameters, frozen from TUNE only."""

    l_max: float = 0.015
    d_max: float = 0.12
    min_seg_frames: int = 6
    evidence_frac: float = 0.6
    lam_pen: float = 10.0
    eps_pen: float = 0.0032             # rev 1c PROPOSED (TUNE MoCap floor p95, rounded up); rev 1b 0.002
    fit_band: float = 0.0032            # rev 1c PROPOSED; rev 1b 0.0 (fit residual |min_j phi|)
    c_w: float = 0.03
    grid_step: float = 0.001
    ramp_s: float = 0.3
    patch_k: int = 20
    grace_rows: int = 50
    carry_speed: float | None = None    # NOT in rev 1b (None = the registered rule); see `contact_reference`
    patch_select: str = "signed"        # "signed" (smallest phi) | "unsigned" (smallest |phi|; diagnostic)

    @staticmethod
    def rev1b(**kw: object) -> ContactRefConfig:
        """The registered rev-1b J (eps_pen 2 mm, no fit band): it FAILS KA-noop on MoCap (module docstring)."""
        return ContactRefConfig(**{"eps_pen": 0.002, "fit_band": 0.0, **kw})  # type: ignore[arg-type]


@dataclass(frozen=True, eq=False)
class GridSDF:
    """Signed distance on a cubic node grid (values[ix, iy, iz] at origin + step * (ix, iy, iz)), trilinear."""

    values: np.ndarray
    origin: np.ndarray
    step: float

    @staticmethod
    def from_mesh(vertices: np.ndarray, faces: np.ndarray, resolution: int = 128, pad: float = 0.05,
                  sign: str = "winding") -> GridSDF:
        """libigl signed distance (negative inside) on the bounding cube of the mesh plus `pad`. `sign`: "winding"
        (the fast generalised winding number: robust for the open, non-manifold meshes of scans and reconstructions)
        or "pseudonormal" (exact for closed manifold meshes only)."""
        import igl

        types = {"winding": igl.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER,
                 "pseudonormal": igl.SIGNED_DISTANCE_TYPE_PSEUDONORMAL}
        if sign not in types:
            raise ValueError(f"sign {sign!r}")

        v = np.asarray(vertices, np.float64)
        lo, hi = v.min(0), v.max(0)
        half = float((hi - lo).max()) / 2.0 + pad
        step = 2.0 * half / (resolution - 1)
        origin = (lo + hi) / 2.0 - half
        ax = origin[None, :] + step * np.arange(resolution)[:, None]
        g = np.stack(np.meshgrid(ax[:, 0], ax[:, 1], ax[:, 2], indexing="ij"), -1).reshape(-1, 3)
        sd = igl.signed_distance(g, v, np.asarray(faces, np.int64), sign_type=types[sign])[0]
        return GridSDF(np.asarray(sd, np.float64).reshape((resolution,) * 3), origin, float(step))

    @staticmethod
    def from_convex_parts(parts: list[np.ndarray], resolution: int = 128, pad: float = 0.05) -> GridSDF:
        """The union of closed convex parts (each given by its vertices; its convex hull is meshed). Distance: one
        libigl unsigned call on the soup of all hull triangles (outside the union the nearest soup triangle is on the
        union's boundary, since a buried face is farther, so the distance is exact there; inside, it is the depth to
        the nearest hull face, a lower bound). Sign: negative iff a node is inside some hull (its half-spaces). The executor's collision geometry is exactly such a union, and a scanned/reconstructed visual mesh is
        often open, which breaks any inside test."""
        import igl
        from scipy.spatial import ConvexHull

        V, Fs, off, hulls = [], [], 0, []
        for v in parts:
            v = np.asarray(v, np.float64)
            hull = ConvexHull(v)
            hulls.append((hull.equations, v[hull.vertices].min(0), v[hull.vertices].max(0)))
            f = hull.simplices.astype(np.int64)
            n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
            flip = np.einsum("ij,ij->i", n, v[f[:, 0]] - v[hull.vertices].mean(0)) < 0
            f[flip] = f[flip][:, ::-1]
            V.append(v)
            Fs.append(f + off)
            off += v.shape[0]
        allv = np.concatenate(V)
        lo, hi = allv.min(0), allv.max(0)
        half = float((hi - lo).max()) / 2.0 + pad
        step = 2.0 * half / (resolution - 1)
        origin = (lo + hi) / 2.0 - half
        ax = origin[None, :] + step * np.arange(resolution)[:, None]
        g = np.stack(np.meshgrid(ax[:, 0], ax[:, 1], ax[:, 2], indexing="ij"), -1).reshape(-1, 3)
        ud = np.asarray(igl.signed_distance(g, allv, np.concatenate(Fs),
                                            sign_type=igl.SIGNED_DISTANCE_TYPE_UNSIGNED)[0], np.float64)
        inside = np.zeros(g.shape[0], bool)
        for eq, vlo, vhi in hulls:
            box = np.flatnonzero(np.all((g >= vlo) & (g <= vhi), axis=1))
            if box.size:
                inside[box[np.all(g[box] @ eq[:, :3].T + eq[:, 3] <= 0.0, axis=1)]] = True
        sd = np.where(inside, -ud, ud)
        return GridSDF(np.asarray(sd, np.float64).reshape((resolution,) * 3), origin, float(step))

    def query(self, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(phi, grad) at points p (..., 3). Outside the grid (far field): the closest surface point q of the clamped
        point pc (q = pc - phi(pc) n(pc)) stands in for p's, so phi = |p - q| (outside) and grad = (p - q) / |p - q|:
        the distance and its direction stay meaningful beyond the 5 cm pad, where (a) still probes up to 2 d_max."""
        p = np.asarray(p, np.float64)
        G = self.values.shape[0]
        u = (p - self.origin) / self.step
        uc = np.clip(u, 0.0, G - 1.0)
        i = np.minimum(np.floor(uc).astype(np.int64), G - 2)
        f = uc - i
        x, y, z = i[..., 0], i[..., 1], i[..., 2]
        fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
        V = self.values
        c = {(a, b, e): V[x + a, y + b, z + e] for a in (0, 1) for b in (0, 1) for e in (0, 1)}
        gx, gy, gz = 1.0 - fx, 1.0 - fy, 1.0 - fz
        phi = (c[0, 0, 0] * gx * gy * gz + c[1, 0, 0] * fx * gy * gz + c[0, 1, 0] * gx * fy * gz
               + c[1, 1, 0] * fx * fy * gz + c[0, 0, 1] * gx * gy * fz + c[1, 0, 1] * fx * gy * fz
               + c[0, 1, 1] * gx * fy * fz + c[1, 1, 1] * fx * fy * fz)
        dx = ((c[1, 0, 0] - c[0, 0, 0]) * gy * gz + (c[1, 1, 0] - c[0, 1, 0]) * fy * gz
              + (c[1, 0, 1] - c[0, 0, 1]) * gy * fz + (c[1, 1, 1] - c[0, 1, 1]) * fy * fz)
        dy = ((c[0, 1, 0] - c[0, 0, 0]) * gx * gz + (c[1, 1, 0] - c[1, 0, 0]) * fx * gz
              + (c[0, 1, 1] - c[0, 0, 1]) * gx * fz + (c[1, 1, 1] - c[1, 0, 1]) * fx * fz)
        dz = ((c[0, 0, 1] - c[0, 0, 0]) * gx * gy + (c[1, 0, 1] - c[1, 0, 0]) * fx * gy
              + (c[0, 1, 1] - c[0, 1, 0]) * gx * fy + (c[1, 1, 1] - c[1, 1, 0]) * fx * fy)
        grad = np.stack([dx, dy, dz], -1) / self.step
        far = np.any(u != uc, axis=-1)
        if np.any(far):
            pc = self.origin + uc * self.step
            q = pc - phi[..., None] * grad / np.maximum(np.linalg.norm(grad, axis=-1, keepdims=True), 1e-12)
            off = p - q
            dist = np.linalg.norm(off, axis=-1)
            phi = np.where(far, dist, phi)
            grad = np.where(far[..., None], off / np.maximum(dist, 1e-12)[..., None], grad)
        return phi, grad

    def value(self, p: np.ndarray) -> np.ndarray:
        """phi alone (the hot path of the offset search): the same trilinear value as `query`, gathered through flat
        indices; points outside the grid fall back to `query`'s far field."""
        p = np.asarray(p, np.float64)
        G = self.values.shape[0]
        u = (p - self.origin) / self.step
        uc = np.clip(u, 0.0, G - 1.0)
        i = np.minimum(np.floor(uc).astype(np.int64), G - 2)
        f = uc - i
        flat = self.values.reshape(-1)
        base = (i[..., 0] * G + i[..., 1]) * G + i[..., 2]
        fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
        c00 = flat[base] * (1.0 - fz) + flat[base + 1] * fz
        c01 = flat[base + G] * (1.0 - fz) + flat[base + G + 1] * fz
        c10 = flat[base + G * G] * (1.0 - fz) + flat[base + G * G + 1] * fz
        c11 = flat[base + G * G + G] * (1.0 - fz) + flat[base + G * G + G + 1] * fz
        phi = (c00 * (1.0 - fy) + c01 * fy) * (1.0 - fx) + (c10 * (1.0 - fy) + c11 * fy) * fx
        far = np.any(u != uc, axis=-1)
        if np.any(far):
            phi[far] = self.query(p[far])[0]
        return phi


@dataclass(frozen=True)
class Segment:
    """One contact segment [start, end] (inclusive frames) and what the gates did with it."""

    start: int
    end: int
    d_raw: float              # argmin of J_k before any gate
    evidence: float           # fraction of rows with residual <= l_max at d_raw
    status: str               # "applied" | "capped" | "no_evidence"
    d: float                  # offset after the per-segment gates (before the windowed cap)


@dataclass(frozen=True, eq=False)
class ContactRefResult:
    rays: np.ndarray              # (n, 3) unit rays
    candidate: np.ndarray         # (n,) bool
    lateral: np.ndarray           # (n,) min lateral distance (inf where no vertex is within d_max along the ray)
    segments: tuple[Segment, ...]
    window_mean_before: float     # mean over the scoring window of |d| before the windowed cap
    beta: float                   # windowed-cap factor (1 when it does not bind)
    field: np.ndarray             # (n,) applied offset per frame (after every gate)
    extra: dict = field(default_factory=dict)


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """(..., 4) wxyz -> (..., 3, 3)."""
    q = np.asarray(q, np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                     2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                     2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(*q.shape[:-1], 3, 3)


def camera_center_exact(qpos_obj: np.ndarray, centering_offset: np.ndarray) -> np.ndarray:
    """The camera-frame origin in the scene frame, from DO AS I DO's `process_dataset` composition: scene = R x_cam -
    world_offset with world_offset = (o_x, o_y, min-z lift) and `centering_offset` o = R t_cam[0], so frame 0's object
    sits at (0, 0, o_z - lift) and the camera at -world_offset = qpos_obj[0, :3] - o. Valid only when frame 0's object
    was not frozen to a later frame (`process_dataset` freezes the object before the first valid hand frame): refuses
    unless frame 0's object x, y are exactly 0."""
    q0 = np.asarray(qpos_obj, np.float64)[0, :3]
    if np.abs(q0[:2]).max() > 1e-9:
        raise ValueError(f"frame 0's object is not at the centring origin ({q0[:2]}): use the raw-dir camera route")
    return q0 - np.asarray(centering_offset, np.float64)


def camera_rays(wrist: np.ndarray, center: np.ndarray) -> np.ndarray:
    """Unit rays from the camera centre through the wrist, per frame (n, 3)."""
    v = np.asarray(wrist, np.float64) - np.asarray(center, np.float64)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def find_segments(candidate: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    """Maximal runs [s, e] (inclusive) of True with length >= min_len."""
    c = np.concatenate([[False], np.asarray(candidate, bool), [False]])
    edges = np.flatnonzero(np.diff(c.astype(np.int8)))
    return [(int(s), int(e) - 1) for s, e in zip(edges[::2], edges[1::2], strict=True) if e - s >= min_len]


def offset_field(n: int, segs: list[tuple[int, int, float]], ramp_frames: int) -> np.ndarray:
    """d(f) (module docstring): segment values; a gap of <= 2 ramp frames between two segments is interpolated from
    one value to the next; longer gaps and the clip's ends ramp linearly to 0 over min(ramp, free interval)."""
    d = np.zeros(n)
    segs = sorted(segs)
    for s, e, v in segs:
        d[s:e + 1] = v
    for k, (s, e, v) in enumerate(segs):
        if k == 0 or s - segs[k - 1][1] - 1 > 2 * ramp_frames:          # clip start or a long gap: ramp to 0
            L_l = min(ramp_frames, s)
            for j in range(1, L_l + 1):
                d[s - j] = v * (1.0 - j / (L_l + 1))
        nxt = segs[k + 1] if k + 1 < len(segs) else None
        if nxt is not None and nxt[0] - e - 1 <= 2 * ramp_frames:        # a short gap: to the next value
            g = nxt[0] - e - 1
            for j in range(1, g + 1):
                d[e + j] = v + (nxt[2] - v) * j / (g + 1)
        else:
            L_r = min(ramp_frames, n - 1 - e)
            for j in range(1, L_r + 1):
                d[e + j] = v * (1.0 - j / (L_r + 1))
    return d


def field_on_rows(d_frames: np.ndarray, ref_dt: float, sim_dt: float, W: int, n_rows: int) -> np.ndarray:
    """d on the executor grid with DO AS I DO's `load_data` sampling (module docstring), in float64."""
    h = int(np.asarray(d_frames).shape[0])
    k = max(1, round(ref_dt / sim_dt))
    x = np.linspace(0.0, h - 1, h * k)
    out = np.interp(x, np.arange(h), np.asarray(d_frames, np.float64))
    out = np.concatenate([np.repeat(out[:1], W), out])
    if out.shape[0] < n_rows:
        out = np.concatenate([out, np.repeat(out[-1:], n_rows - out.shape[0])])
    return out[:n_rows]


def window_scale(d_frames: np.ndarray, ref_dt: float, sim_dt: float, W: int, T: int, grace: int,
                 c_w: float) -> tuple[float, float]:
    """(mean over executor rows [W + grace, T) of |d|, beta): beta = c_w / mean if mean > c_w, else 1."""
    rows = field_on_rows(d_frames, ref_dt, sim_dt, W, T)
    m = float(np.abs(rows[W + grace:T]).mean()) if T > W + grace else 0.0
    return m, (c_w / m if m > c_w else 1.0)


def _local(points: np.ndarray, rays: np.ndarray, obj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Points (n, V, 3) and rays (n, 3) in the object body frame of each frame."""
    R = quat_to_mat(obj[:, 3:7])
    pl = np.einsum("nij,nvi->nvj", R, points - obj[:, None, :3])
    rl = np.einsum("nij,ni->nj", R, rays)
    return pl, rl


def _candidates(sdf: GridSDF, Y: np.ndarray, rl: np.ndarray, cfg: ContactRefConfig) -> tuple[np.ndarray, np.ndarray]:
    phi, g = sdf.query(Y)
    v = phi[..., None] * g / np.maximum(np.linalg.norm(g, axis=-1, keepdims=True), 1e-12)
    a = np.einsum("nkc,nc->nk", v, rl)
    lat = np.linalg.norm(v - a[..., None] * rl[:, None, :], axis=-1)
    lat = np.where(np.abs(a) <= cfg.d_max, lat, np.inf).min(axis=1)
    return lat <= cfg.l_max, lat


def _J(sdf: GridSDF, Y: np.ndarray, rl: np.ndarray, d: np.ndarray, cfg: ContactRefConfig) -> np.ndarray:
    """J(d) for each d (D,), segment patch Y (S, k, 3), rays rl (S, 3); chunked over d."""
    out = np.empty(d.shape[0])
    chunk = max(1, 2_000_000 // max(1, Y.shape[0] * Y.shape[1]))
    for i in range(0, d.shape[0], chunk):
        dd = d[i:i + chunk]
        phi = sdf.value(Y[None] + dd[:, None, None, None] * rl[None, :, None, :])
        pen = np.maximum(-phi - cfg.eps_pen, 0.0)
        m = phi.min(axis=2)
        fit = np.maximum(m, 0.0) + np.maximum(-m - cfg.fit_band, 0.0)
        out[i:i + chunk] = (fit ** 2).sum(axis=1) + cfg.lam_pen * (pen ** 2).sum(axis=(1, 2))
    return out


def _argmin(sdf: GridSDF, Y: np.ndarray, rl: np.ndarray, cfg: ContactRefConfig) -> float:
    span = 2.0 * cfg.d_max
    n = round(span / cfg.grid_step)
    grid = np.linspace(-span, span, 2 * n + 1)
    J = _J(sdf, Y, rl, grid, cfg)
    ok = np.flatnonzero(J <= J.min() + Y.shape[0] * 1e-6)
    d0 = float(grid[ok[np.argmin(np.abs(grid[ok]))]])
    lo, hi = d0 - cfg.grid_step, d0 + cfg.grid_step
    g = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = hi - g * (hi - lo), lo + g * (hi - lo)
    fa, fb = _J(sdf, Y, rl, np.array([a, b]), cfg)
    for _ in range(40):
        if fa <= fb:
            hi, b, fb = b, a, fa
            a = hi - g * (hi - lo)
            fa = _J(sdf, Y, rl, np.array([a]), cfg)[0]
        else:
            lo, a, fa = a, b, fb
            b = lo + g * (hi - lo)
            fb = _J(sdf, Y, rl, np.array([b]), cfg)[0]
    best = 0.5 * (lo + hi)
    J0 = _J(sdf, Y, rl, np.array([d0, best]), cfg)
    return best if J0[1] <= J0[0] else d0


def contact_reference(sdf: GridSDF, verts: np.ndarray, wrist: np.ndarray, obj: np.ndarray, center: np.ndarray,
                      *, ref_dt: float, sim_dt: float, W: int, T: int,
                      cfg: ContactRefConfig | None = None) -> ContactRefResult:
    """(a) on one clip (module docstring). verts (n, V, 3), wrist (n, 3), obj (n, 7) wxyz, center (3,); W, T and the
    grid steps define the scoring window [W + cfg.grace_rows, T) on the executor grid; `cfg` None = the defaults."""
    cfg = cfg if cfg is not None else ContactRefConfig()
    verts = np.asarray(verts, np.float64)
    obj = np.asarray(obj, np.float64)
    n = verts.shape[0]
    rays = camera_rays(wrist, center)
    pl, rl = _local(verts, rays, obj)
    phi0 = sdf.value(pl)
    k = min(cfg.patch_k, verts.shape[1])
    if cfg.patch_select not in ("signed", "unsigned"):
        raise ValueError(f"patch_select {cfg.patch_select!r}")
    key = phi0 if cfg.patch_select == "signed" else np.abs(phi0)
    idx = np.argpartition(key, k - 1, axis=1)[:, :k]
    Y = np.take_along_axis(pl, idx[..., None], axis=1)
    cand, lat = _candidates(sdf, Y, rl, cfg)
    if cfg.carry_speed is not None:
        # Opt-in, NOT the rev-1b rule: a candidate row must also be a CARRY row, the object reference moving
        # >= carry_speed (central difference over +-2 frames, the CPU proxy's own carry definition).
        lo, hi = np.clip(np.arange(n) - 2, 0, n - 1), np.clip(np.arange(n) + 2, 0, n - 1)
        speed = np.linalg.norm(obj[hi, :3] - obj[lo, :3], axis=1) / np.maximum(hi - lo, 1) / ref_dt
        cand = cand & (speed >= cfg.carry_speed)
    segs: list[Segment] = []
    for s, e in find_segments(cand, cfg.min_seg_frames):
        Ys, rs = Y[s:e + 1], rl[s:e + 1]
        d_raw = _argmin(sdf, Ys, rs, cfg)
        resid = np.abs(sdf.value(Ys + d_raw * rs[:, None, :])).min(axis=1)
        ev = float((resid <= cfg.l_max).mean())
        if abs(d_raw) > cfg.d_max:
            status, d = "capped", 0.0
        elif ev < cfg.evidence_frac:
            status, d = "no_evidence", 0.0
        else:
            status, d = "applied", d_raw
        segs.append(Segment(s, e, d_raw, ev, status, d))
    ramp = round(cfg.ramp_s / ref_dt)
    d_f = offset_field(n, [(sg.start, sg.end, sg.d) for sg in segs], ramp)
    m, beta = window_scale(d_f, ref_dt, sim_dt, W, T, cfg.grace_rows, cfg.c_w)
    return ContactRefResult(rays=rays, candidate=cand, lateral=lat, segments=tuple(segs), window_mean_before=m,
                            beta=beta, field=beta * d_f)


def shift_keypoints(kp: Mapping[str, np.ndarray], side: str, d_frames: np.ndarray,
                    rays: np.ndarray) -> dict[str, np.ndarray]:
    """h' = h + d(f) r_f on every hand keypoint of `side` (positions only; the wrist/fingertip quaternions, the object
    and the other side are copied unchanged). Returns a new dict; `kp` is not modified."""
    out = {k: np.array(v, copy=True) for k, v in kp.items()}
    sh = np.asarray(d_frames, np.float64)[:, None] * np.asarray(rays, np.float64)
    for key in (t.format(s=side) for t in _HAND_KEYS):
        a = out[key]
        if a.shape[0] != sh.shape[0]:
            raise ValueError(f"{key}: {a.shape[0]} frames, offset field has {sh.shape[0]}")
        extra = (None,) * (a.ndim - 2)
        delta = sh[(slice(None), *extra, slice(None))]
        a[..., :3] = (a[..., :3].astype(np.float64) + delta).astype(a.dtype)
    return out
