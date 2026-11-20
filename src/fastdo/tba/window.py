"""TBA / TBA-V on one window's stored artefacts (runs on CPU after the production solve).

`load_window` reads a production work dir (`solve.npz`, `contact.npz` when the contact stage ran, `tracks.npz`,
`geometry.npz`, `hands.npz`, `intrinsics.npz`, `mesh.obj`, `sdf.npz`). `run_tba` then:

1. anchors: each keyframe query's ray (under the BA's K) is cast against BASE's posed template at the keyframe;
2. R1 (`fastdo.tba.bundle.solve_bundle`) from BASE's relative poses and anchors;
3. the gauge: "K" holds BASE's window gauge (l_K), "consensus" takes the pointmap's (TBA-V-C);
4. the template is re-posed RIGIDLY on the gauged anchor cloud (Kabsch of the anchors' canonical points at BASE's
   scale s; `finish` says why not s exp(l)), so its scale is tied to the gauge and never re-read from the shape;
   BA rows take ``M_k`` composed with that keyframe pose; other rows keep BASE's pose;
5. R2 (`cfg.carve`): `fastdo.tba.carve` on the re-posed trajectory; re-meshed only when carving applies.

BASE = the written trajectory (post-contact when `contact.npz` exists) unless `base_poses` is given.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from fastdo.tba.bundle import (
    BundleProblem,
    BundleResult,
    apply_log_gauge,
    gauge_consensus,
    gauge_k,
    solve_bundle,
)
from fastdo.tba.carve import (
    CarveResult,
    carve_template,
    hand_silhouette,
    object_region,
    remesh,
    voxel_grid,
)
from fastdo.tba.config import TbaConfig
from fastdo.tba.geometry import kabsch, pixel_rays, posed_points, ray_mesh_depth, relative_poses

__all__ = ["FrameEvidence", "Prepared", "TbaOutput", "Window", "finish", "load_window", "prepare", "run_tba", "scaled_K",
           "surface_centroid"]

logger = logging.getLogger(__name__)
SIDES = ("left", "right")


@dataclass(frozen=True)
class Window:
    """One window's stored artefacts (module docstring)."""

    wd: Path
    frame_idx: np.ndarray
    K: np.ndarray
    image_hw: tuple[int, int]
    tracks: np.ndarray
    vis: np.ndarray
    query_uv: np.ndarray
    key: int
    poses_final: np.ndarray
    poses_solve: np.ndarray
    scale: float
    mesh_v: np.ndarray
    mesh_f: np.ndarray
    sdf_values: np.ndarray
    sdf_origin: np.ndarray
    sdf_step: float
    obj_points: np.ndarray
    obj_offsets: np.ndarray
    hand_verts: dict[str, np.ndarray] = field(default_factory=dict)
    hand_valid: dict[str, np.ndarray] = field(default_factory=dict)
    hand_faces: dict[str, np.ndarray] = field(default_factory=dict)
    has_contact: bool = False


def load_window(wd: Path) -> Window:
    """Read a production work dir (module docstring). Raises ValueError if the artefacts' rows disagree."""
    import trimesh

    wd = Path(wd)
    s = np.load(wd / "solve.npz", allow_pickle=False)
    fr = np.asarray(s["frame_idx"], np.int64)
    poses_solve = np.asarray(s["poses"], np.float64)
    poses_final, has_contact = poses_solve, False
    if (wd / "contact.npz").is_file():
        c = np.load(wd / "contact.npz", allow_pickle=False)
        if not np.array_equal(np.asarray(c["frame_idx"], np.int64), fr):
            raise ValueError(f"{wd}: contact rows differ from solve rows")
        poses_final, has_contact = np.asarray(c["poses"], np.float64), True
    t = np.load(wd / "tracks.npz", allow_pickle=False)
    g = np.load(wd / "geometry.npz", allow_pickle=False)
    for name, arr in (("tracks", t["frame_idx"]), ("geometry", g["frame_idx"])):
        if not np.array_equal(np.asarray(arr, np.int64), fr):
            raise ValueError(f"{wd}: {name} rows differ from solve rows")
    key = int(t["keyframe_index"])
    if int(g["keyframe_index"]) != key:
        raise ValueError(f"{wd}: geometry and tracks keyframes differ")
    mesh = trimesh.load(str(wd / "mesh.obj"), process=False, force="mesh")
    sdf = np.load(wd / "sdf.npz", allow_pickle=False)
    h = np.load(wd / "hands.npz", allow_pickle=False)
    if not np.array_equal(np.asarray(h["frame_idx"], np.int64), fr):
        raise ValueError(f"{wd}: hands rows differ from solve rows")
    verts, valid, faces = {}, {}, {}
    for side in SIDES:
        if f"{side}_vertices" in h.files:
            verts[side] = np.asarray(h[f"{side}_vertices"], np.float64)
            valid[side] = np.asarray(h[f"{side}_valid"], bool)
            faces[side] = np.asarray(h[f"{side}_faces"], np.int64)
    shape = tuple(int(x) for x in g["image_shape"])
    return Window(wd=wd, frame_idx=fr, K=np.asarray(np.load(wd / "intrinsics.npz")["K"], np.float64),
                  image_hw=(shape[1], shape[2]), tracks=np.asarray(t["tracks"], np.float64),
                  vis=np.asarray(t["vis"], bool), query_uv=np.asarray(t["query_uv"], np.float64), key=key,
                  poses_final=poses_final, poses_solve=poses_solve, scale=float(s["scale"]),
                  mesh_v=np.asarray(mesh.vertices, np.float64), mesh_f=np.asarray(mesh.faces, np.int64),
                  sdf_values=np.asarray(sdf["values"], np.float64), sdf_origin=np.asarray(sdf["origin"], np.float64),
                  sdf_step=float(sdf["step"]), obj_points=np.asarray(g["object_points"], np.float32),
                  obj_offsets=np.asarray(g["object_offsets"], np.int64), hand_verts=verts, hand_valid=valid,
                  hand_faces=faces, has_contact=has_contact)


def scaled_K(K: np.ndarray, mult: float) -> np.ndarray:
    """K with both focal lengths times `mult` (principal point kept): the S1-F / KA-GT-F operating points."""
    out = np.array(K, np.float64, copy=True)
    out[0, 0] *= mult
    out[1, 1] *= mult
    return out


def surface_centroid(vertices: np.ndarray, faces: np.ndarray, n: int = 10_000, seed: int = 0) -> np.ndarray:
    """The scorers' template centre: mean of `n` area-uniform surface samples (`e1_score.mesh_centroid`)."""
    import trimesh

    mesh = trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=False)
    pts, _ = trimesh.sample.sample_surface(mesh, n, seed=seed)
    return np.asarray(pts, np.float64).mean(axis=0)


class FrameEvidence:
    """Per-row object regions, UNKNOWN masks and pointmap depths from the stored artefacts (stored K: image facts)."""

    def __init__(self, win: Window, cfg: TbaConfig) -> None:
        self.win, self.cfg = win, cfg
        self.image_hw = win.image_hw
        self._region: dict[int, np.ndarray] = {}
        self._unknown: dict[int, np.ndarray] = {}

    def points(self, k: int) -> np.ndarray:
        o = self.win.obj_offsets
        return self.win.obj_points[o[k]:o[k + 1]].astype(np.float64)

    def region(self, k: int) -> np.ndarray:
        if k not in self._region:
            self._region[k] = object_region(self.points(k), self.win.K, self.image_hw, self.cfg.close_px)
        return self._region[k]

    def unknown(self, k: int) -> np.ndarray:
        if k not in self._unknown:
            if not self.cfg.hand_unknown:
                self._unknown[k] = np.zeros(self.image_hw, bool)
            else:
                vs = [self.win.hand_verts[s][k] for s in self.win.hand_verts if self.win.hand_valid[s][k]]
                faces = next(iter(self.win.hand_faces.values())) if self.win.hand_faces else np.zeros((0, 3), int)
                self._unknown[k] = hand_silhouette(vs, faces, self.win.K, self.image_hw, self.cfg.hand_dilate_px)
        return self._unknown[k]

    def z_obs(self, tracks: np.ndarray, vis: np.ndarray) -> np.ndarray:
        """(T,J) pointmap camera z at the rounded tracked pixel (NaN where the pixel holds no object point)."""
        h, w = self.image_hw
        out = np.full(vis.shape, np.nan)
        for k in range(vis.shape[0]):
            p = self.points(k)
            p = p[np.isfinite(p).all(axis=1) & (p[:, 2] > 0)]
            if not len(p):
                continue
            uv = np.rint(np.stack([self.win.K[0, 0] * p[:, 0] / p[:, 2] + self.win.K[0, 2],
                                   self.win.K[1, 1] * p[:, 1] / p[:, 2] + self.win.K[1, 2]], 1)).astype(np.int64)
            ok = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
            lut = {int(a): float(z) for a, z in zip(uv[ok, 1] * w + uv[ok, 0], p[ok, 2])}
            q = np.rint(tracks[k]).astype(np.int64)
            for i in np.flatnonzero(vis[k]):
                out[k, i] = lut.get(int(q[i, 1] * w + q[i, 0]), np.nan)
        return out


@dataclass(frozen=True)
class TbaOutput:
    """An arm's trajectory and template: ``X = scale R p + t`` with ``p`` on (`mesh_v`, `mesh_f`)."""

    poses: np.ndarray
    scale: float
    mesh_v: np.ndarray
    mesh_f: np.ndarray
    carved: bool
    record: dict[str, Any]
    carve: CarveResult | None = None


def _anchors(win: Window, K: np.ndarray, poses: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
    rays = pixel_rays(win.query_uv, K)
    Vc = posed_points(win.mesh_v, poses[win.key], scale)
    return rays, ray_mesh_depth(rays, Vc, win.mesh_f)


@dataclass(frozen=True)
class Prepared:
    """R1 solved once for a window at one K and BASE (`prepare`); `finish` derives TBA, TBA-V and TBA-V-C from it.

    `d_hit` are the keyframe rays' hits on BASE's posed template (what the template's canonical anchor points are
    read from); `problem.d_base` are the BA's prior centres, equal to `d_hit` unless a mutation (M-ANCHOR) moved them.
    """

    base_poses: np.ndarray
    K: np.ndarray
    rays: np.ndarray
    d_hit: np.ndarray
    anchored: np.ndarray
    problem: BundleProblem
    result: BundleResult
    seconds: float


def prepare(win: Window, cfg: TbaConfig, base_poses: np.ndarray | None = None, tracks: np.ndarray | None = None,
            anchor_offsets: np.ndarray | None = None, base_scale: float | None = None) -> Prepared:
    """Anchors on BASE's posed template and R1 (module docstring steps 1-2).

    `tracks` replaces the stored tracks (M-TRACK) and `anchor_offsets` (J,) metres are added to BASE's anchor depths
    before the solve (M-ANCHOR: the prior centre and the init both move; the template hits do not); both default to
    the stored inputs. `base_poses` / `base_scale` replace BASE's trajectory (TBA-V-RC: the pre-contact solve).
    """
    t0 = time.perf_counter()
    P = np.asarray(win.poses_final if base_poses is None else base_poses, np.float64)
    s = win.scale if base_scale is None else float(base_scale)
    K = scaled_K(win.K, cfg.focal_mult)
    rays, d_hit = _anchors(win, K, P, s)
    anchored = np.isfinite(d_hit) & (d_hit > 0)
    d_prior = d_hit if anchor_offsets is None else d_hit + np.asarray(anchor_offsets, np.float64)
    anchored &= np.isfinite(d_prior) & (d_prior > 0)
    trk = win.tracks if tracks is None else np.asarray(tracks, np.float64)
    prob = BundleProblem(K=K, tracks=trk, vis=win.vis, key=win.key, rays=rays, anchored=anchored,
                         d_base=np.where(anchored, d_prior, np.nan), M0=relative_poses(P, win.key))
    res = solve_bundle(prob, cfg)
    return Prepared(P, K, rays, np.where(anchored, d_hit, np.nan), anchored, prob, res, time.perf_counter() - t0)


def finish(win: Window, cfg: TbaConfig, prep: Prepared, evidence: FrameEvidence | None = None,
           carve: bool | None = None, gauge: str | None = None, base_scale: float | None = None) -> TbaOutput:
    """Gauge, rigid re-pose and (optionally) R2 on a prepared window (module docstring steps 3-5).

    `carve` / `gauge` default to `cfg.carve` / `cfg.gauge`; `base_scale` must be `prepare`'s.

    Gauge K holds BASE's window gauge: the BA solution is scaled by exp(l_K), which puts its anchors at BASE's depths
    in the median, and the template keeps BASE's scale ``s``. (The BA has no scale of its own: its anchors start at,
    and are tied by the prior to, BASE's template hits at scale ``s``, so its implied template scale is
    ``s exp(-l_K)`` and the registered "anchors, translations, template scale scaled by l_K" gives ``s``. Scaling
    ``s`` by exp(l_K) instead would double-count the gauge: a BA that only drifted along the gauge would return a
    template of the wrong size for BASE's anchors.) Gauge "consensus" (TBA-V-C) is then g_exp(l_C - l_K) of the whole
    window -- every translation, the scale -- with l_C the pointmap's: it replaces lambda, as registered.
    """
    t0 = time.perf_counter()
    carve = cfg.carve if carve is None else carve
    gauge = cfg.gauge if gauge is None else gauge
    if gauge not in ("K", "consensus"):
        raise ValueError(f"gauge must be 'K' or 'consensus', got {gauge!r}")
    P, K, prob = prep.base_poses, prep.K, prep.problem
    s = win.scale if base_scale is None else float(base_scale)
    rays, anchored, d_hit = prep.rays, prep.anchored, prep.d_hit
    ev = evidence if evidence is not None else FrameEvidence(win, cfg)
    ell_k = gauge_k(prob, prep.result, d_hit)
    res = apply_log_gauge(prep.result, ell_k)
    rec: dict[str, Any] = {"n_rows": len(P), "n_ba_rows": int(res.ba_frames.sum()), "n_anchored": int(anchored.sum()),
                           "solved": res.solved, "ell_k": ell_k, "nfev": res.nfev, "cost_init": res.cost_init,
                           "cost_final": res.cost_final, "focal_mult": cfg.focal_mult, "gauge": gauge,
                           "carve_enabled": bool(carve)}
    poses = P.copy()
    if res.solved:
        pk = P[win.key]
        canon = ((d_hit[anchored, None] * rays[anchored]) - pk[:3, 3]) @ pk[:3, :3] / s     # R^T (X - t) / s
        Y = res.depths[anchored, None] * rays[anchored]
        R1, t1 = kabsch(s * canon, Y)
        key_pose = np.eye(4)
        key_pose[:3, :3], key_pose[:3, 3] = R1, t1
        rows = res.ba_frames
        poses[rows] = res.M[rows] @ key_pose[None]
        dd = (res.depths - d_hit)[anchored]
        rec["anchor_shift_mm"] = {"median_abs": float(np.median(np.abs(dd)) * 1e3),
                                  "max_abs": float(np.max(np.abs(dd)) * 1e3)}
        rec["repose_rms_mm"] = float(np.sqrt(np.mean(np.sum((s * canon @ R1.T + t1 - Y) ** 2, 1))) * 1e3)
    mesh_v, mesh_f, carved, cres = win.mesh_v, win.mesh_f, False, None
    if carve:
        grid = voxel_grid(win.sdf_values, win.sdf_origin, win.sdf_step, cfg.voxel_upsample)
        cres = carve_template(grid, poses, s, K, ev, cfg)
        rec["carve"] = {"n_inside": len(cres.inside_idx), "removed_frac": cres.removed_frac,
                        "abstained": cres.abstained, "applied": cres.applied,
                        "median_views": float(np.median(cres.n_seen)) if len(cres.n_seen) else 0.0}
        if cres.applied:
            mesh_v, mesh_f = remesh(grid, cres)
            carved = True
    scale = s
    if gauge == "consensus":
        ell_c = gauge_consensus(prob, prep.result, ev.z_obs(prob.tracks, win.vis))
        rec["ell_c"] = ell_c
        lam = math.exp(ell_c - ell_k)
        poses = poses.copy()
        poses[..., :3, 3] *= lam
        scale = s * lam
    rec["seconds"] = prep.seconds + time.perf_counter() - t0
    return TbaOutput(poses=poses, scale=scale, mesh_v=mesh_v, mesh_f=mesh_f, carved=carved, record=rec, carve=cres)


def run_tba(win: Window, cfg: TbaConfig, base_poses: np.ndarray | None = None,
            evidence: FrameEvidence | None = None, tracks: np.ndarray | None = None,
            anchor_offsets: np.ndarray | None = None, base_scale: float | None = None) -> TbaOutput:
    """TBA (``cfg.carve`` False) or TBA-V on one window: `prepare` then `finish` (module docstring)."""
    return finish(win, cfg, prepare(win, cfg, base_poses, tracks, anchor_offsets, base_scale), evidence,
                  base_scale=base_scale)
