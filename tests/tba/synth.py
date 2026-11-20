"""Synthetic windows with a known answer: a rigid convex object, GT poses with parallax, GT tracks.

The object is an ellipsoid mesh (convex, so back-face culling IS the depth test for its own visibility). Tracks are
the exact projections of 96 keyframe-visible surface points (plus optional pixel noise); a point is visible in a
row when its normal faces the camera there.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from fastdo.tba.geometry import project, rotvec_to_matrix

F_PX, HW = 600.0, (480, 640)
K = np.array([[F_PX, 0.0, 320.0], [0.0, F_PX, 240.0], [0.0, 0.0, 1.0]])


@dataclass
class SynthWindow:
    mesh_v: np.ndarray          # canonical, unit bounding-box diagonal
    mesh_f: np.ndarray
    scale: float                # metres per canonical unit
    poses: np.ndarray           # (T,4,4) GT camera-from-object
    tracks: np.ndarray          # (T,J,2)
    vis: np.ndarray             # (T,J)
    query_uv: np.ndarray        # (J,2)
    key: int


def ellipsoid(axes: tuple[float, float, float] = (1.0, 0.7, 0.5), subdiv: int = 3) -> tuple[np.ndarray, np.ndarray]:
    m = trimesh.creation.icosphere(subdivisions=subdiv)
    v = np.asarray(m.vertices) * np.asarray(axes)
    v = v / np.linalg.norm(v.max(0) - v.min(0))
    return v, np.asarray(m.faces, np.int64)


def make_window(n_rows: int = 24, theta_deg: float = 40.0, depth: float = 0.6, size_m: float = 0.15,
                n_tracks: int = 96, noise_px: float = 0.0, seed: int = 0) -> SynthWindow:
    """GT trajectory: the object turns by `theta_deg` about a tilted vertical axis while drifting sideways."""
    rng = np.random.default_rng(seed)
    v, f = ellipsoid()
    scale = size_m
    axis = np.array([0.15, 1.0, 0.1])
    axis /= np.linalg.norm(axis)
    poses = np.zeros((n_rows, 4, 4))
    R0 = rotvec_to_matrix(np.array([0.3, 0.2, 0.1]))
    for k in range(n_rows):
        a = np.radians(theta_deg) * k / (n_rows - 1)
        poses[k, :3, :3] = rotvec_to_matrix(axis * a) @ R0
        poses[k, :3, 3] = (0.05 * np.sin(k / 5.0), -0.02 + 0.03 * k / n_rows, depth + 0.05 * np.cos(k / 7.0))
        poses[k, 3, 3] = 1.0
    mesh = trimesh.Trimesh(v, f, process=False)
    key = 0
    pts, fi = trimesh.sample.sample_surface(mesh, 4000, seed=seed)
    nrm = mesh.face_normals[fi]
    Xk = scale * pts @ poses[key, :3, :3].T + poses[key, :3, 3]
    facing = np.einsum("ij,ij->i", nrm @ poses[key, :3, :3].T, -Xk) > 0.05 * np.linalg.norm(Xk, axis=1)
    pick = rng.choice(np.flatnonzero(facing), n_tracks, replace=False)
    P, N = pts[pick], nrm[pick]
    tracks = np.zeros((n_rows, n_tracks, 2))
    vis = np.zeros((n_rows, n_tracks), bool)
    for k in range(n_rows):
        X = scale * P @ poses[k, :3, :3].T + poses[k, :3, 3]
        tracks[k] = project(X, K) + noise_px * rng.standard_normal((n_tracks, 2))
        vis[k] = np.einsum("ij,ij->i", N @ poses[k, :3, :3].T, -X) > 0.0
    tracks[key] = project(scale * P @ poses[key, :3, :3].T + poses[key, :3, 3], K)
    vis[key] = True
    return SynthWindow(v, f, scale, poses, tracks, vis, tracks[key].copy(), key)


def corrupt(poses: np.ndarray, scale: float, centroid: np.ndarray, lam: float, along_sd_m: float,
            seed: int = 1) -> tuple[np.ndarray, float]:
    """BASE = g_lam(GT) plus an OFF-orbit per-row translation along each row's centre ray (sd `along_sd_m`)."""
    rng = np.random.default_rng(seed)
    out = poses.copy()
    out[:, :3, 3] *= lam
    s = scale * lam
    c = s * out[:, :3, :3] @ centroid + out[:, :3, 3]
    u = c / np.linalg.norm(c, axis=1, keepdims=True)
    out[:, :3, 3] += (along_sd_m * rng.standard_normal(len(out)))[:, None] * u
    return out, s


def as_window(sw: SynthWindow, base_poses: np.ndarray, base_scale: float, with_sdf: bool = False,
              obj_points: list[np.ndarray] | None = None, hands: tuple[np.ndarray, np.ndarray] | None = None):
    """A `fastdo.tba.window.Window` holding BASE = (`base_poses`, `base_scale`) on the synthetic inputs; `hands`
    = ((T,V,3) camera-frame vertices, faces) is stored as a valid right hand."""
    from pathlib import Path

    from fastdo.tba.window import Window

    if with_sdf:
        from fastdo.solver.sdf import mesh_sdf_grid

        g = mesh_sdf_grid(sw.mesh_v, sw.mesh_f, resolution=48, padding=0.15)
        sdf_v, sdf_o, sdf_s = g.values.numpy(), g.origin.numpy(), float(g.step)
    else:
        sdf_v, sdf_o, sdf_s = np.ones((2, 2, 2)), np.zeros(3), 1.0
    pts = obj_points if obj_points is not None else [np.zeros((0, 3))] * len(sw.poses)
    offsets = np.concatenate([[0], np.cumsum([len(p) for p in pts])]).astype(np.int64)
    allp = np.concatenate(pts).astype(np.float32) if offsets[-1] else np.zeros((0, 3), np.float32)
    hv, hval, hf = {}, {}, {}
    if hands is not None:
        hv["right"], hf["right"] = np.asarray(hands[0], np.float64), np.asarray(hands[1], np.int64)
        hval["right"] = np.ones(len(sw.poses), bool)
    return Window(wd=Path("."), frame_idx=np.arange(len(sw.poses), dtype=np.int64), K=K, image_hw=HW,
                  tracks=sw.tracks, vis=sw.vis, query_uv=sw.query_uv, key=sw.key, poses_final=base_poses,
                  poses_solve=base_poses, scale=base_scale, mesh_v=sw.mesh_v, mesh_f=sw.mesh_f, sdf_values=sdf_v,
                  sdf_origin=sdf_o, sdf_step=sdf_s, obj_points=allp, obj_offsets=offsets, hand_verts=hv,
                  hand_valid=hval, hand_faces=hf)


# ---- carving scenes (R2 known answers) --------------------------------------------------------------------------------


def raster(V_cam: np.ndarray, faces: np.ndarray, hw: tuple[int, int] = HW) -> np.ndarray:
    """(H,W) bool silhouette of a camera-frame mesh (cv2 polygon fill, 4 fractional bits; all vertices in front)."""
    import cv2

    m = np.zeros(hw, np.uint8)
    uv = project(V_cam[np.asarray(faces)], K)
    for t in np.rint(uv * 16.0).astype(np.int32):
        cv2.fillConvexPoly(m, t, 1, lineType=cv2.LINE_8, shift=4)
    return m.astype(bool)


def lift(mask: np.ndarray, depth: float = 0.5) -> np.ndarray:
    """Camera-frame points that reproject exactly onto the pixel centres of `mask` (a stand-in pointmap)."""
    v, u = np.nonzero(mask)
    uv = np.stack([u, v], 1).astype(np.float64)
    return depth * np.concatenate([(uv - K[:2, 2]) / K[0, 0], np.ones((len(uv), 1))], axis=1)


@dataclass
class CarveScene:
    sw: SynthWindow
    hand_v: np.ndarray              # (T,Vh,3) camera-frame hand-sphere vertices
    hand_f: np.ndarray
    hand_canon: np.ndarray          # (3,) hand centre in the canonical frame
    hand_radius: float              # canonical units
    view_dir: np.ndarray            # (3,) keyframe viewing direction in the canonical frame
    obj_points: list[np.ndarray]    # per row: lifted visible-object pixels (object minus the hand in front)


def carve_scene(n_rows: int = 16, theta_deg: float = 70.0, hand_radius_m: float = 0.015, size_m: float = 0.22,
                seed: int = 0) -> CarveScene:
    """An ellipsoid (`size_m` diagonal) turning by `theta_deg` with a sphere 'hand' touching its camera-facing side.

    The production mask excludes occluding hand pixels, so each row's object region is the object silhouette minus
    the hand silhouette (the hand sits on the camera side of the object).
    """
    sw = make_window(n_rows=n_rows, theta_deg=theta_deg, size_m=size_m, seed=seed)
    key = sw.key
    R, t = sw.poses[key, :3, :3], sw.poses[key, :3, 3]
    c = sw.scale * R @ sw.mesh_v.mean(0) + t
    view = R.T @ (c / np.linalg.norm(c))                         # canonical direction of the keyframe ray
    # the canonical surface point nearest the camera along the keyframe's centre ray, offset 30% sideways
    side = np.cross(view, [0.0, 1.0, 0.0])
    side /= np.linalg.norm(side)
    target = sw.mesh_v.mean(0) - 0.6 * view + 0.15 * side
    surf = sw.mesh_v[np.argmin(np.linalg.norm(sw.mesh_v - target, axis=1))]
    nrm = surf - sw.mesh_v.mean(0)
    nrm /= np.linalg.norm(nrm)
    r_c = hand_radius_m / sw.scale
    hc = surf + r_c * nrm
    sph = trimesh.creation.icosphere(subdivisions=2, radius=r_c)
    hv = np.asarray(sph.vertices) + hc
    hand_v = np.stack([sw.scale * hv @ sw.poses[k, :3, :3].T + sw.poses[k, :3, 3] for k in range(n_rows)])
    pts = []
    for k in range(n_rows):
        Vo = sw.scale * sw.mesh_v @ sw.poses[k, :3, :3].T + sw.poses[k, :3, 3]
        pts.append(lift(raster(Vo, sw.mesh_f) & ~raster(hand_v[k], sph.faces)))
    return CarveScene(sw, hand_v, np.asarray(sph.faces, np.int64), hc, r_c, view, pts)


def sdf_on_grid(mesh_v: np.ndarray, mesh_f: np.ndarray, lo: np.ndarray, hi: np.ndarray, n: int = 56,
                shifts: np.ndarray | None = None):
    """A `fastdo.tba.carve.VoxelGrid` of the mesh's signed distance on a cube covering [lo, hi]; with `shifts`
    (S,3) the union (min) of the mesh translated by each shift (a swept, over-extended template)."""
    import igl

    from fastdo.tba.carve import VoxelGrid

    half = float((hi - lo).max()) / 2.0 * 1.1
    centre = (lo + hi) / 2.0
    step = 2.0 * half / (n - 1)
    origin = centre - half
    ax = origin[None, :] + step * np.arange(n)[:, None]
    g = np.stack(np.meshgrid(ax[:, 0], ax[:, 1], ax[:, 2], indexing="ij"), -1).reshape(-1, 3)
    vals = np.full(len(g), np.inf)
    for s in (np.zeros((1, 3)) if shifts is None else shifts):
        sd, *_ = igl.signed_distance(g - s, mesh_v.astype(np.float64), np.asarray(mesh_f, np.int64),
                                     sign_type=igl.SIGNED_DISTANCE_TYPE_PSEUDONORMAL)
        vals = np.minimum(vals, np.asarray(sd))
    return VoxelGrid(vals.reshape(n, n, n), origin, step)
