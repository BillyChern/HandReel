"""Camera-frame ground truth for the V2D Track 3 public split, built from its own stereo pair (CPU).

The split gives Vicon ``world_T_object`` and scanned meshes but NO camera-to-mocap extrinsic (README), so the metric
along-ray error of a monocular reconstruction cannot be scored until ``cam_T_world`` is constructed per frame. This
module constructs it WITHOUT our monocular pipeline (an audit must read an independent construction,
feedback-an-audit-cannot-read-what-the-pipeline-wrote):

1. the ego stereo pair (left ``ego_cam_c``, right ``ego_cam_b``, 75 mm) is rectified with OpenCV's 14-coefficient
   rational model (:func:`stereo_rectification`), and rectified disparities become metric points in the LEFT camera
   frame (:func:`points_from_disparity`);
2. the episode's scans, posed by Vicon in the mocap world (their relative geometry is exact), are registered rigidly
   into that cloud by robust point-to-plane ICP (:func:`register_scene`), giving ``c_T_world`` per frame;
3. ``a_T_world = a_T_c @ c_T_world`` moves it to ``ego_cam_a``, the RGB stream our pipeline reads
   (:func:`EgoRig.a_T_c`), and ``a_T_world @ world_T_object`` is the camera-frame GT pose.

Calibration conventions (UNVERIFIED on pixels; decided label-free, :mod:`fastdo.bench.v2d_orientation`):
every 4x4 in ``camera_calibration.json`` is read as ``X_dst = T @ X_src``, and ``extrinsics_to_stereo_left`` targets
the device's ``CAM_B``. The file itself shows the ego extrinsics are expressed in 180-deg-rotated camera frames: the
exo device (``rotation_deg`` 0) has ``CAM_A`` at x = +0.0376 m and ``CAM_C`` at +0.0751 m in ``CAM_B``'s frame, the ego
device (same camera model, ``rotation_deg`` 180) has them at -0.0374 and -0.0755, i.e. ``Rz(pi)`` of the same layout.
That is why the file names ``ego_cam_c`` the LEFT camera of the ego pair ("a 180 degree rotation mirrors the baseline
direction"). The pair's ``left_to_right`` maps ``ego_cam_c`` into ``ego_cam_b`` and is identical to ``ego_cam_c``'s
``extrinsics_to_stereo_left``; its translation is (-B, 0, 0), so the right camera centre ``-R^T T`` lies at x = +B in
the left frame, which is OpenCV's ``stereoRectify`` convention (``T[0] < 0``). A wrong left/right assignment makes
rectified disparities NEGATIVE, which :func:`points_from_disparity` refuses (tested).

Three things are NOT known from the file and are decided on data before any arm output:
whether the stored mp4 frames are upright (rotated by ``rotation_deg``) or native; whether the ego K and distortion
are given in the same rotated frame as the extrinsics; and therefore whether :func:`fastdo.bench.v2d_orientation.
rotate_calib_180` / :func:`~fastdo.bench.v2d_orientation.conj_180` must be applied. The decisive checks are world-up
-> image-up after registration (:func:`~fastdo.bench.v2d_orientation.image_up_cosine`) and a frame-free edge NCC of
triangulated stereo points projected into ``ego_cam_a`` under each hypothesis
(:func:`~fastdo.bench.v2d_orientation.edge_ncc`).

The stereo ruler is ONE scale per device (:func:`device_ruler`), pooled over frames with large static scans; each
frame is then refitted rigidly at that fixed scale. A free per-frame cloud scale is exactly the gauge direction (a
1% error is ~6 mm along the ray at 0.6 m), so it is reported as GT uncertainty, never used as the GT.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from fastdo.bench.v2d_track3 import CameraCalib, StereoPair

logger = logging.getLogger(__name__)

__all__ = [
    "EgoRig",
    "Rectification",
    "RulerEstimate",
    "SceneFit",
    "camera_frame_poses",
    "device_ruler",
    "ego_rig",
    "latency_by_agreement",
    "points_from_disparity",
    "rectify_points",
    "register_scene",
    "stereo_rectification",
]


@dataclass(frozen=True)
class EgoRig:
    """The ego device: the RGB camera ``a`` and the stereo pair (``left``, ``right``) with their transforms."""

    a: CameraCalib
    left: CameraCalib
    right: CameraCalib
    right_T_left: np.ndarray      # (4, 4): X_right = right_T_left @ X_left
    a_T_left: np.ndarray          # (4, 4): X_a = a_T_left @ X_left
    baseline_m: float

    def a_T_c(self) -> np.ndarray:
        """Alias of ``a_T_left`` (the ego stereo-left camera is ``ego_cam_c``)."""
        return self.a_T_left


def _inv(t: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = t[:3, :3].T
    out[:3, 3] = -t[:3, :3].T @ t[:3, 3]
    return out


def ego_rig(cams: dict[str, CameraCalib], pairs: list[StereoPair]) -> EgoRig:
    """Assemble the ego rig under convention H0 (module docstring)."""
    pair = next(p for p in pairs if p.rig == "ego")
    left, right, a = cams[pair.left], cams[pair.right], cams["ego_cam_a"]
    if a.extrinsics_to_stereo_left is None or left.extrinsics_to_stereo_left is None:
        raise ValueError("ego_cam_a and the stereo-left camera need extrinsics_to_stereo_left")
    # Both map into CAM_B's frame as given in the file (180-deg-rotated frames on the ego device; module docstring).
    b_T_a = np.asarray(a.extrinsics_to_stereo_left, np.float64)
    b_T_left = np.asarray(left.extrinsics_to_stereo_left, np.float64)
    if not np.allclose(b_T_left, pair.left_to_right, atol=1e-6):
        raise ValueError("H0 violated: the pair's left_to_right differs from the left camera's extrinsic")
    return EgoRig(a=a, left=left, right=right, right_T_left=np.asarray(pair.left_to_right, np.float64),
                  a_T_left=_inv(b_T_a) @ b_T_left, baseline_m=float(pair.baseline_m))


@dataclass(frozen=True)
class Rectification:
    """``cv2.stereoRectify`` outputs for a (left, right) pair at the calibration resolution."""

    R1: np.ndarray
    R2: np.ndarray
    P1: np.ndarray
    P2: np.ndarray
    Q: np.ndarray
    size: tuple[int, int]

    @property
    def focal_px(self) -> float:
        return float(self.P1[0, 0])

    @property
    def baseline_m(self) -> float:
        """``-P2[0, 3] / P2[0, 0]``: the rectified baseline (positive under OpenCV's convention)."""
        return float(-self.P2[0, 3] / self.P2[0, 0])


def stereo_rectification(left: CameraCalib, right: CameraCalib, right_T_left: np.ndarray,
                         alpha: float = 0.0) -> Rectification:
    """OpenCV rectification of a calibrated pair (``X_right = R X_left + T`` with ``right_T_left = [R | T]``)."""
    import cv2

    if tuple(left.resolution) != tuple(right.resolution):
        raise ValueError("left and right resolutions differ")
    size = (int(left.resolution[0]), int(left.resolution[1]))
    r1, r2, p1, p2, q, _, _ = cv2.stereoRectify(
        left.K, left.distortion.reshape(1, -1), right.K, right.distortion.reshape(1, -1), size,
        np.ascontiguousarray(right_T_left[:3, :3]), np.ascontiguousarray(right_T_left[:3, 3]).reshape(3, 1),
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=alpha)
    return Rectification(R1=r1, R2=r2, P1=p1, P2=p2, Q=q, size=size)


def rectify_points(uv: np.ndarray, cam: CameraCalib, r: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Distorted pixel coordinates (N, 2) -> rectified pixel coordinates (N, 2) (``cv2.undistortPoints``)."""
    import cv2

    pts = np.asarray(uv, np.float64).reshape(-1, 1, 2)
    crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 200, 1e-12)
    out = cv2.undistortPoints(pts, cam.K, cam.distortion.reshape(1, -1), R=r, P=p, criteria=crit)
    return out.reshape(-1, 2)


def points_from_disparity(uv_rect: np.ndarray, disparity: np.ndarray, rect: Rectification) -> np.ndarray:
    """Rectified left pixels (N, 2) and disparities (N,) (``u_left - u_right``, px) -> points in the LEFT camera
    frame (N, 3), metres. Non-positive disparities raise: under a wrong left/right assignment they are negative."""
    d = np.asarray(disparity, np.float64).reshape(-1)
    if np.any(~np.isfinite(d)) or np.any(d <= 0.0):
        raise ValueError("disparities must be finite and > 0 (a negative one means left and right are swapped)")
    uv = np.asarray(uv_rect, np.float64).reshape(-1, 2)
    f, cx, cy = rect.P1[0, 0], rect.P1[0, 2], rect.P1[1, 2]
    z = f * rect.baseline_m / d
    x_rect = np.stack([(uv[:, 0] - cx) * z / f, (uv[:, 1] - cy) * z / rect.P1[1, 1], z], axis=1)
    return x_rect @ rect.R1           # rectified-left -> left camera: X_left = R1^T X_rect


@dataclass(frozen=True)
class SceneFit:
    """Result of :func:`register_scene`: ``cam_T_world``, the cloud's metric correction and the frame checks.

    ``cloud_scale`` multiplies the stereo cloud (about the camera centre) into the scans' metric (Vicon) units: the
    GT inherits the mocap ruler, never the stereo baseline. ``cam_T_world`` maps world points into that corrected
    camera frame (same orientation as the camera; translations in Vicon metres).
    """

    cam_T_world: np.ndarray       # (4, 4)
    cloud_scale: float            # 1.0 when the scale was held fixed
    inlier_rms_m: float           # point-to-plane RMS over inliers
    inlier_frac: float            # fraction of cloud points within `inlier_m` of the model after the fit
    model_coverage: float         # fraction of camera-facing model points with a cloud point within `inlier_m`
    iterations: int
    converged: bool


def _skew_exp(w: np.ndarray) -> np.ndarray:
    th = float(np.linalg.norm(w))
    k = np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])
    if th < 1e-12:
        return np.eye(3) + k
    return np.eye(3) + np.sin(th) / th * k + (1.0 - np.cos(th)) / th ** 2 * (k @ k)


def _skew(v: np.ndarray) -> np.ndarray:
    """(N, 3) -> (N, 3, 3) cross-product matrices."""
    z = np.zeros(len(v))
    return np.stack([np.stack([z, -v[:, 2], v[:, 1]], 1), np.stack([v[:, 2], z, -v[:, 0]], 1),
                     np.stack([-v[:, 1], v[:, 0], z], 1)], 1)


def _gn_stage(mp: np.ndarray, mn: np.ndarray, x: np.ndarray, t: np.ndarray, log_s: float, radius: float,
              estimate_scale: bool, max_iterations: int, tol: float,
              point_weight: float) -> tuple[np.ndarray, float, int, bool]:
    """Robust (Tukey at ``radius``) Gauss-Newton over (omega, v[, log s]) at one search radius.

    Residuals per correspondence: point-to-plane ``n . (q - x)`` plus ``sqrt(point_weight) (q - x)``. Point-to-plane
    alone is blind to sliding along planar faces, and with a free scale three faces of a box meeting at a corner are
    invariant to scaling about that corner, so a small point-to-point term is required (tested: noiseless boxes).
    """
    n_par = 7 if estimate_scale else 6
    sw = float(np.sqrt(point_weight))
    for it in range(1, max_iterations + 1):
        pc = mp @ t[:3, :3].T + t[:3, 3]
        nc = mn @ t[:3, :3].T
        xs = np.exp(log_s) * x
        d, j = cKDTree(pc).query(xs, distance_upper_bound=radius)
        ok = np.isfinite(d)
        if ok.sum() < 2 * n_par:
            return t, log_s, it, False
        q, n, xo, dd = pc[j[ok]], nc[j[ok]], xs[ok], d[ok]
        w = (1.0 - np.minimum((dd / radius) ** 2, 1.0)) ** 2
        e = q - xo
        r_pl = np.einsum("ij,ij->i", n, e)
        a_pl = [np.cross(q, n), n]                              # d r / d (omega, v) for q -> exp(omega) q + v
        a_pt = [-_skew(q), np.broadcast_to(np.eye(3), (len(q), 3, 3))]
        if estimate_scale:
            a_pl.append(-np.einsum("ij,ij->i", n, xo)[:, None])  # d r / d log s for x -> exp(log s) x
            a_pt.append(-xo[:, :, None])
        a1 = np.concatenate(a_pl, axis=1)                       # (K, P)
        a3 = sw * np.concatenate(a_pt, axis=2)                  # (K, 3, P)
        h = (a1 * w[:, None]).T @ a1 + np.einsum("k,kip,kiq->pq", w, a3, a3) + 1e-12 * np.eye(n_par)
        g = (a1 * w[:, None]).T @ r_pl + np.einsum("k,kip,ki->p", w, a3, sw * e)
        delta = -np.linalg.solve(h, g)
        step = np.eye(4)
        step[:3, :3] = _skew_exp(delta[:3])
        step[:3, 3] = delta[3:6]
        t = step @ t
        if estimate_scale:
            log_s += float(delta[6])
        if np.linalg.norm(delta) < tol:
            return t, log_s, it, True
    return t, log_s, max_iterations, False


def register_scene(model_pts: np.ndarray, model_nrm: np.ndarray, cloud: np.ndarray, init_cam_T_world: np.ndarray,
                   estimate_scale: bool = True, inlier_m: float = 0.01, max_corr_m: float = 0.05,
                   max_iterations: int = 60, tol: float = 1e-8, point_weight: float = 0.25,
                   fixed_scale: float = 1.0, scale_bounds: tuple[float, float] = (0.8, 1.25)) -> SceneFit:
    """``cam_T_world`` (and the cloud's metric scale) from posed scan samples and a camera-frame stereo cloud.

    Args:
        model_pts, model_nrm: (M, 3) samples of every scan of the episode posed by Vicon at this frame (world frame,
            metres) and their outward normals; their relative geometry is exact.
        cloud: (N, 3) stereo points in the camera frame (any clutter: hands, table, background).
        init_cam_T_world: (4, 4) initialisation; coarse-to-fine search is the caller's job. A wrong basin shows up
            as low ``model_coverage``.
        estimate_scale: fit one scale of the cloud about the camera centre (default; ``fixed_scale`` is then the
            initial value). A rigid-only fit at the WRONG scale turns the ruler error into a translation error
            (tested). The GT protocol uses the free fit only to estimate one ruler per device
            (:func:`device_ruler`), then refits every frame with ``estimate_scale=False`` at that ruler.
        fixed_scale: the cloud scale held when ``estimate_scale`` is False (the device ruler).
        scale_bounds: a free scale outside this band marks the fit NOT converged. On a single curved, partially
            visible object the free-scale problem has a collapse attractor (cloud and model shrink onto the camera
            centre, scale -> 0) that still reports high coverage (tested), so coverage alone cannot catch it.

    Two stages: ``max_corr_m`` (capture), then ``2 * inlier_m`` (refine), each robust Gauss-Newton on
    point-to-plane plus ``point_weight`` x point-to-point residuals.
    """
    mp = np.asarray(model_pts, np.float64)
    mn = np.asarray(model_nrm, np.float64)
    x = np.asarray(cloud, np.float64)
    t = np.asarray(init_cam_T_world, np.float64).copy()
    if not np.isfinite(fixed_scale) or fixed_scale <= 0.0:
        raise ValueError(f"fixed_scale must be finite and > 0, got {fixed_scale!r}")
    log_s, its, converged = float(np.log(fixed_scale)), 0, False
    for radius in (max_corr_m, 2.0 * inlier_m):
        t, log_s, it, converged = _gn_stage(mp, mn, x, t, log_s, radius, estimate_scale, max_iterations, tol,
                                             point_weight)
        its += it
    pc = mp @ t[:3, :3].T + t[:3, 3]
    nc = mn @ t[:3, :3].T
    xs = np.exp(log_s) * x
    d, j = cKDTree(pc).query(xs)
    inl = d < inlier_m
    r = np.einsum("ij,ij->i", nc[j[inl]], pc[j[inl]] - xs[inl]) if inl.any() else np.array([np.nan])
    facing = np.einsum("ij,ij->i", nc, -pc) > 0.0
    dm = cKDTree(xs).query(pc[facing])[0] if facing.any() else np.array([np.inf])
    scale = float(np.exp(log_s))
    if estimate_scale and not (scale_bounds[0] <= scale <= scale_bounds[1]):
        logger.warning("free cloud scale %.4g left the band %s: fit marked not converged", scale, scale_bounds)
        converged = False
    return SceneFit(cam_T_world=t, cloud_scale=scale, inlier_rms_m=float(np.sqrt(np.mean(r ** 2))),
                    inlier_frac=float(inl.mean()), model_coverage=float(np.mean(dm < inlier_m)), iterations=its,
                    converged=converged)


def camera_frame_poses(cam_T_world: np.ndarray, world_T_object: np.ndarray) -> np.ndarray:
    """(T, 4, 4) camera poses and (T, N, 4, 4) object poses -> (T, N, 4, 4) ``cam_T_object`` (NaN rows stay NaN)."""
    c = np.asarray(cam_T_world, np.float64)
    w = np.asarray(world_T_object, np.float64)
    return np.einsum("tij,tnjk->tnik", c, w)


@dataclass(frozen=True)
class RulerEstimate:
    """One stereo ruler per device: the median free-fit cloud scale over the pooled frames and its scatter."""

    scale: float                  # exp(median log scale)
    log_sd: float                 # sd of the per-frame log scales: the free-scale GT uncertainty (gate: <= 0.01)
    log_mad_sd: float             # 1.4826 x MAD of the log scales (robust to a few bad frames)
    n_frames: int


def device_ruler(free_scales: np.ndarray, ok: np.ndarray | None = None) -> RulerEstimate:
    """Pool per-frame free-scale fits (frames with large static scans) into one ruler per device.

    Args:
        free_scales: (F,) ``SceneFit.cloud_scale`` of free-scale fits.
        ok: (F,) frames passing the per-frame validity rules (coverage, inlier RMS); default all finite.
    """
    s = np.asarray(free_scales, np.float64).reshape(-1)
    m = np.isfinite(s) & (s > 0.0)
    if ok is not None:
        m &= np.asarray(ok, bool).reshape(-1)
    if m.sum() < 3:
        raise ValueError(f"need >= 3 valid frames to pool a ruler, got {int(m.sum())}")
    ls = np.log(s[m])
    med = float(np.median(ls))
    return RulerEstimate(scale=float(np.exp(med)), log_sd=float(np.std(ls, ddof=1)),
                         log_mad_sd=float(1.4826 * np.median(np.abs(ls - med))), n_frames=int(m.sum()))


def latency_by_agreement(lags_s: np.ndarray, disagreement: Callable[[float], float]) -> tuple[float, np.ndarray]:
    """Label-free Vicon-to-video latency: the lag minimising a disagreement between independent per-object fits.

    ``disagreement(lag)`` evaluates, with every Vicon pose sampled at ``t_video + lag``, e.g. the median over
    two-object frames of the distance between the ``cam_T_world`` fitted on object A alone and on object B alone (a
    static object is lag-blind, a moving one is not, so the two agree only at the true lag). Returns the best lag,
    refined by a parabola through the grid minimum and its neighbours, and the curve.
    """
    lags = np.asarray(lags_s, np.float64).reshape(-1)
    curve = np.asarray([float(disagreement(float(lag))) for lag in lags])
    i = int(np.nanargmin(curve))
    if 0 < i < len(lags) - 1:
        y0, y1, y2 = curve[i - 1], curve[i], curve[i + 1]
        den = y0 - 2.0 * y1 + y2
        if den > 0.0 and np.allclose(np.diff(lags), lags[1] - lags[0]):
            return float(lags[i] + 0.5 * (y0 - y2) / den * (lags[1] - lags[0])), curve
    return float(lags[i]), curve
