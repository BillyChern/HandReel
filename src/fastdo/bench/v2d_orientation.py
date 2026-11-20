"""Image-orientation transforms and label-free orientation checks for the V2D ego device (CPU, no frame viewing).

The ego OAK device is mounted upside down (``rotation_deg`` 180), and its extrinsics in ``camera_calibration.json`` are
expressed in 180-deg-rotated camera frames (:mod:`fastdo.bench.v2d_camera_gt` docstring). Whether the stored mp4
frames, the intrinsics and the distortion are in that same frame is not stated, so every hypothesis needs an exact
transform and a decisive check:

- :func:`rotate_calib_180`: one camera's K and 14-coefficient OpenCV distortion for the image rotated by 180 deg
  (pixel ``(u, v) -> (W - 1 - u, H - 1 - v)``, camera frame ``X -> Rz(pi) X``). Radial terms are even and stay;
  tangential ``p1, p2``, thin-prism ``s1..s4`` and tilt ``tau_x, tau_y`` are odd and change sign (known-answer test
  against ``cv2.projectPoints`` on the rotated image);
- :func:`conj_180`: an extrinsic between two rotated frames, ``Rz(pi) T Rz(pi)^T``;
- :func:`image_up_cosine`: after scene registration, world up (Vicon +z; the table is ~0.61 m high) must map to image
  up (camera -y) in upright frames: a head-mounted camera is never upside down for a whole episode;
- :func:`edge_ncc`: frame-free consistency of a hypothesis. Triangulated stereo points carry the stereo-left image's
  edge magnitude; projected into ``ego_cam_a`` under the hypothesis, their values must correlate with ``ego_cam_a``'s
  edge magnitude. A wrong K/distortion/extrinsic hypothesis decorrelates them. Nothing here displays a frame.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from fastdo.bench.v2d_track3 import CameraCalib

__all__ = [
    "RZ_PI",
    "conj_180",
    "edge_ncc",
    "image_up_cosine",
    "rotate_calib_180",
    "rotate_pixels_180",
    "sobel_magnitude",
]

RZ_PI = np.diag([-1.0, -1.0, 1.0, 1.0])
# OpenCV 14-term order: k1 k2 p1 p2 k3 k4 k5 k6 s1 s2 s3 s4 tau_x tau_y; odd terms under (x, y) -> (-x, -y):
_ODD = np.array([2, 3, 8, 9, 10, 11, 12, 13])


def rotate_pixels_180(uv: np.ndarray, resolution: tuple[int, int]) -> np.ndarray:
    """Pixel coordinates (N, 2) of the same points in the image rotated by 180 deg (pixel centres at integers)."""
    w, h = int(resolution[0]), int(resolution[1])
    out = np.asarray(uv, np.float64).copy()
    out[..., 0] = (w - 1) - out[..., 0]
    out[..., 1] = (h - 1) - out[..., 1]
    return out


def rotate_calib_180(cam: CameraCalib) -> CameraCalib:
    """``cam`` for the 180-deg-rotated image: cx' = W-1-cx, cy' = H-1-cy, odd distortion terms negated.

    ``extrinsics_to_stereo_left`` is conjugated by ``Rz(pi)`` on the assumption that its target frame is rotated
    together with this camera (both cameras of one device); use :func:`conj_180` directly for anything else.
    ``rotation_deg`` is advanced by 180 (mod 360) so a double application is the identity.
    """
    w, h = int(cam.resolution[0]), int(cam.resolution[1])
    k = np.asarray(cam.K, np.float64).copy()
    k[0, 2] = (w - 1) - k[0, 2]
    k[1, 2] = (h - 1) - k[1, 2]
    if abs(k[0, 1]) > 0.0:
        raise ValueError("a skewed K is not supported")
    d = np.asarray(cam.distortion, np.float64).reshape(-1).copy()
    if d.size not in (4, 5, 8, 12, 14):
        raise ValueError(f"unsupported distortion length {d.size}")
    d[_ODD[_ODD < d.size]] *= -1.0
    ext = None if cam.extrinsics_to_stereo_left is None else conj_180(cam.extrinsics_to_stereo_left)
    return dataclasses.replace(cam, K=k, distortion=d, extrinsics_to_stereo_left=ext,
                               rotation_deg=int((cam.rotation_deg + 180) % 360))


def conj_180(t_dst_src: np.ndarray) -> np.ndarray:
    """``X_dst = T X_src`` re-expressed between the two 180-deg-rotated frames: ``Rz(pi) T Rz(pi)^T``."""
    return RZ_PI @ np.asarray(t_dst_src, np.float64) @ RZ_PI.T


_WORLD_UP = (0.0, 0.0, 1.0)


def image_up_cosine(cam_T_world: np.ndarray, up_world: tuple[float, float, float] = _WORLD_UP) -> np.ndarray:
    """Cosine between world up mapped into the camera and image up (camera -y), per frame ((4, 4) or (T, 4, 4)).

    Positive (typically > 0.5 for a head camera) when the frames are upright under the tested hypothesis, negative
    when they are upside down.
    """
    c = np.asarray(cam_T_world, np.float64)
    r = c[..., :3, :3]
    u = np.asarray(up_world, np.float64)
    up_cam = r @ (u / np.linalg.norm(u))
    return -up_cam[..., 1] / np.linalg.norm(up_cam, axis=-1)


def sobel_magnitude(img: np.ndarray) -> np.ndarray:
    """Gradient magnitude of a grayscale float image (3x3 Sobel), for :func:`edge_ncc`."""
    import cv2

    g = np.asarray(img, np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def _bilinear(img: np.ndarray, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = img.shape[:2]
    u, v = uv[:, 0], uv[:, 1]
    ok = np.isfinite(u) & np.isfinite(v) & (u >= 0) & (v >= 0) & (u <= w - 1) & (v <= h - 1)
    out = np.full(len(uv), np.nan)
    if not ok.any():
        return out, ok
    uu, vv = u[ok], v[ok]
    u0 = np.minimum(np.floor(uu).astype(np.int64), w - 2)
    v0 = np.minimum(np.floor(vv).astype(np.int64), h - 2)
    a, b = uu - u0, vv - v0
    out[ok] = ((1 - a) * (1 - b) * img[v0, u0] + a * (1 - b) * img[v0, u0 + 1]
               + (1 - a) * b * img[v0 + 1, u0] + a * b * img[v0 + 1, u0 + 1])
    return out, ok


def edge_ncc(src_values: np.ndarray, dst_edges: np.ndarray, uv_dst: np.ndarray, min_points: int = 200) -> float:
    """Normalised cross-correlation between per-point source edge values and the destination edge image sampled at
    the points' projections ``uv_dst`` (N, 2) under one hypothesis. NaN when fewer than ``min_points`` land inside."""
    sv = np.asarray(src_values, np.float64).reshape(-1)
    dv, ok = _bilinear(np.asarray(dst_edges, np.float64), np.asarray(uv_dst, np.float64).reshape(-1, 2))
    ok &= np.isfinite(sv)
    if ok.sum() < min_points:
        return float("nan")
    a, b = sv[ok] - sv[ok].mean(), dv[ok] - dv[ok].mean()
    den = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / den) if den > 0.0 else float("nan")
