"""Trajectory comparison helpers for real-clip head-to-heads with no ground truth.

Used by `run/pipeline/whisking_headtohead.py` to compare our trajectory with the DO AS I DO
baseline's layout: SE(3) interpolation from the 10 fps solve back to the full frame rate, per-frame
translation / geodesic rotation differences, and silhouette reprojection IoU against a common mask.
"""

from __future__ import annotations

import bisect
from collections.abc import Sequence

import cv2
import numpy as np
import torch
from torch import Tensor

from fastdo.geometry.lie import se3_exp, se3_inv, se3_log, so3_log

__all__ = [
    "axis_tilt_errors",
    "interpolate_poses",
    "mask_iou",
    "principal_axis",
    "render_silhouette",
    "rotation_errors",
    "translation_errors",
]


def interpolate_poses(poses: Tensor, frame_idx: Sequence[int], n_frames: int) -> Tensor:
    """Lie-algebra interpolation of a sparse pose trajectory onto every frame.

    Between solved frames `a < f < b`: `T_a Exp(alpha Log(T_a^-1 T_b))`, `alpha = (f-a)/(b-a)`.
    Frames before the first / after the last solved frame hold that end pose.

    Args:
        poses: `(K,4,4)` poses at strictly increasing `frame_idx`.
        frame_idx: `K` frame indices of `poses`.
        n_frames: Output length.

    Returns:
        `(n_frames,4,4)` float32.
    """
    idx = [int(i) for i in frame_idx]
    P = poses.double()
    out = torch.empty(n_frames, 4, 4, dtype=torch.float64)
    for f in range(n_frames):
        j = bisect.bisect_right(idx, f) - 1
        if j < 0:
            out[f] = P[0]
        elif j >= len(idx) - 1 or idx[j] == f:
            out[f] = P[min(j, len(idx) - 1)]
        else:
            alpha = (f - idx[j]) / (idx[j + 1] - idx[j])
            out[f] = P[j] @ se3_exp(alpha * se3_log(se3_inv(P[j]) @ P[j + 1]))
    return out.float()


def translation_errors(poses_a: Tensor, poses_b: Tensor) -> Tensor:
    """`(T,)` Euclidean distance between the two trajectories' translations (metres)."""
    return (poses_a[:, :3, 3].double() - poses_b[:, :3, 3].double()).norm(dim=-1).float()


def rotation_errors(poses_a: Tensor, poses_b: Tensor) -> Tensor:
    """`(T,)` geodesic angle (radians) between the two trajectories' rotations."""
    rel = poses_a[:, :3, :3].double().transpose(1, 2) @ poses_b[:, :3, :3].double()
    return so3_log(rel).norm(dim=-1).float()


def principal_axis(points: np.ndarray) -> np.ndarray:
    """Unit direction of largest variance of `(N,3)` points (a near-symmetric object's long axis)."""
    centred = points - points.mean(axis=0)
    _u, _s, vh = np.linalg.svd(centred, full_matrices=False)
    return vh[0] / np.linalg.norm(vh[0])


def axis_tilt_errors(poses_a: Tensor, poses_b: Tensor, axis: np.ndarray) -> Tensor:
    """`(T,)` angle (radians) between the two trajectories' images of a canonical `axis`.

    For an object near-symmetric about `axis` (a whisk handle), spin about the axis is unobservable
    from its shape; this is the part of the rotation difference that is not such a spin.
    """
    a = torch.as_tensor(axis, dtype=torch.float64)
    da = poses_a[:, :3, :3].double() @ a
    db = poses_b[:, :3, :3].double() @ a
    return torch.arccos((da * db).sum(-1).clamp(-1.0, 1.0)).float()


def render_silhouette(vertices_cam: np.ndarray, faces: np.ndarray, K: Tensor, image_hw: tuple[int, int]
                      ) -> np.ndarray:
    """Binary silhouette of a camera-frame triangle mesh (faces with a vertex behind the camera skipped).

    Args:
        vertices_cam: `(V,3)` camera-frame vertices.
        faces: `(F,3)` vertex indices.
        K: `(3,3)` pixel intrinsics.
        image_hw: `(H,W)`.

    Returns:
        `(H,W)` bool.
    """
    h, w = image_hw
    Kn = K.double().cpu().numpy()
    z = vertices_cam[:, 2]
    z_safe = np.where(z > 1e-6, z, 1.0)
    uv = np.stack([Kn[0, 0] * vertices_cam[:, 0] / z_safe + Kn[0, 2],
                   Kn[1, 1] * vertices_cam[:, 1] / z_safe + Kn[1, 2]], axis=-1)
    tri = uv[faces]                                                          # (F,3,2)
    keep = (z[faces] > 1e-6).all(axis=1)
    keep &= (tri[..., 0].max(1) >= 0) & (tri[..., 0].min(1) < w) & (tri[..., 1].max(1) >= 0) & (tri[..., 1].min(1) < h)
    tri = np.clip(tri[keep], -4 * max(h, w), 4 * max(h, w))
    fixed = np.round(tri * 16.0).astype(np.int32)                            # 4-bit sub-pixel precision
    img = np.zeros((h, w), dtype=np.uint8)
    for t in fixed:
        cv2.fillConvexPoly(img, t, 1, lineType=cv2.LINE_8, shift=4)
    return img.astype(bool)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over union of two boolean masks (NaN when both are empty)."""
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return float("nan")
    return int(np.logical_and(a, b).sum()) / union
