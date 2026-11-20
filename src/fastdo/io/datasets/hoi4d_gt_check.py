"""Reprojection check of HOI4D ground truth: the CAD model, posed with the GT object pose and projected with the
intrinsics, against the annotated 2D object mask.

HOI4D has no depth video here, so the only independent test of a GT pose is 2D: does its silhouette land on the
object's pixels? Three numbers per frame, all computed against the ANNOTATED mask (an independent construction: the
masks were labelled on the RGB frames, the poses on reconstructed point clouds):

- `iou_raw`: IoU(silhouette, object mask).
- `iou_occ`: occlusion-aware IoU, IoU(silhouette minus every OTHER labelled pixel, object mask). A hand holding the
  object hides part of it; the raw IoU punishes the pose for that.
- `shift_px` / `lateral_mm` / `iou_best`: the 2D translation of the silhouette that maximises `iou_occ` (searched
  exactly by FFT correlation over +-`max_shift_px`), its length converted to millimetres at the object's GT depth
  (`|shift| z / f`), and the IoU reached there. `lateral_mm` estimates the image-plane component of the GT
  translation error in the scorer's own unit; `iou_best` tests the rotation and shape once that is removed. The
  along-ray (depth) component is NOT observable in 2D and is not checked by anything here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from scipy.signal import fftconvolve

from fastdo.bench.headtohead import render_silhouette

__all__ = ["FrameCheck", "check_frame", "silhouette"]

_PAD = 8


@dataclass(frozen=True)
class FrameCheck:
    """The reprojection numbers of one frame (module docstring). NaN IoUs when both masks are empty."""

    obj_px: int
    sil_px: int
    iou_raw: float
    iou_occ: float
    shift_px: tuple[int, int]
    lateral_mm: float
    iou_best: float
    depth_m: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def silhouette(vertices: np.ndarray, faces: np.ndarray, pose: np.ndarray, K: np.ndarray,
               image_hw: tuple[int, int]) -> np.ndarray:
    """(H,W) bool silhouette of model-frame `vertices` under the object-to-camera `pose`."""
    pose = np.asarray(pose, np.float64)
    cam = np.asarray(vertices, np.float64) @ pose[:3, :3].T + pose[:3, 3]
    return render_silhouette(cam, np.asarray(faces, np.int64), torch.as_tensor(np.asarray(K, np.float64)), image_hw)


def _iou(inter: float, a: float, b: float) -> float:
    union = a + b - inter
    return float(inter / union) if union > 0 else float("nan")


def check_frame(sil: np.ndarray, obj: np.ndarray, other: np.ndarray, depth_m: float, focal_px: float,
                max_shift_px: int = 160) -> FrameCheck:
    """`FrameCheck` of one frame. `sil`, `obj`, `other` are (H,W) bool; `other` must not overlap `obj`.

    The shift search is exact over the integer grid: with O = obj and C = other, IoU(shift s) =
    |S_s & O| / (|S_s & ~C| + |O| - |S_s & O|), and both |S_s & O| and |S_s & C| are correlations of the
    silhouette with a fixed mask, so one FFT each gives every shift at once. Shifted-out pixels are dropped.
    """
    sil, obj, other = (np.asarray(m, bool) for m in (sil, obj, other))
    if (obj & other).any():
        raise ValueError("`other` overlaps the object mask")
    n_obj, n_sil = int(obj.sum()), int(sil.sum())
    iou_raw = _iou(float((sil & obj).sum()), n_sil, n_obj)
    vis = sil & ~other
    iou_occ = _iou(float((vis & obj).sum()), float(vis.sum()), n_obj)
    if n_obj == 0 or n_sil == 0:
        return FrameCheck(n_obj, n_sil, iou_raw, iou_occ, (0, 0), float("nan"), float("nan"), float(depth_m))
    h, w = sil.shape
    ys, xs = np.nonzero(sil | obj)
    r = int(max_shift_px) + _PAD
    y0, y1 = max(int(ys.min()) - r, 0), min(int(ys.max()) + r + 1, h)
    x0, x1 = max(int(xs.min()) - r, 0), min(int(xs.max()) + r + 1, w)
    s = sil[y0:y1, x0:x1].astype(np.float64)
    o = obj[y0:y1, x0:x1].astype(np.float64)
    c = other[y0:y1, x0:x1].astype(np.float64)
    # corr[dy, dx] = sum_p S(p) M(p + d); fftconvolve(M, S[::-1, ::-1], "full") has the zero shift at (hs-1, ws-1).
    hs, ws = s.shape
    flip = s[::-1, ::-1]
    inter_o = fftconvolve(o, flip, mode="full")
    inter_c = fftconvolve(c, flip, mode="full")
    kept = fftconvolve(np.ones_like(o), flip, mode="full")        # silhouette pixels still inside the crop
    k = min(int(max_shift_px), hs - 1, ws - 1)
    cy, cx = hs - 1, ws - 1
    sl = (slice(cy - k, cy + k + 1), slice(cx - k, cx + k + 1))
    io, ic, kp = (np.rint(a[sl]) for a in (inter_o, inter_c, kept))
    union = (kp - ic) + n_obj - io
    iou = np.where(union > 0, io / np.maximum(union, 1.0), 0.0)
    dy, dx = np.unravel_index(int(np.argmax(iou)), iou.shape)
    shift = (int(dy) - k, int(dx) - k)
    lateral = float(np.hypot(*shift)) * float(depth_m) / float(focal_px) * 1000.0
    return FrameCheck(n_obj, n_sil, iou_raw, iou_occ, shift, lateral, float(iou[dy, dx]), float(depth_m))
