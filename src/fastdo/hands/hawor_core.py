"""Torch side of batched HaWoR: crops, a backbone feature FIFO, temporal heads and row gathering.

Shared module: the hawor env's driver loads this file by path (Python 3.10), and the uv env imports it. It imports torch,
numpy and the standard library only, nothing from `fastdo`, and types its inputs with Protocols, so `hawor_plan`'s
dataclasses or any look-alike can be passed in.

`run_heads` calls HaWoR's own submodules (`st_module`, `mano_head`, `motion_module`, `bbox_est`, `get_trans`) in
`HAWOR.forward_step`'s order, with a per-slot flip, focal length and image centre instead of batch-global ones. Windows
of different clips, hands and image sizes can therefore share one batch, and no slot's output depends on another
window.

Batch invariance (opt-in). cuBLAS/cuDNN choose kernels by shape, so a crop's features and a window's head outputs
can move at float level with the SIZE of the call they run in, which the other clips of the group set. Within one fixed
shape, each row's result does not depend on the other rows. `FeatureBank(pad_to_batch=True)` and
`run_heads(pad_last=True)` therefore run every call at one fixed size: the backbone remainder is padded with zero crops
to `batch`, and the last heads batch is padded to `windows_per_batch` windows with copies of its own first window (a
real window, so valid centre, scale and focal: no NaN path). Padded outputs are dropped. Both default off, which runs
exactly the unpadded calls of before.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "CROP_SIZE",
    "FEATURE_SHAPE",
    "IMG_MEAN",
    "IMG_STD",
    "WINDOW",
    "CropLike",
    "FeatureBank",
    "GeometryLike",
    "HaworLike",
    "HaworOps",
    "HeadOutputs",
    "SlotLike",
    "crop_hands",
    "crop_ulbr",
    "gather_rows",
    "run_backbone",
    "run_heads",
]

CROP_SIZE = 256
WINDOW = 16
IMG_MEAN = (0.485, 0.456, 0.406)
IMG_STD = (0.229, 0.224, 0.225)
FEATURE_SHAPE = (1280, 16, 12)
_WIDTH_TRIM = 32                                                     # 256x256 crop -> 256x192 backbone input


class CropLike(Protocol):
    @property
    def row(self) -> int: ...
    @property
    def center(self) -> tuple[float, float]: ...
    @property
    def scale(self) -> float: ...
    @property
    def flip(self) -> bool: ...


class SlotLike(Protocol):
    @property
    def clip(self) -> int: ...
    @property
    def side(self) -> str: ...
    @property
    def bank0(self) -> int: ...
    @property
    def bank1(self) -> int: ...
    @property
    def alpha(self) -> float: ...
    @property
    def center(self) -> tuple[float, float]: ...
    @property
    def scale(self) -> float: ...
    @property
    def flip(self) -> bool: ...
    @property
    def out_row(self) -> int: ...


class GeometryLike(Protocol):
    @property
    def width(self) -> int: ...
    @property
    def height(self) -> int: ...
    @property
    def focal_px(self) -> float: ...


class HaworLike(Protocol):
    @property
    def backbone(self) -> Callable[..., Any]: ...
    @property
    def st_module(self) -> Callable[..., Any]: ...
    @property
    def mano_head(self) -> Callable[..., Any]: ...
    @property
    def motion_module(self) -> Callable[..., Any]: ...
    def bbox_est(self, center: Tensor, scale: Tensor, img_focal: Tensor, img_center: Tensor) -> Tensor: ...
    def get_trans(self, pred_cam: Tensor, center: Tensor, scale: Tensor, img_focal: Tensor,
                  img_center: Tensor) -> Tensor: ...


@dataclass(frozen=True)
class HaworOps:
    """HaWoR's rotation helpers: 6D -> `(N,3,3)` rotation matrices and `(N,3,3)` -> `(N,3)` axis-angle."""

    rot6d_to_rotmat: Callable[[Tensor], Tensor]
    rotmat_to_aa: Callable[[Tensor], Tensor]


@dataclass(frozen=True)
class HeadOutputs:
    """Per slot: `rotmat (S,16,3,3)` as the network predicts it (mirrored for flipped slots), `aa (S,16,3)` un-mirrored
    axis-angle, `betas (S,10)` and full-frame camera translation `trans (S,3)`; all float32."""

    rotmat: Tensor
    aa: Tensor
    betas: Tensor
    trans: Tensor


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def crop_ulbr(center: tuple[float, float], scale: float, res: int = CROP_SIZE) -> tuple[tuple[int, int], tuple[int, int]]:
    """Upper-left and bottom-right pixel corners of HaWoR's square crop of side `200 * scale` around `center`.

    `T` maps image pixels to crop pixels, `T = [[res/h, 0, res(-cx/h + 1/2)], [0, res/h, res(-cy/h + 1/2)], [0, 0, 1]]`
    with `h = 200 * scale + 1e-6`. A crop corner `p` (1-based) maps back to `trunc(T^-1 [p0 - 1, p1 - 1, 1]) + 1`, and
    the corners are those of `p = (1, 1)` and `(res + 1, res + 1)`, minus one.
    """
    h = 200.0 * float(scale) + 1e-6
    t = np.zeros((3, 3))
    t[0, 0] = t[1, 1] = float(res) / h
    t[0, 2] = res * (-float(center[0]) / h + 0.5)
    t[1, 2] = res * (-float(center[1]) / h + 0.5)
    t[2, 2] = 1.0
    t_inv = np.linalg.inv(t)

    def to_image(p0: float, p1: float) -> np.ndarray:
        return np.dot(t_inv, np.array([p0 - 1.0, p1 - 1.0, 1.0]))[:2].astype(int) + 1

    ul, br = to_image(1, 1) - 1, to_image(res + 1, res + 1) - 1
    return (int(ul[0]), int(ul[1])), (int(br[0]), int(br[1]))


def _crop_one(frame: Tensor, item: CropLike, height: int, width: int) -> Tensor:
    if not (np.isfinite(item.center).all() and np.isfinite(item.scale) and item.scale > 0):
        raise ValueError(f"crop item for row {item.row} has a non-finite centre or non-positive scale")
    (ul_x, ul_y), (br_x, br_y) = crop_ulbr(item.center, item.scale)
    patch = torch.zeros((br_y - ul_y, br_x - ul_x, 3), dtype=torch.uint8, device=frame.device)
    x0, x1, y0, y1 = max(ul_x, 0), min(br_x, width), max(ul_y, 0), min(br_y, height)
    if x1 > x0 and y1 > y0:
        # the mirrored frame's columns [x0, x1) are the frame's columns [W - x1, W - x0), reversed
        region = frame[y0:y1, width - x1:width - x0].flip(1) if item.flip else frame[y0:y1, x0:x1]
        patch[y0 - ul_y:y1 - ul_y, x0 - ul_x:x1 - ul_x] = region
    resized = F.interpolate(patch.permute(2, 0, 1)[None].float(), size=(CROP_SIZE, CROP_SIZE), mode="bilinear",
                            antialias=True, align_corners=False)
    return resized[0].clamp_(0.0, 255.0).floor_()


def crop_hands(frames: Tensor, items: Sequence[CropLike], device: torch.device | str | None = None,
               frames_per_chunk: int = 256) -> Tensor:
    """Normalized `(N,3,256,256)` float32 crops, zero-padded outside the frame; flipped items crop the mirrored frame.

    Args:
        frames: `(T,H,W,3)` uint8 RGB, on any device.
        items: One per crop; `row` indexes `frames`.
        device: Where crops are computed and returned; `frames`' own device when None.
        frames_per_chunk: Frames are moved to `device` this many at a time, only for chunks some item reads.

    Raises:
        ValueError: If `frames` is not `(T,H,W,3)` uint8, or an item has a non-finite centre or non-positive scale.
    """
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != torch.uint8:
        raise ValueError(f"frames must be (T,H,W,3) uint8, got {tuple(frames.shape)} {frames.dtype}")
    if frames_per_chunk < 1:
        raise ValueError(f"frames_per_chunk must be >= 1, got {frames_per_chunk}")
    target = frames.device if device is None else torch.device(device)
    out = torch.empty((len(items), 3, CROP_SIZE, CROP_SIZE), dtype=torch.float32, device=target)
    if not items:
        return out
    height, width = int(frames.shape[1]), int(frames.shape[2])
    chunk_start, chunk = -1, frames[:0]
    for i in sorted(range(len(items)), key=lambda k: items[k].row):
        start = items[i].row // frames_per_chunk * frames_per_chunk
        if start != chunk_start:
            chunk, chunk_start = frames[start:start + frames_per_chunk].to(target), start
        out[i] = _crop_one(chunk[items[i].row - start], items[i], height, width)
    mean = torch.tensor(IMG_MEAN, dtype=torch.float32, device=target).view(1, 3, 1, 1)
    std = torch.tensor(IMG_STD, dtype=torch.float32, device=target).view(1, 3, 1, 1)
    return out.div_(255.0).sub_(mean).div_(std)


def run_backbone(model: HaworLike, crops: Tensor, batch: int = 64, bf16: bool = True) -> Tensor:
    """`(N,1280,16,12)` backbone features of `(N,3,256,256)` crops trimmed to 256x192: bf16 under bfloat16 autocast when
    `bf16`, else float32."""
    if crops.ndim != 4 or tuple(crops.shape[1:]) != (3, CROP_SIZE, CROP_SIZE):
        raise ValueError(f"crops must be (N,3,{CROP_SIZE},{CROP_SIZE}), got {tuple(crops.shape)}")
    dtype = torch.bfloat16 if bf16 else torch.float32
    if crops.shape[0] == 0:
        return torch.zeros((0, *FEATURE_SHAPE), dtype=dtype, device=crops.device)
    features = []
    with torch.no_grad():
        for start in range(0, int(crops.shape[0]), max(1, batch)):
            x = crops[start:start + batch, :, :, _WIDTH_TRIM:-_WIDTH_TRIM]
            if bf16:
                with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16):
                    feats = model.backbone(x)
            else:
                feats = model.backbone(x)
            features.append(feats.to(dtype))
    return torch.cat(features)


class FeatureBank:
    """Backbone features of one clip group in item order, filled through a FIFO of crops.

    `push` runs one backbone batch of `batch` crops each time the FIFO holds that many; `flush` runs the rest. Both
    return the backbone seconds they spent (CUDA-synchronized). With `pad_to_batch`, `flush` pads the rest with
    zero crops to exactly `batch`, runs that one full batch and keeps the first rows, so every backbone call has the
    same shape and a crop's features do not depend on how many crops the group's other clips left in the FIFO.
    """

    def __init__(self, model: HaworLike, batch: int = 64, bf16: bool = True, pad_to_batch: bool = False) -> None:
        if batch < 1:
            raise ValueError(f"batch must be >= 1, got {batch}")
        self._model, self._batch, self._bf16 = model, int(batch), bool(bf16)
        self._pad_to_batch = bool(pad_to_batch)
        self._pending: list[Tensor] = []
        self._n_pending = 0
        self._n_items = 0
        self._features: list[Tensor] = []

    @property
    def n_items(self) -> int:
        """Crops pushed so far, including those still waiting in the FIFO."""
        return self._n_items

    def push(self, crops: Tensor) -> float:
        n = int(crops.shape[0])
        self._n_items += n
        if n:
            self._pending.append(crops)
            self._n_pending += n
        seconds = 0.0
        while self._n_pending >= self._batch:
            seconds += self._run(self._batch)
        return seconds

    def flush(self) -> float:
        if not self._n_pending:
            return 0.0
        if self._pad_to_batch:                                   # one full-shape batch, padded rows dropped
            return self._run(self._n_pending, pad_to=self._batch)
        return self._run(self._n_pending)

    def features(self) -> Tensor:
        """`(n_items,1280,16,12)` features.

        Raises:
            RuntimeError: If crops are still pending (call `flush` first).
        """
        if self._n_pending:
            raise RuntimeError(f"{self._n_pending} crops are still pending; call flush() first")
        if not self._features:
            return torch.zeros((0, *FEATURE_SHAPE), dtype=torch.bfloat16 if self._bf16 else torch.float32)
        if len(self._features) > 1:
            self._features = [torch.cat(self._features)]
        return self._features[0]

    def _take(self, n: int) -> Tensor:
        parts, need = [], n
        while need:
            first = self._pending[0]
            if int(first.shape[0]) <= need:
                parts.append(self._pending.pop(0))
                need -= int(first.shape[0])
            else:
                parts.append(first[:need])
                self._pending[0] = first[need:]
                need = 0
        self._n_pending -= n
        return parts[0] if len(parts) == 1 else torch.cat(parts)

    def _run(self, n: int, pad_to: int | None = None) -> float:
        crops = self._take(n)
        if pad_to is not None and pad_to > n:
            zeros = torch.zeros((pad_to - n, *crops.shape[1:]), dtype=crops.dtype, device=crops.device)
            crops = torch.cat([crops, zeros])
        _sync(crops.device)
        start = time.perf_counter()
        feats = run_backbone(self._model, crops, batch=int(crops.shape[0]), bf16=self._bf16)
        self._features.append(feats if int(feats.shape[0]) == n else feats[:n])
        _sync(crops.device)
        return time.perf_counter() - start


def run_heads(model: HaworLike, ops: HaworOps, bank: Tensor, slots: Sequence[SlotLike],
              geoms: Sequence[GeometryLike], windows_per_batch: int = 16, pad_last: bool = False) -> HeadOutputs:
    """HaWoR's temporal heads over whole windows of slots, `windows_per_batch` windows per forward pass.

    With `pad_last`, a last batch of `b < windows_per_batch` windows is padded to `windows_per_batch` windows by
    repeating that batch's first window (its features, centre, scale, flip, focal and image centre), and the padded
    windows' outputs are dropped; every pass then has the same shape. Off, the last pass runs `b` windows as before.

    Geometry is `geoms[slot.clip]`; the image centre is `(W/2, H/2)`. Flipped slots carry mirrored centres: the bbox
    feature uses them as they are, and the translation un-mirrors the centre (`W - c - 1`) and negates the camera's x
    offset `tx` (`cam[:, 1]`). The axis-angle output negates components 1 and 2 of flipped slots, while `rotmat` stays
    as predicted. `trans` is in HaWoR's camera, focal `focal_px` on both axes and principal point `(W/2, H/2)`, for
    both sides; `fastdo.engine.workers_hands.write_hands_artifact` moves the hand outputs into the clip's projection
    camera.

    Raises:
        ValueError: If `len(slots)` is not a multiple of 16, or `windows_per_batch < 1`.
    """
    n = len(slots)
    if n % WINDOW or windows_per_batch < 1:
        raise ValueError(f"len(slots) must be a multiple of {WINDOW} and windows_per_batch >= 1, got {n}, "
                         f"{windows_per_batch}")
    device = bank.device
    if n == 0:
        z = torch.zeros(0, device=device)
        return HeadOutputs(z.view(0, 16, 3, 3), z.view(0, 16, 3), z.view(0, 10), z.view(0, 3))

    def column(values: list[Any], dtype: torch.dtype) -> Tensor:
        return torch.tensor(values, dtype=dtype, device=device)

    bank0 = column([s.bank0 for s in slots], torch.long)
    bank1 = column([s.bank1 for s in slots], torch.long)
    alpha = column([s.alpha for s in slots], torch.float32)
    center = column([list(s.center) for s in slots], torch.float32)
    scale = column([s.scale for s in slots], torch.float32)
    flip = column([s.flip for s in slots], torch.bool)
    focal = column([geoms[s.clip].focal_px for s in slots], torch.float32)
    img_center = column([[geoms[s.clip].width / 2.0, geoms[s.clip].height / 2.0] for s in slots], torch.float32)
    outs: list[tuple[Tensor, Tensor, Tensor, Tensor]] = []
    step = windows_per_batch * WINDOW
    with torch.no_grad():
        for start in range(0, n, step):
            sl: slice | Tensor = slice(start, min(start + step, n))
            b = keep = (min(start + step, n) - start) // WINDOW
            if pad_last and b < windows_per_batch:               # repeat the first window up to the full shape
                rows = torch.arange(start, start + b * WINDOW, device=device)
                sl = torch.cat([rows, rows[:WINDOW].repeat(windows_per_batch - b)])
                b = windows_per_batch
            a = alpha[sl].view(-1, 1, 1, 1)
            feat = bank[bank0[sl]].float() * (1.0 - a) + bank[bank1[sl]].float() * a
            info = model.bbox_est(center[sl], scale[sl], focal[sl], img_center[sl])
            h, w = int(feat.shape[2]), int(feat.shape[3])
            x = torch.cat([feat, info[:, :, None, None].expand(-1, -1, h, w)], dim=1)
            c = int(x.shape[1])
            x = x.reshape(b, WINDOW, c, h, w).permute(0, 3, 4, 1, 2).reshape(b * h * w, WINDOW, c)
            x = model.st_module(x)
            c = int(x.shape[-1])
            x = x.reshape(b, h, w, WINDOW, c).permute(0, 3, 4, 1, 2).reshape(b * WINDOW, c, h, w)
            pose6d, betas, cam = model.mano_head(x)
            pose6d = model.motion_module(torch.cat([pose6d.reshape(b, WINDOW, -1), info.reshape(b, WINDOW, -1)], -1))
            rotmat = ops.rot6d_to_rotmat(pose6d.reshape(b * WINDOW, -1)).reshape(b * WINDOW, -1, 3, 3).float()
            flipped = flip[sl]
            cam = cam.float().clone()
            cam[flipped, 1] *= -1
            unmirrored = center[sl].clone()
            unmirrored[flipped, 0] = 2.0 * img_center[sl][flipped, 0] - center[sl][flipped, 0] - 1.0
            trans = model.get_trans(cam, unmirrored, scale[sl], focal[sl], img_center[sl])[:, 0].float()
            aa = ops.rotmat_to_aa(rotmat.reshape(-1, 3, 3)).reshape(b * WINDOW, -1, 3).float()
            aa[flipped, :, 1:] *= -1
            if keep < b:                                             # drop the padded windows' outputs
                rotmat, aa, betas, trans = (t[:keep * WINDOW] for t in (rotmat, aa, betas, trans))
            outs.append((rotmat, aa, betas.float(), trans))
    return HeadOutputs(*(torch.cat([o[i] for o in outs]) for i in range(4)))


def gather_rows(out: HeadOutputs, slots: Sequence[SlotLike], clip: int, side: str, n_rows: int) -> dict[str, np.ndarray]:
    """One clip side's read rows: `rot (T,3)`, `hand_pose (T,45)`, `trans (T,3)`, `betas (T,10)` float32 with NaN where
    unread, `valid (T,)` bool and `slot_index (T,)` int64 (-1 where unread).

    Raises:
        ValueError: If a row is read twice or lies outside `[0, n_rows)`.
    """
    picks = [(q, s.out_row) for q, s in enumerate(slots) if s.clip == clip and s.side == side and s.out_row >= 0]
    rows = {"rot": np.full((n_rows, 3), np.nan, np.float32), "hand_pose": np.full((n_rows, 45), np.nan, np.float32),
            "trans": np.full((n_rows, 3), np.nan, np.float32), "betas": np.full((n_rows, 10), np.nan, np.float32),
            "valid": np.zeros(n_rows, bool), "slot_index": np.full(n_rows, -1, np.int64)}
    if not picks:
        return rows
    q = np.array([p[0] for p in picks], dtype=np.int64)
    r = np.array([p[1] for p in picks], dtype=np.int64)
    if int(r.max()) >= n_rows or np.unique(r).size != r.size:
        raise ValueError(f"clip {clip} {side}: rows {r.tolist()} are out of range for {n_rows} rows or read twice")
    index = torch.as_tensor(q, device=out.aa.device)
    aa = out.aa[index].float().cpu().numpy()
    rows["rot"][r] = aa[:, 0]
    rows["hand_pose"][r] = aa[:, 1:].reshape(-1, 45)
    rows["trans"][r] = out.trans[index].float().cpu().numpy()
    rows["betas"][r] = out.betas[index].float().cpu().numpy()
    rows["valid"][r] = True
    rows["slot_index"][r] = q
    return rows
