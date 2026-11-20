"""SAM 2.1 model variants, reduced-resolution mask propagation, and the default-variant rule.

SAM 2.1's video predictor comes in four open sizes on Hugging Face (`SAM2_VARIANTS`). A variant string is
`<size>` for native resolution or `<size>@<px>` for frames downscaled to a `<px>` short side before
propagation; masks always come back at native resolution. `run/pipeline/bench_sam2_variants.py` measures every
size at native 720p and at a 512 px short side on the whisking demo, and `choose_sam2_variant` picks the default
from that table: the fastest row whose mean mask IoU against large at native resolution is at least 0.90, else
`large`. Sweep rows carry variant strings as labels (`variant_label`): a row run at the demo's native 720p is the
bare size (`small`), so a chosen native row runs every clip at its own native resolution, as it was measured,
instead of downscaling 1080p clips to 720 px. Until that sweep is recorded (in its exclusive GPU
window), `mask_worker` defaults to `"large"` (`DEFAULT_SAM2_VARIANT`).

Resolution caveat, read before interpreting a sweep: SAM 2's video predictor resizes every frame to its
`image_size` (1024x1024 for all four 2.1 checkpoints; `sam2/utils/misc.py` `_load_img_as_tensor`) before the
image encoder. A smaller short side therefore saves JPEG staging and frame loading and lowers the resolution
of the predicted masks, but it does not reduce the backbone's compute.

Propagation runs under bf16 autocast on CUDA (`bf16=True`), the precision the sweep measures.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from fastdo.shape.masks_sam2 import propagate_masks_sam2

__all__ = [
    "DEFAULT_SAM2_VARIANT",
    "REFERENCE_LABEL",
    "SAM2_VARIANTS",
    "Sam2Variant",
    "choose_sam2_variant",
    "parse_sam2_variant",
    "propagate_masks_at",
    "resize_short_side",
    "variant_label",
]

SAM2_VARIANTS: Mapping[str, str] = MappingProxyType({
    "large": "facebook/sam2.1-hiera-large",
    "base_plus": "facebook/sam2.1-hiera-base-plus",
    "small": "facebook/sam2.1-hiera-small",
    "tiny": "facebook/sam2.1-hiera-tiny",
})
DEFAULT_SAM2_VARIANT = "large"
# The sweep's reference and the rule's fallback: large at native resolution (the whisking demo's 720p).
REFERENCE_LABEL = DEFAULT_SAM2_VARIANT
_CHUNK = 16  # frames per resize call, bounding the float copy of a 720p clip


@dataclass(frozen=True)
class Sam2Variant:
    """A parsed variant string: model size and optional short side in pixels (None = native)."""

    variant: str
    short_side: int | None = None

    @property
    def model_id(self) -> str:
        return SAM2_VARIANTS[self.variant]

    @property
    def label(self) -> str:
        return self.variant if self.short_side is None else f"{self.variant}@{self.short_side}"


def parse_sam2_variant(text: str) -> Sam2Variant:
    """Parses `"<size>"` or `"<size>@<short side px>"`.

    Raises:
        ValueError: On an unknown size or a short side that is not a positive integer.
    """
    name, sep, px = text.partition("@")
    if name not in SAM2_VARIANTS:
        raise ValueError(f"unknown SAM 2.1 variant {text!r}; expected one of {tuple(SAM2_VARIANTS)}, "
                         "optionally followed by '@<short side px>'")
    if not sep:
        return Sam2Variant(name)
    if not px.isdigit() or int(px) < 1:
        raise ValueError(f"bad SAM 2.1 variant {text!r}: the short side after '@' must be a positive integer")
    return Sam2Variant(name, int(px))


def variant_label(size: str, short_side: int, native_short_side: int) -> str:
    """The variant string of `size` fed a `short_side` px short side on frames whose short side is `native_short_side`.

    The bare size when `short_side` is not below the native short side (`resize_short_side` then leaves the frames
    unchanged), else `<size>@<short_side>`.

    Raises:
        ValueError: On an unknown size or a short side below 1.
    """
    if short_side < 1 or native_short_side < 1:
        raise ValueError(f"short sides must be >= 1, got {short_side} and {native_short_side}")
    return parse_sam2_variant(size if short_side >= native_short_side else f"{size}@{short_side}").label


def choose_sam2_variant(rows: Sequence[Mapping[str, Any]], min_iou: float = 0.90) -> str:
    """The default-variant rule: the fastest row with mean mask IoU >= `min_iou`, else `REFERENCE_LABEL`.

    Args:
        rows: Sweep rows with `label`, `seconds` and `mean_iou` (the JSON rows of bench_sam2_variants.py).
            Rows with a non-finite `seconds` or `mean_iou` are ignored; ties keep the earlier row.
        min_iou: Inclusive lower bound on `mean_iou`.

    Returns:
        A `make_mask_worker` variant string: the chosen row's label, checked with `parse_sam2_variant`.

    Raises:
        ValueError: If the chosen row's label is not a variant string.
    """
    best: tuple[float, str] | None = None
    for row in rows:
        seconds, iou = float(row["seconds"]), float(row["mean_iou"])
        if not (math.isfinite(seconds) and math.isfinite(iou)) or iou < min_iou:
            continue
        if best is None or seconds < best[0]:
            best = (seconds, str(row["label"]))
    return REFERENCE_LABEL if best is None else parse_sam2_variant(best[1]).label


def resize_short_side(frames: Tensor, short_side: int | None) -> Tensor:
    """`(N,H,W,3)` frames downscaled (bilinear, antialiased) so the short side is `short_side`.

    Returns `frames` itself when `short_side` is None or not smaller than the current short side.

    Raises:
        ValueError: If `frames` is not `(N,H,W,3)` or `short_side < 1`.
    """
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"frames must be (N,H,W,3), got {tuple(frames.shape)}")
    if short_side is None:
        return frames
    if short_side < 1:
        raise ValueError(f"short_side must be >= 1, got {short_side}")
    n, h, w = int(frames.shape[0]), int(frames.shape[1]), int(frames.shape[2])
    if min(h, w) <= short_side:
        return frames
    scale = short_side / min(h, w)
    size = (short_side, max(1, round(w * scale))) if h <= w else (max(1, round(h * scale)), short_side)
    out = torch.empty((n, *size, 3), dtype=frames.dtype)
    for start in range(0, n, _CHUNK):
        chunk = frames[start:start + _CHUNK].permute(0, 3, 1, 2).float()
        resized = F.interpolate(chunk, size=size, mode="bilinear", align_corners=False, antialias=True)
        if frames.dtype == torch.uint8:
            resized = resized.round().clamp(0, 255)
        out[start:start + _CHUNK] = resized.to(frames.dtype).permute(0, 2, 3, 1)
    return out


def propagate_masks_at(frames: Tensor, keyframe_index: int, keyframe_mask: Tensor, short_side: int | None = None,
                       device: str = "cuda", predictor: Any = None, bf16: bool = True) -> Tensor:
    """`propagate_masks_sam2` on frames downscaled to `short_side`, with masks returned at native resolution.

    The prompt mask is downscaled with nearest-exact sampling and the propagated masks are upscaled the
    same way. With `short_side=None` (or not below the native short side) this is exactly
    `propagate_masks_sam2(frames, keyframe_index, keyframe_mask, device, predictor)` under the autocast below.

    Args:
        frames: `(N,H,W,3)` uint8 RGB frames.
        keyframe_index: Row of `frames` carrying the prompt.
        keyframe_mask: `(H,W)` boolean prompt mask at native resolution.
        short_side: Short side SAM 2 sees, in pixels, or None for native.
        device: Torch device string for the predictor.
        predictor: A `SAM2VideoPredictor`-compatible object, or None to load hiera-large.
        bf16: Run under bf16 autocast when `device` is CUDA.

    Returns:
        `(N,H,W)` bool masks.

    Raises:
        ValueError: If `frames` or `keyframe_mask` has the wrong shape, or `keyframe_index` is out of range.
    """
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"frames must be (N,H,W,3), got {tuple(frames.shape)}")
    h, w = int(frames.shape[1]), int(frames.shape[2])
    if tuple(keyframe_mask.shape) != (h, w):
        raise ValueError(f"keyframe_mask must be ({h},{w}), got {tuple(keyframe_mask.shape)}")
    small = resize_short_side(frames, short_side)
    small_hw = (int(small.shape[1]), int(small.shape[2]))
    mask = keyframe_mask.to(torch.bool)
    if small_hw != (h, w):
        mask = _resize_masks(mask[None], small_hw)[0]
    device_type = torch.device(device).type
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=bf16 and device_type == "cuda"):
        masks = propagate_masks_sam2(small, keyframe_index, mask, device=device, predictor=predictor)
    return masks if small_hw == (h, w) else _resize_masks(masks, (h, w))


def _resize_masks(masks: Tensor, size: tuple[int, int]) -> Tensor:
    """`(N,h,w)` bool masks resampled to `size` with nearest-exact sampling."""
    out = torch.empty((int(masks.shape[0]), *size), dtype=torch.bool)
    for start in range(0, int(masks.shape[0]), _CHUNK):
        chunk = masks[start:start + _CHUNK, None].float()
        out[start:start + _CHUNK] = F.interpolate(chunk, size=size, mode="nearest-exact")[:, 0] > 0.5
    return out
