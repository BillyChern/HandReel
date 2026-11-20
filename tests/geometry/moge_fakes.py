"""MoGe-2-shaped CPU modules whose `infer` IS moge's `MoGeModel.infer`.

`fastdo.geometry.moge_overlap` replaces `MoGeModel.infer` with a transcription of it and refuses any model whose infer
is something else, so its tests need modules that run the REAL infer post-processing (the scipy shift solve included)
around a cheap forward. Both classes borrow `infer`, `device` and `dtype` from `MoGeModel` and provide `forward` and
`num_tokens_range`; moge must be importable (the `frontend` extra).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from moge.model.v2 import MoGeModel
from moge.utils.geometry_torch import normalized_view_plane_uv
from torch import Tensor

__all__ = ["SceneMoGe", "TinyMoGe", "tag_frames"]


def _tags(image: Tensor) -> Tensor:
    """Pixel (0,0) channel 0 of each `(B,3,H,W)` [0,1] frame, as the uint8 tag written by `tag_frames`."""
    return (image[:, 0, 0, 0] * 255.0).round().long()


def tag_frames(n: int, h: int = 12, w: int = 16, seed: int = 0) -> Tensor:
    """`(n,h,w,3)` uint8 random frames whose pixel (0,0) channel 0 is the row index (the tag)."""
    frames = torch.from_numpy(np.random.default_rng(seed).integers(0, 256, size=(n, h, w, 3), dtype=np.uint8))
    frames[:, 0, 0, 0] = torch.arange(n, dtype=torch.uint8)
    return frames


class _MoGeShaped(torch.nn.Module):
    infer = MoGeModel.infer
    device = MoGeModel.device
    dtype = MoGeModel.dtype

    def __init__(self) -> None:
        super().__init__()
        self.num_tokens_range = [1200, 3600]
        self.log: list[tuple[str, Any]] = []
        self.tokens: list[int] = []            # the `num_tokens` of every forward, in call order


class TinyMoGe(_MoGeShaped):
    """A random linear 'head' on the pixels, run under infer's fp16 autocast (so `points`, `mask` and `metric_scale`
    come out float16, exercising infer's `.float()` casts). Frames tagged in `empty` have an all-zero mask (the
    `uv_lr_i_np.shape[0] < 2` branch of `recover_focal_shift`); in frames tagged in `negative` a checkerboard of masked
    pixels has negative depth (the `mask_binary &= z > 0` branch); every 5th column's mask is exactly 0.5 (infer's
    strict `> 0.5` threshold). `with_mask=False` omits the mask output and `with_normal=True` adds a normal output, for
    infer's other branches. A `num_tokens` other than infer's default 3600 scales the depth by `num_tokens / 3600`
    (the default computation is untouched), so a token count that does not arrive changes the output."""

    def __init__(self, empty: Sequence[int] = (), negative: Sequence[int] = (), with_mask: bool = True,
                 with_normal: bool = False, seed: int = 0) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(3, 8)
        generator = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            self.lin.weight.copy_(torch.randn(8, 3, generator=generator))
            self.lin.bias.copy_(torch.randn(8, generator=generator) * 0.1)
        self.empty, self.negative = list(empty), list(negative)
        self.with_mask, self.with_normal = with_mask, with_normal

    def forward(self, image: Tensor, num_tokens: int) -> dict[str, Tensor]:
        tags = _tags(image)
        self.log.append(("forward", tags.tolist()))
        self.tokens.append(num_tokens)
        _, _, h, w = image.shape
        feat = self.lin(image.permute(0, 2, 3, 1))                           # (B,H,W,8), fp16 under infer's autocast
        uv = normalized_view_plane_uv(w, h, dtype=feat.dtype, device=feat.device)
        z = 1.0 + feat[..., 0].sigmoid()
        if num_tokens != 3600:                                               # infer's default leaves z untouched
            z = z * (num_tokens / 3600)
        negative = torch.isin(tags, torch.tensor(self.negative, dtype=torch.long))[:, None, None]
        checker = ((torch.arange(h)[:, None] + torch.arange(w)[None, :]) % 3 == 0)[None]
        z = torch.where(negative & checker, torch.full_like(z, -0.5), z)
        xy = uv * z[..., None] * 0.9 + 0.05 * feat[..., 1:3]
        out = {"points": torch.cat([xy, z[..., None]], dim=-1),
               "metric_scale": feat[..., 4].mean(dim=(1, 2)).exp()}
        if self.with_mask:
            empty = torch.isin(tags, torch.tensor(self.empty, dtype=torch.long))[:, None, None]
            mask = feat[..., 3].sigmoid()
            mask[..., ::5] = 0.5                                              # infer's threshold is strict: > 0.5
            out["mask"] = torch.where(empty, torch.zeros_like(mask), mask)
        if self.with_normal:
            out["normal"] = F.normalize(feat[..., 5:8], dim=-1)
        return out


class SceneMoGe(_MoGeShaped):
    """`tests.engine.fakes.FakeMoGe`'s scene through the real infer: the forward returns the tag's depth `z` with
    `xy = uv * z / focal` for the focal of `fov_x` (so infer's shift solve lands at ~0 and its depth is ~`z`), mask 1
    where `z` is finite and 0 where the fake marks it invalid (inf), and metric scale 1."""

    def __init__(self, depth: dict[int, np.ndarray], fov_x: float, height: int, width: int) -> None:
        super().__init__()
        self.depth = depth
        self.one = torch.nn.Parameter(torch.ones(()))
        aspect = width / height
        self.focal = aspect / (1 + aspect ** 2) ** 0.5 / math.tan(math.radians(fov_x) / 2)

    def forward(self, image: Tensor, num_tokens: int) -> dict[str, Tensor]:
        tags = _tags(image)
        self.log.append(("forward", tags.tolist()))
        self.tokens.append(num_tokens)
        b, _, h, w = image.shape
        z = torch.stack([torch.from_numpy(self.depth[int(t)]) for t in tags]).to(torch.float32)
        finite = torch.isfinite(z)
        z = torch.where(finite, z, torch.ones_like(z)) * self.one
        uv = normalized_view_plane_uv(w, h, dtype=z.dtype, device=z.device)
        return {"points": torch.cat([uv * z[..., None] / self.focal, z[..., None]], dim=-1),
                "mask": finite.to(torch.float32), "metric_scale": torch.ones(b) * self.one}
