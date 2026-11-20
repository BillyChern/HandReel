"""DexYCB per-frame image, depth and segmentation access, back-projected to camera-frame pointmaps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor

__all__ = ["HAND_SEG_ID", "DexYCBFrames", "FramePaths"]

HAND_SEG_ID = 255


@dataclass(frozen=True)
class FramePaths:
    color: Path
    depth: Path
    labels: Path


class DexYCBFrames:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def frame_paths(self, seq_id: str) -> list[FramePaths]:
        d = self.root / seq_id
        colors = sorted(d.glob("color_*.jpg"))
        out = []
        for c in colors:
            idx = c.stem.split("_")[-1]
            out.append(FramePaths(color=c, depth=d / f"aligned_depth_to_color_{idx}.png", labels=d / f"labels_{idx}.npz"))
        if not out:
            raise FileNotFoundError(f"no color frames under {d}")
        return out

    def load_observations(self, seq_id: str, K: Tensor, object_id: int, stride: int = 3
                          ) -> tuple[Tensor, Tensor, Tensor]:
        paths = self.frame_paths(seq_id)[::stride]
        masks, hands, pmaps = [], [], []
        fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
        for p in paths:
            seg = np.load(p.labels)["seg"]
            depth = np.asarray(Image.open(p.depth), dtype=np.float32) / 1000.0
            H, W = depth.shape
            vv, uu = np.meshgrid(np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32), indexing="ij")
            z = np.where(depth > 0, depth, np.nan)
            pm = np.stack([(uu - cx) * z / fx, (vv - cy) * z / fy, z], axis=-1)
            masks.append(torch.as_tensor(seg == object_id))
            hands.append(torch.as_tensor(seg == HAND_SEG_ID))
            pmaps.append(torch.as_tensor(pm, dtype=torch.float32))
        return torch.stack(masks), torch.stack(hands), torch.stack(pmaps)
