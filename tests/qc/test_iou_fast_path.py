"""The qc IoU fast path (crop counting + box-count closing) must equal the old full-image path exactly."""
from __future__ import annotations

import pytest
import torch

from fastdo.qc import features as F


@pytest.mark.parametrize("radius", [1, 2, 5, 13, 32])
@pytest.mark.parametrize("seed", range(4))
def test_box_count_closing_equals_max_pool_closing(radius: int, seed: int) -> None:
    g = torch.Generator().manual_seed(seed)
    h, w = int(torch.randint(5, 120, (1,), generator=g)), int(torch.randint(5, 120, (1,), generator=g))
    img = torch.rand(h, w, generator=g) < torch.rand(1, generator=g).item() * 0.3   # sparse splats, touching borders
    want = F._close(img.float()[None, None], radius)[0, 0] > 0.5
    assert torch.equal(F._close_binary(img, radius), want)


def _old_iou(masks: torch.Tensor, poses: torch.Tensor, scale: float, pts: torch.Tensor, K: torch.Tensor) -> float:
    sil = F.render_silhouettes(pts, poses, scale, K, (masks.shape[1], masks.shape[2]))
    inter = (sil & masks).sum(dim=(1, 2)).double()
    union = (sil | masks).sum(dim=(1, 2)).double()
    valid = union > 0
    return 0.0 if not bool(valid.any()) else float((inter[valid] / union[valid]).mean())


def test_reprojection_iou_equals_the_full_image_formula() -> None:
    T, H, W = 12, 90, 160
    g = torch.Generator().manual_seed(0)
    pts = torch.randn(600, 3, generator=g)
    pts = 0.05 * pts / pts.norm(dim=1, keepdim=True)
    poses = torch.eye(4).repeat(T, 1, 1)
    poses[:, 0, 3] = torch.linspace(-0.3, 0.3, T)            # slides off both image edges
    poses[:, 2, 3] = 0.5
    poses[5, 2, 3] = -0.5                                      # behind the camera: no render
    K = torch.tensor([[120.0, 0, 79.5], [0, 120.0, 44.5], [0, 0, 1]])
    masks = torch.rand(T, H, W, generator=g) < 0.02
    masks[:, 30:60, 60:100] = True
    masks[7] = False
    got = F.reprojection_iou(masks, poses, 1.0, pts, K)
    assert got == _old_iou(masks, poses, 1.0, pts, K)
    assert torch.equal(F.render_silhouettes(pts, poses, 1.0, K, (H, W)).any(dim=(1, 2)),
                       torch.tensor([bool(p[2, 3] > 0) and abs(float(p[0, 3])) < 0.45 for p in poses]))
