"""The initial-scale ruler of every generated-shape (internet) clip, and its depth-robust variant."""
from __future__ import annotations

import pytest
import torch

from fastdo.solver.init import scale_from_extent, scale_from_lateral_extent


def _scene(h: int = 64, w: int = 64, f: float = 80.0, z_obj: float = 0.8, radius_px: float = 14.0):
    """A fronto-parallel disk of an object at `z_obj` in front of a wall at 3 m, as a metric pointmap."""
    v, u = torch.meshgrid(torch.arange(h, dtype=torch.float64), torch.arange(w, dtype=torch.float64), indexing="ij")
    xn, yn = (u - w / 2) / f, (v - h / 2) / f
    mask = (u - w / 2) ** 2 + (v - h / 2) ** 2 < radius_px ** 2
    z = torch.where(mask, torch.full_like(xn, z_obj), torch.full_like(xn, 3.0))
    pointmap = torch.stack([xn * z, yn * z, z], dim=-1)
    g = torch.linspace(-0.1, 0.1, 5, dtype=torch.float64)
    surface = torch.stack(torch.meshgrid(g, g, g, indexing="ij"), dim=-1).reshape(-1, 3)
    return pointmap, mask, surface


def _fly(pointmap: torch.Tensor, mask: torch.Tensor, fraction: float, depth: float = 3.0) -> torch.Tensor:
    """Push a `fraction` of the masked pixels to `depth` ALONG THEIR OWN RAYS: a depth-edge ("flying")
    pixel keeps its image position and only its depth is wrong, which is exactly what a silhouette does."""
    out = pointmap.clone()
    idx = mask.nonzero()
    pick = idx[torch.linspace(0, idx.shape[0] - 1, int(fraction * idx.shape[0])).long()]
    p = out[pick[:, 0], pick[:, 1]]
    out[pick[:, 0], pick[:, 1]] = p * (depth / p[:, 2:3])
    return out


def test_flying_pixels_inflate_the_3d_extent_but_not_the_lateral_one() -> None:
    pointmap, mask, surface = _scene()
    clean_ext = scale_from_extent(pointmap, mask, surface)
    clean_lat = scale_from_lateral_extent(pointmap, mask, surface)
    dirty = _fly(pointmap, mask, 0.10)                  # 10%: above the 5% the quantile trim can absorb
    dirty_ext = scale_from_extent(dirty, mask, surface)
    dirty_lat = scale_from_lateral_extent(dirty, mask, surface)

    assert dirty_ext > 3.0 * clean_ext, "the fixture must reproduce the pathology, or the test proves nothing"
    assert dirty_lat == pytest.approx(clean_lat, rel=0.02), (
        f"lateral moved {dirty_lat / clean_lat:.3f}x under 10% flying pixels; it should barely move")


def test_the_lateral_ruler_tolerates_contamination_up_to_its_median() -> None:
    """Depth enters only through a median, so the breakdown point is ~50%, not the 3D quantile's 5%."""
    pointmap, mask, surface = _scene()
    clean = scale_from_lateral_extent(pointmap, mask, surface)
    assert scale_from_lateral_extent(_fly(pointmap, mask, 0.40), mask, surface) == pytest.approx(clean, rel=0.02)
    assert scale_from_lateral_extent(_fly(pointmap, mask, 0.60), mask, surface) > 2.0 * clean


def test_both_rulers_are_equivariant_along_the_gauge_orbit() -> None:
    """`X -> lambda X` leaves the image invariant and must scale either ruler by exactly lambda."""
    pointmap, mask, surface = _scene()
    for ruler in (scale_from_extent, scale_from_lateral_extent):
        assert ruler(2.5 * pointmap, mask, surface) == pytest.approx(2.5 * ruler(pointmap, mask, surface))


def test_the_lateral_source_is_reachable_through_registration() -> None:
    from fastdo.engine.workers_solve import registration_init_scale

    pointmap, mask, surface = _scene()
    scale, source = registration_init_scale("lateral", {"backend": "sam3d"}, pointmap, mask, surface)
    assert source == "lateral" and scale == pytest.approx(scale_from_lateral_extent(pointmap, mask, surface))


def test_too_few_points_raises() -> None:
    pointmap, mask, surface = _scene(radius_px=1.0)
    with pytest.raises(ValueError, match="too few"):
        scale_from_lateral_extent(pointmap, mask, surface)
