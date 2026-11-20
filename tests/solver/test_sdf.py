import numpy as np
import torch
import trimesh

from fastdo.solver.sdf import SdfGrid, mesh_sdf_grid, sdf_lookup


def test_sphere_sdf_values_and_gradient() -> None:
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
    sdf = mesh_sdf_grid(np.asarray(mesh.vertices), np.asarray(mesh.faces), resolution=48)
    y = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.8], [0.3, 0.0, 0.0]]], dtype=torch.float64)
    phi, grad = sdf_lookup(sdf, y)
    assert abs(float(phi[0, 0]) + 0.5) < 0.03          # centre: -radius
    assert abs(float(phi[0, 1]) - 0.3) < 0.03          # outside: distance to surface
    assert abs(float(phi[0, 2]) + 0.2) < 0.03
    assert torch.allclose(grad[0, 1], torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64), atol=0.1)
    assert torch.allclose(grad[0, 2], torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64), atol=0.1)


def test_lookup_outside_grid_is_clamped_finite() -> None:
    values = torch.zeros(8, 8, 8, dtype=torch.float64)
    sdf = SdfGrid(values=values, origin=torch.zeros(3, dtype=torch.float64), step=0.1)
    phi, grad = sdf_lookup(sdf, torch.full((1, 1, 3), 5.0, dtype=torch.float64))
    assert torch.isfinite(phi).all() and torch.isfinite(grad).all()


def test_sdf_values_equal_lookup_values() -> None:
    """The value-only lookup is bitwise the value half of `sdf_lookup`, in and outside the grid."""
    from fastdo.solver.sdf import sdf_values

    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    sdf = mesh_sdf_grid(np.asarray(box.vertices), np.asarray(box.faces), resolution=24)
    y = torch.rand(1, 1000, 3, generator=torch.Generator().manual_seed(0), dtype=torch.float64) * 2.0 - 1.0
    assert torch.equal(sdf_values(sdf, y), sdf_lookup(sdf, y)[0])
