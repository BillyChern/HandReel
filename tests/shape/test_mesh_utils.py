"""Mesh normalization, watertightness repair and canonical surface sampling (D4); CPU only, no weights."""

from __future__ import annotations

import logging

import numpy as np
import pytest
import torch
import trimesh

from fastdo.shape.mesh_utils import canonical_samples, check_watertight, normalize_mesh


def _tetrahedron() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int64)
    return vertices, faces


def _unit_cube() -> tuple[np.ndarray, np.ndarray]:
    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    return np.asarray(box.vertices, dtype=np.float64), np.asarray(box.faces, dtype=np.int64)


def test_normalize_mesh_unit_diameter() -> None:
    vertices, faces = _tetrahedron()
    scaled = vertices * 7.0 + np.array([3.0, -2.0, 5.0])
    original = scaled.copy()
    extent = float(np.linalg.norm(scaled.max(0) - scaled.min(0)))

    out_vertices, out_faces, applied_scale = normalize_mesh(scaled, faces)

    lo, hi = out_vertices.min(0), out_vertices.max(0)
    assert abs(float(np.linalg.norm(hi - lo)) - 1.0) < 1e-6
    np.testing.assert_allclose((lo + hi) / 2.0, 0.0, atol=1e-12)      # recentred on the bounding-box centre
    assert abs(1.0 / applied_scale - extent) < 1e-9                   # the applied scale recovers the extent
    assert abs(1.0 / applied_scale - 7.0 * np.sqrt(3.0)) < 1e-9
    np.testing.assert_array_equal(out_faces, faces)
    np.testing.assert_array_equal(scaled, original)                    # the input is not mutated


@pytest.mark.parametrize("vertices", [np.zeros((4, 3)), np.array([[0.0, 0.0, np.nan]] * 4)])
def test_normalize_mesh_rejects_degenerate_or_non_finite_vertices(vertices: np.ndarray) -> None:
    _, faces = _tetrahedron()
    with pytest.raises(ValueError):
        normalize_mesh(vertices, faces)


def test_check_watertight_returns_a_closed_mesh_unchanged(caplog: pytest.LogCaptureFixture) -> None:
    vertices, faces = _unit_cube()
    with caplog.at_level(logging.WARNING):
        out_vertices, out_faces, watertight = check_watertight(vertices, faces)
    assert watertight
    np.testing.assert_array_equal(out_vertices, vertices)
    np.testing.assert_array_equal(out_faces, faces)
    assert not caplog.records


def test_check_watertight_never_raises(caplog: pytest.LogCaptureFixture) -> None:
    vertices, faces = _unit_cube()
    punctured = faces[2:]                                               # one quad face (two triangles) removed
    assert not trimesh.Trimesh(vertices, punctured, process=False).is_watertight

    with caplog.at_level(logging.WARNING):
        out_vertices, out_faces, watertight = check_watertight(vertices, punctured)

    if watertight:
        assert trimesh.Trimesh(out_vertices, out_faces, process=False).is_watertight
    else:
        assert any("watertight" in record.getMessage() for record in caplog.records)
        np.testing.assert_array_equal(out_faces, punctured)


@pytest.mark.parametrize("case", ["single_triangle", "out_of_range_face"])
def test_check_watertight_never_raises_on_unrepairable_input(case: str, caplog: pytest.LogCaptureFixture) -> None:
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    faces = np.array([[0, 1, 2]]) if case == "single_triangle" else np.array([[0, 1, 7]])

    with caplog.at_level(logging.WARNING):
        out_vertices, out_faces, watertight = check_watertight(vertices, faces)

    assert not watertight
    np.testing.assert_array_equal(out_vertices, vertices)               # returned unchanged
    np.testing.assert_array_equal(out_faces, faces)
    assert any("watertight" in record.getMessage() for record in caplog.records)


def test_canonical_samples_deterministic() -> None:
    vertices, faces = _unit_cube()
    points_a, normals_a = canonical_samples(vertices, faces, n=512, seed=3)
    points_b, normals_b = canonical_samples(vertices, faces, n=512, seed=3)

    assert points_a.shape == normals_a.shape == (512, 3)
    assert points_a.dtype == normals_a.dtype == torch.float32
    assert torch.equal(points_a, points_b) and torch.equal(normals_a, normals_b)   # bit-identical
    assert not torch.equal(points_a, canonical_samples(vertices, faces, n=512, seed=4)[0])
    assert torch.allclose(points_a.abs().max(dim=1).values, torch.tensor(0.5), atol=1e-6)  # on the surface
    torch.testing.assert_close(normals_a.norm(dim=1), torch.ones(512))


def test_canonical_samples_orients_normals_outward_for_an_inward_wound_mesh() -> None:
    vertices, faces = _unit_cube()
    points, normals = canonical_samples(vertices, faces[:, ::-1].copy(), n=256, seed=0)
    outward = (normals * (points - points.mean(dim=0))).sum(dim=1)
    assert float(outward.mean()) > 0.0
