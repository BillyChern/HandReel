"""Known answers for fastdo.tba.geometry (ray casting, Kabsch, relative poses), each with a mutation that fires."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from fastdo.tba.geometry import (
    kabsch,
    pixel_rays,
    posed_points,
    project,
    ray_mesh_depth,
    relative_poses,
    rotvec_to_matrix,
)

K = np.array([[600.0, 0.0, 320.0], [0.0, 610.0, 240.0], [0.0, 0.0, 1.0]])


def test_pixel_rays_reproject_exactly() -> None:
    uv = np.array([[10.5, 20.25], [320.0, 240.0], [600.0, 470.0]])
    r = pixel_rays(uv, K)
    assert np.allclose(r[:, 2], 1.0)
    for d in (0.3, 1.7):
        assert np.allclose(project(d * r, K), uv, atol=1e-10)


def test_ray_mesh_depth_hits_the_front_face_of_a_sphere() -> None:
    m = trimesh.creation.icosphere(subdivisions=5, radius=0.1)
    V = np.asarray(m.vertices) + np.array([0.0, 0.0, 0.8])
    rays = pixel_rays(np.array([[320.0, 240.0], [330.0, 245.0], [0.0, 0.0]]), K)
    d = ray_mesh_depth(rays, V, np.asarray(m.faces))
    assert d[0] == pytest.approx(0.7, abs=2e-4)                 # front of the sphere on the optical axis
    assert np.isfinite(d[1]) and d[1] > 0.7
    assert np.isnan(d[2])                                       # misses
    # mutation: the sphere moved 5 cm back moves the hit by exactly 5 cm
    d2 = ray_mesh_depth(rays[:1], V + np.array([0.0, 0.0, 0.05]), np.asarray(m.faces))
    assert d2[0] - d[0] == pytest.approx(0.05, abs=1e-9)


def test_kabsch_recovers_a_rigid_motion_and_refuses_reflections() -> None:
    rng = np.random.default_rng(0)
    a = rng.standard_normal((30, 3))
    R = rotvec_to_matrix(np.array([0.3, -0.5, 0.9]))
    t = np.array([0.1, -0.2, 0.7])
    R1, t1 = kabsch(a, a @ R.T + t)
    assert np.allclose(R1, R, atol=1e-10) and np.allclose(t1, t, atol=1e-10)
    R2, _ = kabsch(a, a * np.array([-1.0, 1.0, 1.0]))          # a mirror image: best PROPER rotation only
    assert np.linalg.det(R2) == pytest.approx(1.0)


def test_relative_poses_map_keyframe_points_to_each_row() -> None:
    rng = np.random.default_rng(1)
    P = np.tile(np.eye(4), (4, 1, 1))
    for k in range(4):
        P[k, :3, :3] = rotvec_to_matrix(rng.standard_normal(3) * 0.3)
        P[k, :3, 3] = rng.standard_normal(3) * 0.1 + np.array([0, 0, 0.6])
    p = rng.standard_normal((10, 3)) * 0.1
    M = relative_poses(P, 2)
    Xkey = posed_points(p, P[2], 0.37)
    for k in range(4):
        assert np.allclose(Xkey @ M[k, :3, :3].T + M[k, :3, 3], posed_points(p, P[k], 0.37), atol=1e-12)
