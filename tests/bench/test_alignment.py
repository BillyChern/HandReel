"""Umeyama similarity and scaled ICP (PyTorch3D `iterative_closest_point(estimate_scale=True)` semantics)."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from fastdo.bench.alignment import SimilarityFit, scaled_icp, umeyama_similarity
from fastdo.bench.metrics import f_score


def _box_points() -> np.ndarray:
    mesh = trimesh.creation.box(extents=[0.16, 0.21, 0.06])
    return np.asarray(trimesh.sample.sample_surface(mesh, 5000, seed=0)[0], dtype=np.float64)


def _rot(deg: tuple[float, float, float]) -> np.ndarray:
    return Rotation.from_euler("xyz", deg, degrees=True).as_matrix()


def test_umeyama_recovers_known_similarity() -> None:
    src = np.random.default_rng(0).normal(size=(500, 3))
    s, r, t = 1.13, _rot((20.0, -35.0, 50.0)), np.array([0.1, -0.2, 0.3])
    dst = s * src @ r.T + t
    s_hat, r_hat, t_hat = umeyama_similarity(src, dst)
    assert abs(s_hat - s) < 1e-9
    assert np.abs(r_hat - r).max() < 1e-9
    assert np.abs(t_hat - t).max() < 1e-9


def test_umeyama_never_returns_reflection() -> None:
    src = np.random.default_rng(0).normal(size=(500, 3))
    dst = src.copy()
    dst[:, 0] = -dst[:, 0]
    _s, r, _t = umeyama_similarity(src, dst)
    assert abs(np.linalg.det(r) - 1.0) < 1e-9


def test_umeyama_rejects_bad_input() -> None:
    src = np.random.default_rng(0).normal(size=(500, 3))
    with pytest.raises(ValueError):
        umeyama_similarity(src, src[:499])
    with pytest.raises(ValueError):
        umeyama_similarity(src[:2], src[:2])


def test_scaled_icp_converges_on_box() -> None:
    g = _box_points()
    pred = 1.08 * g @ _rot((8.0, -8.0, 4.0)).T + np.array([0.03, -0.015, 0.03])
    aligned, fit = scaled_icp([pred], [g])
    assert isinstance(fit, SimilarityFit)
    assert len(aligned) == 1 and aligned[0].shape == g.shape
    assert np.linalg.norm(aligned[0] - g, axis=1).max() < 1e-4
    assert f_score(aligned[0], g, 0.005) == 1.0
    assert fit.converged and 0 < fit.iterations < 100
    assert abs(np.linalg.det(fit.rotation) - 1.0) < 1e-9
    assert fit.rmse < 1e-4
    # The returned similarity is the one that produced the aligned points.
    assert np.allclose(fit.scale * pred @ fit.rotation.T + fit.translation, aligned[0], atol=1e-12)


def test_scaled_icp_single_fit_across_frames() -> None:
    g = _box_points()
    gts = [g @ _rot(deg).T + off for deg, off in (((0.0, 0.0, 0.0), np.array([0.0, 0.0, 0.5])),
                                                   ((25.0, -10.0, 40.0), np.array([0.05, -0.02, 0.55])),
                                                   ((-30.0, 20.0, -15.0), np.array([-0.04, 0.03, 0.6])))]
    s, r, t = 1.05, _rot((3.0, -4.0, 2.0)), np.array([0.01, -0.012, 0.015])
    preds = [s * gk @ r.T + t for gk in gts]
    aligned, fit = scaled_icp(preds, gts)
    assert fit.converged
    for a, gk in zip(aligned, gts, strict=True):
        assert np.linalg.norm(a - gk, axis=1).max() < 1e-4

    preds[1] = preds[1] + np.array([0.02, 0.0, 0.0])            # independent 20 mm offset on frame 1 only
    clip_aligned, _clip_fit = scaled_icp(preds, gts)
    assert np.linalg.norm(clip_aligned[1] - gts[1], axis=1).max() > 5e-3
    for p, gk in zip(preds, gts, strict=True):
        frame_aligned, _frame_fit = scaled_icp([p], [gk])
        assert np.linalg.norm(frame_aligned[0] - gk, axis=1).max() < 1e-4


def test_scaled_icp_rejects_bad_input() -> None:
    g = _box_points()
    with pytest.raises(ValueError):
        scaled_icp([g, g], [g])
    with pytest.raises(ValueError):
        scaled_icp([], [])
    with pytest.raises(ValueError):
        scaled_icp([g[:, :2]], [g])


def test_scaled_icp_identity_on_identical_clouds() -> None:
    g = _box_points()
    aligned, fit = scaled_icp([g], [g])
    assert fit.converged
    assert np.abs(aligned[0] - g).max() < 1e-12
    assert abs(fit.scale - 1.0) < 1e-12 and fit.rmse < 1e-12
