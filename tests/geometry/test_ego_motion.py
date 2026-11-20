"""Camera ego-motion from pointmaps -- registration, pose graph, tracker plan (synthetic known answers)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from fastdo.geometry.ego_motion import (
    EgoMotionConfig,
    load_ego_motion,
    register_similarity,
    rotation_angle_deg,
    save_ego_motion,
    sim3_matrix,
    smooth_trajectory,
    so3_exp,
    so3_log,
    solve_ego_motion,
    umeyama,
)
from fastdo.geometry.ego_tracks import (
    exclusion_masks,
    plan_correspondences,
    solve_from_samples,
    working_size,
)


def _traj(n: int, rng: np.random.Generator, drift: float = 0.03) -> np.ndarray:
    """cam_to_world Sim(3) per row: smooth head motion + a per-row metric-scale wobble."""
    out = [np.eye(4)]
    for _ in range(n - 1):
        r = so3_exp(rng.normal(0, math.radians(1.5), 3))
        t = rng.normal(0, 0.02, 3)
        step = sim3_matrix(1.0, r, t)
        out.append(out[-1] @ step)
    s = np.exp(rng.normal(0, drift, n))
    s[0] = 1.0
    return np.stack([m @ np.diag([s_k, s_k, s_k, 1.0]) for m, s_k in zip(out, s, strict=True)])


def _edges_data(c2w: np.ndarray, rng: np.random.Generator, skips=(1, 3), n_pts=200, outlier=0.2, noise=0.005):
    n = len(c2w)
    world = rng.uniform([-3, -2, 2], [3, 2, 8], (4000, 3))
    edges, xs, ys = [], [], []
    for i in range(n):
        for k in skips:
            j = i + k
            if j >= n:
                continue
            pick = rng.choice(len(world), n_pts, replace=False)
            w = np.c_[world[pick], np.ones(n_pts)]
            xi = (np.linalg.inv(c2w[i]) @ w.T).T[:, :3]
            yj = (np.linalg.inv(c2w[j]) @ w.T).T[:, :3]
            xi = xi * (1 + rng.normal(0, noise, (n_pts, 1)))
            yj = yj * (1 + rng.normal(0, noise, (n_pts, 1)))
            bad = rng.random(n_pts) < outlier
            yj[bad] += rng.normal(0, 0.5, (int(bad.sum()), 3))       # moving things
            edges.append((i, j))
            xs.append(xi)
            ys.append(yj)
    return np.array(edges), xs, ys


def test_so3_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(50):
        w = rng.normal(0, 1, 3)
        w *= min(1.0, 3.0 / np.linalg.norm(w))
        np.testing.assert_allclose(so3_log(so3_exp(w)), w, atol=1e-8)


def test_umeyama_exact():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, (30, 3))
    r = so3_exp(rng.normal(0, 0.5, 3))
    y = 1.3 * x @ r.T + np.array([0.1, -0.2, 0.3])
    s, r2, _t = umeyama(x, y)
    assert s == pytest.approx(1.3)
    np.testing.assert_allclose(r2, r, atol=1e-9)
    s, _, _ = umeyama(x, y, similarity=False)
    assert s == 1.0


def test_register_rejects_outliers():
    rng = np.random.default_rng(2)
    x = rng.uniform([-2, -2, 2], [2, 2, 6], (300, 3))
    r = so3_exp(np.array([0.02, -0.03, 0.01]))
    y = 0.97 * x @ r.T + np.array([0.05, 0.0, -0.02])
    y[:90] += rng.normal(0, 0.4, (90, 3))
    fit = register_similarity(x, y, EgoMotionConfig(), np.random.default_rng(0))
    assert fit.ok
    assert fit.s == pytest.approx(0.97, abs=1e-3)       # a few outliers fall inside the 3% gate by chance
    assert rotation_angle_deg(fit.r.T @ r) < 0.05
    assert 205 <= fit.n_inliers <= 225


def test_register_fails_on_noise():
    rng = np.random.default_rng(3)
    fit = register_similarity(rng.normal(0, 1, (100, 3)) + [0, 0, 4], rng.normal(0, 1, (100, 3)) + [0, 0, 4],
                              EgoMotionConfig(), np.random.default_rng(0))
    assert not fit.ok


@pytest.mark.parametrize("similarity", [True, False])
def test_trajectory_known_answer(similarity):
    rng = np.random.default_rng(4)
    c2w = _traj(40, rng, drift=0.03 if similarity else 0.0)
    edges, xs, ys = _edges_data(c2w, rng)
    cfg = EgoMotionConfig(similarity=similarity, smooth_half_rows=0)   # the graph itself; smoothing tested below
    ego = solve_ego_motion(40, edges, xs, ys, cfg)
    assert ego.valid.all() and ego.reference_row == 0
    assert ego.summary["ok"]
    rot_err = [rotation_angle_deg(ego.cam_to_world[t, :3, :3] / ego.scale[t] @ (c2w[t, :3, :3] /
               np.cbrt(np.linalg.det(c2w[t, :3, :3]))).T) for t in range(40)]
    assert max(rot_err) < 0.3
    trans_err = np.linalg.norm(ego.cam_to_world[:, :3, 3] - c2w[:, :3, 3], axis=1)
    assert trans_err.max() < 0.03
    if similarity:
        true_s = np.cbrt(np.linalg.det(c2w[:, :3, :3]))
        assert np.abs(np.log(ego.scale / true_s)).max() < 0.01


def test_se3_is_biased_under_scale_drift():
    """The reason for Sim(3): the same data with a 3% per-row scale wobble breaks SE(3), not Sim(3)."""
    rng = np.random.default_rng(5)
    c2w = _traj(30, rng, drift=0.03)
    edges, xs, ys = _edges_data(c2w, rng)
    err = {}
    for sim in (True, False):
        ego = solve_ego_motion(30, edges, xs, ys, EgoMotionConfig(similarity=sim, smooth_half_rows=0))
        pts = np.array([0.0, 0.0, 4.0, 1.0])            # a static point 4 m ahead of row 0
        world = c2w[0] @ pts
        cam = [np.linalg.inv(c2w[t]) @ world for t in range(30)]
        back = [ego.cam_to_world[t] @ cam[t] for t in range(30) if ego.valid[t]]
        err[sim] = float(np.max([np.linalg.norm(b[:3] - world[:3]) for b in back]))
    assert err[True] < 0.05 < err[False]


def test_disconnected_rows_invalid_and_loops():
    rng = np.random.default_rng(6)
    c2w = _traj(20, rng)
    edges, xs, ys = _edges_data(c2w, rng)
    for e, (i, j) in enumerate(edges):                   # cut every edge touching row 19
        if j == 19 or i == 19:
            ys[e] = rng.normal(0, 1, ys[e].shape) + [0, 0, 4]
    ego = solve_ego_motion(20, edges, xs, ys, EgoMotionConfig())
    assert not ego.valid[19] and ego.valid[:19].all()
    loops = ego.loop_rot_deg[np.isfinite(ego.loop_rot_deg)]
    assert loops.size > 10 and np.median(loops) < 0.3


def test_save_load(tmp_path):
    rng = np.random.default_rng(7)
    c2w = _traj(10, rng)
    edges, xs, ys = _edges_data(c2w, rng)
    cfg = EgoMotionConfig()
    ego = solve_ego_motion(10, edges, xs, ys, cfg)
    save_ego_motion(tmp_path / "ego_motion.npz", ego, np.arange(10) * 3, cfg, extra={"seconds": 1.0})
    z = load_ego_motion(tmp_path / "ego_motion.npz")
    np.testing.assert_array_equal(z["cam_to_world"], ego.cam_to_world)
    assert z["summary"]["ok"] == ego.summary["ok"] and z["summary"]["seconds"] == 1.0
    assert z["config"]["skips"] == [1, 3]


def test_config_validation():
    with pytest.raises(ValueError):
        EgoMotionConfig(skips=(2, 3))
    with pytest.raises(ValueError):
        EgoMotionConfig.from_mapping({"nope": 1})
    assert EgoMotionConfig.from_mapping({"skips": [3, 1]}).skips == (1, 3)


def _texture(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    import cv2

    base = rng.integers(0, 255, (h // 8, w // 8)).astype(np.uint8)
    img = cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC)
    return np.repeat(img[:, :, None], 3, axis=2)


def test_plan_tracks_a_shift_and_honours_exclusion():
    rng = np.random.default_rng(8)
    big = _texture(520, 700, rng)
    frames = np.stack([big[10 + 2 * t:10 + 2 * t + 480, 20 + 3 * t:20 + 3 * t + 640] for t in range(5)])
    cfg = EgoMotionConfig(long_side=640)
    excl = np.zeros((5, 480, 640), bool)
    excl[:, :, :200] = True
    plan = plan_correspondences(frames, excl, cfg)
    assert [tuple(e) for e in plan.edges] == [(0, 1), (0, 3), (1, 2), (1, 4), (2, 3), (3, 4)]
    for (i, j), ai, aj in zip(plan.edges, plan.edge_i, plan.edge_j, strict=True):
        assert len(ai) > 30
        du = plan.u[aj] - plan.u[ai]
        dv = plan.v[aj] - plan.v[ai]
        assert np.median(du) == -3 * (j - i) and np.median(dv) == -2 * (j - i)
        assert (plan.u[ai] >= 200).all()


def test_solve_from_samples_on_a_rendered_plane():
    """End to end on pixels: a textured plane seen by a translating camera; pointmaps sampled at the plan."""
    rng = np.random.default_rng(9)
    big = _texture(600, 800, rng)
    n, h, w = 6, 480, 640
    frames = np.stack([big[20:20 + h, 30 + 4 * t:30 + 4 * t + w] for t in range(n)])
    f, z = 500.0, 3.0
    uu, vv = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float))
    pm = np.stack([(uu - w / 2) * z / f, (vv - h / 2) * z / f, np.full_like(uu, z)], -1)
    pointmaps = np.stack([pm for _ in range(n)])            # the plane in each camera frame
    cfg = EgoMotionConfig(long_side=640)
    plan = plan_correspondences(frames, np.zeros((n, h, w), bool), cfg)
    samples = pointmaps[plan.frame, plan.v, plan.u]
    ego = solve_from_samples(plan, samples, cfg)
    assert ego.valid.all()
    # 4 px per row at f=500, z=3 -> the camera moves +0.024 m in x per row (image content shifts left)
    np.testing.assert_allclose(ego.cam_to_world[:, 0, 3], 0.024 * np.arange(n), atol=2e-3)


def test_exclusion_hands_and_object():
    cfg = EgoMotionConfig(long_side=320)
    h, w, factor = working_size(480, 640, 320)
    assert (h, w, factor) == (240, 320, 2.0)
    masks = np.zeros((2, 480, 640), bool)
    masks[0, 100:120, 100:120] = True
    joints = {"right": np.array([[[0.1, 0.0, 1.0]] * 3, [[np.nan] * 3] * 3])}
    ex = exclusion_masks(2, 480, 640, cfg, masks, joints, {"right": np.array([True, True])},
                         np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]]))
    assert ex[0, 55, 55] and not ex[1, 55, 55]
    assert ex[0, 239, 185] and ex[0, 120, 185] and not ex[0, 100, 185]


def _const_velocity(n: int) -> np.ndarray:
    out = np.zeros((n, 4, 4))
    for t in range(n):
        out[t] = sim3_matrix(math.exp(0.004 * t), so3_exp(np.array([0.01, -0.02, 0.005]) * t),
                             np.array([0.02, -0.01, 0.03]) * t)
    return out


def test_smoothing_keeps_constant_velocity_interior_and_runs_apart():
    """A constant-velocity Sim(3) trajectory is a fixed point of the symmetric window away from run ends; rows of
    different runs never mix; invalid rows stay NaN; half 0 is the identity."""
    c2w = _const_velocity(30)
    valid = np.ones(30, bool)
    valid[15] = False
    c2w[15] = np.nan
    out = smooth_trajectory(c2w, valid, 3)
    assert np.isnan(out[15]).all()
    for t in [*range(3, 12), *range(19, 27)]:            # >= 3 rows from either end of runs [0,14] and [16,29]
        np.testing.assert_allclose(out[t], c2w[t], atol=1e-9)
    np.testing.assert_array_equal(smooth_trajectory(c2w, valid, 0)[valid], c2w[valid])
    first_only = valid & (np.arange(30) < 15)                # row 14 never sees rows 16..17 of the next run
    np.testing.assert_allclose(out[14], smooth_trajectory(c2w, first_only, 3)[14], atol=1e-12)


def test_smoothing_reduces_row_jitter():
    rng = np.random.default_rng(11)
    truth = _const_velocity(60)
    noisy = truth.copy()
    for t in range(60):
        noisy[t] = sim3_matrix(math.exp(0.004 * t + rng.normal(0, 0.01)),
                               so3_exp(np.array([0.01, -0.02, 0.005]) * t + rng.normal(0, 0.005, 3)),
                               truth[t, :3, 3] + rng.normal(0, 0.01, 3))
    out = smooth_trajectory(noisy, np.ones(60, bool), 3)
    err_raw = np.linalg.norm(noisy[5:55, :3, 3] - truth[5:55, :3, 3], axis=1).mean()
    err_s = np.linalg.norm(out[5:55, :3, 3] - truth[5:55, :3, 3], axis=1).mean()
    assert err_s < 0.6 * err_raw


def test_solve_applies_smoothing_and_keeps_raw():
    rng = np.random.default_rng(4)
    c2w = _traj(40, rng, drift=0.03)
    edges, xs, ys = _edges_data(c2w, rng)
    raw = solve_ego_motion(40, edges, xs, ys, EgoMotionConfig(smooth_half_rows=0))
    sm = solve_ego_motion(40, edges, xs, ys, EgoMotionConfig())
    np.testing.assert_array_equal(sm.cam_to_world_raw, raw.cam_to_world)
    np.testing.assert_allclose(sm.cam_to_world, smooth_trajectory(raw.cam_to_world, raw.valid, 3), atol=1e-12)
    assert sm.summary["smooth_half_rows"] == 3 and raw.summary["smooth_half_rows"] == 0
