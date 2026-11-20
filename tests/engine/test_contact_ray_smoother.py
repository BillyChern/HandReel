"""Contact-gated along-ray log-depth smoother (`fastdo.engine.contact_ray_smoother`), CPU only."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from fastdo.engine.contact_ray_smoother import (
    ContactRaySmootherConfig,
    contact_depth_measurements,
    rts_random_walk,
    smooth_contact_ray,
    smoother_from_mapping,
)
from fastdo.engine.worker_support import load_npz, save_npz
from fastdo.engine.workers_solve import SolveStageConfig, make_solve_worker
from fastdo.qc.interaction import frame_min_contact
from fastdo.solver.synthetic import make_synthetic_clip
from tests.engine.fakes_solve import (
    SMALL_CLIP,
    SMALL_SOLVE,
    RecordingSolve,
    write_synthetic_solve_inputs,
)

_ROOT_CONFIG = Path(__file__).resolve().parents[2] / "run" / "conf" / "config.yaml"
CFG = ContactRaySmootherConfig()
BLOCK = {"touch_gate_mm": 56.0, "max_log_shift": 0.2, "meas_floor_mm": 10.0, "min_gated_frames": 3,
         "q_floor": 1e-6, "patch_k": 20, "contact_eps_mm": 5.0, "hands": "prompt", "grid_mm": [-300.0, 300.0, 2.0]}


# --- a synthetic scene with known depth -------------------------------------------------------------------------

def _sphere(n: int = 600, radius: float = 0.04, seed: int = 0) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal((n, 3))
    return radius * v / np.linalg.norm(v, axis=1, keepdims=True)


def _rot(seed: int) -> np.ndarray:
    q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((3, 3)))
    return q * np.sign(np.linalg.det(q))


def _scene(n: int = 30, log_err: np.ndarray | float = 0.1, touch: np.ndarray | None = None, seed: int = 0):
    """True object centroids on a slow path ~0.5 m away; the SOLVED poses put each centroid at exp(log_err) of its true
    distance along the same ray. Touching rows: a hand cupping the camera-facing side of the ball: the 21 joints and a
    30-vertex contact patch ON the true surface's camera-facing cap, the other 170 vertices 15-60 mm outside that cap
    (the back of the hand, toward the camera); other rows: the whole hand 0.25 m off to the side."""
    mesh = _sphere(seed=seed)
    t = np.arange(n)
    true_c = np.stack([0.05 * np.sin(t / 7.0), 0.02 * np.cos(t / 5.0), 0.5 + 0.03 * np.sin(t / 9.0)], axis=1)
    err = np.broadcast_to(np.asarray(log_err, dtype=float), (n,))
    poses = np.tile(np.eye(4), (n, 1, 1))
    joints, verts = np.zeros((n, 21, 3)), np.zeros((n, 200, 3))
    touch = np.ones(n, bool) if touch is None else touch
    rng = np.random.default_rng(seed + 1)
    for i in range(n):
        rot = _rot(seed + 10 + i)
        poses[i, :3, :3] = rot
        poses[i, :3, 3] = np.exp(err[i]) * true_c[i]                # solved centroid on the true ray (mesh centred)
        surf = mesh @ rot.T                                          # true surface, centred
        nrm = surf / np.linalg.norm(surf, axis=1, keepdims=True)
        cap = np.flatnonzero(nrm @ (-true_c[i] / np.linalg.norm(true_c[i])) > 0.6)   # faces the camera
        j_idx, v_idx = rng.choice(cap, 21, replace=False), rng.choice(cap, 200, replace=True)
        out = np.r_[np.zeros(30), rng.uniform(0.015, 0.06, 170)][:, None]
        hand_j, hand_v = surf[j_idx] + true_c[i], surf[v_idx] + out * nrm[v_idx] + true_c[i]
        off = np.zeros(3) if touch[i] else np.array([0.25, 0.0, 0.0])
        joints[i], verts[i] = hand_j + off, hand_v + off
    return poses, mesh, joints, true_c, verts


def _project(points: np.ndarray) -> np.ndarray:
    return points[..., :2] / points[..., 2:3]


# --- config -----------------------------------------------------------------------------------------------------

def test_config_off_by_default_and_blocks_validate() -> None:
    root = OmegaConf.load(_ROOT_CONFIG)
    assert root.solve_stage.contact_ray_smoother is None
    assert SolveStageConfig.from_mapping(root.solve_stage).contact_ray_smoother is None
    assert SolveStageConfig().contact_ray_smoother is None
    assert smoother_from_mapping(None) is None
    assert smoother_from_mapping(BLOCK) == CFG
    with pytest.raises(ValueError, match="missing"):
        smoother_from_mapping({k: v for k, v in BLOCK.items() if k != "max_log_shift"})
    with pytest.raises(ValueError, match="unknown"):
        smoother_from_mapping({**BLOCK, "lam": 1.0})
    for bad in ({"max_log_shift": 0.0}, {"touch_gate_mm": float("nan")}, {"min_gated_frames": 0},
                {"min_gated_frames": 1.5}, {"patch_k": 0}, {"contact_eps_mm": -1.0}, {"hands": "both"}, {"grid_mm": (0.0, 300.0, 2.0)}, {"q_floor": True}):
        with pytest.raises(ValueError):
            ContactRaySmootherConfig(**{**dataclasses.asdict(CFG), **bad})
    staged = SolveStageConfig.from_mapping({"contact_ray_smoother": BLOCK})
    assert staged.contact_ray_smoother == CFG


# --- RTS --------------------------------------------------------------------------------------------------------

def test_rts_equals_the_dense_gaussian_posterior() -> None:
    rng = np.random.default_rng(3)
    n, q, p0 = 40, 2e-4, 0.04
    y = np.full(n, np.nan)
    meas = rng.choice(n, 12, replace=False)
    y[meas] = rng.normal(0.1, 0.05, size=12)
    r = np.full(n, np.inf)
    r[meas] = rng.uniform(1e-4, 4e-3, size=12)
    xs, ps = rts_random_walk(y, r, q, p0)
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    prior = p0 + q * np.minimum(i, j)                              # cov of a random walk started at N(0, p0)
    h = np.eye(n)[np.sort(meas)]
    rr = np.diag(r[np.sort(meas)])
    gain = prior @ h.T @ np.linalg.inv(h @ prior @ h.T + rr)
    mean = gain @ y[np.sort(meas)]
    cov = prior - gain @ h @ prior
    np.testing.assert_allclose(xs, mean, rtol=0, atol=1e-10)
    np.testing.assert_allclose(ps, np.diag(cov), rtol=0, atol=1e-10)


def test_rts_propagates_to_unmeasured_start_rows() -> None:
    y = np.full(20, np.nan)
    y[10:] = 0.15
    xs, _ = rts_random_walk(y, np.full(20, 1e-6), 1e-5, 0.04)
    assert np.allclose(xs, 0.15, atol=2e-3)                       # rows 0-9 take the grasp rows' evidence


# --- measurement ------------------------------------------------------------------------------------------------

def test_cmin_is_the_interaction_modules_cmin() -> None:
    poses, mesh, joints, _, _verts = _scene(n=8, log_err=0.12)
    valid = np.ones(8, bool)
    cmin, dstar, rad = contact_depth_measurements(poses, 1.0, mesh, joints, valid, CFG.grid())
    np.testing.assert_allclose(cmin, frame_min_contact(poses, 1.0, mesh, joints, valid), rtol=0, atol=1e-9)
    true_r = rad / np.exp(0.12)
    np.testing.assert_allclose(rad + dstar / 1e3, true_r, atol=0.004)   # the argmin finds the true depth (2 mm grid)


# --- known answer, invariance, cap, inactive --------------------------------------------------------------------

def test_known_answer_recovers_true_depth_including_rows_outside_the_gate() -> None:
    n = 30
    touch = np.arange(n) >= 8                                        # the grasp starts at row 8
    err = 0.12 + 0.02 * np.sin(np.arange(n) / 6.0)
    poses, mesh, joints, true_c, verts = _scene(n=n, log_err=err, touch=touch)
    res = smooth_contact_ray(poses, 1.0, mesh, joints, np.ones(n, bool), CFG, verts)
    assert res.active and res.record["n_gated"] == int(touch.sum())
    cen = res.poses[:, :3, 3]
    err_mm = np.linalg.norm(cen - true_c, axis=1) * 1e3
    before = np.linalg.norm(poses[:, :3, 3] - true_c, axis=1) * 1e3
    assert np.median(before) > 50.0
    assert np.max(err_mm[touch]) < 8.0                                # within the 5 mm contact tolerance + grid
    assert np.max(err_mm[:8]) < 15.0                                  # propagated to frame 0, outside the gate


def test_centroid_projection_is_exact_rotation_and_scale_untouched() -> None:
    poses, mesh, joints, _, verts = _scene(n=12, log_err=0.1)
    scale = 1.3
    res = smooth_contact_ray(poses, scale, mesh / scale, joints, np.ones(12, bool), CFG, verts)
    assert res.active
    np.testing.assert_array_equal(res.poses[:, :3, :3], poses[:, :3, :3])
    np.testing.assert_array_equal(res.poses[:, 3], poses[:, 3])
    c0 = scale * np.einsum("tij,j->ti", poses[:, :3, :3], (mesh / scale).mean(0)) + poses[:, :3, 3]
    c1 = scale * np.einsum("tij,j->ti", res.poses[:, :3, :3], (mesh / scale).mean(0)) + res.poses[:, :3, 3]
    np.testing.assert_allclose(_project(c1), _project(c0), rtol=0, atol=1e-14)
    np.testing.assert_allclose(np.log(np.linalg.norm(c1, axis=1) / np.linalg.norm(c0, axis=1)), res.log_shift,
                               atol=1e-12)
    # a fixed-size object is NOT on the gauge orbit: the silhouette scales by exp(-shift) (first order)
    p0 = _project(scale * np.einsum("tij,mj->tmi", poses[:, :3, :3], mesh / scale) + poses[:, None, :3, 3])
    p1 = _project(scale * np.einsum("tij,mj->tmi", res.poses[:, :3, :3], mesh / scale) + res.poses[:, None, :3, 3])
    spread = lambda p: np.linalg.norm(p - p.mean(axis=1, keepdims=True), axis=2).mean(axis=1)
    np.testing.assert_allclose(spread(p1) / spread(p0), np.exp(-res.log_shift), rtol=0.02)


def test_a_hand_already_touching_leaves_the_object_where_it_is() -> None:
    poses, mesh, joints, _, verts = _scene(n=12, log_err=0.0)
    cmin, delta, _ = contact_depth_measurements(poses, 1.0, mesh, joints, np.ones(12, bool), CFG.grid(), verts,
                                                CFG.touch_gate_mm, CFG.patch_k, CFG.contact_eps_mm)
    assert np.all(cmin <= CFG.touch_gate_mm) and np.all(delta == 0.0)
    res = smooth_contact_ray(poses, 1.0, mesh, joints, np.ones(12, bool), CFG, verts)
    assert res.active and np.all(np.abs(res.log_shift) < 1e-12)
    np.testing.assert_allclose(res.poses, poses, rtol=0, atol=1e-12)
    # 3 mm short of exact contact is still contact (contact_eps_mm 5): the smallest sufficient move is 0, not the argmin
    poses, mesh, joints, _, verts = _scene(n=6, log_err=0.006)
    _, delta, _ = contact_depth_measurements(poses, 1.0, mesh, joints, np.ones(6, bool), CFG.grid(), verts,
                                             CFG.touch_gate_mm, CFG.patch_k, CFG.contact_eps_mm)
    assert np.all(delta == 0.0)


def test_the_joint_mean_argmin_pulls_the_object_into_the_palm() -> None:
    """Why the measurement is the contact patch: a pinch at the object's near edge with the rest of the hand in front of
    it. The patch already touches (delta 0); the 21-joint mean's argmin moves the object toward the joints."""
    poses, mesh, joints, _, verts = _scene(n=4, log_err=0.0)
    toward_cam = -poses[:, None, :3, 3] / np.linalg.norm(poses[:, None, :3, 3], axis=2, keepdims=True)
    joints = joints.copy()
    joints[:, 5:] += 0.06 * toward_cam                              # 16 of 21 joints 6 cm in front of the surface
    _, d_joint, _ = contact_depth_measurements(poses, 1.0, mesh, joints, np.ones(4, bool), CFG.grid())
    _, d_patch, _ = contact_depth_measurements(poses, 1.0, mesh, joints, np.ones(4, bool), CFG.grid(), verts,
                                               CFG.touch_gate_mm, CFG.patch_k, CFG.contact_eps_mm)
    assert np.all(d_patch == 0.0) and np.all(d_joint < -20.0)


def test_cap_bounds_every_row() -> None:
    poses, mesh, joints, _, verts = _scene(n=15, log_err=0.45)              # the hand says 0.45 nearer than solved
    cfg = dataclasses.replace(CFG, max_log_shift=0.2)
    res = smooth_contact_ray(poses, 1.0, mesh, joints, np.ones(15, bool), cfg, verts)
    assert res.active and res.record["n_capped"] > 0
    assert np.max(np.abs(res.log_shift)) <= 0.2 + 1e-15
    assert np.allclose(res.log_shift, -0.2)


def test_no_gated_rows_returns_the_same_array() -> None:
    poses, mesh, joints, _, verts = _scene(n=10, log_err=0.1, touch=np.zeros(10, bool))
    res = smooth_contact_ray(poses, 1.0, mesh, joints, np.ones(10, bool), CFG, verts)
    assert not res.active and res.poses is poses and res.record["n_gated"] == 0
    few = np.zeros(10, bool)
    few[:2] = True
    poses, mesh, joints, _, verts = _scene(n=10, log_err=0.1, touch=few)
    res = smooth_contact_ray(poses, 1.0, mesh, joints, np.ones(10, bool), CFG, verts)
    assert not res.active and res.poses is poses and res.record["n_gated"] == 2


def test_grid_edge_argmin_is_not_a_measurement() -> None:
    poses, mesh, joints, _, verts = _scene(n=6, log_err=float(np.log(0.83 / 0.5)))   # true depth ~330 mm nearer
    cmin, dstar, _ = contact_depth_measurements(poses, 1.0, mesh, joints, np.ones(6, bool), CFG.grid())
    assert np.all(dstar == -300.0) and np.all(cmin <= CFG.touch_gate_mm)       # the gate alone would pass them
    res = smooth_contact_ray(poses, 1.0, mesh, joints, np.ones(6, bool), CFG, verts)
    assert res.record["n_gated"] == 0 and res.poses is poses


# --- the solve stage --------------------------------------------------------------------------------------------

def _touching_hands(d: Path, side: str = "right") -> None:
    """Rewrite hands.npz so `side`'s hand cups the FAR side of the solved object 8 % of its distance farther along each
    row's ray (joints and a 40-vertex patch there, the other vertices 0.5 m away), and the prompt chooses `side`."""
    solve, mesh = load_npz(d / "solve.npz"), load_npz(d / "samples.npz")["points"].astype(float)
    hands = dict(load_npz(d / "hands.npz"))
    poses, scale = solve["poses"].astype(float), float(solve["scale"])
    rng = np.random.default_rng(0)
    joints = np.zeros((poses.shape[0], 21, 3), dtype=hands[f"{side}_joints"].dtype)
    verts = np.zeros((poses.shape[0], 778, 3), dtype=hands[f"{side}_vertices"].dtype)
    for t in range(poses.shape[0]):
        surf = scale * (mesh @ poses[t, :3, :3].T) + poses[t, :3, 3]
        c = scale * (poses[t, :3, :3] @ mesh.mean(0)) + poses[t, :3, 3]
        far = np.argsort(surf @ (c / np.linalg.norm(c)))[-60:]      # the far side of the object
        joints[t] = surf[rng.choice(far, 21, replace=False)] + 0.08 * c
        verts[t] = np.r_[surf[rng.choice(far, 40, replace=False)] + 0.08 * c,
                         np.tile(c + np.array([0.5, 0.0, 0.0]), (738, 1))]
    hands[f"{side}_joints"] = joints
    hands[f"{side}_vertices"] = verts
    hands[f"{side}_valid"] = np.ones(poses.shape[0], bool)
    save_npz(d / "hands.npz", **hands)
    (d / "prompt.json").write_text(json.dumps({"choice": {"side": side}}))


def test_solve_stage_inactive_is_byte_identical_and_active_moves_only_poses(tmp_path: Path) -> None:
    clip = make_synthetic_clip(SMALL_CLIP, seed=0)
    on = dataclasses.replace(SMALL_SOLVE, contact_ray_smoother=dataclasses.replace(CFG, min_gated_frames=1))
    off_dir, on_dir = tmp_path / "off" / "c", tmp_path / "on" / "c"
    for d in (off_dir, on_dir):
        write_synthetic_solve_inputs(d, clip)
    make_solve_worker(SMALL_SOLVE, RecordingSolve())("c", off_dir)
    make_solve_worker(on, RecordingSolve())("c", on_dir)
    block = json.loads((on_dir / "solve.json").read_text())["contact_ray_smoother"]
    assert block["active"] is False and block["n_gated"] == 0       # the fake hands never touch
    assert (off_dir / "solve.npz").read_bytes() == (on_dir / "solve.npz").read_bytes()
    assert "contact_ray_smoother" not in json.loads((off_dir / "solve.json").read_text())
    for d in (off_dir, on_dir):
        _touching_hands(d)
    make_solve_worker(SMALL_SOLVE, RecordingSolve())("c", off_dir)
    make_solve_worker(on, RecordingSolve())("c", on_dir)
    block = json.loads((on_dir / "solve.json").read_text())["contact_ray_smoother"]
    assert block["active"] is True and block["side"] == "right" and block["side_source"] == "prompt"
    a, b = load_npz(off_dir / "solve.npz"), load_npz(on_dir / "solve.npz")
    assert a.keys() == b.keys()
    for k in a:
        if k != "poses":
            assert a[k].tobytes() == b[k].tobytes(), k              # track_weight stays the solver's
    assert a["poses"].dtype == b["poses"].dtype
    np.testing.assert_array_equal(a["poses"][:, :3, :3], b["poses"][:, :3, :3])
    assert np.allclose(block["log_shift"], np.log(1.08), atol=0.02)
