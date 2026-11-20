"""Per-clip grasp-onset along-ray constant (`fastdo.engine.grasp_ray_constant`), CPU only."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from fastdo.engine.contact_ray_smoother import ContactRaySmootherConfig
from fastdo.engine.grasp_ray_constant import (
    GraspRayConstantConfig,
    apply_log_constant,
    grasp_constant_from_mapping,
    grasp_ray_constant,
    onset_log_measurements,
)
from fastdo.engine.worker_support import load_npz, save_npz
from fastdo.engine.workers_solve import SolveStageConfig, make_solve_worker
from fastdo.solver.synthetic import make_synthetic_clip
from tests.engine.fakes_solve import (
    SMALL_CLIP,
    SMALL_SOLVE,
    RecordingSolve,
    write_synthetic_solve_inputs,
)

_ROOT_CONFIG = Path(__file__).resolve().parents[2] / "run" / "conf" / "config.yaml"
GAUGE = GraspRayConstantConfig(mode="gauge")
SHIFT = GraspRayConstantConfig(mode="shift")
BLOCK = {"mode": "gauge", "touch_gate_mm": 56.0, "onset_rows": 10, "min_gated_frames": 3, "max_log_shift": 0.2,
         "patch_k": 20, "contact_eps_mm": 5.0, "hands": "prompt", "grid_log": [-0.3, 0.3, 0.004]}


# --- a synthetic scene with known depth -------------------------------------------------------------------------

def _sphere(n: int = 600, radius: float = 0.04, seed: int = 0) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal((n, 3))
    return radius * v / np.linalg.norm(v, axis=1, keepdims=True)


def _rot(seed: int) -> np.ndarray:
    q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((3, 3)))
    return q * np.sign(np.linalg.det(q))


def _scene(n: int = 30, log_err: float = 0.1, touch: np.ndarray | None = None, gauge: bool = True,
           hand_log: np.ndarray | None = None, seed: int = 0):
    """The TRUE object (a 4 cm ball, unit scale) moves on a path ~0.5 m away. The SOLVED clip is off by exp(log_err):
    with `gauge` its scale AND every translation are multiplied (the image-exact gauge), else only the translations
    (a fixed-size object along the rays). Touching rows: 21 joints and a 30-vertex patch on the true surface's
    camera-facing cap (the other 170 vertices 15-60 mm outside it), the hand optionally displaced along its ray by
    exp(hand_log[t]); other rows: the hand 0.25 m to the side. Returns (poses, scale, mesh, joints, verts, true_c)."""
    mesh = _sphere(seed=seed)
    t = np.arange(n)
    true_c = np.stack([0.05 * np.sin(t / 7.0), 0.02 * np.cos(t / 5.0), 0.5 + 0.03 * np.sin(t / 9.0)], axis=1)
    lam = float(np.exp(log_err))
    hand_log = np.zeros(n) if hand_log is None else np.asarray(hand_log, float)
    touch = np.ones(n, bool) if touch is None else touch
    poses = np.tile(np.eye(4), (n, 1, 1))
    joints, verts = np.zeros((n, 21, 3)), np.zeros((n, 200, 3))
    rng = np.random.default_rng(seed + 1)
    for i in range(n):
        rot = _rot(seed + 10 + i)
        poses[i, :3, :3] = rot
        poses[i, :3, 3] = lam * true_c[i]
        surf = mesh @ rot.T
        nrm = surf / np.linalg.norm(surf, axis=1, keepdims=True)
        cap = np.flatnonzero(nrm @ (-true_c[i] / np.linalg.norm(true_c[i])) > 0.6)
        j_idx, v_idx = rng.choice(cap, 21, replace=False), rng.choice(cap, 200, replace=True)
        out = np.r_[np.zeros(30), rng.uniform(0.015, 0.06, 170)][:, None]
        hand_j, hand_v = surf[j_idx] + true_c[i], surf[v_idx] + out * nrm[v_idx] + true_c[i]
        hand_j, hand_v = np.exp(hand_log[i]) * hand_j, np.exp(hand_log[i]) * hand_v
        off = np.zeros(3) if touch[i] else np.array([0.25, 0.0, 0.0])
        joints[i], verts[i] = hand_j + off, hand_v + off
    return poses, (lam if gauge else 1.0), mesh, joints, verts, true_c


def _points(poses: np.ndarray, scale: float, mesh: np.ndarray) -> np.ndarray:
    return scale * np.einsum("tij,mj->tmi", poses[:, :3, :3], mesh) + poses[:, None, :3, 3]


def _project(points: np.ndarray) -> np.ndarray:
    return points[..., :2] / points[..., 2:3]


def _normals(mesh: np.ndarray) -> np.ndarray:
    return mesh / np.linalg.norm(mesh, axis=1, keepdims=True)        # the ball's outward normals


def _run(cfg, poses, scale, mesh, joints, verts, valid=None):
    valid = np.ones(poses.shape[0], bool) if valid is None else valid
    return grasp_ray_constant(poses, scale, mesh, _normals(mesh), joints, valid, cfg, verts)


# --- config -----------------------------------------------------------------------------------------------------

def test_config_off_by_default_and_blocks_validate() -> None:
    root = OmegaConf.load(_ROOT_CONFIG)
    assert root.solve_stage.grasp_ray_constant is None
    assert SolveStageConfig.from_mapping(root.solve_stage).grasp_ray_constant is None
    assert SolveStageConfig().grasp_ray_constant is None
    assert grasp_constant_from_mapping(None) is None
    assert grasp_constant_from_mapping(BLOCK) == GAUGE
    assert grasp_constant_from_mapping({**BLOCK, "mode": "shift"}) == SHIFT
    with pytest.raises(ValueError, match="missing"):
        grasp_constant_from_mapping({k: v for k, v in BLOCK.items() if k != "onset_rows"})
    with pytest.raises(ValueError, match="unknown"):
        grasp_constant_from_mapping({**BLOCK, "lam": 1.0})
    for bad in ({"mode": "both"}, {"max_log_shift": 0.0}, {"touch_gate_mm": float("nan")}, {"onset_rows": 0},
                {"min_gated_frames": 1.5}, {"patch_k": 0}, {"contact_eps_mm": -1.0}, {"hands": "both"},
                {"grid_log": (0.0, 0.3, 0.004)}, {"grid_log": (-0.3, 0.3, 0.0)}, {"max_log_shift": True},
                {"max_log_shift": 0.5}):
        with pytest.raises(ValueError):
            GraspRayConstantConfig(**{**dataclasses.asdict(GAUGE), **bad})
    assert SolveStageConfig.from_mapping({"grasp_ray_constant": BLOCK}).grasp_ray_constant == GAUGE
    smoother = dataclasses.asdict(ContactRaySmootherConfig())
    with pytest.raises(ValueError, match="at most one"):
        SolveStageConfig.from_mapping({"grasp_ray_constant": BLOCK, "contact_ray_smoother": smoother})


# --- known answers ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("log_err", [0.12, -0.09])
def test_known_answer_gauge_recovers_true_depth_and_size_on_every_row(log_err: float) -> None:
    n = 30
    touch = np.arange(n) >= 8                                         # the grasp starts at row 8
    poses, scale, mesh, joints, verts, true_c = _scene(n=n, log_err=log_err, touch=touch, gauge=True)
    res = _run(GAUGE, poses, scale, mesh, joints, verts)
    assert res.active and res.record["n_gated"] == int(touch.sum()) and res.record["onset_rows"][0] == 8
    assert abs(res.log_constant + log_err) < 0.015                   # 5 mm contact band at 0.5 m (0.01) + grid
    assert abs(np.log(res.scale)) < 0.015                            # the true size is back
    err_mm = np.linalg.norm(res.poses[:, :3, 3] - true_c, axis=1) * 1e3
    before = np.linalg.norm(poses[:, :3, 3] - true_c, axis=1) * 1e3
    assert np.median(before) > 40.0 and np.max(err_mm) < 8.0          # every row, including the 8 before the grasp


def test_known_answer_shift_recovers_true_depth_at_fixed_size() -> None:
    n = 30
    touch = np.arange(n) >= 5
    poses, scale, mesh, joints, verts, true_c = _scene(n=n, log_err=0.12, touch=touch, gauge=False)
    res = _run(SHIFT, poses, scale, mesh, joints, verts)
    assert res.active and abs(res.log_constant + 0.12) < 0.015
    assert res.scale == scale
    np.testing.assert_array_equal(res.poses[:, :3, :3], poses[:, :3, :3])
    err_mm = np.linalg.norm(res.poses[:, :3, 3] - true_c, axis=1) * 1e3
    assert np.max(err_mm) < 8.0


def test_a_known_extra_gauge_moves_the_estimate_by_exactly_that_amount() -> None:
    """The KA the study runs on real solves (KA-G): compose a known gauge exp(k) with the solve; the estimate moves by
    -k (up to the grid), in either direction, including when the solve already touches."""
    for log_err in (0.05, 0.0):
        poses, scale, mesh, joints, verts, _ = _scene(n=20, log_err=log_err, gauge=True)
        base = _run(GAUGE, poses, scale, mesh, joints, verts).log_constant
        for k in (-0.1, -0.03, 0.1):
            p2, s2 = apply_log_constant(poses, scale, mesh, k, "gauge")
            est = _run(GAUGE, p2, s2, mesh, joints, verts).log_constant
            assert abs((est - base) + k) <= 0.0081                     # two grid steps


def test_a_touching_band_is_measured_at_its_middle_not_its_edge() -> None:
    """A hand whose contact vertices lie on a surface PARALLEL to the ray touches over a band of depths. The
    maximum-contact point is the band's middle, so a known move of the object moves the estimate by exactly -k."""
    n, half = 12, 0.03                                                 # a 6 cm tall side wall along the ray
    rng = np.random.default_rng(0)
    wall = np.stack([np.full(400, 0.04), rng.uniform(-0.03, 0.03, 400), rng.uniform(-half, half, 400)], axis=1)
    mesh = np.concatenate([wall, wall * [-1, 1, 1]])                  # two walls, outward normals +-x
    normals = np.zeros_like(mesh)
    normals[:400, 0], normals[400:, 0] = 1.0, -1.0
    poses = np.tile(np.eye(4), (n, 1, 1))
    poses[:, 2, 3] = 0.5
    finger = np.stack([np.full(60, 0.04), rng.uniform(-0.02, 0.02, 60), rng.uniform(-0.01, 0.01, 60)], axis=1)
    verts = np.concatenate([finger + [0, 0, 0.5], np.tile([0.3, 0.0, 0.5], (140, 1))])[None].repeat(n, 0)
    joints = verts[:, :21]
    res = grasp_ray_constant(poses, 1.0, mesh, normals, joints, np.ones(n, bool), SHIFT, verts)
    assert abs(res.log_constant) <= 0.0041                            # the band is centred on the solved depth
    for k in (-0.04, 0.04):
        p2, _ = apply_log_constant(poses, 1.0, mesh, k, "shift")
        est = grasp_ray_constant(p2, 1.0, mesh, normals, joints, np.ones(n, bool), SHIFT, verts).log_constant
        assert abs(est - res.log_constant + k) <= 0.0081


# --- what the constant can and cannot change -------------------------------------------------------------------

def test_gauge_preserves_every_projection_exactly() -> None:
    poses, scale, mesh, joints, verts, _ = _scene(n=12, log_err=0.1, gauge=True)
    res = _run(GAUGE, poses, scale, mesh, joints, verts)
    assert res.active
    np.testing.assert_array_equal(res.poses[:, :3, :3], poses[:, :3, :3])
    np.testing.assert_array_equal(res.poses[:, 3], poses[:, 3])
    lam = np.exp(res.log_constant)
    np.testing.assert_allclose(res.poses[:, :3, 3], lam * poses[:, :3, 3], rtol=1e-15, atol=0)
    assert res.scale == pytest.approx(lam * scale, rel=1e-15)
    np.testing.assert_allclose(_project(_points(res.poses, res.scale, mesh)), _project(_points(poses, scale, mesh)),
                               rtol=0, atol=1e-12)


@pytest.mark.parametrize("cfg", [GAUGE, SHIFT])
def test_one_scalar_per_clip_cannot_cancel_motion(cfg: GraspRayConstantConfig) -> None:
    """Every row's centroid moves by the SAME log factor along its ray, so the camera-frame displacement between any
    two rows is the solved one times exp(c): relative motion is scaled, never cancelled (the contact-gated smoother's
    failure mode)."""
    poses, scale, mesh, joints, verts, _ = _scene(n=16, log_err=0.1, gauge=cfg.mode == "gauge")
    res = _run(cfg, poses, scale, mesh, joints, verts)
    cen = lambda p, s: s * np.einsum("tij,j->ti", p[:, :3, :3], mesh.mean(0)) + p[:, :3, 3]
    c0, c1 = cen(poses, scale), cen(res.poses, res.scale)
    lam = np.exp(res.log_constant)
    np.testing.assert_allclose(c1, lam * c0, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(c1[1:] - c1[0], lam * (c0[1:] - c0[0]), rtol=1e-10, atol=1e-15)


def test_only_the_onset_rows_set_the_constant() -> None:
    n = 30
    hand_log = np.where(np.arange(n) < 10, 0.0, 0.15)                 # after the onset the hand drifts 0.15 farther
    poses, scale, mesh, joints, verts, _ = _scene(n=n, log_err=0.1, gauge=True, hand_log=hand_log)
    res = _run(GAUGE, poses, scale, mesh, joints, verts)
    assert res.record["onset_rows"] == list(range(10))
    assert abs(res.log_constant + 0.1) < 0.015


def test_a_hand_already_touching_is_a_fixed_point() -> None:
    poses, scale, mesh, joints, verts, _ = _scene(n=12, log_err=0.0)
    res = _run(GAUGE, poses, scale, mesh, joints, verts)
    assert abs(res.log_constant) <= 0.0041                            # one grid step
    np.testing.assert_allclose(res.poses[:, :3, 3], poses[:, :3, 3], rtol=0.0042, atol=0)
    assert res.scale == pytest.approx(scale, rel=0.0042)


def test_cap_bounds_the_constant() -> None:
    poses, scale, mesh, joints, verts, _ = _scene(n=12, log_err=0.26, gauge=True)
    res = _run(GAUGE, poses, scale, mesh, joints, verts)
    assert res.active and res.record["capped"] and res.log_constant == -0.2


def test_too_few_gated_rows_returns_the_same_objects() -> None:
    poses, scale, mesh, joints, verts, _ = _scene(n=10, log_err=0.1, touch=np.zeros(10, bool))
    res = _run(GAUGE, poses, scale, mesh, joints, verts)
    assert not res.active and res.poses is poses and res.scale == scale and res.record["n_gated"] == 0
    few = np.zeros(10, bool)
    few[:2] = True
    poses, scale, mesh, joints, verts, _ = _scene(n=10, log_err=0.1, touch=few)
    res = _run(GAUGE, poses, scale, mesh, joints, verts)
    assert not res.active and res.poses is poses and res.record["n_gated"] == 2


def test_grid_edge_argmin_is_not_a_measurement() -> None:
    poses, scale, mesh, joints, verts, _ = _scene(n=6, log_err=0.45, gauge=True)   # outside the +-0.3 grid
    cmin, ell, gated = onset_log_measurements(poses, scale, mesh, _normals(mesh), joints, np.ones(6, bool), verts,
                                              GAUGE)
    assert np.all(cmin <= GAUGE.touch_gate_mm)                        # the gate alone would pass them
    assert not gated.any() and np.all(np.isnan(ell))
    res = _run(GAUGE, poses, scale, mesh, joints, verts)
    assert not res.active and res.poses is poses


# --- the solve stage --------------------------------------------------------------------------------------------

LOG_TRUE = -float(np.log(1.08))


def _touching_hands(d: Path, side: str = "right", log_true: float = LOG_TRUE) -> None:
    """Rewrite hands.npz so `side`'s hand touches the camera-facing side of the solved surface scaled about the camera
    by exp(log_true) (the gauge image of the object; < 0: the hand hovers in front of the solved object), and the
    prompt chooses `side`."""
    solve, mesh = load_npz(d / "solve.npz"), load_npz(d / "samples.npz")["points"].astype(float)
    hands = dict(load_npz(d / "hands.npz"))
    poses, scale = solve["poses"].astype(float), float(solve["scale"])
    rng = np.random.default_rng(0)
    joints = np.zeros((poses.shape[0], 21, 3), dtype=hands[f"{side}_joints"].dtype)
    verts = np.zeros((poses.shape[0], 778, 3), dtype=hands[f"{side}_vertices"].dtype)
    for t in range(poses.shape[0]):
        surf = np.exp(log_true) * (scale * (mesh @ poses[t, :3, :3].T) + poses[t, :3, 3])
        c = surf.mean(0)
        near = np.argsort(surf @ (c / np.linalg.norm(c)))[:60]       # the camera-facing side
        joints[t] = surf[rng.choice(near, 21, replace=False)]
        verts[t] = np.r_[surf[rng.choice(near, 40, replace=False)], np.tile(c + np.array([0.5, 0.0, 0.0]), (738, 1))]
    hands[f"{side}_joints"] = joints
    hands[f"{side}_vertices"] = verts
    hands[f"{side}_valid"] = np.ones(poses.shape[0], bool)
    save_npz(d / "hands.npz", **hands)
    (d / "prompt.json").write_text(json.dumps({"choice": {"side": side}}))


def test_solve_stage_inactive_is_byte_identical_and_active_applies_the_gauge(tmp_path: Path) -> None:
    clip = make_synthetic_clip(SMALL_CLIP, seed=0)
    on = dataclasses.replace(SMALL_SOLVE, grasp_ray_constant=dataclasses.replace(GAUGE, min_gated_frames=1))
    off_dir, on_dir = tmp_path / "off" / "c", tmp_path / "on" / "c"
    for d in (off_dir, on_dir):
        write_synthetic_solve_inputs(d, clip)
    make_solve_worker(SMALL_SOLVE, RecordingSolve())("c", off_dir)
    make_solve_worker(on, RecordingSolve())("c", on_dir)
    block = json.loads((on_dir / "solve.json").read_text())["grasp_ray_constant"]
    assert block["active"] is False and block["n_gated"] == 0       # the fake hands never touch
    assert (off_dir / "solve.npz").read_bytes() == (on_dir / "solve.npz").read_bytes()
    assert "grasp_ray_constant" not in json.loads((off_dir / "solve.json").read_text())
    for d in (off_dir, on_dir):
        _touching_hands(d)
    make_solve_worker(SMALL_SOLVE, RecordingSolve())("c", off_dir)
    make_solve_worker(on, RecordingSolve())("c", on_dir)
    meta = json.loads((on_dir / "solve.json").read_text())
    block = meta["grasp_ray_constant"]
    assert block["active"] is True and block["side"] == "right" and block["side_source"] == "prompt"
    lam = float(np.exp(block["log_constant"]))
    assert abs(block["log_constant"] - LOG_TRUE) < 0.02             # 5 mm band at the clip's ~0.6 m depth + grid
    a, b = load_npz(off_dir / "solve.npz"), load_npz(on_dir / "solve.npz")
    assert a.keys() == b.keys()
    for k in a:
        if k not in ("poses", "scale", "covariance"):
            assert a[k].tobytes() == b[k].tobytes(), k              # track_weight stays the solver's
    assert a["poses"].dtype == b["poses"].dtype
    np.testing.assert_array_equal(a["poses"][:, :3, :3], b["poses"][:, :3, :3])
    np.testing.assert_allclose(b["poses"][:, :3, 3], lam * a["poses"][:, :3, 3], rtol=1e-6)
    assert float(b["scale"]) == pytest.approx(lam * float(a["scale"]), rel=1e-12)
    assert meta["solver"]["scale"] == pytest.approx(float(b["scale"]), rel=1e-12)
    np.testing.assert_allclose(b["covariance"][:, :3, :3], lam ** 2 * a["covariance"][:, :3, :3], rtol=1e-5)
    np.testing.assert_allclose(b["covariance"][:, 3:, 3:], a["covariance"][:, 3:, 3:], rtol=0, atol=0)
