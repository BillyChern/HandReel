"""World-frame object motion features and the opt-in qc world-motion screen."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from fastdo.engine.workers_qc import QcStageConfig, make_qc_worker
from fastdo.geometry.ego_motion import EgoMotionConfig, save_ego_motion, sim3_matrix, so3_exp, solve_ego_motion
from fastdo.qc.world_motion import (
    WorldMotionScreenConfig,
    world_motion_features,
    world_screen_decision,
)
from tests.engine.fakes_qc import write_solved_clip
from tests.qc.test_interaction import _ROOT_CONFIG, _passing, _rule


def _camera(t: int) -> np.ndarray:
    """cam_to_world: the head turns 2 deg/row and walks 3 cm/row, with a 2% per-row metric-scale wobble."""
    out = []
    for k in range(t):
        s = 1.0 + 0.02 * math.sin(k)
        out.append(sim3_matrix(s, so3_exp(np.array([0.0, math.radians(2.0 * k), 0.0])), np.array([0.03 * k, 0, 0])))
    return np.stack(out)


def _object_in_camera(c2w: np.ndarray, world_pos: np.ndarray, mesh_centroid: np.ndarray, scale: float) -> np.ndarray:
    """Object poses (camera frame) that put the object's centroid at `world_pos[t]` (identity world orientation)."""
    poses = np.tile(np.eye(4), (len(c2w), 1, 1))
    for t, m in enumerate(c2w):
        inv = np.linalg.inv(m)
        r = inv[:3, :3] / np.cbrt(np.linalg.det(inv[:3, :3]))
        poses[t, :3, :3] = r
        cam_c = inv[:3, :3] @ world_pos[t] + inv[:3, 3]
        poses[t, :3, 3] = cam_c - scale * r @ mesh_centroid
    return poses


def test_a_static_object_is_static_in_the_world_and_moves_in_the_camera() -> None:
    t = 30
    c2w = _camera(t)
    mesh = np.random.default_rng(0).uniform(-0.05, 0.05, (50, 3)) + [0.0, 0.0, 0.01]
    world = np.tile([0.2, 0.1, 1.5], (t, 1))
    poses = _object_in_camera(c2w, world, mesh.mean(0), 1.0)
    feats, per = world_motion_features(poses, 1.0, mesh, np.ones(t), c2w, np.ones(t, bool), True)
    assert feats.obj_world_disp_mm < 1e-6 and feats.obj_world_rot_deg < 1e-3
    assert feats.obj_cam_disp_mm > 100 and feats.obj_cam_rot_deg > 20
    assert feats.obj_moving_frac == 0.0 and feats.used_frac == 1.0 and feats.cam_path_mm > 800


def test_a_carried_object_moves_in_the_world_with_the_hand() -> None:
    t = 30
    c2w = _camera(t)
    mesh = np.random.default_rng(1).uniform(-0.05, 0.05, (50, 3))
    world = np.stack([np.array([0.2 + 0.01 * k, 0.1, 1.5]) for k in range(t)])      # 29 cm over the window
    poses = _object_in_camera(c2w, world, mesh.mean(0), 1.0)
    hand_w = world + [0.0, 0.05, 0.0]
    joints = np.stack([(np.linalg.inv(c2w[k]) @ np.r_[hand_w[k], 1.0])[:3] for k in range(t)])[:, None, :]
    conf = np.ones(t)
    conf[:3] = 0.0                                                                   # unobserved rows are not used
    feats, _ = world_motion_features(poses, 1.0, mesh, conf, c2w, np.ones(t, bool), True,
                                     {"right": np.repeat(joints, 21, 1)}, {"right": np.ones(t, bool)})
    assert feats.used_frac == pytest.approx(27 / 30)
    assert 100 < feats.obj_world_disp_mm < 150 and feats.obj_moving_frac == 1.0
    assert feats.comotion == pytest.approx(1.0)
    assert feats.hand_world_disp_mm == pytest.approx(feats.obj_world_disp_mm, rel=0.2)


def test_screen_decision_abstains_on_bad_ego_and_few_rows() -> None:
    t = 10
    c2w = _camera(t)
    mesh = np.zeros((4, 3)) + np.eye(4, 3) * 0.05
    poses = _object_in_camera(c2w, np.tile([0.0, 0.0, 1.0], (t, 1)), mesh.mean(0), 1.0)
    cfg = WorldMotionScreenConfig(enabled=True, min_value=20.0)
    feats, _ = world_motion_features(poses, 1.0, mesh, np.ones(t), c2w, np.ones(t, bool), True)
    assert world_screen_decision(feats, cfg) == (False, "fail")
    bad, _ = world_motion_features(poses, 1.0, mesh, np.ones(t), c2w, np.ones(t, bool), False)
    assert world_screen_decision(bad, cfg) == (True, "abstain_bad_ego")
    few, _ = world_motion_features(poses, 1.0, mesh, np.r_[np.ones(2), np.zeros(8)], c2w, np.ones(t, bool), True)
    assert world_screen_decision(few, cfg) == (True, "abstain_few_rows")
    assert world_screen_decision(feats, WorldMotionScreenConfig(enabled=True, min_value=0.0)) == (True, "pass")


def test_screen_config() -> None:
    with pytest.raises(ValueError, match="min_value is null"):
        WorldMotionScreenConfig(enabled=True)
    with pytest.raises(ValueError, match="unknown keys"):
        WorldMotionScreenConfig.from_mapping({"typo": 1})
    with pytest.raises(ValueError, match="feature"):
        WorldMotionScreenConfig(feature="touch_frac")
    root = OmegaConf.load(_ROOT_CONFIG)
    assert QcStageConfig.from_mapping(root.qc_stage).world_motion_screen == WorldMotionScreenConfig()
    got = QcStageConfig.from_mapping({"world_motion_screen": {"enabled": True, "min_value": 40}}).world_motion_screen
    assert got.enabled and got.min_value == 40.0


# --- the qc stage: off is byte-identical (and reads no ego_motion.npz); on records and can only REJECT -------------


def _write_ego(clip: Path, t: int, ok: bool = True) -> None:
    rows = np.asarray(np.load(clip / "solve.npz")["frame_idx"])
    rng = np.random.default_rng(0)
    world = rng.uniform([-2, -1, 2], [2, 1, 6], (300, 3))
    c2w = _camera(t)
    edges, xs, ys = [], [], []
    for i in range(t - 1):
        edges.append((i, i + 1))
        w = np.c_[world, np.ones(len(world))]
        xs.append((np.linalg.inv(c2w[i]) @ w.T).T[:, :3])
        ys.append((np.linalg.inv(c2w[i + 1]) @ w.T).T[:, :3])
    ego = solve_ego_motion(t, np.array(edges), xs, ys, EgoMotionConfig(skips=(1,), ok_max_loop_deg=1e9))
    summary = dict(ego.summary) | {"ok": ok}          # one-step edges only: no loops, so set the self-check here
    save_ego_motion(clip / "ego_motion.npz", type(ego)(**{**ego.__dict__, "summary": summary}), rows,
                    EgoMotionConfig())


def test_a_disabled_world_screen_is_byte_identical_and_reads_nothing(tmp_path: Path) -> None:
    clip = tmp_path / "clip"
    write_solved_clip(clip, seed=3)
    out = {}
    for arm, screen in (("none", None), ("off", WorldMotionScreenConfig()),
                        ("off_with_value", WorldMotionScreenConfig(enabled=False, min_value=1e6))):
        make_qc_worker(_rule(), features_fn=_passing, world_motion_screen=screen)("c", clip)   # no ego_motion.npz
        out[arm] = ((clip / "qc.json").read_bytes(), (clip / "clip_record.npz").read_bytes())
        (clip / "qc.json").unlink()
        (clip / "clip_record.npz").unlink()
    assert out["none"] == out["off"] == out["off_with_value"]
    assert "world_motion" not in json.loads(out["none"][0])


def test_an_enabled_world_screen_records_rejects_and_abstains(tmp_path: Path) -> None:
    for arm, value, ok, accepted, reason in (("keep", 0.0, True, True, "pass"), ("drop", 1e6, True, False, "fail"),
                                             ("abstain", 1e6, False, True, "abstain_bad_ego")):
        clip = tmp_path / arm
        write_solved_clip(clip, seed=3)
        _write_ego(clip, 6, ok=ok)
        make_qc_worker(_rule(), features_fn=_passing,
                       world_motion_screen=WorldMotionScreenConfig(enabled=True, min_value=value))("c", clip)
        qc = json.loads((clip / "qc.json").read_text())
        assert qc["accepted"] is accepted and qc["world_motion"]["reason"] == reason
        assert qc["rejected_for_no_world_motion"] is (not accepted)
        assert qc["world_motion"]["features"]["used_frac"] == 1.0
        assert (clip / "clip_record.npz").exists() is accepted


def test_an_enabled_world_screen_needs_the_ego_motion(tmp_path: Path) -> None:
    clip = tmp_path / "clip"
    write_solved_clip(clip, seed=3)
    with pytest.raises(FileNotFoundError, match="ego_motion"):
        make_qc_worker(_rule(), features_fn=_passing,
                       world_motion_screen=WorldMotionScreenConfig(enabled=True, min_value=1.0))("c", clip)
