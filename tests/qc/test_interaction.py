"""Window-level interaction features (`fastdo.qc.interaction`) and the opt-in qc screen."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from fastdo.contact import ContactConstants, contact_correction
from fastdo.engine.workers_qc import QcStageConfig, make_qc_worker
from fastdo.qc import AcceptanceRule, ConformalCalibration, QcFeatures
from fastdo.qc.interaction import (
    DEFAULT_TOUCH_GATE_MM,
    InteractionScreenConfig,
    frame_min_contact,
    interaction_features,
    longest_run,
    screen_passes,
)
from tests.engine.fakes_qc import write_solved_clip

_ROOT_CONFIG = Path(__file__).resolve().parents[2] / "run" / "conf" / "config.yaml"
PASSING = QcFeatures(reprojection_iou=0.9, laplace_std_ok_frac=1.0, ot_residual_rms_frac=0.02, penetration_depth_m=0.0,
                     track_inlier_ratio=0.9)
CONST = ContactConstants(a=1.0, b_mm=0.0, mu_mm=0.0, sig_mm=50.0)


def _scene(t: int = 8, seed: int = 0) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    """A unit-ish sphere of surface samples 0.5 m in front of the camera, and hand joints near / far from it."""
    rng = np.random.default_rng(seed)
    pts = rng.normal(size=(400, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)                  # radius 1 in the object frame
    scale = 0.05                                                         # 5 cm radius
    poses = np.tile(np.eye(4), (t, 1, 1))
    poses[:, :3, 3] = [0.0, 0.0, 0.5]
    near = np.array([0.0, 0.0, 0.5]) + np.array([0.06, 0.0, 0.0]) + rng.normal(scale=0.005, size=(t, 21, 3))
    far = near + np.array([0.4, 0.0, 0.0])
    return poses, scale, pts, near, far


def test_frame_min_contact_matches_the_contact_stage_minimum() -> None:
    poses, scale, pts, near, far = _scene()
    joints = np.concatenate([near[:4], far[4:]])
    valid = np.array([True] * 7 + [False])
    stage = contact_correction(poses, scale, pts, joints, valid, CONST, touch_gate_mm=1e9).min_cost_mm
    mine = frame_min_contact(poses, scale, pts, joints, valid)
    assert np.array_equal(np.isnan(stage), np.isnan(mine)) and np.isnan(mine[7])
    assert np.allclose(mine[:7], stage[:7], rtol=0, atol=1e-9)


def test_touching_needs_the_hand_near_the_surface_at_some_depth() -> None:
    poses, scale, pts, near, far = _scene()
    valid = np.ones(8, bool)
    assert (frame_min_contact(poses, scale, pts, near, valid) <= DEFAULT_TOUCH_GATE_MM).all()
    assert (frame_min_contact(poses, scale, pts, far, valid) > DEFAULT_TOUCH_GATE_MM).all()


def test_c_min_is_immune_to_an_along_ray_error_of_the_object() -> None:
    """The slide makes C_min invariant to where along its ray the object sits (to the 2 mm grid)."""
    poses, scale, pts, near, _ = _scene()
    moved = poses.copy()
    moved[:, :3, 3] *= 1.3                                              # 150 mm further along the same ray
    valid = np.ones(8, bool)
    a = frame_min_contact(poses, scale, pts, near, valid)
    b = frame_min_contact(moved, scale, pts, near, valid)
    assert np.abs(a - b).max() <= 2.0


def test_window_features_take_either_hand_and_count_runs() -> None:
    poses, scale, pts, near, far = _scene(t=10)
    left = far.copy()
    right = np.concatenate([far[:3], near[3:8], far[8:]])               # touches rows 3..7 only
    valid = {"left": np.ones(10, bool), "right": np.ones(10, bool)}
    feats, per_row = interaction_features(poses, scale, pts, {"left": left, "right": right}, valid)
    assert feats.touch_frac == pytest.approx(0.5) and feats.longest_touch_run == 5
    assert feats.touch_frac_right == pytest.approx(0.5) and feats.touch_frac_left == 0.0
    assert feats.hand_valid_frac == 1.0 and np.array_equal(per_row["cmin"], np.fmin(per_row["cmin_left"],
                                                                                      per_row["cmin_right"]))
    valid["right"][:] = False                                           # an invalid side cannot touch
    feats2, _ = interaction_features(poses, scale, pts, {"left": left, "right": right}, valid)
    assert feats2.touch_frac == 0.0 and feats2.longest_touch_run == 0


def test_rows_without_a_hand_are_not_touching() -> None:
    poses, scale, pts, near, _ = _scene()
    none = {"left": np.zeros(8, bool), "right": np.zeros(8, bool)}
    feats, per_row = interaction_features(poses, scale, pts, {"left": near, "right": near}, none)
    assert feats.touch_frac == 0.0 and feats.hand_valid_frac == 0.0 and np.isnan(feats.touch_frac_of_valid)
    assert np.isnan(per_row["cmin"]).all()


def test_longest_run() -> None:
    assert longest_run(np.array([0, 1, 1, 0, 1, 1, 1, 0], bool)) == 3
    assert longest_run(np.zeros(4, bool)) == 0 and longest_run(np.array([], bool)) == 0


def test_screen_config_refuses_an_unset_threshold_and_unknown_keys() -> None:
    assert InteractionScreenConfig() == InteractionScreenConfig.from_mapping(None)
    with pytest.raises(ValueError, match="min_value is null"):
        InteractionScreenConfig(enabled=True)
    with pytest.raises(ValueError, match="unknown keys"):
        InteractionScreenConfig.from_mapping({"enabled": False, "typo": 1})
    with pytest.raises(ValueError, match="feature"):
        InteractionScreenConfig(feature="cmin_median_mm")
    root = OmegaConf.load(_ROOT_CONFIG)
    assert QcStageConfig.from_mapping(root.qc_stage).interaction_screen == InteractionScreenConfig()
    cfg = QcStageConfig.from_mapping({"interaction_screen": {"enabled": True, "min_value": 0.26}})
    assert cfg.interaction_screen.enabled and cfg.interaction_screen.min_value == 0.26


def test_screen_passes_on_the_threshold() -> None:
    poses, scale, pts, near, far = _scene(t=10)
    right = np.concatenate([far[:3], near[3:8], far[8:]])
    feats, _ = interaction_features(poses, scale, pts, {"left": far, "right": right},
                                    {"left": np.ones(10, bool), "right": np.ones(10, bool)})
    assert screen_passes(feats, InteractionScreenConfig(enabled=True, min_value=0.5))
    assert not screen_passes(feats, InteractionScreenConfig(enabled=True, min_value=0.51))
    with pytest.raises(ValueError, match="touch gate"):
        screen_passes(feats, InteractionScreenConfig(enabled=True, min_value=0.1, touch_gate_mm=40.0))


# --- the qc stage: off is byte-identical; on adds the block and can only REJECT ------------------------------------


def _rule() -> AcceptanceRule:
    return AcceptanceRule(calibration=ConformalCalibration(threshold=0.5, alpha=0.1, n_calibration=20),
                          logistic_weights=np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0]), feature_mean=np.zeros(5),
                          feature_std=np.ones(5), tau=0.02)


def _passing(*args: object, **kwargs: object) -> QcFeatures:
    return PASSING


def test_a_disabled_screen_leaves_the_qc_artifacts_byte_identical(tmp_path: Path) -> None:
    out = {}
    clip = tmp_path / "clip"                    # one directory: the record embeds the mesh path
    write_solved_clip(clip, seed=3)
    for arm, screen in (("none", None), ("off", InteractionScreenConfig()),
                        ("off_with_value", InteractionScreenConfig(enabled=False, min_value=0.9))):
        make_qc_worker(_rule(), features_fn=_passing, interaction_screen=screen)("c", clip)
        out[arm] = ((clip / "qc.json").read_bytes(), (clip / "clip_record.npz").read_bytes())
        (clip / "qc.json").unlink()
        (clip / "clip_record.npz").unlink()
    assert out["none"] == out["off"] == out["off_with_value"]
    assert "interaction" not in json.loads(out["none"][0])


def test_an_enabled_screen_records_and_rejects_below_threshold(tmp_path: Path) -> None:
    for arm, value, accepted in (("keep", 0.0, True), ("drop", 1.0, False)):
        clip = tmp_path / arm
        write_solved_clip(clip, seed=3)
        (clip / "clip_record.npz").write_bytes(b"stale")
        make_qc_worker(_rule(), features_fn=_passing,
                       interaction_screen=InteractionScreenConfig(enabled=True, min_value=value))("c", clip)
        qc = json.loads((clip / "qc.json").read_text())
        assert qc["accepted"] is accepted and qc["rejected_for_no_interaction"] is (not accepted)
        assert qc["interaction"]["passed"] is accepted and qc["interaction"]["min_value"] == value
        assert set(qc["interaction"]["features"]) >= {"touch_frac", "longest_touch_run_frac", "hand_valid_frac"}
        assert (clip / "clip_record.npz").exists() is accepted
