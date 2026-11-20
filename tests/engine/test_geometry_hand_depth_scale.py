"""`geometry_stage.hand_depth_scale`, the per-side relative correction of the hand depth the metric scale is
anchored on.

What must hold:
- off (null / absent) and factors of exactly 1.0 are byte-identical to today's worker;
- a factor f on the anchoring side multiplies the clip's metric scale (and every per-frame scale) by exactly f -- not
  1/f -- so f = 1 / (1 + r) cancels a hand placed a relative r too far and recovers the true metric depth;
- only the side that anchors the clip is corrected, on both pointmap storages and under a row stride;
- the pixels the anchor reads do not move (the joints are rescaled about the camera centre).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from fastdo.bench.bench_stages import build_stage_fns, geometry_hand_depth_scale
from fastdo.engine.workers import make_geometry_worker
from fastdo.engine.workers_geometry import HAND_SIDES_AUTO, check_hand_depth_scale
from fastdo.engine.workers_hands import SIDES
from fastdo.geometry.moge_pointmaps import hand_anchor_pixels, hand_anchored_scale
from tests.engine.fakes import (
    FALSE_SCALE,
    SCENE_H,
    SCENE_K,
    SCENE_N,
    SCENE_W,
    FakeMoGe,
    true_depth,
    write_geometry_scene,
)

ROOT = Path(__file__).resolve().parents[2]
DECODE_SOURCE_FRAMES = (0, 3, 6, 9)
R_RIGHT, R_LEFT = 0.021, -0.011               # measured side biases: right hands +2.1% far, left -1.1% near
STORAGES = ("dense", "masked")


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _same_arrays(got: dict[str, np.ndarray], want: dict[str, np.ndarray]) -> None:
    assert set(got) == set(want)
    for key, expected in want.items():
        assert got[key].dtype == expected.dtype and got[key].shape == expected.shape, key
        assert got[key].tobytes() == expected.tobytes(), key


def _scene(clip: Path, hands: str, bias: float = 0.0) -> FakeMoGe:
    """`write_geometry_scene` (right hand at the TRUE depth on every decode row), then:

    - `right`: the right hand rescaled about the camera centre by (1 + bias): HaWoR placing it a relative `bias` far;
    - `left_only`: the same hand moved to the LEFT side (right invalid on every row), rescaled by (1 + bias);
    - `both`: `right` plus a left hand at 1.3x the right hand's depth, valid everywhere.
    """
    model = write_geometry_scene(clip)
    h = _load(clip / "hands.npz")
    joints = (h["right_joints"].astype(np.float64) * (1.0 + bias)).astype(np.float32)
    n = h["right_valid"].shape[0]
    if hands in ("right", "both"):
        h["right_joints"] = joints
    if hands == "both":
        h["left_joints"], h["left_valid"] = (joints * 1.3).astype(np.float32), np.ones(n, bool)
    if hands == "left_only":
        h["left_joints"], h["left_valid"] = joints, h["right_valid"].copy()
        h["right_joints"], h["right_valid"] = joints * 0.5, np.zeros(n, bool)
    np.savez(clip / "hands.npz", **h)  # type: ignore[arg-type]  # numpy's stub types `allow_pickle` into **kwds
    masks = np.random.default_rng(3).random((SCENE_N, SCENE_H, SCENE_W)) < 0.3
    masks[:, 4:12, 6:20] = True
    np.savez(clip / "mask.npz", masks=masks, frame_idx=np.array(DECODE_SOURCE_FRAMES, dtype=np.int64),
             keyframe_index=np.array(0, dtype=np.int64))
    return model


def _geometry(tmp_path: Path, name: str, hands: str, storage: str, bias: float = 0.0,
              **kwargs: Any) -> dict[str, np.ndarray]:
    clip = tmp_path / name
    model = _scene(clip, hands, bias)
    make_geometry_worker(device="cpu", model=model, pointmap_storage=storage, **kwargs)("clip", clip)
    return _load(clip / "geometry.npz")


def _config() -> Any:
    return OmegaConf.load(ROOT / "run" / "conf" / "config.yaml")


# --- the config key and validation ----------------------------------------------------------------------------------


def test_the_production_default_is_off_and_reaches_the_registered_worker() -> None:
    cfg = _config()
    assert cfg.geometry_stage.hand_depth_scale is None
    assert geometry_hand_depth_scale(cfg) is None
    assert geometry_hand_depth_scale(OmegaConf.create({})) is None
    fns, labels = build_stage_fns(cfg, ROOT, [])
    assert fns["geometry"].hand_depth_scale is None
    assert labels["geometry"] == "computed"


def test_a_configured_correction_reaches_the_registered_worker() -> None:
    cfg = _config()
    OmegaConf.update(cfg, "geometry_stage.hand_sides", "auto")
    OmegaConf.update(cfg, "geometry_stage.hand_depth_scale", {"left": 1.011, "right": 0.979}, merge=False)
    fns, labels = build_stage_fns(cfg, ROOT, [])
    assert fns["geometry"].hand_depth_scale == {"left": 1.011, "right": 0.979}
    assert fns["geometry"].hand_sides == HAND_SIDES_AUTO
    assert labels["geometry"] == "computed (hand anchor auto) (hand depth x{'left': 1.011, 'right': 0.979})"


def test_factors_of_exactly_one_are_off() -> None:
    assert check_hand_depth_scale({"left": 1.0, "right": 1}) is None
    cfg = _config()
    OmegaConf.update(cfg, "geometry_stage.hand_depth_scale", {"left": 1.0, "right": 1.0}, merge=False)
    fns, labels = build_stage_fns(cfg, ROOT, [])
    assert fns["geometry"].hand_depth_scale is None and labels["geometry"] == "computed"


@pytest.mark.parametrize("bad", [
    {"right": 0.98}, {"left": 1.01}, {"left": 1.0, "right": 0.98, "both": 1.0}, {"Left": 1.0, "right": 0.98},
    {"left": True, "right": 0.98}, {"left": "1.0", "right": 0.98}, {"left": None, "right": 0.98},
    {"left": math.nan, "right": 0.98}, {"left": math.inf, "right": 0.98}, {"left": 0.0, "right": 0.98},
    {"left": -1.0, "right": 0.98}, {"left": 1.0, "right": 2.1}, {"left": 1.0, "right": 0.021},
    {"left": 1.0, "right": 0.79}, {"left": 1.26, "right": 1.0}, [1.0, 0.98], 0.98, "auto", True,
])
def test_anything_else_is_refused_at_construction(bad: Any) -> None:
    with pytest.raises(ValueError, match="hand_depth_scale"):
        check_hand_depth_scale(bad)
    with pytest.raises(ValueError, match="hand_depth_scale"):
        make_geometry_worker(device="cpu", hand_depth_scale=bad)
    cfg = _config()
    OmegaConf.update(cfg, "geometry_stage.hand_depth_scale", bad, merge=False)
    with pytest.raises(ValueError, match="geometry_stage.hand_depth_scale"):
        build_stage_fns(cfg, ROOT, [])


def test_the_range_edges_are_accepted() -> None:
    assert check_hand_depth_scale({"left": 0.8, "right": 1.25}) == {"left": 0.8, "right": 1.25}


def test_a_custom_scale_fn_is_refused_with_a_correction() -> None:
    with pytest.raises(ValueError, match="custom scale_fn"):
        make_geometry_worker(scale_fn=lambda *a: 1.0, device="cpu", hand_depth_scale={"left": 1.0, "right": 0.98})
    make_geometry_worker(scale_fn=lambda *a: 1.0, device="cpu", hand_depth_scale={"left": 1.0, "right": 1.0})


# --- byte identity when nothing is corrected ------------------------------------------------------------------------


@pytest.mark.parametrize("storage", STORAGES)
@pytest.mark.parametrize(("hands", "sides"), [("right", ("right",)), ("both", ("right",)), ("both", SIDES),
                                              ("left_only", HAND_SIDES_AUTO)])
@pytest.mark.parametrize("scale", [None, {"left": 1.0, "right": 1.0}])
def test_off_and_unit_factors_are_byte_identical_to_the_default_worker(tmp_path: Path, storage: str, hands: str,
                                                                       sides: Any, scale: Any) -> None:
    got = _geometry(tmp_path, "got", hands, storage, bias=R_RIGHT, hand_sides=sides, hand_depth_scale=scale)
    want = _geometry(tmp_path, "want", hands, storage, bias=R_RIGHT, hand_sides=sides)
    _same_arrays(got, want)


@pytest.mark.parametrize("storage", STORAGES)
def test_a_factor_on_the_side_that_does_not_anchor_changes_nothing(tmp_path: Path, storage: str) -> None:
    right = _geometry(tmp_path, "r", "right", storage, hand_sides=HAND_SIDES_AUTO,
                      hand_depth_scale={"left": 1.2, "right": 1.0})
    _same_arrays(right, _geometry(tmp_path, "r0", "right", storage, hand_sides=HAND_SIDES_AUTO))
    left = _geometry(tmp_path, "l", "left_only", storage, hand_sides=HAND_SIDES_AUTO,
                     hand_depth_scale={"left": 1.0, "right": 0.85})
    _same_arrays(left, _geometry(tmp_path, "l0", "left_only", storage, hand_sides=HAND_SIDES_AUTO))


# --- direction and magnitude ----------------------------------------------------------------------------------------


def _object_depth(g: dict[str, np.ndarray]) -> np.ndarray:
    """Every stored point's depth, whichever storage wrote it (masked: object pixels; dense: all pixels)."""
    if "object_points" in g:
        return np.concatenate([g["object_points"][:, 2], g["keyframe_pointmap"][..., 2].reshape(-1)])
    return g["pointmaps"][..., 2].reshape(-1)


@pytest.mark.parametrize("storage", STORAGES)
@pytest.mark.parametrize(("hands", "bias", "side"), [("right", R_RIGHT, "right"), ("left_only", R_LEFT, "left")])
def test_the_factor_multiplies_the_metric_scale_and_one_over_one_plus_r_recovers_the_truth(
        tmp_path: Path, storage: str, hands: str, bias: float, side: str) -> None:
    """The scene's hand sits a relative `bias` from the true depth. Uncorrected, the anchor reads the scale
    (1 + bias) too large (the metric scene is too deep/large: the right-handed +scale bias). The correction
    f = 1 / (1 + bias) multiplies the scale by f -- exactly, per frame and pooled -- and gives back the true scale,
    and the stored points' depth returns to the scene's true depth."""
    truth = _geometry(tmp_path, "truth", hands, storage, hand_sides=HAND_SIDES_AUTO)
    assert float(truth["scale"]) == pytest.approx(FALSE_SCALE, rel=1e-6)
    off = _geometry(tmp_path, "off", hands, storage, bias=bias, hand_sides=HAND_SIDES_AUTO)
    assert off["hand_sides"].tolist() == [side]
    assert float(off["scale"]) == pytest.approx(FALSE_SCALE * (1.0 + bias), rel=1e-6)   # the bias, uncorrected
    f = 1.0 / (1.0 + bias)
    other = 0.9 if side == "left" else 1.1              # the other side's factor must not be applied
    cal = _geometry(tmp_path, "cal", hands, storage, bias=bias, hand_sides=HAND_SIDES_AUTO,
                    hand_depth_scale={side: f, ("right" if side == "left" else "left"): other})
    assert float(cal["scale"]) / float(off["scale"]) == pytest.approx(f, rel=1e-6)       # x f, not x 1/f
    assert (float(cal["scale"]) < float(off["scale"])) == (bias > 0)                     # far hand -> shrink
    assert float(cal["scale"]) == pytest.approx(FALSE_SCALE, rel=1e-6)
    finite = np.isfinite(off["frame_scale"])
    assert finite.sum() == SCENE_N and np.array_equal(finite, np.isfinite(cal["frame_scale"]))
    np.testing.assert_allclose(cal["frame_scale"][finite] / off["frame_scale"][finite], f, rtol=1e-6)
    np.testing.assert_allclose(_object_depth(cal), _object_depth(truth), rtol=1e-6)
    np.testing.assert_allclose(_object_depth(cal), _object_depth(off) * f, rtol=1e-6)
    assert cal["hand_depth_scale"].dtype == np.float64 and cal["hand_depth_scale"].tolist() == [f]
    assert "hand_depth_scale" not in off


@pytest.mark.parametrize("storage", STORAGES)
def test_the_correction_holds_under_a_row_stride(tmp_path: Path, storage: str) -> None:
    off = _geometry(tmp_path, "off", "right", storage, bias=R_RIGHT, row_stride=2)
    cal = _geometry(tmp_path, "cal", "right", storage, bias=R_RIGHT, row_stride=2,
                    hand_depth_scale={"left": 1.0, "right": 1.0 / (1.0 + R_RIGHT)})
    assert float(cal["scale"]) == pytest.approx(FALSE_SCALE, rel=1e-6)
    finite = np.isfinite(off["frame_scale"])
    np.testing.assert_allclose(cal["frame_scale"][finite] / off["frame_scale"][finite], 1.0 / (1.0 + R_RIGHT),
                               rtol=1e-6)


def test_pooled_sides_weight_each_joint_by_its_own_side() -> None:
    """With both hands anchoring, each joint's depth takes its side's factor: equal to rescaling each hand's joints
    about the camera centre, and the anchor's pixels do not move."""
    rng = np.random.default_rng(7)
    t, j = 3, 42
    pm = torch.from_numpy(np.stack([np.stack([rng.random((SCENE_H, SCENE_W)), rng.random((SCENE_H, SCENE_W)),
                                              0.4 + rng.random((SCENE_H, SCENE_W))], -1) for _ in range(t)])
                          .astype(np.float32))
    u, v = rng.integers(0, SCENE_W, (t, j)) + 0.25, rng.integers(0, SCENE_H, (t, j)) + 0.25
    z = 0.5 + rng.random((t, j))
    K = SCENE_K
    joints = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], -1)
    valid = torch.ones((t, j), dtype=torch.bool)
    f = np.r_[np.full(21, 1.013), np.full(21, 0.978)]
    scaled = torch.from_numpy((joints * f[None, :, None]).astype(np.float64))
    k_t = torch.from_numpy(K)
    got = hand_anchored_scale(pm, torch.from_numpy(joints), k_t, valid, depth_scale=torch.from_numpy(f))
    want = hand_anchored_scale(pm, scaled, k_t, valid)
    torch.testing.assert_close(got, want, rtol=1e-6, atol=0)
    plain = hand_anchor_pixels(torch.from_numpy(joints), k_t, valid, (t, SCENE_H, SCENE_W))
    corrected = hand_anchor_pixels(torch.from_numpy(joints), k_t, valid, (t, SCENE_H, SCENE_W),
                                   depth_scale=torch.from_numpy(f))
    for a, b in zip(plain[:3], corrected[:3], strict=True):
        assert torch.equal(a, b)                                          # the same pixels
    assert plain[3].numel() == t * j                                      # every joint is read
    per_joint = np.tile(f, t)                                             # (frame, joint) order
    torch.testing.assert_close(corrected[3], plain[3] * torch.from_numpy(per_joint), rtol=0, atol=0)
    with pytest.raises(ValueError, match="one factor per joint"):
        hand_anchor_pixels(torch.from_numpy(joints), k_t, valid, (t, SCENE_H, SCENE_W),
                           depth_scale=torch.ones(21, dtype=torch.float64))


def test_the_uncorrected_true_depth_scene_is_the_fake_models_scale(tmp_path: Path) -> None:
    """Sanity of the instrument above: the fake MoGe depth is the true depth / FALSE_SCALE."""
    got = _geometry(tmp_path, "g", "right", "dense")
    np.testing.assert_allclose(got["pointmaps"][0][..., 2], true_depth(0), rtol=1e-6)
