"""`geometry_stage.hand_sides` from the config to the `geometry` worker, and what `auto` does and does not change.

The engine anchored every clip's metric scale on the right hand (`DEFAULT_HAND_SIDES`), so a clip whose only valid
hand is the left failed at geometry after its whole front end was paid. `auto` recovers those clips; on every
clip with a valid right decode row it must be byte-identical to `right`, so no existing number moves.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from omegaconf import OmegaConf

from fastdo.bench.bench_stages import GEOMETRY_HAND_SIDES, build_stage_fns, geometry_hand_sides
from fastdo.engine.workers import DEFAULT_HAND_SIDES, make_geometry_worker
from fastdo.engine.workers_geometry import HAND_SIDES_AUTO
from tests.engine.fakes import (
    FALSE_SCALE,
    SCENE_H,
    SCENE_N,
    SCENE_W,
    FakeMoGe,
    write_geometry_scene,
)

ROOT = Path(__file__).resolve().parents[2]
DECODE_SOURCE_FRAMES = (0, 3, 6, 9)            # `write_geometry_scene`'s decode rows, as hands.npz source frames


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _same_arrays(got: dict[str, np.ndarray], want: dict[str, np.ndarray], skip: tuple[str, ...] = ()) -> None:
    """Every key, dtype, shape and byte (NaN payloads included), except `skip`."""
    assert set(got) == set(want)
    for key, expected in want.items():
        if key in skip:
            continue
        assert got[key].dtype == expected.dtype and got[key].shape == expected.shape, key
        assert got[key].tobytes() == expected.tobytes(), key


def _config() -> Any:
    return OmegaConf.load(ROOT / "run" / "conf" / "config.yaml")


def _scene(clip: Path, hands: str) -> FakeMoGe:
    """`write_geometry_scene` (right hand valid on every row) with its `hands.npz` rewritten:

    - `right`: unchanged (the left hand invalid);
    - `both`: the left hand ALSO valid on every row, at 1.3x the right hand's depth, so a rule that pooled or preferred
      the left would change the scale;
    - `right_once`: the right hand valid on ONE decode row only, the left (at 1.3x depth) on every row, so a rule that
      picked the side with more valid frames would pick the left;
    - `left_only`: the right hand's joints moved to the left side and the right hand invalid on every decode row, but
      valid on source frame 1, which is NOT a decode row (a rule that looked past the decode rows would keep right).
    """
    model = write_geometry_scene(clip)
    h = _load(clip / "hands.npz")
    right, n = h["right_joints"].copy(), h["right_valid"].shape[0]
    if hands in ("both", "right_once"):
        h["left_joints"], h["left_valid"] = (right * 1.3).astype(np.float32), np.ones(n, bool)
    if hands == "right_once":
        h["right_valid"] = np.zeros(n, bool)
        h["right_valid"][DECODE_SOURCE_FRAMES[2]] = True
    if hands == "left_only":
        h["left_joints"], h["left_valid"] = right, h["right_valid"].copy()
        h["right_joints"], h["right_valid"] = right * 0.5, np.zeros(n, bool)
        h["right_valid"][1] = True
    np.savez(clip / "hands.npz", **h)  # type: ignore[arg-type]  # numpy's stub types `allow_pickle` into **kwds
    masks = np.random.default_rng(3).random((SCENE_N, SCENE_H, SCENE_W)) < 0.3   # the same mask in every scene
    masks[:, 4:12, 6:20] = True
    np.savez(clip / "mask.npz", masks=masks, frame_idx=np.array(DECODE_SOURCE_FRAMES, dtype=np.int64),
             keyframe_index=np.array(0, dtype=np.int64))
    return model


def _geometry(tmp_path: Path, name: str, hands: str, storage: str, **kwargs: Any) -> dict[str, np.ndarray]:
    clip = tmp_path / name
    model = _scene(clip, hands)
    make_geometry_worker(device="cpu", model=model, pointmap_storage=storage, **kwargs)("clip", clip)
    return _load(clip / "geometry.npz")


# --- the config key -------------------------------------------------------------------------------------------------


def test_the_production_default_is_right_and_reaches_the_registered_worker() -> None:
    cfg = _config()
    assert cfg.geometry_stage.hand_sides == "right"
    assert geometry_hand_sides(cfg) == DEFAULT_HAND_SIDES == ("right",)
    assert geometry_hand_sides(OmegaConf.create({})) == DEFAULT_HAND_SIDES                         # no block
    assert geometry_hand_sides(OmegaConf.create({"geometry_stage": {"row_stride": 1}})) == DEFAULT_HAND_SIDES
    fns, labels = build_stage_fns(cfg, ROOT, [])
    assert fns["geometry"].hand_sides == DEFAULT_HAND_SIDES
    assert labels["geometry"] == "computed"                          # the default run's label does not move


@pytest.mark.parametrize(("value", "expected"), [("auto", HAND_SIDES_AUTO), ("left", ("left",)),
                                                 ("right", ("right",))])
def test_each_value_reaches_the_registered_worker(value: str, expected: Any) -> None:
    cfg = _config()
    OmegaConf.update(cfg, "geometry_stage.hand_sides", value)
    fns, labels = build_stage_fns(cfg, ROOT, [])
    assert fns["geometry"].hand_sides == expected == GEOMETRY_HAND_SIDES[value]
    assert labels["geometry"] == ("computed" if value == "right" else f"computed (hand anchor {value})")


@pytest.mark.parametrize("bad", ["both", "Right", "AUTO", "", None, ["left"], ["left", "right"], 1, True])
def test_anything_else_is_refused_at_construction(bad: Any) -> None:
    with pytest.raises(ValueError, match="geometry_stage.hand_sides"):
        geometry_hand_sides(OmegaConf.create({"geometry_stage": {"hand_sides": bad}}))
    cfg = _config()
    OmegaConf.update(cfg, "geometry_stage.hand_sides", bad, merge=False)
    with pytest.raises(ValueError, match="geometry_stage.hand_sides"):
        build_stage_fns(cfg, ROOT, [])


# --- what the values do ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("storage", ["dense", "masked"])
@pytest.mark.parametrize("hands", ["right", "both"])
def test_the_configured_default_is_byte_identical_to_the_default_worker(tmp_path: Path, storage: str,
                                                                         hands: str) -> None:
    got = _geometry(tmp_path, "cfg", hands, storage, hand_sides=geometry_hand_sides(_config()))
    want = _geometry(tmp_path, "default", hands, storage)
    _same_arrays(got, want)
    assert got["hand_sides"].tolist() == ["right"]


@pytest.mark.parametrize("storage", ["dense", "masked"])
@pytest.mark.parametrize("hands", ["right", "both", "right_once"])
def test_auto_is_byte_identical_to_right_wherever_the_right_hand_has_a_valid_decode_row(
        tmp_path: Path, storage: str, hands: str) -> None:
    auto = _geometry(tmp_path, "auto", hands, storage, hand_sides=GEOMETRY_HAND_SIDES["auto"])
    right = _geometry(tmp_path, "right", hands, storage, hand_sides=GEOMETRY_HAND_SIDES["right"])
    _same_arrays(auto, right)
    assert auto["hand_sides"].tolist() == ["right"]
    if hands == "both":                        # the left hand was valid too, and would have moved the scale
        left = _geometry(tmp_path, "left", hands, storage, hand_sides=GEOMETRY_HAND_SIDES["left"])
        assert float(left["scale"]) != pytest.approx(float(right["scale"]), rel=0.05)


@pytest.mark.parametrize("storage", ["dense", "masked"])
def test_auto_recovers_a_left_only_clip_and_records_the_side(tmp_path: Path, storage: str) -> None:
    # today's default drops the clip at geometry: the left-only failure, reproduced
    with pytest.raises(ValueError, match=r"metric scale must be finite and positive.*hand_sides=\('right',\)"):
        _geometry(tmp_path, "right", "left_only", storage, hand_sides=GEOMETRY_HAND_SIDES["right"])
    auto = _geometry(tmp_path, "auto", "left_only", storage, hand_sides=GEOMETRY_HAND_SIDES["auto"])
    assert auto["hand_sides"].tolist() == ["left"]                        # provenance for the stratified study
    assert float(auto["scale"]) == pytest.approx(FALSE_SCALE, rel=0.02)
    # the left hand holds the right-handed scene's joints, so it must give exactly the right-handed scene's geometry
    mirror = _geometry(tmp_path, "mirror", "right", storage)
    _same_arrays(auto, mirror, skip=("hand_sides",))
    forced = _geometry(tmp_path, "forced", "left_only", storage, hand_sides=GEOMETRY_HAND_SIDES["left"])
    _same_arrays(forced, auto)
