"""The geometry stage's opt-in ego-motion -- geometry.npz byte-identical on/off, dense == gathered samples."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from fastdo.engine.workers import make_geometry_worker
from fastdo.geometry.ego_motion import EGO_MOTION_FILE, EgoMotionConfig
from tests.engine.fakes import SCENE_H, SCENE_N, SCENE_W, FakeMoGe, write_decode
from tests.engine.test_workers_geometry import _assert_same_arrays, _load, _masked_scene, _spy_paths

EGO = EgoMotionConfig(enabled=True, long_side=64, min_distance_px=3, min_inliers=4, min_inlier_frac=0.0,
                      max_corners=80, ransac_thresh_rel=0.5)


def _texture(clip: Path, tag_offset: int, seed: int) -> None:
    """Replaces the scene's blank decode rows with a shifting texture (the tag pixel kept), so corners exist."""
    rng = np.random.default_rng(seed)
    big = np.kron(rng.integers(0, 255, (20, 24)), np.ones((4, 4))).astype(np.uint8)
    frames = np.stack([np.repeat(big[r:r + SCENE_H, 2 * r:2 * r + SCENE_W, None], 3, axis=2) for r in range(SCENE_N)])
    frames[:, 0, 0, 0] = tag_offset + np.arange(SCENE_N)
    write_decode(clip, frames, frame_idx=[0, 3, 6, 9])


@pytest.fixture(autouse=True)
def _no_hand_exclusion(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fake scene's joints are scattered over the whole frame, so their box would exclude every pixel."""
    from fastdo.engine import geometry_ego

    monkeypatch.setattr(geometry_ego, "HANDS_FILE", "no_hands_for_the_exclusion.npz")


def _scenes(tmp_path: Path) -> tuple[list[Path], FakeMoGe]:
    dirs = [tmp_path / n for n in "ab"]
    fakes = [_masked_scene(d, tag_offset=10 * i, seed=i) for i, d in enumerate(dirs)]
    for i, d in enumerate(dirs):
        _texture(d, 10 * i, seed=i)
    return dirs, FakeMoGe(fakes[0].depth | fakes[1].depth)


def _run(dirs: list[Path], model: FakeMoGe, **kwargs: Any) -> list[dict[str, np.ndarray]]:
    make_geometry_worker(device="cpu", model=model, **kwargs).run_many([(d.name, d) for d in dirs])
    out = []
    for d in dirs:
        out.append({"geometry": _load(d / "geometry.npz"),
                    "ego": _load(d / EGO_MOTION_FILE) if (d / EGO_MOTION_FILE).is_file() else None})
        (d / "geometry.npz").unlink()
        (d / EGO_MOTION_FILE).unlink(missing_ok=True)
    return out


@pytest.mark.parametrize("storage", ["masked", "dense"])
def test_ego_off_writes_nothing_and_on_leaves_geometry_byte_identical(tmp_path: Path, storage: str) -> None:
    dirs, model = _scenes(tmp_path)
    off = _run(dirs, model, pointmap_storage=storage)
    disabled = _run(dirs, model, pointmap_storage=storage, ego_motion=EgoMotionConfig())
    on = _run(dirs, model, pointmap_storage=storage, ego_motion=EGO)
    for a, b, c in zip(off, disabled, on, strict=True):
        assert a["ego"] is None and b["ego"] is None and c["ego"] is not None
        _assert_same_arrays(b["geometry"], a["geometry"])
        _assert_same_arrays(c["geometry"], a["geometry"])
        assert c["ego"]["edge_n_points"].sum() > 0


def test_gathered_and_dense_paths_give_the_same_ego_motion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dirs, model = _scenes(tmp_path)
    calls = _spy_paths(monkeypatch)
    gathered = _run(dirs, model, pointmap_storage="masked", ego_motion=EGO)
    assert calls == {"dense": 0, "gathered": 1}
    split = _run(dirs, model, pointmap_storage="masked", ego_motion=EGO, max_frames_per_pass=3)   # dense fallback
    dense = _run(dirs, model, pointmap_storage="dense", ego_motion=EGO)
    assert calls["dense"] == 3 + 1
    for g, s, d in zip(gathered, split, dense, strict=True):
        for key in ("cam_to_world", "scale", "valid", "edges", "edge_n_points", "edge_n_inliers", "edge_rel"):
            assert g["ego"][key].tobytes() == s["ego"][key].tobytes() == d["ego"][key].tobytes(), key


def test_ego_needs_row_stride_one() -> None:
    with pytest.raises(ValueError, match="row_stride"):
        make_geometry_worker(device="cpu", model=object(), row_stride=2, ego_motion=EGO)
    make_geometry_worker(device="cpu", model=object(), row_stride=2, ego_motion=EgoMotionConfig())   # off: fine
