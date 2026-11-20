"""Tests for the D3 camera stage workers: `intrinsics` (amendments) and `geometry` (R3 scale_fn, R8 K pass-through)."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import Tensor

from fastdo.engine import workers_geometry
from fastdo.engine.workers import DEFAULT_HAND_SIDES, make_geometry_worker, make_intrinsics_worker
from fastdo.geometry.intrinsics import (
    camera_matrix,
    pixel_intrinsics,
    read_intrinsics,
    write_intrinsics,
)
from fastdo.geometry.moge_model import fov_x_from_intrinsics
from fastdo.geometry.moge_pointmaps import hand_anchored_scale
from fastdo.hands.hand_camera import CAMERA_KEY, HAWOR_CAMERA_KEY, hawor_camera
from tests.engine.fakes import (
    FALSE_SCALE,
    NORMALIZED_K,
    SCENE_H,
    SCENE_K,
    SCENE_N,
    SCENE_W,
    FakeMoGe,
    forbid_full_frames,
    full_load_decode_rows,
    random_frames,
    true_depth,
    write_decode,
    write_geometry_scene,
)


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _true_pointmaps() -> np.ndarray:
    """The scene's metric pointmaps: true depth back-projected through `SCENE_K`."""
    fx, fy, cx, cy = float(SCENE_K[0, 0]), float(SCENE_K[1, 1]), float(SCENE_K[0, 2]), float(SCENE_K[1, 2])
    vv, uu = np.meshgrid(np.arange(SCENE_H, dtype=np.float32), np.arange(SCENE_W, dtype=np.float32), indexing="ij")
    maps = []
    for row in range(SCENE_N):
        z = true_depth(row)
        maps.append(np.stack([(uu - cx) * z / fx, (vv - cy) * z / fy, z], axis=-1))
    return np.stack(maps).astype(np.float32)


# --- intrinsics ---------------------------------------------------------------------------------------


def test_intrinsics_worker_given_camera_json(tmp_path: Path) -> None:
    write_decode(tmp_path, random_frames(4))
    (tmp_path / "camera.json").write_text(json.dumps({"fx": 999.0292358398438, "fy": 999.0292167663574,
                                                       "cx": 640, "cy": 360}))
    model = FakeMoGe()

    make_intrinsics_worker(device="cpu", model=model)("clip0", tmp_path)

    K, source = read_intrinsics(tmp_path)
    np.testing.assert_array_equal(K, camera_matrix(999.0292358398438, 999.0292167663574, 640.0, 360.0))
    assert source == "given"
    assert model.batches == []                                        # the model is called 0 times


def test_intrinsics_worker_moge_keyframes(tmp_path: Path) -> None:
    frames = np.zeros((46, 6, 8, 3), np.uint8)
    frames[:, 0, 0, 0] = np.arange(46)                                # FakeMoGe reads the row back from this tag
    write_decode(tmp_path, frames)
    model = FakeMoGe()

    make_intrinsics_worker(device="cpu", model=model)("clip0", tmp_path)

    assert [tag for batch in model.batches for tag in batch] == [0, 22, 45]
    K, source = read_intrinsics(tmp_path)
    assert source == "moge2"
    np.testing.assert_array_equal(K, pixel_intrinsics(np.repeat(NORMALIZED_K[None], 3, axis=0), width=8, height=6))


def test_intrinsics_worker_reads_only_its_keyframe_rows_and_writes_what_the_full_read_wrote(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The keyframe rows are read alone, and MoGe-2 sees the same bytes -- so the same K -- as before."""
    seen: list[Tensor] = []

    def pixel_dependent(frames: Tensor, device: str = "cuda", model: Any = None,
                        batch_size: int | None = None) -> Tensor:
        seen.append(frames.clone())
        f = frames.double().mean(dim=(1, 2, 3)) / 255.0 + 0.5              # (N,) a focal that depends on every pixel
        k = torch.zeros(frames.shape[0], 3, 3, dtype=torch.float32)
        k[:, 0, 0], k[:, 1, 1], k[:, 0, 2], k[:, 1, 2], k[:, 2, 2] = f.float(), 1.3 * f.float(), 0.5, 0.5, 1.0
        return k

    monkeypatch.setattr(workers_geometry.moge_model, "moge_intrinsics", pixel_dependent)
    frames = random_frames(46, h=6, w=8, seed=7)
    runs = {}
    for arm in ("before", "rows"):
        clip = tmp_path / arm
        write_decode(clip, frames)
        with pytest.MonkeyPatch.context() as mp:
            if arm == "before":
                mp.setattr(workers_geometry, "read_decode_rows", full_load_decode_rows)
            else:
                forbid_full_frames(mp)
            make_intrinsics_worker(n_keyframes=5, device="cpu", model=FakeMoGe())("clip0", clip)
        runs[arm] = (clip / "intrinsics.npz").read_bytes()

    assert runs["rows"] == runs["before"]
    before, rows = seen
    assert rows.dtype == before.dtype == torch.uint8 and rows.shape == before.shape == (5, 6, 8, 3)
    assert rows.numpy().tobytes() == before.numpy().tobytes() == frames[[0, 11, 22, 34, 45]].tobytes()


# --- geometry -----------------------------------------------------------------------------------------


def test_geometry_worker_applies_hand_anchored_scale(tmp_path: Path) -> None:
    model = write_geometry_scene(tmp_path)                            # MoGe depth = true depth / 2.5

    make_geometry_worker(device="cpu", model=model)("clip0", tmp_path)

    out = _load(tmp_path / "geometry.npz")
    # `hand_sides` is provenance: which hand actually set the scale. It is load-bearing for the stratified
    # left-vs-right bias study, because under HAND_SIDES_AUTO the side varies per clip and a pooled number
    # over a mixed population would describe the mixture, not either side.
    assert set(out) == {"pointmaps", "frame_idx", "scale", "frame_scale", "hand_sides"}
    assert out["pointmaps"].shape == (SCENE_N, SCENE_H, SCENE_W, 3) and out["pointmaps"].dtype == np.float32
    assert out["frame_idx"].tolist() == [0, 3, 6, 9] and out["frame_idx"].dtype == np.int64
    truth = _true_pointmaps()
    for row in range(SCENE_N):
        extent = np.linalg.norm(out["pointmaps"][row].reshape(-1, 3).max(0) - out["pointmaps"][row].reshape(-1, 3).min(0))
        true_extent = np.linalg.norm(truth[row].reshape(-1, 3).max(0) - truth[row].reshape(-1, 3).min(0))
        assert abs(extent / true_extent - 1.0) < 0.02
    assert float(out["scale"]) == pytest.approx(FALSE_SCALE, rel=0.02)
    # Per frame, not only pooled: hand rows matched by row position instead of source frame index give per-frame
    # scales of about [2.5, 4.63, 4.32, 2.18], whose lower median over 4 frames is still 2.5. Only
    # `frame_scale` exposes that mismatch.
    assert out["frame_scale"].shape == (SCENE_N,) and out["frame_scale"].dtype == np.float32
    np.testing.assert_allclose(out["frame_scale"], FALSE_SCALE, rtol=0.02)


def test_geometry_worker_default_scale_fn_is_hand_anchored_scale() -> None:
    assert inspect.signature(make_geometry_worker).parameters["scale_fn"].default is hand_anchored_scale


def test_geometry_worker_uses_the_injected_scale_fn(tmp_path: Path) -> None:
    model = write_geometry_scene(tmp_path)
    calls: list[tuple[tuple[int, ...], Tensor, Tensor, Tensor]] = []

    def fused_scale(pointmaps: Tensor, joints: Tensor, K: Tensor, valid: Tensor) -> float:
        calls.append((tuple(pointmaps.shape), joints.clone(), K.clone(), valid.clone()))
        return 1.75

    make_geometry_worker(scale_fn=fused_scale, device="cpu", model=model)("clip0", tmp_path)

    assert len(calls) == 1
    points_shape, joints, K, valid = calls[0]
    assert points_shape == (SCENE_N, SCENE_H, SCENE_W, 3)
    hands = _load(tmp_path / "hands.npz")
    assert DEFAULT_HAND_SIDES == ("right",)                           # D2's single measured side
    np.testing.assert_array_equal(joints.numpy(), hands["right_joints"][[0, 3, 6, 9]])   # decode rows only
    assert tuple(valid.shape) == (SCENE_N, 21) and bool(valid.all())
    assert torch.equal(K, torch.from_numpy(SCENE_K))
    out = _load(tmp_path / "geometry.npz")
    assert float(out["scale"]) == 1.75
    expected_z = np.stack([true_depth(r) for r in range(SCENE_N)]) / FALSE_SCALE * 1.75
    np.testing.assert_allclose(out["pointmaps"][..., 2], expected_z, rtol=1e-5)


def test_geometry_worker_hand_sides_select_the_joints_that_feed_the_scale(tmp_path: Path) -> None:
    model = write_geometry_scene(tmp_path)
    seen: list[tuple[Tensor, Tensor]] = []

    def spy_scale(pointmaps: Tensor, joints: Tensor, K: Tensor, valid: Tensor) -> Tensor:
        seen.append((joints.clone(), valid.clone()))
        return hand_anchored_scale(pointmaps, joints, K, valid)

    make_geometry_worker(scale_fn=spy_scale, device="cpu", model=model, hand_sides=("left", "right"))(
        "clip0", tmp_path)

    hands = _load(tmp_path / "hands.npz")
    joints, valid = seen[0]
    expected = np.concatenate([hands["left_joints"], hands["right_joints"]], axis=1)[[0, 3, 6, 9]]
    np.testing.assert_array_equal(joints.numpy(), expected)           # both hands pooled, left then right
    assert not bool(valid[:, :21].any()) and bool(valid[:, 21:].all())
    assert float(_load(tmp_path / "geometry.npz")["scale"]) == pytest.approx(FALSE_SCALE, rel=0.02)

    with pytest.raises(ValueError, match=r"hand_sides=\('left',\)"):   # the scene's left hand is never valid
        make_geometry_worker(device="cpu", model=model, hand_sides=("left",))("clip0", tmp_path)
    for bad in [(), ("middle",), ("right", "right")]:
        with pytest.raises(ValueError, match="hand_sides"):
            make_geometry_worker(device="cpu", model=model, hand_sides=bad)


def test_geometry_worker_passes_the_intrinsics_stage_K_into_moge(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = write_geometry_scene(tmp_path)
    seen: list[Tensor | None] = []
    real_run_moge = workers_geometry.run_moge

    def spy(frames: Tensor, device: str = "cuda", model: Any = None, intrinsics: Tensor | None = None,
            batch_size: int | None = None, overlap: bool = False) -> Tensor:
        seen.append(intrinsics)
        return real_run_moge(frames, device=device, model=model, intrinsics=intrinsics, batch_size=batch_size,
                             overlap=overlap)

    monkeypatch.setattr(workers_geometry, "run_moge", spy)

    make_geometry_worker(device="cpu", model=model)("clip0", tmp_path)

    K, _ = read_intrinsics(tmp_path)
    assert len(seen) == 1 and seen[0] is not None
    assert torch.equal(seen[0], torch.from_numpy(K))
    fov_x = fov_x_from_intrinsics(torch.from_numpy(K), SCENE_W)
    assert model.kwargs and all(kwargs == {"fov_x": fov_x} for kwargs in model.kwargs)


def test_geometry_worker_rejects_hands_missing_a_decode_frame(tmp_path: Path) -> None:
    model = write_geometry_scene(tmp_path)
    hands = _load(tmp_path / "hands.npz")
    np.savez(tmp_path / "hands.npz", **{k: (v if v.ndim == 0 or v.shape[0] != 12 else v[:9])
                                        for k, v in hands.items()})  # type: ignore[arg-type]  # numpy's stub types `allow_pickle` into **kwds

    with pytest.raises(ValueError, match=r"frame index 9\b"):
        make_geometry_worker(device="cpu", model=model)("clip0", tmp_path)
    assert not (tmp_path / "geometry.npz").exists()


def test_geometry_worker_requires_the_intrinsics_and_hands_artifacts(tmp_path: Path) -> None:
    write_decode(tmp_path, random_frames(2))
    with pytest.raises(FileNotFoundError, match="intrinsics"):
        make_geometry_worker(device="cpu", model=FakeMoGe())("clip0", tmp_path)
    write_intrinsics(tmp_path, SCENE_K, "given")
    with pytest.raises(FileNotFoundError, match="hands"):
        make_geometry_worker(device="cpu", model=FakeMoGe())("clip0", tmp_path)


def test_geometry_worker_concatenates_clips_that_share_a_camera(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dirs = [tmp_path / "a", tmp_path / "b", tmp_path / "c"]
    fakes = [write_geometry_scene(d, tag_offset=10 * i) for i, d in enumerate(dirs)]
    write_intrinsics(dirs[2], camera_matrix(60.0, 60.0, 31.0, 23.5), "given")   # clip c: another camera
    model = FakeMoGe(fakes[0].depth | fakes[1].depth | fakes[2].depth)
    frames_per_call: list[int] = []
    real_run_moge = workers_geometry.run_moge

    def spy(frames: Tensor, device: str = "cuda", model: Any = None, intrinsics: Tensor | None = None,
            batch_size: int | None = None, overlap: bool = False) -> Tensor:
        frames_per_call.append(int(frames.shape[0]))
        return real_run_moge(frames, device=device, model=model, intrinsics=intrinsics, batch_size=batch_size,
                             overlap=overlap)

    monkeypatch.setattr(workers_geometry, "run_moge", spy)

    make_geometry_worker(device="cpu", model=model).run_many([(d.name, d) for d in dirs])

    assert frames_per_call == [2 * SCENE_N, SCENE_N]                  # a + b in one MoGe call, c on its own
    for d in dirs[:2]:
        assert float(_load(d / "geometry.npz")["scale"]) == pytest.approx(FALSE_SCALE, rel=0.02)
    assert (dirs[2] / "geometry.npz").is_file()


def test_geometry_worker_splits_long_clips_into_bounded_moge_calls(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    split, whole = [tmp_path / "split" / n for n in "ab"], [tmp_path / "whole" / n for n in "ab"]
    fakes = [write_geometry_scene(d, tag_offset=10 * i) for i, d in enumerate(split)]
    for i, d in enumerate(whole):
        write_geometry_scene(d, tag_offset=10 * i)
    model = FakeMoGe(fakes[0].depth | fakes[1].depth)
    frames_per_call: list[int] = []
    real_run_moge = workers_geometry.run_moge

    def spy(frames: Tensor, device: str = "cuda", model: Any = None, intrinsics: Tensor | None = None,
            batch_size: int | None = None, overlap: bool = False) -> Tensor:
        frames_per_call.append(int(frames.shape[0]))
        return real_run_moge(frames, device=device, model=model, intrinsics=intrinsics, batch_size=batch_size,
                             overlap=overlap)

    monkeypatch.setattr(workers_geometry, "run_moge", spy)

    make_geometry_worker(max_frames_per_pass=3, device="cpu", model=model).run_many([(d.name, d) for d in split])
    assert frames_per_call == [3, 3, 2]                               # a0-a2 | a3 b0 b1 | b2 b3: no call over 3
    make_geometry_worker(device="cpu", model=model).run_many([(d.name, d) for d in whole])
    assert frames_per_call[3:] == [2 * SCENE_N]

    for d_split, d_whole in zip(split, whole, strict=True):
        got, expected = _load(d_split / "geometry.npz"), _load(d_whole / "geometry.npz")
        assert set(got) == set(expected)
        for key in expected:
            np.testing.assert_array_equal(got[key], expected[key])


# --- the hands' camera must be the projection camera ---------------------------------------------------------------


def test_geometry_worker_rejects_uncorrected_hands_under_an_offset_principal_point(tmp_path: Path) -> None:
    """`write_geometry_scene`'s hands.npz records no camera correction (as hands.npz once did). With a given K whose
    principal point is 11.5 px from the image centre, projecting those HaWoR outputs through K is the camera-mismatch
    bug: a hard error."""
    model = write_geometry_scene(tmp_path)
    write_intrinsics(tmp_path, camera_matrix(60.0, 60.0, 20.5, 30.0), "given")

    with pytest.raises(ValueError, match=r"principal point"):
        make_geometry_worker(device="cpu", model=model)("clip0", tmp_path)
    assert not (tmp_path / "geometry.npz").exists()


def test_geometry_worker_rejects_hands_recorded_in_another_camera(tmp_path: Path) -> None:
    model = write_geometry_scene(tmp_path)
    other = camera_matrix(60.0, 60.0, 20.5, 30.0)
    hands = _load(tmp_path / "hands.npz")
    np.savez(tmp_path / "hands.npz", **hands, **{CAMERA_KEY: other.astype(np.float64),  # type: ignore[arg-type]
                                                 HAWOR_CAMERA_KEY: hawor_camera(other, SCENE_W, SCENE_H)})

    with pytest.raises(ValueError, match=r"camera"):
        make_geometry_worker(device="cpu", model=model)("clip0", tmp_path)
    assert not (tmp_path / "geometry.npz").exists()


def test_geometry_worker_accepts_hands_recorded_in_its_camera(tmp_path: Path) -> None:
    model = write_geometry_scene(tmp_path)
    hands = _load(tmp_path / "hands.npz")
    np.savez(tmp_path / "hands.npz", **hands, **{CAMERA_KEY: SCENE_K.astype(np.float64),  # type: ignore[arg-type]
                                                 HAWOR_CAMERA_KEY: hawor_camera(SCENE_K, SCENE_W, SCENE_H)})

    make_geometry_worker(device="cpu", model=model)("clip0", tmp_path)

    assert float(_load(tmp_path / "geometry.npz")["scale"]) == pytest.approx(FALSE_SCALE, rel=0.02)


# --- geometry on every k-th row --------------------------------------------------------


def _write_mask(clip: Path, keyframe_index: int) -> None:
    np.savez(clip / "mask.npz", masks=np.ones((SCENE_N, SCENE_H, SCENE_W), dtype=bool),
             frame_idx=np.array([0, 3, 6, 9], dtype=np.int64), keyframe_index=np.array(keyframe_index, dtype=np.int64))


def test_geometry_worker_row_stride_runs_moge_on_every_kth_row_and_the_keyframe(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Stride 2 with the keyframe on row 3: MoGe-2 sees rows [0, 2, 3]; row 1 is NaN; the rows it saw are exactly the
    stride-1 rows (the fake's per-frame scale is constant, so the pooled scale cannot differ)."""
    full, strided = tmp_path / "full", tmp_path / "strided"
    model = write_geometry_scene(full)
    write_geometry_scene(strided)
    _write_mask(strided, keyframe_index=3)
    seen: list[int] = []
    real_run_moge = workers_geometry.run_moge

    def spy(frames: Tensor, device: str = "cuda", model: Any = None, intrinsics: Tensor | None = None,
            batch_size: int | None = None, overlap: bool = False) -> Tensor:
        seen.append(int(frames.shape[0]))
        return real_run_moge(frames, device=device, model=model, intrinsics=intrinsics, batch_size=batch_size,
                             overlap=overlap)

    make_geometry_worker(device="cpu", model=model)("clip0", full)
    monkeypatch.setattr(workers_geometry, "run_moge", spy)
    make_geometry_worker(device="cpu", model=model, row_stride=2)("clip0", strided)

    assert seen == [3]
    ref, out = _load(full / "geometry.npz"), _load(strided / "geometry.npz")
    assert out["pointmaps"].shape == ref["pointmaps"].shape
    np.testing.assert_array_equal(out["frame_idx"], ref["frame_idx"])          # every decode row keeps its index
    assert np.isnan(out["pointmaps"][1]).all() and np.isnan(out["frame_scale"][1])
    for row in (0, 2, 3):
        np.testing.assert_allclose(out["pointmaps"][row], ref["pointmaps"][row], rtol=1e-6)
    assert float(out["scale"]) == pytest.approx(float(ref["scale"]), rel=1e-6)


def test_geometry_worker_row_stride_is_validated(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="row_stride"):
        make_geometry_worker(device="cpu", row_stride=0)
    model = write_geometry_scene(tmp_path)                                    # no mask.npz: no keyframe to keep
    with pytest.raises(FileNotFoundError, match="mask"):
        make_geometry_worker(device="cpu", model=model, row_stride=2)("clip0", tmp_path)


def test_geometry_row_stride_reaches_the_worker_through_the_registered_stage_builder() -> None:
    """Reachability: the config key must arrive at the worker the engine actually builds, not only the factory."""
    from omegaconf import OmegaConf

    from fastdo.bench.bench_stages import build_stage_fns, geometry_row_stride

    root = Path(__file__).resolve().parents[2]
    cfg = OmegaConf.load(root / "run" / "conf" / "config.yaml")
    assert geometry_row_stride(cfg) == 1                                       # production default: every row
    OmegaConf.update(cfg, "geometry_stage.row_stride", 2)
    fns = build_stage_fns(cfg, root, [])[0]
    assert fns["geometry"].row_stride == 2
    for bad in (0, 1.5, True):
        with pytest.raises(ValueError, match="row_stride"):
            geometry_row_stride(OmegaConf.create({"geometry_stage": {"row_stride": bad}}))


def test_default_pass_is_one_s2_window_and_even():
    """A 100-row S2 window must fit one call (no host-buffer reassembly), and the size must stay even so MoGe's
    batch-of-2 pairs the same rows as the old 64-frame passes (the outputs were bit-identical on the real model)."""
    from fastdo.geometry.moge_model import _BATCH_SIZE
    assert workers_geometry.DEFAULT_MAX_FRAMES_PER_PASS >= 100
    assert workers_geometry.DEFAULT_MAX_FRAMES_PER_PASS % _BATCH_SIZE == 0


def test_even_pass_sizes_pair_the_same_rows(tmp_path):
    """The model sees the same frame pairs at pass size 2, 4 and the default, so a pass-size change cannot move a
    frame into a different fp16 batch."""
    seen = []
    for size in (2, 4, workers_geometry.DEFAULT_MAX_FRAMES_PER_PASS):
        dirs = [tmp_path / f"s{size}" / n for n in "abc"]
        fakes = [write_geometry_scene(d, tag_offset=10 * i) for i, d in enumerate(dirs)]
        model = FakeMoGe(fakes[0].depth | fakes[1].depth | fakes[2].depth)
        make_geometry_worker(max_frames_per_pass=size, device="cpu", model=model).run_many([(d.name, d) for d in dirs])
        seen.append(sorted(tuple(b) for b in model.batches))
    assert seen[0] == seen[1] == seen[2]


# --- masked storage gathers only the kept pixels on the device -----------------------------------------------------


def _masked_scene(clip: Path, tag_offset: int = 0, keyframe_index: int = 2, seed: int = 0) -> FakeMoGe:
    """`write_geometry_scene` plus a `mask.npz` of moving rectangles and scattered pixels (frame 1 empty), and MoGe
    marking ~10% of pixels invalid -- some on the hand joints' pixels, some under the mask -- so NaN positions and the
    scale's non-finite-sample branch are both exercised."""
    model = write_geometry_scene(clip, tag_offset=tag_offset)
    rng = np.random.default_rng(100 + seed)
    anchor = np.random.default_rng(0)                                 # `write_geometry_scene`'s joint pixels
    us, vs = anchor.integers(2, SCENE_W - 2, 21), anchor.integers(2, SCENE_H - 2, 21)
    for r in range(SCENE_N):
        depth = model.depth[tag_offset + r].copy()
        depth[rng.random((SCENE_H, SCENE_W)) < 0.1] = np.inf
        depth[vs[: 3 + r], us[: 3 + r]] = np.inf
        model.depth[tag_offset + r] = depth
    masks = rng.random((SCENE_N, SCENE_H, SCENE_W)) < 0.05
    for r in range(SCENE_N):
        masks[r, 5 + 3 * r:20 + 3 * r, 10 + 4 * r:40 + 2 * r] = True
    masks[1] = False
    np.savez(clip / "mask.npz", masks=masks, frame_idx=np.array([0, 3, 6, 9], dtype=np.int64),
             keyframe_index=np.array(keyframe_index, dtype=np.int64))
    return model


def _left_handed(clip: Path) -> None:
    """Moves the scene's right hand to the left side (the right one invalid), for `hand_sides="auto"`."""
    hands = _load(clip / "hands.npz")
    hands["left_joints"], hands["left_valid"] = hands["right_joints"], hands["right_valid"]
    hands["right_joints"], hands["right_valid"] = np.zeros_like(hands["right_joints"]), np.zeros(12, bool)
    np.savez(clip / "hands.npz", **hands)  # type: ignore[arg-type]  # numpy's stub types `allow_pickle` into **kwds


def _assert_same_arrays(got: dict[str, np.ndarray], expected: dict[str, np.ndarray]) -> None:
    """Every key, dtype, shape and byte (so NaN positions and payloads too)."""
    assert set(got) == set(expected)
    for key, want in expected.items():
        assert got[key].dtype == want.dtype and got[key].shape == want.shape, key
        assert got[key].tobytes() == want.tobytes(), key


def _reference(dirs: list[Path], model: FakeMoGe, **kwargs: Any) -> list[dict[str, np.ndarray]]:
    """The masked artifact before device-side gathering: the dense one, packed with `pack_masked` as `_write` does."""
    from fastdo.engine.geometry_io import pack_masked

    make_geometry_worker(device="cpu", model=model, pointmap_storage="dense", **kwargs).run_many(
        [(d.name, d) for d in dirs])
    out = []
    for d in dirs:
        dense, mask = _load(d / "geometry.npz"), _load(d / "mask.npz")
        stored = pack_masked(dense.pop("pointmaps"), mask["masks"], int(mask["keyframe_index"]))
        out.append(dense | stored)
        (d / "geometry.npz").unlink()
    return out


def _spy_paths(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    calls = {"dense": 0, "gathered": 0}
    real_run_moge, real_batches = workers_geometry.run_moge, workers_geometry.run_moge_batches

    def dense(*args: Any, **kwargs: Any) -> Tensor:
        calls["dense"] += 1
        return real_run_moge(*args, **kwargs)

    def gathered(*args: Any, **kwargs: Any) -> None:
        calls["gathered"] += 1
        real_batches(*args, **kwargs)

    monkeypatch.setattr(workers_geometry, "run_moge", dense)
    monkeypatch.setattr(workers_geometry, "run_moge_batches", gathered)
    return calls


@pytest.mark.parametrize("hand_sides", [("right",), ("left", "right"), "auto"])
def test_masked_storage_gathered_pass_writes_the_packed_dense_arrays(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hand_sides: Any) -> None:
    """Three clips, two sharing one call (clip b's keyframe on its last row, its hand on the left) and one on another
    camera: the gathered path writes byte-for-byte the arrays the dense path packs."""
    dirs = [tmp_path / n for n in "abc"]
    fakes = [_masked_scene(d, tag_offset=10 * i, keyframe_index=(2, 3, 0)[i], seed=i) for i, d in enumerate(dirs)]
    if hand_sides == "auto":
        _left_handed(dirs[1])
    write_intrinsics(dirs[2], camera_matrix(60.0, 60.0, 31.0, 23.5), "given")   # clip c: another camera
    model = FakeMoGe(fakes[0].depth | fakes[1].depth | fakes[2].depth)
    expected = _reference(dirs, model, hand_sides=hand_sides)

    calls = _spy_paths(monkeypatch)
    make_geometry_worker(device="cpu", model=model, hand_sides=hand_sides, pointmap_storage="masked").run_many(
        [(d.name, d) for d in dirs])

    assert calls == {"dense": 0, "gathered": 2}
    for d, ref in zip(dirs, expected, strict=True):
        got = _load(d / "geometry.npz")
        assert np.isnan(got["object_points"]).any() and np.isfinite(got["object_points"]).any()
        assert np.isnan(got["frame_scale"]).sum() < SCENE_N                    # the scale is exercised, not all-NaN
        _assert_same_arrays(got, ref)
    if hand_sides == "auto":
        assert _load(dirs[1] / "geometry.npz")["hand_sides"].tolist() == ["left"]


def test_masked_storage_row_stride_gathered_pass_matches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Stride 2, keyframe on row 3: MoGe sees rows [0, 2, 3]; row 1's object pixels are NaN as in the dense path."""
    dirs = [tmp_path / n for n in "ab"]
    fakes = [_masked_scene(d, tag_offset=10 * i, keyframe_index=3, seed=i) for i, d in enumerate(dirs)]
    for d in dirs:                                                    # row 1 must hold object pixels to be NaN'd
        mask = _load(d / "mask.npz")
        mask["masks"][1, 30:40, 30:50] = True
        np.savez(d / "mask.npz", **mask)  # type: ignore[arg-type]  # numpy's stub types `allow_pickle` into **kwds
    model = FakeMoGe(fakes[0].depth | fakes[1].depth)
    expected = _reference(dirs, model, row_stride=2)

    calls = _spy_paths(monkeypatch)
    make_geometry_worker(device="cpu", model=model, pointmap_storage="masked", row_stride=2).run_many(
        [(d.name, d) for d in dirs])

    assert calls == {"dense": 0, "gathered": 1}
    for d, ref in zip(dirs, expected, strict=True):
        got = _load(d / "geometry.npz")
        row1 = got["object_points"][got["object_offsets"][1]:got["object_offsets"][2]]
        assert row1.size and np.isnan(row1).all() and np.isnan(got["frame_scale"][1])
        _assert_same_arrays(got, ref)


def test_masked_storage_falls_back_to_the_dense_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A clip split across calls, and a custom `scale_fn` (its pixels are unknown), take the dense path: same arrays."""
    dirs = [tmp_path / n for n in "ab"]
    fakes = [_masked_scene(d, tag_offset=10 * i, seed=i) for i, d in enumerate(dirs)]
    model = FakeMoGe(fakes[0].depth | fakes[1].depth)
    expected = _reference(dirs, model)

    calls = _spy_paths(monkeypatch)
    make_geometry_worker(max_frames_per_pass=3, device="cpu", model=model, pointmap_storage="masked").run_many(
        [(d.name, d) for d in dirs])
    assert calls == {"dense": 3, "gathered": 0}                       # a0-a2 | a3 b0 b1 | b2 b3
    for d, ref in zip(dirs, expected, strict=True):
        _assert_same_arrays(_load(d / "geometry.npz"), ref)

    def custom(pointmaps: Tensor, joints: Tensor, K: Tensor, valid: Tensor) -> Tensor:
        return hand_anchored_scale(pointmaps, joints, K, valid)

    make_geometry_worker(scale_fn=custom, device="cpu", model=model, pointmap_storage="masked").run_many(
        [(d.name, d) for d in dirs])
    assert calls == {"dense": 4, "gathered": 0}
    for d, ref in zip(dirs, expected, strict=True):
        _assert_same_arrays(_load(d / "geometry.npz"), ref)


def test_masked_storage_gathered_pass_checks_the_mask(tmp_path: Path) -> None:
    """The gathered path refuses what `_write` + `pack_masked` refuse, before running MoGe."""
    model = _masked_scene(tmp_path)
    mask = _load(tmp_path / "mask.npz")
    np.savez(tmp_path / "mask.npz", **(mask | {"frame_idx": np.array([0, 3, 6, 10])}))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="not the geometry rows"):
        make_geometry_worker(device="cpu", model=model, pointmap_storage="masked")("clip0", tmp_path)
    np.savez(tmp_path / "mask.npz", **(mask | {"keyframe_index": np.array(4)}))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="keyframe_index 4 out of range"):
        make_geometry_worker(device="cpu", model=model, pointmap_storage="masked")("clip0", tmp_path)
    assert model.batches == [] and not (tmp_path / "geometry.npz").exists()


# --- geometry_stage.moge_batch_size (MoGe-2 frames per forward) -------------------------------------------------


@pytest.mark.parametrize("storage", ["dense", "masked"])
def test_geometry_worker_moge_batch_size_default_is_todays_batches_and_8_differs(tmp_path: Path, storage: str) -> None:
    """Default and explicit 2: FakeMoGe sees today's pairs, and the artifacts are byte-identical; 8: one forward of the
    4 rows (the fake is per-frame, so its artifact is identical too -- only the batching moved). Both storage paths
    (masked = the gathered `run_moge_batches`, dense = `run_moge`)."""
    artifacts: dict[Any, bytes] = {}
    batches: dict[Any, list[list[int]]] = {}
    for batch in (None, 2, 8):
        clip = tmp_path / f"b{batch}"
        model = _masked_scene(clip)
        kwargs = {} if batch is None else {"moge_batch_size": batch}
        make_geometry_worker(device="cpu", model=model, pointmap_storage=storage, **kwargs)("clip0", clip)
        artifacts[batch], batches[batch] = (clip / "geometry.npz").read_bytes(), model.batches
    assert batches[None] == batches[2] == [[0, 1], [2, 3]]
    assert batches[8] == [[0, 1, 2, 3]]
    assert artifacts[2] == artifacts[None] and artifacts[8] == artifacts[None]


def test_intrinsics_worker_moge_batch_size(tmp_path: Path) -> None:
    """The intrinsics keyframes go through MoGe in `moge_batch_size` chunks; the default is today's 2."""
    frames = np.zeros((46, 6, 8, 3), np.uint8)
    frames[:, 0, 0, 0] = np.arange(46)
    got: dict[Any, tuple[list[list[int]], bytes]] = {}
    for batch in (None, 2, 8):
        clip = tmp_path / f"b{batch}"
        write_decode(clip, frames)
        model = FakeMoGe()
        kwargs = {} if batch is None else {"moge_batch_size": batch}
        make_intrinsics_worker(n_keyframes=5, device="cpu", model=model, **kwargs)("clip0", clip)
        got[batch] = (model.batches, (clip / "intrinsics.npz").read_bytes())
    assert got[None][0] == got[2][0] == [[0, 11], [22, 34], [45]]
    assert got[8][0] == [[0, 11, 22, 34, 45]]
    assert got[None][1] == got[2][1] == got[8][1]


@pytest.mark.parametrize("bad", [0, -2, 1.5, True, "8"])
def test_moge_batch_size_is_validated_at_construction(bad: Any) -> None:
    with pytest.raises(ValueError, match="moge_batch_size"):
        make_geometry_worker(device="cpu", moge_batch_size=bad)
    with pytest.raises(ValueError, match="moge_batch_size"):
        make_intrinsics_worker(device="cpu", moge_batch_size=bad)


def test_moge_batch_size_reaches_both_workers_through_the_registered_stage_builder(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reachability: `geometry_stage.moge_batch_size` must arrive at the workers `build_stage_fns` builds -- the
    geometry worker's attribute, and the value the intrinsics worker hands `moge_intrinsics` (spied: the bench builds
    on cuda)."""
    from omegaconf import OmegaConf

    from fastdo.bench.bench_stages import build_stage_fns, geometry_moge_batch_size

    root = Path(__file__).resolve().parents[2]
    cfg = OmegaConf.load(root / "run" / "conf" / "config.yaml")
    assert geometry_moge_batch_size(cfg) == 2                                  # production default: today's 2
    assert geometry_moge_batch_size(OmegaConf.create({})) == 2                 # no block: today's 2
    seen: list[Any] = []

    def spy(frames: Tensor, device: str = "cuda", model: Any = None, batch_size: int | None = None) -> Tensor:
        seen.append(batch_size)
        return torch.from_numpy(np.repeat(NORMALIZED_K[None], frames.shape[0], axis=0))

    monkeypatch.setattr(workers_geometry.moge_model, "moge_intrinsics", spy)
    monkeypatch.setattr(workers_geometry.moge_model, "_load_model", lambda device: FakeMoGe())
    write_decode(tmp_path, random_frames(4, h=6, w=8))
    for batch in (2, 8):
        OmegaConf.update(cfg, "geometry_stage.moge_batch_size", batch)
        fns = build_stage_fns(cfg, root, [])[0]
        assert fns["geometry"].moge_batch_size == batch
        fns["intrinsics"]("clip0", tmp_path)
    assert seen == [2, 8]
    for bad in (0, 1.5, True, "8"):
        with pytest.raises(ValueError, match="moge_batch_size"):
            geometry_moge_batch_size(OmegaConf.create({"geometry_stage": {"moge_batch_size": bad}}))
