"""Tests for the hands stage worker and its `hands.npz` artifact.

`hands.npz` rows are exactly `decode.npz` rows on every path, selected from the source npz by source
frame index, so the hand T equals the solution T downstream (`ClipRecord.validate`, the retarget worker).
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import numpy as np
import pytest

from fastdo.engine.workers import hands_worker, make_hands_worker, write_hands_artifact
from fastdo.geometry.intrinsics import camera_matrix, read_intrinsics, write_intrinsics
from fastdo.hands.hand_camera import CAMERA_KEY, CAMERA_KEYS, HAWOR_CAMERA_KEY
from fastdo.hands.hawor_batched import HandsCallReport, HandsClipInput, HandsClipResult, HandsConfig
from tests.baseline_paths import baseline_path
from tests.engine.fakes import make_hawor_npz, random_frames, write_decode
from tests.hands.fake_hawor_driver import write_valid_hand_npz

HANDS_CFG = HandsConfig(hawor_python=Path("/fake/python"), hawor_root=Path("/fake/HaWoR"))

WHISKING_HANDS = baseline_path("reconstruction/whisking/whisking/all_hand_meshes.npz")
_SIDES = ("left", "right")
_SIDE_KEYS = ("params", "betas", "trans", "joints", "valid", "vertices", "faces", "infilled")


def _hands_keys() -> set[str]:
    return {f"{side}_{key}" for side in _SIDES for key in _SIDE_KEYS} | {"frame_idx"} | set(CAMERA_KEYS)


def _assert_pinned_layout(data: np.lib.npyio.NpzFile, t: int) -> None:
    assert set(data.files) == _hands_keys()
    pinned = {"params": ((t, 48), np.float32), "betas": ((10,), np.float32), "trans": ((t, 3), np.float32),
              "joints": ((t, 21, 3), np.float32), "valid": ((t,), np.bool_), "vertices": ((t, 778, 3), np.float32),
              "faces": ((1552, 3), np.int32), "infilled": ((t,), np.bool_)}
    for side in _SIDES:
        for key, (shape, dtype) in pinned.items():
            arr = data[f"{side}_{key}"]
            assert arr.shape == shape and arr.dtype == dtype, f"{side}_{key}: {arr.shape} {arr.dtype}"
    assert data["frame_idx"].shape == (t,) and data["frame_idx"].dtype == np.int64
    for key in CAMERA_KEYS:                                              # the camera the outputs are in
        assert data[key].shape == (3, 3) and data[key].dtype == np.float64, f"{key}: {data[key].shape} {data[key].dtype}"


class _SpyBatched:
    """A one-clip `run_hands_batched` stand-in: records its calls and writes a valid npz when `status` is done."""

    def __init__(self, status: str = "done") -> None:
        self.status = status
        self.calls: list[tuple[list[HandsClipInput], Path]] = []

    def __call__(self, clips: list[HandsClipInput], cfg: HandsConfig, job_dir: Path) -> HandsCallReport:
        self.calls.append((list(clips), Path(job_dir)))
        (clip,) = clips
        with np.load(clip.frames_path, allow_pickle=False) as data:
            frame_idx = data["frame_idx"]
        if self.status == "done":
            write_valid_hand_npz(clip.out_npz, frame_idx)
        error = "" if self.status == "done" else "detector exploded"
        result = HandsClipResult(clip.clip_id, self.status, error, len(frame_idx), {"total": 1.0})
        return HandsCallReport(0, 1.0, 0.5, 1.0, 2.0, (result,))


def _write_camera(artifact_dir: Path, fx: float) -> None:
    write_intrinsics(artifact_dir, camera_matrix(fx, fx, 16.0, 12.0), "given")


def test_hands_worker_batched_path_without_npz(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_decode(tmp_path, random_frames(3))
    _write_camera(tmp_path, 999.03)
    spy = _SpyBatched()
    monkeypatch.setattr("fastdo.engine.workers_hands.run_hands_batched", spy)

    make_hands_worker(HANDS_CFG)("clip0", tmp_path)

    assert len(spy.calls) == 1
    (clip,), job_dir = spy.calls[0]
    assert clip.focal_px == pytest.approx(999.03, abs=1e-3)
    assert clip.frames_path == tmp_path / "decode.npz" and clip.out_npz == tmp_path / "hands_hawor.npz"
    assert job_dir == tmp_path / "hands_job"
    with np.load(tmp_path / "hands.npz", allow_pickle=False) as data:
        _assert_pinned_layout(data, 3)
        assert len(data.files) == 19 and data["frame_idx"].tolist() == [0, 3, 6]
        np.testing.assert_array_equal(data[CAMERA_KEY], read_intrinsics(tmp_path)[0].astype(np.float64))


def test_hands_worker_requires_intrinsics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_decode(tmp_path, random_frames(3))
    spy = _SpyBatched()
    monkeypatch.setattr("fastdo.engine.workers_hands.run_hands_batched", spy)
    with pytest.raises(ValueError, match="intrinsics"):
        make_hands_worker(HANDS_CFG)("clip0", tmp_path)
    assert spy.calls == []


def test_hands_worker_none_config_raises(tmp_path: Path) -> None:
    write_decode(tmp_path, random_frames(3))
    with pytest.raises(RuntimeError, match="HandsConfig"):
        hands_worker("clip0", tmp_path)
    assert not (tmp_path / "hands.npz").exists()


def test_hands_worker_prefers_precomputed_npz(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_decode(tmp_path, random_frames(2))                                 # frame_idx 0, 3
    _write_camera(tmp_path, 500.0)
    make_hawor_npz(tmp_path / "all_hand_meshes.npz", t=2, frame_idx=np.arange(2) * 3)
    spy = _SpyBatched()
    monkeypatch.setattr("fastdo.engine.workers_hands.run_hands_batched", spy)

    make_hands_worker(HANDS_CFG)("clip0", tmp_path)

    assert spy.calls == [] and (tmp_path / "hands.npz").is_file()


def test_hands_worker_failed_batched_clip_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_decode(tmp_path, random_frames(3))
    _write_camera(tmp_path, 500.0)
    monkeypatch.setattr("fastdo.engine.workers_hands.run_hands_batched", _SpyBatched(status="failed"))
    with pytest.raises(RuntimeError, match="detector exploded"):
        make_hands_worker(HANDS_CFG)("clip0", tmp_path)
    assert not (tmp_path / "hands.npz").exists()


def test_hands_worker_reads_the_npz_from_sources_or_the_artifact_dir(tmp_path: Path) -> None:
    infilled = np.array([False, True, True, False, False])
    source = make_hawor_npz(tmp_path / "src" / "all_hand_meshes.npz", t=5, frame_idx=np.arange(5) * 3,
                            left_infilled=infilled)
    clip0, clip1 = tmp_path / "clip0", tmp_path / "clip1"
    write_decode(clip0, random_frames(5))                                    # frame_idx 0, 3, ..., 12
    write_decode(clip1, random_frames(2))                                    # frame_idx 0, 3
    for clip in (clip0, clip1):
        _write_camera(clip, 500.0)
    shutil.copy(source, clip1 / "all_hand_meshes.npz")

    make_hands_worker(None, sources={"clip0": source})("clip0", clip0)
    hands_worker("clip1", clip1)

    with np.load(clip0 / "hands.npz", allow_pickle=False) as data:
        _assert_pinned_layout(data, 5)
        assert data["frame_idx"].tolist() == [0, 3, 6, 9, 12]
        np.testing.assert_array_equal(data["left_infilled"], infilled)
        assert not data["right_infilled"].any()
    with np.load(clip1 / "hands.npz", allow_pickle=False) as data:
        _assert_pinned_layout(data, 2)
        np.testing.assert_array_equal(data["left_infilled"], infilled[:2])


@pytest.mark.skipif(not WHISKING_HANDS.is_file(), reason="shipped whisking hand npz not available")
def test_write_hands_artifact_whisking(tmp_path: Path) -> None:
    # The whisking clip's MoGe-2 camera has its principal point at the image centre: the shear is the identity.
    write_hands_artifact(WHISKING_HANDS, tmp_path, frame_idx=np.arange(0, 138, 3),
                         K=camera_matrix(1000.0, 1000.0, 640.0, 360.0), image_size=(1280, 720))

    path = tmp_path / "hands.npz"
    with zipfile.ZipFile(path) as archive:
        assert all(info.compress_type == zipfile.ZIP_STORED for info in archive.infolist())
    with np.load(path, allow_pickle=False) as data, np.load(WHISKING_HANDS) as raw:
        _assert_pinned_layout(data, 46)
        assert data["frame_idx"].tolist() == list(range(0, 136, 3))
        for side in _SIDES:
            params = np.concatenate([raw[f"{side}_rot"], raw[f"{side}_hand_pose"]], axis=1)
            np.testing.assert_array_equal(data[f"{side}_params"], params[::3])
            np.testing.assert_array_equal(data[f"{side}_vertices"], raw[f"{side}_vertices"][::3])
            np.testing.assert_array_equal(data[f"{side}_joints"], raw[f"{side}_joints"][::3])
            np.testing.assert_array_equal(data[f"{side}_faces"], raw[f"{side}_faces"])
            assert not data[f"{side}_infilled"].any()


def test_write_hands_artifact_reads_decode_rows(tmp_path: Path) -> None:
    right_valid = np.ones(10, bool)
    right_valid[3] = False
    source = make_hawor_npz(tmp_path / "src.npz", t=10, right_valid=right_valid, left_valid=np.zeros(10, bool))
    clip = tmp_path / "clip"
    write_decode(clip, random_frames(3))                                     # frame_idx 0, 3, 6
    _write_camera(clip, 500.0)

    write_hands_artifact(source, clip)

    rows = [0, 3, 6]
    with np.load(clip / "hands.npz", allow_pickle=False) as data, np.load(source) as raw:
        _assert_pinned_layout(data, 3)
        assert data["frame_idx"].tolist() == rows
        for side in _SIDES:
            params = np.concatenate([raw[f"{side}_rot"], raw[f"{side}_hand_pose"]], axis=1)
            np.testing.assert_array_equal(data[f"{side}_params"], params[rows])
            np.testing.assert_array_equal(data[f"{side}_trans"], raw[f"{side}_trans"][rows])
            np.testing.assert_array_equal(data[f"{side}_vertices"], raw[f"{side}_vertices"][rows])
        assert data["right_valid"].tolist() == [True, False, True] and not data["left_valid"].any()
        # betas: mean over the selected valid rows, else (no valid row) over the selected rows
        np.testing.assert_allclose(data["right_betas"], raw["right_betas"][[0, 6]].mean(axis=0), atol=1e-6)
        np.testing.assert_allclose(data["left_betas"], raw["left_betas"][rows].mean(axis=0), atol=1e-6)


def test_write_hands_artifact_selects_by_source_frame_index_not_row_position(tmp_path: Path) -> None:
    source = make_hawor_npz(tmp_path / "src.npz", t=10, frame_idx=np.arange(10, dtype=np.int64) * 3)

    write_hands_artifact(source, tmp_path / "clip", frame_idx=np.array([3, 9, 27]),
                         K=camera_matrix(500.0, 500.0, 16.0, 12.0), image_size=(32, 24))

    with np.load(tmp_path / "clip" / "hands.npz", allow_pickle=False) as data, np.load(source) as raw:
        assert data["frame_idx"].tolist() == [3, 9, 27]
        np.testing.assert_array_equal(data["right_joints"], raw["right_joints"][[1, 3, 9]])


def test_write_hands_artifact_rejects_missing_frame(tmp_path: Path) -> None:
    source = make_hawor_npz(tmp_path / "src.npz", t=10)
    clip = tmp_path / "clip"
    write_decode(clip, random_frames(5))                                     # frame_idx 0, 3, 6, 9, 12

    with pytest.raises(ValueError, match=r"frame index 12\b"):
        write_hands_artifact(source, clip)
    assert not (clip / "hands.npz").exists()


@pytest.mark.parametrize(("key", "rows"), [("right_infilled", 11), ("left_infilled", 9),
                                           ("right_betas", 11), ("left_betas", 9)])
def test_write_hands_artifact_rejects_per_row_arrays_of_another_length(tmp_path: Path, key: str, rows: int) -> None:
    per_row: np.ndarray = np.zeros(rows, bool) if key.endswith("_infilled") else np.zeros((rows, 10), np.float32)
    # `seed` is passed explicitly so that mypy matches the expanded dict only against `**overrides`.
    source = make_hawor_npz(tmp_path / "src.npz", t=10, seed=0, **{key: per_row})
    clip = tmp_path / "clip"
    write_decode(clip, random_frames(3))                                     # frame_idx 0, 3, 6: in range for all

    with pytest.raises(ValueError, match=rf"inconsistent row counts .*'{key}': {rows}\b"):
        write_hands_artifact(source, clip)
    assert not (clip / "hands.npz").exists()


def test_write_hands_artifact_accepts_per_clip_betas_and_selects_given_infilled(tmp_path: Path) -> None:
    source = make_hawor_npz(tmp_path / "src.npz", t=10, right_betas=np.full(10, 0.5, np.float32),
                            right_infilled=np.arange(10) % 2 == 1)
    clip = tmp_path / "clip"
    write_decode(clip, random_frames(3))                                     # frame_idx 0, 3, 6
    _write_camera(clip, 500.0)

    write_hands_artifact(source, clip)

    with np.load(clip / "hands.npz", allow_pickle=False) as data:
        _assert_pinned_layout(data, 3)
        np.testing.assert_array_equal(data["right_betas"], np.full(10, 0.5, np.float32))   # (10,): one per clip
        assert data["right_infilled"].tolist() == [False, True, False]
        assert not data["left_infilled"].any()


# --- HaWoR outputs in the projection camera ------------------------------------------------------------------------


def _positive_depth_source(path: Path, t: int, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    overrides: dict[str, np.ndarray] = {}
    for side in _SIDES:
        for key, shape in (("joints", (t, 21, 3)), ("vertices", (t, 778, 3)), ("trans", (t, 3))):
            points = rng.uniform(-0.2, 0.2, size=shape)
            points[..., 2] = rng.uniform(0.3, 1.5, size=shape[:-1])
            overrides[f"{side}_{key}"] = points.astype(np.float32)
    return make_hawor_npz(path, t=t, seed=seed, **overrides)


def _project(points: np.ndarray, K: np.ndarray) -> np.ndarray:
    p, k = np.asarray(points, np.float64), np.asarray(K, np.float64)
    return np.stack([k[0, 0] * p[..., 0] / p[..., 2] + k[0, 2], k[1, 1] * p[..., 1] / p[..., 2] + k[1, 2]], axis=-1)


def test_write_hands_artifact_shears_hawor_outputs_into_the_projection_camera(tmp_path: Path) -> None:
    source = _positive_depth_source(tmp_path / "src.npz", t=10)
    clip = tmp_path / "clip"
    write_decode(clip, random_frames(3, h=48, w=64))                         # frame_idx 0, 3, 6; HaWoR centre (32, 24)
    K = camera_matrix(60.0, 58.0, 25.0, 30.5)                                # a given calibration, off-centre
    write_intrinsics(clip, K, "given")

    write_hands_artifact(source, clip)

    rows = [0, 3, 6]
    hawor_K = np.array([[60.0, 0.0, 32.0], [0.0, 60.0, 24.0], [0.0, 0.0, 1.0]])
    with np.load(clip / "hands.npz", allow_pickle=False) as data, np.load(source) as raw:
        _assert_pinned_layout(data, 3)
        np.testing.assert_array_equal(data[CAMERA_KEY], K.astype(np.float64))
        np.testing.assert_array_equal(data[HAWOR_CAMERA_KEY], hawor_K)
        for side in _SIDES:
            for key in ("joints", "vertices", "trans"):
                before = raw[f"{side}_{key}"][rows].astype(np.float64)
                after = data[f"{side}_{key}"]
                assert after.dtype == np.float32
                np.testing.assert_array_equal(after[..., 2], raw[f"{side}_{key}"][rows][..., 2])      # depth
                expected_x = before[..., 0] + before[..., 2] * (32.0 - 25.0) / 60.0
                expected_y = before[..., 1] + before[..., 2] * (24.0 - 30.5) / 58.0
                np.testing.assert_allclose(after[..., 0], expected_x, rtol=0, atol=1e-6)
                np.testing.assert_allclose(after[..., 1], expected_y, rtol=0, atol=1e-6)
                # a HaWoR point's pixel (K's focals, HaWoR's principal point) is the stored point's pixel through K
                hawor_pixels = _project(before, np.array([[60.0, 0, 32.0], [0, 58.0, 24.0], [0, 0, 1]]))
                assert float(np.abs(_project(after, K) - hawor_pixels).max()) < 1e-4
            params = np.concatenate([raw[f"{side}_rot"], raw[f"{side}_hand_pose"]], axis=1)
            np.testing.assert_array_equal(data[f"{side}_params"], params[rows])                        # untouched
            np.testing.assert_array_equal(data[f"{side}_valid"], raw[f"{side}_valid"][rows])


def test_write_hands_artifact_is_a_bitwise_no_op_for_moge2_intrinsics(tmp_path: Path) -> None:
    source = _positive_depth_source(tmp_path / "src.npz", t=10, seed=1)
    clip = tmp_path / "clip"
    write_decode(clip, random_frames(3, h=48, w=64))
    write_intrinsics(clip, camera_matrix(61.0, 55.5, 32.0, 24.0), "moge2")   # principal point at the image centre

    write_hands_artifact(source, clip)

    with np.load(clip / "hands.npz", allow_pickle=False) as data, np.load(source) as raw:
        _assert_pinned_layout(data, 3)
        for side in _SIDES:
            for key in ("joints", "vertices", "trans"):
                np.testing.assert_array_equal(data[f"{side}_{key}"], raw[f"{side}_{key}"][[0, 3, 6]])


def test_write_hands_artifact_requires_the_projection_camera(tmp_path: Path) -> None:
    source = make_hawor_npz(tmp_path / "src.npz", t=10)
    clip = tmp_path / "clip"
    write_decode(clip, random_frames(3))

    with pytest.raises(FileNotFoundError, match="intrinsics"):
        write_hands_artifact(source, clip)
    assert not (clip / "hands.npz").exists()
