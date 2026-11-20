"""Tests for the HaWoR npz hand loader, against a synthetic npz fixture."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fastdo.hands.hawor_npz import fingertip_indices, load_hawor_npz

_T = 5


def _make_npz(tmp_path: Path, **overrides: np.ndarray) -> Path:
    rng = np.random.default_rng(0)
    data = {}
    for side in ("left", "right"):
        data[f"{side}_vertices"] = rng.standard_normal((_T, 778, 3)).astype(np.float32)
        data[f"{side}_faces"] = np.zeros((1552, 3), dtype=np.int32)
        data[f"{side}_joints"] = rng.standard_normal((_T, 21, 3)).astype(np.float32)
        data[f"{side}_trans"] = rng.standard_normal((_T, 3)).astype(np.float32)
        data[f"{side}_rot"] = rng.standard_normal((_T, 3)).astype(np.float32)
        data[f"{side}_hand_pose"] = rng.standard_normal((_T, 45)).astype(np.float32)
        data[f"{side}_betas"] = np.tile(np.arange(10, dtype=np.float32) * (1 if side == "right" else -1), (_T, 1))
        data[f"{side}_valid"] = np.ones(_T, dtype=bool)
    data.update(overrides)
    path = tmp_path / "all_hand_meshes.npz"
    np.savez(path, **data)
    return path


def test_params_concat_and_betas_mean(tmp_path: Path) -> None:
    path = _make_npz(tmp_path)
    with np.load(path) as raw:
        expected_params = np.concatenate([raw["right_rot"], raw["right_hand_pose"]], axis=1)

    track = load_hawor_npz(path, side="right")

    assert track.params.shape == (_T, 48)
    np.testing.assert_allclose(track.params, expected_params, atol=1e-6)
    # betas are constant 0..9 per frame for "right" -> mean is 0..9 unchanged.
    np.testing.assert_allclose(track.betas, np.arange(10, dtype=np.float32), atol=1e-6)
    assert track.trans.shape == (_T, 3) and track.joints.shape == (_T, 21, 3)
    assert bool(track.valid.all())


def test_left_side_uses_left_arrays(tmp_path: Path) -> None:
    path = _make_npz(tmp_path)
    track = load_hawor_npz(path, side="left")
    # left betas were tiled as -1 * arange(10).
    np.testing.assert_allclose(track.betas, -np.arange(10, dtype=np.float32), atol=1e-6)


def test_invalid_side_raises(tmp_path: Path) -> None:
    path = _make_npz(tmp_path)
    with pytest.raises(ValueError, match="side"):
        load_hawor_npz(path, side="both")


def test_shape_mismatch_raises(tmp_path: Path) -> None:
    bad_pose = np.zeros((_T, 44), dtype=np.float32)  # wrong last dim (44 instead of 45)
    path = _make_npz(tmp_path, right_hand_pose=bad_pose)
    with pytest.raises(ValueError, match="right_hand_pose"):
        load_hawor_npz(path, side="right")


def test_valid_combines_finite_and_provided_flag(tmp_path: Path) -> None:
    trans = np.zeros((_T, 3), dtype=np.float32)
    trans[2] = np.nan  # frame 2: non-finite -> invalid regardless of the provided flag
    provided_valid = np.ones(_T, dtype=bool)
    provided_valid[4] = False  # frame 4: finite but explicitly flagged invalid
    path = _make_npz(tmp_path, right_trans=trans, right_valid=provided_valid)

    track = load_hawor_npz(path, side="right")

    expected = np.array([True, True, False, True, False])
    np.testing.assert_array_equal(track.valid, expected)


def test_valid_falls_back_to_finite_only_without_valid_key(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    path = tmp_path / "no_valid.npz"
    data = {
        "right_rot": rng.standard_normal((_T, 3)).astype(np.float32),
        "right_hand_pose": rng.standard_normal((_T, 45)).astype(np.float32),
        "right_trans": rng.standard_normal((_T, 3)).astype(np.float32),
        "right_joints": rng.standard_normal((_T, 21, 3)).astype(np.float32),
        "right_betas": rng.standard_normal((_T, 10)).astype(np.float32),
    }
    data["right_trans"][1] = np.inf
    np.savez(path, **data)

    track = load_hawor_npz(path, side="right")

    assert track.valid.tolist() == [True, False, True, True, True]


def test_fingertip_indices() -> None:
    idx = fingertip_indices()
    assert idx.tolist() == [745, 317, 444, 556, 673]
    assert idx.dtype.__str__() == "torch.int64"
