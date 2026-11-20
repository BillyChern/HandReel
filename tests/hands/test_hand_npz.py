"""Tests for `fastdo.hands.hand_npz`: hold rows, the 23-key npz contract, and the MANO wrist cap."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from fastdo.hands.hand_npz import (
    NPZ_SIDE_KEYS,
    PRODUCER,
    SideArrays,
    assemble_side,
    closed_faces,
    nearest_valid_rows,
    validate_hand_npz,
    write_hand_npz,
)
from fastdo.hands.hawor_npz import load_hawor_npz
from tests.hands.hawor_env import HAWOR_PYTHON

# a unit cube, outward-oriented; the last two rows are the top (z = 1) face
CUBE = np.array([[0, 2, 1], [0, 3, 2], [0, 1, 5], [0, 5, 4], [3, 7, 6], [3, 6, 2],
                 [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5], [4, 5, 6], [4, 6, 7]], dtype=np.int64)
FACES = {"left": np.zeros((1552, 3), np.int32), "right": np.zeros((1552, 3), np.int32)}


def _rows(t: int, valid: np.ndarray, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {"rot": rng.standard_normal((t, 3)).astype(np.float32),
            "hand_pose": rng.standard_normal((t, 45)).astype(np.float32),
            "trans": rng.standard_normal((t, 3)).astype(np.float32),
            "betas": rng.standard_normal((t, 10)).astype(np.float32), "valid": valid.astype(bool)}


def _side(t: int, valid: np.ndarray, seed: int = 0) -> SideArrays:
    rng = np.random.default_rng(seed + 100)
    box = np.where(valid[:, None], rng.uniform(0, 500, (t, 5)), np.nan)
    return assemble_side(_rows(t, valid, seed), rng.standard_normal((t, 778, 3)).astype(np.float32),
                         rng.standard_normal((t, 21, 3)).astype(np.float32), np.zeros(t, bool), box)


def test_nearest_valid_rows() -> None:
    valid = np.array([False, False, True, False, False, False, True, False])
    assert nearest_valid_rows(valid).tolist() == [2, 2, 2, 2, 2, 6, 6, 6]
    assert nearest_valid_rows(valid).dtype == np.int64
    assert nearest_valid_rows(np.zeros(4, bool)).tolist() == [-1, -1, -1, -1]


def test_assemble_side_hold_rows_and_betas() -> None:
    valid = np.array([False, True, True, False, False, True])
    rows = _rows(6, valid)
    rng = np.random.default_rng(3)
    vertices = rng.standard_normal((6, 778, 3)).astype(np.float32)
    joints = rng.standard_normal((6, 21, 3)).astype(np.float32)

    side = assemble_side(rows, vertices, joints, np.zeros(6, bool), np.full((6, 5), np.nan))

    for dst, src in ((0, 1), (3, 2), (4, 5)):
        for name, source in (("trans", rows["trans"]), ("rot", rows["rot"]), ("hand_pose", rows["hand_pose"]),
                             ("vertices", vertices), ("joints", joints)):
            np.testing.assert_array_equal(getattr(side, name)[dst], source[src])
    mean = rows["betas"][[1, 2, 5]].mean(axis=0)
    for row in (0, 3, 4):
        np.testing.assert_allclose(side.betas[row], mean, atol=1e-6)
    np.testing.assert_array_equal(side.valid, valid)
    np.testing.assert_array_equal(side.box, np.zeros((6, 5), np.float32))


def test_write_hand_npz_keys_dtypes_and_loader(tmp_path: Path) -> None:
    t = 46
    left_valid = np.ones(t, bool)
    left_valid[41:43] = False
    frame_idx = np.arange(t, dtype=np.int64) * 3
    left, right = _side(t, left_valid, seed=0), _side(t, np.ones(t, bool), seed=1)
    path = tmp_path / "hands_hawor.npz"

    write_hand_npz(path, frame_idx, 10.0, left, right, FACES)

    assert validate_hand_npz(path) == t
    with np.load(path, allow_pickle=False) as data:
        expected = {f"{s}_{k}" for s in ("left", "right") for k in NPZ_SIDE_KEYS} | {"frame_idx", "fps", "producer"}
        assert set(data.files) == expected and len(expected) == 23
        assert all(np.isfinite(data[k]).all() for k in data.files if data[k].dtype.kind == "f")
        assert str(data["producer"]) == PRODUCER and data["fps"].dtype == np.float64
    for name, arrays, mask in (("left", left, left_valid), ("right", right, np.ones(t, bool))):
        track = load_hawor_npz(path, name)
        track.validate(t, f"{name}_hand")
        np.testing.assert_allclose(track.params, np.concatenate([arrays.rot, arrays.hand_pose], 1), atol=1e-6)
        np.testing.assert_array_equal(track.valid, mask)
        np.testing.assert_allclose(track.betas, arrays.betas[mask].mean(axis=0), atol=1e-6)
    assert not list(tmp_path.glob("*.tmp*"))


def _replace(side: SideArrays, **fields: np.ndarray) -> SideArrays:
    from dataclasses import replace

    return replace(side, **fields)


def test_write_hand_npz_rejects_non_finite(tmp_path: Path) -> None:
    t = 46
    left = _side(t, np.ones(t, bool))
    trans = left.trans.copy()
    trans[5, 1] = np.nan
    path = tmp_path / "bad.npz"
    with pytest.raises(ValueError, match="left_trans"):
        write_hand_npz(path, np.arange(t, dtype=np.int64), 10.0, _replace(left, trans=trans), _side(t, np.ones(t, bool)),
                       FACES)
    assert not path.exists() and not list(tmp_path.iterdir())


def test_write_hand_npz_rejects_bad_shape(tmp_path: Path) -> None:
    t = 46
    left = _side(t, np.ones(t, bool))
    path = tmp_path / "bad.npz"
    with pytest.raises(ValueError, match="left_vertices"):
        write_hand_npz(path, np.arange(t, dtype=np.int64), 10.0,
                       _replace(left, vertices=np.zeros((t, 777, 3), np.float32)), _side(t, np.ones(t, bool)), FACES)
    assert not path.exists()


def test_write_hand_npz_side_without_detections(tmp_path: Path) -> None:
    t = 46
    right = _side(t, np.zeros(t, bool), seed=4)
    for name in ("vertices", "joints", "trans", "rot", "hand_pose", "betas", "box"):
        assert not getattr(right, name).any(), name
    path = tmp_path / "hands_hawor.npz"

    write_hand_npz(path, np.arange(t, dtype=np.int64), 10.0, _side(t, np.ones(t, bool)), right, FACES)

    track = load_hawor_npz(path, "right")
    assert not track.valid.any()
    assert np.isfinite(track.betas).all() and not track.betas.any()


def test_validate_hand_npz_rejects_missing_key_and_unsorted_frames(tmp_path: Path) -> None:
    t = 8
    path = tmp_path / "hands_hawor.npz"
    write_hand_npz(path, np.arange(t, dtype=np.int64), 10.0, _side(t, np.ones(t, bool)), _side(t, np.ones(t, bool)),
                   FACES)
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}

    missing = tmp_path / "missing.npz"
    np.savez(missing, **{k: v for k, v in arrays.items() if k != "right_box"})
    with pytest.raises(ValueError, match="right_box"):
        validate_hand_npz(missing)
    unsorted = tmp_path / "unsorted.npz"
    np.savez(unsorted, **(arrays | {"frame_idx": arrays["frame_idx"][::-1].copy()}))
    with pytest.raises(ValueError, match="frame_idx"):
        validate_hand_npz(unsorted)


def _directed_edges(faces: np.ndarray) -> list[tuple[int, int]]:
    return [(int(f[i]), int(f[(i + 1) % 3])) for f in faces for i in range(3)]


def test_closed_faces_caps_single_boundary_loop() -> None:
    faces = closed_faces(CUBE[:10])

    right, left = faces["right"], faces["left"]
    assert right.shape == (12, 3) and right.dtype == np.int32
    np.testing.assert_array_equal(right[:10], CUBE[:10])
    edges = _directed_edges(right)
    assert len(edges) == len(set(edges))
    assert all((j, i) in set(edges) for i, j in edges)
    assert set(right[10:, 0].tolist()) == {4}                        # the loop's smallest vertex
    np.testing.assert_array_equal(left, right[:, [0, 2, 1]])


def test_closed_faces_rejects_two_loops() -> None:
    with pytest.raises(ValueError, match="loop"):
        closed_faces(CUBE[2:10])


@pytest.mark.skipif(not HAWOR_PYTHON.is_file(), reason="hawor python not available")
def test_shared_files_import_under_hawor_python(tmp_path: Path) -> None:
    from fastdo.hands.hawor_batched import DRIVER_SCRIPT, driver_env

    env = driver_env(os.environ) | {"CUDA_VISIBLE_DEVICES": ""}
    proc = subprocess.run([str(HAWOR_PYTHON), str(DRIVER_SCRIPT), "--import-check"], capture_output=True, text=True,
                          env=env, cwd=tmp_path, timeout=300, check=False)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    assert "import-check ok" in proc.stdout
