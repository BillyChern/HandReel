"""Tests for `fastdo.geometry.intrinsics` (D3 amendments): camera matrices, MoGe-2 normalized intrinsics to
pixel intrinsics, and the `intrinsics.npz` artifact."""

from __future__ import annotations

import json
from pathlib import Path
from typing import BinaryIO

import numpy as np
import pytest

from fastdo.geometry.intrinsics import (
    INTRINSICS_FILE,
    INTRINSICS_SOURCES,
    camera_matrix,
    pixel_intrinsics,
    read_camera_json,
    read_intrinsics,
    write_intrinsics,
)


def test_camera_matrix_layout_and_dtype() -> None:
    K = camera_matrix(500.0, 510.0, 320.5, 240.25)
    assert K.dtype == np.float32 and K.shape == (3, 3)
    np.testing.assert_array_equal(K, np.array([[500.0, 0, 320.5], [0, 510.0, 240.25], [0, 0, 1]], np.float32))


def test_pixel_intrinsics_median() -> None:
    fx_w, fy_h = [0.78, 0.80, 0.79], [1.41, 1.38, 1.40]          # deliberately unsorted
    normalized = np.zeros((3, 3, 3), np.float32)
    normalized[:, 0, 0], normalized[:, 1, 1] = fx_w, fy_h
    normalized[:, 0, 2] = normalized[:, 1, 2] = 0.5
    normalized[:, 2, 2] = 1.0

    K = pixel_intrinsics(normalized, width=1280, height=720)

    assert K.dtype == np.float32 and K.shape == (3, 3)
    assert K[0, 0] == pytest.approx(1011.2, abs=1e-3)
    assert K[1, 1] == pytest.approx(1008.0, abs=1e-3)
    assert K[0, 2] == pytest.approx(640.0, abs=1e-3)
    assert K[1, 2] == pytest.approx(360.0, abs=1e-3)
    assert K[0, 1] == 0.0 and K[2, 2] == 1.0


def test_pixel_intrinsics_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="N,3,3"):
        pixel_intrinsics(np.zeros((3, 3), np.float32), width=64, height=48)


def test_read_camera_json(tmp_path: Path) -> None:
    path = tmp_path / "camera.json"
    path.write_text(json.dumps({"fx": 999.0292358398438, "fy": 999.0292167663574, "cx": 640, "cy": 360}))
    np.testing.assert_array_equal(read_camera_json(path),
                                  camera_matrix(999.0292358398438, 999.0292167663574, 640.0, 360.0))


def test_intrinsics_npz_roundtrip_loads_without_pickle(tmp_path: Path) -> None:
    K = camera_matrix(1011.2, 1008.0, 640.0, 360.0)
    write_intrinsics(tmp_path, K, "moge2")

    with np.load(tmp_path / INTRINSICS_FILE, allow_pickle=False) as data:
        assert set(data.files) == {"K", "source"}
        assert data["K"].dtype == np.float32 and data["K"].shape == (3, 3)
        assert data["source"].shape == () and data["source"].dtype.kind == "U"
    K_read, source = read_intrinsics(tmp_path)
    np.testing.assert_array_equal(K_read, K)
    assert source == "moge2" and source in INTRINSICS_SOURCES
    assert not list(tmp_path.glob("*.tmp*"))                         # atomic write leaves no debris


def test_read_intrinsics_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_intrinsics(tmp_path)


def test_write_intrinsics_validates_source_and_shape(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="source"):
        write_intrinsics(tmp_path, camera_matrix(1.0, 1.0, 0.0, 0.0), "guess")
    with pytest.raises(ValueError, match="3,3"):
        write_intrinsics(tmp_path, np.eye(4, dtype=np.float32), "given")


class _Interrupted(BaseException):
    """Neither an OSError nor an Exception, like KeyboardInterrupt."""


def test_write_intrinsics_removes_its_tmp_file_on_any_exception(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def interrupted_savez(file: BinaryIO, **arrays: np.ndarray) -> None:
        file.write(b"PK partial archive")
        raise _Interrupted

    monkeypatch.setattr(np, "savez", interrupted_savez)
    with pytest.raises(_Interrupted):
        write_intrinsics(tmp_path, camera_matrix(1011.2, 1008.0, 640.0, 360.0), "moge2")
    assert not list(tmp_path.iterdir())                               # neither intrinsics.npz nor its .tmp
