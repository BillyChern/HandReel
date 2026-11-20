"""Tests for the letterbox crop of the encoder stage (`fastdo.engine.workers_frames`): fakes only, no weights."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from fastdo.encoder.dinov2 import encode_crops
from fastdo.engine.workers_frames import (
    CROP_MODES,
    center_crops,
    letterbox_crops,
    make_encode_worker,
)
from tests.engine.fakes import FakeDinov2, random_frames, write_decode


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def test_letterbox_keeps_the_frame_edges() -> None:
    frames = torch.zeros(1, 36, 64, 3, dtype=torch.uint8)
    frames[:, :, 0] = 255

    letterbox = letterbox_crops(frames, 32)                   # content rows 7-24, padding (124, 116, 104) elsewhere
    assert bool((letterbox[:, :, 7:25, 0] > 100).all()) and bool((letterbox[:, :, 7:25, 1:] < 100).all())
    assert not bool((center_crops(frames, 32) > 100).any())


def test_letterbox_pads_with_imagenet_mean() -> None:
    out = letterbox_crops(torch.full((1, 36, 64, 3), 50, dtype=torch.uint8), 32)

    assert out.shape == (1, 3, 32, 32) and out.dtype == torch.uint8
    assert bool((out[:, :, 7:25] == 50).all())
    for row in (0, 6, 25, 31):
        assert out[0, :, row, :].tolist() == [[124] * 32, [116] * 32, [104] * 32]


def test_encode_worker_records_crop(tmp_path: Path) -> None:
    frames = random_frames(5, h=24, w=32)
    write_decode(tmp_path / "lb", frames)
    write_decode(tmp_path / "center", frames)

    make_encode_worker(device="cpu", crop_size=14, model=FakeDinov2(), crop="letterbox")("lb", tmp_path / "lb")
    make_encode_worker(device="cpu", crop_size=14, model=FakeDinov2())("center", tmp_path / "center")

    letterbox, center = _load(tmp_path / "lb" / "encoder.npz"), _load(tmp_path / "center" / "encoder.npz")
    assert CROP_MODES == ("center", "letterbox")
    assert letterbox["crop"].shape == () and str(letterbox["crop"]) == "letterbox"
    assert str(center["crop"]) == "center" and center["frame_idx"].tolist() == [0, 3, 6, 9, 12]
    frames_t = torch.from_numpy(frames)
    expected_center = encode_crops(FakeDinov2(), center_crops(frames_t, 14)).numpy()
    np.testing.assert_allclose(center["tokens"], expected_center, rtol=0, atol=1e-6)        # as D3's own test
    expected_letterbox = encode_crops(FakeDinov2(), letterbox_crops(frames_t, 14)).numpy()
    np.testing.assert_allclose(letterbox["tokens"], expected_letterbox, rtol=0, atol=1e-6)


def test_make_encode_worker_rejects_unknown_crop() -> None:
    with pytest.raises(ValueError, match="stretch"):
        make_encode_worker(device="cpu", model=FakeDinov2(), crop="stretch")
