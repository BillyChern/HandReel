"""Tests for `fastdo.gate.features`: uniform row sampling, gate features and the encoder-archive reader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from fastdo.gate.features import N_GATE_FRAMES, gate_features, read_gate_tokens, sample_rows
from tests.gate.fakes import write_encoder_npz


def test_sample_rows_uniform_with_repeats() -> None:
    assert N_GATE_FRAMES == 8
    assert sample_rows(100).tolist() == [0, 14, 28, 42, 57, 71, 85, 99]
    assert sample_rows(20).tolist() == [0, 3, 5, 8, 11, 14, 16, 19]
    assert sample_rows(5).tolist() == [0, 1, 1, 2, 2, 3, 3, 4]
    assert sample_rows(1).tolist() == [0] * 8
    assert sample_rows(20).dtype == np.int64
    with pytest.raises(ValueError):
        sample_rows(0)


def test_gate_features_shape() -> None:
    tokens = torch.randn(20, 384, generator=torch.Generator().manual_seed(0))
    hand_conf = torch.linspace(0, 1, 20)

    out = gate_features(tokens, hand_conf)

    rows = torch.from_numpy(sample_rows(20))
    assert out.shape == (8, 385) and out.dtype == torch.float32
    assert torch.equal(out[:, :384], tokens[rows])
    assert torch.equal(out[:, 384], hand_conf[rows])
    assert torch.equal(gate_features(tokens)[:, 384], torch.zeros(8))


def test_gate_features_rejects_bad_inputs() -> None:
    tokens = torch.randn(20, 384)
    with pytest.raises(ValueError):
        gate_features(torch.randn(20))
    with pytest.raises(ValueError):
        gate_features(torch.randn(0, 384))
    with pytest.raises(ValueError):
        gate_features(tokens, torch.zeros(19))
    bad_conf = torch.zeros(20)
    bad_conf[3] = 1.5
    with pytest.raises(ValueError):
        gate_features(tokens, bad_conf)
    nan_tokens = tokens.clone()
    nan_tokens[7, 11] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        gate_features(nan_tokens)


def test_read_gate_tokens_crop_and_rows(tmp_path: Path) -> None:
    write_encoder_npz(tmp_path / "d3", np.ones((30, 16)), np.arange(0, 90, 3), crop=None)
    tokens, frame_idx, crop = read_gate_tokens(tmp_path / "d3")
    assert crop == "center" and tokens.shape == (30, 16) and tokens.dtype == torch.float32
    assert frame_idx.tolist() == list(range(0, 90, 3))

    write_encoder_npz(tmp_path / "lb", np.ones((30, 16)), np.arange(30), crop="letterbox")
    assert read_gate_tokens(tmp_path / "lb")[2] == "letterbox"

    write_encoder_npz(tmp_path / "bad", np.ones((30, 16)), np.arange(29), crop="center")
    with pytest.raises(ValueError):
        read_gate_tokens(tmp_path / "bad")
    with pytest.raises(FileNotFoundError, match="run the 'encoder' stage"):
        read_gate_tokens(tmp_path / "missing")
