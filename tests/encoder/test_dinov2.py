"""Tests for `fastdo.encoder.dinov2` (D3): pooled CLS + mean-patch descriptors under bf16, with a fake model."""

from __future__ import annotations

import sys

import pytest
import torch

from fastdo.encoder import encode_crops, load_dinov2
from fastdo.encoder.dinov2 import IMAGENET_MEAN, IMAGENET_STD, MODEL_ID
from tests.engine.fakes import FakeDinov2


def test_encode_crops_pools_cls_and_mean_patch_tokens_under_bf16() -> None:
    crops = torch.randint(0, 256, (5, 3, 14, 14), dtype=torch.uint8)
    model = FakeDinov2()

    out = encode_crops(model, crops, batch_size=2)

    assert out.shape == (5, 2 * FakeDinov2.hidden) and out.dtype == torch.float32
    assert model.batch_sizes == [2, 2, 1]
    assert all(model.autocast_bf16)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    normalized = (crops.float() / 255.0 - mean) / std
    torch.testing.assert_close(torch.cat(model.inputs), normalized)
    # The fake's token t holds mean(input) + t: CLS (t = 0) is the mean, the patch mean (t = 1..3) is mean + 2.
    m = normalized.mean(dim=(1, 2, 3))[:, None]
    expected = torch.cat([m.expand(5, FakeDinov2.hidden), (m + 2.0).expand(5, FakeDinov2.hidden)], dim=1)
    torch.testing.assert_close(out, expected)


def test_encode_crops_rejects_non_square_crops_and_bad_batch_size() -> None:
    with pytest.raises(ValueError, match="N,3,S,S"):
        encode_crops(FakeDinov2(), torch.zeros(2, 3, 14, 12))
    with pytest.raises(ValueError, match="batch_size"):
        encode_crops(FakeDinov2(), torch.zeros(2, 3, 14, 14), batch_size=0)


def test_load_dinov2_without_transformers_raises_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    assert MODEL_ID == "facebook/dinov2-large"
    monkeypatch.setitem(sys.modules, "transformers", None)             # makes `import transformers` fail
    with pytest.raises(RuntimeError, match="transformers"):
        load_dinov2("cpu")
