"""Tests for `fastdo.gate.bundle`: weights-only gate bundles with calibration, crop and card."""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path

import pytest
import torch

from fastdo.gate.bundle import GATE_BUNDLE_VERSION, load_gate_bundle, save_gate_bundle
from fastdo.gate.calibrate import GateCalibration
from fastdo.gate.model import GateConfig, InteractionGate

_CAL = GateCalibration(threshold=0.25, alpha=0.05, k=2, n_pos=41, n_neg=0, guaranteed_recall=1 - 2 / 42,
                       empirical_recall=40 / 41, empirical_fpr=None)


class _Custom:
    pass


def _model() -> InteractionGate:
    torch.manual_seed(0)
    return InteractionGate(GateConfig(token_dim=16, hidden=8)).eval()


def test_gate_bundle_roundtrip(tmp_path: Path) -> None:
    model, path = _model(), tmp_path / "gate_bundle.pt"
    card = {"promoted": True, "val_auroc": 0.97, "counts": {"fit": 10}, "reasons": []}

    sha = save_gate_bundle(path, model, _CAL, "letterbox", card)
    bundle = load_gate_bundle(path)

    x = torch.randn(3, 8, 17)
    with torch.no_grad():
        assert all(torch.equal(a, b) for a, b in zip(model(x), bundle.model(x), strict=True))
    assert dataclasses.asdict(bundle.calibration) == dataclasses.asdict(_CAL)
    assert bundle.crop == "letterbox" and bundle.card == card and bundle.hand_conf_source == "none"
    assert bundle.config == model.cfg and not bundle.model.training
    assert bundle.sha256 == hashlib.sha256(path.read_bytes()).hexdigest() == sha
    assert GATE_BUNDLE_VERSION == 1 and not list(tmp_path.glob("*.tmp"))


def test_gate_bundle_rejects_bad_version_and_pickled_objects(tmp_path: Path) -> None:
    path = tmp_path / "gate_bundle.pt"
    save_gate_bundle(path, _model(), _CAL, "center", {})
    payload = torch.load(path, weights_only=True)
    payload["version"] = 2
    torch.save(payload, tmp_path / "v2.pt")
    with pytest.raises(ValueError, match="version"):
        load_gate_bundle(tmp_path / "v2.pt")
    payload["version"] = 1
    payload["card"] = {"object": _Custom()}
    torch.save(payload, tmp_path / "pickled.pt")
    with pytest.raises(ValueError, match="weights-only"):
        load_gate_bundle(tmp_path / "pickled.pt")


def test_gate_bundle_rejects_unknown_crop(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="stretch"):
        save_gate_bundle(tmp_path / "gate_bundle.pt", _model(), _CAL, "stretch", {})
    assert not (tmp_path / "gate_bundle.pt").exists()
