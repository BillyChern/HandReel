"""Tests for `fastdo.qc.rule_io`: the versioned JSON form of a fitted `AcceptanceRule`."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from fastdo.qc import AcceptanceRule, ConformalCalibration, QcFeatures, accept
from fastdo.qc.rule_io import (
    ACCEPTANCE_RULE_FORMAT,
    ACCEPTANCE_RULE_VERSION,
    FEATURE_NAMES,
    acceptance_rule_from_dict,
    acceptance_rule_sha256,
    acceptance_rule_to_dict,
    load_acceptance_rule,
    read_acceptance_rule_provenance,
    save_acceptance_rule,
)

_ARRAYS = ("logistic_weights", "feature_mean", "feature_std")


def _rule(seed: int = 3) -> AcceptanceRule:
    rng = np.random.default_rng(seed)
    weights, mean, std = rng.normal(size=6), rng.normal(size=5), rng.uniform(0.5, 2.0, size=5)
    return AcceptanceRule(calibration=ConformalCalibration(threshold=0.12345678901234567, alpha=0.1, n_calibration=57),
                          logistic_weights=weights, feature_mean=mean, feature_std=std, tau=0.02)


def _write(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload))            # json's default writes NaN as the literal `NaN`
    return path


def test_acceptance_rule_json_roundtrip_is_bit_exact(tmp_path: Path) -> None:
    rule, path = _rule(), tmp_path / "qc" / "acceptance_rule.json"

    save_acceptance_rule(path, rule, provenance={"sequences": ["a/b/c"]})
    loaded = load_acceptance_rule(path)

    for name in _ARRAYS:
        original, restored = getattr(rule, name), getattr(loaded, name)
        assert restored.dtype == np.float64 and original.tobytes() == restored.tobytes(), name
        assert not restored.flags.writeable, name
    assert loaded.calibration.threshold == rule.calibration.threshold
    assert loaded.calibration.alpha == rule.calibration.alpha and loaded.tau == rule.tau
    assert type(loaded.calibration.n_calibration) is int and loaded.calibration.n_calibration == 57
    rng = np.random.default_rng(4)
    for _ in range(20):
        features = QcFeatures(*rng.normal(size=5).tolist())
        assert accept(loaded, features) == accept(rule, features)
    payload = json.loads(path.read_text())
    assert (payload["format"], payload["version"], payload["feature_names"]) == \
        ("fastdo.qc.AcceptanceRule", 1, list(FEATURE_NAMES))
    assert (ACCEPTANCE_RULE_FORMAT, ACCEPTANCE_RULE_VERSION) == ("fastdo.qc.AcceptanceRule", 1)
    assert FEATURE_NAMES == tuple(QcFeatures.__dataclass_fields__)      # QcFeatures field order
    assert read_acceptance_rule_provenance(path) == {"sequences": ["a/b/c"]}
    assert [p.name for p in path.parent.iterdir()] == ["acceptance_rule.json"]          # no .tmp sibling left
    assert re.fullmatch(r"[0-9a-f]{64}", acceptance_rule_sha256(path))
    in_memory = acceptance_rule_from_dict(acceptance_rule_to_dict(rule))
    assert in_memory.logistic_weights.tobytes() == rule.logistic_weights.tobytes()


def test_load_acceptance_rule_fails_loudly(tmp_path: Path) -> None:
    missing = tmp_path / "absent.json"
    with pytest.raises(FileNotFoundError, match=re.escape(str(missing))):
        load_acceptance_rule(missing)
    good = acceptance_rule_to_dict(_rule())
    names = list(good["feature_names"])
    with pytest.raises(ValueError, match="format"):
        load_acceptance_rule(_write(tmp_path / "format.json", {**good, "format": "other"}))
    with pytest.raises(ValueError, match="version"):
        load_acceptance_rule(_write(tmp_path / "version.json", {**good, "version": 2}))
    with pytest.raises(ValueError, match="feature"):
        load_acceptance_rule(_write(tmp_path / "names.json", {**good, "feature_names": [names[1], names[0], *names[2:]]}))
    with pytest.raises(ValueError):
        load_acceptance_rule(_write(tmp_path / "weights.json", {**good, "logistic_weights": good["logistic_weights"][:5]}))
    with pytest.raises(ValueError):
        load_acceptance_rule(_write(tmp_path / "nan.json", {**good, "feature_mean": [float("nan"), *good["feature_mean"][1:]]}))
    assert "NaN" in (tmp_path / "nan.json").read_text()
    with pytest.raises(ValueError):
        load_acceptance_rule(_write(tmp_path / "extra.json", {**good, "notes": "x"}))
    with pytest.raises(ValueError):
        load_acceptance_rule(_write(tmp_path / "missing_key.json", {k: v for k, v in good.items() if k != "tau"}))
    with pytest.raises(ValueError):
        load_acceptance_rule(_write(tmp_path / "std.json", {**good, "feature_std": [0.0, *good["feature_std"][1:]]}))
    with pytest.raises(ValueError):
        load_acceptance_rule(_write(tmp_path / "text.json", {**good, "tau": "0.02"}))


def test_save_acceptance_rule_is_atomic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path, rule_a = tmp_path / "rule.json", _rule(3)
    save_acceptance_rule(path, rule_a)

    def failing_replace(src: Any, dst: Any) -> None:
        raise OSError("simulated failure in os.replace")

    monkeypatch.setattr(os, "replace", failing_replace)
    with pytest.raises(OSError, match="simulated failure"):
        save_acceptance_rule(path, _rule(5))
    monkeypatch.undo()

    assert load_acceptance_rule(path).logistic_weights.tobytes() == rule_a.logistic_weights.tobytes()
    assert [p.name for p in tmp_path.iterdir()] == ["rule.json"]
