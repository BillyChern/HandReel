"""Tests for the qc stage built from config: a persisted rule, and a loud failure without one."""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from fastdo.engine.workers_qc import (
    QcStageConfig,
    compute_qc_features,
    make_qc_worker,
    make_qc_worker_from_config,
    read_qc_inputs,
)
from fastdo.qc import AcceptanceRule, ConformalCalibration, QcFeatures
from fastdo.qc.rule_io import save_acceptance_rule
from tests.engine.fakes_qc import write_solved_clip
from tests.engine.test_workers_qc import PASSING

_ROOT_CONFIG = Path(__file__).resolve().parents[2] / "run" / "conf" / "config.yaml"


def _rule(bias: float) -> AcceptanceRule:
    """Score `expit(-bias)` whatever the features: bias 10 accepts (4.5e-5 <= 0.5), bias -10 rejects (0.99995)."""
    return AcceptanceRule(calibration=ConformalCalibration(threshold=0.5, alpha=0.1, n_calibration=20),
                          logistic_weights=np.array([bias, 0.0, 0.0, 0.0, 0.0, 0.0]), feature_mean=np.zeros(5),
                          feature_std=np.ones(5), tau=0.02)


def _passing(*args: Any, **kwargs: Any) -> QcFeatures:
    return PASSING


class _Capture:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> QcFeatures:
        self.calls.append((args, kwargs))
        return PASSING


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if dataclasses.is_dataclass(a) and not isinstance(a, type):
        return all(_same(getattr(a, f.name), getattr(b, f.name)) for f in dataclasses.fields(a))
    return bool(a == b)


def test_qc_stage_config_from_the_hydra_block() -> None:
    root = OmegaConf.load(_ROOT_CONFIG)
    assert QcStageConfig.from_mapping(root.qc_stage) == QcStageConfig(rule_path=None, laplace_std_threshold=0.01)
    assert QcStageConfig.from_mapping({"rule_path": "r.json"}).rule_path == "r.json"


def test_qc_worker_from_config_fails_loudly_without_a_rule(tmp_path: Path) -> None:
    for rule_path in (None, ""):
        with pytest.raises(ValueError, match=r"qc_stage\.rule_path") as info:
            make_qc_worker_from_config(QcStageConfig(rule_path=rule_path))
        assert "calibrate_acceptance" in str(info.value) and "accept-all" in str(info.value)
    with pytest.raises(FileNotFoundError):
        make_qc_worker_from_config(QcStageConfig(rule_path=str(tmp_path / "absent.json")))
    assert inspect.signature(make_qc_worker).parameters["rule"].default is inspect.Parameter.empty


def test_qc_worker_from_config_loads_the_rule_and_records_it(tmp_path: Path) -> None:
    for name, bias, accepted in (("accepting", 10.0, True), ("rejecting", -10.0, False)):
        clip, path = tmp_path / name, tmp_path / f"{name}_rule.json"
        write_solved_clip(clip)
        save_acceptance_rule(path, _rule(bias))

        make_qc_worker_from_config(QcStageConfig(rule_path=str(path)), features_fn=_passing)(name, clip)

        qc = json.loads((clip / "qc.json").read_text())
        assert qc["accepted"] is accepted, name
        assert qc["rule"] == {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                              "version": 1}
        assert (clip / "clip_record.npz").exists() is accepted, name


def test_qc_worker_from_config_records_the_hash_of_the_parsed_bytes(tmp_path: Path,
                                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    """The rule file is read once, and the recorded sha256 is of the bytes that were parsed. The
    file is replaced by a rejecting rule right after every read of it; the verdict and the hash stay the first file's."""
    clip, path, other = tmp_path / "clip", tmp_path / "rule.json", tmp_path / "other.json"
    write_solved_clip(clip)
    save_acceptance_rule(path, _rule(10.0))                          # accepts
    save_acceptance_rule(other, _rule(-10.0))                        # rejects
    first, replacement = path.read_bytes(), other.read_bytes()
    reads: list[str] = []

    def replaced_after_reading(read: Any) -> Any:
        def wrapper(self: Path, *args: Any, **kwargs: Any) -> Any:
            out = read(self, *args, **kwargs)
            if self.resolve() == path.resolve():
                reads.append(read.__name__)
                path.write_bytes(replacement)
            return out

        return wrapper

    monkeypatch.setattr(Path, "read_bytes", replaced_after_reading(Path.read_bytes))
    monkeypatch.setattr(Path, "read_text", replaced_after_reading(Path.read_text))
    worker = make_qc_worker_from_config(QcStageConfig(rule_path=str(path)), features_fn=_passing)
    monkeypatch.undo()
    worker("clip", clip)

    qc = json.loads((clip / "qc.json").read_text())
    assert reads == ["read_bytes"]
    assert qc["rule"]["sha256"] == hashlib.sha256(first).hexdigest() != hashlib.sha256(replacement).hexdigest()
    assert qc["accepted"] is True and path.read_bytes() == replacement


def test_read_qc_inputs_and_compute_qc_features_are_the_workers_path(tmp_path: Path) -> None:
    clip = tmp_path / "clip0"
    write_solved_clip(clip)
    capture = _Capture()

    features = compute_qc_features(read_qc_inputs(clip), capture, 0.02)
    make_qc_worker(_rule(10.0), capture, 0.02)("clip0", clip)

    assert features == PASSING
    (direct, direct_kwargs), (in_worker, worker_kwargs) = capture.calls
    assert direct_kwargs == worker_kwargs == {"laplace_std_threshold": 0.02}
    assert len(direct) == len(in_worker) == 11
    for i, (a, b) in enumerate(zip(direct, in_worker, strict=True)):
        assert _same(a, b), i
    assert json.loads((clip / "qc.json").read_text())["rule"] is None     # make_qc_worker called directly
