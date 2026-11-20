"""The throughput bench registers the `retarget` worker only when `retarget_stage.enabled` (opt-in).

- Default config: no retarget worker, the stage keeps its NOT_REGISTERED label (a default run is unchanged, its
  'pending' column included).
- Enabled with a qc rule: a retarget worker built from `retarget_stage` (driftqc flag passed through), labelled with its
  side rule and solver options.
- Enabled without a registered qc stage: refused before anything runs (the worker reads qc.json / clip_record.npz).
- A non-bool `enabled` is refused.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from fastdo.bench import bench_stages
from fastdo.bench.bench_stages import NOT_REGISTERED, build_stage_fns, retarget_enabled
from fastdo.qc import AcceptanceRule, ConformalCalibration
from fastdo.qc.rule_io import save_acceptance_rule

_REPO = Path(__file__).resolve().parents[2]


def _cfg(*overrides: str) -> DictConfig:
    with initialize_config_dir(config_dir=str(_REPO / "run/conf"), version_base=None):
        return compose(config_name="config", overrides=[f"paths.root={_REPO}", *overrides])


def _rule(path: Path) -> Path:
    save_acceptance_rule(path, AcceptanceRule(
        calibration=ConformalCalibration(threshold=0.5, alpha=0.1, n_calibration=20),
        logistic_weights=np.zeros(6), feature_mean=np.zeros(5), feature_std=np.ones(5)))
    return path


def test_default_config_does_not_register_retarget() -> None:
    assert retarget_enabled(OmegaConf.load(_REPO / "run/conf/config.yaml")) is False
    assert retarget_enabled(OmegaConf.create({})) is False
    fns, labels = build_stage_fns(_cfg(), _REPO, [])
    assert "retarget" not in fns and labels["retarget"] == NOT_REGISTERED["retarget"]


def test_enabled_registers_the_worker_with_the_stage_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def make(cfg: Any, **kwargs: Any) -> Any:
        seen["cfg"], seen["kwargs"] = cfg, kwargs
        return lambda clip_id, artifact_dir: None

    monkeypatch.setattr(bench_stages, "make_retarget_worker", make)
    cfg = _cfg(f"qc_stage.rule_path={_rule(tmp_path / 'rule.json')}", "retarget_stage.enabled=true",
               "retarget_stage.side_from_prompt=true", "+retarget_stage.retarget.joint_limits=projected",
               "+retarget_stage.retarget.lam_tip=1000.0")
    fns, labels = build_stage_fns(cfg, _REPO, [])
    assert callable(fns["retarget"]) and seen["kwargs"] == {"driftqc_enabled": False}
    assert seen["cfg"].side_from_prompt is True and seen["cfg"].retarget.joint_limits == "projected"
    assert labels["retarget"] == ("computed (kinematic reference, CPU; side from prompt, joint_limits projected, "
                                  "lam_tip 1000)")


def test_enabled_without_qc_is_refused() -> None:
    with pytest.raises(ValueError, match="qc stage is not registered"):
        build_stage_fns(_cfg("retarget_stage.enabled=true"), _REPO, [])


def test_a_non_bool_enabled_is_refused() -> None:
    with pytest.raises(ValueError, match="retarget_stage.enabled"):
        retarget_enabled(OmegaConf.create({"retarget_stage": {"enabled": "yes"}}))
