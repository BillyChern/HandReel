"""Tests for `run/pipeline/calibrate_acceptance.py`: the CPU rule fit from engine artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from fastdo.qc import QcFeatures
from fastdo.qc.rule_io import load_acceptance_rule, read_acceptance_rule_provenance
from run.pipeline import calibrate_acceptance as calibration
from tests.engine.fakes_qc import write_solved_clip
from tests.engine.fakes_solve import monotone_features, shifted_gt, write_template_meta

_REPO = Path(__file__).resolve().parents[2]


# This file's synthetic clips are constructed around a FIXED tau, so it pins one rather than inheriting
# `acceptance_calibration.tau` from the config. That value is a research parameter: when P2 retuned it from
# 0.02 to 0.05 to match the reported metric, every clip here became "good" and the fit refused with
# "24 good, 0 bad" -- a red suite caused by a number this test does not test.
TEST_TAU = 0.02


def _cfg(tmp_path: Path, *overrides: str) -> DictConfig:
    base = [f"paths.root={_REPO}", f"acceptance_calibration.tau={TEST_TAU}",
            f"acceptance_calibration.artifact_root={tmp_path / 'engine'}",
            f"acceptance_calibration.out_path={tmp_path / 'qc' / 'acceptance_rule.json'}",
            f"acceptance_calibration.features_out={tmp_path / 'qc' / 'features.json'}"]
    with initialize_config_dir(config_dir=str(_REPO / "run/conf"), version_base=None):
        return compose(config_name="config", overrides=[*base, *overrides])


def test_calibrate_acceptance_fits_and_saves_a_loadable_rule(tmp_path: Path) -> None:
    errors = np.linspace(0.002, 0.05, 24)                  # 9 good, 15 bad at TEST_TAU
    table: list[tuple[str, str]] = []
    gts: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for i, error in enumerate(errors):
        clip_id, seq = f"clip{i:02d}", f"s/q/c{i}"
        gts[seq] = shifted_gt(write_solved_clip(tmp_path / "engine" / clip_id, seed=i), float(error))
        write_template_meta(tmp_path / "engine" / clip_id)
        table.append((clip_id, seq))
    rows = iter(monotone_features(errors))

    def features_fn(*args: Any, **kwargs: Any) -> QcFeatures:
        return next(rows)                                   # clips are scored in table order

    # alpha 0.2: the split keeps round(4.5) = 4 good clips for the fit, 5 calibrate, conformal_rank(5, 0.2) = 5.
    cfg = _cfg(tmp_path, "acceptance_calibration.alpha=0.2")
    rule = calibration.run_calibration(cfg, _REPO, gt_poses_fn=gts.__getitem__, features_fn=features_fn,
                                       clip_table=table)

    out = tmp_path / "qc" / "acceptance_rule.json"
    assert load_acceptance_rule(out).logistic_weights.tobytes() == rule.logistic_weights.tobytes()
    provenance = read_acceptance_rule_provenance(out)
    assert (provenance["n_clips"], provenance["n_good"], provenance["n_bad"]) == (24, 9, 15)
    assert provenance["coverage_guaranteed"] is True and provenance["n_skipped"] == 0
    assert provenance["clip_ids"] == [c for c, _ in table] and provenance["sequences"] == [s for _, s in table]
    assert provenance["shape_source"] == "template (GT YCB mesh)" and provenance["alpha"] == 0.2
    assert datetime.fromisoformat(provenance["created_utc"]).utcoffset() == timedelta(0)
    features = json.loads((tmp_path / "qc" / "features.json").read_text())
    assert len(features["clips"]) == 24 and features["skipped"] == []
    assert all(abs(row["pose_error_m"] - e) <= 1e-6 for row, e in zip(features["clips"], errors, strict=True))
    assert [row["good"] for row in features["clips"]] == [bool(e < TEST_TAU) for e in errors]

    with pytest.raises(FileExistsError):
        calibration.run_calibration(cfg, _REPO, gt_poses_fn=gts.__getitem__, features_fn=features_fn,
                                    clip_table=table)
    with pytest.raises(ValueError, match="dexycb_sequences"):
        calibration.run_calibration(_cfg(tmp_path, "acceptance_calibration.dexycb_sequences=[]"), _REPO)
