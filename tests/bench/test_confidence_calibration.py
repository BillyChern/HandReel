"""Confidence calibration statistics, and the stored DexYCB calibration run's thresholds."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.stats import spearmanr

from fastdo.bench.confidence_calibration import (
    TARGET_COVERAGE,
    FrameSamples,
    calibrate_scale,
    calibration_summary,
    confidence_at,
    coverage,
    frame_translation_errors,
    merge_samples,
    samples_from_eval_report,
    spearman,
    statistic_from_confidence,
    within_clip_spearman_median,
)
from fastdo.solver import SolverConfig

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_translation_errors_are_per_frame_origin_distances_with_missing_ground_truth() -> None:
    gt = np.tile(np.eye(4), (3, 1, 1))
    pred = gt.copy()
    pred[:, :3, 3] = [[0.003, 0.004, 0.0], [0.0, 0.0, 0.01], [1.0, 1.0, 1.0]]
    errors = frame_translation_errors(pred, gt, np.array([True, True, False]))
    assert errors[0] == pytest.approx(0.005) and errors[1] == pytest.approx(0.01) and errors[2] is None


def test_statistic_round_trips_through_confidence_and_zero_is_unsupported() -> None:
    z = np.array([0.0, 0.3, 1.0, 4.0, np.inf])
    conf = confidence_at(z, 1.7)
    assert conf[-1] == 0.0 and conf[0] == 1.0
    back = statistic_from_confidence(conf.astype(np.float32), 1.7)
    assert np.isinf(back[-1]) and np.allclose(back[:-1], z[:-1], atol=1e-6)


def test_spearman_is_against_the_negated_error_with_ties() -> None:
    rng = np.random.default_rng(0)
    err = rng.uniform(0, 0.02, 200)
    conf = np.round(np.exp(-err / 0.004) + rng.normal(0, 0.05, 200), 1)       # rounding makes ties
    assert spearman(conf, err) == pytest.approx(-float(spearmanr(conf, err).statistic), abs=1e-12)
    assert spearman(np.exp(-err), err) == pytest.approx(1.0)


def test_coverage_counts_accurate_frames_above_the_confidence_threshold() -> None:
    err = np.array([0.001, 0.002, 0.004, 0.006, 0.02])
    conf = np.array([0.9, 0.4, 0.6, 0.9, 0.1])
    assert coverage(conf, err) == pytest.approx(2 / 3)
    assert math.isnan(coverage(conf, err + 1.0))


def test_calibrated_scale_is_the_smallest_two_figure_scale_meeting_coverage() -> None:
    rng = np.random.default_rng(1)
    z = rng.uniform(0.2, 3.0, 400)
    err = np.where(rng.uniform(size=400) < 0.7, 0.002, 0.02)
    scale = calibrate_scale(z, err)
    assert coverage(confidence_at(z, scale), err) >= TARGET_COVERAGE
    step = 10.0 ** (math.floor(math.log10(scale)) - 1)
    assert coverage(confidence_at(z, scale - step), err) < TARGET_COVERAGE
    assert scale == float(f"{scale:.2g}")
    with pytest.raises(ValueError, match="unreachable"):
        calibrate_scale(np.where(err < 0.005, np.inf, 1.0), err)
    with pytest.raises(ValueError, match="below"):
        calibrate_scale(z, err + 1.0)


def _report(init_mode: str) -> dict[str, object]:
    return {"init_mode": init_mode, "confidence_scale": 2.0,
            "rows": [{"sequence": "s/a/1", "frame_confidence": [0.9, 0.5, 0.0],
                      "frame_translation_error_m": [0.001, None, 0.03]},
                     {"sequence": "s/b/1", "frame_confidence": [0.2], "frame_translation_error_m": [0.01]}]}


def test_eval_reports_become_scale_free_samples_grouped_by_init_mode_and_sequence() -> None:
    samples = merge_samples([samples_from_eval_report(_report("pnp")), samples_from_eval_report(_report("gt"))])
    assert samples.init_mode == ("pnp",) * 3 + ("gt",) * 3 and samples.sequence[:3] == ("s/a/1", "s/a/1", "s/b/1")
    assert np.allclose(samples.statistic[[0, 2]], [-2.0 * math.log(0.9), -2.0 * math.log(0.2)])
    assert np.isinf(samples.statistic[1]) and np.allclose(samples.error_m[:3], [0.001, 0.03, 0.01])
    summary = calibration_summary(samples, 2.0)
    assert summary["all"]["n_frames"] == 6 and summary["all"]["n_unsupported"] == 2
    assert summary["all"]["coverage"] == pytest.approx(1.0)
    assert summary["all"]["coverage_supported"] == pytest.approx(1.0)
    assert set(summary["by_init_mode"]) == {"pnp", "gt"} and summary["by_sequence"]["s/b/1"]["n_frames"] == 2
    assert json.loads(json.dumps(summary)) == summary                        # strict JSON: no NaN / inf
    assert within_clip_spearman_median(summary) == pytest.approx(1.0)       # s/b/1 has a constant confidence: skipped
    bad = _report("pnp")
    bad["rows"] = [{"sequence": "x", "frame_confidence": [0.5], "frame_translation_error_m": []}]
    with pytest.raises(ValueError, match="confidences"):
        samples_from_eval_report(bad)


def _outputs_roots() -> list[Path]:
    """`outputs/` of this checkout and, from a git worktree, of the main checkout (the artifacts are untracked)."""
    roots = [REPO_ROOT / "outputs"]
    git = REPO_ROOT / ".git"
    if git.is_file():                        # worktree: "gitdir: <main>/.git/worktrees/<name>"
        roots.append(Path(git.read_text().split("gitdir:", 1)[1].strip()).parents[1].parent / "outputs")
    return roots


def _stored_calibration() -> dict[str, Any] | None:
    for root in _outputs_roots():
        path = root / "eval" / "confidence_calibration.json"
        if path.is_file():
            return json.loads(path.read_text())
    return None


def _stored_dexycb() -> dict[str, Any]:
    stored = _stored_calibration()
    if stored is None:
        pytest.skip("outputs/eval/confidence_calibration.json not found (run run/pipeline/calibrate_confidence.py)")
    assert stored["mode"] == "oracle" and sorted(stored["init_modes"]) == ["gt", "gt_perturbed", "pnp"]
    assert len(stored["sequences"]) == 3
    return stored


def test_stored_dexycb_calibration_meets_coverage_at_the_shipped_scale() -> None:
    """Coverage on the stored run (run/pipeline/calibrate_confidence.py): three DexYCB oracle sequences, all
    three init modes, confidence > 0.5 on >= 90% of the frames with error < 5 mm, at the scale and mass floor
    `SolverConfig` ships. The DexYCB frames alone bind the scale; the synthetic fixture is reported, never binding
    (its occluded frames score 0.0 by definition and pushed a max(dexycb, synthetic) rule to 1.6)."""
    stored = _stored_dexycb()
    assert stored["confidence_scale"] == SolverConfig().confidence_scale
    assert stored["confidence_mass_floor"] == SolverConfig().confidence_mass_floor
    assert stored["binding"] == "dexycb" and stored["confidence_scale"] == stored["scale_dexycb"]
    assert stored["dexycb"]["all"]["coverage"] >= TARGET_COVERAGE, stored["dexycb"]["all"]
    assert stored["synthetic"]["all"]["n_frames"] == 120, stored["synthetic"]["all"]   # reported at the shipped scale


def test_stored_dexycb_calibration_ranks_frames_within_clips() -> None:
    """Within-clip ranking on the stored run: the median over the DexYCB oracle sequences of the within-clip
    Spearman(confidence, -error) is > 0. The confidence is a within-clip relative signal; the pooled Spearman
    over clips, once gated at >= 0.6, is no longer a gate and is only reported. At the shipped mass floor."""
    stored = _stored_dexycb()
    assert stored["confidence_mass_floor"] == SolverConfig().confidence_mass_floor
    by_sequence = stored["dexycb"]["by_sequence"]
    assert sorted(by_sequence) == sorted(stored["sequences"])
    assert all(v["spearman"] is not None for v in by_sequence.values()), by_sequence
    median = float(np.median([v["spearman"] for v in by_sequence.values()]))
    assert median > 0.0, {k: v["spearman"] for k, v in by_sequence.items()}
    assert stored["dexycb_within_clip_spearman_median"] == pytest.approx(median, abs=1e-12)


def test_frame_samples_are_frozen() -> None:
    samples = FrameSamples(statistic=np.zeros(1), error_m=np.zeros(1), init_mode=("gt",), sequence=("s",))
    with pytest.raises(AttributeError):
        samples.init_mode = ("pnp",)  # type: ignore[misc]
