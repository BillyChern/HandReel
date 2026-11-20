"""Tests for `fastdo.engine.qc_calibration`: pose-error labels, skipped clips and the coverage check."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from fastdo.engine.qc_calibration import (
    CalibrationClip,
    calibration_clip,
    collect_calibration_clips,
    fit_rule_from_clips,
    template_frame_poses,
)
from fastdo.qc import QcFeatures
from tests.engine.fakes_qc import write_solved_clip
from tests.engine.fakes_solve import monotone_features, shifted_gt, write_template_meta
from tests.engine.test_workers_qc import PASSING

_LOGGER = "fastdo.engine.qc_calibration"


def _passing(*args: Any, **kwargs: Any) -> QcFeatures:
    return PASSING


def _monotone_clips(n: int) -> list[CalibrationClip]:
    errors = np.linspace(0.002, 0.05, n)
    return [CalibrationClip(f"c{i}", f"s/q/c{i}", features, float(e), 6, 6, 1.0)
            for i, (e, features) in enumerate(zip(errors, monotone_features(errors), strict=True))]


def test_template_frame_poses_inverts_the_shape_normalization() -> None:
    rng = np.random.default_rng(5)
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = Rotation.from_rotvec(rng.normal(size=3)).as_matrix(), rng.normal(size=3)
    scale, ns, c = 0.7, 3.0, np.array([0.1, -0.2, 0.05])
    raw = rng.normal(size=(50, 3))
    meta = {"backend": "template", "normalization_scale": ns, "normalization_center": c.tolist()}

    out, metric_scale = template_frame_poses(pose[None], scale, meta)

    assert metric_scale == 0.7 * 3.0                      # the product itself (0.7 * 3.0 != 2.1 in float64)
    canonical = (raw - c) * ns
    np.testing.assert_allclose(scale * canonical @ pose[:3, :3].T + pose[:3, 3],
                               metric_scale * raw @ out[0, :3, :3].T + out[0, :3, 3], rtol=0.0, atol=1e-12)
    np.testing.assert_array_equal(out[0, :3, :3], pose[:3, :3])
    with pytest.raises(ValueError):
        template_frame_poses(pose[None], scale, {**meta, "backend": "sam3d"})


def test_calibration_clip_labels_with_pose_error_m_on_valid_gt_frames(tmp_path: Path) -> None:
    clip = tmp_path / "clip0"
    solution = write_solved_clip(clip, t=6, stride=3)
    write_template_meta(clip)
    gt, valid = shifted_gt(solution, 0.01)
    gt[6, 0, 3] += 1.0                                   # source frame 6 (solve row 2): far off, and invalid
    valid[6] = False

    result = calibration_clip("clip0", "s/q/c", clip, (gt, valid), features_fn=_passing)

    assert result.pose_error_m == pytest.approx(0.01, abs=1e-6)
    assert (result.n_gt_frames, result.n_frames, result.features) == (5, 6, PASSING)
    assert (result.clip_id, result.sequence, result.metric_scale) == ("clip0", "s/q/c", solution.scale * 1.0)
    with pytest.raises(ValueError, match="valid"):
        calibration_clip("clip0", "s/q/c", clip, (gt, np.zeros_like(valid)), features_fn=_passing)


def test_collect_calibration_clips_records_skipped_clips(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    table = [(f"clip{i}", f"s/q/c{i}") for i in range(3)]
    gts = {}
    for i, (clip_id, seq) in enumerate(table):
        gts[seq] = shifted_gt(write_solved_clip(tmp_path / clip_id, seed=i), 0.005)
        write_template_meta(tmp_path / clip_id)
    (tmp_path / "clip1" / "solve.npz").unlink()

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        clips, skipped = collect_calibration_clips(tmp_path, table, gts.__getitem__, features_fn=_passing)

    assert [c.clip_id for c in clips] == ["clip0", "clip2"]
    assert [(s["clip_id"], s["sequence"]) for s in skipped] == [("clip1", "s/q/c1")] and set(skipped[0]) == \
        {"clip_id", "sequence", "reason"}
    assert "solve" in skipped[0]["reason"]
    assert len([r for r in caplog.records if r.name == _LOGGER and r.levelno == logging.WARNING]) == 1


def test_fit_rule_from_clips_refuses_an_unguaranteed_threshold(caplog: pytest.LogCaptureFixture) -> None:
    # n = 12: 5 good clips at tau 0.02; the stratified split keeps 3 good calibration clips; conformal_rank(3, 0.1) = 4.
    with pytest.raises(ValueError, match="coverage"):
        fit_rule_from_clips(_monotone_clips(12))
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        rule = fit_rule_from_clips(_monotone_clips(12), allow_unguaranteed=True)
    assert rule.calibration.n_calibration == 3
    mine = [r for r in caplog.records if r.name == _LOGGER]
    assert len(mine) == 1 and mine[0].levelno == logging.WARNING and "coverage" in mine[0].getMessage()

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        rule = fit_rule_from_clips(_monotone_clips(60))     # 23 good; 12 fit, 11 calibrate; conformal_rank(11, 0.1) = 11
    assert rule.calibration.n_calibration == 11
    assert not [r for r in caplog.records if r.name in (_LOGGER, "fastdo.qc.conformal")]
    with pytest.raises(ValueError, match="good"):
        fit_rule_from_clips(_monotone_clips(12)[5:])        # no good clip at all


def test_the_calibration_labels_the_poses_the_qc_stage_will_actually_score(tmp_path: Path) -> None:
    """P2's load-bearing invariant: the rule must be FIT on the pose source it is DEPLOYED on.

    `workers_qc.read_qc_inputs` swaps in the contact stage's corrected poses when `contact.npz` is fresh, so the
    deployed qc stage scores the corrected trajectory. `calibration_clip` calls the same reader, and this test
    exists so that a future refactor cannot split the two paths without going red: a rule fitted on uncorrected
    poses and used on corrected ones is calibrated against a distribution the engine no longer produces.
    """
    from fastdo.engine.workers_contact import CONTACT_FILE, _sha256
    from fastdo.engine.worker_support import SOLVE_FILE, save_npz

    d = tmp_path / "clip"
    solution = write_solved_clip(d)
    write_template_meta(d)
    n = int(np.asarray(solution.poses).shape[0])
    gt = shifted_gt(solution, 0.0)

    before = calibration_clip("clip", "s/q/c", d, gt)

    shift = np.zeros((n, 3), dtype=np.float64)
    shift[:, 2] = 0.05                                    # 50 mm along +z, a correction qc would score
    moved = np.asarray(solution.poses, dtype=np.float32).copy()
    moved[:, :3, 3] += shift.astype(np.float32)
    save_npz(d / CONTACT_FILE, poses=moved, shift_mm=(shift * 1000.0).astype(np.float32),
             frame_idx=np.asarray(np.load(d / "decode.npz")["frame_idx"], dtype=np.int64),
             solve_sha256=np.array(_sha256(d / SOLVE_FILE)))

    after = calibration_clip("clip", "s/q/c", d, gt)
    assert after.pose_error_m != pytest.approx(before.pose_error_m), (
        "calibration_clip ignored contact.npz: it would label the uncorrected solve while the qc stage scores the "
        "corrected one")
    assert after.pose_error_m == pytest.approx(before.pose_error_m + 0.05, abs=2e-3)
