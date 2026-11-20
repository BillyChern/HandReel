"""Kinematic-path fixes: the replay's success criteria read the reference
object only on the rows the retarget USED (rows past the clip end or unobserved are the solver's extrapolations),
and the reported DexPilot residual is that of the unfiltered solve, not of the low-passed hand stream."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fastdo.robot.kinematic_path import KinematicPathConfig, load_reconstruction, run_kinematic_path
from tests.robot import synthetic


def test_reference_motion_is_read_on_used_rows_only(g1, tmp_path: Path) -> None:
    """Known answer: the last 4 rows are unobserved (confidence 0) and their object pose is an extrapolation 0.5 m
    off. The reference displacement is the 10 cm slide between the first and last USED rows (rows 0 and 15: 7.9 cm);
    reading row 19 instead (the pre-fix `object_ref[-1]`) gives ~0.51 m."""
    obj = np.tile(np.eye(4), (synthetic.N_ROWS, 1, 1))
    obj[:, 0, 3] = synthetic.OBJECT_XY[0]
    obj[:, 1, 3] = synthetic.OBJECT_XY[1] + np.linspace(0.0, 0.10, synthetic.N_ROWS)
    obj[:, 2, 3] = synthetic.TABLE_Z + 0.5 * 0.06 + 0.001
    obj[16:, 0, 3] += 0.5
    conf = np.ones(synthetic.N_ROWS)
    conf[16:] = 0.0
    clip = synthetic.write_clip(g1, tmp_path / "tail", object_world=obj, confidence=conf)
    result = run_kinematic_path(load_reconstruction(clip), KinematicPathConfig(table_xy=synthetic.OBJECT_XY), g1)
    assert result.used_rows[:16].all() and not result.used_rows[16:].any()
    ref = result.object_ref
    expected = float(np.linalg.norm(ref[15, :3, 3] - ref[0, :3, 3]))
    assert expected == pytest.approx(0.10 * 15 / 19, abs=1e-6)
    m = result.metrics
    assert m.reference_displacement == pytest.approx(expected, abs=1e-9)
    assert m.object_final_error < 0.1                                # to row 15's pose, not the extrapolation


def test_hand_residual_is_the_unfiltered_retarget(g1, tmp_path: Path) -> None:
    """Known answer: the human hand switches pose at row 10. The unfiltered DexPilot solve follows it within ~1.7 mm;
    the low-passed stream lags (5.3 mm at row 10 at the clip's 10 Hz), which the pre-fix residual reported as a
    retargeting error. The exported hand stream stays low-passed."""
    hand = np.tile(synthetic.HAND_Q, (synthetic.N_ROWS, 1))
    hand[10:] = [0.2, -0.1, -0.7, 0.7, 0.6, 0.8, 0.7]
    clip = synthetic.write_clip(g1, tmp_path / "step", hand_q=hand)
    result = run_kinematic_path(load_reconstruction(clip), KinematicPathConfig(table_xy=synthetic.OBJECT_XY), g1)
    assert result.hand_vec_err[10] < 3e-3
    assert np.nanmax(result.hand_vec_err) < 3e-3
    assert np.abs(result.hand_q[10] - hand[10]).max() > 0.05          # the exported stream still lags the step


def test_reference_skips_observed_but_untrusted_rows(g1, tmp_path: Path) -> None:
    """Known answer: rows 16-19 are observed but with object confidence 0.05 (< 0.1, about 3.2 noise floors) and
    0.5 m off. They stay USED (the hand path is the human's), but the reference end is row 15, the last trusted row.
    Mutation (reference on every used row) reads row 19: ~0.51 m."""
    obj = np.tile(np.eye(4), (synthetic.N_ROWS, 1, 1))
    obj[:, 0, 3] = synthetic.OBJECT_XY[0]
    obj[:, 1, 3] = synthetic.OBJECT_XY[1] + np.linspace(0.0, 0.10, synthetic.N_ROWS)
    obj[:, 2, 3] = synthetic.TABLE_Z + 0.5 * 0.06 + 0.001
    obj[16:, 0, 3] += 0.5
    conf = np.ones(synthetic.N_ROWS)
    conf[16:] = 0.05
    clip = synthetic.write_clip(g1, tmp_path / "tail", object_world=obj, confidence=conf)
    result = run_kinematic_path(load_reconstruction(clip), KinematicPathConfig(table_xy=synthetic.OBJECT_XY), g1)
    assert result.used_rows.all()
    assert result.metrics.reference_displacement == pytest.approx(0.10 * 15 / 19, abs=1e-6)
