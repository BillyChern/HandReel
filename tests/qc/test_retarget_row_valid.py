"""Row support for retargeting: observed rows, bridged single-row gaps, never extrapolations."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from fastdo.qc.export import retarget_row_valid
from tests.qc.records import make_record


def _record(confidence: list[float], depth: float = 0.6) -> object:
    t = len(confidence)
    record = make_record(np.arange(t) * 3)
    pose = record.object_pose.copy()
    pose[:, 2, 3] = depth
    return dataclasses.replace(record, confidence=np.asarray(confidence, dtype=np.float32), object_pose=pose)


def test_observed_rows_are_valid_and_a_hand_invalid_row_never_is() -> None:
    rec = _record([0.5, 0.4, 0.6, 0.5])
    hand = np.array([True, False, True, True])
    assert retarget_row_valid(rec, hand).tolist() == [True, False, True, True]


def test_a_single_interior_row_between_observed_rows_is_bridged() -> None:
    """Stride-2 geometry leaves isolated rows without pointmaps; between two observed rows the pose is an
    interpolation over 0.2 s (5 Hz keys: +1-3 mm in levers.md section 4), so the row is usable."""
    rec = _record([0.5, 0.0, 0.6, 0.0, 0.5])
    assert retarget_row_valid(rec, np.ones(5, dtype=bool)).tolist() == [True] * 5


def test_a_bridged_row_must_be_physically_plausible() -> None:
    rec = _record([0.5, 0.0, 0.6])
    pose = rec.object_pose.copy()
    pose[1, 2, 3] = -0.2                                        # the object behind the camera: never bridged
    rec = dataclasses.replace(rec, object_pose=pose)
    assert retarget_row_valid(rec, np.ones(3, dtype=bool)).tolist() == [True, False, True]


def test_longer_gaps_and_edge_runs_stay_masked() -> None:
    """On the Ego4D pilot: long interior gaps (141 of 152 in gaps >= 10) and edge runs are prior-only extrapolations."""
    rec = _record([0.0, 0.5, 0.0, 0.0, 0.5, 0.6, 0.0])
    assert retarget_row_valid(rec, np.ones(7, dtype=bool)).tolist() == [False, True, False, False, True, True, False]


@pytest.mark.parametrize("max_bridge", [0, 2])
def test_max_bridge_sets_the_longest_bridged_gap(max_bridge: int) -> None:
    rec = _record([0.5, 0.0, 0.0, 0.5])
    want = [True, max_bridge >= 2, max_bridge >= 2, True]
    assert retarget_row_valid(rec, np.ones(4, dtype=bool), max_bridge=max_bridge).tolist() == want
