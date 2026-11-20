"""Tests for `fastdo.gate.labels`: 100DOH contact labels, synthetic stand-ins, DexYCB windows and partitions."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest
import torch

from fastdo.bench.bench_clips import doh_clip_name
from fastdo.gate.dataset import check_dexycb_ids, dexycb_gate_items
from fastdo.gate.labels import (
    HEADS,
    DexycbFrameLabels,
    dexycb_frame_labels,
    dexycb_partition,
    dexycb_windows,
    doh_interact_label,
    synthetic_stand_in_labels,
)
from fastdo.solver.synthetic import SyntheticConfig, make_synthetic_clip
from tests.bench.fake_doh import install_file_access_spy

_ALLOWED = ["20200820-subject-03", "20200903-subject-04", "20200908-subject-05", "20200918-subject-06"]


def _entry(**extra: Any) -> dict[str, Any]:
    return {"category": "drink", "video_id": "x_Yz-9", "start_seconds": 1, "end_seconds": 11, **extra}


def test_doh_interact_label() -> None:
    assert HEADS == ("interact", "rigid", "visibility", "keyframe")
    cases = {(3,): True, (0, 3): True, (4, 4): False, (1, 4): False, (2,): False}
    for states, expected in cases.items():
        assert doh_interact_label(_entry(hands=[{"contact_state": s} for s in states])) is expected
    with pytest.raises(ValueError, match=doh_clip_name(_entry())):
        doh_interact_label(_entry())
    with pytest.raises(ValueError):
        doh_interact_label(_entry(hands=[{"contact_state": 3}, {"side": "left"}]))


def test_synthetic_stand_in_labels_matches_occluder_semantics() -> None:
    occluded = make_synthetic_clip(SyntheticConfig(n_frames=20, n_surface=256, n_tracks=8, occlusion=0.6), seed=0)
    clear = make_synthetic_clip(SyntheticConfig(n_frames=20, n_surface=256, n_tracks=8, occlusion=0.0), seed=0)

    labels = synthetic_stand_in_labels([occluded, clear])

    assert labels.interact.tolist() == [1.0, 0.0]
    assert labels.rigid.tolist() == [1.0, 1.0]
    assert bool(((labels.visibility > 0) & (labels.visibility <= 1)).all())
    assert labels.mask.tolist() == [[True, True, True, False]] * 2
    assert labels.keyframe.shape == (2, 8)
    results = []
    for n_touching in (19, 20, 21):
        hands = torch.zeros(100, 8, 8, dtype=torch.bool)
        hands[:n_touching, 0, 0] = True
        masks = torch.ones(100, 8, 8, dtype=torch.bool) & ~hands
        clip = dataclasses.replace(occluded, masks=masks, hand_masks=hands)
        results.append(bool(synthetic_stand_in_labels([clip]).interact[0]))
    assert results == [False, True, True]


def test_dexycb_frame_labels_distances_and_invalid_rows() -> None:
    poses = np.zeros((6, 3, 4))
    poses[:, :, :3] = np.eye(3)
    poses[[0, 1, 5], 2, 3] = 0.5
    poses[3, 2, 3] = 0.52
    poses[2] = 0.0
    poses[4] = -1.0
    poses[5, 0, 1] = np.nan
    joints = np.zeros((6, 21, 3))
    joints[..., 2] = 0.45
    joints[1] = -1.0

    labels = dexycb_frame_labels(joints, np.zeros((1, 3)), poses, np.array([10, 20, 0, 40, 40, 40]))

    np.testing.assert_allclose(labels.distance_m, [0.05, np.inf, np.nan, 0.07, np.nan, np.nan], atol=1e-6)
    assert labels.displacement_m[3] == pytest.approx(0.02, abs=1e-9)
    assert np.isnan(labels.displacement_m[[2, 4, 5]]).all()
    np.testing.assert_allclose(labels.visibility, [0.25, 0.5, 0, 1, 1, 1])
    with pytest.raises(ValueError):
        dexycb_frame_labels(joints, np.zeros((1, 3)), poses, np.zeros(6))


def _window_labels(**changes: Any) -> DexycbFrameLabels:
    distance = np.array([0.050] * 8 + [0.020] * 4 + [0.005] * 18)
    displacement = np.where(np.arange(30) < 15, 0.0, 0.02)
    visibility = np.ones(30)
    visibility[21] = 0.5
    base = DexycbFrameLabels(distance_m=distance, displacement_m=displacement, visibility=visibility)
    return dataclasses.replace(base, **changes)


def test_dexycb_windows_rule() -> None:
    labels = _window_labels()

    positive, negative = dexycb_windows(labels, stride=3)

    assert negative == list(range(8)) and positive == list(range(0, 30, 3))
    items = dexycb_gate_items("S/q/c", labels, 3, "fit", "tok.npz")
    pos, neg = items
    assert (pos.item_id, pos.source, pos.interact, pos.rows) == ("dexycb_pos_S/q/c", "dexycb_pos", 1.0,
                                                                tuple(range(0, 30, 3)))
    assert pos.row_keyframe == (0.0,) * 4 + (1.0,) * 6
    assert pos.row_visibility == (1.0,) * 7 + (0.5,) + (1.0,) * 2
    assert (neg.item_id, neg.source, neg.interact, neg.rows) == ("dexycb_neg_S/q/c", "dexycb_neg", 0.0, tuple(range(8)))
    assert neg.row_visibility == (1.0,) * 8 and neg.row_keyframe is None
    assert all(item.clip_key == "S/q/c" and item.split is None and item.partition == "fit" for item in items)
    only_neg = dexycb_gate_items("S/q/c", labels, 3, "fit", "tok.npz", contact_m=0.001)
    assert [item.source for item in only_neg] == ["dexycb_neg"]
    only_pos = dexycb_gate_items("S/q/c", labels, 3, "fit", "tok.npz", min_negative_frames=9)
    assert [item.source for item in only_pos] == ["dexycb_pos"]

    nan_distance = labels.distance_m.copy()
    nan_distance[5] = np.nan
    assert dexycb_windows(_window_labels(distance_m=nan_distance), 3)[1] == list(range(5))
    short = labels.distance_m.copy()
    short[3:8] = 0.020
    assert dexycb_windows(_window_labels(distance_m=short), 3)[1] is None
    assert dexycb_windows(_window_labels(distance_m=np.full(30, 0.050)), 3)[0] is None


def test_dexycb_partition_rule() -> None:
    ids = [f"A/s{i}/c" for i in range(10)] + ["B/t0/c", "B/t1/c"]

    parts = dexycb_partition(ids, val_subject="B")

    assert parts["B/t0/c"] == parts["B/t1/c"] == "val"
    assert parts["A/s4/c"] == parts["A/s9/c"] == "val_seq"
    assert sorted(i for i, p in parts.items() if p == "fit") == sorted(f"A/s{i}/c" for i in (0, 1, 2, 3, 5, 6, 7, 8))
    labels = _window_labels()
    assert {item.partition for item in dexycb_gate_items("A/s4/c", labels, 3, parts["A/s4/c"], "t.npz")} == {"val_seq"}
    with pytest.raises(ValueError):
        dexycb_partition(ids, val_subject="C")
    with pytest.raises(ValueError):
        dexycb_partition(["B/t0/c", "B/t1/c"], val_subject="B")


def test_check_dexycb_ids_refuses_before_io(monkeypatch: pytest.MonkeyPatch) -> None:
    refused = ["20200928-subject-07/x/836212060125", "20201002-subject-08/x/836212060125",
               "20201015-subject-09/x/836212060125"]
    spy = install_file_access_spy(monkeypatch)
    for seq_id in refused:
        with pytest.raises(ValueError, match=seq_id):
            check_dexycb_ids(["20200820-subject-03/20200820_135508/836212060125", seq_id], _ALLOWED)
    calls = list(spy.calls)
    monkeypatch.undo()
    assert calls == []
    ok = "20200820-subject-03/20200820_135508/836212060125"
    assert check_dexycb_ids([ok], _ALLOWED) == [ok]
