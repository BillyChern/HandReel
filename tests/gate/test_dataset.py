"""Tests for `fastdo.gate.dataset`: 100DOH train items, the gate index and its loader, DexYCB exclusions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from fastdo.gate.dataset import (
    GateItem,
    check_dexycb_ids,
    doh_gate_items,
    excluded_dexycb_sequences,
    load_gate_dataset,
    write_gate_index,
)
from fastdo.gate.features import sample_rows
from tests.bench.fake_doh import install_file_access_spy, write_fake_doh_bundle
from tests.gate.fakes import add_doh_hands

_STATES = {"boardgame": [3], "drink": [4, 4], "cooking": [3], "misc": [3]}


def test_doh_gate_items_train_only_never_opens_test(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = write_fake_doh_bundle(tmp_path / "100DOH")
    add_doh_hands(bundle.root, _STATES)

    spy = install_file_access_spy(monkeypatch)
    items = doh_gate_items(bundle.root, tmp_path / "tokens")
    monkeypatch.undo()

    assert [(i.item_id, i.partition, i.split, i.source, i.rows) for i in items] == [
        ("gate_100doh_boardgame_3Bsm34JRL7w_64_74", "calib", "train", "100doh", ()),
        ("gate_100doh_drink_x_Yz-9_1_11", "check", "train", "100doh", ())]
    assert [i.interact for i in items] == [1.0, 0.0]
    assert all(i.row_visibility is None and i.row_keyframe is None for i in items)
    for marker in bundle.held_out_markers:
        assert spy.touched(marker) == []


def _item(source: str, token_file: Path, rows: tuple[int, ...], **extra: object) -> GateItem:
    fields: dict[str, object] = {"item_id": f"{source}_x", "source": source, "clip_key": "x", "split": None,
                                 "token_file": str(token_file), "rows": rows, "interact": 1.0,
                                 "row_visibility": None, "row_keyframe": None, "partition": "fit"}
    fields.update(extra)
    return GateItem(**fields)  # type: ignore[arg-type]


def test_write_and_load_gate_index_roundtrip(tmp_path: Path) -> None:
    tokens = np.random.default_rng(0).standard_normal((30, 16)).astype(np.float32)
    np.savez(tmp_path / "t.npz", tokens=tokens, frame_idx=np.arange(30))
    pos_rows = tuple(range(0, 30, 3))
    pos_vis = tuple(float(v) for v in np.linspace(0.1, 1.0, 10))
    pos_kf = (0.0,) * 4 + (1.0,) * 6
    items = [
        _item("100doh", tmp_path / "t.npz", (), split="train", partition="calib"),
        _item("dexycb_pos", tmp_path / "t.npz", pos_rows, row_visibility=pos_vis, row_keyframe=pos_kf),
        _item("dexycb_neg", tmp_path / "t.npz", tuple(range(8)), interact=0.0, row_visibility=(0.5,) * 8,
              partition="val"),
    ]
    index = tmp_path / "index.json"

    sha = write_gate_index(index, items, {"crop": "letterbox"})
    data = load_gate_dataset(index, {"fit", "val", "calib"})

    assert sha == hashlib.sha256(index.read_bytes()).hexdigest()
    assert json.loads(index.read_text())["provenance"] == {"crop": "letterbox"}
    assert data.features.shape == (3, 8, 17)
    assert data.sources == ["100doh", "dexycb_pos", "dexycb_neg"] and data.partitions == ["calib", "fit", "val"]
    assert data.item_ids == ["100doh_x", "dexycb_pos_x", "dexycb_neg_x"]
    assert data.labels.mask.tolist() == [[True, False, False, False], [True, False, True, True],
                                         [True, False, True, False]]
    picks = sample_rows(10)
    assert torch.equal(data.features[1, :, :16], torch.from_numpy(tokens[np.asarray(pos_rows)[picks]]))
    assert torch.equal(data.features[0, :, :16], torch.from_numpy(tokens[sample_rows(30)]))
    assert float(data.labels.visibility[1]) == pytest.approx(float(np.mean(np.asarray(pos_vis)[picks])))
    assert data.labels.keyframe[1].tolist() == [pos_kf[i] for i in picks]
    assert data.labels.interact.tolist() == [1.0, 1.0, 0.0]
    assert load_gate_dataset(index, {"val_seq"}).features.shape[0] == 0
    assert load_gate_dataset(index, {"fit"}).item_ids == ["dexycb_pos_x"]


def test_load_gate_dataset_refuses_test_split_items(tmp_path: Path) -> None:
    np.savez(tmp_path / "t.npz", tokens=np.zeros((8, 4), np.float32), frame_idx=np.arange(8))
    write_gate_index(tmp_path / "index.json", [_item("100doh", tmp_path / "t.npz", (), split="test",
                                                     partition="calib")], {})
    with pytest.raises(ValueError, match="test"):
        load_gate_dataset(tmp_path / "index.json", {"fit"})


def test_excluded_sequences_are_refused_before_io(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "gauge.json").write_text(json.dumps(
        {"sets": {"old": {"sequences": ["20200903_104655"]}}, "generated_utc": "2026-09-13T16:41:15+00:00"}))
    (tmp_path / "ids.json").write_text(json.dumps({"sequences": ["20200820-subject-03/20200820_140000/836212060125"]}))
    allowed = ["20200820-subject-03", "20200903-subject-04"]

    excluded = excluded_dexycb_sequences([tmp_path / "gauge.json", tmp_path / "ids.json"])

    assert excluded == frozenset({"20200903_104655", "20200820_140000"})
    spy = install_file_access_spy(monkeypatch)
    for refused in ("20200903-subject-04/20200903_104655/836212060125",
                    "20200820-subject-03/20200820_140000/836212060125"):
        with pytest.raises(ValueError, match=refused):
            check_dexycb_ids([refused], allowed, excluded)
    with pytest.raises(ValueError, match="serial"):
        check_dexycb_ids(["20200820-subject-03/20200820_150000/932122060857"], allowed, excluded)
    with pytest.raises(ValueError, match="twice"):
        check_dexycb_ids(["20200820-subject-03/20200820_1/836212060125"] * 2, allowed, excluded)
    calls = list(spy.calls)
    monkeypatch.undo()
    assert calls == []
    with pytest.raises(FileNotFoundError):
        excluded_dexycb_sequences([tmp_path / "missing.json"])
