"""The gate token dataset: items over per-clip token files, the JSON index, and the tensor loader.

An item is one clip window over one token file (a per-row DINOv2 `encoder.npz`, moved out of the engine work
directory). Sources and roles:

- `100doh`: a whole train-split clip, `calib` when its anchor annotation shows portable-object contact, else `check`.
  Test-split entries are dropped by `doh_train_entries` before any path is built, and the loader refuses a 100DOH item
  whose split is not `"train"`.
- `dexycb_pos` / `dexycb_neg`: the positive and no-contact windows of one DexYCB sequence (`dexycb_windows`), both in
  the sequence's partition: `fit`, `val` (subject-held-out) or `val_seq` (sequence-held-out).

The loader samples `N_GATE_FRAMES` rows of each item's window with `sample_rows`, exactly as the deployed gate stage
does, and labels heads by source: 100doh interact only; dexycb_pos interact, visibility and keyframe; dexycb_neg
interact and visibility. Rigid is never labelled.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from fastdo.bench.bench_clips import doh_clip_name, doh_train_entries
from fastdo.gate.dataset_dexycb import (
    check_dexycb_ids,
    excluded_dexycb_sequences,
    read_dexycb_frame_labels,
    read_dexycb_id_file,
)
from fastdo.gate.features import N_GATE_FRAMES, gate_features, sample_rows
from fastdo.gate.labels import (
    HEADS,
    DexycbFrameLabels,
    GateLabels,
    dexycb_windows,
    doh_interact_label,
)

__all__ = [
    "DOH_ITEM_PREFIX",
    "GATE_INDEX_VERSION",
    "PARTITIONS",
    "SOURCES",
    "GateDataset",
    "GateItem",
    "check_dexycb_ids",
    "dexycb_gate_items",
    "dexycb_token_name",
    "doh_gate_items",
    "excluded_dexycb_sequences",
    "item_counts",
    "load_gate_dataset",
    "read_dexycb_frame_labels",
    "read_dexycb_id_file",
    "read_gate_index",
    "write_gate_index",
]

GATE_INDEX_VERSION = 1
DOH_ITEM_PREFIX = "gate_100doh_"
SOURCES: tuple[str, ...] = ("100doh", "dexycb_pos", "dexycb_neg")
PARTITIONS: tuple[str, ...] = ("fit", "val", "val_seq", "calib", "check")
_MASKS: dict[str, tuple[bool, bool, bool, bool]] = {
    "100doh": (True, False, False, False),
    "dexycb_pos": (True, False, True, True),
    "dexycb_neg": (True, False, True, False),
}


@dataclass(frozen=True)
class GateItem:
    """One clip window: `rows` index the token file's rows (`()`: every row); `row_visibility` and `row_keyframe` are
    per window row (None where the source does not label them)."""

    item_id: str
    source: str
    clip_key: str
    split: str | None
    token_file: str
    rows: tuple[int, ...]
    interact: float
    row_visibility: tuple[float, ...] | None
    row_keyframe: tuple[float, ...] | None
    partition: str


@dataclass(frozen=True)
class GateDataset:
    """`features (B, 8, D + 1)` and `labels` of the selected items, with their ids, sources and partitions."""

    features: Tensor
    labels: GateLabels
    item_ids: list[str]
    sources: list[str]
    partitions: list[str]


def doh_gate_items(root: Path, token_dir: Path) -> list[GateItem]:
    """The 100DOH train-split items of `<root>/manifest.json` (the only file read), token files under `token_dir`.

    Raises:
        ValueError: If an entry lacks its contact annotation (`doh_interact_label`).
    """
    entries, _ = doh_train_entries(Path(root))
    items: list[GateItem] = []
    for entry in entries:
        name = doh_clip_name(entry)
        item_id = DOH_ITEM_PREFIX + name
        positive = doh_interact_label(entry)
        items.append(GateItem(item_id=item_id, source="100doh", clip_key=name, split=str(entry["official_split"]),
                              token_file=str(Path(token_dir) / f"{item_id}.npz"), rows=(), interact=float(positive),
                              row_visibility=None, row_keyframe=None, partition="calib" if positive else "check"))
    return items


def dexycb_token_name(seq_id: str) -> str:
    """The token file stem of a DexYCB sequence: `gate_dexycb_<subject>__<sequence>__<serial>`."""
    return "gate_dexycb_" + seq_id.replace("/", "__")


def dexycb_gate_items(seq_id: str, labels: DexycbFrameLabels, stride: int, partition: str, token_file: str,
                      contact_m: float = 0.015, free_m: float = 0.030, still_m: float = 0.005,
                      min_negative_frames: int = 4) -> list[GateItem]:
    """Zero, one or two items (`dexycb_pos_<id>`, `dexycb_neg_<id>`) from `dexycb_windows` with these thresholds.

    Positive keyframe targets are `distance_m < contact_m` at each window row (NaN gives 0.0).
    """
    positive, negative = dexycb_windows(labels, stride, contact_m, free_m, still_m, min_negative_frames)
    items: list[GateItem] = []
    for source, rows in (("dexycb_pos", positive), ("dexycb_neg", negative)):
        if rows is None:
            continue
        keyframe = (tuple(1.0 if bool(labels.distance_m[r] < contact_m) else 0.0 for r in rows)
                    if source == "dexycb_pos" else None)
        items.append(GateItem(item_id=f"{source}_{seq_id}", source=source, clip_key=seq_id, split=None,
                              token_file=token_file, rows=tuple(int(r) for r in rows),
                              interact=1.0 if source == "dexycb_pos" else 0.0,
                              row_visibility=tuple(float(labels.visibility[r]) for r in rows), row_keyframe=keyframe,
                              partition=partition))
    return items


def item_counts(items: Sequence[GateItem]) -> dict[str, dict[str, int]]:
    """Item counts per source, per partition, and per `<partition>/<source>`."""
    return {"source": dict(sorted(Counter(i.source for i in items).items())),
            "partition": dict(sorted(Counter(i.partition for i in items).items())),
            "partition_source": dict(sorted(Counter(f"{i.partition}/{i.source}" for i in items).items()))}


def write_gate_index(path: Path, items: Sequence[GateItem], provenance: Mapping[str, Any]) -> str:
    """Writes `{"version", "provenance", "items"}` as JSON via a tmp sibling and `os.replace`; returns its sha256."""
    payload = {"version": GATE_INDEX_VERSION, "provenance": dict(provenance),
               "items": [dataclasses.asdict(item) for item in items]}
    data = json.dumps(payload, indent=1).encode()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return hashlib.sha256(data).hexdigest()


def read_gate_index(path: Path) -> tuple[list[GateItem], dict[str, Any]]:
    """`(items, provenance)` of an index written by `write_gate_index`.

    Raises:
        ValueError: On another version, an unknown source or partition, or a 100DOH item whose split is not "train".
    """
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict) or payload.get("version") != GATE_INDEX_VERSION:
        raise ValueError(f"{path} is not a version-{GATE_INDEX_VERSION} gate index")
    items: list[GateItem] = []
    for raw in payload["items"]:
        item = GateItem(**{**raw, "rows": tuple(int(r) for r in raw["rows"]),
                           "row_visibility": None if raw["row_visibility"] is None else tuple(raw["row_visibility"]),
                           "row_keyframe": None if raw["row_keyframe"] is None else tuple(raw["row_keyframe"])})
        if item.source not in SOURCES or item.partition not in PARTITIONS:
            raise ValueError(f"{path}: item {item.item_id} has source {item.source!r} and partition "
                             f"{item.partition!r}; known: {SOURCES} and {PARTITIONS}")
        if item.source == "100doh" and item.split != "train":
            raise ValueError(f"{path}: 100DOH item {item.item_id} has split {item.split!r}; only train clips are "
                             "used (the test split is reserved for the final evaluation)")
        items.append(item)
    return items, dict(payload.get("provenance") or {})


def load_gate_dataset(index_path: Path, partitions: Collection[str]) -> GateDataset:
    """The tensors of the index items in `partitions` (module docstring), in index order.

    Raises:
        ValueError: As `read_gate_index`, on an unknown partition, a window outside its token file, per-row labels
            that do not match the window, or token files of different widths.
    """
    wanted = {str(p) for p in partitions}
    if wanted - set(PARTITIONS):
        raise ValueError(f"unknown partitions {sorted(wanted - set(PARTITIONS))}; known: {PARTITIONS}")
    items = [item for item in read_gate_index(index_path)[0] if item.partition in wanted]
    features: list[Tensor] = []
    visibility: list[float] = []
    keyframe: list[np.ndarray] = []
    cached: tuple[str, np.ndarray] | None = None
    for item in items:
        if cached is None or cached[0] != item.token_file:
            with np.load(item.token_file, allow_pickle=False) as archive:
                cached = (item.token_file, np.asarray(archive["tokens"], dtype=np.float32))
        tokens = cached[1]
        window = np.asarray(item.rows, dtype=np.int64) if item.rows else np.arange(tokens.shape[0], dtype=np.int64)
        if tokens.ndim != 2 or window.size == 0 or int(window.min()) < 0 or int(window.max()) >= tokens.shape[0]:
            raise ValueError(f"item {item.item_id}: window rows do not fit {item.token_file} {tokens.shape}")
        if features and int(tokens.shape[1]) + 1 != int(features[0].shape[1]):
            raise ValueError(f"item {item.item_id}: token width {tokens.shape[1]} differs from earlier items")
        picks = sample_rows(int(window.size))
        features.append(gate_features(torch.from_numpy(tokens[window])))
        row_vis, row_kf = item.row_visibility, item.row_keyframe
        if item.source != "100doh" and (row_vis is None or len(row_vis) != window.size):
            raise ValueError(f"item {item.item_id}: row_visibility must have one value per window row")
        if item.source == "dexycb_pos" and (row_kf is None or len(row_kf) != window.size):
            raise ValueError(f"item {item.item_id}: row_keyframe must have one value per window row")
        visibility.append(0.0 if row_vis is None else float(np.mean(np.asarray(row_vis, dtype=np.float64)[picks])))
        keyframe.append(np.zeros(N_GATE_FRAMES) if row_kf is None or item.source != "dexycb_pos"
                        else np.asarray(row_kf, dtype=np.float64)[picks])
    b = len(items)
    labels = GateLabels(interact=torch.tensor([item.interact for item in items], dtype=torch.float32),
                        rigid=torch.zeros(b), visibility=torch.tensor(visibility, dtype=torch.float32),
                        keyframe=(torch.from_numpy(np.stack(keyframe)).float() if b
                                  else torch.zeros(0, N_GATE_FRAMES)),
                        mask=torch.tensor([_MASKS[item.source] for item in items], dtype=torch.bool).reshape(
                            b, len(HEADS)))
    return GateDataset(features=torch.stack(features) if b else torch.zeros(0, N_GATE_FRAMES, 0), labels=labels,
                       item_ids=[item.item_id for item in items], sources=[item.source for item in items],
                       partitions=[item.partition for item in items])
