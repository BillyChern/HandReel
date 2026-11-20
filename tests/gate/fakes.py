"""Fixtures for the gate tests: fabricated token batches and indexes, a stub gate, encoder archives, a DexYCB tree.

Nothing here loads weights, touches the GPU, reads real data or downloads anything.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
import trimesh
import yaml
from PIL import Image
from torch import Tensor

from fastdo.gate.dataset import GateItem, write_gate_index
from fastdo.gate.labels import GateLabels
from fastdo.gate.model import GateConfig, GateOutput, InteractionGate
from fastdo.io.datasets.dexycb import ycb_object_name

__all__ = ["SERIAL", "StubGate", "add_doh_hands", "fabricated_gate_batch", "write_encoder_npz",
           "write_fabricated_gate_index", "write_gate_dexycb_tree"]

SERIAL = "836212060125"
GRASPED_YCB_ID = 5
_DEXYCB_PARTITIONS = ("fit", "val", "val_seq")


def fabricated_gate_batch(n_pos: int, n_neg: int, dim: int, seed: int, delta: float = 5.0,
                          n_shifted: int = 2) -> tuple[Tensor, GateLabels]:
    """`(n_pos + n_neg, 8, dim + 1)` float32 features, positives first: i.i.d. N(0, 1) tokens, `delta` added to token
    coordinate 0 in `n_shifted` frames of each positive (`randperm(8)[:n_shifted]`), a zero hand_conf column. Only
    interact is labelled."""
    gen = torch.Generator().manual_seed(seed)
    n = n_pos + n_neg
    tokens = torch.randn((n, 8, dim), generator=gen)
    for i in range(n_pos):
        frames = torch.randperm(8, generator=gen)[:n_shifted]
        tokens[i, frames, 0] += delta
    features = torch.cat([tokens, torch.zeros(n, 8, 1)], dim=-1)
    labels = GateLabels(interact=torch.cat([torch.ones(n_pos), torch.zeros(n_neg)]), rigid=torch.zeros(n),
                        visibility=torch.zeros(n), keyframe=torch.zeros(n, 8),
                        mask=torch.tensor([True, False, False, False]).expand(n, 4).clone())
    return features, labels


def write_fabricated_gate_index(root: Path, counts: Mapping[str, tuple[int, int]], dim: int = 16, seed: int = 0,
                                val_delta: float = 5.0) -> Path:
    """An index of `fabricated_gate_batch` clips per partition (`(n_pos, n_neg)`, seeded `seed + position`; `val` uses
    `val_delta`, every other partition delta 5), each written as an 8-row token file. fit/val/val_seq clips are DexYCB
    window items (visibility 1.0, positive keyframe targets 0.0); calib/check clips are 100DOH train items."""
    token_dir = root / "tokens"
    token_dir.mkdir(parents=True, exist_ok=True)
    items: list[GateItem] = []
    for offset, (partition, (n_pos, n_neg)) in enumerate(counts.items()):
        delta = val_delta if partition == "val" else 5.0
        features, _ = fabricated_gate_batch(n_pos, n_neg, dim, seed + offset, delta=delta)
        for i in range(n_pos + n_neg):
            positive, doh = i < n_pos, partition not in _DEXYCB_PARTITIONS
            source = "100doh" if doh else ("dexycb_pos" if positive else "dexycb_neg")
            item_id = f"gate_100doh_{partition}_{i:03d}" if doh else f"{source}_S/{partition}_{i:03d}/{SERIAL}"
            token_file = token_dir / f"{partition}_{i:03d}.npz"
            np.savez(token_file, tokens=features[i, :, :dim].numpy(), frame_idx=np.arange(8, dtype=np.int64))
            items.append(GateItem(
                item_id=item_id, source=source, clip_key=item_id, split="train" if doh else None,
                token_file=str(token_file), rows=() if doh else tuple(range(8)), interact=float(positive),
                row_visibility=None if doh else (1.0,) * 8,
                row_keyframe=(0.0,) * 8 if positive and not doh else None, partition=partition))
    path = root / "index.json"
    write_gate_index(path, items, {"crop": "letterbox", "fabricated": True})
    return path


class StubGate(InteractionGate):
    """An `InteractionGate` whose `p_interact` is fixed: one score per clip of the batch, or one score for every clip."""

    def __init__(self, scores: Sequence[float], token_dim: int = 16) -> None:
        super().__init__(GateConfig(token_dim=token_dim, hidden=4, dropout=0.0))
        self.fixed = torch.tensor([float(s) for s in scores], dtype=torch.float32)

    def forward(self, features: Tensor) -> GateOutput:
        base = super().forward(features)
        b = int(features.shape[0])
        p = self.fixed.expand(b).clone() if self.fixed.numel() == 1 else self.fixed.clone()
        if p.shape[0] != b:
            raise ValueError(f"StubGate holds {self.fixed.numel()} scores for a batch of {b}")
        return base._replace(p_interact=p, interact_logit=torch.logit(p))


def write_encoder_npz(artifact_dir: Path, tokens: np.ndarray, frame_idx: np.ndarray, crop: str | None) -> None:
    """An `encoder.npz` as the encoder stage writes it; `crop=None` omits the key, as in D3 archives."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    arrays = {"tokens": np.asarray(tokens, np.float32), "frame_idx": np.asarray(frame_idx, np.int64)}
    if crop is not None:
        arrays["crop"] = np.array(crop)
    np.savez(artifact_dir / "encoder.npz", **arrays)  # type: ignore[arg-type]


def add_doh_hands(root: Path, states: Mapping[str, Sequence[int]]) -> None:
    """Adds `hands[].contact_state` to every entry of `<root>/manifest.json`, by category."""
    path = root / "manifest.json"
    entries = json.loads(path.read_text())
    for entry in entries:
        entry["hands"] = [{"contact_state": int(s)} for s in states[entry["category"]]]
    path.write_text(json.dumps(entries))


def write_gate_dexycb_tree(root: Path, seq_ids: Sequence[str], n_frames: int = 12, n_free: int = 6) -> None:
    """A DexYCB-shaped tree: grasped object `GRASPED_YCB_ID` (a 2 cm box) still at z = 0.5 m, the hand 14 cm away in
    the first `n_free` frames and 3 mm from the box face after; colour JPEGs, `labels_*.npz` and `meta.yml`."""
    model = root / "models" / ycb_object_name(GRASPED_YCB_ID) / "textured_simple.obj"
    model.parent.mkdir(parents=True, exist_ok=True)
    trimesh.creation.box(extents=(0.02, 0.02, 0.02)).export(str(model))
    rng = np.random.default_rng(0)
    for seq_id in seq_ids:
        camera = root / seq_id
        camera.mkdir(parents=True)
        (camera.parent / "meta.yml").write_text(yaml.safe_dump(
            {"serials": [SERIAL], "num_frames": n_frames, "ycb_ids": [2, GRASPED_YCB_ID], "ycb_grasp_ind": 1}))
        for t in range(n_frames):
            Image.fromarray(rng.integers(0, 256, (12, 16, 3), dtype=np.uint8)).save(camera / f"color_{t:06d}.jpg")
            seg = np.zeros((12, 16), np.uint8)
            seg[2:6, 3:3 + 1 + t % 5] = GRASPED_YCB_ID
            pose_y = np.zeros((2, 3, 4), np.float32)
            pose_y[:, :, :3] = np.eye(3, dtype=np.float32)
            pose_y[:, 2, 3] = 0.5
            joint_3d = np.zeros((1, 21, 3), np.float32)
            joint_3d[..., 2] = 0.35 if t < n_free else 0.487
            np.savez(camera / f"labels_{t:06d}.npz", seg=seg, pose_y=pose_y, joint_3d=joint_3d)
