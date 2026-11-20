"""Duck-typed model fakes and artifact writers shared by the engine-worker tests.

Nothing here loads weights, touches the GPU or downloads anything.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch import Tensor

from fastdo.geometry.intrinsics import camera_matrix, write_intrinsics

NORMALIZED_K = np.array([[0.79, 0.0, 0.5], [0.0, 1.40, 0.5], [0.0, 0.0, 1.0]], np.float32)


class FakeDinov2:
    """`Dinov2Model`-shaped: `model(pixel_values=(B,3,S,S)).last_hidden_state` is `(B, 1+P, hidden)`.

    Token `t` of sample `b` holds `mean(pixel_values[b]) + t` in every channel, so the CLS token (t = 0)
    and the patch-token mean (t = 1..P) are both known in closed form.
    """

    hidden = 4
    n_patches = 3

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []
        self.autocast_bf16: list[bool] = []
        self.inputs: list[Tensor] = []

    def __call__(self, pixel_values: Tensor) -> Any:
        device_type = pixel_values.device.type
        self.batch_sizes.append(int(pixel_values.shape[0]))
        self.autocast_bf16.append(torch.is_autocast_enabled(device_type)
                                  and torch.get_autocast_dtype(device_type) == torch.bfloat16)
        self.inputs.append(pixel_values.detach().clone())
        b = pixel_values.shape[0]
        offsets = torch.arange(1.0 + self.n_patches)[None, :, None]
        tokens = pixel_values.float().mean(dim=(1, 2, 3))[:, None, None] + offsets
        return SimpleNamespace(last_hidden_state=tokens.expand(b, 1 + self.n_patches, self.hidden))


class FakeOnlineTracker:
    """`CoTrackerOnlinePredictor`-shaped: `.step`, a priming call, then windowed calls whose output grows by
    `chunk - step` frames; track `t` of query `j` sits at `query_uv[j] + t`."""

    step = 4

    def __init__(self) -> None:
        self.total = 0
        self.queries: Tensor | None = None

    def __call__(self, video_chunk: Tensor, is_first_step: bool = False,
                 queries: Tensor | None = None) -> tuple[Tensor | None, Tensor | None]:
        if is_first_step:
            self.total, self.queries = 0, queries
            return None, None
        assert self.queries is not None
        length = int(video_chunk.shape[1])
        self.total = length if self.total == 0 else self.total + length - self.step
        uv = self.queries[0, :, 1:]
        tracks = uv[None, None] + torch.arange(float(self.total))[None, :, None, None]
        return tracks, torch.ones(1, self.total, uv.shape[0], dtype=torch.bool)


class FakeMoGe:
    """`MoGeModel.infer`-shaped. Pixel (0,0) channel 0 of each input frame carries a tag; `depth[tag]` is the
    `(H,W)` depth it returns (x, y are junk: `run_moge(intrinsics=K)` rebuilds them from depth through K).
    It also returns the fixed normalized intrinsics `NORMALIZED_K` and records every call's kwargs."""

    def __init__(self, depth: dict[int, np.ndarray] | None = None) -> None:
        self.depth = depth or {}
        self.batches: list[list[int]] = []
        self.kwargs: list[dict[str, Any]] = []

    def infer(self, image: Tensor, **kwargs: Any) -> dict[str, Tensor]:
        tags = (image[:, 0, 0, 0] * 255.0).round().long().tolist()
        self.batches.append(tags)
        self.kwargs.append(kwargs)
        b, _, h, w = image.shape
        points = torch.zeros(b, h, w, 3)
        for i, tag in enumerate(tags):
            if tag in self.depth:
                points[i, ..., 2] = torch.from_numpy(self.depth[tag])
        return {"points": points, "intrinsics": torch.from_numpy(NORMALIZED_K).expand(b, 3, 3).clone()}


def random_frames(n: int, h: int = 24, w: int = 32, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, size=(n, h, w, 3), dtype=np.uint8)


def write_decode(artifact_dir: Path, frames: np.ndarray, frame_idx: list[int] | None = None,
                 fps: float = 10.0, stride: int | None = None, pts: list[float] | None = None,
                 timebase_mismatch: bool | None = None) -> None:
    """A `decode.npz` as the decode stage writes it (stride-3 source indices unless `frame_idx` is given); the
    `stride` key is written only when `stride` is given, as in archives from before the decode stage wrote it.

    `pts` and `timebase_mismatch` are the window decode's measured-time members, written only when given --
    the window decode path records them and the whole-file path does not, so both shapes must be constructible."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    idx = np.arange(len(frames), dtype=np.int64) * 3 if frame_idx is None else np.asarray(frame_idx, np.int64)
    payload: dict[str, np.ndarray] = {"frames": frames.astype(np.uint8), "frame_idx": idx,
                                      "fps": np.float64(fps)}
    if stride is not None:
        payload["stride"] = np.int64(stride)
    if pts is not None:
        payload["pts"] = np.asarray(pts, np.float64)
    if timebase_mismatch is not None:
        payload["timebase_mismatch"] = np.int64(int(timebase_mismatch))
    np.savez(artifact_dir / "decode.npz", **payload)


def forbid_full_frames(monkeypatch: Any) -> None:
    """Makes any `np.load(...)["frames"]` raise, so a caller passes only if it reads decode rows alone."""
    real = np.lib.npyio.NpzFile.__getitem__

    def guarded(self: np.lib.npyio.NpzFile, key: str) -> Any:
        if key == "frames":
            raise AssertionError("the full frames array was loaded")
        return real(self, key)

    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", guarded)


def full_load_decode_rows(path: Path, rows: Any) -> tuple[np.ndarray, np.ndarray]:
    """`read_decode_rows` as the readers did it before row-only reads: the whole `frames` array, then the rows."""
    with np.load(path, allow_pickle=False) as data:
        return np.asarray(data["frames"])[np.asarray(rows, dtype=np.intp)], np.asarray(data["frame_idx"], np.int64)


def make_hawor_npz(path: Path, t: int, seed: int = 0, **overrides: np.ndarray) -> Path:
    """A HaWoR `all_hand_meshes.npz`-shaped file with `t` rows and per-row distinct betas."""
    rng = np.random.default_rng(seed)
    data: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        data[f"{side}_vertices"] = rng.standard_normal((t, 778, 3)).astype(np.float32)
        data[f"{side}_faces"] = np.arange(1552 * 3, dtype=np.int32).reshape(1552, 3)
        data[f"{side}_joints"] = rng.standard_normal((t, 21, 3)).astype(np.float32)
        data[f"{side}_trans"] = rng.standard_normal((t, 3)).astype(np.float32)
        data[f"{side}_rot"] = rng.standard_normal((t, 3)).astype(np.float32)
        data[f"{side}_hand_pose"] = rng.standard_normal((t, 45)).astype(np.float32)
        data[f"{side}_betas"] = rng.standard_normal((t, 10)).astype(np.float32)
        data[f"{side}_valid"] = np.ones(t, dtype=bool)
    data.update(overrides)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **data)  # type: ignore[arg-type]  # numpy's stub types `allow_pickle` into **kwds
    return path


# --- a geometry scene with a known metric scale ------------------------------------------------------

SCENE_H, SCENE_W, SCENE_N, FALSE_SCALE = 48, 64, 4, 2.5
SCENE_K = camera_matrix(60.0, 60.0, 31.5, 23.5)


def true_depth(row: int) -> np.ndarray:
    u = np.arange(SCENE_W, dtype=np.float32)[None, :].repeat(SCENE_H, 0)
    return (0.5 + 0.004 * u + 0.05 * row).astype(np.float32)


def write_geometry_scene(artifact_dir: Path, tag_offset: int = 0) -> FakeMoGe:
    """Decode rows at source frames 0, 3, 6, 9, `intrinsics.npz` = `SCENE_K`, and a full-rate `hands.npz`
    (T = 12). Only rows at those source frames hold right-hand joints consistent with the true depth; the
    others sit at twice the depth. Matching hand rows by row position instead of by source frame index therefore
    gives per-frame scales of about [2.5, 4.63, 4.32, 2.18] instead of 2.5 each. Their lower median over the 4
    frames is still 2.5, so a test of the row-matching rule must check `frame_scale`, not only the pooled
    `scale`. Returns a `FakeMoGe` whose depth is the true depth divided by `FALSE_SCALE`."""
    frames = np.zeros((SCENE_N, SCENE_H, SCENE_W, 3), np.uint8)
    frames[:, 0, 0, 0] = tag_offset + np.arange(SCENE_N)
    write_decode(artifact_dir, frames, frame_idx=[0, 3, 6, 9])
    write_intrinsics(artifact_dir, SCENE_K, "moge2")
    rng = np.random.default_rng(0)
    us, vs = rng.integers(2, SCENE_W - 2, 21), rng.integers(2, SCENE_H - 2, 21)
    fx, fy = float(SCENE_K[0, 0]), float(SCENE_K[1, 1])
    cx, cy = float(SCENE_K[0, 2]), float(SCENE_K[1, 2])
    right = np.zeros((12, 21, 3), np.float32)
    for t in range(12):
        z = true_depth(t // 3)[vs, us] * (1.0 if t % 3 == 0 else 2.0)
        right[t] = np.stack([(us + 0.25 - cx) * z / fx, (vs + 0.25 - cy) * z / fy, z], axis=-1)
    arrays: dict[str, np.ndarray] = {"frame_idx": np.arange(12, dtype=np.int64)}
    for side, joints, valid in (("left", np.zeros_like(right), False), ("right", right, True)):
        arrays |= {f"{side}_params": np.zeros((12, 48), np.float32), f"{side}_betas": np.zeros(10, np.float32),
                   f"{side}_trans": np.zeros((12, 3), np.float32), f"{side}_joints": joints,
                   f"{side}_valid": np.full(12, valid), f"{side}_vertices": np.zeros((12, 778, 3), np.float32),
                   f"{side}_faces": np.zeros((1552, 3), np.int32), f"{side}_infilled": np.zeros(12, bool)}
    np.savez(artifact_dir / "hands.npz", **arrays)  # type: ignore[arg-type]  # numpy's stub types `allow_pickle` into **kwds
    return FakeMoGe({tag_offset + r: true_depth(r) / FALSE_SCALE for r in range(SCENE_N)})
