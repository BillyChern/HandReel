"""Gate input features: 8 uniformly sampled encoder rows of a clip, plus a hand-confidence column.

The gate reads the encoder stage's `encoder.npz`: `tokens (N, D)` float32, `frame_idx (N,)` int64 and a 0-d
`crop` string (`fastdo.engine.workers_frames.CROP_MODES`). An archive without `crop` predates the key, when the only
crop was `"center"`. `hand_conf` is zeros until a hand detector runs before the gate.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from fastdo.engine.worker_support import load_npz, require_artifact
from fastdo.engine.workers_frames import ENCODER_FILE

__all__ = ["DEFAULT_CROP", "N_GATE_FRAMES", "gate_features", "read_gate_tokens", "sample_rows"]

N_GATE_FRAMES: int = 8
DEFAULT_CROP = "center"      # the crop of encoder archives written before the `crop` key existed


def sample_rows(n_rows: int, k: int = N_GATE_FRAMES) -> np.ndarray:
    """`(k,)` int64 rows spread uniformly over `[0, n_rows - 1]`: `round(linspace(0, n_rows - 1, k))`.

    Rows repeat when `n_rows < k`. `linspace` never lands on a .5 here (`i * (n_rows - 1) / 7` has an odd denominator),
    so numpy's round-half-even never matters.

    Raises:
        ValueError: If `n_rows < 1` or `k < 1`.
    """
    if n_rows < 1 or k < 1:
        raise ValueError(f"sample_rows needs n_rows >= 1 and k >= 1, got n_rows {n_rows} and k {k}")
    return np.round(np.linspace(0, n_rows - 1, k)).astype(np.int64)


def gate_features(encoder_tokens: Tensor, hand_conf: Tensor | None = None) -> Tensor:
    """`(N_GATE_FRAMES, D + 1)` float32: the tokens at `sample_rows(N)` with the matching `hand_conf` as last column.

    Args:
        encoder_tokens: `(N, D)` per-row encoder descriptors, N >= 1.
        hand_conf: `(N,)` hand confidence in [0, 1]; None gives zeros.

    Raises:
        ValueError: On a shape mismatch, N == 0, a non-finite token, or `hand_conf` outside [0, 1].
    """
    if encoder_tokens.ndim != 2 or int(encoder_tokens.shape[0]) == 0:
        raise ValueError(f"encoder tokens must be (N, D) with N >= 1, got {tuple(encoder_tokens.shape)}")
    n = int(encoder_tokens.shape[0])
    if not bool(torch.isfinite(encoder_tokens).all()):
        raise ValueError("encoder tokens hold non-finite values")
    if hand_conf is None:
        conf = torch.zeros(n, dtype=torch.float32)
    else:
        if tuple(hand_conf.shape) != (n,):
            raise ValueError(f"hand_conf must be ({n},), matching the tokens, got {tuple(hand_conf.shape)}")
        if not bool((torch.isfinite(hand_conf) & (hand_conf >= 0) & (hand_conf <= 1)).all()):
            raise ValueError("hand_conf must lie in [0, 1]")
        conf = hand_conf.float()
    rows = torch.from_numpy(sample_rows(n))
    return torch.cat([encoder_tokens[rows].float(), conf[rows, None]], dim=1)


def read_gate_tokens(artifact_dir: Path) -> tuple[Tensor, np.ndarray, str]:
    """`(tokens (N, D) float32, frame_idx (N,) int64, crop)` from `<artifact_dir>/encoder.npz`.

    Raises:
        FileNotFoundError: If the encoder stage has not run for the clip.
        ValueError: If `tokens` is not 2-D, or its row count differs from `frame_idx`'s.
    """
    data = load_npz(require_artifact(Path(artifact_dir) / ENCODER_FILE, "encoder"))
    tokens = np.asarray(data["tokens"], dtype=np.float32)
    frame_idx = np.asarray(data["frame_idx"], dtype=np.int64)
    if tokens.ndim != 2 or frame_idx.ndim != 1 or tokens.shape[0] != frame_idx.shape[0]:
        raise ValueError(f"{Path(artifact_dir) / ENCODER_FILE}: tokens {tokens.shape} and frame_idx {frame_idx.shape} "
                         "must have one row per decode row")
    crop = str(data["crop"]) if "crop" in data else DEFAULT_CROP
    return torch.from_numpy(tokens), frame_idx, crop
