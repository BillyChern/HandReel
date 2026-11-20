"""Configuration of the amortized tracker network."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from fastdo.tracker.features import FRAME_FEATURE_DIM, frame_feature_dim

__all__ = ["TrackerConfig"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrackerConfig:
    """Sizes and output ranges of `AmortizedTracker`; every constant is fixed a priori (data rules).

    `feature_dim` must equal `frame_feature_dim(n_params, n_bins, n_clusters)`. `damping_log10_range` bounds the
    per-clip Marquardt damping; `damping_init` (the fine level's `lam0`) and `weight_init` set the zero-initialized
    output heads' biases so an untrained model starts as the solver.
    """

    feature_dim: int = FRAME_FEATURE_DIM
    n_params: int = 7
    n_bins: int = 16
    n_clusters: int = 64
    width: int = 384
    n_layers: int = 6
    n_heads: int = 6
    ffn_ratio: int = 4
    max_distance: int = 32
    head_hidden: int = 128
    max_tracks: int = 128
    weight_floor: float = 1e-3
    weight_init: float = 0.9
    precision_floor: float = 1e-6
    damping_log10_range: tuple[float, float] = (-6.0, 2.0)
    damping_init: float = 1e-3

    def __post_init__(self) -> None:
        expected = frame_feature_dim(self.n_params, self.n_bins, self.n_clusters)
        if self.feature_dim != expected:
            raise ValueError(f"feature_dim {self.feature_dim} != frame_feature_dim(n_params={self.n_params}, "
                             f"n_bins={self.n_bins}, n_clusters={self.n_clusters}) = {expected}")
        if self.n_params < 7:
            raise ValueError(f"n_params must be >= 7 (pose and log-scale), got {self.n_params}")
        sizes = {"n_layers": self.n_layers, "n_heads": self.n_heads, "ffn_ratio": self.ffn_ratio,
                 "max_distance": self.max_distance, "head_hidden": self.head_hidden, "max_tracks": self.max_tracks}
        for name, value in sizes.items():
            if value < 1:
                raise ValueError(f"{name} must be >= 1, got {value}")
        if self.width % self.n_heads != 0:
            raise ValueError(f"width {self.width} is not divisible by n_heads {self.n_heads}")
        if not 0.0 < self.weight_floor < self.weight_init < 1.0:
            raise ValueError(f"need 0 < weight_floor < weight_init < 1, got {self.weight_floor}, {self.weight_init}")
        if self.precision_floor <= 0.0:
            raise ValueError(f"precision_floor must be > 0, got {self.precision_floor}")
        lo, hi = self.damping_log10_range
        if not lo < hi:
            raise ValueError(f"damping_log10_range must be increasing, got {self.damping_log10_range}")
        if not 10.0 ** lo <= self.damping_init <= 10.0 ** hi:
            raise ValueError(f"damping_init {self.damping_init} outside [1e{lo:g}, 1e{hi:g}]")
