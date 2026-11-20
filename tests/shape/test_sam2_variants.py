"""Tests for `fastdo.shape.sam2_variants` (D3, R4): variant parsing, the default-selection rule on a fabricated
sweep table, and reduced-resolution propagation against a resolution-aware fake predictor."""

from __future__ import annotations

import inspect
import math
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
import torch

from fastdo.engine.workers import make_mask_worker
from fastdo.shape.sam2_variants import (
    DEFAULT_SAM2_VARIANT,
    REFERENCE_LABEL,
    SAM2_VARIANTS,
    choose_sam2_variant,
    parse_sam2_variant,
    propagate_masks_at,
    resize_short_side,
    variant_label,
)

# A fabricated sweep table, shaped like bench_sam2_variants.py's JSON rows (seconds and IoU are invented). Rows at the
# demo's native 720p carry the bare size as their label, as `variant_label` writes them.
_TABLE = [
    {"label": "large", "seconds": 13.3, "mean_iou": 1.0},
    {"label": "large@512", "seconds": 12.9, "mean_iou": 0.97},
    {"label": "base_plus", "seconds": 8.1, "mean_iou": 0.95},
    {"label": "base_plus@512", "seconds": 7.7, "mean_iou": 0.93},
    {"label": "small", "seconds": 5.2, "mean_iou": 0.91},
    {"label": "small@512", "seconds": 4.9, "mean_iou": 0.89},
    {"label": "tiny", "seconds": 4.1, "mean_iou": 0.86},
    {"label": "tiny@512", "seconds": 3.8, "mean_iou": 0.84},
]


def test_choose_sam2_variant_picks_the_fastest_with_mean_iou_at_least_090() -> None:
    assert choose_sam2_variant(_TABLE) == "small"
    assert choose_sam2_variant(_TABLE, min_iou=0.94) == "base_plus"
    assert choose_sam2_variant(_TABLE, min_iou=0.89) == "small@512"            # the bound is inclusive


def test_choose_sam2_variant_falls_back_to_large_at_native_resolution() -> None:
    assert REFERENCE_LABEL == DEFAULT_SAM2_VARIANT == "large"
    no_reference = [row for row in _TABLE if row["label"] != "large"]
    assert choose_sam2_variant(no_reference, min_iou=0.99) == "large"
    assert choose_sam2_variant([], min_iou=0.90) == "large"


def test_choose_sam2_variant_ignores_non_finite_rows_and_breaks_ties_by_table_order() -> None:
    rows = [{"label": "tiny@512", "seconds": 1.0, "mean_iou": math.nan},
            {"label": "small", "seconds": 2.0, "mean_iou": 0.95},
            {"label": "small@512", "seconds": 2.0, "mean_iou": 0.96}]
    assert choose_sam2_variant(rows) == "small"


def test_a_chosen_native_row_runs_clips_at_their_own_native_resolution() -> None:
    assert variant_label("small", 720, native_short_side=720) == "small"
    assert variant_label("small", 1080, native_short_side=720) == "small"      # not a downscale either
    assert variant_label("small", 512, native_short_side=720) == "small@512"
    assert parse_sam2_variant(choose_sam2_variant(_TABLE)).short_side is None  # a 1080p clip stays at 1080p
    with pytest.raises(ValueError, match="variant"):
        choose_sam2_variant([{"label": "small@720p", "seconds": 1.0, "mean_iou": 0.99}])
    with pytest.raises(ValueError, match="short sides"):
        variant_label("small", 0, native_short_side=720)


def test_parse_sam2_variant() -> None:
    assert set(SAM2_VARIANTS) == {"large", "base_plus", "small", "tiny"}
    assert SAM2_VARIANTS["base_plus"] == "facebook/sam2.1-hiera-base-plus"
    large = parse_sam2_variant("large")
    assert (large.variant, large.short_side, large.model_id) == ("large", None, "facebook/sam2.1-hiera-large")
    small = parse_sam2_variant("small@512")
    assert (small.variant, small.short_side, small.label) == ("small", 512, "small@512")
    for bad in ("huge", "small@", "small@-3", "small@abc"):
        with pytest.raises(ValueError, match="variant"):
            parse_sam2_variant(bad)


def test_mask_worker_defaults_to_large_until_the_sweep_is_recorded() -> None:
    assert DEFAULT_SAM2_VARIANT == "large"
    assert inspect.signature(make_mask_worker).parameters["variant"].default == "large"


def test_resize_short_side_downscales_only() -> None:
    frames = torch.randint(0, 256, (2, 12, 20, 3), dtype=torch.uint8)
    assert resize_short_side(frames, None) is frames
    assert resize_short_side(frames, 12) is frames and resize_short_side(frames, 64) is frames
    small = resize_short_side(frames, 6)
    assert small.shape == (2, 6, 10, 3) and small.dtype == torch.uint8


class _EchoPredictor:
    """Resolution-aware fake: every propagated frame gets the prompt mask it was given, at the prompt's own
    resolution, and it records the frame size it saw."""

    def __init__(self) -> None:
        self.prompt: np.ndarray | None = None

    def init_state(self, video_path: str) -> dict[str, str]:
        return {"video_path": video_path}

    def add_new_mask(self, state: dict[str, str], frame_idx: int, obj_id: int, mask: np.ndarray) -> None:
        self.prompt = np.asarray(mask, dtype=bool)

    def propagate_in_video(self, state: dict[str, str], start_frame_idx: int,
                           reverse: bool = False) -> Iterator[tuple[int, list[int], Any]]:
        assert self.prompt is not None
        logits = torch.where(torch.from_numpy(self.prompt), 10.0, -10.0)[None, None]
        frames = range(start_frame_idx, -1, -1) if reverse else range(start_frame_idx, 4)
        for fidx in frames:
            yield fidx, [1], logits


def test_propagate_masks_at_reduced_short_side_returns_native_resolution_masks() -> None:
    pytest.importorskip("PIL")
    frames = torch.zeros(4, 8, 12, 3, dtype=torch.uint8)
    prompt = torch.zeros(8, 12, dtype=torch.bool)
    prompt[2:6, 4:10] = True                                        # aligned to the 2-px grid of a 4x6 frame
    predictor = _EchoPredictor()

    masks = propagate_masks_at(frames, 1, prompt, short_side=4, predictor=predictor, device="cpu")

    assert predictor.prompt is not None and predictor.prompt.shape == (4, 6)
    assert masks.shape == (4, 8, 12) and masks.dtype == torch.bool
    assert bool((masks == prompt[None]).all())
