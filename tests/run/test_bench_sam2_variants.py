"""Tests for `run/pipeline/bench_sam2_variants.py` (D3, R4) with an injected fake predictor and clock.

No GPU run happens here: the real sweep runs in an exclusive GPU window.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir

from run.pipeline import bench_sam2_variants as bench

_REPO = Path(__file__).resolve().parents[2]
_COST = {"facebook/sam2.1-hiera-large": 8.0, "facebook/sam2.1-hiera-tiny": 2.0}   # fake seconds at 16x24


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class _SweepPredictor:
    """Echoes the prompt it was given on every frame (tiny drops odd frames) and advances the fake clock by a
    per-model cost proportional to the prompt's pixel count."""

    def __init__(self, model_id: str, clock: _Clock) -> None:
        self.model_id, self.clock = model_id, clock
        self.prompt: np.ndarray | None = None
        self.n_runs = 0

    def init_state(self, video_path: str) -> dict[str, str]:
        self.n_runs += 1
        return {"video_path": video_path}

    def add_new_mask(self, state: dict[str, str], frame_idx: int, obj_id: int, mask: np.ndarray) -> None:
        self.prompt = np.asarray(mask, dtype=bool)
        self.clock.now += _COST[self.model_id] * self.prompt.size / (16 * 24)

    def propagate_in_video(self, state: dict[str, str], start_frame_idx: int,
                           reverse: bool = False) -> Iterator[tuple[int, list[int], Any]]:
        assert self.prompt is not None
        n = len(list(Path(state["video_path"]).glob("*.jpg")))
        frames = range(start_frame_idx, -1, -1) if reverse else range(start_frame_idx, n)
        for fidx in frames:
            keep = not (self.model_id.endswith("tiny") and fidx % 2 == 1)
            logits = torch.where(torch.from_numpy(self.prompt & keep), 10.0, -10.0)[None, None]
            yield fidx, [1], logits


def _inputs() -> tuple[torch.Tensor, int, torch.Tensor]:
    frames = torch.zeros(4, 16, 24, 3, dtype=torch.uint8)
    prompt = torch.zeros(16, 24, dtype=torch.bool)
    prompt[4:12, 6:18] = True                                        # aligned to the 2-px grid of the 8x12 run
    return frames, 1, prompt


def _sweep(clock: _Clock, loads: list[str], predictors: list[_SweepPredictor],
           warmup_frames: int = 0) -> list[dict[str, Any]]:
    def load(model_id: str) -> _SweepPredictor:
        loads.append(model_id)
        clock.now += 100.0                                           # model load: must not be timed
        predictors.append(_SweepPredictor(model_id, clock))
        return predictors[-1]

    frames, keyframe_index, prompt = _inputs()
    return bench.run_sweep(frames, keyframe_index, prompt, sizes=("large", "tiny"), short_sides=(16, 8),
                           load_predictor=load, device="cpu", warmup_frames=warmup_frames, clock=clock)


def test_run_sweep_times_propagation_only_and_scores_iou_against_the_first_row() -> None:
    pytest.importorskip("PIL")
    clock, loads, predictors = _Clock(), list[str](), list[_SweepPredictor]()

    rows = _sweep(clock, loads, predictors)

    assert [row["label"] for row in rows] == ["large", "large@8", "tiny", "tiny@8"]   # 16 px is native: bare size
    assert loads == ["facebook/sam2.1-hiera-large", "facebook/sam2.1-hiera-tiny"]   # one load per size
    assert [row["seconds"] for row in rows] == pytest.approx([8.0, 2.0, 2.0, 0.5])
    assert [row["mean_iou"] for row in rows] == pytest.approx([1.0, 1.0, 0.5, 0.5])
    assert [row["min_frame_iou"] for row in rows] == pytest.approx([1.0, 1.0, 0.0, 0.0])
    assert [row["sam2_input_hw"] for row in rows] == [[16, 24], [8, 12], [16, 24], [8, 12]]
    assert all(row["n_frames"] == 4 for row in rows)
    assert rows[1]["model_id"] == "facebook/sam2.1-hiera-large" and rows[1]["short_side"] == 8


def test_run_sweep_warm_up_is_untimed() -> None:
    pytest.importorskip("PIL")
    clock, loads, predictors = _Clock(), list[str](), list[_SweepPredictor]()

    rows = _sweep(clock, loads, predictors, warmup_frames=2)

    assert [row["seconds"] for row in rows] == pytest.approx([8.0, 2.0, 2.0, 0.5])
    assert [p.n_runs for p in predictors] == [4, 4]                  # 2 short sides x (warm-up + timed run)


def test_sweep_report_writes_json_and_markdown_with_the_chosen_default(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    rows = _sweep(_Clock(), [], [])

    report = bench.sweep_report(rows, min_iou=0.90, meta={"video": "fake.mp4", "bf16": True})
    json_path, md_path = bench.write_sweep_outputs(tmp_path / "out", report)

    data = json.loads(json_path.read_text())
    assert data["rows"] == rows
    assert data["reference"] == "large" and data["min_iou"] == 0.90
    assert data["chosen"] == "large@8"                               # fastest row with mean IoU >= 0.90
    assert data["meta"] == {"video": "fake.mp4", "bf16": True}
    table = md_path.read_text()
    assert "| label |" in table and all(f"| {row['label']} |" in table for row in rows)


def test_sweep_labels_and_config() -> None:
    assert bench.sweep_labels(("large", "small"), (720, 512), native_short_side=720) == [
        "large", "large@512", "small", "small@512"]
    assert bench.sweep_labels(("small",), (720, 512), native_short_side=1080) == ["small@720", "small@512"]
    with initialize_config_dir(config_dir=str(_REPO / "run/conf"), version_base=None):
        cfg = compose(config_name="config")
    assert list(cfg.sam2_bench.sizes) == ["large", "base_plus", "small", "tiny"]
    assert list(cfg.sam2_bench.short_sides) == [720, 512]
    assert cfg.sam2_bench.bf16 is True and cfg.sam2_bench.min_iou == 0.90
