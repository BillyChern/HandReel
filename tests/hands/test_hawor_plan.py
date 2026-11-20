"""Tests for `fastdo.hands.hawor_plan`: crop items, source-grid and sampled slot timelines, bank offsets, groups."""

from __future__ import annotations

import numpy as np
import pytest

from fastdo.hands.hawor_plan import (
    BOX_DILATE,
    TEMPORAL_MODES,
    WINDOW,
    ClipGeometry,
    Slot,
    box_center_scale,
    plan_clip,
    plan_groups,
)

GEOM = ClipGeometry(width=640, height=360, focal_px=500.0)


def _boxes(n: int, detected: list[int], seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    boxes = np.full((n, 5), np.nan)
    for row in detected:
        x1, y1 = rng.uniform(50, 400), rng.uniform(30, 200)
        boxes[row] = [x1, y1, x1 + rng.uniform(40, 120), y1 + rng.uniform(40, 120), rng.uniform(0.3, 0.9)]
    return boxes


def _fixture() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    frame_idx = np.arange(16, dtype=np.int64) * 3
    boxes = {"left": _boxes(16, [0, 1, 2, 3, 4, *range(6, 16)], seed=1), "right": _boxes(16, list(range(16)), seed=2)}
    return frame_idx, boxes


def test_constants() -> None:
    assert BOX_DILATE == 1.2 and WINDOW == 16 and TEMPORAL_MODES == ("source_grid", "sampled")


def test_box_center_scale() -> None:
    center, scale = box_center_scale(np.array([100.0, 200.0, 180.0, 260.0]))
    assert abs(center[0] - 140.0) <= 1e-12 and abs(center[1] - 230.0) <= 1e-12
    assert abs(scale - 0.48) <= 1e-12


def _reads(slots: list[Slot], side: str) -> list[int]:
    return sorted(s.out_row for s in slots if s.side == side and s.out_row >= 0)


def test_plan_clip_source_grid_windows_and_reads() -> None:
    frame_idx, boxes = _fixture()

    items, slots = plan_clip(0, frame_idx, boxes, GEOM, bank_offset=0)

    assert len(items) == 31
    assert len(slots) == 96 and len(slots) % WINDOW == 0
    assert sum(s.side == "left" for s in slots) == 16 + 32          # timelines of 13 and 28 slots: 1 + 2 windows
    assert sum(s.side == "right" for s in slots) == 48              # one 46-slot timeline: 3 windows
    assert _reads(slots, "left") == [0, 1, 2, 3, 4, *range(6, 16)]
    assert _reads(slots, "right") == list(range(16))
    bank_of = {(it.side, it.row): i for i, it in enumerate(items)}
    for slot in slots:
        assert slot.clip == 0
        assert any(np.isclose(slot.alpha, a) for a in (0.0, 1 / 3, 2 / 3))
        assert slot.flip == (slot.side == "left")
        if slot.out_row >= 0:
            assert slot.alpha == 0.0 and slot.bank0 == slot.bank1 == bank_of[(slot.side, slot.out_row)]
    # the first left window: 13 timeline slots (frames 0..12), then 3 copies of the last one marked unread
    assert [s.out_row for s in slots[13:16]] == [-1, -1, -1]
    assert all(s.bank0 == slots[12].bank0 and s.center == slots[12].center for s in slots[13:16])
    for item in items:
        cx = box_center_scale(boxes[item.side][item.row])[0][0]
        assert item.flip == (item.side == "left")
        assert item.center[0] == pytest.approx(640 - cx - 1 if item.side == "left" else cx, abs=1e-9)
    left_read = next(s for s in slots if s.side == "left" and s.out_row == 7)
    assert left_read.center == items[bank_of[("left", 7)]].center


def test_plan_clip_sampled_mode() -> None:
    frame_idx, boxes = _fixture()

    items, slots = plan_clip(0, frame_idx, boxes, GEOM, bank_offset=0, temporal="sampled")

    assert len(items) == 31
    assert len(slots) == 48
    assert all(s.alpha == 0.0 for s in slots)
    assert sum(s.out_row >= 0 for s in slots) == 31


def test_plan_clip_irregular_stride() -> None:
    frame_idx = np.array([0, 3, 7, 10], dtype=np.int64)
    boxes = {"left": np.full((4, 5), np.nan), "right": _boxes(4, [0, 1, 2, 3])}

    items, slots = plan_clip(0, frame_idx, boxes, GEOM, bank_offset=0)

    assert len(items) == 4 and len(slots) == 16
    assert sum(s.out_row >= 0 for s in slots) == 4
    at5 = slots[5]                                                   # timeline u = 0..10, slot index == u
    assert (at5.bank0, at5.bank1) == (1, 2) and at5.alpha == pytest.approx(0.5, abs=1e-12)
    c1, c2 = items[1].center, items[2].center
    assert at5.center[0] == pytest.approx((c1[0] + c2[0]) / 2, abs=1e-9)
    assert at5.scale == pytest.approx((items[1].scale + items[2].scale) / 2, abs=1e-12)


def test_plan_clip_bank_offset() -> None:
    frame_idx, boxes = _fixture()

    items, slots = plan_clip(3, frame_idx, boxes, GEOM, bank_offset=100)

    assert len(items) == 31
    assert {s.bank0 for s in slots} | {s.bank1 for s in slots} <= set(range(100, 131))
    assert all(s.clip == 3 for s in slots) and all(it.clip == 3 for it in items)
    reads = {s.bank0 for s in slots if s.out_row >= 0}
    assert reads == set(range(100, 131))


def test_plan_clip_rejects_unknown_temporal() -> None:
    frame_idx, boxes = _fixture()
    with pytest.raises(ValueError, match="temporal"):
        plan_clip(0, frame_idx, boxes, GEOM, bank_offset=0, temporal="interp30")


def test_plan_groups_closes_on_clip_and_crop_limits() -> None:
    assert plan_groups([3000, 2000, 500], clips_per_group=8, max_crops_per_group=4096) == [[0], [1, 2]]
    assert plan_groups([5000, 10], clips_per_group=8, max_crops_per_group=4096) == [[0], [1]]
    assert plan_groups([10, 5000, 10], clips_per_group=8, max_crops_per_group=4096) == [[0], [1], [2]]
    assert plan_groups([1, 1, 1], clips_per_group=2, max_crops_per_group=4096) == [[0, 1], [2]]
    assert plan_groups([], clips_per_group=2, max_crops_per_group=4096) == []
