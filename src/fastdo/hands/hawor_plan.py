"""Crop items, head slots and clip groups for batched HaWoR (D3b).

Shared module: numpy and the standard library only; nothing from `fastdo` or the other shared modules.

`plan_clip` turns one clip's filtered boxes into backbone crop items (one per detected side and row) and head slots.
HaWoR's temporal modules run on fixed windows of `WINDOW` frames at the source frame rate, so with
`temporal="source_grid"` a detection chunk's timeline is every source frame between its first and last row. A frame
between two decoded rows interpolates their backbone features, centres and scales, and outputs are read back only at
the decoded rows. The last window of a timeline repeats its last slot, marked unread. `temporal="sampled"` feeds the
decoded rows directly as the timeline (the 10 fps ablation). Left hands use the horizontally mirrored frame, as HaWoR
does.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np

__all__ = [
    "BOX_DILATE",
    "TEMPORAL_MODES",
    "WINDOW",
    "ClipGeometry",
    "CropItem",
    "Slot",
    "box_center_scale",
    "closes_group",
    "plan_clip",
    "plan_groups",
]

BOX_DILATE = 1.2
WINDOW = 16
TEMPORAL_MODES: tuple[str, ...] = ("source_grid", "sampled")
_SIDES = ("left", "right")


@dataclass(frozen=True)
class ClipGeometry:
    """Decoded frame size and the single focal length HaWoR uses (its image centre is always `(W/2, H/2)`)."""

    width: int
    height: int
    focal_px: float


@dataclass(frozen=True)
class CropItem:
    """One backbone crop: `center` is in mirrored-image pixels when `flip` (left hand)."""

    clip: int
    row: int
    side: str
    center: tuple[float, float]
    scale: float
    flip: bool


@dataclass(frozen=True)
class Slot:
    """One head input: features `(1 - alpha) * bank[bank0] + alpha * bank[bank1]`; `out_row` is the read row or -1."""

    clip: int
    side: str
    bank0: int
    bank1: int
    alpha: float
    center: tuple[float, float]
    scale: float
    flip: bool
    out_row: int


def box_center_scale(box: np.ndarray, dilate: float = BOX_DILATE) -> tuple[tuple[float, float], float]:
    """Box centre and HaWoR scale `max(w, h) / 200 * dilate` of an `[x1, y1, x2, y2, ...]` box."""
    x1, y1, x2, y2 = (float(v) for v in np.asarray(box, dtype=np.float64).reshape(-1)[:4])
    w, h = x2 - x1, y2 - y1
    return (x1 + w / 2.0, y1 + h / 2.0), max(w, h) / 200.0 * dilate


def _runs(rows: np.ndarray) -> list[list[int]]:
    if rows.size == 0:
        return []
    return [run.tolist() for run in np.split(rows, np.flatnonzero(np.diff(rows) != 1) + 1)]


def _mix(a: CropItem, b: CropItem, alpha: float) -> tuple[tuple[float, float], float]:
    if alpha == 0.0:
        return a.center, a.scale
    center = ((1.0 - alpha) * a.center[0] + alpha * b.center[0], (1.0 - alpha) * a.center[1] + alpha * b.center[1])
    return center, (1.0 - alpha) * a.scale + alpha * b.scale


def _timeline(run: list[int], frame_idx: np.ndarray, item_of: Mapping[int, tuple[int, CropItem]],
              temporal: str) -> list[Slot]:
    def slot(j: int, k: int, alpha: float, out_row: int) -> Slot:
        (bank0, a), (bank1, b) = item_of[j], item_of[k]
        center, scale = _mix(a, b, alpha)
        return Slot(a.clip, a.side, bank0, bank1, alpha, center, scale, a.flip, out_row)

    if temporal == "sampled":
        return [slot(j, j, 0.0, j) for j in run]
    timeline = []
    for pos, j in enumerate(run):
        timeline.append(slot(j, j, 0.0, j))
        if pos + 1 < len(run):
            k = run[pos + 1]
            f0, f1 = int(frame_idx[j]), int(frame_idx[k])
            timeline.extend(slot(j, k, (u - f0) / (f1 - f0), -1) for u in range(f0 + 1, f1))
    return timeline


def _windows(timeline: list[Slot]) -> list[Slot]:
    slots: list[Slot] = []
    for start in range(0, len(timeline), WINDOW):
        window = timeline[start:start + WINDOW]
        slots.extend(window)
        slots.extend([replace(window[-1], out_row=-1)] * (WINDOW - len(window)))
    return slots


def plan_clip(clip: int, frame_idx: np.ndarray, boxes: Mapping[str, np.ndarray], geom: ClipGeometry,
              bank_offset: int, temporal: str = "source_grid") -> tuple[list[CropItem], list[Slot]]:
    """Crop items and head slots of one clip (see the module docstring).

    Args:
        clip: The clip's index in its group (`Slot.clip`, `CropItem.clip`).
        frame_idx: `(T,)` strictly increasing source frame indices of the decoded rows.
        boxes: `"left"` / `"right"` -> `(T,5)` filtered boxes, NaN rows where undetected; a missing side has none.
        geom: The clip's geometry (the width mirrors left-hand centres).
        bank_offset: Feature-bank index of this clip's first crop item.
        temporal: One of `TEMPORAL_MODES`.

    Returns:
        `(items, slots)`: items in side order (left, right) then row order, item `i` at bank index `bank_offset + i`;
        slots in whole windows of `WINDOW`, every detected (side, row) read exactly once at `alpha == 0`.

    Raises:
        ValueError: On an unknown `temporal`, a non-increasing `frame_idx`, or boxes that are not `(T,5)`.
    """
    if temporal not in TEMPORAL_MODES:
        raise ValueError(f"temporal must be one of {TEMPORAL_MODES}, got {temporal!r}")
    idx = np.asarray(frame_idx, dtype=np.int64).reshape(-1)
    if idx.size > 1 and not bool((np.diff(idx) > 0).all()):
        raise ValueError("frame_idx must be strictly increasing")
    items: list[CropItem] = []
    runs: dict[str, list[list[int]]] = {}
    item_of: dict[str, dict[int, tuple[int, CropItem]]] = {}
    for side in _SIDES:
        item_of[side], runs[side] = {}, []
        if side not in boxes:
            continue
        arr = np.asarray(boxes[side], dtype=np.float64)
        if arr.shape != (idx.size, 5):
            raise ValueError(f"boxes[{side!r}] must be ({idx.size},5), got {arr.shape}")
        rows = np.flatnonzero(np.isfinite(arr).all(axis=1))
        for row in rows.tolist():
            (cx, cy), scale = box_center_scale(arr[row])
            flip = side == "left"
            item = CropItem(clip, row, side, (geom.width - cx - 1.0 if flip else cx, cy), scale, flip)
            item_of[side][row] = (bank_offset + len(items), item)
            items.append(item)
        runs[side] = _runs(rows)
    slots: list[Slot] = []
    for side in _SIDES:
        for run in runs[side]:
            slots.extend(_windows(_timeline(run, idx, item_of[side], temporal)))
    return items, slots


def closes_group(group_clips: int, group_crops: int, next_crops: int, clips_per_group: int,
                 max_crops_per_group: int) -> bool:
    """Whether a non-empty group must close before a clip with `next_crops` crops joins it: it already holds
    `clips_per_group` clips, or the crops would exceed `max_crops_per_group`. A lone clip over the limit runs alone."""
    return group_clips > 0 and (group_clips >= clips_per_group or group_crops + next_crops > max_crops_per_group)


def plan_groups(crop_counts: Sequence[int], clips_per_group: int, max_crops_per_group: int) -> list[list[int]]:
    """Clip indices grouped in order by `closes_group`.

    Raises:
        ValueError: If `clips_per_group` or `max_crops_per_group` is below 1.
    """
    if clips_per_group < 1 or max_crops_per_group < 1:
        raise ValueError(f"clips_per_group and max_crops_per_group must be >= 1, got {clips_per_group}, "
                         f"{max_crops_per_group}")
    groups: list[list[int]] = []
    current: list[int] = []
    crops = 0
    for index, count in enumerate(crop_counts):
        if closes_group(len(current), crops, int(count), clips_per_group, max_crops_per_group):
            groups.append(current)
            current, crops = [], 0
        current.append(index)
        crops += int(count)
    if current:
        groups.append(current)
    return groups
