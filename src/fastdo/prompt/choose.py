"""Which frame, and which object, the mask stage should be prompted with.

The mask stage needs one `(row, mask)` pair per window: the object a hand is manipulating, on a frame where it is
being manipulated. Raw internet video carries no label for that, so a hand-object detector supplies it.
This module holds the part that needs no detector, no GPU and no pixels: turning
per-frame detections into one choice, or into a REFUSAL.

**Why refusal is the important half.** The yield law's numerator is the rate of usable trajectories, and a prompt
stage that always answers converts every unusable window into an accepted one. Screening, not solving, is what
decides internet-scale throughput, so a window with no confident
hand-object contact must fail loudly and be counted, never be prompted with a guess.

**Why the score is `interaction_score * p_held_like`.** Hands23 reports a 5-way contact state, an interaction score,
and a "touch" distribution over how the object is touched. `p_held_like` is the mass on the held-like classes, which
is what separates a portable object being manipulated from a fixed surface being leaned on. The product demands both:
confident interaction AND a held object.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

__all__ = ["GATES", "HAND_CONTACT_OBJECT", "PromptChoice", "PromptThresholds", "candidate_rows", "choose_prompt",
           "first_failed_gate", "gate_trace", "mask_borders", "mask_is_plausible"]

HAND_CONTACT_OBJECT = 3   # Hands23's contact class for "object_contact" (its training-order mapping)
_BORDER_LIMIT = 3         # touching this many frame borders is a background blowout, not an object


@dataclass(frozen=True)
class PromptThresholds:
    """The gates a prompt must clear.

    Attributes:
        min_score: Lowest acceptable `interaction_score * p_held_like`. Set from CALIB or fixed a priori; the stage
            config has no accept-all default.
        hand_score: Lowest hand-detection score.
        obj_score: Lowest object-detection score.
        area_frac: Allowed mask area as a fraction of the frame, `(low, high)`.
    """

    min_score: float = 0.0
    hand_score: float = 0.9
    obj_score: float = 0.9
    area_frac: tuple[float, float] = field(default=(0.002, 0.40))

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_score <= 1.0:
            raise ValueError(f"min_score must be in [0, 1], got {self.min_score}")
        for name in ("hand_score", "obj_score"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {getattr(self, name)}")
        lo, hi = self.area_frac
        if not 0.0 <= lo < hi <= 1.0:
            raise ValueError(f"area_frac must satisfy 0 <= low < high <= 1, got {self.area_frac}")


@dataclass(frozen=True)
class PromptChoice:
    """The chosen prompt: a decode row, the object box on it, and why it won."""

    row: int
    box: list[float]
    score: float
    side: str
    hand_score: float
    obj_score: float
    interaction_score: float
    p_held_like: float


def candidate_rows(n_rows: int, stride: int) -> list[int]:
    """Decode rows to run the detector on: every `stride`-th row, always including the first and the last.

    Raises:
        ValueError: If `stride < 1` or `n_rows < 1`.
    """
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    if n_rows < 1:
        raise ValueError(f"n_rows must be >= 1, got {n_rows}")
    rows = sorted({*range(0, n_rows, stride), n_rows - 1})
    return rows


def choose_prompt(per_row: Mapping[int, Sequence[Mapping[str, object]]],
                  thresholds: PromptThresholds) -> PromptChoice | None:
    """The best (row, object) pair over the detector's per-row hand records, or None to REFUSE.

    A record must have a confident hand, `contact_idx == HAND_CONTACT_OBJECT`, and a linked object box with a
    confident score. Among those, the highest `interaction_score * p_held_like` wins; ties break by lower row, then
    larger box area, then smaller x0, so the answer never depends on iteration order.

    Args:
        per_row: `decode row -> the row's hand records` (Hands23's parsed fields).
        thresholds: The gates to apply.

    Returns:
        The choice, or None when nothing clears the gates.
    """
    best: tuple[float, int, float, float] | None = None
    choice: PromptChoice | None = None
    for row in sorted(per_row):
        for rec in per_row[row]:
            box = rec.get("obj_box")
            if box is None:
                continue
            if float(rec.get("hand_score", 0.0)) < thresholds.hand_score:
                continue
            if int(rec.get("contact_idx", -1)) != HAND_CONTACT_OBJECT:
                continue
            if float(rec.get("obj_score", 0.0)) < thresholds.obj_score:
                continue
            score = float(rec.get("interaction_score", 0.0)) * float(rec.get("p_held_like", 0.0))
            if score < thresholds.min_score or score <= 0.0:
                continue
            b = [float(v) for v in box]  # type: ignore[union-attr]
            area = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
            key = (-score, row, -area, b[0])
            if best is None or key < best:
                best = key
                choice = PromptChoice(row=int(row), box=b, score=score, side=str(rec.get("side", "")),
                                      hand_score=float(rec.get("hand_score", 0.0)),
                                      obj_score=float(rec.get("obj_score", 0.0)),
                                      interaction_score=float(rec.get("interaction_score", 0.0)),
                                      p_held_like=float(rec.get("p_held_like", 0.0)))
    return choice


#: The gates `choose_prompt` applies to a record, in the order it applies them (`first_failed_gate`).
GATES: tuple[str, ...] = ("obj_box", "hand_score", "contact_idx", "obj_score", "score")


def first_failed_gate(rec: Mapping[str, object], thresholds: PromptThresholds) -> str | None:
    """The first of `GATES` that `choose_prompt` rejects `rec` on, or None when the record is eligible.

    The same predicate as the body of `choose_prompt`'s loop, in the same order; `tests/prompt/test_choose.py`
    checks on randomised records that `choose_prompt` picks exactly among the records this returns None for.
    """
    if rec.get("obj_box") is None:
        return "obj_box"
    if float(rec.get("hand_score", 0.0)) < thresholds.hand_score:  # type: ignore[arg-type]
        return "hand_score"
    if int(rec.get("contact_idx", -1)) != HAND_CONTACT_OBJECT:  # type: ignore[call-overload]
        return "contact_idx"
    if float(rec.get("obj_score", 0.0)) < thresholds.obj_score:  # type: ignore[arg-type]
        return "obj_score"
    score = float(rec.get("interaction_score", 0.0)) * float(rec.get("p_held_like", 0.0))  # type: ignore[arg-type]
    if score < thresholds.min_score or score <= 0.0:
        return "score"
    return None


def gate_trace(per_row: Mapping[int, Sequence[Mapping[str, object]]],
               thresholds: PromptThresholds) -> list[dict[str, object]]:
    """Every record of every row with its `row` and `first_failed_gate` (None = eligible), rows ascending.

    A refusal used to leave no record, so which gate a no-contact window failed was unknowable. Pure, so
    stored records can be replayed through any gate variant on CPU.
    """
    out: list[dict[str, object]] = []
    for row in sorted(per_row):
        for rec in per_row[row]:
            out.append({"row": int(row), **dict(rec), "first_failed_gate": first_failed_gate(rec, thresholds)})
    return out


def mask_is_plausible(mask: np.ndarray, thresholds: PromptThresholds) -> bool:
    """Whether a box-prompted mask can be an object: non-empty, inside the area band, not a border blowout.

    A box prompt fails in three ways, and all three are silent without this check: it returns nothing, it returns the
    background (which touches most borders), or it returns a speck.
    """
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 2 or m.size == 0:
        return False
    frac = float(m.sum()) / float(m.size)
    lo, hi = thresholds.area_frac
    if not lo <= frac <= hi:
        return False
    return mask_borders(m) < _BORDER_LIMIT


def mask_borders(mask: np.ndarray) -> int:
    """How many of the 4 frame borders a 2-D mask touches (`mask_is_plausible` refuses at 3 or more)."""
    m = np.asarray(mask, dtype=bool)
    return int(m[0].any()) + int(m[-1].any()) + int(m[:, 0].any()) + int(m[:, -1].any())
