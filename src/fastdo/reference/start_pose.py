"""Consumer hook: the consolidated start pose at a contact onset.

`start_pose_at_onset(ref, onset_row)` returns the consolidated pose of the rest segment that ENDS at the contact
onset, or None with a reason. A segment qualifies when it was consolidated, lies entirely before `onset_row`, and every
row strictly between its last row and `onset_row` is `near` or `guard` (no touched, unknown-hand, no-pose or other
rest row), with at most `max_gap_rows` of them. It reads the conditioner's output only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fastdo.reference.conditioner import ConditionedReference
from fastdo.reference.frames import WorldSource, to_record

__all__ = ["StartPose", "start_pose_at_onset"]

_GAP_OK = ("near", "guard")


@dataclass(frozen=True)
class StartPose:
    rotation_world: np.ndarray        # (3,3) O
    centre_world: np.ndarray          # (3,) p = world position of c*
    pose_world: np.ndarray            # (4,4) origin-referenced world pose (O, p - O c*)
    pose_record_at_onset: np.ndarray  # (4,4) pre-ego record pose of the consolidated state at `onset_row`
    segment_rows: tuple[int, int]     # first, last row of the segment
    n_vote: int
    n_replaced: int
    spread: dict                      # weighted RMS / p95 / max: centre mm, angle deg (voting rows)
    covariance: np.ndarray            # (6,6) Laplace covariance of the mean [centre; world-left rotation]
    world_source: str
    gap_rows: int                     # rows between the segment's end and the onset


def start_pose_at_onset(ref: ConditionedReference, onset_row: int, max_gap_rows: int = 10
                        ) -> tuple[StartPose | None, str]:
    """`(StartPose, "ok")` or `(None, reason)` (module docstring). Reasons: `no_world_frame`, `onset_out_of_range`,
    `no_segment_before_onset`, `segment_refused:<reason>`, `segment_short`, `gap_not_clear:<state>`, `gap_too_long`."""
    n = len(ref.row_state)
    if not 0 <= int(onset_row) < n:
        return None, "onset_out_of_range"
    if ref.world_source == "none":
        return None, "no_world_frame"
    before = [s for s in ref.segments if int(s.rows[-1]) < int(onset_row)]
    if not before:
        return None, "no_segment_before_onset"
    seg = max(before, key=lambda s: int(s.rows[-1]))
    gap = np.arange(int(seg.rows[-1]) + 1, int(onset_row))
    bad = [str(ref.row_state[r]) for r in gap if ref.row_state[r] not in _GAP_OK]
    if bad:
        return None, f"gap_not_clear:{bad[0]}"
    if len(gap) > max_gap_rows:
        return None, "gap_too_long"
    if seg.status == "rest_short":
        return None, "segment_short"
    if seg.status != "consolidated":
        return None, f"segment_refused:{seg.reason}"
    O, p = np.asarray(seg.O), np.asarray(seg.p)
    pw = np.eye(4)
    pw[:3, :3] = O
    pw[:3, 3] = p - O @ ref.c_star
    w = ref.world
    one = WorldSource(w.kind, w.Q[[onset_row]], w.u[[onset_row]], w.sigma[[onset_row]], w.reason)
    rec = to_record(p[None], O[None], ref.c_star, one)[0]
    return StartPose(O, p, pw, rec, (int(seg.rows[0]), int(seg.rows[-1])), seg.n_vote, len(seg.replaced),
                     dict(seg.spread), np.asarray(seg.cov), ref.world_source, len(gap)), "ok"
