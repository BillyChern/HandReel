"""Pass@S verification: ONE selection score and top-4 certification.

Per clip, the GPU executor rolls out S_c prior samples (open loop) and S_pi seeded tracker rollouts (+1
deterministic). Each rollout is scored by the registered scorer. The ONE selection score is the normalised SR_MT
margin
    m = max(obj_pos / 0.03 m, obj_rot / 30 deg, tip / 0.06 m, joint / 0.08 m)
(lower is better; the thresholds are the registered scorer's), computed only among GPU passes of
SR_MT AND M_i AND PLAUS (`success`). Nothing else in a score (DO AS I DO's reward, SR_SPIDER, the kind of candidate)
influences the order; ties keep the candidates' seed order. The top `TOP_M` = 4 are certified IN THAT ORDER until one
certifies (prior samples by open-loop CPU replay, tracker rollouts closed loop: the caller's `certify`); a CPU false
negative loses an acceptance, never admits a trajectory.

pass@S (`pass_at_s`): whether any of the first S candidates of a pool (in seed order; seeds are prefix-consistent)
passes on the GPU, for S in {1, 4, 16, 64, 128}.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["S_GRID", "THRESHOLDS", "TOP_M", "Candidate", "Selection", "certify_top", "gpu_pass", "margin",
           "pass_at_s", "rank"]

THRESHOLDS: dict[str, float] = {"obj_pos_m": 0.03, "obj_rot_rad": math.radians(30.0), "tip_m": 0.06, "joint_m": 0.08}
TOP_M = 4
S_GRID = (1, 4, 16, 64, 128)


@dataclass(frozen=True)
class Candidate:
    cid: str
    kind: str                     # "prior" (open loop) or "tracker" (closed loop)
    seed_index: int
    score: dict[str, Any] = field(default_factory=dict, compare=False)


def gpu_pass(score: dict[str, Any]) -> bool:
    return bool(score.get("success"))


def margin(score: dict[str, Any]) -> float:
    """The normalised SR_MT margin (module docstring)."""
    return max(float(score[k]) / t for k, t in THRESHOLDS.items())


def rank(cands: Sequence[Candidate]) -> list[Candidate]:
    """GPU passes ordered by the margin, ties in seed order."""
    order = {c.cid: i for i, c in enumerate(cands)}
    ok = [c for c in cands if gpu_pass(c.score)]
    return sorted(ok, key=lambda c: (margin(c.score), order[c.cid]))


@dataclass
class Selection:
    accepted: Candidate | None
    tried: list[tuple[str, bool]]          # (cid, certified) in certification order
    ranked: list[str]


def certify_top(cands: Sequence[Candidate], certify: Callable[[Candidate], bool], top: int = TOP_M) -> Selection:
    """Certify the ranked candidates in order, at most `top` of them, until one certifies."""
    ranked = rank(cands)
    tried: list[tuple[str, bool]] = []
    for c in ranked[:top]:
        ok = bool(certify(c))
        tried.append((c.cid, ok))
        if ok:
            return Selection(c, tried, [x.cid for x in ranked])
    return Selection(None, tried, [x.cid for x in ranked])


def pass_at_s(cands: Sequence[Candidate], grid: Sequence[int] = S_GRID) -> dict[int, bool]:
    """Any GPU pass among the first S candidates in seed order (candidates must be in seed order)."""
    idx = [c.seed_index for c in cands]
    if idx != sorted(idx):
        raise ValueError("candidates must be in seed order")
    return {int(S): any(gpu_pass(c.score) for c in cands[:S]) for S in grid}
