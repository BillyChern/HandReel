"""The prompt stage's choice and refusal logic (S2 precondition P3), with no detector and no GPU.

This is where the yield hazard lives. A prompt stage that never refuses turns every unusable window into a false
accept, which inflates exactly the number S2 exists to measure
(screening dominates at internet scale). So the refusals are
tested as hard as the choices, and every choice is required to be deterministic.
"""

from __future__ import annotations

import pytest

from fastdo.prompt.choose import (
    HAND_CONTACT_OBJECT,
    PromptThresholds,
    candidate_rows,
    choose_prompt,
    mask_is_plausible,
)


def hand(*, score: float = 0.95, contact: int = HAND_CONTACT_OBJECT, obj_score: float = 0.95,
         held: float = 0.8, interaction: float = 0.9, box: tuple[float, float, float, float] = (10, 10, 60, 60),
         side: str = "right") -> dict:
    return {"hand_score": score, "contact_idx": contact, "side": side, "interaction_score": interaction,
            "obj_score": obj_score, "p_held_like": held, "obj_box": list(box)}


T = PromptThresholds(min_score=0.1, hand_score=0.9, obj_score=0.9)


def test_candidate_rows_include_the_ends_and_respect_the_stride() -> None:
    assert candidate_rows(20, stride=5) == [0, 5, 10, 15, 19]
    assert candidate_rows(20, stride=12) == [0, 12, 19]
    assert candidate_rows(1, stride=5) == [0]
    with pytest.raises(ValueError, match="stride"):
        candidate_rows(10, stride=0)
    with pytest.raises(ValueError, match="rows"):
        candidate_rows(0, stride=5)


def test_it_picks_the_highest_interaction_times_held_like() -> None:
    per_row = {0: [hand(interaction=0.5, held=0.5)],          # 0.25
               5: [hand(interaction=0.9, held=0.9)],          # 0.81  <- best
               10: [hand(interaction=0.95, held=0.4)]}        # 0.38
    got = choose_prompt(per_row, T)
    assert got is not None and got.row == 5
    assert got.score == pytest.approx(0.81)
    assert got.side == "right"


def test_a_hand_touching_nothing_or_itself_is_ignored() -> None:
    per_row = {0: [hand(contact=0)],        # no_contact
               5: [hand(contact=1)],        # self / other-person, depending on the head's mapping
               10: [hand(contact=2)]}
    assert choose_prompt(per_row, T) is None


def test_low_scores_are_ignored_and_an_empty_result_refuses() -> None:
    assert choose_prompt({0: [hand(score=0.5)]}, T) is None            # weak hand
    assert choose_prompt({0: [hand(obj_score=0.2)]}, T) is None         # weak object
    assert choose_prompt({0: [hand(interaction=0.01, held=0.01)]}, T) is None  # below min_score
    assert choose_prompt({}, T) is None
    assert choose_prompt({0: []}, T) is None


def test_a_hand_with_no_linked_object_refuses() -> None:
    h = hand()
    del h["obj_box"]
    assert choose_prompt({0: [h]}, T) is None


def test_ties_break_deterministically_and_never_by_dict_order() -> None:
    a = hand(box=(0, 0, 40, 40))            # area 1600
    b = hand(box=(5, 5, 65, 65))            # area 3600, same score
    assert choose_prompt({3: [a], 7: [b]}, T).row == 3                 # lower row first
    first = choose_prompt({3: [a, b]}, T)
    assert first is not None and first.box == [5.0, 5.0, 65.0, 65.0]   # then the larger box
    # the same inputs in a different insertion order give the same answer
    assert choose_prompt({7: [b], 3: [a]}, T).row == 3
    assert choose_prompt({3: [b, a]}, T).box == [5.0, 5.0, 65.0, 65.0]


def test_mask_plausibility_refuses_the_three_ways_a_box_prompt_fails() -> None:
    import numpy as np

    frame = (48, 64)
    ok = np.zeros(frame, bool); ok[10:30, 10:40] = True                # 600 of 3072 px = 19.5 %
    assert mask_is_plausible(ok, T)

    assert not mask_is_plausible(np.zeros(frame, bool), T)             # empty
    assert not mask_is_plausible(np.ones(frame, bool), T)              # the whole frame
    tiny = np.zeros(frame, bool); tiny[0:1, 0:2] = True                # 0.065 %
    assert not mask_is_plausible(tiny, T)

    # a blowout that touches three borders but is under the area cap
    blow = np.zeros(frame, bool); blow[:, 0:20] = True; blow[0, :] = True
    assert not mask_is_plausible(blow, T)


def test_thresholds_reject_nonsense() -> None:
    with pytest.raises(ValueError, match="min_score"):
        PromptThresholds(min_score=-0.1)
    with pytest.raises(ValueError, match="area_frac"):
        PromptThresholds(area_frac=(0.4, 0.002))
    with pytest.raises(ValueError, match="hand_score"):
        PromptThresholds(hand_score=1.5)


# --- the gate trace a refusal records ----------------------------------------------------------------


def test_first_failed_gate_is_choose_prompts_own_predicate() -> None:
    """On randomised records, choose_prompt picks exactly among the records first_failed_gate calls eligible."""
    import random

    from fastdo.prompt.choose import GATES, first_failed_gate, gate_trace

    rng = random.Random(251)
    for trial in range(400):
        per_row = {}
        for row in rng.sample(range(40), rng.randint(0, 5)):
            per_row[row] = [hand(score=rng.choice([0.5, 0.89, 0.9, 0.99]), contact=rng.choice([0, 1, 3, 3, 4]),
                                 obj_score=rng.choice([0.2, 0.9, 0.95]), held=rng.choice([0.0, 0.05, 0.5, 1.0]),
                                 interaction=rng.choice([0.0, 0.1, 0.9]),
                                 box=(rng.randint(0, 20), 0, rng.randint(21, 60), 50)) for _ in range(rng.randint(0, 3))]
            if rng.random() < 0.2 and per_row[row]:
                per_row[row][0] = {k: v for k, v in per_row[row][0].items() if k != "obj_box"}
        thresholds = PromptThresholds(min_score=rng.choice([0.0, 0.1, 0.5]), hand_score=0.9, obj_score=0.9)
        trace = gate_trace(per_row, thresholds)
        assert len(trace) == sum(len(v) for v in per_row.values())
        assert all(r["first_failed_gate"] in (*GATES, None) for r in trace)
        eligible = [r for r in trace if r["first_failed_gate"] is None]
        choice = choose_prompt(per_row, thresholds)
        assert (choice is None) == (not eligible), trial
        if choice is not None:
            assert any(r["row"] == choice.row and [float(v) for v in r["obj_box"]] == choice.box for r in eligible)
        in_order = [rec for row in sorted(per_row) for rec in per_row[row]]
        assert [first_failed_gate(rec, thresholds) for rec in in_order] == [t["first_failed_gate"] for t in trace]


def test_gate_trace_names_the_first_gate_in_choose_prompts_order() -> None:
    from fastdo.prompt.choose import gate_trace

    per_row = {5: [hand(score=0.5, contact=0)], 0: [hand(contact=1), hand(obj_score=0.1), hand(held=0.0),
                                                    {k: v for k, v in hand().items() if k != "obj_box"}, hand()]}
    trace = gate_trace(per_row, T)
    assert [r["row"] for r in trace] == [0, 0, 0, 0, 0, 5]
    assert [r["first_failed_gate"] for r in trace] == ["contact_idx", "obj_score", "score", "obj_box", None,
                                                       "hand_score"]
