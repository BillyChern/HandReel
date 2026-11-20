"""The corpus-economics model (`fastdo.bench.yield_model`).

These are regression tests on three mistakes that were actually made on 2026-09-21, not on arithmetic. The
arithmetic is trivial; what is being pinned is that the model cannot be *applied* the way it was misapplied
— a contended `c_post` silently deflating a result, a per-video cost charged per window, a ratio quoted
without the denominator that determines it.
"""

from __future__ import annotations

import math

import pytest

from fastdo.bench.yield_model import (MEASURED, Seconds, seconds_per_window, speedup_of_improving,
                                      windows_per_gpu_hour, yield_per_gpu_hour)

# The original note's inputs, so the regression below is against the real published figures.
TABLE_BEFORE, TABLE_AFTER = 2.29, 0.007        # long-video frame table, per window
SCREEN = 1.07                                  # decode + encoder + crop, per window (upper end)
A_GATE = 0.085                                 # a at gate FPR 0.00


def test_a_contended_c_post_is_refused_by_name() -> None:
    """An inflated c_post UNDER-states every c_pre improvement, so it cannot be used silently."""
    with pytest.raises(ValueError, match="card='contended'"):
        yield_per_gpu_hour(0.085, 1.0, 1.0, c_pre=3.36, a=A_GATE, c_post=MEASURED["c_post_pilot"])
    # ... and is allowed when the caller says so, which is what makes the refusal informative.
    assert yield_per_gpu_hour(0.085, 1.0, 1.0, c_pre=3.36, a=A_GATE,
                              c_post=MEASURED["c_post_pilot"], allow_contended=True) > 0


def test_the_exclusive_measurement_needs_no_permission() -> None:
    assert MEASURED["shape_ego4d"].trustworthy
    assert not MEASURED["shape_ego4d_pilot"].trustworthy
    assert not MEASURED["c_post_partly_corrected"].trustworthy      # shape is, the other stages are not


def test_the_published_1_15x_is_reproduced_and_then_corrected() -> None:
    """The contended-c_post finding itself, as a regression: only c_post moves between these two lines.

    1.15x was published this morning from a contended c_post. Reproducing it is what established that the
    original reasoning was sound and that the input was the problem — so both numbers are pinned here.
    """
    contended = speedup_of_improving(TABLE_BEFORE, TABLE_AFTER,
                                     SCREEN + A_GATE * float(MEASURED["c_post_pilot"]), card="contended")
    corrected = speedup_of_improving(TABLE_BEFORE, TABLE_AFTER,
                                     SCREEN + A_GATE * float(MEASURED["c_post_partly_corrected"]))
    assert contended.ratio == pytest.approx(1.15, abs=0.01)         # the published figure
    assert corrected.ratio == pytest.approx(1.26, abs=0.01)         # what it is with a cleaner denominator
    assert corrected.ratio > contended.ratio


def test_the_ratio_is_monotonically_decreasing_in_the_remainder() -> None:
    """The remainder algebra, asserted as the property rather than at one point.

    S(R) = (x + R)/(x' + R) with x > x' is strictly decreasing in R. This is why an over-measured remainder
    under-values finished work, and the direction of that bias is the whole point.
    """
    ratios = [speedup_of_improving(10.0, 1.0, R).ratio for R in (0.0, 1.0, 5.0, 25.0, 100.0, 1000.0)]
    assert ratios == sorted(ratios, reverse=True)
    assert all(a > b for a, b in zip(ratios, ratios[1:]))
    assert ratios[0] == pytest.approx(10.0)                         # R = 0: the bare ratio
    assert ratios[-1] < 1.01                                        # R -> inf: the improvement vanishes


def test_a_speedup_carries_the_remainder_it_was_taken_against() -> None:
    """A ratio without its denominator is the shape of the contended-c_post error, so it cannot be returned alone."""
    s = speedup_of_improving(TABLE_BEFORE, TABLE_AFTER, MEASURED["c_post_pilot"])
    assert s.remainder == pytest.approx(161.3)
    assert s.card == "contended"
    assert any("LOWER bound" in c for c in s.caveats)
    assert "card=contended" in str(s) and "161.30" in str(s)


def test_the_cost_model_has_a_per_video_term_and_it_matters() -> None:
    """Decode is 258.07 s for a video's first window and 3.22 s for every later one."""
    c_w, c_v = float(MEASURED["c_window_decode"]), float(MEASURED["c_video_decode"])
    assert seconds_per_window(c_w, c_v, 1) == pytest.approx(c_w + c_v)
    assert seconds_per_window(c_w, c_v, 1) / seconds_per_window(c_w, c_v, 1000) > 50
    # W is a throughput parameter: more windows per video is strictly cheaper per window.
    per_window = [seconds_per_window(c_w, c_v, W) for W in (1, 5.2, 20, 50, 1000)]
    assert all(a > b for a, b in zip(per_window, per_window[1:]))
    assert math.isclose(per_window[-1], c_w, rel_tol=0.1)           # W -> inf recovers the per-window cost


def test_windows_per_video_must_be_positive_rather_than_defaulting() -> None:
    with pytest.raises(ValueError, match="windows_per_video"):
        seconds_per_window(3.22, 258.07, 0)


def test_throughput_at_the_pilots_own_w_matches_what_m140_recorded() -> None:
    """The pilot ran 47 windows over 9 videos: W = 5.2, and that is why its decode looked enormous."""
    assert windows_per_gpu_hour(111.0, float(MEASURED["c_video_decode"]), 47 / 9) == pytest.approx(
        3600 / 160.6, rel=0.02)


def test_every_measured_constant_names_the_ledger_entry_behind_it() -> None:
    """A constant without a traceable source is the failure this registry exists to prevent: every source says how
    the figure was obtained (`measured`, `pilot`, `derived` or `note`) and then what it is."""
    for name, s in MEASURED.items():
        kind, _, what = s.source.partition(": ")
        assert kind in ("measured", "pilot", "derived", "note") and len(what.split()) >= 3, (
            f"{name}: {s.source!r} does not say how it was obtained and what it is")
        assert s.card in ("exclusive", "contended", "unknown"), name


def test_the_packet_route_constants_are_internally_consistent() -> None:
    """c_pre on the packet route IS decode + prompt -- not a separately-typed number that can drift."""
    assert float(MEASURED["c_pre_packet_route"]) == pytest.approx(
        float(MEASURED["decode_packet_route"]) + float(MEASURED["prompt_21_rows"]), abs=0.01)


def test_the_packet_route_is_quoted_against_the_floor_it_cannot_beat() -> None:
    """The warm window decode is the floor; the packet route pays ~1 s of addressing above it."""
    floor, packet = float(MEASURED["decode_warm_window"]), float(MEASURED["decode_packet_route"])
    assert floor < packet < 1.5 * floor
    cold = float(MEASURED["decode_cold_frame_table"])
    assert cold / packet > 50                                     # the 60x cold-path result


def test_the_superseded_contended_figures_are_kept_and_marked() -> None:
    """Kept, because the published 1.15x arithmetic needs them -- and marked, so they cannot be reused."""
    for name in ("shape_ego4d_pilot", "prompt_pilot", "c_post_pilot"):
        assert MEASURED[name].card == "contended"
        assert not MEASURED[name].trustworthy
    assert float(MEASURED["prompt_pilot"]) / float(MEASURED["prompt_21_rows"]) > 10    # the 10.6x


def test_c_post_is_still_flagged_unknown_because_most_of_it_is() -> None:
    """The largest open number in the model must not be able to masquerade as measured."""
    assert MEASURED["c_post_partly_corrected"].card == "unknown"
    with pytest.raises(ValueError, match="not 'exclusive'"):
        yield_per_gpu_hour(0.09, 1.0, 1.0, c_pre=float(MEASURED["c_pre_packet_route"]), a=0.085,
                           c_post=MEASURED["c_post_partly_corrected"])


def test_the_quiet_card_measurements_are_marked_exclusive() -> None:
    """The registry's whole purpose: provenance is enforced, not merely recorded.

    Added after a mutation check: flipping `decode_packet_route` from "exclusive" to "unknown" failed
    nothing, so the card field was decorative for every constant that carries a real measurement. These
    five were each measured with the card or machine checked empty, and a downgrade of
    any of them must fail here rather than quietly re-enable `allow_contended` prompts downstream.
    """
    for name in ("shape_ego4d", "prompt_21_rows", "decode_cold_frame_table", "decode_warm_window",
                 "decode_packet_route", "c_pre_packet_route"):
        assert MEASURED[name].card == "exclusive", f"{name} lost its measured provenance"
        assert MEASURED[name].trustworthy
    # ... and a trustworthy c_pre needs no permission, unlike c_post.
    assert yield_per_gpu_hour(0.09, 1.0, 1.0, c_pre=float(MEASURED["c_pre_packet_route"]), a=0.085,
                              c_post=MEASURED["decode_warm_window"]) > 0
