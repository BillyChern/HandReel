# Copyright 2026 The alongray authors
# SPDX-License-Identifier: Apache-2.0
"""A cluster drawn k times in a bootstrap draw is k clusters in the sequence-weighted mean."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest

from alongray.coverage_io import Method, Protocol, UnitSpec
from alongray.coverage_report import _means, aggregate_scores
from alongray.pool import cluster_bootstrap

SCORES = [0.0, 0.1, 0.3, 1.0]            # four objects, one sequence and one equal-frame unit each


def test_a_sequence_drawn_three_times_counts_three_times() -> None:
    w, seqs = [5, 5, 5, 5], ["s0", "s1", "s2", "s3"]
    got = _means(SCORES, w, seqs, [0, 0, 0, 3])
    assert got == {"frame_weighted": pytest.approx(0.25), "sequence_weighted": pytest.approx(0.25)}   # not 0.5
    assert _means(SCORES, w, seqs, [0, 1, 2, 3])["sequence_weighted"] == pytest.approx(0.35)


def _by_label(vals: list[float], w: list[int], seqs: list[str], idx: list[int]) -> float:
    """The earlier statistic: one mean per sequence LABEL in the draw, however often the sequence was drawn."""
    rows: dict[str, list[int]] = {}
    for i in idx:
        rows.setdefault(seqs[i], []).append(i)
    return float(np.mean([sum(vals[i] * w[i] for i in g) / sum(w[i] for i in g) for g in rows.values()]))


def test_every_draw_of_four_clusters_gives_the_enumerated_interval() -> None:
    """All 4^4 equally likely ordered draws. With multiplicity the linear 2.5% and 97.5% percentiles of the
    sequence-weighted mean are [0.05, 0.775]; collapsing by label gives [0.05, 0.65] and differs on 192 draws."""
    w, seqs = [5, 5, 5, 5], ["s0", "s1", "s2", "s3"]
    kept, collapsed = [], []
    for pick in itertools.product(range(4), repeat=4):
        kept.append(_means(SCORES, w, seqs, list(pick))["sequence_weighted"])
        collapsed.append(_by_label(SCORES, w, seqs, list(pick)))
    assert np.percentile(kept, [2.5, 97.5]) == pytest.approx([0.05, 0.775])
    assert np.percentile(collapsed, [2.5, 97.5]) == pytest.approx([0.05, 0.65])
    assert sum(abs(a - b) > 1e-12 for a, b in zip(kept, collapsed, strict=True)) == 192


def test_an_object_drawn_twice_brings_two_copies_of_each_of_its_sequences() -> None:
    vals, w, seqs = [1.0, 3.0, 10.0], [2, 2, 2], ["s0", "s1", "s2"]              # object A = s0, s1; object B = s2
    got = _means(vals, w, seqs, [0, 1, 0, 1])
    assert got["sequence_weighted"] == pytest.approx(2.0) and got["frame_weighted"] == pytest.approx(2.0)
    assert _means(vals, w, seqs, [2, 0, 1])["sequence_weighted"] == pytest.approx((10.0 + 1.0 + 3.0) / 3)


def test_a_sequence_shared_by_two_objects_keeps_the_point_under_the_identity_draw() -> None:
    """Units A:S = 0, A:T = 9, B:S = 0, one frame each, clusters = objects. Sequence S lies in both objects. The
    point is the mean over sequences, (0 + 9) / 2 = 4.5, and the draw that takes each object once must give it
    back; grouping by (drawn object, sequence) would give (0 + 9 + 0) / 3 = 3."""
    vals, w, seqs, obj = [0.0, 9.0, 0.0], [1, 1, 1], ["S", "T", "S"], ["A", "A", "B"]
    assert _means(vals, w, seqs, [0, 1, 2])["sequence_weighted"] == pytest.approx(4.5)
    seen: list[list[int]] = []
    cluster_bootstrap(lambda idx: seen.append(idx) or 0.0, obj, 200, 0)
    for idx in seen:
        if sorted(idx) == [0, 1, 2]:                                  # the identity draw, in either order
            assert _means(vals, w, seqs, idx)["sequence_weighted"] == pytest.approx(4.5)
    assert any(sorted(idx) == [0, 1, 2] for idx in seen)
    twice_a = _means(vals, w, seqs, [0, 1, 0, 1])                    # A, A: S has 2 of its 2 frames, T has 2 of 1
    assert twice_a["sequence_weighted"] == pytest.approx((1.0 * 0.0 + 2.0 * 9.0) / 3.0)
    assert _means(vals, w, seqs, [2, 2])["sequence_weighted"] == pytest.approx(0.0)   # B, B: only S is drawn
    assert _means([1.0, 5.0], [3, 1], ["S", "S"], [0, 1, 1])["sequence_weighted"] == pytest.approx(13 / 5)


def _method(name: str) -> Method:
    scores = {f"u{i}": {"err": v, "n_frames": 5} for i, v in enumerate(SCORES)}
    return Method(name=name, folder=Path("."), outputs="scores", provenance={}, modality={}, inputs={},
                  conventions={"aggregation": "sequence"}, units={}, scores=scores, counts={}, reported={},
                  timing_fields=())


def test_the_reported_interval_keeps_the_copies() -> None:
    units = tuple(UnitSpec(f"u{i}", f"s{i}", "cam", f"o{i}", tuple(range(5)), frozenset(range(5))) for i in range(4))
    proto = Protocol(name="p", provenance={}, modality={}, conventions={"aggregation": "sequence"}, cluster="object",
                     frame_rate=None, overlap_policy="", symmetry={}, units=units, published_counts=None)
    common = {u.unit_id: list(range(5)) for u in units}
    out, _ = aggregate_scores(proto, _method("a"), _method("b"), common, n_boot=20000, seed=3)
    row = out["err"]["a"]
    lo, hi = row["ci95_cluster_bootstrap"]["sequence_weighted"]
    assert row["sequence_weighted"] == pytest.approx(0.35) and lo == pytest.approx(0.05, abs=0.03)
    assert hi == pytest.approx(0.775, abs=0.03)                       # 0.65 if a repeated sequence counted once
    assert row["ci95_cluster_bootstrap"]["frame_weighted"] == pytest.approx([lo, hi])   # equal frames: same statistic
