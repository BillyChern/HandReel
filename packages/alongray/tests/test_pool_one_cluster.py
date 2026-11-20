# Copyright 2026 The alongray authors. SPDX-License-Identifier: Apache-2.0
"""The cluster bootstrap with fewer than two distinct clusters has no between-cluster information: its interval is
NOT_ESTIMABLE, never a zero-width interval. With two or more clusters every output is bit-identical to 0.3.1
(tests/data/pool_bootstrap_golden_0.3.1.json, written by the unchanged 0.3.1 code)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from alongray import pool
from alongray.pool import NOT_ESTIMABLE, cluster_bootstrap, pool_rows

GOLDEN = json.loads((Path(__file__).resolve().parent / "data" / "pool_bootstrap_golden_0.3.1.json").read_text())


ADDED_SINCE_GOLDEN = ("assignment",)       # fields added after the golden outputs were recorded; checked elsewhere


def _hex(o: object) -> object:
    if isinstance(o, dict):
        return {k: _hex(v) for k, v in o.items() if k not in ADDED_SINCE_GOLDEN}
    if isinstance(o, (list, tuple)):
        return [_hex(v) for v in o]
    if isinstance(o, float):
        return float.hex(o)
    return o


def test_zero_clusters_is_not_estimable() -> None:
    calls: list[list[int]] = []
    assert cluster_bootstrap(lambda ix: calls.append(ix) or 0.0, [], 50, 0) == NOT_ESTIMABLE
    assert calls == []                                       # the statistic is never evaluated on no unit


def test_one_cluster_is_not_estimable() -> None:
    """Several units, one cluster: 0.3.1 returned the zero-width interval (stat, stat); now NOT_ESTIMABLE."""
    x = np.array([0.2, 0.5, 0.9])
    assert cluster_bootstrap(lambda ix: float(np.mean(x[ix])), ["v", "v", "v"], 200, 0) == NOT_ESTIMABLE
    assert NOT_ESTIMABLE == "NOT_ESTIMABLE"


def test_one_cluster_call_leaves_a_shared_generator_as_0_3_1_did() -> None:
    """A numpy-behaviour check: a one-cluster call leaves a shared generator where 0.3.1's left it. (On numpy 2.x
    ``integers(0, 1, 1)`` draws no bits, so this holds whether or not the draws are made; the regression guard for the
    later groups is ``one_then_two_group_B`` in the test below.)"""
    g1, g2 = np.random.default_rng(9), np.random.default_rng(9)
    cluster_bootstrap(lambda ix: 0.0, ["v", "v"], 100, g1)
    for _ in range(100):
        g2.integers(0, 1, 1)
    assert g1.integers(0, 10**9) == g2.integers(0, 10**9)


def test_two_clusters_give_a_proper_interval() -> None:
    x = np.array([0.2, 0.5, 0.9, 0.1])
    ci = cluster_bootstrap(lambda ix: float(np.mean(x[ix])), ["a", "a", "b", "b"], 400, 0)["value"]
    assert isinstance(ci, tuple) and ci[0] < ci[1]


def test_pool_rows_one_cluster_group_reports_not_estimable_and_keeps_the_estimate() -> None:
    rows = GOLDEN["rows"]["one_then_two"]
    res = pool_rows(rows, group_key="corpus", cluster_key="video", n_boot=300, seed=5)
    a = res["A"]                                            # one cluster ("only"), five units
    assert a["n_clusters"] == 1 and a["n_units"] == 5
    assert a["pooled_share_ci95"] == NOT_ESTIMABLE
    assert "fewer than 2" in a["pooled_share_ci95_reason"]
    no_boot = pool_rows([r for r in rows if r["corpus"] == "A"], cluster_key="video")["all"]
    assert _hex(a["pooled_share"]) == _hex(no_boot["pooled_share"])     # the point estimate is kept
    # the two-cluster group after it is bit-identical to 0.3.1 (the shared generator advanced as before)
    assert _hex(res["B"]) == GOLDEN["output"]["one_then_two_group_B"]
    assert "pooled_share_ci95_reason" not in res["B"]


@pytest.mark.parametrize("case", ["two_clusters", "three_clusters"])
def test_two_or_more_clusters_bit_identical_to_0_3_1(case: str) -> None:
    res = pool_rows(GOLDEN["rows"][case], cluster_key="video", n_boot=300, seed=5)
    assert _hex(res) == GOLDEN["output"][case]


def test_pool_cli_prints_the_marker(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    p = tmp_path / "rows.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in GOLDEN["rows"]["one_then_two"] if r["corpus"] == "A"))
    assert pool.main([str(p), "--cluster-key", "video", "--boot", "50"]) == 0
    out = json.loads(capsys.readouterr().out)["all"]
    assert out["pooled_share_ci95"] == "NOT_ESTIMABLE" and out["n_clusters"] == 1


def test_coverage_one_cluster_interval_is_not_estimable(tmp_path: Path) -> None:
    """alongray-coverage's known-answer case id_collision has one cluster: its interval is NOT_ESTIMABLE, while a
    two-cluster case keeps its interval."""
    from alongray.coverage import compare_folders
    from alongray.coverage_fixtures import write_known_answers

    write_known_answers(tmp_path)

    def rows(case: str) -> dict:
        d = tmp_path / case
        return compare_folders(d / "protocol.json", d / "a", d / "b", None, n_boot=50)["aggregates"]["err_mm"]

    for m, row in rows("id_collision").items():
        assert row["n_clusters"] == 1 and row["ci95_cluster_bootstrap"] == NOT_ESTIMABLE, m
        assert "fewer than 2" in row["ci95_cluster_bootstrap_reason"], m
        assert row["frame_weighted"] == pytest.approx(20.0)            # the point estimate is kept
    for m, row in rows("aggregation").items():
        assert row["n_clusters"] == 2 and len(row["ci95_cluster_bootstrap"]["sequence_weighted"]) == 2, m
