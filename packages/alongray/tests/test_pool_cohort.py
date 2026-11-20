# Copyright 2026 The alongray authors
# SPDX-License-Identifier: Apache-2.0
"""Cohort denominators and the failure policy of alongray.pool.

A pooled number over the units a method delivered says nothing about the units it failed on. Every group therefore
reports how many units were assigned, scored, failed, excluded (with their reasons) and missing (assigned but with no
row), and the failure policy decides whether a group with failed or missing units is pooled over its scored units
("report", the default) or refused ("refuse").
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from alongray import e_par_split, pool_rows, pool_splits, split_of_row
from alongray.pool import main


def _split(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    g = np.column_stack([rng.normal(0, 0.05, 12), rng.normal(0, 0.05, 12), rng.uniform(0.45, 0.75, 12)])
    return dataclasses.asdict(e_par_split(rng.uniform(0.9, 1.1) * g + rng.normal(scale=0.01, size=g.shape), g))


def _rows() -> list[dict]:
    rows = [{"unit_id": f"u{i}", "video": f"v{i // 2}", "split": _split(i)} for i in range(4)]
    rows.append({"unit_id": "u4", "video": "v2", "status": "failed"})
    rows.append({"unit_id": "u5", "video": "v2", "status": "excluded", "reason": "ground truth missing"})
    return rows


def test_every_group_reports_its_denominators() -> None:
    res = pool_rows(_rows(), cluster_key="video", n_boot=20)["all"]
    assert (res["n_assigned"], res["n_scored"], res["n_failed"], res["n_excluded"], res["n_missing"]) == (6, 4, 1, 1, 0)
    assert res["excluded_reasons"] == {"ground truth missing": 1}
    assert res["failure_policy"] == "report"
    assert "4 of 6 assigned" in res["cohort"] and "1 failed" in res["cohort"] and "not in any statistic" in res["cohort"]
    scored = pool_splits([split_of_row(r) for r in _rows()[:4]], 1e3)                # the scored units alone
    assert res["median_e_par"] == scored["median_e_par"] and res["n_units"] == 4


def test_the_refuse_policy_refuses_a_group_with_failures() -> None:
    with pytest.raises(ValueError, match="failed"):
        pool_rows(_rows(), failure_policy="refuse")
    ok = [r for r in _rows() if r.get("status") != "failed"]          # an excluded unit with a reason is allowed
    assert pool_rows(ok, failure_policy="refuse")["all"]["n_scored"] == 4
    with pytest.raises(ValueError, match="failure_policy"):
        pool_rows(_rows(), failure_policy="ignore")


def test_an_assigned_manifest_counts_units_with_no_row() -> None:
    assigned = [f"u{i}" for i in range(6)] + ["u9"]
    res = pool_rows(_rows(), assigned=assigned)["all"]
    assert res["n_assigned"] == 7 and res["n_missing"] == 1 and "1 missing" in res["cohort"]
    with pytest.raises(ValueError, match="missing"):
        pool_rows(_rows(), assigned=assigned, failure_policy="refuse")
    with pytest.raises(ValueError, match="not in the assigned"):
        pool_rows(_rows(), assigned=["u0", "u1"])
    by_group = pool_rows([{**r, "corpus": "A"} for r in _rows()], group_key="corpus",
                         assigned={**{f"u{i}": "A" for i in range(6)}, "u7": "B"})
    assert by_group["B"]["n_assigned"] == 1 and by_group["B"]["n_missing"] == 1 and by_group["B"]["n_scored"] == 0
    assert "median_e_par" not in by_group["B"]


@pytest.mark.parametrize("bad, match", [
    ({"unit_id": "u5", "video": "v2", "status": "excluded"}, "reason"),
    ({"unit_id": "u4", "video": "v2", "status": "failed", "split": _split(9)}, "failed"),
    ({"unit_id": "u4", "video": "v2", "status": "lost"}, "status"),
    ({"unit_id": "u0", "video": "v0", "split": _split(0)}, "twice"),
])
def test_malformed_cohort_rows_are_refused(bad: dict, match: str) -> None:
    rows = [r for r in _rows() if r["unit_id"] != bad["unit_id"] or bad["unit_id"] == "u0"] + [bad]
    with pytest.raises(ValueError, match=match):
        pool_rows(rows)


def test_command_line_takes_the_manifest_and_the_policy(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    f = tmp_path / "rows.jsonl"
    f.write_text("".join(json.dumps(r) + "\n" for r in _rows()))
    m = tmp_path / "units.txt"
    m.write_text("# synthetic\n" + "".join(f"u{i}\n" for i in range(7)))
    assert main([str(f), "--assigned", str(m)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["all"]["n_missing"] == 1 and out["all"]["n_assigned"] == 7
    assert main([str(f), "--failure-policy", "refuse"]) == 2
    assert "failed" in capsys.readouterr().err
