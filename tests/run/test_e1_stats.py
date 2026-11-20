"""CPU tests for the E1 statistics (run/study/e1_stats.py): ITT mapping, bootstrap test, Holm, verdicts, sign test."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "run" / "study"))

import e1_stats as T

NOM = {"H": T.NOMINAL, "D": T.NOMINAL}


def test_itt_bad_values() -> None:
    d = T.paired_d([10, 10, 10, 20], [20, 10, 10, 10], [True, False, True, False], [True, True, False, False])
    assert d.tolist() == [pytest.approx(math.log(0.5)), T.BAD, -T.BAD, 0.0]
    with pytest.raises(ValueError):
        T.paired_d([0.0], [1.0], [True], [True])


def _rows(corpus: str, d: list[float], n_clusters: int, run: list[bool] | None = None) -> list[dict]:
    rows = []
    for i, x in enumerate(d):
        rows.append({"unit_id": f"{corpus}{i}", "corpus": corpus, "cluster": f"c{i % n_clusters}",
                     "stratum": f"s{i % 2}" if corpus == "H" else None,
                     "run": True if run is None else run[i],
                     "ours": {"delivered": True, "e_mm": 50.0 * math.exp(x)}, "theirs": {"delivered": True, "e_mm": 50.0}})
    return rows


def test_clear_parity_is_noninferior_and_clear_loss_is_inferior() -> None:
    rng = np.random.default_rng(0)
    good = list(-0.5 + 0.05 * rng.standard_normal(40))
    bad = list(0.5 + 0.05 * rng.standard_normal(40))
    out = T.summarise(_rows("H", good, 20) + _rows("D", bad, 20), NOM, b=2000)
    h, d = out["corpora"]["H"], out["corpora"]["D"]
    assert h["itt"]["p"] == pytest.approx(1 / 2001) and h["verdict"] == "NON-INFERIOR" and h["lower_error"]
    assert d["verdict"] == "INFERIOR" and d["itt"]["lb95_one_sided"] > T.MARGIN
    assert out["holm"]["pass_level"] == {"D": None, "H": "0.025"}


def test_incomplete_below_80_percent_run() -> None:
    run = [True] * 79 + [False] * 21
    out = T.summarise(_rows("D", [-0.5] * 100, 50, run), NOM, b=500)
    rec = out["corpora"]["D"]
    assert rec["n_run"] == 79 and rec["n_not_run"] == 21 and rec["verdict"] == "INCOMPLETE"
    assert out["holm"]["p"]["D"] == 1.0
    ok = T.summarise(_rows("D", [-0.5] * 100, 50, [True] * 80 + [False] * 20), NOM, b=500)
    assert ok["corpora"]["D"]["verdict"] == "NON-INFERIOR"


@pytest.mark.parametrize(("p_h", "p_d", "want"), [
    (0.01, 0.04, {"H": "0.025", "D": "0.05"}),     # H first step, then D at 0.05
    (0.04, 0.04, {"H": None, "D": None}),          # neither clears the first step
    (0.02, 0.06, {"H": "0.025", "D": None}),
    (0.03, 0.02, {"H": "0.05", "D": "0.025"}),
])
def test_holm_table(p_h: float, p_d: float, want: dict) -> None:
    assert T.holm({"H": p_h, "D": p_d}, NOM) == want


def test_holm_uses_calibrated_thresholds() -> None:
    thr = {"H": {"0.025": 0.0245, "0.05": 0.05}, "D": {"0.025": 0.014, "0.05": 0.0365}}
    assert T.holm({"H": 0.024, "D": 0.04}, thr) == {"H": "0.025", "D": None}
    assert T.holm({"H": 0.026, "D": 0.013}, thr) == {"H": "0.05", "D": "0.025"}


def test_cluster_sign_test_matches_scipy() -> None:
    d = [-1, -1, 0.5, 0.0, -0.2, 0.3, T.MARGIN, 1.0]
    cl = ["a", "a", "b", "c", "d", "e", "f", "g"]
    r = T.cluster_sign_test(d, cl)
    # per-cluster medians: a -1, b .5, c 0, d -.2, e .3, f margin (tie), g 1 -> fav 3 (a, c, d), unf 3 (b, e, g)
    assert (r["favourable"], r["unfavourable"], r["ties"]) == (3, 3, 1)
    assert r["p"] == pytest.approx(binomtest(3, 6, 0.5, alternative="greater").pvalue)


def test_bootstrap_is_deterministic_and_stratified() -> None:
    rng = np.random.default_rng(3)
    d = rng.standard_normal(30)
    cl = [f"c{i // 2}" for i in range(30)]
    st = [f"s{(i // 2) % 3}" for i in range(30)]
    a, b = T.bootstrap_medians(d, cl, st, 300, 1), T.bootstrap_medians(d, cl, st, 300, 1)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, T.bootstrap_medians(d, cl, st, 300, 2))
    with pytest.raises(ValueError, match="spans strata"):
        T.bootstrap_medians(d, cl, [f"s{i % 3}" for i in range(30)], 10, 1)


def test_complete_case_drops_undelivered_and_itt_keeps_them() -> None:
    rows = _rows("D", [-0.5] * 10, 10)
    rows[0]["ours"] = {"delivered": False, "e_mm": None}
    rows[1]["theirs"] = {"delivered": False, "e_mm": None}
    out = T.summarise(rows, NOM, b=300)["corpora"]["D"]
    assert out["itt"]["n"] == 10 and out["complete_case"]["n"] == 8
    assert out["delivered_ours"] == 9 and out["delivered_theirs"] == 9 and out["n_both"] == 8
    assert out["counts_complete_case"]["wins_ours"]["count"] == 8
