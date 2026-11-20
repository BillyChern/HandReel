"""The E1 paper cells come only from the verdict record: every cell is pending without one; with one, each cell is
the registered field (log-scale CI exponentiated). The per-corpus block uses the REAL `e1_stats.corpus_test` keys."""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "run" / "study"))
_SPEC = importlib.util.spec_from_file_location("e1_table", ROOT / "run" / "paper" / "e1_table.py")
T = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(T)

import e1_stats as S  # noqa: E402


def _corpus(d: np.ndarray, verdict: str, lower: bool = False) -> dict:
    itt = {k: v for k, v in S.corpus_test(d, [str(i) for i in range(d.size)], None, b=200, seed=1).items()
           if k != "meds"}
    return {"n_assigned": 72, "n_run": 58, "n_analysed": 56, "delivered_ours": 55, "delivered_theirs": 52, "itt": itt,
            "verdict": verdict, "lower_error": lower}


def _record() -> dict:
    d = np.log(np.linspace(0.8, 1.2, 40))
    corp = {"H": _corpus(d, "NON-INFERIOR", lower=True), "D": _corpus(d + 0.3, "INCONCLUSIVE")}
    cal = {"H": dict(_corpus(d, "NON-INFERIOR"), delivered_ours=50), "D": _corpus(d + 0.3, "INCONCLUSIVE")}
    return {"primary_O_EST": {"primary": {"corpora": corp}}, "secondary_S2_O_CAL": {"primary": {"corpora": cal}},
            "headline": {"H": {"headline_non_inferior": True}, "D": {"headline_non_inferior": False}}}


def test_without_a_record_every_cell_is_pending() -> None:
    tex = T.render(None, "none")
    assert tex.count(r"{\TBDr}") == (len(T.FIELDS) - len(T.SUFFIX_FIELDS)) * 2 and "GENERATED" in tex
    assert r"\renewcommand{\EoneHEstLower}{}" in tex                    # a suffix is empty, never a second placeholder


def test_cells_are_the_registered_fields() -> None:
    rec = _record()
    c = T.cells(rec)
    h = rec["primary_O_EST"]["primary"]["corpora"]["H"]["itt"]
    assert c["H"]["EstRatio"] == f"{h['ratio']:.2f}"
    lo, hi = h["ci95"]
    assert c["H"]["EstCI"] == f"[{math.exp(lo):.2f}, {math.exp(hi):.2f}]"       # log scale -> ratio scale
    assert c["H"]["Run"] == "58" and c["H"]["Assigned"] == "72" and c["H"]["EstVerdict"] == "non-inferior"
    assert c["H"]["Headline"] == "yes" and c["D"]["Headline"] == "no" and c["D"]["EstVerdict"] == "inconclusive"
    assert c["H"]["Analysed"] == "56"
    assert c["H"]["EstLower"] == ", lower error" and c["D"]["EstLower"] == "" and c["H"]["CalLower"] == ""
    assert c["H"]["CalDelOurs"] == "50" and c["H"]["CalDelTheirs"] == "52" and c["H"]["CalVerdict"] == "non-inferior"


def test_main_writes_the_pending_file(tmp_path) -> None:
    out = tmp_path / "e1.tex"
    assert T.main(["--out", str(out)]) == 0
    assert r"\renewcommand{\EoneHEstRatio}{\TBDr}" in out.read_text()
