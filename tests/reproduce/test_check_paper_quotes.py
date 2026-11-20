"""scripts/check_paper_quotes.py on a synthetic main paper and synthetic documents: a cited exhibit number must exist
with the declared caption, a supplement number is a finding, a quotation must be verbatim in the main paper, the title
must match, and no document of the release may contain a wording of the script's finite pattern list. The pattern
check also runs on the shipped documents.

The script is not a semantic check of claims. The tests at the end assert its limits: a paraphrase, a changed unit
and a claim attached to the wrong table all pass. A manual audit of each results sentence against the submitted paper
covers those cases (docs/paper_exhibits.md records the audit made for the release)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location("check_paper_quotes", ROOT / "scripts" / "check_paper_quotes.py")
Q = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(Q)

MAIN = """HandReel: Fast Hand-Object Reconstruction from Video
without Per-Frame Pose Sampling
Fig. 1. HandReel reconstructs a hand-held object as one mesh that it poses in every frame.
1. Introduction
HandReel's seconds follow the third setting of Table 2. The baseline took 130× as long per window as that
share, on the nine of 19 attempted windows that both completed.
123
Algorithm 1 The windowed solve. Levenberg-Marquardt minimises E.
5.2. Reconstruction Speed and Error
Table 1. Held-out object-centre error. HandReel's error is non-
inferior to the baseline's.
Table 2. Reconstruction time per window. On Ego4D.
"""


def _root(tmp_path: Path, doc: str, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "doc.md").write_text(doc)
    (tmp_path / "CITATION.cff").write_text(f'title: "{Q.TITLE}"\n')
    (tmp_path / "README.md").write_text(f"# {Q.TITLE}\n\n## Results\n")
    monkeypatch.setattr(Q, "DOCS", ["doc.md"])
    monkeypatch.setattr(Q, "SECTION_DOCS", ["README.md"])
    return tmp_path


def test_declared_exhibits_and_verbatim_quotes_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    doc = ('As "Results" says, the stage "took 130× as long per window as that share" (Table 2, row 2; Sec. 5.2;\n'
           'Fig. 1; Algorithm 1). The supplement, Additional Experiments, gives the windows.\n'
           '```\nprint("not a quote")\n```\n')
    bad, n = Q.check(_root(tmp_path, doc, monkeypatch), MAIN)
    assert bad == [] and n["in_main"] == 1 and n["own_section"] == 1
    assert n["exhibits"] == 4                       # Table 2, Sec. 5.2, Fig. 1 and Algorithm 1


@pytest.mark.parametrize(("doc", "finding"), [
    ('"took 130x as long per window as that share"', "QUOTE NOT FOUND"),            # x for the paper's sign
    ('"a sentence of the supplement or of an older draft"', "QUOTE NOT FOUND"),
    ("a main table this paper lacks (Table 3)", "NOT IN THE PDF"),
    ("a table above the main paper's last (Table 25)", "NOT IN THE MAIN PAPER"),
    ("a section this release does not declare (Sec. 9.9)", "CITED, IDENTITY NOT DECLARED"),
    ("a supplement number (Supp. Tab. 7)", "SUPPLEMENT NUMBER"),
    ("a supplement number (supplement Tables 25 and 27 to 30)", "SUPPLEMENT NUMBER"),
    ("a supplement number (the supplement's Fig. 7)", "SUPPLEMENT NUMBER"),
    ("a supplement appendix (Appendix E.4)", "SUPPLEMENT NUMBER"),
    ("a supplement section (Sec. F)", "SUPPLEMENT NUMBER"),
])
def test_altered_quotes_unknown_exhibits_and_supplement_numbers_fail(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, doc: str, finding: str) -> None:
    bad, _ = Q.check(_root(tmp_path, doc, monkeypatch), MAIN)
    assert any(b.startswith(finding) for b in bad), bad


def test_the_supplement_is_cited_without_numbers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    doc = "The supplement's run-settings table lists each run (the supplement, Experiment Details)."
    bad, n = Q.check(_root(tmp_path, doc, monkeypatch), MAIN)
    assert bad == [] and n["cites"] == 0


def test_a_number_that_moved_to_another_exhibit_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    moved = MAIN.replace("Table 1. Held-out object-centre error", "Table 1. Stage times for one video")
    bad, _ = Q.check(_root(tmp_path, "the held-out errors (Table 1)", monkeypatch), moved)
    assert any(b.startswith("WRONG EXHIBIT: Table 1") for b in bad), bad


def test_a_body_sentence_that_ends_in_a_table_number_is_not_taken_for_the_caption(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """MAIN's introduction ends a sentence with "Table 2." before the caption of Table 2."""
    bad, _ = Q.check(_root(tmp_path, "the reconstruction times (Table 2)", monkeypatch), MAIN)
    assert bad == []


def test_the_title_must_match(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _root(tmp_path, "no citation here", monkeypatch)
    (root / "CITATION.cff").write_text('title: "HandReel: Hand-Object Reconstruction from Video"\n')
    bad, _ = Q.check(root, MAIN)
    assert bad == ["TITLE MISSING OR DIFFERENT: CITATION.cff"]
    (root / "CITATION.cff").write_text(f'title: "{Q.TITLE.replace("Hand-Object", "Hand–Object")}"\n')   # an en dash
    bad, _ = Q.check(root, MAIN)
    assert bad == ["TITLE MISSING OR DIFFERENT: CITATION.cff"]


def test_ranges_and_parentheticals_are_expanded() -> None:
    got = Q.cited("Tables 1 to 3; Table 2 (row 1) and 5; Secs. 5.1-5.3; Supp. Tabs. 39 and 40; Figs. 1 and 2")
    assert got == [("Table", n) for n in ("1", "2", "3", "2", "5")] + [("Figure", n) for n in ("1", "2")] + [
        ("Sec", s) for s in ("5.1", "5.2", "5.3")]


# One sentence per listed wording; each must match a pattern.
FORBIDDEN_SENTENCES = [
    "End to end the pipeline is 1.5-3.1x as fast.",
    "End to end the pipeline is 1.5–3.1× as fast.",
    "End to end the pipeline is 1.5 to 3.1x as fast.",
    "HandReel is about 2x end to end.",
    "HandReel is as accurate as the baseline.",
    "HandReel is faster at equal quality.",
    "HandReel halved the error.",
    "HandReel is faster end to end.",
    "The baseline took 128x as long.",
    "The baseline took 128× as long.",
    "HandReel is two orders of magnitude faster.",
    "HandReel handles video of any length.",
    "HandReel's error was non-inferior at a 15% margin on HOI4D and on DexYCB.",
    "HandReel's error was non-inferior at a 15% margin on HOI4D and DexYCB.",
    "FoundationPose was not timed.",
    "For many videos HandReel gives a 120x speed-up.",
    "For many videos the speedup is large.",
    "HandReel's warm batched runs took a 17 s per-window share.",
    "HandReel took 17 s per window.",
]


@pytest.mark.parametrize("sentence", FORBIDDEN_SENTENCES)
def test_each_listed_wording_fails(tmp_path: Path, sentence: str) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "page.md").write_text(f"# A page\n\nSome text.\n{sentence}\nMore text.\n")
    bad = Q.forbidden_wording(tmp_path)
    assert bad and all(b.startswith("FORBIDDEN WORDING docs/page.md") for b in bad), bad


def test_the_papers_own_wording_passes(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text(
        "DO AS I DO's reconstruction stage took 130x as long per window as HandReel's per-window share of batched\n"
        "runs: 2,100 s against a 17 s per-window share. Each batched run loaded its models once, and the timing\n"
        "includes that load. By a sum of separately measured stage times, the full pipeline would take about 3x as\n"
        "long: 3,069 s against 991 s. On DexYCB the ratio was 0.85, which is non-inferior, but that verdict is not\n"
        "robust if joint failures count against HandReel. The object is welded during the warm-up rows.\n")
    assert Q.forbidden_wording(tmp_path) == []


def test_the_pattern_check_reads_docstrings_and_skips_the_registered_protocol(tmp_path: Path) -> None:
    for rel, text in (("reproduce/tool.py", '"""HandReel is as accurate as the baseline."""\nX = "as accurate as"\n'),
                      ("reproduce/heldout/protocol/prereg.md", "our warm share, as registered\n"),
                      ("third_party/x/README.md", "orders of magnitude\n"),
                      ("src/pkg/code.py", '"""any length"""\n')):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text)
    assert Q.pattern_docs(tmp_path) == ["reproduce/tool.py"]
    bad = Q.forbidden_wording(tmp_path)
    assert len(bad) == 1 and bad[0].startswith("FORBIDDEN WORDING reproduce/tool.py: as accurate as")


def test_the_shipped_documents_match_no_listed_wording() -> None:
    """The release's own documents: fails on any wording of the list above in README.md, docs/, release/,
    reproduce/, CITATION.cff or a script's docstring. A pass shows that no pattern matches, not that the claims are
    right."""
    docs = Q.pattern_docs(ROOT)
    for rel in ("README.md", "CITATION.cff", "docs/paper_exhibits.md", "release/QUICKSTART.md", "reproduce/README.md",
                "reproduce/heldout/README.md", "reproduce/e1_verdict_from_scores.py"):
        assert rel in docs, rel
    assert not any(d.startswith(("reproduce/heldout/protocol/", "third_party/")) for d in docs)
    assert Q.forbidden_wording(ROOT) == []


def test_every_allowance_names_a_shipped_document_and_a_rule() -> None:
    names = {name for name, _ in Q.FORBIDDEN}
    for (rel, name), reason in Q.ALLOWED.items():
        assert (ROOT / rel).is_file() and name in names and reason


def test_the_shipped_documents_cite_no_supplement_number() -> None:
    """Needs no PDF: no document in DOCS cites a numbered supplement table, figure, section or appendix."""
    for rel in Q.DOCS:
        assert Q.supplement_numbers(Q.doc_text(ROOT / rel)) == [], rel


# --- Limits of the script, asserted so that nobody reads a clean exit as a check of claims. Each case below is wrong
# --- against the submitted paper and passes. A manual claim-to-evidence audit of each results sentence covers them.

PARAPHRASES_NOT_CAUGHT = [
    "HandReel matches the baseline's accuracy.",                            # "as accurate as", in other words
    "End to end, HandReel roughly doubles the pipeline's throughput.",      # "about 2x end to end"
    "HandReel cut the object-centre error in two on DexYCB.",               # "halved"; and the wrong corpus
    "HandReel reconstructs videos of arbitrary duration.",                  # "any length"
    "The many-videos application runs over a hundred times quicker with HandReel.",   # a ratio for application 2
    "Nobody measured how long FoundationPose takes.",                       # "FoundationPose was not timed"
]


@pytest.mark.parametrize("sentence", PARAPHRASES_NOT_CAUGHT)
def test_limit_a_paraphrase_of_a_listed_wording_is_not_caught(tmp_path: Path, sentence: str) -> None:
    """NOT caught: the patterns are a finite list of wordings. The manual audit covers paraphrases."""
    (tmp_path / "README.md").write_text(f"# Results\n\n{sentence}\n")
    assert Q.forbidden_wording(tmp_path) == []


CHANGED_UNITS_NOT_CAUGHT = [
    # the paper: 2,100 s against a 17 s per-window share; about 2,000 s against 66 s; 56.8 mm against 101.3 mm
    "DO AS I DO took 2,100 min per window against HandReel's 17 min per-window share of batched runs.",
    "DO AS I DO took about 2,000 ms per window against HandReel's 66 ms.",
    "The median error was 56.8 cm for HandReel and 101.3 cm for DO AS I DO.",
    "The ratio was 130x, with a 95% confidence interval of 97 to 170, on 20 windows.",     # the paper: 9 of 19
]


@pytest.mark.parametrize("sentence", CHANGED_UNITS_NOT_CAUGHT)
def test_limit_a_changed_unit_or_count_is_not_caught(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                     sentence: str) -> None:
    """NOT caught: no check compares a number or its unit with the paper. The manual audit covers them."""
    bad, _ = Q.check(_root(tmp_path, sentence, monkeypatch), MAIN)        # includes the wording patterns
    assert bad == []


WRONG_ASSOCIATIONS_NOT_CAUGHT = [
    "Table 1 reports the reconstruction speed of both pipelines.",          # its caption: held-out object-centre error
    "HandReel's object-centre error was half of the baseline's (Table 2).",     # its caption: reconstruction time
    "Algorithm 1 gives the quality check (Sec. 5.2).",                      # the windowed solve; speed and error
]


@pytest.mark.parametrize("sentence", WRONG_ASSOCIATIONS_NOT_CAUGHT)
def test_limit_a_claim_attached_to_the_wrong_exhibit_is_not_caught(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                   sentence: str) -> None:
    """NOT caught: the citation check binds a number to its caption, not the sentence to the exhibit. Each sentence
    cites an exhibit that exists under its declared caption and says something that the exhibit does not show. The
    manual audit covers them."""
    bad, n = Q.check(_root(tmp_path, sentence, monkeypatch), MAIN)
    assert bad == [] and n["cites"] >= 1


def test_the_success_text_states_the_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                           capsys: pytest.CaptureFixture[str]) -> None:
    """A clean exit says what was checked and that it is not a semantic check; --help says so too."""
    root = _root(tmp_path, "the held-out errors (Table 1)", monkeypatch)
    paper = tmp_path / "main.txt"
    paper.write_text(MAIN)
    assert Q.main(["--main", str(paper), "--root", str(root)]) == 0
    out = capsys.readouterr().out
    assert "no configured-pattern violation found" in out and "not a semantic check of claims" in out
    assert "exists in the PDF with the declared caption" in out and "no numbered supplement citation" in out
    assert "the title matches" in out and "does not make" not in out
    assert Q.main(["--patterns-only", "--root", str(root)]) == 0
    out = capsys.readouterr().out
    assert "no configured-pattern violation found" in out and "not a semantic check of claims" in out
    with pytest.raises(SystemExit):
        Q.main(["--help"])
    assert "not a semantic check of claims" in " ".join(capsys.readouterr().out.split())
    assert "not a semantic check of claims" in Q.__doc__
