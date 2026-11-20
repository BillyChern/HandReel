#!/usr/bin/env python3
"""Number, quotation and wording-pattern checks of this release's documents against the submitted main paper's PDF.

    python3 scripts/check_paper_quotes.py --main <main paper .pdf or .txt>

This is not a semantic check of claims. It checks that cited numbers exist under the declared captions and that a
finite, hand-written list of wordings is absent. It does not test whether a sentence says what its cited exhibit
shows: a paraphrase of a listed wording, a changed unit, or a claim attached to the wrong table all pass. A manual
audit of each results sentence against the paper covers those; `docs/paper_exhibits.md` records the audit made for
this release.

The PDF is not part of this release; pass its path. A `.pdf` is converted with `pdftotext` (poppler), or with PyMuPDF
when `pdftotext` is not on PATH; a `.txt` is read as already extracted. Five checks:

1. Cited numbers of the main paper. Every table, figure, section and algorithm number that a document in `DOCS`
   cites must exist in the PDF, and must be listed in `IDENTITY` with the start of its caption or heading, which must
   match the PDF. So a number that moved to another exhibit fails until its identity is checked again. The check
   binds a number to a caption. It does not bind the sentence around the citation to that exhibit.
2. No supplement numbers. The supplement's numbering is not final, so no document may cite a numbered supplement
   table, figure, section, appendix or algorithm ("Supp. Tab. 7", "Appendix E", "supplement Table 25"), and a bare
   "Table N" or "Fig. N" above the main paper's last table or figure is a finding too. Documents name the supplement's
   parts without numbers ("the supplement, Additional Experiments").
3. Quotations. Every passage in double quotes ("..." or the curly pair) in the Markdown documents, outside code
   blocks and outside inline code that itself holds a quote character, must be the name of a section of this
   release's own documents (a heading or a bold paragraph lead), a passage that `NOT_PAPER` lists with its own source
   (a log line, a file's text, another work's title, a column name), or verbatim in the main paper. So a quotation
   of the supplement or of an older draft fails. The Python files, the manifest and the documents in `NUMBERS_ONLY`
   are checked for cited numbers only. Matching ignores whitespace, ASCII hyphens and underscores, backticks and the
   forms of quote marks and ligatures, because PDF text extraction breaks lines and drops end-of-line hyphens.
   Nothing else is relaxed.
4. The title: `CITATION.cff` and `README.md` must give the paper's title as the main PDF prints it.
5. Configured wording patterns (`FORBIDDEN`): a finite list of wordings that the submitted paper does not use for its
   results, each a regular expression, plus the rule that "17 s" stands near "per-window share". No document of the
   release (`pattern_docs`) may contain a match. This check needs no PDF: `forbidden_wording(root)` runs it alone,
   and `--patterns-only` runs it from the command line. A document that passes can still state a wrong claim in
   other words.

Prints one line per finding and a summary; exits 0 only when every check passes, 1 otherwise, 2 on a usage error.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SECTION_DOCS = ["README.md", "release/QUICKSTART.md", "docs/paper_exhibits.md", "docs/record_names.md",
                "docs/audit.md", "docs/timing_protocol.md", "docs/output.md", "reproduce/README.md",
                "reproduce/heldout/README.md", "release/reference/v2d_t3_ep011_cam_a/README.md", "release/MODELS.md",
                "THIRD_PARTY_LICENSES.md"]
DOCS = ["README.md", "release/QUICKSTART.md", "docs/paper_exhibits.md", "docs/record_names.md", "docs/output.md",
        "docs/audit.md", "docs/timing_protocol.md",
        "reproduce/README.md", "reproduce/heldout/README.md", "release/reference/v2d_t3_ep011_cam_a/README.md",
        "reproduce/e1_verdict_from_scores.py", "reproduce/alongray_shares.py", "reproduce/regenerate_tables.py",
        "reproduce/make_released_tables.py", "reproduce/heldout/regenerate_unit_errors.py", "run/paper/e1_table.py",
        "reproduce/manifest.json"]
# Documents in DOCS that quote their own terms, not the paper: checked for cited numbers only, as the Python files are.
NUMBERS_ONLY = ["docs/audit.md", "docs/timing_protocol.md"]
TITLE = "HandReel: Fast Hand-Object Reconstruction from Video without Per-Frame Pose Sampling"

# Quoted passages that are not quotations of the paper, with their source.
NOT_PAPER: dict[str, str] = {
    "the hand meshes": "the text of release/reference/v2d_t3_ep011_cam_a/ATTRIBUTION.txt",
    "Skipping the post-processing step": "SAM 2's log line",
    "modified: reconstructed by this pipeline": "the text of the run's ATTRIBUTION.txt",
    "build from source": "the section comment of the release/env/*.pip.txt locks",
    "HOI4D: A 4D Egocentric Dataset for Category-Level Human-Object Interaction": "the HOI4D paper's title",
    "DexYCB: A Benchmark for Capturing Hand Grasping of Objects": "the DexYCB paper's title",
    "read by": "a column name of reproduce/README.md's records table",
    "No sealed file and no ours-side held-out output exists or was read.":
        "the text of reproduce/heldout/protocol/2026-10-14-e1-amendment-A-U-unseal.md",
}

# Every main-paper exhibit the documents may cite: the start of its caption (tables, figures, algorithms) or its
# heading (sections), as the submitted main paper prints it.
IDENTITY: dict[tuple[str, str], str] = {
    ("Table", "1"): "Held-out object-centre error",
    ("Table", "2"): "Reconstruction time per window",
    ("Table", "3"): "Reported and measured reconstruction times",
    ("Table", "4"): "Error ratio on fresh HOI4D windows under three depth settings",
    ("Table", "5"): "Stage times for one video, measured separately",
    ("Figure", "1"): "HandReel reconstructs a hand-held object as one mesh",
    ("Figure", "2"): "HandReel is fast without per-frame pose sampling",
    ("Figure", "3"): "One selected clip from start to end",
    ("Figure", "4"): "Six selected development clips with varied objects",
    ("Figure", "5"): "The same Ego4D reconstruction times apply to both applications",
    ("Algorithm", "1"): "The windowed solve",
    ("Sec", "1"): "Introduction",
    ("Sec", "2"): "Related Work",
    ("Sec", "3"): "Problem Statement",
    ("Sec", "4"): "Method",
    ("Sec", "4.1"): "Windows, Keyframes and Perception",
    ("Sec", "4.2"): "The Windowed Solve",
    ("Sec", "4.3"): "Metric Scale and the Scale-Depth Ambiguity",
    ("Sec", "4.4"): "Depth Layers and Quality Check",
    ("Sec", "5"): "Experiments",
    ("Sec", "5.1"): "Setup",
    ("Sec", "5.2"): "Reconstruction Speed and Error",
    ("Sec", "5.3"): "Application 1: Video to Robot Trajectory",
    ("Sec", "5.4"): "Application 2: Many Videos at Once",
    ("Sec", "6"): "Limitations and Future Work",
    ("Sec", "7"): "Conclusion",
}

# Wordings that the submitted paper does not use for its results: (name, pattern). A finite, hand-written list, not
# a test of claims. Matching is case-insensitive, on text with whitespace runs collapsed.
_X = r"\s*(?:x|×)"
FORBIDDEN: list[tuple[str, str]] = [
    ("an end-to-end range of 1.5 to 3.1 (the paper: about 3x, by a sum of separately measured stage times)",
     r"1\.5\s*(?:-|–|—|−|to)\s*3\.1"),
    ("about 2x end to end", r"\b2" + _X + r"\s+end[ -]to[ -]end"),
    ("as accurate as", r"\bas accurate as\b"),
    ("at equal quality", r"\bat equal quality\b"),
    ("halved", r"\bhalv(?:ed|es|ing)\b"),
    ("faster end to end", r"\bfaster end[ -]to[ -]end\b"),
    ("128x (the paper: 130x)", r"\b128\s*(?:x\b|×)"),
    ("orders of magnitude", r"\borders? of magnitude\b"),
    ("any length", r"\bany length\b"),
    (("non-inferior at a 15% margin on HOI4D and on DexYCB, as a blanket claim (the paper: the DexYCB verdict is not "
      "robust if joint failures count against HandReel)"),
     r"non-?inferior at a 15\s?% margin on HOI4D and (?:on )?DexYCB"),
    ("FoundationPose was not timed", r"FoundationPose[^.]{0,80}\bnot timed\b|\bnot timed\b[^.]{0,80}FoundationPose"),
    ("a speed-up (for the many-videos application the paper gives reconstruction time only)", r"\bspeed-?ups?\b"),
    ("warm, for batched runs (each batched run loaded its models once, and the timing includes the load)",
     r"\bwarm\b(?!-up)"),
]
# "17 s" is only ever HandReel's per-window share of batched runs: the phrase must stand near each occurrence.
SHARE_VALUE, SHARE_PHRASE, SHARE_BEFORE, SHARE_AFTER = r"\b17 s\b", "per-window share", 300, 150
# (document, name of the FORBIDDEN entry) pairs that are allowed, with the reason.
ALLOWED: dict[tuple[str, str], str] = {
    ("docs/timing_protocol.md", FORBIDDEN[-1][0]):
        "a general checklist with no result of this project; it defines the cold and warm regimes that a study must "
        "report apart",
    ("docs/timing_protocol.md", FORBIDDEN[-2][0]):
        "a general checklist with no result of this project; it tells a study not to claim a like-for-like speedup",
}
# Not documents of the release: third-party texts, the registered protocol records (byte-pinned, as registered) and
# this checker with its test, which hold the patterns themselves.
PATTERN_SKIP = ("third_party/", "licenses/", "reproduce/heldout/protocol/", "scripts/check_paper_quotes.py",
              "tests/", ".git/", ".venv/", "outputs/", "data/")
PATTERN_PY_DIRS = ("reproduce", "scripts", "run/paper", "release")


def norm(s: str) -> str:
    """The matching form: no whitespace, ASCII hyphens, underscores or backticks; one form of quotes and ligatures."""
    for a, b in (("’", "'"), ("‘", "'"), ("“", '"'), ("”", '"'), ("ﬁ", "fi"),
                 ("ﬂ", "fl"), ("­", "")):
        s = s.replace(a, b)
    return re.sub(r"[\s\-_`]+", "", s)


def pdf_text(path: Path) -> str:
    """The text of `path`: read as is for `.txt`; for a PDF, pdftotext, else PyMuPDF."""
    if path.suffix.lower() == ".txt":
        return path.read_text(encoding="utf-8")
    if shutil.which("pdftotext"):
        return subprocess.run(["pdftotext", "-enc", "UTF-8", str(path), "-"], capture_output=True, text=True,
                              check=True).stdout
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise SystemExit("check_paper_quotes: neither pdftotext nor PyMuPDF is available; pass .txt files") from None
    with fitz.open(str(path)) as doc:
        return "\n".join(page.get_text() for page in doc)


def clean_lines(text: str) -> list[str]:
    """Lines of the paper's text without margin line numbers and running headers: a line of one to four digits, a
    line in capitals that says CONFIDENTIAL, and a line that is one short word in capitals (a template's header)."""
    return [ln for ln in text.splitlines()
            if not re.fullmatch(r"\s*\d{1,4}\s*", ln) and not ("CONFIDENTIAL" in ln and ln == ln.upper())
            and not re.fullmatch(r"\s*[A-Z]{3,6}\s*", ln)]


def doc_text(path: Path) -> str:
    """The prose of one document: fenced code removed; for the manifest, its exhibit labels only."""
    raw = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        return "\n".join(x for o in json.loads(raw).get("outputs", []) for x in o.get("exhibits", []))
    if path.suffix == ".py":
        return "\n".join(re.findall(r'"""(.*?)"""', raw, re.DOTALL) + re.findall(r"#[^\n]*", raw))
    return re.sub(r"(?ms)^```.*?^```", "", raw)


def quotes(text: str) -> list[str]:
    """Double-quoted passages of a document's prose, line breaks joined."""
    text = re.sub(r"`[^`\n]*[\"“”][^`\n]*`", " CODE ", text)   # inline code that holds a quote mark
    text = re.sub(r"\s*\n\s*", " ", text)
    found = re.findall(r"“([^”]{1,800})”", text)
    found += [q for q in re.findall(r'"([^"]{1,800})"', text) if q.strip()]
    return [q.strip() for q in found]


ITEM = {"Table": r"\d+", "Figure": r"\d+", "Sec": r"\d+(?:\.\d+)?", "Algorithm": r"\d+"}
CITES = [  # (pattern of the keyword, kind): citations of the main paper
    (r"(?<!DO AS I DO )\b(?:Tabs?\.|Tables?)(?= \d)", "Table"),
    (r"\b(?:Figs?\.|Figures?)(?= \d)", "Figure"),
    (r"\b(?:Secs?\.|Sections?)(?= \d)", "Sec"),
    (r"\bAlgorithms?(?= \d)", "Algorithm"),
]
# A numbered citation of the supplement, in any of the forms the documents have used.
SUPP_NUMBERED = re.compile(
    r"(?:Supp\.|[Ss]upplement(?:ary(?: [Mm]aterial)?|'s)?)\s+(?:Tabs?\.|Tables?|Figs?\.|Figures?|Secs?\.|Sections?"
    r"|Appendi(?:x|ces)|App\.|Algorithms?)\s*(?:[A-K]\b|\d)[\w.]*"
    r"|\bAppendi(?:x|ces)\s+[A-K]\b(?:\.\d+)?|\bApp\.\s*[A-K]\b(?:\.\d+)?|\bSecs?\.\s+[A-K]\b(?:\.\d+)?")


def supplement_numbers(text: str) -> list[str]:
    """Every numbered citation of the supplement in a document's prose."""
    return [m.group(0) for m in SUPP_NUMBERED.finditer(re.sub(r"\s*\n\s*", " ", text))]


def cited(text: str) -> list[tuple[str, str]]:
    """Every (kind, number) of the main paper a document cites, ranges ("Tables 1 to 5", "5.1-5.4") expanded.
    Numbered citations of the supplement are left out: `supplement_numbers` reports them."""
    text = SUPP_NUMBERED.sub(" ", re.sub(r"\s*\n\s*", " ", text))
    out = []
    for kw, kind in CITES:
        for m in re.finditer(kw, text):
            pos = m.end()
            items: list[str] = []
            while True:
                rest = text[pos:]
                sep = (re.match(r"\s*(?:,\s*(?:and\s+)?|and\s+|to\s+|-|–|\s)\s*", rest) if items
                       else re.match(r"\s*", rest))
                if sep is None:
                    break
                k = re.match(ITEM[kind], rest[sep.end():])
                if k is None:
                    break
                is_range = sep.group(0).strip() in ("to", "-", "–")
                tok = k.group(0)
                if is_range and items and tok.isdigit() and items[-1].isdigit():
                    items += [str(i) for i in range(int(items[-1]) + 1, int(tok) + 1)]
                elif is_range and items and "." in tok and "." in items[-1] and tok[0] == items[-1][0]:
                    head = tok.split(".")[0]
                    a, b = int(items[-1].split(".")[1]), int(tok.split(".")[1])
                    items += [f"{head}.{i}" for i in range(a + 1, b + 1)]
                else:
                    items.append(tok)
                pos += sep.end() + k.end()
                par = re.match(r"\s*\([^()]*(?:\([^()]*\)[^()]*)*\)", text[pos:])     # "2 (row 1) and 5"
                if par and re.match(r"\s*(?:,|and\b)", text[pos + par.end():]):
                    pos += par.end()
            out += [(kind, it) for it in items]
    return out


def section_names(root: Path) -> set[str]:
    """Headings and bold paragraph leads of this release's own Markdown documents."""
    names: set[str] = set()
    for rel in SECTION_DOCS:
        p = root / rel
        if p.is_file():
            raw = p.read_text(encoding="utf-8")
            heads = {re.sub(r"^\d+\.\s*", "", h).strip() for h in re.findall(r"(?m)^#+\s+(.+?)\s*$", raw)}
            names |= heads | {re.sub(r"\s*\([^()]*\)$", "", h) for h in heads}     # "... (not Apache-2.0)"
            names |= {b.strip().rstrip(".") for b in re.findall(r"\*\*([^*\n]{2,80})\*\*", raw)}
    return names


def captions(lines: list[str]) -> dict[tuple[str, str], list[str]]:
    """(kind, number) -> the normalised texts that follow "Table N. ", "Fig. N. " or "Algorithm N " and the section
    headings, from the extracted text of the main PDF. A sentence of the body can end in "Table 4." too, so a number
    can have several candidates; the caption is one of them."""
    flat = " ".join(lines)
    caps: dict[tuple[str, str], list[str]] = {}
    for kind, word in (("Table", r"Table"), ("Figure", r"Fig(?:ure|\.)")):
        for m in re.finditer(rf"\b{word} (\d+)\. (?=(.{{0,160}}))", flat):       # lookahead: captions can abut
            caps.setdefault((kind, m.group(1)), []).append(norm(m.group(2)))
    for m in re.finditer(r"\bAlgorithm (\d+) (?=([A-Z].{0,160}))", flat):          # "Algorithm 1 The windowed solve."
        caps.setdefault(("Algorithm", m.group(1)), []).append(norm(m.group(2)))
    for ln in lines:
        head = re.fullmatch(r"(\d)\.(?:(\d)\.)? ([A-Z].{2,60})", ln.strip())
        if head:
            num = head.group(1) + (f".{head.group(2)}" if head.group(2) else "")
            caps.setdefault(("Sec", num), []).append(norm(head.group(3)))
    return caps


def pattern_docs(root: Path) -> list[str]:
    """The documents of the release that the wording-pattern check reads: every Markdown file, `CITATION.cff`, `NOTICE`
    and the docstrings and comments of the Python files under `PATTERN_PY_DIRS`, without the paths in
    `PATTERN_SKIP`."""
    found: list[str] = []
    stack = [root]
    while stack:
        here = stack.pop()
        for p in sorted(here.iterdir()):
            rel = p.relative_to(root).as_posix()
            if p.is_dir():
                if not p.is_symlink() and not (rel + "/").startswith(PATTERN_SKIP) and p.name != "__pycache__":
                    stack.append(p)
            elif rel.startswith(PATTERN_SKIP):
                continue
            elif (p.suffix == ".md" or rel in ("CITATION.cff", "NOTICE")
                  or (p.suffix == ".py" and rel.startswith(tuple(d + "/" for d in PATTERN_PY_DIRS)))):
                found.append(rel)
    return sorted(found)


def forbidden_wording(root: Path, docs: list[str] | None = None) -> list[str]:
    """One finding per pattern of `FORBIDDEN` that a document matches, and per "17 s" that does not stand near
    "per-window share". Needs no PDF. A pattern check only: an empty result does not show that the claims are
    right."""
    bad: list[str] = []
    for rel in (pattern_docs(root) if docs is None else docs):
        path = root / rel
        raw = doc_text(path) if path.suffix == ".py" else path.read_text(encoding="utf-8")
        text = re.sub(r"\s+", " ", raw)
        for name, pattern in FORBIDDEN:
            m = re.search(pattern, text, re.IGNORECASE)
            if m and (rel, name) not in ALLOWED:
                where = text[max(0, m.start() - 40):m.end() + 40]
                bad.append(f"FORBIDDEN WORDING {rel}: {name}: \"...{where}...\"")
        for m in re.finditer(SHARE_VALUE, text):
            near = text[max(0, m.start() - SHARE_BEFORE):m.end() + SHARE_AFTER]
            if SHARE_PHRASE not in near:
                bad.append(f"FORBIDDEN WORDING {rel}: 17 s without \"{SHARE_PHRASE}\" beside it: "
                           f"\"...{text[max(0, m.start() - 40):m.end() + 40]}...\"")
    return bad


def check(root: Path, main_text: str, log: list[str] | None = None) -> tuple[list[str], dict[str, int]]:
    """All findings (empty when everything passes) and the counts for the summary; `log` collects one line per
    quotation and per cited exhibit."""
    log = [] if log is None else log
    lines = clean_lines(main_text)
    body = norm(" ".join(lines))
    caps = captions(lines)
    own = section_names(root)
    bad: list[str] = []
    n = {"quotes": 0, "in_main": 0, "own_section": 0, "not_paper": 0, "cites": 0, "exhibits": 0, "patterns": 0}
    seen: set[tuple[str, str]] = set()
    for rel in DOCS:
        text = doc_text(root / rel)
        for q in (quotes(text) if rel.endswith(".md") and rel not in NUMBERS_ONLY else []):
            n["quotes"] += 1
            if q in own:
                n["own_section"] += 1
                where = "a section of this release"
            elif q in NOT_PAPER:
                n["not_paper"] += 1
                where = "not the paper's: " + NOT_PAPER[q]
            elif norm(q) in body:
                n["in_main"] += 1
                where = "main paper, verbatim"
            else:
                bad.append(f"QUOTE NOT FOUND {rel}: \"{q}\"")
                where = "NOT FOUND"
            log.append(f"quote [{where}] {rel}: \"{q}\"")
        for s in supplement_numbers(text):
            bad.append(f"SUPPLEMENT NUMBER {rel}: \"{s}\" (cite the supplement without numbers)")
        for c in cited(text):
            n["cites"] += 1
            seen.add(c)
            if c not in IDENTITY:
                last = max((int(k[1]) for k in caps if k[0] == c[0] and k[1].isdigit()), default=0)
                if c[0] != "Sec" and c[1].isdigit() and int(c[1]) > last:
                    bad.append(f"NOT IN THE MAIN PAPER {rel}: {c[0]} {c[1]} (the main paper's last is {last}; cite "
                               "the supplement without numbers)")
                else:
                    bad.append(f"CITED, IDENTITY NOT DECLARED {rel}: {c[0]} {c[1]}")
    for c in sorted(seen):
        if c not in IDENTITY:
            continue
        n["exhibits"] += 1
        got = caps.get(c)
        if got is None:
            bad.append(f"NOT IN THE PDF: {c[0]} {c[1]}")
        elif not any(g.startswith(norm(IDENTITY[c])) for g in got):
            bad.append(f"WRONG EXHIBIT: {c[0]} {c[1]} is \"{got[0][:60]}...\", expected \"{IDENTITY[c]}\"")
        else:
            log.append(f"exhibit [ok] {c[0]} {c[1]}: \"{IDENTITY[c]}...\"")
    if norm(TITLE) not in body:
        bad.append(f"TITLE NOT IN THE MAIN PDF: {TITLE}")
    for rel in ("CITATION.cff", "README.md"):
        if TITLE not in " ".join((root / rel).read_text(encoding="utf-8").split()):
            bad.append(f"TITLE MISSING OR DIFFERENT: {rel}")
    matches = forbidden_wording(root)
    n["patterns"] = len(matches)
    return bad + matches, n


LIMIT = ("This is not a semantic check of claims: it does not test whether a sentence says what its cited exhibit "
         "shows, and a paraphrase or a changed unit passes the patterns. Audit each results sentence against the paper "
         "by hand.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], epilog=LIMIT)
    ap.add_argument("--main", type=Path, help="the main paper, .pdf or extracted .txt")
    ap.add_argument("--root", type=Path, default=ROOT, help="the release checkout (default: this one)")
    ap.add_argument("--patterns-only", action="store_true",
                    help="run only the wording-pattern check (a finite list of regular expressions); needs no PDF")
    ap.add_argument("--list", action="store_true", help="also print every quotation and cited exhibit with its result")
    a = ap.parse_args(argv)
    if a.patterns_only:
        bad = forbidden_wording(a.root)
        for line in bad:
            print(line)
        print(f"wording patterns: {len(pattern_docs(a.root))} documents, {len(FORBIDDEN) + 1} patterns")
        print(f"ok: no configured-pattern violation found. {LIMIT}" if not bad else f"FAIL: {len(bad)} finding(s)")
        return 0 if not bad else 1
    if a.main is None:
        ap.error("--main is required (or pass --patterns-only)")
    if not a.main.is_file():
        print(f"check_paper_quotes: {a.main} not found", file=sys.stderr)
        return 2
    log: list[str] = []
    bad, n = check(a.root, pdf_text(a.main), log)
    if a.list:
        print("\n".join(log))
    for line in bad:
        print(line)
    print(f"quotations: {n['quotes']} ({n['in_main']} in the main paper, {n['own_section']} naming a section of this "
          f"release, {n['not_paper']} listed as not the paper's); citations: {n['cites']} of {n['exhibits']} "
          f"main-paper exhibits, each number checked against its declared caption or heading; wording-pattern "
          f"matches: {n['patterns']} ({len(FORBIDDEN) + 1} patterns)")
    print("ok: every cited exhibit number exists in the PDF with the declared caption or heading; no numbered "
          "supplement citation; every quotation is a section name, a declared passage or verbatim in the main paper; "
          f"the title matches; no configured-pattern violation found. {LIMIT}" if not bad
          else f"FAIL: {len(bad)} finding(s)")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
