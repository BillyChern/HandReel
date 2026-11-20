"""CPU tests for reproduce/regenerate_tables.py on a synthetic paper source (no paper records are needed).

The synthetic tree has the paper source's layout: `tools/make_tables.py` (writes one table from one record) and
`records/` (that record and an E1 verdict record in the shape `run/paper/e1_table.py` reads).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location("regenerate_tables", ROOT / "reproduce" / "regenerate_tables.py")
R = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(R)

GENERATOR = '''import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
v = json.load(open(os.path.join(HERE, "..", "records", "a.json")))["v"]
os.makedirs(os.path.join(HERE, "..", "tables"), exist_ok=True)
open(os.path.join(HERE, "..", "tables", "t.tex"), "w").write(f"value {v}\\n")
'''


def _corpus(n: int) -> dict:
    return {"n_assigned": n, "n_run": n, "n_analysed": n, "delivered_ours": n, "delivered_theirs": n,
            "itt": {"ratio": 0.5, "ci95": [-1.0, -0.5]}, "verdict": "NON-INFERIOR", "lower_error": True}


def _verdict() -> dict:
    corpora = {"H": _corpus(3), "D": _corpus(4)}
    return {"primary_O_EST": {"primary": {"corpora": corpora}},
            "secondary_S2_O_CAL": {"primary": {"corpora": corpora}},
            "headline": {"H": {"headline_non_inferior": True}, "D": {"headline_non_inferior": False}}}


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _paper(tmp_path: Path, value: int = 7) -> tuple[Path, Path]:
    src = tmp_path / "paper"
    (src / "tools").mkdir(parents=True)
    (src / "records").mkdir()
    (src / "tools" / "make_tables.py").write_text(GENERATOR)
    (src / "records" / "a.json").write_text(json.dumps({"v": value}))
    (src / "records" / "verdict.json").write_text(json.dumps(_verdict()))
    ref = tmp_path / "ref_e1.tex"           # the expected E1 macros: a reference render of the same record
    subprocess.run([sys.executable, "-s", "-B", str(R.E1_TABLE), "--verdict", str(src / "records" / "verdict.json"),
                    "--out", str(ref)], check=True)
    manifest = {
        "paper_revision": "synthetic",
        "generator": {"path": "tools/make_tables.py", "sha256": _sha((src / "tools" / "make_tables.py").read_bytes())},
        "e1_verdict_record": "verdict.json",
        "records": [{"path": f"records/{n}", "sha256": _sha((src / "records" / n).read_bytes())}
                    for n in ("a.json", "verdict.json")],
        "outputs": [{"file": "tables/t.tex", "sha256": _sha(b"value 7\n")},
                    {"file": "e1_results.tex", "root": "out", "compare": "body", "sha256": R.sha256_body(ref)}],
    }
    m = tmp_path / "manifest.json"
    m.write_text(json.dumps(manifest))
    return src, m


def _tree_digest(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): _sha(p.read_bytes()) for p in sorted(root.rglob("*")) if p.is_file()}


def test_matching_inputs_regenerate_every_output_and_leave_the_source_alone(tmp_path: Path) -> None:
    src, m = _paper(tmp_path)
    before = _tree_digest(src)
    assert R.main(["--paper-src", str(src), "--out", str(tmp_path / "out"), "--manifest", str(m)]) == 0
    rep = json.loads((tmp_path / "out" / "report.json").read_text())
    assert rep["ok"] and all(o["ok"] for o in rep["outputs"]) and len(rep["outputs"]) == 2
    assert _tree_digest(src) == before


def test_a_changed_record_is_refused_before_anything_runs(tmp_path: Path) -> None:
    src, m = _paper(tmp_path)
    (src / "records" / "a.json").write_text(json.dumps({"v": 8}))
    assert R.main(["--paper-src", str(src), "--out", str(tmp_path / "out"), "--manifest", str(m)]) == 1
    rep = json.loads((tmp_path / "out" / "report.json").read_text())
    assert rep["error"] == "input sha256 mismatch" and rep["runs"] == []


def test_unpinned_run_reports_the_changed_output(tmp_path: Path) -> None:
    src, m = _paper(tmp_path)
    (src / "records" / "a.json").write_text(json.dumps({"v": 8}))
    out = tmp_path / "out"
    assert R.main(["--paper-src", str(src), "--out", str(out), "--manifest", str(m), "--allow-unpinned"]) == 1
    rep = json.loads((out / "report.json").read_text())
    assert [o["file"] for o in rep["outputs"] if not o["ok"]] == ["tables/t.tex"]


def test_missing_records_are_reported(tmp_path: Path) -> None:
    src, m = _paper(tmp_path)
    (src / "records" / "verdict.json").unlink()
    assert R.main(["--paper-src", str(src), "--out", str(tmp_path / "out"), "--manifest", str(m)]) == 1
    assert "records/verdict.json" in json.loads((tmp_path / "out" / "report.json").read_text())["error"]


def test_out_must_be_empty_and_outside_the_source(tmp_path: Path) -> None:
    src, m = _paper(tmp_path)
    with pytest.raises(SystemExit):
        R.main(["--paper-src", str(src), "--out", str(src / "out"), "--manifest", str(m)])
    busy = tmp_path / "busy"
    busy.mkdir()
    (busy / "x").write_text("x")
    with pytest.raises(SystemExit):
        R.main(["--paper-src", str(src), "--out", str(busy), "--manifest", str(m)])


def test_shipped_manifest_pins_the_generator_every_record_and_the_main_tables() -> None:
    m = json.loads((ROOT / "reproduce" / "manifest.json").read_text())
    assert len(m["generator"]["sha256"]) == 64 and m["paper_revision"]
    assert all(len(r["sha256"]) == 64 and r["release"].startswith(("NOT RELEASED", "PROPOSED FOR RELEASE"))
               for r in m["records"])
    files = {o["file"]: o for o in m["outputs"]}
    assert {"tables/main.tex", "tables/bench.tex", "e1_results.tex"} <= set(files)
    assert files["e1_results.tex"]["compare"] == "body"
    recs = {r["path"].removeprefix("records/") for r in m["records"]}
    assert all(set(o["reads"]) <= recs for o in m["outputs"])


def test_repin_is_one_command_and_the_new_pins_then_check(tmp_path: Path) -> None:
    src, m = _paper(tmp_path)
    (src / "records" / "a.json").write_text(json.dumps({"v": 9}))            # the paper source moved on
    assert R.main(["--paper-src", str(src), "--out", str(tmp_path / "check"), "--manifest", str(m)]) == 1
    assert R.main(["--paper-src", str(src), "--out", str(tmp_path / "repin"), "--manifest", str(m),
                   "--repin", "rev2"]) == 0
    new = json.loads(m.read_text())
    assert new["paper_revision"] == "rev2"
    assert {o["file"] for o in new["outputs"]} == {"tables/t.tex", "e1_results.tex"}
    assert next(o for o in new["outputs"] if o["file"] == "tables/t.tex")["sha256"] == _sha(b"value 9\n")
    assert all(r["release"].startswith("NOT RELEASED") for r in new["records"])
    assert R.main(["--paper-src", str(src), "--out", str(tmp_path / "recheck"), "--manifest", str(m)]) == 0


def _shipped(m: dict) -> list[dict]:
    return [r for r in m["records"] if r["release"].startswith("PROPOSED FOR RELEASE")]


def test_the_shipped_records_are_exactly_the_proposed_ones_and_feed_the_released_tables() -> None:
    m = json.loads((ROOT / "reproduce" / "manifest.json").read_text())
    shipped = _shipped(m)
    on_disk = sorted(p.relative_to(ROOT / "reproduce").as_posix() for p in (ROOT / "reproduce" / "records").rglob("*")
                     if p.is_file())
    assert on_disk == sorted(r.get("release_path", r["path"]) for r in shipped)
    assert all(_sha((ROOT / "reproduce" / r.get("release_path", r["path"])).read_bytes()) == r["sha256"] for r in shipped)
    assert not [p for p in on_disk if re.search(r"(^|/)s[0-9]{3}", p)]      # release names carry no study numbers
    names = {r["path"].removeprefix("records/") for r in shipped}
    released = [o for o in m["outputs"] if "released" in o]
    assert {o["released"].get("file", o["file"]) for o in released} == {
        f"tables/{t}.tex" for t in ("solve", "optim", "stride", "hyper", "weak_rows")}
    assert all(set(o["reads"]) <= names for o in released)
    # every output whose records are all shipped is regenerated by the released check
    assert [o["file"] for o in m["outputs"] if "released" not in o and o.get("root") != "out" and
            set(o["reads"]) <= names] == []


def test_the_released_check_regenerates_the_five_tables_from_this_folder(tmp_path: Path) -> None:
    out = tmp_path / "out"
    assert R.main(["--out", str(out)]) == 0
    rep = json.loads((out / "report.json").read_text())
    assert rep["mode"] == "released" and rep["ok"] and len(rep["outputs"]) == 5 and len(rep["inputs"]) == 14
    assert all((out / o["file"]).read_text().startswith("% GENERATED by reproduce/make_released_tables.py")
               for o in rep["outputs"])


def _copied_reproduce(tmp_path: Path):
    """A copy of reproduce/ (script, released generator, manifest, records) loaded as its own module."""
    d = tmp_path / "reproduce"
    shutil.copytree(ROOT / "reproduce", d, ignore=shutil.ignore_patterns("__pycache__"))
    spec = importlib.util.spec_from_file_location("regenerate_tables_copy", d / "regenerate_tables.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return d, mod


def test_a_changed_shipped_record_stops_the_released_check(tmp_path: Path) -> None:
    d, mod = _copied_reproduce(tmp_path)
    rec = d / "records" / "weak_rows_summary.json"
    data = json.loads(rec.read_text())
    data["n_windows"] = 45
    rec.write_text(json.dumps(data))
    assert mod.main(["--out", str(tmp_path / "out")]) == 1
    rep = json.loads((tmp_path / "out" / "report.json").read_text())
    assert rep["runs"] == [] and [i["path"] for i in rep["inputs"] if not i["ok"]] == [
        "reproduce/records/weak_rows_summary.json"]


def test_an_unlisted_file_in_the_records_folder_stops_the_released_check(tmp_path: Path) -> None:
    d, mod = _copied_reproduce(tmp_path)
    (d / "records" / "extra.json").write_text("{}")
    assert mod.main(["--out", str(tmp_path / "out")]) == 1
    assert json.loads((tmp_path / "out" / "report.json").read_text())["unlisted_files"] == ["records/extra.json"]


def test_a_changed_table_line_fails_and_a_changed_comment_line_does_not(tmp_path: Path) -> None:
    t = tmp_path / "t.tex"
    t.write_text("% header\n\\begin{table}\n  x & 1 \\\\\n\\end{table}\n")
    base = R.sha256_content(t)
    t.write_text("% another header\n  % note\n\\begin{table}\n  x & 1 \\\\\n\\end{table}\n")
    assert R.sha256_content(t) == base
    t.write_text("% header\n\\begin{table}\n  x & 2 \\\\\n\\end{table}\n")
    assert R.sha256_content(t) != base


def test_the_released_check_refuses_repin_and_unpinned_runs(tmp_path: Path) -> None:
    for extra in (["--repin", "x"], ["--allow-unpinned"]):
        with pytest.raises(SystemExit):
            R.main(["--out", str(tmp_path / "o"), *extra])


def test_repin_also_rewrites_the_released_content_hash(tmp_path: Path) -> None:
    src, m = _paper(tmp_path)
    man = json.loads(m.read_text())
    man["outputs"][0]["released"] = {"script": "x", "content_sha256": "stale"}
    m.write_text(json.dumps(man))
    (src / "records" / "a.json").write_text(json.dumps({"v": 9}))
    assert R.main(["--paper-src", str(src), "--out", str(tmp_path / "repin"), "--manifest", str(m),
                   "--repin", "rev2"]) == 0
    t = next(o for o in json.loads(m.read_text())["outputs"] if o["file"] == "tables/t.tex")
    assert t["released"] == {"script": "x", "content_sha256": _sha(b"value 9\n")}
