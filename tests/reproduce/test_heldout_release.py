"""reproduce/heldout/, regenerate_unit_errors.py and e1_verdict_from_scores.py: no per-unit error derived from ground
truth ships; the verdict check skips without a regenerated file, and on a regenerated one (synthetic errors here) it
computes and compares every aggregate value, the median errors included; the shipped files hold identifiers only."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
HELDOUT = ROOT / "reproduce" / "heldout"
_SPEC = importlib.util.spec_from_file_location("e1_verdict_from_scores", ROOT / "reproduce" / "e1_verdict_from_scores.py")
V = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(V)
_RSPEC = importlib.util.spec_from_file_location("regenerate_unit_errors", HELDOUT / "regenerate_unit_errors.py")
R = importlib.util.module_from_spec(_RSPEC)
_RSPEC.loader.exec_module(R)


def _synthetic_results(seed: int = 7) -> list[dict]:
    """One result row per assigned unit with synthetic errors (no dataset involved): most units run and deliver."""
    rng = np.random.default_rng(seed)
    rows = []
    for uid in sorted(R.load_units()):
        run = bool(rng.random() > 0.1)
        side = {}
        for key, scale in (("ours", 60.0), ("theirs", 100.0)):
            ok = run and bool(rng.random() > 0.2)
            side[key] = {"delivered": ok, "e_mm": float(rng.lognormal(np.log(scale), 0.6)) if ok else None}
        rows.append({"unit_id": uid, "run": run, **side})
    return rows


def _write(tmp_path: Path, rows: list[dict]) -> Path:
    res, out = tmp_path / "results.jsonl", tmp_path / "unit_errors_est.json"
    res.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert R.main(["--results", str(res), "--out", str(out)]) == 0
    return out


def test_no_per_unit_error_file_ships() -> None:
    """Per-unit errors derived from HOI4D or DexYCB ground truth are never part of the release."""
    assert not (HELDOUT / "e1_" "scores_est.json").exists()
    for p in HELDOUT.glob("*.json"):
        if p.name != R.DEFAULT_OUT.name:                                    # a user's own regenerated copy
            assert '"e_mm"' not in p.read_text(), p.name


def test_without_a_regenerated_file_the_check_is_skipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                         capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(V, "DEFAULT_SCORES", tmp_path / "unit_errors_est.json")
    monkeypatch.setattr(V, "ROOT", tmp_path)
    assert V.main([]) == 0
    out = capsys.readouterr().out
    assert out.startswith("skipped:") and "regenerate_unit_errors.py" in out
    assert V.main(["--scores", str(tmp_path / "missing.json")]) == 2


def test_a_regenerated_file_is_checked_cell_by_cell(tmp_path: Path) -> None:
    """Synthetic errors give values that differ from the references (exit 1); with those values as the reference
    every value matches (exit 0), and the median row is the median of the both-delivered units' errors."""
    rows = _synthetic_results()
    path = _write(tmp_path, rows)
    out = tmp_path / "cells.json"
    assert V.main(["--scores", str(path), "--json", str(out)]) == 1
    cells = json.loads(out.read_text())["cells"]
    an = [r for r in json.loads(path.read_text())["rows"] if r["run"] and r["removed"] is None]
    both = [r for r in an if r["corpus"] == "H" and r["ours"]["delivered"] and r["theirs"]["delivered"]]
    med = (np.median([r["ours"]["e_mm"] for r in both]), np.median([r["theirs"]["e_mm"] for r in both]))
    assert cells["H"]["median_mm"] == f"{med[0]:.1f}/{med[1]:.1f}"
    assert cells["H"]["delivered"].endswith(f"/{len(both)}")
    old = V.PRINTED
    try:
        V.PRINTED = cells
        assert V.main(["--scores", str(path)]) == 0
    finally:
        V.PRINTED = old


def test_the_median_row_is_checked(tmp_path: Path) -> None:
    """With the cells of one regenerated file as the reference, doubling our HOI4D errors on the units both delivered
    moves the median cell, and the check then fails (exit 1)."""
    rows = _synthetic_results()
    thr = V.ST.load_thresholds(HELDOUT / "e1-power-size.json")
    base = V.cells(V.load_rows(_write(tmp_path, rows)), thr)
    for r in rows:
        if r["unit_id"].startswith("H:") and r["ours"]["delivered"] and r["theirs"]["delivered"]:
            r["ours"]["e_mm"] *= 2.0
    path = _write(tmp_path, rows)
    assert V.cells(V.load_rows(path), thr)["H"]["median_mm"] != base["H"]["median_mm"]
    old = V.PRINTED
    try:
        V.PRINTED = base
        assert V.main(["--scores", str(path)]) == 1
    finally:
        V.PRINTED = old


@pytest.mark.parametrize(("change", "message"), [
    (lambda rows: rows.pop(), "assigned units have no row"),
    (lambda rows: rows.append(dict(rows[0])), "given twice"),
    (lambda rows: rows[0].update(unit_id="H:not_a_unit"), "is not a unit"),
    (lambda rows: rows[0].update(ours={"delivered": True, "e_mm": None}), "e_mm exactly when delivered"),
    (lambda rows: rows[0].update(run=False, ours={"delivered": True, "e_mm": 1.5}), "carries no delivered result"),
    (lambda rows: rows.__setitem__(0, [rows[0]["unit_id"]]), "row 1: not a JSON object"),
    (lambda rows: rows.__setitem__(0, "a string"), "row 1: not a JSON object"),
    (lambda rows: rows[0].update(score=3), "unknown key(s) score"),
    (lambda rows: rows[0].update(corpus="D" if rows[0]["unit_id"].startswith("H:") else "H"),
     "but the shipped unit lists give"),
])
def test_the_regenerator_refuses_bad_rows(tmp_path: Path, change, message: str,
                                          capsys: pytest.CaptureFixture[str]) -> None:
    rows = _synthetic_results()
    change(rows)
    res = tmp_path / "results.jsonl"
    res.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert R.main(["--results", str(res), "--out", str(tmp_path / "u.json")]) == 1
    err = capsys.readouterr().err
    assert message in err and "Traceback" not in err and err.count("\n") == 1
    assert not (tmp_path / "u.json").exists()


def test_rows_that_repeat_the_lists_values_are_accepted() -> None:
    units = R.load_units()
    rows = [{**r, **{k: units[r["unit_id"]][k] for k in R.FROM_LISTS}} for r in _synthetic_results()]
    assert len(R.build(rows)["rows"]) == 180


def test_the_regenerator_marks_the_registered_removals_and_computes_centre_errors() -> None:
    doc = R.build(_synthetic_results())
    assert sorted(r["unit_id"] for r in doc["rows"] if r["removed"]) == sorted(R.REMOVED)
    units = R.load_units()
    assert all(u in units for u in R.REMOVED)
    frames = np.arange(0, 10)
    gt = np.zeros((10, 3)) + [0.0, 0.0, 0.5]
    pred = gt + [0.003, 0.004, 0.0]                                          # 5 mm off on every frame
    valid = np.ones(10, bool)
    valid[3] = False
    assert R.unit_centre_error_mm(frames, pred, frames, gt, valid, scored=list(frames)) == pytest.approx(5.0)
    assert R.unit_centre_error_mm(frames, pred, frames, gt, valid, scored=[0, 2]) == pytest.approx(5.0)
    with pytest.raises(ValueError, match="not in the ground truth"):
        R.unit_centre_error_mm(frames, pred, frames + 100, gt, valid, scored=list(frames))
    with pytest.raises(ValueError, match="no finite prediction"):                # a missing scored frame is refused
        R.unit_centre_error_mm(frames[:-1], pred[:-1], frames, gt, valid, scored=list(frames))
    nan = pred.copy()
    nan[5] = np.nan
    with pytest.raises(ValueError, match="no finite prediction"):
        R.unit_centre_error_mm(frames, nan, frames, gt, valid, scored=list(frames))
    with pytest.raises(ValueError, match="repeat"):                              # a repeated prediction row is refused
        R.unit_centre_error_mm(np.r_[frames, 4], np.r_[pred, pred[4:5]], frames, gt, valid, scored=list(frames))
    with pytest.raises(ValueError, match="repeat"):
        R.unit_centre_error_mm(frames, pred, frames, gt, valid, scored=[0, 0, 2])
    assert R.unit_centre_error_mm(frames, nan, frames, gt, valid, scored=[0, 3, 4]) == pytest.approx(5.0)  # 3 invalid
    with pytest.raises(ValueError, match="integer"):                             # 3.5 is refused, never read as 3
        R.unit_centre_error_mm(frames, pred, frames, gt, valid, scored=[0, 3.5])
    with pytest.raises(ValueError, match="integer"):
        R.complete_centres([0, 3.5], frames, pred, np.ones(10, bool))
    bad_gt = gt.copy()
    bad_gt[4, 0] = np.inf                                                        # marked valid, centre not finite
    with pytest.raises(ValueError, match=r"\[4\] are marked valid but"):
        R.unit_centre_error_mm(frames, pred, frames, bad_gt, valid, scored=list(frames))
    bad_gt[4], bad_gt[3, 0] = gt[4], np.nan                                      # frame 3 is NOT valid: no refusal
    assert R.unit_centre_error_mm(frames, pred, frames, bad_gt, valid, scored=list(frames)) == pytest.approx(5.0)
    h = next(v["unit"] for v in units.values() if v["corpus"] == "H")
    assert R.processed_frames(h) == list(range(h["start"], h["stop"], h["stride"]))


def test_completion_rule_differs_from_averaging_the_delivered_frames() -> None:
    """Frames 0, 3, 6 with a constant ground truth, 10 mm off at 0, 40 mm at 6 and no pose at 3: the fill takes
    frame 0 for frame 3 (earlier on the tie), so the unit scores 20 mm; averaging the two delivered frames gives 25."""
    grid, gt = [0, 3, 6], np.zeros((3, 3)) + [0.0, 0.0, 0.5]
    pf, px = np.array([0, 6]), np.array([[0.010, 0.0, 0.5], [0.040, 0.0, 0.5]])
    done = R.complete_centres(grid, pf, px, np.array([True, True]))
    assert np.allclose(done, [px[0], px[0], px[1]])
    assert R.unit_centre_error_mm(np.array(grid), done, np.array(grid), gt, np.ones(3, bool), grid) == \
        pytest.approx(20.0)
    assert R.unit_centre_error_mm(pf, px, np.array(grid), gt, np.ones(3, bool), [0, 6]) == pytest.approx(25.0)
    with pytest.raises(ValueError, match="no finite prediction"):
        R.unit_centre_error_mm(pf, px, np.array(grid), gt, np.ones(3, bool), grid)
    # an unusable or non-finite pose is filled from the nearest usable one; later frame wins only when nearer
    pf3, px3 = np.array([0, 3, 6]), np.array([[0.010, 0.0, 0.5], [np.nan, 0.0, 0.5], [0.040, 0.0, 0.5]])
    assert np.allclose(R.complete_centres(grid, pf3, px3, np.ones(3, bool)), [px[0], px[0], px[1]])
    assert np.allclose(R.complete_centres([0, 5, 6], pf, px, np.array([True, True])), [px[0], px[1], px[1]])
    assert np.allclose(R.complete_centres(grid, pf3, np.nan_to_num(px3), np.array([False, True, True]))[0],
                       np.nan_to_num(px3)[1])
    with pytest.raises(ValueError, match="no usable pose"):
        R.complete_centres(grid, pf, px, np.array([False, False]))
    with pytest.raises(ValueError, match="repeat"):
        R.complete_centres(grid, np.array([0, 0]), px, np.array([True, True]))


@pytest.mark.parametrize(("name", "corpus", "n", "dataset"), [
    ("units_hoi4d_72.json", "H", 72, "HOI4D (https://hoi4d.github.io/), CC BY-NC 4.0"),
    ("units_dexycb_s08_108.json", "D", 108, "DexYCB (https://dex-ycb.github.io/), CC BY-NC 4.0")])
def test_the_unit_lists_are_identifiers(name: str, corpus: str, n: int, dataset: str) -> None:
    doc = json.loads((HELDOUT / name).read_text())
    assert doc["corpus"] == corpus and doc["n_units"] == n == len(doc["units"]) and doc["dataset"] == dataset
    assert len({u["unit_id"] for u in doc["units"]}) == n
    for u in doc["units"]:
        assert "shift_px" not in u and all(not isinstance(v, (list, dict)) for v in u.values())
        if corpus == "D":
            assert u["subject"] == "20201002-subject-08"


def test_the_thresholds_are_the_registered_calibration() -> None:
    sys.path.insert(0, str(ROOT / "run" / "study"))
    import e1_stats

    thr = e1_stats.load_thresholds(HELDOUT / "e1-power-size.json")
    assert thr["H"] == {"0.025": 0.025, "0.05": 0.05}
    assert thr["D"] == pytest.approx({"0.025": 0.017991004497751123, "0.05": 0.03798100949525238})


def test_the_protocol_copies_match_their_redaction_log() -> None:
    log = json.loads((HELDOUT / "protocol" / "redactions.json").read_text())
    shas = {e["file"]: e["change"] for e in log if e["line"] == 0}
    files = sorted(p.name for p in (HELDOUT / "protocol").glob("*.md"))
    assert files == sorted(shas) and len(files) == 11
    host = re.compile(r"/w[o]rkspace|fastdo-w[o]rktrees|GPT_VLA")
    # internal references edited for public release: record files, scratch paths, record numbers, role wording
    internal = re.compile(r"p[l]an/|t[e]mp/|#\d{3}|c[o]ntroller|\bU[S]ER\b|\ba[g]ents?\b|session login|origin/main|\bmain [0-9a-f]{8}\b")
    for name, change in shas.items():
        released = change.split()[-1]
        body = (HELDOUT / "protocol" / name).read_bytes()
        assert hashlib.sha256(body).hexdigest() == released, name
        assert not host.search(body.decode()), name
        assert not internal.search(body.decode()), (name, internal.search(body.decode()))
    edits = [e for e in log if e["line"] > 0]
    assert all(e["file"] in shas and e["change"] for e in edits) and len(edits) > 100


def test_the_qc_rule_constants_load_and_round_to_the_pinned_values() -> None:
    """The constants-only copy of the registered QC rule loads as a version-1 AcceptanceRule, its constants round to
    the six-significant-figure values pinned here, and it holds no path, clip or sequence ID."""
    from fastdo.qc import load_acceptance_rule

    path = HELDOUT / "qc_rule_constants.json"
    rule = load_acceptance_rule(path)
    assert rule.calibration.threshold == pytest.approx(0.690775, abs=5e-7) and rule.calibration.n_calibration == 19
    doc = json.loads(path.read_text())
    sig6 = [float(f"{v:.6g}") for v in doc["logistic_weights"] + doc["feature_mean"] + doc["feature_std"]]
    assert sig6 == [0.238523, 0.66135, 1.14023, -0.237001, -0.0824555, -0.76298,
                    0.567015, 0.532037, 0.107748, 0.0134436, 0.534917,
                    0.131976, 0.415348, 0.0249662, 0.0113688, 0.168629]
    assert doc["provenance"]["registered_sha256_prefix_of_original"] == "e564b52e92d3"
    assert not {"sequences", "clip_ids", "skipped", "git_commit", "artifact_root", "features_file"} & set(
        doc["provenance"])
    text = path.read_text()
    assert "/" not in text and not re.search(r"subject-0\d|\d{8}_\d{6}", text)
    assert hashlib.sha256(path.read_bytes()).hexdigest()[:12] != "e564b52e92d3"


def test_the_derivation_script_rebuilds_the_copy_from_a_rule_with_full_provenance(tmp_path: Path) -> None:
    """derive_qc_rule_constants.py keeps the constants and the fit's counts, drops IDs and paths, and gives the shipped
    bytes; an input that is not the registered file is refused without --allow-unregistered."""
    spec = importlib.util.spec_from_file_location("derive_qc", HELDOUT / "derive_qc_rule_constants.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    copy = json.loads((HELDOUT / "qc_rule_constants.json").read_text())
    full = {**copy, "provenance": {**{k: copy["provenance"][k] for k in mod.PROVENANCE_KEYS},
                                   "git_commit": "0" * 40, "sequences": ["x/y/z"], "artifact_root": "/abs/path"}}
    src = tmp_path / "rule.json"
    src.write_text(json.dumps(full))
    with pytest.raises(SystemExit, match="not the registered"):
        mod.main([str(src), "--out", str(tmp_path / "o.json")])
    assert mod.main([str(src), "--check", str(HELDOUT / "qc_rule_constants.json"), "--allow-unregistered"]) == 0


def test_a_corpus_with_no_unit_delivered_by_both_names_what_does_not_exist(tmp_path: Path) -> None:
    """A3 #10861: every unit runs, HandReel delivers the even units and DO AS I DO the odd ones, so no unit has
    both. The registered failure-rule ratio still exists; the complete-case cells are named unavailable, with no
    crash, no NaN and no LaTeX placeholder."""
    rows = []
    for i, uid in enumerate(sorted(R.load_units())):
        ours, theirs = i % 2 == 0, i % 2 == 1
        rows.append({"unit_id": uid, "run": True, "ours": {"delivered": ours, "e_mm": 40.0 + i if ours else None},
                     "theirs": {"delivered": theirs, "e_mm": 90.0 + i if theirs else None}})
    out = tmp_path / "cells.json"
    assert V.main(["--scores", str(_write(tmp_path, rows)), "--json", str(out)]) == 1
    for c in ("H", "D"):
        cell = json.loads(out.read_text())["cells"][c]
        assert cell["delivered"].endswith("/0")
        assert cell["complete_case"] == cell["within_50mm"] == cell["median_mm"] == V.NO_PAIR
        assert cell["mitt"].split(",")[0] != V.UNAVAILABLE and "nan" not in json.dumps(cell) and "TBD" not in json.dumps(cell)


def test_an_interval_end_past_the_float_range_is_named() -> None:
    assert V.ratio_interval({"ratio": 2.0, "ci95": [0.0, 800.0]}) == "2.00 [1.00, above the float range]"
    assert V.ratio_interval({"ratio": 2.0, "ci95": [math.log(0.5), math.log(4.0)]}) == "2.00 [0.50, 4.00]"
    assert V.ratio_interval({"n": 0, "median": None}) == V.UNAVAILABLE
    assert V.ratio_interval({"ratio": None, "ci95": [0.0, 1.0]}) == V.UNAVAILABLE
    assert V.ratio_interval({"ratio": 2.0, "ci95": None}) == "2.00 unavailable"
    block = {"ratio": 1.37, "ci95": [-0.2, 0.6]}
    assert V.ratio_interval(block) == f"{V.ET._ratio(1.37)} {V.ET._ci([-0.2, 0.6])}"   # finite values: unchanged


def test_an_unregistered_input_is_never_written_or_reported_as_the_registered_rule(tmp_path: Path,
                                                                                capsys: pytest.CaptureFixture[str]) -> None:
    """A3 #10856: with --allow-unregistered the output must say what it is. The written file carries no registered
    prefix and names the input's sha256; the check prints that a match is about the constants only."""
    spec = importlib.util.spec_from_file_location("derive_qc", HELDOUT / "derive_qc_rule_constants.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    shipped = (HELDOUT / "qc_rule_constants.json").read_bytes()
    copy = json.loads(shipped)
    full = {**copy, "provenance": {**{k: copy["provenance"][k] for k in mod.PROVENANCE_KEYS}, "sequences": ["x/y/z"]}}
    src = tmp_path / "rule.json"
    src.write_text(json.dumps(full))
    sha = hashlib.sha256(src.read_bytes()).hexdigest()
    out = tmp_path / "o.json"
    assert mod.main([str(src), "--out", str(out), "--allow-unregistered"]) == 0
    assert f"UNREGISTERED input {sha}: not the registered rule" in capsys.readouterr().out
    wrote = json.loads(out.read_text())
    prov = wrote["provenance"]
    assert out.read_bytes() != shipped and prov["registered"] is False and prov["input_sha256"] == sha
    assert prov["copy"].startswith("UNREGISTERED") and "registered_sha256_prefix_of_original" not in prov
    assert "e564b52e92d3" not in out.read_text()
    assert {k: wrote[k] for k in mod.CONSTANT_KEYS} == {k: copy[k] for k in mod.CONSTANT_KEYS}
    assert mod.main([str(src), "--check", str(HELDOUT / "qc_rule_constants.json"), "--allow-unregistered"]) == 0
    said = capsys.readouterr().out
    assert "matches" in said and f"UNREGISTERED input {sha}" in said and "not that it is the registered file" in said
    assert mod.derive(full) == shipped                                # the registered path still gives the shipped bytes


def test_per_frame_flags_must_be_boolean_or_zero_one() -> None:
    """A plain cast reads NaN and the string 'False' as true. The regenerator refuses such flags instead."""
    import importlib.util

    import numpy as np
    import pytest as _pytest

    spec = importlib.util.spec_from_file_location(
        "regen_flags", Path(__file__).resolve().parents[2] / "reproduce/heldout/regenerate_unit_errors.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod._flags([True, False], "usable").tolist() == [True, False]
    assert mod._flags(np.array([1, 0, 1]), "gt_valid").tolist() == [True, False, True]
    for bad in ([1.0, float("nan")], ["True", "False"], [0, 2], [0.0, 1.0]):
        with _pytest.raises(ValueError, match="Boolean or 0/1"):
            mod._flags(bad, "usable")
    frames, xyz = np.array([0, 3, 6]), np.zeros((3, 3))
    with _pytest.raises(ValueError, match="'usable' must be Boolean"):
        mod.complete_centres([0, 3, 6], frames, xyz, np.array([1.0, np.nan, 1.0]))

