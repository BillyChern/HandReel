# Copyright 2026 The alongray authors
# SPDX-License-Identifier: Apache-2.0
"""Known answers for alongray-coverage: every synthetic case reads as its stated verdict, with exact counts."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from alongray.coverage import EXIT, compare_folders, main
from alongray.coverage_fixtures import CASES, EXPECTED, EXPECTED_INFO, write_known_answers


@pytest.fixture(scope="module")
def ka(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("ka")
    write_known_answers(root)
    return root


def run(ka: Path, case: str, ref: bool = False, **kw: object) -> dict:
    d = ka / case
    return compare_folders(d / "protocol.json", d / "a", d / "b", d / "ref" if ref else None, **kw)


def codes(rep: dict) -> set[str]:
    return {f.split(":")[0] for f in rep["flags"]}


def info_codes(rep: dict) -> set[str]:
    return {f.split(":")[0] for f in rep["info"]}


@pytest.mark.parametrize("ref", [False, True])
@pytest.mark.parametrize("case", CASES)
def test_case_verdict_and_flags(ka: Path, case: str, ref: bool) -> None:
    """The flag set is EXACT (no subset check), with and without the reference."""
    rep = run(ka, case, ref=ref)
    verdict, flags = EXPECTED[case]
    assert rep["verdict"] == verdict, rep["flags"] + rep["reasons"]
    assert codes(rep) == set(flags), rep["flags"]
    assert info_codes(rep) == set(EXPECTED_INFO.get(case, [])), rep["info"]
    if verdict == "COMPARABLE":
        assert rep["flags"] == []
    assert rep["speed"]["status"].startswith("NOT_INFERRED")


def test_expected_json_matches_the_table(ka: Path) -> None:
    for case in CASES:
        e = json.loads((ka / case / "expected.json").read_text())
        assert (e["verdict"], e["flags"], e["info"]) == (*EXPECTED[case], EXPECTED_INFO.get(case, []))


def flow(rep: dict, m: str) -> dict:
    return {s["stage"]: s for s in rep["flow"][m]}


def test_identical_counts_and_checks(ka: Path) -> None:
    rep = run(ka, "identical", ref=True)
    f = flow(rep, "b")
    assert [(s, f[s]["units"], f[s]["frames"]) for s in f] == [
        ("assigned", 3, 18), ("annotated", 3, 17), ("delivered", 3, 17), ("declared_source_identity", 3, 17),
        ("on_grid", 3, 18), ("delivered_annotated", 3, 17)]
    c = rep["common_ids"]
    assert (c["units"], c["frames"], c["sequences"], c["clusters"], c["a_only_frames"], c["b_only_frames"]) == \
        (3, 17, 2, 2, 0, 0)
    chk = rep["checks"]["a~b"]
    assert chk["median_scale_ratio"] == pytest.approx(1.0, abs=1e-12)
    assert chk["median_relative_discrepancy"] == pytest.approx(0.0, abs=1e-12)
    assert list(rep["checks"]) == ["a~b"]
    assert {k: v["result"] for k, v in rep["reference_checks"].items()} == {"ref~a": "PASS", "ref~b": "PASS"}
    assert all(v == "VERIFIED" for k, v in rep["matching"].items() if k != "symmetry")


def test_origin_shift_recovers_the_offset(ka: Path) -> None:
    o = run(ka, "origin_shift")["checks"]["a~b"]["origin_offset_mm"]["box"]
    assert o["offset_mm"] == pytest.approx([30.0, 0.0, 0.0], abs=1e-9)
    assert o["explained"] == pytest.approx(1.0, abs=1e-12)
    assert run(ka, "identical")["checks"]["a~b"]["origin_offset_mm"]["box"]["norm_mm"] == pytest.approx(0, abs=1e-12)


def test_unit_swap_and_its_declared_control(ka: Path) -> None:
    rep = run(ka, "unit_swap", ref=True)
    assert rep["checks"]["a~b"]["median_scale_ratio"] == pytest.approx(1000.0, rel=1e-9)
    rc = rep["reference_checks"]
    assert (rc["ref~b"]["result"], rc["ref~b"]["flags"]) == ("FLAGGED", ["UNIT_OR_SCALE_SUSPECT"])
    assert (rc["ref~a"]["result"], rc["ref~a"]["flags"]) == ("PASS", [])
    ok = run(ka, "unit_declared")
    assert ok["checks"]["a~b"]["median_scale_ratio"] == pytest.approx(1.0, abs=1e-12)
    assert ok["conventions"]["length_unit"]["status"] == "CONVERTED"


def test_missing_frames_flow(ka: Path) -> None:
    rep = run(ka, "missing_frames")
    f = flow(rep, "b")
    assert f["delivered"]["units"] == 2 and f["delivered"]["dropped_units"] == {"failed: qc_rejected": 1}
    assert (f["delivered_annotated"]["units"], f["delivered_annotated"]["frames"]) == (2, 11)
    c = rep["common_ids"]
    assert (c["units"], c["frames"], c["a_only_frames"], c["b_only_frames"]) == (2, 11, 6, 0)
    assert rep["delivery_share_of_annotated_frames"] == pytest.approx({"a": 1.0, "b": 11 / 17})
    assert "SCORES_NOT_ON_COMMON_FRAMES: a: 1 common units" in " ".join(rep["flags"])
    assert not any(f.startswith("SCORES_NOT_ON_COMMON_FRAMES: b") for f in rep["flags"])


def test_aggregation_both_weightings(ka: Path) -> None:
    rep = run(ka, "aggregation", n_boot=200)
    row = rep["aggregates"]["err_mm"]["a"]
    assert row["frame_weighted"] == pytest.approx((6 * 10 + 3 * 40 + 8 * 50) / 17, rel=1e-12)
    assert row["sequence_weighted"] == pytest.approx(((6 * 10 + 3 * 40) / 9 + 50) / 2, rel=1e-12)
    assert (row["declared_aggregation"], row["protocol_aggregation"], row["cluster"], row["n_clusters"]) == \
        ("frame", "sequence", "sequence", 2)
    lo, hi = row["ci95_cluster_bootstrap"]["sequence_weighted"]
    assert 20.0 - 1e-9 <= lo <= hi <= 50.0 + 1e-9
    assert rep["conventions"]["aggregation"]["status"] == "MISMATCH"


def test_symmetric_object(ka: Path) -> None:
    rep = run(ka, "symmetric")
    chk = rep["checks"]["a~b"]
    assert chk["median_rotation_deg"] == pytest.approx(180.0, abs=1e-6)
    assert chk["median_rotation_deg_symmetry_reduced"] == pytest.approx(0.0, abs=1e-5)
    assert rep["conventions"]["symmetry"]["status"] == "MISMATCH" and rep["matching"]["symmetry"] == "MISMATCH"
    assert run(ka, "identical")["conventions"]["symmetry"]["status"] == "NOT_APPLICABLE"


@pytest.mark.parametrize(("case", "text"), [("modality", "input modality 'rgbd'"), ("provenance", "split 'train'"),
                                            ("duplicate_keys_refused", "overlapping windows")])
def test_refusals_say_why(ka: Path, case: str, text: str) -> None:
    rep = run(ka, case)
    assert rep["verdict"] == "REFUSED" and any(text in r for r in rep["reasons"]), rep["reasons"]
    assert "common_ids" not in rep


def test_duplicates_kept_with_multiplicity(ka: Path) -> None:
    k = run(ka, "duplicate_keys_kept")["protocol_keys"]
    assert (k["keyed_frames"], k["distinct_keys"], k["duplicated_keys"], k["extra_copies"]) == (21, 19, 2, 2)
    assert k["distinct_physical_frames"] == 19
    assert run(ka, "duplicate_keys_kept")["common_ids"]["distinct_physical_frames"] == 18


def test_reported_only_is_not_pairable(ka: Path) -> None:
    rep = run(ka, "reported_only")
    assert rep["reasons"][0].startswith("reported-only, not pairable")
    o = rep["reported_cohort_overlap"]
    assert (o["units"], o["frames"], o["sequences"], o["cohort_frames_declared"]) == (3, 8, 2, 9)
    assert "common_ids" not in rep and "aggregates" not in rep


def test_counts_only(ka: Path) -> None:
    c = run(ka, "counts_only")["counts"]
    assert c["derived"] == {"a_only": 2, "b_only": 3, "neither": 1}
    assert all(c["checks"].values())


def test_counts_only_inconsistent_is_an_input_error(ka: Path, tmp_path: Path) -> None:
    d = tmp_path / "bad"
    subprocess.run(["cp", "-r", str(ka / "counts_only"), str(d)], check=True)
    p = json.loads((d / "protocol.json").read_text())
    p["published_counts"]["both"] = 12
    (d / "protocol.json").write_text(json.dumps(p))
    assert main([str(d / "protocol.json"), str(d / "a"), str(d / "b")]) == 2


def test_timing_fields_are_listed_and_never_used(ka: Path) -> None:
    rep = run(ka, "timing")
    assert set(rep["speed"]["ignored_timing_fields"]["a"]) >= {"wall_seconds", "units.S0:w0.gpu_seconds"}
    assert rep["speed"]["ignored_timing_fields"]["b"] == ["fps"]
    text = json.dumps({k: v for k, v in rep.items() if k != "speed"}).lower()
    assert "speed" not in text and "speedup" not in text


def test_camera_world_round_trip(ka: Path) -> None:
    rep = run(ka, "camera_world")
    assert rep["checks"]["a~b"]["median_relative_discrepancy"] == pytest.approx(0.0, abs=1e-12)
    assert rep["conventions"]["frame"]["status"] == "CONVERTED"


def test_frame_offset(ka: Path) -> None:
    rep = run(ka, "frame_base_undeclared")
    assert any("after shifting by -1" in f for f in rep["flags"])
    assert rep["matching"]["source_frame"] == "MISMATCH"
    assert run(ka, "frame_base_declared")["common_ids"]["frames"] == 17


def test_id_collision_excludes_the_wrong_source(ka: Path) -> None:
    rep = run(ka, "id_collision")
    f = flow(rep, "b")["declared_source_identity"]
    assert f["units"] == 2 and sum(f["dropped_units"].values()) == 2
    assert rep["common_ids"]["units"] == 2 and rep["common_ids"]["cameras"] == 1


def test_vanishing_failure_is_counted(ka: Path) -> None:
    rep = run(ka, "vanishing_failure")
    d = flow(rep, "b")["delivered"]
    assert d["units"] == 2 and d["dropped_units"] == {"absent from the manifest (no status)": 1}
    assert flow(rep, "b")["assigned"]["units"] == 3


def test_ids_only_on_request(ka: Path) -> None:
    assert "ids" not in run(ka, "identical")["common_ids"]
    ids = run(ka, "identical", include_ids=True)["common_ids"]["ids"]
    assert ids["S0:w1"] == [20, 22, 26]


def test_unit_file_outside_the_folder_is_refused(ka: Path, tmp_path: Path) -> None:
    d = tmp_path / "esc"
    subprocess.run(["cp", "-r", str(ka / "identical"), str(d)], check=True)
    m = json.loads((d / "b" / "manifest.json").read_text())
    m["units"]["S0:w0"]["file"] = "../a/S0_w0.npz"
    (d / "b" / "manifest.json").write_text(json.dumps(m))
    assert main([str(d / "protocol.json"), str(d / "a"), str(d / "b")]) == 2


def test_duplicate_frame_ids_in_a_file_are_refused(ka: Path, tmp_path: Path) -> None:
    d = tmp_path / "dup"
    subprocess.run(["cp", "-r", str(ka / "identical"), str(d)], check=True)
    z = dict(np.load(d / "b" / "S0_w0.npz"))
    z = {k: np.concatenate([v, v[:1]]) for k, v in z.items()}
    np.savez(d / "b" / "S0_w0.npz", **z)
    assert main([str(d / "protocol.json"), str(d / "a"), str(d / "b")]) == 2


def test_cli_exit_codes_and_summary(ka: Path, tmp_path: Path) -> None:
    for case, (verdict, _) in EXPECTED.items():
        d = ka / case
        assert main([str(d / "protocol.json"), str(d / "a"), str(d / "b"), "--out", str(tmp_path / "r.json")]) \
            == EXIT[verdict], case
    out = subprocess.run([sys.executable, "-m", "alongray.coverage", str(ka / "missing_frames" / "protocol.json"),
                          str(ka / "missing_frames" / "a"), str(ka / "missing_frames" / "b"), "--summary"],
                         capture_output=True, text=True, check=False,
                         env={"PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"), "PATH": "/usr/bin"})
    assert out.returncode == 1, out.stderr
    assert "VERDICT COMPARABLE_WITH_FLAGS" in out.stdout and "failed: qc_rejected 1" in out.stdout


def test_write_known_answers_cli(tmp_path: Path) -> None:
    assert main(["--write-known-answers", str(tmp_path / "k")]) == 0
    assert sorted(p.name for p in (tmp_path / "k").iterdir()) == sorted(CASES)


def test_declared_opengl_is_converted_and_checked(ka: Path) -> None:
    chk = run(ka, "camera_gl_declared")["checks"]["a~b"]
    assert chk["median_relative_discrepancy"] == pytest.approx(0.0, abs=1e-12)
    assert run(ka, "camera_gl_declared")["conventions"]["camera_axes"]["status"] == "CONVERTED"


def test_undeclared_provenance_is_refused(ka: Path, tmp_path: Path) -> None:
    d = tmp_path / "prov"
    subprocess.run(["cp", "-r", str(ka / "identical"), str(d)], check=True)
    m = json.loads((d / "b" / "manifest.json").read_text())
    del m["provenance"]["dataset"]
    (d / "b" / "manifest.json").write_text(json.dumps(m))
    rep = compare_folders(d / "protocol.json", d / "a", d / "b")
    assert rep["verdict"] == "REFUSED" and "b: provenance.dataset undeclared" in rep["reasons"]


@pytest.mark.parametrize(("name_a", "name_b", "text"), [
    ("same", "same", "method name 'same' is given by 2 inputs"),
    ("a", "status", "method name 'status' is a column of the report"),
    ("protocol", "b", "method name 'protocol' is a column of the report")])
def test_ambiguous_method_names_are_refused(ka: Path, tmp_path: Path, name_a: str, name_b: str, text: str) -> None:
    """Two methods under one name, or a name that equals a column of the convention table, would overwrite each
    other's inputs and counts in the report: refused before any comparison, with no common-ID numbers."""
    d = tmp_path / "names"
    subprocess.run(["cp", "-r", str(ka / "identical"), str(d)], check=True)
    for folder, name in (("a", name_a), ("b", name_b)):
        m = json.loads((d / folder / "manifest.json").read_text())
        m["method"] = name
        (d / folder / "manifest.json").write_text(json.dumps(m))
    rep = compare_folders(d / "protocol.json", d / "a", d / "b")
    assert rep["verdict"] == "REFUSED" and any(text in r for r in rep["reasons"]), rep["reasons"]
    assert "common_ids" not in rep
    # nothing keyed by method name exists in a name refusal, so no method's metadata can overwrite another's
    assert set(rep) == {"protocol", "methods", "cluster", "verdict", "reasons", "flags", "info"}
    assert rep["methods"] == [name_a, name_b] and rep["flags"] == []


def test_manifest_with_escaping_path_is_refused_on_load(ka: Path, tmp_path: Path) -> None:
    from alongray.coverage_io import load_method
    d = tmp_path / "esc2"
    subprocess.run(["cp", "-r", str(ka / "identical" / "b"), str(d)], check=True)
    m = json.loads((d / "manifest.json").read_text())
    m["units"]["S0:w0"]["file"] = "/etc/passwd"
    (d / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(ValueError, match="relative path inside the folder"):
        load_method(d)


def test_delivered_unit_with_no_frames_is_counted(ka: Path, tmp_path: Path) -> None:
    d = tmp_path / "empty"
    subprocess.run(["cp", "-r", str(ka / "identical"), str(d)], check=True)
    np.savez(d / "b" / "S0_w1.npz", frame_ids=np.zeros(0, np.int64), object_position=np.zeros((0, 3)),
             object_rotation=np.zeros((0, 3, 3)), timestamps=np.zeros(0))
    rep = compare_folders(d / "protocol.json", d / "a", d / "b")
    assert flow(rep, "b")["on_grid"]["dropped_units"] == {"delivered, no finite frame on the protocol grid": 1}
    assert rep["common_ids"]["units"] == 2


def test_score_aggregates_of_both_methods_rest_on_the_same_units() -> None:
    """Two common units, one method finite only on the first and the other only on the second: neither aggregate is
    printed for that metric, because no unit has a finite value from both. With a third unit finite for both, each
    aggregate rests on that unit alone and the two excluded units are counted, listed and flagged."""
    from pathlib import Path as _P

    from alongray.coverage_io import Method, Protocol, UnitSpec
    from alongray.coverage_report import aggregate_scores

    def unit(i: int) -> UnitSpec:
        return UnitSpec(f"u{i}", f"s{i}", "c", "o", (0, 1), frozenset({0, 1}))

    def method(name: str, vals: list[float | None]) -> Method:
        scores = {f"u{i}": {"n_frames": 2, "err_mm": v, "other": 1.0} for i, v in enumerate(vals)}
        return Method(name, _P("."), "x", {}, {}, {}, {"aggregation": "frame"}, {}, scores, {}, {}, ())

    def proto(n: int) -> Protocol:
        return Protocol("p", {}, {}, {"aggregation": "frame"}, "sequence", None, "x", {},
                        tuple(unit(i) for i in range(n)), None)

    common2 = {"u0": [0, 1], "u1": [0, 1]}
    agg, flags = aggregate_scores(proto(2), method("a", [1.0, None]), method("b", [float("nan"), 2.0]), common2)
    assert set(agg) == {"err_mm", "other"}                # no estimate, and the excluded units are still named
    for name in ("a", "b"):
        row = agg["err_mm"][name]
        assert (row["frame_weighted"], row["sequence_weighted"], row["n_units"]) == (None, None, 0)
        assert row["excluded_units"] == ["u0", "u1"] and "no common unit" in row["unavailable"]
    assert any(f.startswith("SCORES_NOT_FINITE_FOR_BOTH: err_mm: 2 common units") for f in flags)
    assert agg["other"]["a"]["n_excluded_not_finite_in_both"] == 0 and agg["other"]["a"]["excluded_units"] == []
    common3 = {**common2, "u2": [0, 1]}
    agg, flags = aggregate_scores(proto(3), method("a", [1.0, None, 5.0]), method("b", [float("inf"), 2.0, 7.0]),
                                  common3)
    ra, rb = agg["err_mm"]["a"], agg["err_mm"]["b"]
    assert (ra["n_units"], rb["n_units"]) == (1, 1) and ra["excluded_units"] == rb["excluded_units"] == ["u0", "u1"]
    assert (ra["frame_weighted"], rb["frame_weighted"]) == (pytest.approx(5.0), pytest.approx(7.0))   # u2 for both
    assert ra["n_excluded_not_finite_in_both"] == 2
    assert sum(f.startswith("SCORES_NOT_FINITE_FOR_BOTH: err_mm: 2") for f in flags) == 1
    agg, _ = aggregate_scores(proto(3), method("a", [1.0, True, 5.0]), method("b", [3.0, 2.0, 7.0]), common3)
    assert agg["err_mm"]["a"]["excluded_units"] == ["u1"]                 # a boolean is not a score


def test_a_metric_key_missing_from_one_row_does_not_drop_the_metric_and_zero_frames_carry_no_weight() -> None:
    """u1 supplies err_mm in both methods and u2 omits the key in one: the metric is aggregated on u1 and u2 is
    excluded and named. A unit with zero frames is excluded from both methods and never divides by zero."""
    from pathlib import Path as _P

    from alongray.coverage_io import Method, Protocol, UnitSpec
    from alongray.coverage_report import aggregate_scores

    def proto(n: int) -> Protocol:
        return Protocol("p", {}, {}, {"aggregation": "frame"}, "sequence", None, "x", {},
                        tuple(UnitSpec(f"u{i}", f"s{i}", "c", "o", (0, 1), frozenset({0, 1})) for i in range(n)), None)

    def method(name: str, scores: dict) -> Method:
        return Method(name, _P("."), "x", {}, {}, {}, {"aggregation": "frame"}, {}, scores, {}, {}, ())

    common = {"u0": [0, 1], "u1": [0, 1]}
    a = method("a", {"u0": {"n_frames": 2, "err_mm": 4.0}, "u1": {"n_frames": 2, "err_mm": 9.0}})
    b = method("b", {"u0": {"n_frames": 2, "err_mm": 6.0}, "u1": {"n_frames": 2}})          # u1 omits the key
    agg, flags = aggregate_scores(proto(2), a, b, common)
    assert (agg["err_mm"]["a"]["frame_weighted"], agg["err_mm"]["b"]["frame_weighted"]) == (4.0, 6.0)
    assert agg["err_mm"]["a"]["excluded_units"] == agg["err_mm"]["b"]["excluded_units"] == ["u1"]
    assert any(f.startswith("SCORES_NOT_FINITE_FOR_BOTH: err_mm: 1 common units") for f in flags)
    z = method("b", {"u0": {"n_frames": 0, "err_mm": 6.0}, "u1": {"n_frames": 2, "err_mm": 8.0}})   # zero frames
    agg, _ = aggregate_scores(proto(2), a, z, common)
    ra, rb = agg["err_mm"]["a"], agg["err_mm"]["b"]
    assert (ra["frame_weighted"], rb["frame_weighted"], ra["excluded_units"]) == (9.0, 8.0, ["u0"])
    assert np.isfinite([ra["frame_weighted"], ra["sequence_weighted"], rb["sequence_weighted"]]).all()   # no NaN
    both_zero = method("b", {"u0": {"n_frames": 0, "err_mm": 6.0}, "u1": {"n_frames": 0, "err_mm": 8.0}})
    agg, _ = aggregate_scores(proto(2), a, both_zero, common)
    assert agg["err_mm"]["a"]["frame_weighted"] is None and agg["err_mm"]["a"]["excluded_units"] == ["u0", "u1"]


def test_a_common_unit_scored_by_one_method_only_is_excluded_by_name() -> None:
    """A supplies rows for u0 and u1, B for u0 only: the paired estimate on u0 stands, u1 is named as excluded, and
    a metric that only A's u1 row carries is reported as unavailable instead of vanishing."""
    from pathlib import Path as _P

    from alongray.coverage_io import Method, Protocol, UnitSpec
    from alongray.coverage_report import aggregate_scores

    proto = Protocol("p", {}, {}, {"aggregation": "frame"}, "sequence", None, "x", {},
                     tuple(UnitSpec(f"u{i}", f"s{i}", "c", "o", (0, 1), frozenset({0, 1})) for i in range(2)), None)

    def method(name: str, scores: dict) -> Method:
        return Method(name, _P("."), "x", {}, {}, {}, {"aggregation": "frame"}, {}, scores, {}, {}, ())

    a = method("a", {"u0": {"n_frames": 2, "err_mm": 4.0}, "u1": {"n_frames": 2, "err_mm": 9.0, "only_a": 1.0}})
    b = method("b", {"u0": {"n_frames": 2, "err_mm": 6.0}})
    agg, flags = aggregate_scores(proto, a, b, {"u0": [0, 1], "u1": [0, 1]})
    assert (agg["err_mm"]["a"]["frame_weighted"], agg["err_mm"]["b"]["frame_weighted"]) == (4.0, 6.0)
    assert agg["err_mm"]["a"]["excluded_units"] == agg["err_mm"]["b"]["excluded_units"] == ["u1"]
    assert agg["only_a"]["a"]["frame_weighted"] is None and agg["only_a"]["b"]["excluded_units"] == ["u0", "u1"]
    assert any(f.startswith("SCORES_MISSING: b: 1 common units") for f in flags)
    assert any(f.startswith("SCORES_NOT_FINITE_FOR_BOTH: err_mm: 1 common units lack a score row") for f in flags)
    none, _ = aggregate_scores(proto, a, method("b", {}), {"u0": [0, 1], "u1": [0, 1]})   # B supplies no scores
    assert none == {}
