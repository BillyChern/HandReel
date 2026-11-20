# Copyright 2026 The alongray authors
# SPDX-License-Identifier: Apache-2.0
"""Adversarial cases for alongray-coverage (D1-D10), as known answers."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from alongray import coverage
from alongray.coverage import compare_folders, main
from alongray.coverage_fixtures import misread_quaternion_order, truth, write_known_answers

README = Path(__file__).resolve().parents[1] / "README.md"


@pytest.fixture(scope="module")
def ka(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("ka")
    write_known_answers(root)
    return root


def clone(ka: Path, case: str, dst: Path) -> Path:
    shutil.copytree(ka / case, dst)
    return dst


def jload(p: Path) -> dict:
    return json.loads(p.read_text())


def jsave(p: Path, d: object) -> None:
    p.write_text(json.dumps(d, indent=1))


def edit_npz(folder: Path, fn: Callable[[dict], None]) -> None:
    for f in sorted(folder.glob("*.npz")):
        z = {k: np.asarray(v) for k, v in np.load(f).items()}
        fn(z)
        np.savez(f, **z)


def run(d: Path, ref: bool = False, **kw: object) -> dict:
    return compare_folders(d / "protocol.json", d / "a", d / "b", d / "ref" if ref else None, **kw)


def codes(rep: dict, key: str = "flags") -> set[str]:
    return {f.split(":")[0] for f in rep[key]}


def big(ka: Path, d: Path, n_units: int = 20, nfr: int = 10) -> Path:
    """A stride-1 protocol of ``n_units`` units of ``nfr`` frames; a, b and ref deliver the truth everywhere."""
    base = jload(ka / "identical" / "protocol.json")
    base["units"] = [{"unit_id": f"S{i}:w0", "sequence": f"S{i}", "camera": "c0", "object": "box",
                      "frames": list(range(nfr)), "scorable": list(range(nfr))} for i in range(n_units)]
    d.mkdir(parents=True)
    jsave(d / "protocol.json", base)
    for m in ("a", "b", "ref"):
        man = jload(ka / "identical" / m / "manifest.json")
        man["units"], man["scores"] = {}, {}
        (d / m).mkdir()
        for i, u in enumerate(base["units"]):
            pp, rr = truth(i % 3, u["frames"])
            fn = f"S{i}_w0.npz"
            np.savez(d / m / fn, frame_ids=np.array(u["frames"]), object_position=pp, object_rotation=rr,
                     timestamps=np.array(u["frames"]) / 30.0)
            man["units"][u["unit_id"]] = {"status": "delivered", "file": fn, "sequence": u["sequence"],
                                          "camera": "c0", "object": "box"}
        jsave(d / m / "manifest.json", man)
    return d


# D1: reference checks are pass/fail only, on the common IDs; screening statistics are labelled.

def test_reference_checks_are_pass_fail_only(ka: Path) -> None:
    rep = run(ka / "identical", ref=True)
    assert set(rep["checks"]) == {"a~b"}
    assert rep["checks"]["a~b"]["label"] == "screening statistic, not a score"
    for name, chk in rep["reference_checks"].items():
        assert set(chk) == {"result", "flags", "frames_checked", "label"}, name
        assert chk["result"] == "PASS" and chk["flags"] == []
        assert "not a per-method accuracy" in chk["label"]


def test_reference_checks_run_on_the_common_ids(ka: Path) -> None:
    rep = run(ka / "missing_frames", ref=True)
    assert rep["common_ids"]["frames"] == 11
    assert {k: v["frames_checked"] for k, v in rep["reference_checks"].items()} == {"ref~a": 11, "ref~b": 11}


def test_reference_flag_text_withholds_the_value(ka: Path, tmp_path: Path) -> None:
    d = clone(ka, "identical", tmp_path / "c")
    edit_npz(d / "b", lambda z: z.__setitem__("object_position", z["object_position"] * 3.0))
    rep = run(d, ref=True)
    ref_flags = [f for f in rep["flags"] if "ref~b" in f]
    assert ref_flags and all("withheld" in f for f in ref_flags), ref_flags
    assert not any(ch.isdigit() for f in ref_flags for ch in f.split("ref~b", 1)[1]), ref_flags


# D2: an undeclared 0/1-based shift on a stride-1 grid.

def test_stride1_one_based_ids_are_flagged(ka: Path) -> None:
    rep = run(ka / "frame_base_undeclared_stride1")
    assert any("after shifting by -1" in f for f in rep["flags"]), rep["flags"]
    assert rep["matching"]["source_frame"] == "MISMATCH"
    assert rep["common_ids"]["frames"] == 76


def test_stride1_twenty_frame_units(ka: Path, tmp_path: Path) -> None:
    d = big(ka, tmp_path / "dense", n_units=20, nfr=20)
    edit_npz(d / "b", lambda z: z.__setitem__("frame_ids", z["frame_ids"] + 1))
    assert "FRAME_ID_OFFSET_SUSPECT" in codes(run(d))


def test_correct_stride1_ids_are_not_flagged(ka: Path, tmp_path: Path) -> None:
    rep = run(big(ka, tmp_path / "dense", n_units=20, nfr=20), ref=True)
    assert rep["verdict"] == "COMPARABLE" and rep["flags"] == []


def test_reference_picks_the_shift_the_grid_cannot_see(ka: Path, tmp_path: Path) -> None:
    """b delivers truth frames 0..17 labelled 1..18: every id lies on the grid, so the grid screen is blind; the
    between-method screen and the reference screen are not."""
    d = big(ka, tmp_path / "interior", n_units=6, nfr=20)

    def interior(z: dict) -> None:
        for k in list(z):
            z[k] = z[k][:18]
        z["frame_ids"] = z["frame_ids"] + 1
        z["timestamps"] = z["frame_ids"] / 30.0
    edit_npz(d / "b", interior)
    plain = run(d)
    assert not any(f.startswith("FRAME_ID_OFFSET_SUSPECT: b:") for f in plain["flags"]), plain["flags"]
    assert any(f.startswith("FRAME_ID_OFFSET_SUSPECT: a~b") for f in plain["flags"]), plain["flags"]
    rep = run(d, ref=True)
    assert any(f.startswith("FRAME_ID_OFFSET_SUSPECT: ref~b") and "shift -1" in f for f in rep["flags"]), rep["flags"]
    assert rep["reference_checks"]["ref~a"]["result"] == "PASS"
    assert rep["matching"]["source_frame"] == "MISMATCH"


# D3: the origin check fires only when a constant OBJECT-frame offset clearly beats a constant camera-frame one.

@pytest.mark.parametrize(("name", "fn", "expected"), [
    ("scale_3pct", lambda z: z.__setitem__("object_position", z["object_position"] * 1.03), set()),
    ("scale_3x", lambda z: z.__setitem__("object_position", z["object_position"] * 3.0),
     {"FRAME_OR_TRANSFORM_SUSPECT"}),
    ("mirror_x", lambda z: z.__setitem__("object_position", z["object_position"] * np.array([-1.0, 1, 1])), set()),
    ("offset_20cm", lambda z: z.__setitem__("object_position", z["object_position"] + np.array([0.2, 0, 0])), set()),
])
def test_global_errors_are_not_origin_offsets(ka: Path, tmp_path: Path, name: str, fn: Callable,
                                              expected: set[str]) -> None:
    d = clone(ka, "identical", tmp_path / name)
    edit_npz(d / "b", fn)
    rep = run(d, ref=True)
    assert codes(rep) == expected, rep["flags"]


def test_three_percent_scale_is_not_caught(ka: Path, tmp_path: Path) -> None:
    """The documented limit (README): a 3% global scale reads COMPARABLE, ratio 1.03."""
    d = clone(ka, "identical", tmp_path / "s")
    edit_npz(d / "b", lambda z: z.__setitem__("object_position", z["object_position"] * 1.03))
    rep = run(d)
    assert rep["verdict"] == "COMPARABLE"
    assert rep["checks"]["a~b"]["median_scale_ratio"] == pytest.approx(1.03, rel=1e-12)


def test_origin_offset_without_rotation_spread_is_not_attributed(ka: Path, tmp_path: Path) -> None:
    """With one fixed rotation an object-frame offset is also a camera-frame offset: no origin flag."""
    d = clone(ka, "identical", tmp_path / "still")
    r0 = truth(0, [0])[1][0]

    def still(z: dict) -> None:
        z["object_rotation"] = np.broadcast_to(r0, z["object_rotation"].shape).copy()
    for m in ("a", "b"):
        edit_npz(d / m, still)
    edit_npz(d / "b", lambda z: z.__setitem__("object_position", z["object_position"]
                                              + z["object_rotation"] @ np.array([0.03, 0.0, 0.0])))
    assert "ORIGIN_OFFSET_SUSPECT" not in codes(run(d))


def test_origin_shift_still_reports_both_fits(ka: Path) -> None:
    o = run(ka / "origin_shift")["checks"]["a~b"]["origin_offset_mm"]["box"]
    assert o["object_vs_camera_residual"] == pytest.approx(0.0, abs=1e-12)


# D4: a rotation-convention screen.

def test_quaternion_order_is_flagged(ka: Path) -> None:
    rep = run(ka / "rotation_convention", ref=True)
    assert rep["checks"]["a~b"]["median_rotation_deg"] > 100.0
    assert codes(rep) == {"ROTATION_DISAGREEMENT"}
    assert rep["reference_checks"]["ref~b"]["flags"] == ["ROTATION_DISAGREEMENT"]
    assert rep["matching"]["object"] == "MISMATCH"


def test_rotation_screen_needs_rotations_declared_comparable(ka: Path, tmp_path: Path) -> None:
    d = clone(ka, "identical", tmp_path / "q")
    edit_npz(d / "b", lambda z: z.__setitem__("object_rotation", misread_quaternion_order(z["object_rotation"])))
    assert "ROTATION_DISAGREEMENT" not in codes(run(d))


def test_symmetric_object_is_not_a_rotation_disagreement(ka: Path) -> None:
    assert "ROTATION_DISAGREEMENT" not in codes(run(ka / "symmetric"))


# D5: malformed input exits 2, never 1.

def _no_scorable(d: Path) -> None:
    p = jload(d / "protocol.json")
    for u in p["units"]:
        u["scorable"] = []
    jsave(d / "protocol.json", p)


def _unit_entry_str(d: Path) -> None:
    m = jload(d / "b" / "manifest.json")
    m["units"]["S0:w0"] = "delivered"
    jsave(d / "b" / "manifest.json", m)


def _protocol_unit_list(d: Path) -> None:
    p = jload(d / "protocol.json")
    p["units"][0] = ["S0:w0"]
    jsave(d / "protocol.json", p)


def _bad_zip(d: Path) -> None:
    (d / "b" / "S0_w0.npz").write_bytes(b"PK\x03\x04" + b"\x00" * 64)


def _score_entry_str(d: Path) -> None:
    m = jload(d / "b" / "manifest.json")
    m["scores"]["S0:w0"] = 3.0
    jsave(d / "b" / "manifest.json", m)


@pytest.mark.parametrize(("case", "mutate", "text"), [
    ("identical", _no_scorable, "no annotated (scorable) frame"),
    ("identical", _unit_entry_str, "unit S0:w0 must be a JSON object"),
    ("identical", _protocol_unit_list, "protocol unit 0 must be a JSON object"),
    ("identical", _bad_zip, "not a readable .npz"),
    ("identical", _score_entry_str, "scores.S0:w0 must be a JSON object"),
])
def test_malformed_input_exits_2(ka: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], case: str,
                                 mutate: Callable[[Path], None], text: str) -> None:
    d = clone(ka, case, tmp_path / "m")
    mutate(d)
    assert main([str(d / "protocol.json"), str(d / "a"), str(d / "b")]) == 2
    assert text in capsys.readouterr().err


def test_counts_only_protocol_with_per_frame_methods_exits_2(ka: Path, tmp_path: Path,
                                                             capsys: pytest.CaptureFixture[str]) -> None:
    d = clone(ka, "counts_only", tmp_path / "c")
    for m in ("a", "b"):
        shutil.rmtree(d / m)
        shutil.copytree(ka / "identical" / m, d / m)
    assert main([str(d / "protocol.json"), str(d / "a"), str(d / "b")]) == 2
    assert "per-frame outputs need a protocol with units" in capsys.readouterr().err


def test_any_other_exception_exits_2(ka: Path, monkeypatch: pytest.MonkeyPatch,
                                     capsys: pytest.CaptureFixture[str]) -> None:
    def boom(*_: object, **__: object) -> dict:
        raise KeyError("unexpected")
    monkeypatch.setattr(coverage, "compare_folders", boom)
    d = ka / "identical"
    assert main([str(d / "protocol.json"), str(d / "a"), str(d / "b")]) == 2
    assert "internal error" in capsys.readouterr().err


# D6: "declared source identity", never "physical source matched".

def test_source_identity_is_called_declared(ka: Path) -> None:
    stages = [s["stage"] for s in run(ka / "identical")["flow"]["b"]]
    assert "declared_source_identity" in stages and "source_matched" not in stages
    text = README.read_text() + (coverage.__doc__ or "")
    assert "physical source matched" not in text and "declared source identity" in text


def test_a_copied_identity_is_not_detected(ka: Path, tmp_path: Path) -> None:
    """The documented limit: a file holding another sequence's trajectory under a copied identity passes."""
    d = clone(ka, "identical", tmp_path / "copy")
    z1, z0 = dict(np.load(d / "b" / "S1_w0.npz")), dict(np.load(d / "b" / "S0_w0.npz"))
    n = len(z0["frame_ids"])
    z0["object_position"], z0["object_rotation"] = z1["object_position"][:n], z1["object_rotation"][:n]
    np.savez(d / "b" / "S0_w0.npz", **z0)
    assert run(d)["verdict"] == "COMPARABLE"


# D7: DELIVERY_DIFFERS is informational.

def test_delivery_differs_on_self_exclusion(ka: Path, tmp_path: Path) -> None:
    d = big(ka, tmp_path / "excl")
    m = jload(d / "b" / "manifest.json")
    for uid in ("S7:w0", "S8:w0"):
        m["units"][uid] = {"status": "excluded", "reason": "low confidence (method's own filter)"}
    jsave(d / "b" / "manifest.json", m)
    rep = run(d)
    assert rep["verdict"] == "COMPARABLE" and rep["flags"] == []
    assert codes(rep, "info") == {"DELIVERY_DIFFERS"}
    assert any("2 excluded" in f for f in rep["info"]), rep["info"]


def test_delivery_differs_on_dropped_rows(ka: Path, tmp_path: Path) -> None:
    d = big(ka, tmp_path / "rows")
    edit_npz(d / "b", lambda z: [z.__setitem__(k, v[1:]) for k, v in list(z.items())])
    rep = run(d)
    assert codes(rep, "info") == {"DELIVERY_DIFFERS"}
    assert rep["delivery_share_of_annotated_frames"] == pytest.approx({"a": 1.0, "b": 0.9})


def test_no_delivery_info_when_delivery_is_equal(ka: Path) -> None:
    assert run(ka / "identical")["info"] == []


# D8: the reference's own coverage and flags are reported.

def test_reference_coverage_and_nonfinite_rows(ka: Path, tmp_path: Path) -> None:
    d = clone(ka, "identical", tmp_path / "nan")

    def nanify(z: dict) -> None:
        pos = z["object_position"].copy()
        pos[::2] = np.nan
        z["object_position"] = pos
    edit_npz(d / "ref", nanify)
    rep = run(d, ref=True)
    r = rep["reference"]
    assert r["name"] == "ref"
    last = r["flow"][-1]
    assert (last["stage"], last["nonfinite_frames"]) == ("delivered_annotated", 9)
    assert any(f.startswith("NONFINITE_ROWS: ref: 9") for f in rep["info"]), rep["info"]
    assert rep["verdict"] == "COMPARABLE"


def test_reference_axes_flag_is_reported(ka: Path, tmp_path: Path) -> None:
    d = clone(ka, "identical", tmp_path / "gl")
    edit_npz(d / "ref", lambda z: z.__setitem__("object_position", z["object_position"] * np.array([1.0, -1, -1])))
    rep = run(d, ref=True)
    assert "CAMERA_AXES_SUSPECT: ref: 3 units" in rep["flags"]
    assert "CAMERA_AXES_SUSPECT" in rep["reference"]["flags"][0]


# D9: source_frame VERIFIED needs a flow for both methods.

def test_reported_only_source_frame_is_unverified(ka: Path) -> None:
    assert run(ka / "reported_only")["matching"]["source_frame"] == "UNVERIFIED"


# D10: the README states the scale limit.

def test_readme_states_the_scale_limit() -> None:
    text = " ".join(README.read_text().split())
    assert "few-percent global scale errors are not caught" in text.lower()
    assert "1.0127" in text


def test_summary_shows_info_reference_flow_and_checks(ka: Path) -> None:
    from alongray.coverage_report import format_summary
    text = format_summary(run(ka / "missing_frames", ref=True))
    assert "INFO DELIVERY_DIFFERS: b: 1 failed of 3 assigned units" in text
    assert "reference ref: delivers 1.00 of the annotated frames" in text
    assert "reference check ref~a PASS on 11 common frames" in text
    assert "screening statistic, not a score" in text


# Further cases: disjoint references, unit flow and nits.

def _ref_disjoint(d: Path) -> None:
    edit_npz(d / "ref", lambda z: z.__setitem__("frame_ids", z["frame_ids"] + 1))  # off a stride-2 grid


def _ref_all_nan(d: Path) -> None:
    edit_npz(d / "ref", lambda z: z.__setitem__("object_position", np.full_like(z["object_position"], np.nan)))


def _ref_no_positions(d: Path) -> None:
    edit_npz(d / "ref", lambda z: z.pop("object_position"))


@pytest.mark.parametrize("mutate", [_ref_disjoint, _ref_all_nan, _ref_no_positions])
def test_reference_that_checks_nothing_is_not_a_pass(ka: Path, tmp_path: Path, mutate: Callable) -> None:
    from alongray.coverage_report import format_summary
    d = clone(ka, "identical", tmp_path / "r")
    mutate(d)
    rep = run(d, ref=True)
    for chk in rep["reference_checks"].values():
        assert (chk["result"], chk["frames_checked"]) == ("NOT_CHECKED", 0)
    assert "REFERENCE_NOT_CHECKED" in codes(rep) and rep["verdict"] == "COMPARABLE_WITH_FLAGS"
    text = format_summary(rep)
    assert "PASS on 0" not in text and "NOT_CHECKED on 0" in text


def _shift_drop_last(d: Path) -> None:
    shutil.rmtree(d / "b")
    shutil.copytree(d / "a", d / "b")
    m = jload(d / "b" / "manifest.json")
    m["method"] = "b"
    jsave(d / "b" / "manifest.json", m)

    def fn(z: dict) -> None:
        for k in list(z):
            z[k] = z[k][:-1]
        z["frame_ids"] = z["frame_ids"] + 1
    edit_npz(d / "b", fn)


def test_stride1_shift_with_the_last_frame_dropped_is_flagged_without_a_reference(ka: Path, tmp_path: Path) -> None:
    """Every shifted id lies on the grid, so the grid screen is blind; the between-method screen is not."""
    d = clone(ka, "frame_base_undeclared_stride1", tmp_path / "s")
    _shift_drop_last(d)
    rep = run(d)
    assert any(f.startswith("FRAME_ID_OFFSET_SUSPECT: a~b") for f in rep["flags"]), rep["flags"]
    assert rep["matching"]["source_frame"] == "MISMATCH"


def test_between_method_shift_screen_ignores_independent_noise(ka: Path, tmp_path: Path) -> None:
    d = big(ka, tmp_path / "noise", n_units=6, nfr=20)
    rng = np.random.default_rng(0)
    edit_npz(d / "b", lambda z: z.__setitem__("object_position", z["object_position"]
                                              + rng.normal(0.0, 0.005, z["object_position"].shape)))
    assert "FRAME_ID_OFFSET_SUSPECT" not in codes(run(d))


@pytest.mark.parametrize(("key", "compare"), [("object_position", ["object_position"]),
                                              ("object_rotation", ["object_position", "object_rotation"])])
def test_delivered_unit_missing_a_compared_array_is_flagged(ka: Path, tmp_path: Path, key: str,
                                                            compare: list[str]) -> None:
    d = clone(ka, "identical", tmp_path / "miss")
    p = jload(d / "protocol.json")
    p["modality"]["compare"] = compare
    jsave(d / "protocol.json", p)
    edit_npz(d / "b", lambda z: z.pop(key))
    rep = run(d)
    assert "COMPARED_ARRAY_MISSING: b: 3 units" in rep["flags"], rep["flags"]
    assert rep["verdict"] == "COMPARABLE_WITH_FLAGS"


def test_rotation_transpose_below_the_screen_reads_comparable(ka: Path, tmp_path: Path) -> None:
    """The documented limit: a transpose disagrees by twice the rotation angle; on these rotations about 53 deg."""
    d = clone(ka, "identical", tmp_path / "t")
    p = jload(d / "protocol.json")
    p["modality"]["compare"] = ["object_position", "object_rotation"]
    jsave(d / "protocol.json", p)
    edit_npz(d / "b", lambda z: z.__setitem__("object_rotation", np.swapaxes(z["object_rotation"], -1, -2)))
    rep = run(d)
    assert rep["verdict"] == "COMPARABLE"
    assert 45.0 < rep["checks"]["a~b"]["median_rotation_deg"] < 60.0


def test_readme_gives_the_thresholds() -> None:
    text = " ".join(README.read_text().split())
    assert "`DISCREPANCY_TOL` = 0.5" in text and "20 cm offset at 0.5 m reads COMPARABLE" in text
    assert "60 deg" in text and "about 38 deg" in text and "about 53 deg" in text


def test_string_in_optional_published_count_exits_2(ka: Path, tmp_path: Path,
                                                    capsys: pytest.CaptureFixture[str]) -> None:
    d = clone(ka, "counts_only", tmp_path / "c")
    p = jload(d / "protocol.json")
    p["published_counts"]["removed"] = "1"
    jsave(d / "protocol.json", p)
    assert main([str(d / "protocol.json"), str(d / "a"), str(d / "b")]) == 2
    assert "published_counts.removed must be a non-negative integer" in capsys.readouterr().err


def test_unreadable_npz_leaves_no_open_file(ka: Path, tmp_path: Path) -> None:
    import gc
    import warnings
    d = clone(ka, "identical", tmp_path / "z")
    (d / "b" / "S0_w0.npz").write_bytes(b"PK\x03\x04" + b"\x00" * 64)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ResourceWarning)
        with pytest.raises(ValueError, match="not a readable .npz"):
            run(d)
        gc.collect()
    assert not [x for x in w if issubclass(x.category, ResourceWarning)], [str(x.message) for x in w]
