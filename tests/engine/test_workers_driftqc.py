"""driftqc stage: constants, detector loader, v1 TRUNC range rule and the stage worker.

The worker tests run on the 24-row synthetic clip of
tests/engine/fakes_driftqc.py. The frozen files are read under FROZEN_ROOT (env FASTDO_DRIFTQC_ROOT,
default the repo root); a test that needs them skips only when they are absent.
Unit-case inputs come from constants_v2_2.json `unit_cases_v1` / `unit_cases_labels` at test time; numeric m values are
derived from the constants (flagged rows +-2 tau_abs with both signs, unflagged finite rows 0.0, NaN rows NaN, row 0
NaN as sg_stat returns at the keyframe).
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from fastdo.engine import workers_driftqc as wd
from fastdo.engine.manifest import STAGE_NAMES, ClipManifest
from fastdo.engine.orchestrator import STAGE_GPU, EngineConfig, StageFn, run_clip
from fastdo.engine.valid_range import (
    DRIFTQC_FILE,
    VALID_RANGE_FILE,
    VALID_RANGE_KEYS,
    RowRange,
    frame_idx_sha256,
    resolve_valid_range,
    slice_artifact,
)
from fastdo.engine.worker_support import load_npz, read_frame_idx, save_npz
from fastdo.engine.workers_driftqc import (
    DRIFT_REJECTED_PREFIX,
    FROZEN_CONSTANTS_SHA256,
    FROZEN_DETECTOR_SHA256,
    FROZEN_TRACKWEIGHT_CONSTANTS_SHA256,
    RECORD_LABEL,
    DriftDecision,
    DriftQcConfigError,
    DriftQcConstants,
    DriftQcStageConfig,
    DriftRejected,
    decide_range,
    driftqc_range,
    load_driftqc_constants,
    load_frozen_detector,
    make_driftqc_worker,
    with_driftqc,
)
from tests.engine.fakes_driftqc import N_ROWS, driftqc_clip, write_driftqc_inputs

FROZEN_ROOT = Path(os.environ.get("FASTDO_DRIFTQC_ROOT", "/opt/fastdo/handreel"))
FROZEN_ABSENT = ("frozen driftqc files absent (not shipped): set FASTDO_DRIFTQC_ROOT to a directory holding "
                 "temp/driftqc/constants_v2_2.json, temp/trackweight/constants_v2.json and "
                 "temp/trackweight/code/v1_weights.py")
CONSTANTS = FROZEN_ROOT / "temp/driftqc/constants_v2_2.json"
TRACKWEIGHT_CONSTANTS = FROZEN_ROOT / "temp/trackweight/constants_v2.json"
DETECTOR = FROZEN_ROOT / "temp/trackweight/code/v1_weights.py"

requires_frozen = pytest.mark.skipif(
    not (CONSTANTS.is_file() and TRACKWEIGHT_CONSTANTS.is_file() and DETECTOR.is_file()),
    reason=FROZEN_ABSENT,
)

_Expect = tuple[str, tuple[int, int] | None, int | None, int | None, str | None]  # decision, kept, onset, cut_b, reason

# Section 6 E2 table. Inputs (flags, NaN rows) are asserted equal to the constants entries.
UNIT_TABLE: dict[str, tuple[list[int], list[int], _Expect]] = {
    "U1": ([], [], ("ACCEPT", (0, 23), None, None, None)),
    "U2": (list(range(9, 24)), [], ("TRUNCATE", (0, 7), 9, 7, None)),
    "U3": ([9, 11], [10], ("TRUNCATE", (0, 7), 9, 7, None)),
    "U4": ([7, 8], [], ("TRUNCATE", (0, 5), 7, 5, None)),
    "U5": ([6, 7], [], ("REJECT", None, 6, 4, "short")),
    "U6": ([], list(range(10, 24)), ("ACCEPT", (0, 23), None, None, None)),
    "U7": ([9], [], ("ACCEPT", (0, 23), None, None, None)),
    "U8": ([23], [], ("ACCEPT", (0, 23), None, None, None)),
    "U9": ([22, 23], [], ("TRUNCATE", (0, 20), 22, 20, None)),
    "U10": ([1, 2], [], ("REJECT", None, 1, -1, "short")),
    "U11": ([9, 16], list(range(10, 16)), ("TRUNCATE", (0, 7), 9, 7, None)),
    "U12": ([], [], ("ACCEPT", (0, 23), None, None, None)),  # |m| = tau_abs exactly at 9, 10 (both signs)
}
LABEL_TABLE: dict[str, tuple[int | None, _Expect]] = {
    "UL1": (6, ("REJECT", None, 6, 4, "short")),
    "UL2": (1, ("REJECT", None, 1, -1, "short")),
    "UL3": (9, ("TRUNCATE", (0, 7), 9, 7, None)),
    "UL4": (None, ("ACCEPT", (0, 23), None, None, None)),
}


def _rows(spec: Any) -> list[int]:
    """A constants row list: a JSON list, or an inclusive "a-b" string."""
    if isinstance(spec, str):
        lo, hi = (int(v) for v in spec.split("-"))
        return list(range(lo, hi + 1))
    return [int(v) for v in spec]


def _raw_constants() -> dict[str, Any]:
    return json.loads(CONSTANTS.read_bytes())


def _case(section: str, case_id: str) -> dict[str, Any]:
    matches = [c for c in _raw_constants()[section] if c["id"] == case_id]
    assert len(matches) == 1, case_id
    return matches[0]


def _m_from_pattern(n: int, flags: list[int], nan: list[int], tau: float) -> np.ndarray:
    m = np.zeros(n, dtype=np.float64)
    m[0] = np.nan
    for i, t in enumerate(sorted(flags)):
        m[t] = 2.0 * tau * (1.0 if i % 2 == 0 else -1.0)
    m[nan] = np.nan
    return m


def _assert_decision(dec: DriftDecision, n: int, expect: _Expect) -> None:
    decision, kept, onset, cut_b, reason = expect
    assert dec.n_rows == n
    assert dec.decision == decision
    assert dec.kept == kept
    assert dec.onset == onset
    assert dec.cut_b == cut_b
    assert dec.reason == reason
    assert dec.end_f == (n if onset is None else onset)


def _assert_expect_text(text: str, expect: _Expect) -> None:
    """The constants' `expect` wording names the same decision, kept range and reason as the table."""
    decision, kept, _, _, reason = expect
    assert text.replace(";", " ").split()[0] == decision, text
    if kept is not None and decision == "TRUNCATE":
        assert f"[{kept[0]},{kept[1]}]" in text, text
    if reason is not None:
        assert reason in text, text


@pytest.fixture(scope="module")
def constants() -> DriftQcConstants:
    if not (CONSTANTS.is_file() and TRACKWEIGHT_CONSTANTS.is_file()):
        pytest.skip(FROZEN_ABSENT)
    return load_driftqc_constants(CONSTANTS, TRACKWEIGHT_CONSTANTS)


@pytest.fixture(scope="module")
def detector() -> Any:
    if not DETECTOR.is_file():
        pytest.skip(FROZEN_ABSENT)
    return load_frozen_detector(DETECTOR)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _copy_pair(tmp_path: Path) -> tuple[Path, Path]:
    c, t = tmp_path / "constants_v2_2.json", tmp_path / "constants_v2.json"
    shutil.copyfile(CONSTANTS, c)
    shutil.copyfile(TRACKWEIGHT_CONSTANTS, t)
    return c, t


def _rewrite(path: Path, edit: Any) -> None:
    obj = json.loads(path.read_bytes())
    edit(obj)
    path.write_text(json.dumps(obj, indent=1))


# ---------------------------------------------------------------------------------------------------- constants


@requires_frozen
def test_constants_values_from_frozen_files(constants: DriftQcConstants) -> None:
    assert constants.pair_floor_m == 0.03
    assert constants.patch_r == 2
    assert constants.min_pairs == 3
    assert constants.tau_abs == 0.3
    assert constants.K_consec == 2
    assert constants.keyframe_v1 == 0
    assert constants.s_margin_rows == 1
    assert constants.L_min_rows == 6
    assert constants.sha256 == FROZEN_CONSTANTS_SHA256 == _sha(CONSTANTS)
    assert constants.trackweight_sha256 == FROZEN_TRACKWEIGHT_CONSTANTS_SHA256 == _sha(TRACKWEIGHT_CONSTANTS)
    for name in ("patch_r", "min_pairs", "K_consec", "keyframe_v1", "s_margin_rows", "L_min_rows"):
        assert type(getattr(constants, name)) is int, name
    for name in ("pair_floor_m", "tau_abs"):
        assert type(getattr(constants, name)) is float, name


@requires_frozen
def test_constants_sha_mismatch_raises(tmp_path: Path) -> None:
    c, t = _copy_pair(tmp_path)
    c.write_bytes(c.read_bytes() + b"\n")
    with pytest.raises(DriftQcConfigError, match="sha256"):
        load_driftqc_constants(c, t)
    (tmp_path / "b").mkdir()
    c2, t2 = _copy_pair(tmp_path / "b")
    t2.write_bytes(t2.read_bytes() + b" ")
    with pytest.raises(DriftQcConfigError, match="sha256"):
        load_driftqc_constants(c2, t2)


@requires_frozen
def test_trackweight_constants_disagreement_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    c, t = _copy_pair(tmp_path)
    _rewrite(t, lambda o: o["SG_flag_for_CTRL_droplate"].__setitem__("tau_abs_m", 0.35))
    monkeypatch.setattr(wd, "FROZEN_CONSTANTS_SHA256", _sha(c))
    monkeypatch.setattr(wd, "FROZEN_TRACKWEIGHT_CONSTANTS_SHA256", _sha(t))
    with pytest.raises(DriftQcConfigError, match="tau_abs"):
        load_driftqc_constants(c, t)


@requires_frozen
def test_non_null_B_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    c, t = _copy_pair(tmp_path)
    _rewrite(c, lambda o: o["range_v1_TRUNC"].__setitem__("B_blind_run", 3))
    monkeypatch.setattr(wd, "FROZEN_CONSTANTS_SHA256", _sha(c))
    with pytest.raises(DriftQcConfigError, match="B_blind_run"):
        load_driftqc_constants(c, t)


# ---------------------------------------------------------------------------------------------------- detector


@requires_frozen
def test_detector_loader_accepts_frozen_file(detector: Any) -> None:
    for name in ("lift_patch", "sg_stat", "run_onset", "sg_onset"):
        assert callable(getattr(detector, name)), name
    assert detector.__name__ not in sys.modules
    assert detector.sg_onset(np.array([np.nan, 0.0, 0.4, np.nan, -0.5]), 0.3, 2) == 2


@requires_frozen
def test_detector_loader_rejects_one_byte_mutation(tmp_path: Path) -> None:
    raw = bytearray(DETECTOR.read_bytes())
    pos = raw.index(b"if x > thr:") + len(b"if x ")
    raw[pos:pos + 1] = b"<"  # one byte: the strict flag inequality flipped
    copy = tmp_path / "v1_weights.py"
    copy.write_bytes(bytes(raw))
    assert len(raw) == DETECTOR.stat().st_size
    with pytest.raises(DriftQcConfigError, match="sha256"):
        load_frozen_detector(copy)


def test_detector_loader_requires_callables(tmp_path: Path) -> None:
    copy = tmp_path / "partial.py"
    copy.write_text("def lift_patch():\n    pass\n\nsg_stat = run_onset = 1\n")
    with pytest.raises(DriftQcConfigError, match="sg_onset"):
        load_frozen_detector(copy, expected_sha256=_sha(copy))


# ---------------------------------------------------------------------------------------------------- range rule


@requires_frozen
@pytest.mark.parametrize("case_id", sorted(UNIT_TABLE, key=lambda k: int(k[1:])))
def test_unit_case(case_id: str, constants: DriftQcConstants, detector: Any) -> None:
    case = _case("unit_cases_v1", case_id)
    flags, nan, expect = UNIT_TABLE[case_id]
    n = int(case["N"])
    assert n == 24
    if case_id == "U12":
        assert _rows(case["m_equal_tau_at"]) == [9, 10]
        m = _m_from_pattern(n, [], [], constants.tau_abs)
        m[9], m[10] = constants.tau_abs, -constants.tau_abs
    else:
        assert _rows(case["flags"]) == flags
        assert _rows(case["nan"]) == nan
        m = _m_from_pattern(n, flags, nan, constants.tau_abs)
    _assert_expect_text(case["expect"], expect)
    dec = decide_range(m, constants, detector.sg_onset)
    _assert_decision(dec, n, expect)


@requires_frozen
@pytest.mark.parametrize("case_id", sorted(LABEL_TABLE))
def test_UL(case_id: str, constants: DriftQcConstants, detector: Any) -> None:
    case = _case("unit_cases_labels", case_id)
    onset, expect = LABEL_TABLE[case_id]
    n = int(case["N"])
    assert n == 24
    assert case["anchor_onset"] == onset
    _assert_expect_text(case["expect"], expect)
    if expect[3] is not None:
        assert f"b = {expect[3]}" in case["expect"].replace("−", "-")
    else:
        assert "no cut" in case["expect"]
    flags = [] if onset is None else list(range(onset, onset + constants.K_consec))
    m = _m_from_pattern(n, flags, [], constants.tau_abs)
    dec = decide_range(m, constants, detector.sg_onset)
    _assert_decision(dec, n, expect)


def _stub_constants(**over: Any) -> DriftQcConstants:
    base: dict[str, Any] = {"pair_floor_m": 0.03, "patch_r": 2, "min_pairs": 3, "tau_abs": 0.3, "K_consec": 2,
                            "keyframe_v1": 0, "s_margin_rows": 1, "L_min_rows": 6, "sha256": "c",
                            "trackweight_sha256": "t"}
    base.update(over)
    return DriftQcConstants(**base)


def test_decide_range_calls_onset_fn_with_constants() -> None:
    seen: list[tuple[Any, ...]] = []

    def onset_fn(m: np.ndarray, tau: float, k: int) -> int | None:
        seen.append((m, tau, k))
        return 9

    m = np.zeros(24)
    dec = decide_range(m, _stub_constants(tau_abs=0.25, K_consec=3), onset_fn)
    assert len(seen) == 1 and seen[0][0] is m and seen[0][1:] == (0.25, 3)
    assert (dec.decision, dec.kept, dec.cut_b) == ("TRUNCATE", (0, 7), 7)


def test_decide_range_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="1-D"):
        decide_range(np.zeros((24, 1)), _stub_constants(), lambda m, t, k: None)
    with pytest.raises(ValueError, match="empty"):
        decide_range(np.zeros(0), _stub_constants(), lambda m, t, k: None)
    with pytest.raises(ValueError, match="onset"):
        decide_range(np.zeros(24), _stub_constants(), lambda m, t, k: 24)


def test_exception_types_and_messages() -> None:
    exc = DriftRejected(f"{DRIFT_REJECTED_PREFIX}: short")
    assert str(exc) == "drift rejected: short"
    assert not isinstance(exc, ValueError)
    assert issubclass(DriftQcConfigError, ValueError)
    assert len(FROZEN_DETECTOR_SHA256) == 64
    assert (wd.SEGMENT_ID, wd.RECORD_LABEL) == ("#s0", "QC-less (no acceptance rule yet; the rule is deferred)")


# ---------------------------------------------------------------------------------------------------- E3 stage worker

_DRIFTQC_KEYS = {"version", "clip_id", "segment_id", "label", "n_rows", "frame_idx_sha256", "keyframe_index", "m",
                 "flags", "blind_rows", "onset", "end_f", "decision", "range", "reason", "cut_b", "B", "constants",
                 "detector", "cpu_seconds", "peak_rss_kib", "float64_cast_bytes"}
_RUN_DEPENDENT = ("cpu_seconds", "peak_rss_kib")  # process clock and process-wide RSS high-water mark


@pytest.fixture(scope="module")
def stage_cfg() -> DriftQcStageConfig:
    if not (CONSTANTS.is_file() and TRACKWEIGHT_CONSTANTS.is_file() and DETECTOR.is_file()):
        pytest.skip(FROZEN_ABSENT)
    return DriftQcStageConfig(enabled=True, constants_path=str(CONSTANTS),
                              trackweight_constants_path=str(TRACKWEIGHT_CONSTANTS), detector_path=str(DETECTOR))


def _json_or_none(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_bytes()) if path.is_file() else None


def _run_worker(root: Path, cfg: DriftQcStageConfig, contraction_from: int | None, clip_id: str = "clip0",
                **fixture: Any) -> tuple[Path, dict[str, Any], dict[str, Any] | None]:
    """Writes the fixture under root/clip_id, runs the worker (a DriftRejected propagates) and reads both records."""
    artifact_dir = root / clip_id
    write_driftqc_inputs(artifact_dir, driftqc_clip(0), contraction_from, **fixture)
    make_driftqc_worker(cfg)(clip_id, artifact_dir)
    doc = _json_or_none(artifact_dir / DRIFTQC_FILE)
    assert doc is not None
    return artifact_dir, doc, _json_or_none(artifact_dir / VALID_RANGE_FILE)


def _reference_m(artifact_dir: Path, constants: DriftQcConstants, detector: Any) -> np.ndarray:
    mask, geo, tr = (load_npz(artifact_dir / name) for name in ("mask.npz", "geometry.npz", "tracks.npz"))
    return np.asarray(detector.sg_stat(geo["pointmaps"].astype(np.float64), mask["masks"].astype(bool),
                                       tr["query_uv"].astype(np.float64), tr["tracks"].astype(np.float64),
                                       tr["vis"].astype(bool), constants.pair_floor_m, constants.min_pairs,
                                       constants.patch_r, kf=0))


def _m_of(doc: dict[str, Any]) -> np.ndarray:
    return np.array([np.nan if v is None else v for v in doc["m"]], dtype=np.float64)


def _normalised_sha(raw: bytes) -> str:
    doc = json.loads(raw)
    assert json.dumps(doc, indent=2).encode() == raw  # deterministic serialisation, so the sha can be recomputed
    doc.update(dict.fromkeys(_RUN_DEPENDENT, 0))
    return hashlib.sha256(json.dumps(doc, indent=2).encode()).hexdigest()


def _assert_records_identical(dir_a: Path, dir_b: Path) -> None:
    raw_a, raw_b = (d.joinpath(DRIFTQC_FILE).read_bytes() for d in (dir_a, dir_b))
    doc_a, doc_b = json.loads(raw_a), json.loads(raw_b)
    for doc in (doc_a, doc_b):
        for key in _RUN_DEPENDENT:
            doc.pop(key)
    assert doc_a == doc_b
    assert _normalised_sha(raw_a) == _normalised_sha(raw_b)
    vr_a, vr_b = (_json_or_none(d / VALID_RANGE_FILE) for d in (dir_a, dir_b))
    assert vr_a is not None and vr_b is not None
    assert vr_a["driftqc_sha256"] == hashlib.sha256(raw_a).hexdigest()
    assert vr_b["driftqc_sha256"] == hashlib.sha256(raw_b).hexdigest()
    assert {k: v for k, v in vr_a.items() if k != "driftqc_sha256"} == \
        {k: v for k, v in vr_b.items() if k != "driftqc_sha256"}


@requires_frozen
def test_engine_m_equals_reference_sg_stat(tmp_path: Path, stage_cfg: DriftQcStageConfig,
                                           constants: DriftQcConstants, detector: Any) -> None:
    artifact_dir, doc, _ = _run_worker(tmp_path, stage_cfg, 9)
    tol = float(_raw_constants()["dev_screen"]["G1_port_tol"])
    assert tol == 1e-6
    ref, m = _reference_m(artifact_dir, constants, detector), _m_of(doc)
    assert m.shape == ref.shape == (N_ROWS,)
    assert np.array_equal(np.isnan(m), np.isnan(ref))
    finite = np.isfinite(ref)
    assert finite.sum() >= 10 and (~finite).sum() >= 2
    assert np.max(np.abs(m[finite] - ref[finite])) <= tol


@requires_frozen
def test_worker_accept_healthy(tmp_path: Path, stage_cfg: DriftQcStageConfig, constants: DriftQcConstants,
                               detector: Any) -> None:
    artifact_dir, doc, vr = _run_worker(tmp_path, stage_cfg, None)
    assert detector.sg_onset(_reference_m(artifact_dir, constants, detector), constants.tau_abs,
                             constants.K_consec) is None
    assert (doc["decision"], doc["onset"], doc["end_f"], doc["range"], doc["cut_b"], doc["reason"]) == \
        ("ACCEPT", None, N_ROWS, [0, 23], None, None)
    assert vr is not None and vr["range"] == [0, 23] and vr["decision"] == "ACCEPT"


@requires_frozen
def test_worker_truncate_contracted(tmp_path: Path, stage_cfg: DriftQcStageConfig, constants: DriftQcConstants,
                                    detector: Any) -> None:
    artifact_dir, doc, vr = _run_worker(tmp_path, stage_cfg, 9)
    assert detector.sg_onset(_reference_m(artifact_dir, constants, detector), constants.tau_abs,
                             constants.K_consec) == 9
    assert (doc["decision"], doc["onset"], doc["end_f"], doc["range"], doc["cut_b"], doc["reason"]) == \
        ("TRUNCATE", 9, 9, [0, 7], 7, None)
    assert vr is not None and vr["range"] == [0, 7] and vr["decision"] == "TRUNCATE"


@requires_frozen
def test_valid_range_json_round_trip_read_and_slice(tmp_path: Path, stage_cfg: DriftQcStageConfig) -> None:
    for contraction_from, stop in ((None, None), (9, 7)):
        artifact_dir, _, _ = _run_worker(tmp_path, stage_cfg, contraction_from, clip_id=f"c{contraction_from}")
        decode = read_frame_idx(artifact_dir)
        rng = resolve_valid_range(artifact_dir, decode, driftqc_enabled=True)
        if stop is None:
            assert rng is None  # full range: the identity path
            continue
        assert rng == RowRange(stop=stop, n_rows=N_ROWS)
        for artifact in ("mask.npz", "geometry.npz", "tracks.npz"):
            out = slice_artifact(load_npz(artifact_dir / artifact), artifact, rng)
            assert np.array_equal(out["frame_idx"], decode[:stop + 1]), artifact


@requires_frozen
def test_reject_writes_driftqc_json_then_raises_prefixed(tmp_path: Path, stage_cfg: DriftQcStageConfig,
                                                         constants: DriftQcConstants, detector: Any) -> None:
    artifact_dir = tmp_path / "clip0"
    write_driftqc_inputs(artifact_dir, driftqc_clip(0), 6)
    assert detector.sg_onset(_reference_m(artifact_dir, constants, detector), constants.tau_abs,
                             constants.K_consec) == 6
    with pytest.raises(DriftRejected) as exc:
        make_driftqc_worker(stage_cfg)("clip0", artifact_dir)
    assert str(exc.value) == "drift rejected: short"
    doc = _json_or_none(artifact_dir / DRIFTQC_FILE)
    assert doc is not None
    assert (doc["decision"], doc["onset"], doc["end_f"], doc["range"], doc["cut_b"], doc["reason"]) == \
        ("REJECT", 6, 6, None, 4, "short")
    assert not (artifact_dir / VALID_RANGE_FILE).exists()


@requires_frozen
def test_worker_removes_stale_artifacts_first(tmp_path: Path, stage_cfg: DriftQcStageConfig) -> None:
    artifact_dir = tmp_path / "clip0"
    write_driftqc_inputs(artifact_dir, driftqc_clip(0), 6)
    (artifact_dir / DRIFTQC_FILE).write_text('{"stale": true}')
    (artifact_dir / VALID_RANGE_FILE).write_text('{"stale": true}')
    with pytest.raises(DriftRejected):
        make_driftqc_worker(stage_cfg)("clip0", artifact_dir)
    assert not (artifact_dir / VALID_RANGE_FILE).exists()
    assert "stale" not in json.loads((artifact_dir / DRIFTQC_FILE).read_bytes())


@requires_frozen
def test_orchestrator_marks_reject_failed_with_prefix(tmp_path: Path, stage_cfg: DriftQcStageConfig) -> None:
    cfg = EngineConfig(artifact_root=tmp_path / "engine", stages=("driftqc", "solve"))
    manifest = ClipManifest.create(["healthy", "rejected"], ["test", "test"], [N_ROWS, N_ROWS])
    write_driftqc_inputs(cfg.artifact_root / "healthy", driftqc_clip(0), None)
    write_driftqc_inputs(cfg.artifact_root / "rejected", driftqc_clip(0), 6)
    solved: list[str] = []
    fns: dict[str, StageFn] = {"driftqc": make_driftqc_worker(stage_cfg),
                               "solve": lambda clip_id, artifact_dir: solved.append(clip_id)}
    for clip_id in manifest.clip_ids():
        run_clip(clip_id, cfg, fns, manifest)
    assert solved == ["healthy"]
    assert manifest.status("healthy", "driftqc").state == "done"
    status = manifest.status("rejected", "driftqc")
    assert status.state == "failed" and status.error.startswith(DRIFT_REJECTED_PREFIX)
    assert status.error == "drift rejected: short"
    assert manifest.status("rejected", "solve").state == "pending"


@requires_frozen
def test_U13_keyframe_5_is_config_error(tmp_path: Path, stage_cfg: DriftQcStageConfig,
                                        constants: DriftQcConstants, detector: Any) -> None:
    artifact_dir = tmp_path / "clip0"
    write_driftqc_inputs(artifact_dir, driftqc_clip(0), None, keyframe_index=5)
    with pytest.raises(DriftQcConfigError) as exc:
        make_driftqc_worker(stage_cfg)("clip0", artifact_dir)
    assert not isinstance(exc.value, DriftRejected)
    assert not str(exc.value).startswith(DRIFT_REJECTED_PREFIX)
    assert not (artifact_dir / DRIFTQC_FILE).exists() and not (artifact_dir / VALID_RANGE_FILE).exists()
    mask, geo, tr = (load_npz(artifact_dir / name) for name in ("mask.npz", "geometry.npz", "tracks.npz"))
    with pytest.raises(DriftQcConfigError, match="keyframe"):
        driftqc_range(geo["pointmaps"], mask["masks"], tr["query_uv"], tr["tracks"], tr["vis"], 5, constants,
                      detector)


@requires_frozen
def test_U14_frame_scale_perturbation_identical(tmp_path: Path, stage_cfg: DriftQcStageConfig) -> None:
    perturbed = np.ones(N_ROWS, dtype=np.float32)
    perturbed[1::2] *= 1.7
    dir_a, _, _ = _run_worker(tmp_path / "a", stage_cfg, 9)
    dir_b, _, _ = _run_worker(tmp_path / "b", stage_cfg, 9, frame_scale=perturbed)
    assert np.array_equal(load_npz(dir_b / "geometry.npz")["frame_scale"], perturbed)
    assert not np.array_equal(load_npz(dir_a / "geometry.npz")["frame_scale"], perturbed)
    _assert_records_identical(dir_a, dir_b)


@requires_frozen
def test_worker_runs_without_frame_scale_key(tmp_path: Path, stage_cfg: DriftQcStageConfig) -> None:
    dir_a, _, _ = _run_worker(tmp_path / "a", stage_cfg, 9)
    dir_b, _, _ = _run_worker(tmp_path / "b", stage_cfg, 9, write_frame_scale=False)
    assert "frame_scale" not in load_npz(dir_b / "geometry.npz")
    _assert_records_identical(dir_a, dir_b)


@requires_frozen
def test_worker_passes_saved_arrays_to_sg_stat(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                               stage_cfg: DriftQcStageConfig, constants: DriftQcConstants) -> None:
    real = load_frozen_detector(DETECTOR)
    seen: dict[str, Any] = {}
    spy = ModuleType("spy_detector")
    for name in ("lift_patch", "run_onset", "sg_onset"):
        setattr(spy, name, getattr(real, name))

    def sg_stat(pm: np.ndarray, masks: np.ndarray, q: np.ndarray, tr: np.ndarray, vis: np.ndarray, *args: Any,
                **kwargs: Any) -> np.ndarray:
        seen.update(pm=pm, masks=masks, q=q, tr=tr, vis=vis, args=args, kwargs=kwargs)
        return np.asarray(real.sg_stat(pm, masks, q, tr, vis, *args, **kwargs))

    spy.sg_stat = sg_stat  # type: ignore[attr-defined]
    loaded: list[str] = []

    def fake_loader(path: Path, expected_sha256: str = FROZEN_DETECTOR_SHA256) -> ModuleType:
        loaded.append(str(path))
        return spy

    monkeypatch.setattr(wd, "load_frozen_detector", fake_loader)
    artifact_dir, _, _ = _run_worker(tmp_path, stage_cfg, 9)
    assert loaded == [str(DETECTOR)]
    mask, geo, tr = (load_npz(artifact_dir / name) for name in ("mask.npz", "geometry.npz", "tracks.npz"))
    assert seen["vis"].dtype == np.bool_ and seen["vis"].shape == tr["vis"].shape
    assert seen["vis"].tobytes() == tr["vis"].tobytes()
    assert seen["pm"].dtype == np.float64
    assert np.array_equal(seen["pm"], geo["pointmaps"].astype(np.float64), equal_nan=True)
    assert np.array_equal(seen["masks"], mask["masks"]) and seen["masks"].dtype == np.bool_
    assert np.array_equal(seen["tr"], tr["tracks"].astype(np.float64)) and seen["tr"].dtype == np.float64
    assert np.array_equal(seen["q"], tr["query_uv"].astype(np.float64)) and seen["q"].dtype == np.float64
    assert seen["args"] == (constants.pair_floor_m, constants.min_pairs, constants.patch_r)
    assert seen["kwargs"] == {"kf": 0}


@requires_frozen
def test_driftqc_json_schema_and_values(tmp_path: Path, stage_cfg: DriftQcStageConfig) -> None:
    artifact_dir, doc, vr = _run_worker(tmp_path, stage_cfg, 9)
    raw = (artifact_dir / DRIFTQC_FILE).read_bytes()
    assert set(doc) == _DRIFTQC_KEYS
    decode = read_frame_idx(artifact_dir)
    assert (doc["version"], doc["clip_id"], doc["segment_id"], doc["label"]) == (1, "clip0", "#s0", RECORD_LABEL)
    assert (doc["n_rows"], doc["frame_idx_sha256"], doc["keyframe_index"]) == (N_ROWS, frame_idx_sha256(decode), 0)
    m = doc["m"]
    assert len(m) == N_ROWS and m[0] is None and all(v is None or isinstance(v, float) for v in m)
    assert doc["flags"] == [v is not None and bool(np.isfinite(v)) and abs(v) > 0.3 for v in m]
    assert doc["blind_rows"] == [t for t in range(1, N_ROWS) if m[t] is None]
    assert doc["blind_rows"] and 0 not in doc["blind_rows"]
    assert (doc["onset"], doc["end_f"], doc["decision"], doc["range"], doc["reason"], doc["cut_b"], doc["B"]) == \
        (9, 9, "TRUNCATE", [0, 7], None, 7, "infinity")
    values = {"pair_floor_m": 0.03, "patch_r": 2, "min_pairs": 3, "tau_abs": 0.3, "K_consec": 2, "keyframe_v1": 0,
              "s_margin_rows": 1, "L_min_rows": 6}
    assert doc["constants"] == {"path": str(CONSTANTS), "sha256": FROZEN_CONSTANTS_SHA256,
                                "trackweight_path": str(TRACKWEIGHT_CONSTANTS),
                                "trackweight_sha256": FROZEN_TRACKWEIGHT_CONSTANTS_SHA256, "values": values}
    assert doc["detector"] == {"path": str(DETECTOR), "sha256": FROZEN_DETECTOR_SHA256}
    assert isinstance(doc["cpu_seconds"], float) and doc["cpu_seconds"] >= 0.0
    assert isinstance(doc["peak_rss_kib"], int) and doc["peak_rss_kib"] > 0
    geo, tr = load_npz(artifact_dir / "geometry.npz"), load_npz(artifact_dir / "tracks.npz")
    assert doc["float64_cast_bytes"] == 8 * (geo["pointmaps"].size + tr["tracks"].size + tr["query_uv"].size)
    assert vr is not None and list(vr) == list(VALID_RANGE_KEYS)
    assert vr == {"version": 1, "clip_id": "clip0", "segment_id": "#s0", "n_rows": N_ROWS,
                  "frame_idx_sha256": frame_idx_sha256(decode), "range": [0, 7], "decision": "TRUNCATE",
                  "driftqc_sha256": hashlib.sha256(raw).hexdigest()}


@requires_frozen
def test_worker_cpu_only_and_records_cpu_seconds(tmp_path: Path, stage_cfg: DriftQcStageConfig) -> None:
    # What this means is "the WORKER does not touch the GPU", which is not the same as "CUDA is uninitialized
    # in this process": the session-wide `_cuda_budget` fixture in tests/conftest.py calls `apply_gpu_budget`,
    # whose `torch.cuda.get_device_properties` initializes CUDA before any test body runs. Asserting
    # `not is_initialized()` therefore measured the FIXTURE, passing only on a host with no visible GPU and
    # failing here for a reason that has nothing to do with driftqc. Measure the delta across the worker instead.
    torch_mod = sys.modules.get("torch")
    cuda_live = torch_mod is not None and torch_mod.cuda.is_initialized()
    before = (cuda_live, torch_mod.cuda.memory_allocated() if cuda_live else 0)

    _, doc, _ = _run_worker(tmp_path, stage_cfg, None)

    torch_mod = sys.modules.get("torch")
    cuda_live_after = torch_mod is not None and torch_mod.cuda.is_initialized()
    after = (cuda_live_after, torch_mod.cuda.memory_allocated() if cuda_live_after else 0)
    assert after == before, "the driftqc worker initialized or allocated CUDA memory; it is a CPU-only stage"
    assert isinstance(doc["cpu_seconds"], float) and doc["cpu_seconds"] >= 0.0
    assert STAGE_GPU.get("driftqc", False) is False
    tree = ast.parse(Path(wd.__file__).read_text())
    imported = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import)
                for alias in node.names}
    imported |= {(node.module or "").split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert "torch" not in imported


def test_make_worker_refuses_disabled_config() -> None:
    with pytest.raises(ValueError, match="not enabled"):
        make_driftqc_worker(DriftQcStageConfig())
    with pytest.raises(ValueError, match="detector_path"):
        make_driftqc_worker(DriftQcStageConfig(enabled=True, constants_path="c", trackweight_constants_path="t"))


def test_stage_config_from_mapping() -> None:
    assert DriftQcStageConfig.from_mapping({}) == DriftQcStageConfig()
    assert DriftQcStageConfig.from_mapping({"enabled": False, "constants_path": None}) == DriftQcStageConfig()
    full = {"enabled": True, "constants_path": "c", "trackweight_constants_path": "t", "detector_path": "d"}
    assert DriftQcStageConfig.from_mapping(full) == DriftQcStageConfig(True, "c", "t", "d")
    with pytest.raises(ValueError, match="unknown"):
        DriftQcStageConfig.from_mapping({"enabled": False, "tau_abs": 0.2})
    with pytest.raises(ValueError, match="trackweight_constants_path"):
        DriftQcStageConfig.from_mapping({**full, "trackweight_constants_path": None})
    with pytest.raises(ValueError, match="enabled"):
        DriftQcStageConfig.from_mapping({"enabled": "yes"})


@requires_frozen
def test_make_worker_asserts_hashes_at_construction(tmp_path: Path, stage_cfg: DriftQcStageConfig) -> None:
    copy = tmp_path / "v1_weights.py"
    copy.write_bytes(DETECTOR.read_bytes() + b"\n")
    bad = DriftQcStageConfig(True, stage_cfg.constants_path, stage_cfg.trackweight_constants_path, str(copy))
    with pytest.raises(DriftQcConfigError, match="sha256"):
        make_driftqc_worker(bad)


@requires_frozen
def test_worker_checks_rows_and_keyframes(tmp_path: Path, stage_cfg: DriftQcStageConfig) -> None:
    worker = make_driftqc_worker(stage_cfg)
    artifact_dir = tmp_path / "kf"
    write_driftqc_inputs(artifact_dir, driftqc_clip(0), None)
    tracks = load_npz(artifact_dir / "tracks.npz")
    save_npz(artifact_dir / "tracks.npz", **{**tracks, "keyframe_index": np.array(1, dtype=np.int64)})
    with pytest.raises(ValueError, match="keyframe") as exc:
        worker("kf", artifact_dir)
    assert not isinstance(exc.value, DriftQcConfigError)
    rows_dir = tmp_path / "rows"
    write_driftqc_inputs(rows_dir, driftqc_clip(0), None)
    geometry = load_npz(rows_dir / "geometry.npz")
    save_npz(rows_dir / "geometry.npz", **{**geometry, "frame_idx": geometry["frame_idx"] + 1})
    with pytest.raises(ValueError, match="R7"):
        worker("rows", rows_dir)
    for d in (artifact_dir, rows_dir):
        assert not (d / DRIFTQC_FILE).exists()


@requires_frozen
def test_driftqc_range_shapes(constants: DriftQcConstants, detector: Any) -> None:
    clip = driftqc_clip(0)
    pm, masks = clip.pointmaps.numpy(), clip.masks.numpy()
    tracks, vis = clip.tracks.numpy(), clip.track_vis.numpy()
    m, dec = driftqc_range(pm, masks, tracks[0], tracks, vis, 0, constants, detector)
    assert m.dtype == np.float64 and m.shape == (N_ROWS,) and dec.n_rows == N_ROWS
    with pytest.raises(ValueError, match="shape"):
        driftqc_range(pm, masks[:-1], tracks[0], tracks, vis, 0, constants, detector)
    with pytest.raises(ValueError, match="shape"):
        driftqc_range(pm, masks, tracks[0], tracks, vis[:, :-1], 0, constants, detector)


def test_with_driftqc_order() -> None:
    staged = with_driftqc(STAGE_NAMES)
    assert "driftqc" not in STAGE_NAMES
    assert staged.index("driftqc") == staged.index("tracks") + 1 < staged.index("solve")
    assert tuple(s for s in staged if s != "driftqc") == STAGE_NAMES
    assert with_driftqc(staged) == staged
    assert with_driftqc(("decode", "tracks")) == ("decode", "tracks", "driftqc")
    with pytest.raises(ValueError, match="tracks"):
        with_driftqc(("decode", "solve"))
    with pytest.raises(ValueError, match="solve"):
        with_driftqc(("solve", "tracks"))


# ------------------------------------------------------------------------ tests that kill surviving worker mutations


def _run_with_spy_m(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cfg: DriftQcStageConfig,
                    m: np.ndarray) -> dict[str, Any]:
    """Runs the worker with a spy detector whose sg_stat returns `m` (the real onset functions stay) and reads
    driftqc.json."""
    real = load_frozen_detector(DETECTOR)
    spy = ModuleType("spy_detector")
    for name in ("lift_patch", "run_onset", "sg_onset"):
        setattr(spy, name, getattr(real, name))
    spy.sg_stat = lambda *args, **kwargs: m.copy()  # type: ignore[attr-defined]
    monkeypatch.setattr(wd, "load_frozen_detector", lambda path, expected_sha256=FROZEN_DETECTOR_SHA256: spy)
    _, doc, _ = _run_worker(tmp_path, cfg, None)
    return doc


@requires_frozen
def test_worker_flag_strict_at_tau_abs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage_cfg: DriftQcStageConfig,
                                      constants: DriftQcConstants) -> None:
    """m exactly tau_abs (both signs) is not a flag (strict >); the '>' to '>=' mutation fails here."""
    m = np.zeros(N_ROWS, dtype=np.float64)
    m[0] = np.nan
    m[9], m[10], m[15] = constants.tau_abs, -constants.tau_abs, 2.0 * constants.tau_abs
    doc = _run_with_spy_m(tmp_path, monkeypatch, stage_cfg, m)
    assert doc["m"][9] == 0.3 and doc["m"][10] == -0.3
    assert doc["flags"][9] is False and doc["flags"][10] is False
    assert doc["flags"][15] is True and sum(doc["flags"]) == 1
    assert (doc["decision"], doc["onset"]) == ("ACCEPT", None)


@requires_frozen
@pytest.mark.parametrize("row1_nan", [True, False], ids=["row1_nan", "row1_finite"])
def test_worker_blind_rows_scan_starts_at_row_1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                stage_cfg: DriftQcStageConfig, row1_nan: bool) -> None:
    """The blind-row scan is eligible from t = 1 (row 0, the keyframe, is excluded). With a NaN at
    row 1, blind_rows starts at 1, so a scan starting at row 2 fails; with row 1 finite, it is not blind."""
    m = np.zeros(N_ROWS, dtype=np.float64)
    m[[0, 5]] = np.nan
    if row1_nan:
        m[1] = np.nan
    doc = _run_with_spy_m(tmp_path, monkeypatch, stage_cfg, m)
    assert doc["m"][0] is None and 0 not in doc["blind_rows"]
    assert doc["blind_rows"] == ([1, 5] if row1_nan else [5])
