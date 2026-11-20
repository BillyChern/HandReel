"""driftqc stage (v1 = TRUNC): frozen constants and SG detector, the anchor range rule and the CPU stage worker.

- The three frozen sha256 values are module constants, asserted on every load; the files are read by configured path
  and nothing is vendored into src/. `load_frozen_detector` execs exactly the asserted bytes of
  v1_weights.py into a fresh, unregistered module; `load_driftqc_constants` also requires B_blind_run null and the
  trackweight SG values to agree.
- `decide_range` is the anchor range rule, with the onset from the frozen `sg_onset`; `driftqc_range` is
  the range function (shared with the N0a adapter): lossless float64 widening, then `sg_stat` on the saved arrays.
- The worker (opt-in) writes `driftqc.json`, then `valid_range.json` for ACCEPT and TRUNCATE, and raises
  `DriftRejected("drift rejected: short")` for REJECT. `geometry.npz` `frame_scale` is never accessed.

No torch import: the stage is CPU-only and never creates a CUDA context.
"""

from __future__ import annotations

import hashlib
import json
import os
import resource
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from fastdo.engine.valid_range import (
    DRIFTQC_FILE,
    VALID_RANGE_FILE,
    VALID_RANGE_KEYS,
    frame_idx_sha256,
)
from fastdo.engine.valid_range import VALID_RANGE_VERSION as _VALID_RANGE_VERSION
from fastdo.engine.worker_support import load_npz, read_frame_idx, require_artifact
from fastdo.engine.workers_qc import check_decode_rows

if TYPE_CHECKING:
    from fastdo.engine.orchestrator import StageFn

__all__ = [
    "DRIFTQC_FILE", "DRIFT_REJECTED_PREFIX", "FROZEN_CONSTANTS_SHA256", "FROZEN_DETECTOR_SHA256",
    "FROZEN_TRACKWEIGHT_CONSTANTS_SHA256", "RECORD_LABEL", "SEGMENT_ID", "Decision", "DriftDecision",
    "DriftQcConfigError", "DriftQcConstants", "DriftQcStageConfig", "DriftRejected", "decide_range", "driftqc_range",
    "load_driftqc_constants", "load_frozen_detector", "make_driftqc_worker", "with_driftqc",
]

FROZEN_DETECTOR_SHA256 = "a12bcac842091f1e50d9955da0f7b134a45941212c8d466cf23fc9fb1663637a"  # v1_weights.py
FROZEN_CONSTANTS_SHA256 = "a07f3baf54fe1355651dbe345e7c041dca1b1b05553c5e2e4bfee367342a224b"  # constants_v2_2.json
FROZEN_TRACKWEIGHT_CONSTANTS_SHA256 = "9a0c703943ec6c3a15ebce8579aebf1710dbc85ffe366d31a3b89b6b255d1e65"  # v2.json
DRIFT_REJECTED_PREFIX = "drift rejected"
SEGMENT_ID = "#s0"
RECORD_LABEL = "QC-less (no acceptance rule yet; the rule is deferred)"
Decision = Literal["ACCEPT", "TRUNCATE", "REJECT"]

_DETECTOR_CALLABLES = ("lift_patch", "sg_stat", "run_onset", "sg_onset")
_DETECTOR_MODULE_NAME = "fastdo_driftqc_frozen_v1_weights"
# detector_SG key -> SG_flag_for_CTRL_droplate key (the five shared SG constants).
_SHARED_SG_KEYS = (("pair_floor_m", "pair_floor_m"), ("patch_r", "patch_r"), ("min_pairs", "min_pairs"),
                   ("tau_abs", "tau_abs_m"), ("K_consec", "K_consec"))
_FLOAT_KEYS = frozenset({"pair_floor_m", "tau_abs"})
_VALUE_KEYS = ("pair_floor_m", "patch_r", "min_pairs", "tau_abs", "K_consec", "keyframe_v1", "s_margin_rows",
               "L_min_rows")
_PATH_KEYS = ("constants_path", "trackweight_constants_path", "detector_path")
DRIFTQC_JSON_VERSION = 1


class DriftRejected(Exception):
    """The anchor range rule's REJECT: `str(exc) == "drift rejected: short"` (the GateRejected pattern)."""


class DriftQcConfigError(ValueError):
    """A driftqc configuration error (hash, constants, keyframe); its message never starts with the reject prefix."""


@dataclass(frozen=True)
class DriftQcConstants:
    """detector_SG and range_v1_TRUNC values (B_blind_run is null, i.e. B = infinity) plus both files' sha256."""

    pair_floor_m: float
    patch_r: int
    min_pairs: int
    tau_abs: float
    K_consec: int
    keyframe_v1: int
    s_margin_rows: int
    L_min_rows: int
    sha256: str
    trackweight_sha256: str


@dataclass(frozen=True)
class DriftDecision:
    """One clip's anchor decision."""

    n_rows: int
    onset: int | None
    end_f: int  # onset, or n_rows
    decision: Decision
    kept: tuple[int, int] | None  # inclusive; (0, n_rows-1) ACCEPT, (0, b0) TRUNCATE, None REJECT
    reason: str | None  # None or "short"
    cut_b: int | None  # end_f - 1 - s for TRUNCATE/REJECT (may be negative); None for ACCEPT


def _read_asserted(path: Path, expected_sha256: str, what: str) -> bytes:
    """The file's bytes, read once, after asserting their sha256."""
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise DriftQcConfigError(f"{what} {path} unreadable: {exc}") from exc
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha256:
        raise DriftQcConfigError(f"{what} {path} sha256 {actual} != frozen {expected_sha256}")
    return raw


def load_frozen_detector(path: Path, expected_sha256: str = FROZEN_DETECTOR_SHA256) -> ModuleType:
    """Exec the frozen SG detector's asserted bytes into a fresh module.

    Reads the bytes once, asserts sha256, execs exactly those bytes into a fresh module (no import-system lookup, no
    .pyc, not registered in sys.modules); asserts callables lift_patch, sg_stat, run_onset, sg_onset.

    Raises:
        DriftQcConfigError: On a sha256 mismatch, an unreadable file or a missing callable.
    """
    raw = _read_asserted(path, expected_sha256, "driftqc detector")
    module = ModuleType(_DETECTOR_MODULE_NAME)
    module.__file__ = str(path)
    code = compile(raw, str(path), "exec", dont_inherit=True)
    exec(code, module.__dict__)  # noqa: S102 - exactly the sha256-asserted frozen bytes
    missing = [name for name in _DETECTOR_CALLABLES if not callable(getattr(module, name, None))]
    if missing:
        raise DriftQcConfigError(f"driftqc detector {path} lacks callables {missing}")
    return module


def _section(obj: Any, key: str, path: Path) -> dict[str, Any]:
    section = obj.get(key) if isinstance(obj, dict) else None
    if not isinstance(section, dict):
        raise DriftQcConfigError(f"{path}: section {key!r} absent or not an object")
    return section


def _number(section: dict[str, Any], key: str, where: str, *, is_float: bool) -> float | int:
    if key not in section:
        raise DriftQcConfigError(f"{where}: key {key!r} absent")
    value = section[key]
    ok = (isinstance(value, (int, float)) if is_float else isinstance(value, int)) and not isinstance(value, bool)
    if not ok:
        raise DriftQcConfigError(f"{where}[{key!r}] = {value!r} is not {'a number' if is_float else 'an integer'}")
    return float(value) if is_float else int(value)


def load_driftqc_constants(path: Path, trackweight_path: Path) -> DriftQcConstants:
    """The frozen driftqc constants, with both files' sha256 asserted.

    Reads constants_v2_2.json detector_SG {pair_floor_m, patch_r, min_pairs, tau_abs, K_consec, keyframe_v1} and
    range_v1_TRUNC {s_margin_rows, L_min_rows, B_blind_run}; asserts B_blind_run is None; asserts trackweight
    SG_flag_for_CTRL_droplate {pair_floor_m, patch_r, min_pairs, tau_abs_m, K_consec} equal the detector_SG values.

    Raises:
        DriftQcConfigError: On a sha256 mismatch, a missing or mistyped key, a non-null B_blind_run or a disagreement.
    """
    path, trackweight_path = Path(path), Path(trackweight_path)
    try:
        doc = json.loads(_read_asserted(path, FROZEN_CONSTANTS_SHA256, "driftqc constants"))
        tw_doc = json.loads(_read_asserted(trackweight_path, FROZEN_TRACKWEIGHT_CONSTANTS_SHA256,
                                           "trackweight constants"))
    except json.JSONDecodeError as exc:
        raise DriftQcConfigError(f"constants JSON unparseable: {exc}") from exc
    det = _section(doc, "detector_SG", path)
    rng = _section(doc, "range_v1_TRUNC", path)
    tw = _section(tw_doc, "SG_flag_for_CTRL_droplate", trackweight_path)
    det_where, rng_where = f"{path} detector_SG", f"{path} range_v1_TRUNC"
    values: dict[str, Any] = {key: _number(det, key, det_where, is_float=key in _FLOAT_KEYS)
                              for key in ("pair_floor_m", "patch_r", "min_pairs", "tau_abs", "K_consec",
                                          "keyframe_v1")}
    for key in ("s_margin_rows", "L_min_rows"):
        values[key] = _number(rng, key, rng_where, is_float=False)
    if "B_blind_run" not in rng or rng["B_blind_run"] is not None:
        raise DriftQcConfigError(f"{rng_where}: B_blind_run must be null (B = infinity), got "
                                 f"{rng.get('B_blind_run', '<absent>')!r}")
    tw_where = f"{trackweight_path} SG_flag_for_CTRL_droplate"
    for det_key, tw_key in _SHARED_SG_KEYS:
        tw_value = _number(tw, tw_key, tw_where, is_float=det_key in _FLOAT_KEYS)
        if tw_value != values[det_key]:
            raise DriftQcConfigError(f"constants disagree: detector_SG {det_key}={values[det_key]!r} vs "
                                     f"trackweight {tw_key}={tw_value!r}")
    return DriftQcConstants(**values, sha256=FROZEN_CONSTANTS_SHA256,
                            trackweight_sha256=FROZEN_TRACKWEIGHT_CONSTANTS_SHA256)


def decide_range(m: np.ndarray, constants: DriftQcConstants,
                 onset_fn: Callable[[np.ndarray, float, int], int | None]) -> DriftDecision:
    """Pure. onset = onset_fn(m, tau_abs, K_consec) (the frozen sg_onset); the anchor range rule.

    end_f = onset if an onset exists else N; end_f == N: ACCEPT [0, N-1]; b0 = end_f - 1 - s; b0 + 1 >= L_min:
    TRUNCATE [0, b0]; else REJECT "short". Every non-ACCEPT decision is a cut at b = b0 (may be negative).

    Raises:
        ValueError: If m is not a non-empty 1-D array, or onset_fn returns an onset outside [0, N-1].
    """
    if m.ndim != 1:
        raise ValueError(f"m must be 1-D, got shape {m.shape}")
    n = int(m.shape[0])
    if n == 0:
        raise ValueError("m is empty")
    raw_onset = onset_fn(m, constants.tau_abs, constants.K_consec)
    onset = None if raw_onset is None else int(raw_onset)
    if onset is not None and not 0 <= onset < n:
        raise ValueError(f"onset {onset} outside [0, {n - 1}]")
    end_f = n if onset is None else onset
    if end_f == n:
        return DriftDecision(n_rows=n, onset=None, end_f=n, decision="ACCEPT", kept=(0, n - 1), reason=None,
                             cut_b=None)
    b0 = end_f - 1 - constants.s_margin_rows
    if b0 + 1 >= constants.L_min_rows:
        return DriftDecision(n_rows=n, onset=onset, end_f=end_f, decision="TRUNCATE", kept=(0, b0), reason=None,
                             cut_b=b0)
    return DriftDecision(n_rows=n, onset=onset, end_f=end_f, decision="REJECT", kept=None, reason="short", cut_b=b0)


def driftqc_range(pointmaps: np.ndarray, masks: np.ndarray, query_uv: np.ndarray, tracks: np.ndarray,
                  vis: np.ndarray, keyframe_index: int, constants: DriftQcConstants,
                  detector: ModuleType) -> tuple[np.ndarray, DriftDecision]:
    """The B1 range function (the N0a adapter calls it): `(m (N,) float64, decision)`.

    Shapes (N,H,W,3), (N,H,W), (J,2), (N,J,2), (N,J); ValueError on disagreement. keyframe_index !=
    constants.keyframe_v1 -> DriftQcConfigError (U13). Casts pointmaps, tracks, query_uv to float64 (lossless widening,
    as trackweight v1_stats.py does; no scale), masks and vis to bool; m = detector.sg_stat(pm, masks, q, tr, vis,
    pair_floor_m, min_pairs, patch_r, kf=0)."""
    pm, mk, q, tr, vs = (np.asarray(a) for a in (pointmaps, masks, query_uv, tracks, vis))
    shapes = f"pointmaps {pm.shape}, masks {mk.shape}, query_uv {q.shape}, tracks {tr.shape}, vis {vs.shape}"
    if pm.ndim != 4 or pm.shape[3] != 3 or pm.shape[0] == 0 or q.ndim != 2 or q.shape[1] != 2:
        raise ValueError(f"driftqc_range input shapes disagree: {shapes}")
    n, h, w = pm.shape[:3]
    j = q.shape[0]
    if mk.shape != (n, h, w) or tr.shape != (n, j, 2) or vs.shape != (n, j):
        raise ValueError(f"driftqc_range input shapes disagree: {shapes}")
    if int(keyframe_index) != constants.keyframe_v1:
        raise DriftQcConfigError(f"keyframe_index {int(keyframe_index)} != keyframe_v1 {constants.keyframe_v1}: a "
                                 f"configuration error, not a drift decision; no backward scan runs")
    m = np.asarray(detector.sg_stat(pm.astype(np.float64), mk.astype(bool), q.astype(np.float64),
                                    tr.astype(np.float64), vs.astype(bool), constants.pair_floor_m,
                                    constants.min_pairs, constants.patch_r, kf=constants.keyframe_v1),
                   dtype=np.float64)
    if m.shape != (n,):
        raise ValueError(f"sg_stat returned shape {m.shape}, expected ({n},)")
    return m, decide_range(m, constants, detector.sg_onset)


@dataclass(frozen=True)
class DriftQcStageConfig:
    """The `driftqc_stage` config block. Opt-in: disabled by default; when enabled every path must be set."""

    enabled: bool = False
    constants_path: str | None = None
    trackweight_constants_path: str | None = None
    detector_path: str | None = None

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> DriftQcStageConfig:
        """From the `driftqc_stage` block (dict or OmegaConf node); ValueError on an unknown key, a non-bool `enabled`,
        or `enabled` with any path unset."""
        known = ("enabled", *_PATH_KEYS)
        unknown = sorted({str(key) for key in mapping} - set(known))
        if unknown:
            raise ValueError(f"driftqc_stage has unknown keys {unknown}; known keys: {list(known)}")
        enabled = mapping.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ValueError(f"driftqc_stage.enabled must be bool, got {enabled!r}")  # noqa: TRY004 - spec: ValueError
        paths = [None if mapping.get(key) is None else str(mapping.get(key)) for key in _PATH_KEYS]
        cfg = cls(enabled, paths[0], paths[1], paths[2])
        if enabled:
            cfg.required_paths()
        return cfg

    def required_paths(self) -> tuple[Path, Path, Path]:
        """(constants, trackweight constants, detector) paths; ValueError naming every unset one."""
        values = (self.constants_path, self.trackweight_constants_path, self.detector_path)
        missing = [key for key, value in zip(_PATH_KEYS, values, strict=True) if not value]
        if missing:
            raise ValueError(f"driftqc_stage is enabled but {missing} unset; every path must be set")
        return Path(str(values[0])), Path(str(values[1])), Path(str(values[2]))


def _write_atomic(path: Path, raw: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_bytes(raw)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def make_driftqc_worker(cfg: DriftQcStageConfig) -> StageFn:
    """The `driftqc` stage worker; loads and asserts the detector and constants here, before any clip. Raises
    ValueError if the stage is not enabled or a path is unset, DriftQcConfigError on a hash or constants failure."""
    if not cfg.enabled:
        raise ValueError("driftqc_stage is not enabled: the driftqc stage is opt-in")
    constants_path, trackweight_path, detector_path = cfg.required_paths()
    detector = load_frozen_detector(detector_path)
    constants = load_driftqc_constants(constants_path, trackweight_path)
    provenance = {
        "B": "infinity",
        "constants": {"path": str(constants_path), "sha256": constants.sha256,
                      "trackweight_path": str(trackweight_path),
                      "trackweight_sha256": constants.trackweight_sha256,
                      "values": {key: getattr(constants, key) for key in _VALUE_KEYS}},
        "detector": {"path": str(detector_path), "sha256": FROZEN_DETECTOR_SHA256},
    }

    def driftqc_worker(clip_id: str, artifact_dir: Path) -> None:
        artifact_dir = Path(artifact_dir)
        for name in (DRIFTQC_FILE, VALID_RANGE_FILE):
            (artifact_dir / name).unlink(missing_ok=True)
        decode_idx = read_frame_idx(artifact_dir)
        mask = load_npz(require_artifact(artifact_dir / "mask.npz", "mask"))
        with np.load(require_artifact(artifact_dir / "geometry.npz", "geometry"), allow_pickle=False) as z:
            if "pointmaps_format" in z.files:
                # driftqc samples pointmaps at TRACKED pixels, which can leave the object mask -- exactly
                # what it detects. A masked artifact has no values there; refuse rather than read NaNs as drift.
                raise ValueError(f"{clip_id}: geometry.npz is masked; driftqc needs dense pointmaps -- "
                                 "write geometry with pointmap_storage='dense' whenever driftqc is enabled")
            pointmaps, geometry_idx = np.asarray(z["pointmaps"]), np.asarray(z["frame_idx"])
        tracks = load_npz(require_artifact(artifact_dir / "tracks.npz", "tracks"))
        check_decode_rows(mask["frame_idx"], decode_idx, "mask.npz")
        check_decode_rows(geometry_idx, decode_idx, "geometry.npz")
        check_decode_rows(tracks["frame_idx"], decode_idx, "tracks.npz")
        keyframe = int(mask["keyframe_index"])
        if int(tracks["keyframe_index"]) != keyframe:
            raise ValueError(f"{clip_id}: tracks.npz keyframe_index {int(tracks['keyframe_index'])} != mask.npz "
                             f"keyframe_index {keyframe}")
        t0 = time.process_time()
        m, dec = driftqc_range(pointmaps, mask["masks"], tracks["query_uv"], tracks["tracks"], tracks["vis"],
                               keyframe, constants, detector)
        cpu_seconds = time.process_time() - t0
        finite = np.isfinite(m)
        idx_sha = frame_idx_sha256(decode_idx)
        record: dict[str, Any] = {
            "version": DRIFTQC_JSON_VERSION, "clip_id": clip_id, "segment_id": SEGMENT_ID, "label": RECORD_LABEL,
            "n_rows": dec.n_rows, "frame_idx_sha256": idx_sha, "keyframe_index": keyframe,
            "m": [float(v) if ok else None for v, ok in zip(m, finite, strict=True)],
            "flags": [bool(ok and abs(v) > constants.tau_abs) for v, ok in zip(m, finite, strict=True)],
            "blind_rows": [t for t in range(1, dec.n_rows) if not finite[t]],  # row 0 is the keyframe
            "onset": dec.onset, "end_f": dec.end_f, "decision": dec.decision,
            "range": None if dec.kept is None else list(dec.kept), "reason": dec.reason, "cut_b": dec.cut_b,
            **provenance, "cpu_seconds": cpu_seconds,
            "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "float64_cast_bytes": np.dtype(np.float64).itemsize * (pointmaps.size + tracks["tracks"].size
                                                                   + tracks["query_uv"].size),
        }
        raw = json.dumps(record, indent=2, allow_nan=False).encode()
        _write_atomic(artifact_dir / DRIFTQC_FILE, raw)
        if dec.kept is not None:
            values = {"version": _VALID_RANGE_VERSION, "clip_id": clip_id, "segment_id": SEGMENT_ID,
                      "n_rows": dec.n_rows, "frame_idx_sha256": idx_sha, "range": list(dec.kept),
                      "decision": dec.decision, "driftqc_sha256": hashlib.sha256(raw).hexdigest()}
            valid_range = {key: values[key] for key in VALID_RANGE_KEYS}
            _write_atomic(artifact_dir / VALID_RANGE_FILE, json.dumps(valid_range, indent=2).encode())
        if dec.decision == "REJECT":
            raise DriftRejected(f"{DRIFT_REJECTED_PREFIX}: {dec.reason}")

    return driftqc_worker


def with_driftqc(stages: tuple[str, ...]) -> tuple[str, ...]:
    """`stages` with "driftqc" inserted immediately after "tracks" (idempotent). ValueError if "tracks" is absent,
    "solve" precedes "tracks", or "driftqc" is present elsewhere."""
    stages = tuple(stages)
    if "tracks" not in stages:
        raise ValueError(f"with_driftqc needs a 'tracks' stage, got {stages}")
    at = stages.index("tracks")
    if "solve" in stages and stages.index("solve") < at:
        raise ValueError(f"'solve' precedes 'tracks' in {stages}")
    if "driftqc" in stages:
        if stages.count("driftqc") == 1 and stages.index("driftqc") == at + 1:
            return stages
        raise ValueError(f"'driftqc' must follow 'tracks' immediately (once), got {stages}")
    return (*stages[:at + 1], "driftqc", *stages[at + 1:])
