"""Canonical artifact digests and the default engine pipeline for the driftqc default-identity proof (also
loaded by file path in a two-commit comparison, so it imports nothing beyond the package's own code).

`run_default_pipeline(tmp)` runs three synthetic clips (`make_synthetic_clip(SMALL_CLIP, seed=s)`, s = 0, 1, 2) through
`("decode", "mask", "shape", "intrinsics", "hands", "geometry", "tracks", "solve", "qc")` with the default config:
the synthetic front end, `make_solve_worker(SMALL_SOLVE, solve_fn=RecordingSolve())` and the qc rule of
test_engine_solve_qc_e2e.py. It returns `digest_tree` of the artifact root.

Canonical digest (`artifact_digest`), because raw npz bytes carry zip entry write times:
- `.npz`: sha256 over key, `dtype.str`, shape and C-order `tobytes()`, keys sorted.
- `.json`: sha256 of `json.dumps(obj, sort_keys=True)` after dropping the wall-clock fields in `JSON_WALL_CLOCK`.
- Run location: when `run_root` is given, its absolute path prefix inside JSON strings and npz string arrays is replaced
  by `RUN_ROOT_TOKEN` (qc.json `rule.path` and clip_record.npz `object_mesh_path` embed the run's own directory).
- `manifest.parquet`: the rows with every `*_seconds` column dropped, columns sorted.
- `profile.json`: excluded (wall time). Everything else (e.g. `mesh.obj`): raw sha256.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from fastdo.engine.manifest import ClipManifest
from fastdo.engine.orchestrator import EngineConfig, run_manifest
from fastdo.engine.workers_qc import QcStageConfig, make_qc_worker_from_config
from fastdo.engine.workers_solve import make_solve_worker
from fastdo.qc import AcceptanceRule, ConformalCalibration
from fastdo.qc.rule_io import save_acceptance_rule
from fastdo.solver.synthetic import make_synthetic_clip
from tests.engine.fakes_solve import (
    SMALL_CLIP,
    SMALL_SOLVE,
    RecordingSolve,
    synthetic_front_end_stage_fns,
)

STAGES = ("decode", "mask", "shape", "intrinsics", "hands", "geometry", "tracks", "solve", "qc")
SEEDS = (0, 1, 2)
ENGINE_DIR = "engine"
MANIFEST_FILE = "manifest.parquet"
EXCLUDED = frozenset({"profile.json"})
RUN_ROOT_TOKEN = "<run_root>"
# (file name, key path) of JSON values that are wall-clock time, dropped before hashing.
JSON_WALL_CLOCK: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("solve.json", ("solver", "seconds")),
    ("shape.json", ("phase_seconds",)),
)


def _sha256_json(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def _prefixes(run_root: Path | None) -> tuple[str, ...]:
    if run_root is None:
        return ()
    return tuple(sorted({str(Path(run_root)), str(Path(run_root).resolve())}, key=len, reverse=True))


def _relocate(text: str, prefixes: tuple[str, ...]) -> str:
    for prefix in prefixes:
        text = text.replace(prefix, RUN_ROOT_TOKEN)
    return text


def _relocate_json(obj: Any, prefixes: tuple[str, ...]) -> Any:
    if isinstance(obj, str):
        return _relocate(obj, prefixes)
    if isinstance(obj, dict):
        return {k: _relocate_json(v, prefixes) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_relocate_json(v, prefixes) for v in obj]
    return obj


def _npz_digest(path: Path, prefixes: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    with np.load(path, allow_pickle=False) as data:
        for key in sorted(data.files):
            array = np.ascontiguousarray(data[key])
            if prefixes and array.dtype.kind == "U":
                array = np.asarray(np.vectorize(lambda v: _relocate(str(v), prefixes), otypes=[str])(array)) \
                    if array.size else array
            digest.update(f"{key}|{array.dtype.str}|{array.shape}|".encode())
            digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _json_digest(path: Path, prefixes: tuple[str, ...]) -> str:
    obj = _relocate_json(json.loads(path.read_text()), prefixes)
    for name, keys in JSON_WALL_CLOCK:
        if path.name != name:
            continue
        parent = obj
        for key in keys[:-1]:
            parent = parent.get(key) if isinstance(parent, dict) else None
        if isinstance(parent, dict):
            parent.pop(keys[-1], None)
    return _sha256_json(obj)


def _manifest_digest(path: Path) -> str:
    rows = pq.read_table(path).to_pylist()
    kept = [{k: row[k] for k in sorted(row) if not k.endswith("_seconds")} for row in rows]
    return _sha256_json(sorted(kept, key=lambda row: str(row.get("clip_id"))))


def artifact_digest(path: Path, run_root: Path | None = None) -> str:
    """The canonical digest of one artifact file (module docstring)."""
    path, prefixes = Path(path), _prefixes(run_root)
    if path.name == MANIFEST_FILE:
        return _manifest_digest(path)
    if path.suffix == ".npz":
        return _npz_digest(path, prefixes)
    if path.suffix == ".json":
        return _json_digest(path, prefixes)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest_tree(root: Path, run_root: Path | None = None) -> dict[str, str]:
    """`{relative posix path: artifact_digest(p, run_root)}` for every file under `root` except `EXCLUDED` names."""
    root = Path(root)
    return {p.relative_to(root).as_posix(): artifact_digest(p, run_root) for p in sorted(root.rglob("*"))
            if p.is_file() and p.name not in EXCLUDED}


def _write_rule(path: Path) -> Path:
    save_acceptance_rule(path, AcceptanceRule(
        calibration=ConformalCalibration(threshold=0.5, alpha=0.1, n_calibration=20),
        logistic_weights=np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0]), feature_mean=np.zeros(5), feature_std=np.ones(5)))
    return path


def run_default_pipeline(tmp: Path) -> dict[str, str]:
    """Runs the default pipeline (module docstring) under `tmp` and returns `digest_tree(tmp / ENGINE_DIR, tmp)`.
    Inputs (templates, rule) are written under `tmp / "inputs"`, outside the digested root."""
    tmp = Path(tmp)
    inputs, root = tmp / "inputs", tmp / ENGINE_DIR
    inputs.mkdir(parents=True, exist_ok=True)
    clips = {f"clip{s}": make_synthetic_clip(SMALL_CLIP, seed=s) for s in SEEDS}
    fns = synthetic_front_end_stage_fns(clips, inputs / "templates")
    fns["solve"] = make_solve_worker(SMALL_SOLVE, solve_fn=RecordingSolve())
    fns["qc"] = make_qc_worker_from_config(QcStageConfig(rule_path=str(_write_rule(inputs / "rule.json"))))
    ids = list(clips)
    manifest = ClipManifest.create(ids, ["synthetic"] * len(ids), [SMALL_CLIP.n_frames] * len(ids))
    run_manifest(manifest, EngineConfig(artifact_root=root, stages=STAGES, manifest_path=root / MANIFEST_FILE), fns)
    return digest_tree(root, run_root=tmp)
