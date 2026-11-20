"""Batched execution for the throughput benchmark: the `bench.batched` block, the batched stages, the report
fields, and an artifact-tree comparison of a single-stream run against a batched one.

    python -m fastdo.bench.bench_batched compare A B --out report.json

`compare_artifact_dirs` reads every `.npz` and `.json` under two artifact roots, except `profile.json`, `hands_job/`
and `.hands_jobs/`. Float arrays report `max_abs_diff` (NaN positions must match), bool and integer arrays their
mismatch fraction, and other arrays (0-d strings such as the encoder's crop key) exact equality. JSON files are
compared with the keys `phase_seconds`, `seconds` and `created_utc` dropped at every depth.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf

from fastdo.bench.bench_stages import driftqc_enabled, mask_compile, source_profile_config
from fastdo.engine.apportion import check_apportionment, read_batch_log
from fastdo.engine.batched_runner import (
    HOST_CAP_GB,
    BatchedEngineConfig,
    BatchRunReport,
    CardAdmissionConfig,
    MemoryLedgerConfig,
    OomRetryConfig,
    StageBatchConfig,
    check_stage_budgets,
)
from fastdo.engine.batched_stages import (
    BatchedStage,
    ConcurrentClipStage,
    FrameBatchedStage,
    HandsBatchStage,
    SolveBatchStage,
)
from fastdo.engine.manifest import STAGE_NAMES, ClipManifest
from fastdo.engine.orchestrator import STAGE_GPU, EngineConfig, StageFn
from fastdo.engine.source_profile import profiled_solve_halves
from fastdo.engine.worker_support import BatchedWorker
from fastdo.engine.workers_contact import ContactStageConfig
from fastdo.engine.workers_solve import SolveStageConfig
from fastdo.engine.workers_video import make_mask_worker, make_tracks_worker
from fastdo.hands.hawor_batched import hands_config_from_cfg

__all__ = ["batched_engine_config", "build_batched_stages", "card_admission", "check_single_stream_fns",
           "compare_artifact_dirs", "execution_provenance", "main", "memory_ledger", "oom_retry", "pipeline_resources",
           "prewarm_stages"]

logger = logging.getLogger(__name__)

EXCLUDED_FILES = frozenset({"profile.json"})
EXCLUDED_DIRS = frozenset({"hands_job", ".hands_jobs"})
VOLATILE_KEYS = frozenset({"phase_seconds", "seconds", "created_utc", "cache_hit", "cache_seconds"})


def batched_engine_config(bench: DictConfig, engine: EngineConfig) -> BatchedEngineConfig:
    """`BatchedEngineConfig` from `bench.batched`.

    Raises `ValueError` on an invalid stage block: a NaN budget, a key outside `STAGE_NAMES` (which would leave the
    real stage silently unbatched), a `max_batch > 1` on a stage still marked `unmeasured`, or -- as
    `StageBudgetError` -- a stage whose declared batch memory passes the cap in force, or one that batches while
    declaring neither `budget_gb` nor `fixed_cost_gb`. The cap in force is the lower of the block's own `cap_gb`
    (else `HOST_CAP_GB`) and the cap `apply_gpu_budget` installs in this process, so a config is never certified
    against a cap the process does not apply; `check_stage_budgets` logs the disagreement and names both. The
    keys `pipeline_waves`, `pipeline_depth` and `pipeline_resources` (validated by `pipeline_resources`) are read too,
    and the key `memory_ledger` (validated by `memory_ledger`), and the keys `card_admission` and `oom_retry`
    (validated by `card_admission` and `oom_retry`), and the key `prewarm` (validated by `prewarm_stages`).
    """
    block = bench.batched
    stages = cast("dict[str, Any]", OmegaConf.to_container(block.get("stages") or {}, resolve=True))
    unknown = sorted(str(stage) for stage in stages if stage not in STAGE_NAMES)
    if unknown:
        raise ValueError(f"bench.batched.stages has unknown stage keys {unknown}; valid keys: {list(STAGE_NAMES)}")
    wave = block.get("max_clips_per_wave")
    stage_batches = {str(s): StageBatchConfig(**(v or {})) for s, v in stages.items()}
    cap = block.get("cap_gb")
    check_stage_budgets(stage_batches, HOST_CAP_GB if cap is None else float(cap))
    return BatchedEngineConfig(engine=engine,
                               stage_batches=stage_batches,
                               max_clips_per_wave=None if wave is None else int(wave),
                               retry_singly=bool(block.get("retry_singly", True)),
                               prefetch_decode=bool(block.get("prefetch_decode", False)),
                               pipeline_waves=bool(block.get("pipeline_waves", False)),
                               pipeline_depth=int(block.get("pipeline_depth", 2)),
                               pipeline_resources=pipeline_resources(bench),
                               memory_ledger=memory_ledger(bench),
                               card_admission=card_admission(bench),
                               oom_retry=oom_retry(bench),
                               prewarm=prewarm_stages(bench))


def pipeline_resources(bench: DictConfig) -> dict[str, str]:
    """`bench.batched.pipeline_resources` as `stage -> resource class`; `{}` without the key.

    Raises:
        ValueError: On a key that is not a known stage (`STAGE_GPU`), or on `shape` placed outside the `gpu` class
            while `bench.shape.backend` is not `sam3d`: only SAM 3D runs in a worker subprocess, and an in-process
            backend (TRELLIS) off the `gpu` class would run CUDA beside the other in-process stages.
    """
    raw = cast("dict[str, Any]", OmegaConf.to_container(bench.batched.get("pipeline_resources") or {}, resolve=True))
    resources = {str(stage): str(cls) for stage, cls in raw.items()}
    unknown = sorted(stage for stage in resources if stage not in STAGE_GPU)
    if unknown:
        raise ValueError(f"bench.batched.pipeline_resources has unknown stage keys {unknown}; valid keys: "
                         f"{sorted(STAGE_GPU)}")
    backend = str((bench.get("shape") or {}).get("backend", ""))
    if resources.get("shape", "gpu") != "gpu" and backend != "sam3d":
        raise ValueError(f"bench.batched.pipeline_resources puts shape in {resources['shape']!r}, but "
                         f"bench.shape.backend {backend!r} runs in-process: only sam3d runs in a worker subprocess")
    return resources


# Stages whose `pipeline_resources` class may be a worker subprocess (`*_worker`): SAM 3D for shape, HaWoR for hands.
_SUBPROCESS_STAGES = frozenset({"shape", "hands"})


def memory_ledger(bench: DictConfig) -> MemoryLedgerConfig | None:
    """`bench.batched.memory_ledger`: None when the key is absent or null (off), else `{resident_gb: <float>}`
    with an optional `costs_gb: {stage: <per-attempt GiB>}` that replaces those stages' declared costs
    (`MemoryLedgerConfig`).

    The ledger's capacity is the applied per-process cap (`FASTDO_GPU_MEM_GB`, else 12.0) minus `resident_gb`, checked
    here, at config load. It gates only in-process CUDA classes, telling them from worker subprocesses by the
    `_worker` suffix (`fastdo.engine.pipelined_runner.runs_cuda_in_process`), so a `*_worker` class on a stage that is
    not a subprocess stage (`shape`, `hands`) is refused: it would run CUDA in this process ungated. The ledger acts
    only while `pipeline_waves` is on; set without it, it is logged and ignored.

    Raises:
        ValueError: On a block that is not a mapping of `resident_gb` and optionally `costs_gb`; on `resident_gb` not
            a finite number >= 0 or not below the applied cap; on a `costs_gb` key outside `STAGE_GPU` or a value
            not a finite number >= 0; on a disabled cap (`FASTDO_GPU_MEM_GB` <= 0); on a `*_worker`
            class assigned to a stage other than `shape`/`hands`.
    """
    raw = bench.batched.get("memory_ledger")
    if raw is None:
        return None
    block = OmegaConf.to_container(raw, resolve=True) if isinstance(raw, DictConfig) else raw
    if not isinstance(block, dict) or "resident_gb" not in block or set(block) - {"resident_gb", "costs_gb"}:
        raise ValueError("bench.batched.memory_ledger must be null or {resident_gb: <float>, costs_gb: {stage: "
                         f"<float>}} (costs_gb optional), got {block!r}")
    costs = block.get("costs_gb") or {}
    ledger = MemoryLedgerConfig(resident_gb=block["resident_gb"], costs_gb=costs)
    ledger.capacity_gb()
    misplaced = sorted(stage for stage, cls in pipeline_resources(bench).items()
                       if cls.endswith("_worker") and stage not in _SUBPROCESS_STAGES)
    if misplaced:
        raise ValueError(f"bench.batched.memory_ledger gates in-process classes only and reads a `*_worker` class as a "
                         f"subprocess, but pipeline_resources puts the in-process stage(s) {misplaced} in one: rename "
                         "the class")
    if not bool(bench.batched.get("pipeline_waves", False)):
        logger.warning("bench.batched.memory_ledger is set but pipeline_waves is off: the sequential runner runs one "
                       "GPU attempt at a time and ignores the ledger")
    return ledger


def _block(bench: DictConfig, key: str) -> Any:
    raw = bench.batched.get(key)
    return OmegaConf.to_container(raw, resolve=True) if isinstance(raw, (DictConfig, ListConfig)) else raw


def card_admission(bench: DictConfig) -> CardAdmissionConfig | None:
    """`bench.batched.card_admission` (`fastdo.engine.card_admission`): None when the key is absent or null
    (off), else `{min_free_gb: {<stage or class>: GiB}, poll_s: <s>, max_wait_s: <s>}` (`poll_s` and `max_wait_s`
    optional, 5 and 600).

    A `min_free_gb` key names a GPU stage (`STAGE_GPU` true) or the resource class of one (`pipeline_resources`, else
    `gpu`): a CPU-only stage never allocates on the card, and a key naming nothing would silently gate nothing.

    Raises:
        ValueError: On a block that is not such a mapping; on a stray key; on a `min_free_gb` key that is neither a GPU
            stage nor a GPU stage's class, or a value not a finite number >= 0; on `poll_s` not a finite number > 0
            or `max_wait_s` not a finite number >= 0.
    """
    block = _block(bench, "card_admission")
    if block is None:
        return None
    keys = {"min_free_gb", "poll_s", "max_wait_s", "reclaim_cache"}
    if not isinstance(block, dict) or "min_free_gb" not in block or set(block) - keys:
        raise ValueError("bench.batched.card_admission must be null or {min_free_gb: {<stage or class>: <GiB>}, "
                         f"poll_s: <s>, max_wait_s: <s>, reclaim_cache: <bool>}} (all but min_free_gb optional), "
                         f"got {block!r}")
    need = block["min_free_gb"]
    if not isinstance(need, dict):          # a ValueError, as every other bench.batched refusal
        raise ValueError(  # noqa: TRY004
            f"bench.batched.card_admission.min_free_gb must map stage or class -> GiB, got {need!r}")
    resources = pipeline_resources(bench)
    gpu_stages = {stage for stage, gpu in STAGE_GPU.items() if gpu}
    valid = gpu_stages | {resources.get(stage) or "gpu" for stage in gpu_stages}
    unknown = sorted(str(key) for key in need if key not in valid)
    if unknown:
        raise ValueError(f"bench.batched.card_admission.min_free_gb has keys {unknown} that are neither a GPU stage "
                         f"nor a GPU stage's resource class; valid keys: {sorted(valid)}")
    return CardAdmissionConfig(min_free_gb=need, **{k: v for k, v in block.items() if k != "min_free_gb"})


def prewarm_stages(bench: DictConfig) -> tuple[str, ...]:
    """`bench.batched.prewarm` (`fastdo.engine.prewarm`): the stages whose worker-subprocess model the pipelined
    runner loads at run start; `()` when the key is absent, null or empty (today's first-use loads). Only `shape` and
    `hands` offer a hook today (the SAM 3D worker and the warm HaWoR driver); another listed stage is logged at run
    start and skipped. Read only while `pipeline_waves` is on (logged and ignored otherwise).

    Raises:
        ValueError: On a value that is not a list of distinct known stage names (`STAGE_GPU`).
    """
    raw = _block(bench, "prewarm")
    if raw is None:
        return ()
    if not isinstance(raw, list) or any(not isinstance(stage, str) for stage in raw) or len(set(raw)) != len(raw):
        raise ValueError(f"bench.batched.prewarm must be a list of distinct stage names, got {raw!r}")
    unknown = sorted(stage for stage in raw if stage not in STAGE_GPU)
    if unknown:
        raise ValueError(f"bench.batched.prewarm has unknown stages {unknown}; valid: {sorted(STAGE_GPU)}")
    if raw and not bool(bench.batched.get("pipeline_waves", False)):
        logger.warning("bench.batched.prewarm is set but pipeline_waves is off: the sequential runner loads on first "
                       "use and ignores it")
    return tuple(raw)


def oom_retry(bench: DictConfig) -> OomRetryConfig | None:
    """`bench.batched.oom_retry` (`fastdo.engine.card_admission`): None when the key is absent or null (off:
    today's single retries), else `{max_total_wait_s: <s>, backoff_s: [<initial s>, <max s>]}`.

    Raises:
        ValueError: On a block that is not exactly those two keys; on `max_total_wait_s` not a finite number > 0; on
            `backoff_s` not two finite numbers with 0 < initial <= max.
    """
    block = _block(bench, "oom_retry")
    if block is None:
        return None
    if not isinstance(block, dict) or set(block) != {"max_total_wait_s", "backoff_s"}:
        raise ValueError("bench.batched.oom_retry must be null or {max_total_wait_s: <s>, backoff_s: [<initial s>, "
                         f"<max s>]}}, got {block!r}")
    backoff = block["backoff_s"]
    if not isinstance(backoff, list):       # a ValueError, as every other bench.batched refusal
        raise ValueError(  # noqa: TRY004
            f"bench.batched.oom_retry.backoff_s must be [initial, max], got {backoff!r}")
    return OomRetryConfig(max_total_wait_s=block["max_total_wait_s"],
                          backoff_s=cast("tuple[float, float]", tuple(backoff)))


def build_batched_stages(cfg: DictConfig, fns: Mapping[str, StageFn], job_root: Path,
                         lock_path: Path) -> dict[str, StageFn | BatchedStage]:
    """The real workers of `build_stage_fns` as batched stages: encoder/geometry frame-batched, mask/tracks on slot
    pools (built from the same `bench` keys), hands batched when `bench.hands_source == "batched"`, solve batched
    (the two halves, `prepare_solve` with `driftqc_enabled` from `driftqc_stage.enabled`; `finish_solve` reads the
    applied range from the call's context; `solve_stage.depth_anchor_gate` is passed to the stage; each clip's
    source-domain profile via `profiled_solve_halves`); every other stage keeps its `StageFn`."""
    bench = cfg.bench
    out: dict[str, StageFn | BatchedStage] = dict(fns)
    for stage in ("encoder", "geometry"):
        worker = fns.get(stage)
        if isinstance(worker, BatchedWorker):
            out[stage] = FrameBatchedStage(worker)
        elif worker is not None:
            logger.warning("bench.batched: the %s worker (%s) is not a BatchedWorker; it falls back to a plain StageFn "
                           "and runs one clip per batch", stage, type(worker).__name__)
    if "mask" in fns:
        compile_mode = mask_compile(cfg)
        out["mask"] = ConcurrentClipStage(lambda: make_mask_worker(str(bench.mask_variant), mask_compile=compile_mode))
    if "tracks" in fns:
        out["tracks"] = ConcurrentClipStage(lambda: make_tracks_worker(n_queries=int(bench.n_track_queries)))
    if "hands" in fns and str(bench.hands_source) == "batched":
        out["hands"] = HandsBatchStage(hands_config_from_cfg(cfg), job_root, lock_path, fns["hands"])
    if "solve" in fns:
        solve_cfg = SolveStageConfig.from_mapping(cfg.solve_stage)
        # each clip's source-domain profile (fastdo.engine.source_profile), as in the single-stream worker
        prepare, finish = profiled_solve_halves(
            solve_cfg, ContactStageConfig.from_mapping(cfg.get("contact_stage")).enabled, source_profile_config(cfg),
            driftqc_enabled=driftqc_enabled(cfg))
        out["solve"] = SolveBatchStage(prepare, finish, solve_cfg.solver_config(),
                                       depth_anchor_gate=solve_cfg.depth_anchor_gate,
                                       branch_select=solve_cfg.branch_select)
    return out


def check_single_stream_fns(fns: Mapping[str, object]) -> None:
    """Raises `TypeError` naming the first stage whose value is not callable as a `StageFn`."""
    for stage, fn in fns.items():
        if not callable(fn):
            raise TypeError(f"stage {stage!r} is a {type(fn).__name__}, not a StageFn: single-stream runs take plain "
                            "StageFns; set bench.batched.enabled=true to run a BatchedStage")


def execution_provenance(manifest: ClipManifest, run: BatchRunReport | None) -> dict[str, Any]:
    """The yield report's execution fields; the batched ones re-check the manifest against the batch log."""
    if run is None:
        return {"execution": "single_stream", "batch_log": None, "batch_run": None, "apportionment": None,
                "orchestration_overhead_seconds": None}
    check = check_apportionment(manifest, read_batch_log(run.batch_log))
    return {"execution": "batched", "batch_log": str(run.batch_log),
            "batch_run": asdict(run) | {"batch_log": str(run.batch_log)},
            "apportionment": {"unattributed_ns": check.unattributed_ns},
            "orchestration_overhead_seconds": run.overhead_ns / 1e9}


def _files(root: Path) -> set[str]:
    found: set[str] = set()
    for path in Path(root).rglob("*"):
        rel = path.relative_to(root)
        if (path.is_file() and path.suffix in (".npz", ".json") and path.name not in EXCLUDED_FILES
                and not EXCLUDED_DIRS & set(rel.parts[:-1])):
            found.add(rel.as_posix())
    return found


def _compare_arrays(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    entry: dict[str, Any] = {"dtype_match": x.dtype == y.dtype, "shape_match": x.shape == y.shape}
    if not (entry["dtype_match"] and entry["shape_match"]):
        return entry | {"equal": False}
    if x.dtype.kind in "fc":
        nan_x, nan_y = np.isnan(x), np.isnan(y)
        both = ~(nan_x | nan_y)
        a, b = x[both], y[both]
        diff = np.where(a == b, 0.0, np.abs(a.astype(np.complex128 if x.dtype.kind == "c" else np.float64) - b))
        max_abs = float(diff.max()) if diff.size else 0.0
        nan_match = bool(np.array_equal(nan_x, nan_y))
        return entry | {"nan_positions_match": nan_match, "max_abs_diff": max_abs,
                        "equal": nan_match and max_abs == 0.0}
    if x.dtype.kind in "biu":
        fraction = float(np.count_nonzero(x != y)) / x.size if x.size else 0.0
        return entry | {"mismatch_fraction": fraction, "equal": fraction == 0.0}
    return entry | {"equal": bool(np.array_equal(x, y))}


def _strip(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _strip(v) for k, v in value.items() if k not in VOLATILE_KEYS}
    if isinstance(value, list):
        return [_strip(v) for v in value]
    return value


def compare_artifact_dirs(a: Path, b: Path) -> dict[str, Any]:
    """Per-file, per-array comparison of two artifact roots (module docstring); `n_mismatched` counts the arrays and
    JSON files that are not exactly equal, and missing files are listed per side."""
    files_a, files_b = _files(Path(a)), _files(Path(b))
    arrays: dict[str, dict[str, Any]] = {}
    documents: dict[str, dict[str, bool]] = {}
    n_mismatched = 0
    for rel in sorted(files_a & files_b):
        if rel.endswith(".json"):
            equal = _strip(json.loads((Path(a) / rel).read_text())) == _strip(json.loads((Path(b) / rel).read_text()))
            documents[rel] = {"equal": equal}
            n_mismatched += not equal
            continue
        with np.load(Path(a) / rel, allow_pickle=False) as da, np.load(Path(b) / rel, allow_pickle=False) as db:
            entries: dict[str, Any] = {}
            for key in sorted(set(da.files) | set(db.files)):
                if key not in da.files or key not in db.files:
                    entries[key] = {"missing_in": "a" if key not in da.files else "b", "equal": False}
                else:
                    entries[key] = _compare_arrays(np.asarray(da[key]), np.asarray(db[key]))
                n_mismatched += not entries[key]["equal"]
        arrays[rel] = entries
    return {"a": str(a), "b": str(b), "n_files": len(files_a & files_b), "missing_in_a": sorted(files_b - files_a),
            "missing_in_b": sorted(files_a - files_b), "n_mismatched": n_mismatched, "arrays": arrays,
            "json": documents}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m fastdo.bench.bench_batched")
    commands = parser.add_subparsers(dest="command", required=True)
    compare = commands.add_parser("compare", help="compare two artifact roots")
    compare.add_argument("a", type=Path)
    compare.add_argument("b", type=Path)
    compare.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = compare_artifact_dirs(args.a, args.b)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    logger.info("compared %d files: %d mismatched, %d/%d missing; wrote %s", report["n_files"], report["n_mismatched"],
                len(report["missing_in_a"]), len(report["missing_in_b"]), args.out)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
