"""The batched `hands` stage: a manifest's eligible clips through `run_hands_batched`, one model load per call.

Usage, with `fns` the stage workers and `cfg` the engine config:
1. `run_manifest(manifest, replace(cfg, stages=cfg.stages[:cfg.stages.index("hands")]), fns)` -- the RUN'S
   OWN stage list, not `STAGE_NAMES`: S2 inserts `prompt` before `mask` via `workers_prompt.with_prompt`,
   and `prompt` is not in `STAGE_NAMES`, so slicing that tuple would silently skip the prompt stage and
   leave `mask` with no `mask_prompt.npz` to read;
2. `run_hands_stage(manifest, cfg, hands_cfg, job_root)`;
3. `run_manifest(manifest, cfg, fns)` with `skip_done=True` and `fns["hands"] = make_hands_worker(hands_cfg)`. This
   retries failed clips one at a time and handles clips that have a precomputed `all_hand_meshes.npz`.

A clip is eligible when `hands` is pending, `decode` and `intrinsics` are both done, and it has no precomputed npz. Each
call holds `gpu_lock(<artifact_root>/.gpu.lock)`. When a call raises `RuntimeError` or `TimeoutError`, the clips its
partial `response.json` reports done, with a valid npz, are kept. Every other clip of that call is marked failed with
the exception text, and the stage moves on to the next call.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

from fastdo.engine.manifest import ClipManifest
from fastdo.engine.orchestrator import EngineConfig, gpu_lock
from fastdo.engine.worker_support import DECODE_FILE
from fastdo.engine.workers_hands import HANDS_OUTPUT, HAWOR_FILE, write_hands_artifact
from fastdo.geometry.intrinsics import read_intrinsics
from fastdo.hands.hawor_batched import (
    HandsCallReport,
    HandsClipInput,
    HandsClipResult,
    HandsConfig,
    WarmHandsDriver,
    read_response,
    run_hands_batched,
    verify_clip_result,
)

__all__ = ["HANDS_OUTPUT", "INTRINSICS_MISSING", "eligible_clips", "hands_focal_length", "run_hands_call",
           "run_hands_stage"]

logger = logging.getLogger(__name__)

STAGE = "hands"
# `make_hands_worker`'s text (`fastdo.engine.workers_hands`), so both modes record the same `hands_error`.
INTRINSICS_MISSING = "hands_worker needs intrinsics.npz; run the intrinsics stage first"


def hands_focal_length(artifact_dir: Path) -> float:
    """HaWoR's single focal length `K[0, 0]` from `<artifact_dir>/intrinsics.npz`.

    Raises:
        ValueError: `INTRINSICS_MISSING` when the file does not exist (a parity test pins it to the single-stream text).
    """
    try:
        K, _ = read_intrinsics(Path(artifact_dir))
    except FileNotFoundError:
        raise ValueError(INTRINSICS_MISSING) from None
    return float(K[0, 0])


def eligible_clips(manifest: ClipManifest, engine_cfg: EngineConfig) -> list[str]:
    """Clip ids the batched stage should run, in manifest order."""
    return [clip_id for clip_id in manifest.pending(STAGE)
            if manifest.status(clip_id, "decode").state == "done"
            and manifest.status(clip_id, "intrinsics").state == "done"
            and not (engine_cfg.artifact_root / clip_id / HAWOR_FILE).is_file()]


def _salvage(clips: Sequence[HandsClipInput], job_dir: Path, exc: Exception) -> HandsCallReport:
    message = f"{type(exc).__name__}: {exc}"
    partial = None
    response = job_dir / "response.json"
    if response.is_file():
        try:
            partial = read_response(response, returncode=-1)
        except (OSError, ValueError, KeyError, TypeError) as read_exc:
            logger.warning("unreadable partial response %s: %s", response, read_exc)
    done = {} if partial is None else {r.clip_id: r for r in partial.clips if r.status == "done"}
    results = []
    for clip in clips:
        kept = verify_clip_result(clip, done[clip.clip_id]) if clip.clip_id in done else None
        results.append(kept if kept is not None and kept.status == "done"
                       else HandsClipResult(clip.clip_id, "failed", message, 0, {}))
    logger.warning("hands call in %s failed (%s); kept %d of %d clips", job_dir, message,
                   sum(r.status == "done" for r in results), len(clips))
    if partial is None:
        return HandsCallReport(-1, 0.0, 0.0, 0.0, 0.0, tuple(results))
    return HandsCallReport(-1, partial.load_seconds, partial.warmup_seconds, partial.compute_seconds,
                           partial.peak_cuda_gb, tuple(results), batch_invariant=partial.batch_invariant,
                           deterministic=partial.deterministic)


def run_hands_call(clips: Sequence[HandsClipInput], hands_cfg: HandsConfig, job_dir: Path,
                   lock_path: Path, warm: WarmHandsDriver | None = None) -> tuple[HandsCallReport, dict[str, str]]:
    """One `run_hands_batched` call under `gpu_lock(lock_path)` (salvaged on `RuntimeError`/`TimeoutError`), then
    `hands.npz` for each done clip. Returns the report and `clip_id -> error` for every clip left without `hands.npz`.
    With `warm` the call is a request to that warm server; the lock is still held for this call only."""
    try:
        with gpu_lock(lock_path):
            report = (run_hands_batched(clips, hands_cfg, job_dir) if warm is None     # the non-warm call as before
                      else run_hands_batched(clips, hands_cfg, job_dir, warm=warm))
    except (RuntimeError, TimeoutError) as exc:
        report = _salvage(clips, job_dir, exc)
    by_id = {result.clip_id: result for result in report.clips}
    errors: dict[str, str] = {}
    for clip in clips:
        result = by_id.get(clip.clip_id)
        if result is None or result.status != "done":
            errors[clip.clip_id] = "no result for this clip" if result is None else result.error
            continue
        try:
            write_hands_artifact(clip.out_npz, clip.out_npz.parent)
        except Exception as exc:  # noqa: BLE001 - one clip's artifact failure must not stop the others
            errors[clip.clip_id] = f"{type(exc).__name__}: {exc}"
    return report, errors


def _run_call(manifest: ClipManifest, engine_cfg: EngineConfig, hands_cfg: HandsConfig,
              clips: Sequence[HandsClipInput], job_dir: Path) -> HandsCallReport:
    report, errors = run_hands_call(clips, hands_cfg, job_dir, engine_cfg.artifact_root / ".gpu.lock")
    shared_seconds = (report.load_seconds + report.warmup_seconds) / len(clips)
    by_id = {result.clip_id: result for result in report.clips}
    for clip in clips:
        if clip.clip_id in errors:
            manifest.mark(clip.clip_id, STAGE, "failed", 0.0, error=errors[clip.clip_id])
            continue
        total = float(by_id[clip.clip_id].seconds.get("total", 0.0))
        manifest.mark(clip.clip_id, STAGE, "done", total + shared_seconds)
    return report


def run_hands_stage(manifest: ClipManifest, engine_cfg: EngineConfig, hands_cfg: HandsConfig,
                    job_root: Path) -> HandsCallReport | None:
    """Runs every eligible clip in calls of at most `hands_cfg.max_clips_per_call` clips (job dirs
    `job_root / call_0000`, ...). A done clip gets `hands.npz` and `seconds = total + (load + warm-up) / clips in call`.
    Saves the manifest after each call when `engine_cfg.manifest_path` is set.

    Returns:
        The last call's report, or None when no clip was eligible.
    """
    eligible = eligible_clips(manifest, engine_cfg)
    last: HandsCallReport | None = None
    for call_index, start in enumerate(range(0, len(eligible), hands_cfg.max_clips_per_call)):
        clips = []
        for clip_id in eligible[start:start + hands_cfg.max_clips_per_call]:
            artifact_dir = engine_cfg.artifact_root / clip_id
            try:
                K, _ = read_intrinsics(artifact_dir)
            except (OSError, ValueError, KeyError) as exc:
                manifest.mark(clip_id, STAGE, "failed", 0.0, error=f"{type(exc).__name__}: {exc}")
                continue
            clips.append(HandsClipInput(clip_id, artifact_dir / DECODE_FILE, float(K[0, 0]),
                                        artifact_dir / HANDS_OUTPUT))
        if clips:
            last = _run_call(manifest, engine_cfg, hands_cfg, clips, Path(job_root) / f"call_{call_index:04d}")
        if engine_cfg.manifest_path is not None:
            manifest.save(engine_cfg.manifest_path)
    return last
