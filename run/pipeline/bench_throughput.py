"""End-to-end throughput benchmark and yield report: whisking/DexYCB fallback set, 100DOH train.

Builds a `ClipManifest` from the benchmark clip list, runs `run_manifest` through every registered stage worker and
writes `<bench.out_dir>/manifest.parquet` and `<bench.out_dir>/yield_report.json`. Each clip's artifacts and its own
`profile.json` (one `Profiler` per clip, `run_clip`) land in `<bench.artifact_root>/<clip_id>/`.

Clips (`fastdo.bench.bench_clips`): the shipped whisking demo `bench.whisking_copies` times, the DexYCB ids listed in
`bench.dexycb_sequences` (under `dexycb.root`) and the `official_split == "train"` entries of
`<bench.doh_root>/manifest.json`. The 100DOH test-split clips are reserved for the final evaluation: they are dropped
before any path is built, so they are never opened, and no clip file is copied anywhere. Each manifest row's `source`
records its split (`"100doh:train"`). Without a 100DOH train clip (default `bench.doh_root: null`) the script logs that
raw-stream throughput (report key `s2_raw_stream`) is unmeasured and runs the fallback set. Raw-stream throughput also
stays unmeasured, with a WARNING naming the reason, whenever the interaction gate did not run on the clips (unbuilt,
or excluded from `bench.stages`), whatever stages ran, and unless every 100DOH clip finishes every registered stage
it ran. It never searches the filesystem for datasets.

Stages: `bench.stages`, run in `STAGE_NAMES` order (the opt-in driftqc stage is accepted only with
`driftqc_stage.enabled`, right after tracks: `known_stages`); `fastdo.bench.bench_stages.build_stage_fns` builds the workers and
their status labels. Registered workers are decode (ffmpeg; JPEG sequences for DexYCB), encoder, mask (SAM 2.1
`bench.mask_variant`), shape (`bench.shape`; `bench.dexycb_shape: template` gives DexYCB clips their GT YCB mesh,
reported as borrowed, for calibration runs), intrinsics (MoGe-2, or DexYCB's `camera.json`), hands (`bench.hands_source`:
batched HaWoR, or the demo's precomputed npz, reported as borrowed), geometry, tracks, solve (the pose solver on
`solve_stage.device`) and qc (only with `qc_stage.rule_path`, a saved AcceptanceRule). With real workers, qc in
`bench.stages` without a rule raises before anything runs: the qc stage has no accept-all default, so the benchmark never
reports a silently QC-less yield. Two stages have no worker and their columns stay pending at zero cost:
- gate: without `gate_stage.model_path` (the gate bundle), accept-all through the persisted flag `gate_status:
  "unbuilt"`, so the yield is QC-only.
- retarget: opt-in (`retarget_stage.enabled`; it then runs on qc-accepted clips, CPU), else pending.

`clips_per_gpu_hour` covers only the stage columns that ran, so `stage_status`, `stage_states`, `unmeasured_stages` and
`stages_done_on_every_clip` are persisted next to it, and a WARNING names the gap whenever the run is not end to end.
With `bench.batched.enabled` the stages run in cross-clip batches through `run_manifest_batched`
(`fastdo.bench.bench_batched`), each batch's wall apportioned to its clips by frames; the report records `execution`.
Keyframe prompts and DexYCB calibration are written before the run and not timed. `bench.overwrite` first removes the
benchmark clips' own artifact directories, so no stale artifact is read.

GPU run. PRECONDITION: no other job on the card. FASTDO_GPU_MEM_GB=32 is required: the default 12 GiB per-process
cap is below the default config's stage budgets, and SAM 3D alone needs 19 GiB. On a host shared with other GPU jobs,
prefix the command with `flock "$FASTDO_GPU_TOOLS_DIR/gpu.lock"`. The default `bench.stages` leaves the encoder
out because `transformers` is not installed (a failed encoder would stop every clip at stage 2, and its only
consumer, the gate, is unbuilt); requesting it without `transformers` fails fast. The default `bench.stages`
includes qc, which raises without a rule: until one is saved (`run/pipeline/calibrate_acceptance.py`) the command
excludes qc explicitly, and afterwards it passes `qc_stage.rule_path=<the saved rule>` with the default stages
instead. The command exits 1 when every clip failed a stage (the report then times no finished clip).
  cd ${FASTDO_ROOT} && env FASTDO_GPU_MEM_GB=32 FASTDO_CPU_THREADS=2 \
    OMP_NUM_THREADS=2 FASTDO_SAM3D_PYTHON=/path/to/sam3d/bin/python \
    uv run --extra dev --extra frontend python run/pipeline/bench_throughput.py paths.root=${FASTDO_ROOT} \
    'bench.stages=[decode,gate,mask,shape,intrinsics,hands,geometry,tracks,solve,retarget]'
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import os
import shutil
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from fastdo import hostpaths
from fastdo.bench import YieldReport, compute_yield_report
from fastdo.bench.bench_batched import (
    batched_engine_config,
    build_batched_stages,
    check_single_stream_fns,
    execution_provenance,
)
from fastdo.bench.bench_clips import (
    RAW_STREAM_SOURCES,
    SOURCES,
    BenchClip,
    dexycb_clips,
    dexycb_intrinsics,
    dexycb_prompt,
    doh_clips,
    keyframe_row,
    whisking_clips,
    whisking_prompt,
    windowed_clips_from_inventory,
    write_camera_json,
)
from fastdo.bench.bench_stages import (
    NOT_REGISTERED,
    build_stage_fns,
    driftqc_enabled,
    source_profile_config,
)
from fastdo.bench.bench_stages import resolve_path as _resolve
from fastdo.bench.bench_stages import whisking_paths as _whisking_paths
from fastdo.bench.gpu_budget import apply_cpu_budget, apply_gpu_budget
from fastdo.bench.hoi4d_bench import SOURCE as HOI4D_SOURCE
from fastdo.bench.hoi4d_bench import HOI4DWindow, hoi4d_clips, hoi4d_prompt, load_windows
from fastdo.bench.oakink2_bench import SOURCES as OAKINK2_SOURCES
from fastdo.bench.oakink2_bench import OakInk2Window, oakink2_clips
from fastdo.bench.oakink2_bench import load_windows as load_oakink2_windows
from fastdo.bench.yield_report import (
    GATE_UNBUILT,
    clip_rate_agreement,
    clip_total_seconds,
    read_qc_verdicts,
    stage_order,
    stage_state_counts,
    write_yield_report,
)
from fastdo.engine import (
    STAGE_NAMES,
    ClipManifest,
    EngineConfig,
    StageFn,
    run_manifest,
    with_driftqc,
    write_mask_prompt,
)
from fastdo.engine.batched_runner import BatchRunReport, applied_cap_gb, run_manifest_batched
from fastdo.engine.batched_stages import BatchedStage, HandsBatchStage
from fastdo.engine.decode_index import write_index_audit
from fastdo.engine.source_domain import (
    DOMAIN_FILE,
    SourceDomain,
    read_domain_record,
    write_domain_record,
)
from fastdo.engine.source_profile import check_run_profile
from fastdo.engine.workers_contact import ContactStageConfig, with_contact
from fastdo.engine.workers_frames import timebase_flags
from fastdo.engine.workers_gate import GATE_REJECTED_PREFIX, gate_outcome_counts
from fastdo.engine.workers_prompt import (
    PROMPT_REFUSED_PREFIX,
    PromptStageConfig,
    prompt_outcome_counts,
    with_prompt,
)
from fastdo.engine.workers_solve import SolveStageConfig
from fastdo.io.datasets.hoi4d import HOI4DReader
from fastdo.io.datasets.oakink2 import OakInk2Reader
from fastdo.io.video import count_decoded_frames, screened_frame_count

logger = logging.getLogger(__name__)

MANIFEST_FILE = "manifest.parquet"
REPORT_FILE = "yield_report.json"
INJECTED = "injected"
EXCLUDED = "excluded (bench.stages)"
SUBMODULE_HINT = ("the whisking demo ships in the DO AS I DO baseline (third_party/do-as-i-do): run "
                  "`bash scripts/fetch_baseline.sh` in this checkout, or set whisking.demo_root to a DO AS I "
                  "DO checkout's reconstruction/whisking")
GATE_BUILT = "built"
GATE_COST_ONLY = "gate-cost only (contact-centred 100DOH train clips; not S2)"
STAGE_PACKAGES: dict[str, str] = {"encoder": "transformers"}    # real stage -> Python package its model loads through
RATE_DEFINITION = ("3600 / (total_seconds / n_clips), total_seconds summing every clip's {stage}_seconds manifest "
                   "columns in every state; for one clip it is ProfileReport.clips_per_gpu_hour")


def contact_enabled(cfg: DictConfig) -> bool:
    """`ContactStageConfig.from_mapping(cfg.contact_stage).enabled` (the along-ray contact correction); a config without
    the block is disabled, and an invalid block raises here rather than mid-run.

    Raises:
        ValueError: On an invalid `contact_stage` block (unknown key, bad cell, hands or grid).
    """
    return ContactStageConfig.from_mapping(cfg.get("contact_stage") or {}).enabled


def known_stages(cfg: DictConfig) -> tuple[str, ...]:
    """`STAGE_NAMES`, plus each opt-in stage the config enables: `driftqc` after `tracks` (`with_driftqc`,
    `driftqc_stage.enabled`) and `contact` after `solve` (`with_contact`, `contact_stage.enabled`)."""
    stages = with_driftqc(STAGE_NAMES) if driftqc_enabled(cfg) else STAGE_NAMES
    if PromptStageConfig.from_mapping(cfg.get("prompt_stage") or {}).enabled:
        stages = with_prompt(stages)
    return with_contact(stages) if contact_enabled(cfg) else stages


def pipeline_stages(cfg: DictConfig) -> tuple[str, ...]:
    """`bench.stages` in `known_stages` order (null: every known stage). `driftqc` is accepted only when
    `driftqc_stage.enabled` (the stage is opt-in).

    Raises:
        ValueError: On a stage name outside `known_stages`, or `driftqc` with `driftqc_stage.enabled` false.
    """
    known = known_stages(cfg)
    requested = cfg.bench.get("stages")
    if requested is None:
        return known
    names = {str(stage) for stage in requested}
    if "prompt" in names and "prompt" not in known:
        raise ValueError("bench.stages lists prompt, but prompt_stage.enabled is false: the prompt stage is opt-in"
                         "; set prompt_stage.enabled=true with a min_score, or drop prompt from bench.stages")
    if "contact" in names and "contact" not in known:
        raise ValueError("bench.stages lists contact, but contact_stage.enabled is false: the contact stage is opt-in"
                         "; set contact_stage.enabled=true, or drop contact from bench.stages")
    if "driftqc" in names and "driftqc" not in known:
        raise ValueError("bench.stages lists driftqc, but driftqc_stage.enabled is false: the driftqc stage is opt-in"
                         "; set driftqc_stage.enabled=true with its paths, or drop driftqc from bench.stages")
    unknown = sorted(names - set(known))
    if unknown:
        raise ValueError(f"bench.stages has unknown stages {unknown}; known stages: {list(known)}")
    return tuple(stage for stage in known if stage in names)


def _windows_frame_counter(route: str | None, count_frames: Callable[[Path], int]) -> Callable[[Path], int]:
    """`D_v`'s numerator for WINDOWED clips: the decoded count by default, the packet table when opted into.

    Counting decoded frames is O(video) -- 13.55 s for a 203 s file, up to 235.7 s for a 1033 s one -- and
    it is paid while PLACING windows, before anything is known about whether a window is worth processing.
    Over the 21-video Ego4D draw that is over an hour of fixed cost on the front of every run.

    The container count is acceptable for a source whose `nb_frames` has been validated against decoded
    counts (exact on all 21 Ego4D videos), and the trim screen makes it safe per file rather than per
    corpus. `screened_frame_count` is that route, and it falls back to decoding on a file with container
    evidence of trimming, so the screen is one-sided.

    The default stays `decoded`: the packet-table route is a permission a run opts into, not one it inherits from a
    corpus it may not be running on. An unknown route RAISES rather than falling back, because a silent fallback
    would either cost the hour the run meant to save or take a permission it never asked for.

    Resolved by module attribute so a test can substitute the counter and observe which route ran.
    """
    route = "decoded" if route is None else str(route)
    if route == "decoded":
        return count_frames
    if route == "screened":
        return screened_frame_count
    raise ValueError(f"bench.windows_frame_count must be 'decoded' or 'screened', got {route!r}")


def _selected_splits(windows: list[BenchClip], splits: Sequence[str] | None,
                     max_per_video: int | None) -> list[BenchClip]:
    """Keep the named splits, and REFUSE to spend TEST on a truncated run.

    `bench.windows_max_per_video` is **null for the TEST run**. Read the other way round, a run that SETS it is
    by construction not the TEST run, so it has no business opening TEST content: TEST is one-shot and executes
    automatically once CALIB is clean, so there is no round-trip in which anyone notices a pilot already spent it.

    A rule that lives only in prose does not fail loudly. A bounded pilot over
    the real 21-video Ego4D inventory selects **24 TEST windows against 18 CALIB** ones, and nothing in
    the runner would have said so.

    `windows_splits` (null = every split) is how a pilot says which half it is running, and it is the
    supported way to satisfy the guard rather than a way around it: the guard fires on what was SELECTED.
    """
    if splits is not None:
        keep = {str(s) for s in splits}
        windows = [c for c in windows if c.split in keep]
    if max_per_video is not None:
        spent = sorted({c.clip_id for c in windows if c.split == "TEST"})
        if spent:
            raise ValueError(
                f"a truncated run (bench.windows_max_per_video={max_per_video}) selected {len(spent)} TEST "
                f"windows, e.g. {spent[:3]}. S2 Freeze-1 requires max_per_video to be null for the TEST "
                f"run, so a truncated run must not open TEST content. Set bench.windows_splits=[CALIB] for "
                f"a pilot, or max_per_video=null for the freeze's own TEST run.")
    return windows


def _selected_ids(windows: list[BenchClip], ids: Sequence[str] | None) -> list[BenchClip]:
    """Keep only the named window ids (`bench.windows_ids`; null = all). Every id must exist among the selected
    windows, and a run that names ids is a hand-picked study run, so it may not select a TEST window."""
    if ids is None:
        return windows
    wanted = [str(i) for i in ids]
    by_id = {c.clip_id: c for c in windows}
    missing = [i for i in wanted if i not in by_id]
    if missing:
        raise ValueError(f"bench.windows_ids names {len(missing)} windows that are not selected, e.g. {missing[:3]}")
    kept = [by_id[i] for i in dict.fromkeys(wanted)]
    if any(c.split == "TEST" for c in kept):
        raise ValueError("bench.windows_ids selects a TEST window; a hand-picked run must not open TEST content")
    return kept


def build_clips(cfg: DictConfig, root: Path,
                count_frames: Callable[[Path], int] = count_decoded_frames) -> list[BenchClip]:
    """The benchmark clip list: whisking copies, then the configured DexYCB ids, then the 100DOH clips.

    Raises:
        ValueError: If the list is empty or repeats a clip id.
    """
    bench = cfg.bench
    clips = whisking_clips(_whisking_paths(cfg, root)["video"], int(cfg.whisking.n_frames_full),
                           int(bench.whisking_copies))
    sequences = [str(seq) for seq in (bench.get("dexycb_sequences") or [])]
    if sequences:
        clips += dexycb_clips(_resolve(root, cfg.dexycb.root), sequences)
    doh_root = bench.get("doh_root")
    clips += doh_clips(None if doh_root is None else _resolve(root, doh_root), count_frames=count_frames)
    # The raw-stream measure's own input: 10 s windows of long EK-100/Ego4D videos. Without this the
    # windowed builder is unreachable from the registered runner, which is how two earlier study
    # runners shipped "done" but unrunnable.
    inventory = bench.get("windows_inventory")
    if inventory is not None:
        windows = windowed_clips_from_inventory(_resolve(root, inventory), root=root,
                                                max_per_video=bench.get("windows_max_per_video"),
                                                count_frames=_windows_frame_counter(
                                                    bench.get("windows_frame_count"), count_frames))
        clips += _selected_ids(_selected_splits(windows, bench.get("windows_splits"),
                                                bench.get("windows_max_per_video")), bench.get("windows_ids"))
    windows = _hoi4d_windows(cfg, root)
    if windows:
        clips += hoi4d_clips(HOI4DReader(_resolve(root, cfg.hoi4d.root)), windows)
    oak = _oakink2_windows(cfg, root)
    if oak:
        clips += oakink2_clips(OakInk2Reader(_resolve(root, cfg.oakink2.root)), oak)
    ids = [clip.clip_id for clip in clips]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError(f"the benchmark needs at least one clip and distinct clip ids, got {ids}")
    return clips


def _oakink2_windows(cfg: DictConfig, root: Path) -> list[OakInk2Window]:
    """The `bench.oakink2_windows` file's windows, `[]` when unset."""
    path = OmegaConf.select(cfg, "bench.oakink2_windows", default=None)
    if path is None:
        return []
    data = json.loads(_resolve(root, path).read_text())
    return load_oakink2_windows(data["windows"] if isinstance(data, dict) else data)


def _hoi4d_windows(cfg: DictConfig, root: Path) -> list[HOI4DWindow]:
    """The `bench.hoi4d_windows` file's windows, `[]` when unset."""
    path = OmegaConf.select(cfg, "bench.hoi4d_windows", default=None)
    if path is None:
        return []
    data = json.loads(_resolve(root, path).read_text())
    return load_windows(data["windows"] if isinstance(data, dict) else data)


# A stage that REFUSES raises, and `run_clip` marks it "failed" with `str(exc)` -- the manifest has no separate
# state -- so a refusal is told from an error only by the message's stable prefix. A refusal
# FINISHES a window and must not count as unfinished, or one declined window reports the whole run
# unmeasured. Keyed by stage, because only these stages refuse; any other "failed" is a real error.
REFUSAL_PREFIXES: dict[str, str] = {"gate": GATE_REJECTED_PREFIX, "prompt": PROMPT_REFUSED_PREFIX}


def s2_status(cfg: DictConfig, clips: list[BenchClip], manifest: ClipManifest | None = None,
              measured: Sequence[str] = (), gate_status: str = GATE_UNBUILT) -> str:
    """How raw-stream throughput (on 100DOH's train split) stands. "unmeasured..." without 100DOH clips. After the run
    it is "measured..." only when the interaction gate ran on the clips (`gate_status` is `GATE_BUILT` and "gate" is in
    `measured`, the registered stages the run executed) and every 100DOH clip finished every stage of `measured`.
    Otherwise it is "unmeasured...", naming the reason: the gate unbuilt or excluded by `bench.stages`, whatever
    stages ran, and the first unfinished clip. Raw-stream throughput is never reported without the gate. A
    RECORDED REFUSAL finishes a clip -- a gate rejection or a prompt refusal, per `REFUSAL_PREFIXES` -- so a
    declined window is not unfinished; with no measured stage after gate it is `GATE_COST_ONLY`."""
    doh = [clip.clip_id for clip in clips if clip.source in RAW_STREAM_SOURCES]
    if not doh:
        doh_root = cfg.bench.get("doh_root")
        where = "bench.doh_root is null" if doh_root is None else f"bench.doh_root {doh_root} holds no train-split clips"
        return (f"unmeasured: no raw-stream clips ({where}, and bench.windows_inventory adds none); "
                "set bench.doh_root to the 100DOH bundle root, or bench.windows_inventory to a window "
                "inventory of EK-100/Ego4D videos, to measure S2")
    if manifest is None:
        return f"pending: {len(doh)} raw-stream clips selected"
    stops = {c: next((s for s in measured if manifest.status(c, s).state != "done"), None) for c in doh}
    rejected = {c for c, s in stops.items()
                if s in REFUSAL_PREFIXES and manifest.status(c, s).error.startswith(REFUSAL_PREFIXES[s])}
    unfinished = [c for c, stage in stops.items() if stage is not None and c not in rejected]
    gate_ran = gate_status == GATE_BUILT and "gate" in measured
    if measured and not unfinished and gate_ran:
        if not any(stage in measured for stage in STAGE_NAMES[STAGE_NAMES.index("gate") + 1:]):
            return f"{GATE_COST_ONLY}: measured over the registered stages {list(measured)} on {len(doh)} raw-stream clips"
        return f"measured over the registered stages {list(measured)} on {len(doh)} raw-stream clips"
    details: list[str] = []
    if unfinished:
        first, stage = unfinished[0], cast("str", stops[unfinished[0]])
        status = manifest.status(first, stage)
        details.append(f"{first} stopped at {stage} ({status.state}: {status.error[:160]})")
    elif not measured:
        details.append("no registered stage ran")
    if not gate_ran:
        details.append(_gate_not_run(gate_status))
    return (f"unmeasured: {len(doh) - len(unfinished)} of {len(doh)} raw-stream clips finished the registered stages "
            f"{list(measured)}; {'; '.join(details)}")


def _gate_not_run(gate_status: str) -> str:
    """Why the interaction gate did not run on the clips, for the raw-stream status."""
    if gate_status == GATE_UNBUILT:
        return f"the interaction gate is {GATE_UNBUILT}, so no clip passed a gate: S2 needs the built gate"
    if gate_status == EXCLUDED:
        return f"the interaction gate is {EXCLUDED}, so it did not run on the clips: S2 needs gate in bench.stages"
    return f"the interaction gate did not run on the clips (gate_status {gate_status!r}): S2 needs the gate to run"


def check_stage_packages(stages: Sequence[str],
                         find_spec: Callable[[str], object | None] = importlib.util.find_spec) -> None:
    """Fails fast when a requested real stage's Python package is missing.

    Raises:
        RuntimeError: Naming the stage and package; otherwise every clip would fail at that stage and the headline
            would cover only the stages before it.
    """
    missing = {stage: pkg for stage, pkg in STAGE_PACKAGES.items() if stage in stages and find_spec(pkg) is None}
    if missing:
        raise RuntimeError(f"bench.stages requests {sorted(missing)}, whose package {sorted(set(missing.values()))} is "
                           "not installed: every clip would fail there. Install it or drop the stage from bench.stages.")


def prepare_artifacts(cfg: DictConfig, root: Path, clips: list[BenchClip], artifact_root: Path,
                      write_inputs: bool, resume: bool = False) -> dict[str, str]:
    """Removes the clips' stale artifact directories (`bench.overwrite`), then, for real workers, writes the untimed
    keyframe prompts and DexYCB `camera.json`. Returns `clip_id -> prompt source`.

    On `resume`, an existing clip directory is KEPT and its inputs are NOT rewritten. Both halves matter: the
    artifacts are what a resumed run reuses, and rewriting the inputs would overwrite a `mask_prompt.npz` that the
    prompt stage produced on the previous attempt with the dataset label -- silently changing what the mask stage
    propagates from.

    Every clip's `source_domain.json` (its `BenchClip.domain`, the carrier the profiled solve and contact workers
    read) is written here, before any stage and whatever `write_inputs` is; a kept (resumed) directory must already
    declare the same domain or have none.

    Raises:
        ValueError: If `resume` and `bench.overwrite` are both set (one reuses artifacts, the other deletes them), or a
            kept directory declares another source domain.
        FileExistsError: If a clip directory exists and neither `bench.overwrite` nor `resume` is set.
    """
    if resume and bool(cfg.bench.overwrite):
        raise ValueError("bench.resume and bench.overwrite contradict each other: resume reuses the artifacts that "
                         "overwrite deletes. Set bench.overwrite=false to resume a killed run.")
    kept = {clip.clip_id for clip in clips if resume and (artifact_root / clip.clip_id).exists()}
    for clip in clips:
        clip_dir = artifact_root / clip.clip_id
        if clip_dir.exists() and clip.clip_id not in kept:
            if not bool(cfg.bench.overwrite):
                raise FileExistsError(f"{clip_dir} exists; set bench.overwrite=true to remove the benchmark clip's artifacts")
            shutil.rmtree(clip_dir)
    for clip in clips:
        clip_dir = artifact_root / clip.clip_id
        declared = read_domain_record(clip_dir)
        if declared is not None and declared is not clip.domain:
            raise ValueError(f"{clip_dir / DOMAIN_FILE} declares {declared.value}, but this run's {clip.clip_id} is "
                             f"{cast('SourceDomain', clip.domain).value}: a resumed directory cannot change domain")
        write_domain_record(clip_dir, clip.clip_id, clip.source, cast("SourceDomain", clip.domain))
    if not write_inputs:
        return {}
    prompts: dict[str, str] = {}
    whisking = [clip for clip in clips if clip.source == "whisking"]
    if whisking:
        paths, keyframe = _whisking_paths(cfg, root), int(cfg.whisking.keyframe_frame_idx)
        require_demo_files(paths, ("layout_json", "mesh_path"))
        mask = whisking_prompt(paths["layout_json"], paths["mesh_path"], keyframe)
        row = keyframe_row(int(cfg.whisking.n_frames_full), int(cfg.bench.stride), keyframe)
        for clip in whisking:
            if clip.clip_id not in kept:
                write_mask_prompt(artifact_root / clip.clip_id, row, mask)
            prompts[clip.clip_id] = f"SAM 3D keyframe layout silhouette (source frame {keyframe}, decode row {row})"
    dexycb_root = _resolve(root, cfg.dexycb.root)
    for clip in clips:
        if clip.source == "dexycb":
            seq = clip.path.relative_to(dexycb_root).as_posix()
            if clip.clip_id not in kept:
                write_mask_prompt(artifact_root / clip.clip_id, 0, dexycb_prompt(dexycb_root, seq))
                write_camera_json(artifact_root / clip.clip_id, dexycb_intrinsics(dexycb_root, seq))
            prompts[clip.clip_id] = "DexYCB ground-truth label map of frame 0"
        elif clip.source == "100doh":
            prompts[clip.clip_id] = "none: no keyframe prompt source (detector) yet, so the mask stage fails"
        elif clip.source in OAKINK2_SOURCES:
            prompts[clip.clip_id] = ("none written: the prompt stage chooses the object (prompt_stage.enabled); "
                                     "no camera.json (estimated intrinsics, as on internet video)")
    hoi4d = {w.clip_id: w for w in _hoi4d_windows(cfg, root)}
    if any(clip.source == HOI4D_SOURCE for clip in clips):
        reader = HOI4DReader(_resolve(root, cfg.hoi4d.root))
        for clip in clips:
            if clip.source != HOI4D_SOURCE:
                continue
            w = hoi4d[clip.clip_id]
            if clip.clip_id not in kept:
                write_mask_prompt(artifact_root / clip.clip_id, 0, hoi4d_prompt(reader, w.sequence, w.start))
                write_camera_json(artifact_root / clip.clip_id, reader.intrinsics(w.sequence))
            prompts[clip.clip_id] = (f"HOI4D ground-truth object label of frame {w.start} (decode row 0); "
                                     "camera.json = the shipped calibration")
    if any(clip.source == "100doh" for clip in clips):
        logger.warning("100DOH clips have no keyframe mask prompt source yet; their mask stage will fail")
    return prompts


def require_demo_files(paths: Mapping[str, Path], keys: Sequence[str]) -> None:
    """Checks that the whisking demo files `keys` of `paths` exist before they are read.

    Raises:
        FileNotFoundError: Naming the missing files and how to fetch the submodule that holds them.
    """
    missing = [str(paths[key]) for key in keys if not paths[key].is_file()]
    if missing:
        raise FileNotFoundError(f"missing {', '.join(missing)}: {SUBMODULE_HINT}")


def failed_clips(manifest: ClipManifest) -> dict[str, str]:
    """`{clip_id: "<stage>: <error>"}` of every clip with a failed stage (its first failed stage in pipeline order)."""
    out: dict[str, str] = {}
    for clip_id in manifest.clip_ids():
        for stage in stage_order(manifest):
            status = manifest.status(clip_id, stage)
            if status.state == "failed":
                out[clip_id] = f"{stage}: {status.error}"
                break
    return out


def exit_status(manifest: ClipManifest) -> int:
    """1 when the run has clips and every one of them failed a stage (the report then times no finished clip), else
    0. A partial failure stays 0: the report and its WARNING describe it."""
    failed = failed_clips(manifest)
    n_clips = len(manifest.clip_ids())
    if n_clips == 0 or len(failed) < n_clips:
        return 0
    logger.error("every clip failed (%d of %d), so the yield report times no finished clip; first failure per clip: "
                 "%s", len(failed), n_clips, "; ".join(f"{c}: {e}" for c, e in failed.items()))
    return 1


RESUME_SETTINGS_FILE = "resume_settings.json"
# Small files that the config names by path and whose BYTES decide a verdict: the same path can hold another rule.
RULE_FILE_KEYS = ("qc_stage.rule_path", "driftqc_stage.constants_path", "driftqc_stage.trackweight_constants_path")


def rule_file_digests(cfg: DictConfig, root: Path) -> dict[str, str | None]:
    """sha256 of each rule file the config names (`RULE_FILE_KEYS`): None for an unset key, "absent" for a path
    that is no file."""
    out: dict[str, str | None] = {}
    for key in RULE_FILE_KEYS:
        value = OmegaConf.select(cfg, key, default=None)
        if not value:
            out[key] = None
            continue
        path = _resolve(root, str(value))
        out[key] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "absent"
    return out


def settings_digest(cfg: DictConfig, stages: Sequence[str], root: Path) -> str:
    """sha256 of the run's settings: the whole config, the stage list and the bytes of the rule files the config
    names, without the two keys that only say how to START a run (`bench.resume`, `bench.overwrite`). The config
    is taken unresolved, so a value that an interpolation reads from the environment is not covered; every config
    key is. Other files that the config names by path (weights, inventories) are covered by path only."""
    doc = cast("dict[str, Any]", OmegaConf.to_container(cfg, resolve=False))
    doc["bench"] = {k: v for k, v in doc.get("bench", {}).items() if k not in ("resume", "overwrite")}
    text = json.dumps({"config": doc, "stages": list(stages), "rule_files": rule_file_digests(cfg, root)},
                      sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()


def check_resume_settings(path: Path, digest: str) -> None:
    """Refuse to resume a manifest under settings other than the ones that wrote it.

    A resumed run skips every stage marked done. Under changed settings (an added screen, another stage list,
    another threshold) the skipped results belong to the old settings, and an earlier QC acceptance could stand
    beside a stage that the new settings fail.

    Raises:
        ValueError: If the previous run left no settings record, or its record differs from `digest`.
    """
    if not path.is_file():
        raise ValueError(f"{path.parent} has a manifest and no settings record ({path.name}), so the settings of "
                         "the previous run cannot be compared: start over with a fresh bench.artifact_root and a "
                         "fresh bench.out_dir.")
    if json.loads(path.read_text()).get("settings_sha256") != digest:
        raise ValueError(f"{path.parent} was written under different settings (stage list or config): a resumed "
                         "run must use the settings of the run it resumes. For changed settings use a fresh "
                         "bench.artifact_root and a fresh bench.out_dir.")


def resumed_manifest(clips: list[BenchClip], manifest_path: Path, resume: bool, stages: Sequence[str]) -> ClipManifest:
    """The run's manifest: the previous attempt's when `resume`, otherwise a fresh one.

    The orchestrator saves the manifest after EVERY stage transition, so a run killed mid-corpus -- which on a
    shared card is a matter of when, not whether -- has a durable record of
    what finished. Loading it back and leaving `skip_done` on is what makes the rerun cost the lost clip instead of
    the whole corpus. Only a stage marked "done" is skipped, so a stage the kill caught mid-flight reruns, and
    artifact writes are atomic (`save_npz`), so no half-written file survives to be read as finished.

    Raises:
        ValueError: If the saved manifest is for a different clip set.
    """
    ids = [clip.clip_id for clip in clips]
    if not (resume and manifest_path.is_file()):
        return ClipManifest.create(ids, [clip.manifest_source for clip in clips], [clip.n_frames for clip in clips])
    manifest = ClipManifest.load(manifest_path)
    if sorted(manifest.clip_ids()) != sorted(ids):
        raise ValueError(f"{manifest_path} was written for {len(manifest.clip_ids())} clips and this run has "
                         f"{len(ids)}: a resumed run must have the same clip set. Delete the manifest to start over.")
    done = sum(1 for clip_id in ids for stage in stages if manifest.status(clip_id, stage).state == "done")
    logger.info("resuming %s: %d of %d clip-stages already done", manifest_path, done, len(ids) * len(stages))
    return manifest


def run_bench(cfg: DictConfig, root: Path, stage_fns: Mapping[str, StageFn | BatchedStage] | None = None,
              count_frames: Callable[[Path], int] = count_decoded_frames) -> YieldReport:
    """Runs the benchmark and writes the manifest and the yield report; `stage_fns` replaces the real workers."""
    bench = cfg.bench
    out_dir, artifact_root = _resolve(root, bench.out_dir), _resolve(root, bench.artifact_root)
    stages = pipeline_stages(cfg)
    batched = bool(OmegaConf.select(cfg, "bench.batched.enabled", default=False))
    declared_cap = OmegaConf.select(cfg, "bench.batched.cap_gb", default=None)
    if stage_fns is not None and not batched:
        check_single_stream_fns(stage_fns)
    if stage_fns is None:
        check_stage_packages(stages)
    clips = build_clips(cfg, root, count_frames)
    # Refuse a config that contradicts a clip's source-domain profile BEFORE anything is built or deleted (the
    # real workers repeat the check in build_stage_fns; injected ones are covered here)
    check_run_profile({clip.clip_id: cast("SourceDomain", clip.domain) for clip in clips},
                      SolveStageConfig.from_mapping(cfg.solve_stage), contact_enabled(cfg), source_profile_config(cfg))
    s2 = s2_status(cfg, clips)
    if s2.startswith("unmeasured"):
        by_source = Counter(clip.source for clip in clips)
        logger.warning("S2 (raw-stream throughput) is unmeasured (%s). Running %d clips: %s.",
                       s2.removeprefix("unmeasured: "), len(clips),
                       ", ".join(f"{n} {src}" for src, n in sorted(by_source.items())))
    fns: dict[str, StageFn | BatchedStage]
    if stage_fns is None:
        real, status = build_stage_fns(cfg, root, clips)
        fns = {**real}
        if "qc" in stages and "qc" not in fns:
            raise ValueError(f"bench.stages requests qc, but the qc stage is {status['qc']}: the qc stage has no "
                             "accept-all default, so the yield would be silently QC-less. Set qc_stage.rule_path to a "
                             "saved AcceptanceRule, or drop qc from bench.stages.")
        hostpaths.check_stage_paths(cfg, stages)   # before any artifact is written or model loaded
    else:
        fns = dict(stage_fns)
        status = {stage: INJECTED if stage in fns else NOT_REGISTERED.get(stage, "not registered")
                  for stage in known_stages(cfg)}
    status = {stage: label if stage in stages else EXCLUDED for stage, label in status.items()}
    resume = bool(OmegaConf.select(cfg, "bench.resume", default=False))
    manifest_path, settings_path = out_dir / MANIFEST_FILE, out_dir / RESUME_SETTINGS_FILE
    digest = settings_digest(cfg, stages, root)
    if resume and bool(bench.overwrite):
        raise ValueError("bench.resume and bench.overwrite contradict each other: resume reuses the artifacts that "
                         "overwrite deletes. Set bench.overwrite=false to resume a killed run.")
    resuming = resume and manifest_path.is_file()
    if resuming:
        check_resume_settings(settings_path, digest)       # before anything is written or run
    # Without a manifest there is no run to resume and nothing to compare settings with: clip directories that
    # exist are then refused as on any fresh run, never adopted.
    prompts = prepare_artifacts(cfg, root, clips, artifact_root, write_inputs=stage_fns is None, resume=resuming)

    manifest = resumed_manifest(clips, manifest_path, resume, stages)
    manifest.save(manifest_path)
    settings_path.write_text(json.dumps({"settings_sha256": digest}) + "\n")
    engine_cfg = EngineConfig(artifact_root=artifact_root, stages=stages, system="fastdo_engine_bench",
                              manifest_path=manifest_path)
    selected = {stage: fn for stage, fn in fns.items() if stage in stages}
    batch_run: BatchRunReport | None = None
    if batched:
        mapping = selected if stage_fns is not None else build_batched_stages(
            cfg, cast("dict[str, StageFn]", selected), artifact_root / ".hands_jobs", artifact_root / ".gpu.lock")
        try:
            batch_run = run_manifest_batched(manifest, batched_engine_config(bench, engine_cfg), mapping)
        finally:
            hands = mapping.get("hands")
            if isinstance(hands, HandsBatchStage):
                hands.close()                  # the warm HaWoR driver lives exactly as long as the run
    else:
        run_manifest(manifest, engine_cfg, cast("dict[str, StageFn]", selected))

    gate_built = "gate" in fns and "gate" in stages
    gate_status = GATE_BUILT if gate_built else (GATE_UNBUILT if "gate" not in fns else EXCLUDED)
    report = compute_yield_report(manifest, gate_built=gate_built,
                                  qc_verdicts=read_qc_verdicts(artifact_root, manifest.clip_ids()))
    counts = stage_state_counts(manifest)
    total = sum(report.per_stage_seconds.values())
    # `stage_order(manifest)`, not `STAGE_NAMES`: the raw-stream run's `prompt` is inserted by
    # `with_prompt` and is not in that tuple, so iterating it means the stage can never be reported as
    # complete -- nor, at the warning below, ever be named as the gap, which is the one job that warning has.
    ordered = stage_order(manifest)
    every_clip = [stage for stage in ordered if report.n_clips and counts[stage]["done"] == report.n_clips]
    agreement, compared = clip_rate_agreement(manifest, artifact_root)
    if any(clip.source == "100doh" for clip in clips):
        s2 = s2_status(cfg, clips, manifest, [stage for stage in stages if stage in fns], gate_status=gate_status)
        if s2.startswith("unmeasured"):
            logger.warning("S2 (raw-stream throughput) is %s", s2)
    # The decoded pixels are deleted later -- by the Ego4D licence's cleanup, or by shard pruning -- and
    # `decode.npz` carries the decode index's `frame_idx` in the same file. Written
    # HERE, at the end of the run and before anything can delete them, so the gate stays checkable. It was
    # the absence of exactly this call that left the gate's fallback correct, tested and unreachable.
    index_audit_path = write_index_audit(engine_cfg.artifact_root, manifest.clip_ids())

    provenance: dict[str, Any] = {
        "gate_status": gate_status,
        "decode_index_audit": str(index_audit_path),
        "gate_outcomes": gate_outcome_counts(manifest),
        # The prompt refusal rate is read from this. Before PROMPT_REFUSED_PREFIX existed a refusal and a
        # crash were the same manifest row, so the rate could only be scraped from log lines -- where each
        # refusal appears twice (batch and clip) and is easy to double count.
        "prompt_outcomes": prompt_outcome_counts(manifest),
        # The timebase check. A flagged window stays in every analysis, so the flag
        # has to appear in the run's own record; `unrecorded` is kept separate because a decode that
        # never measured its pts has not passed the check, it just did not take it.
        "timebase_flags": timebase_flags(engine_cfg.artifact_root, manifest.clip_ids()),
        "yield_basis": ("gate and QC acceptance" if gate_built else
                        f"QC-only: the interaction gate is {gate_status} and counts every clip as gate-accepted"),
        "clips_per_gpu_hour_definition": RATE_DEFINITION,
        "accepted_clips_per_gpu_hour": report.yield_fraction * report.clips_per_gpu_hour if report.n_qc_accepted else 0.0,
        "total_seconds": total,
        "clip_seconds": {clip_id: clip_total_seconds(manifest, clip_id) for clip_id in manifest.clip_ids()},
        "profile_formula_max_abs_diff": agreement,
        "profile_formula_clips_compared": compared,
        "stage_status": status,
        "stage_states": counts,
        "unmeasured_stages": [stage for stage in STAGE_NAMES if counts[stage]["done"] == 0],
        "stages_done_on_every_clip": every_clip,
        "complete_pipeline": gate_built and all(counts[s]["done"] == report.n_clips for s in STAGE_NAMES)
        and not any(label.startswith("borrowed") for label in status.values()),
        "s2_raw_stream": s2,
        "sources": {source: sum(clip.source == source for clip in clips) for source in SOURCES},
        # Clips per source domain and the profile switch; each solved clip's own record is source_profile.json
        "source_domains": {d.value: sum(clip.domain is d for clip in clips) for d in SourceDomain},
        "source_profile": {"study_override": source_profile_config(cfg).study_override},
        "clip_splits": {clip.clip_id: clip.split for clip in clips if clip.split is not None},
        "mask_prompts": prompts,
        "stages_run": list(stages),
        "manifest": str(manifest_path),
        "artifact_root": str(artifact_root),
        # All three cap fields, so a run's provenance cannot disagree with itself: the
        # raw environment variable (None when unset), the cap this process actually applies under
        # `apply_gpu_budget`'s rule (the variable, else gpu_budget.DEFAULT_GB; None when opted out with <= 0), and
        # the `bench.batched.cap_gb` the per-stage budgets are declared and validated against.
        "gpu_mem_cap_gb": os.environ.get("FASTDO_GPU_MEM_GB"),
        "applied_gpu_mem_cap_gb": applied_cap_gb(),
        "declared_cap_gb": None if declared_cap is None else float(declared_cap),
        **execution_provenance(manifest, batch_run),
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    write_yield_report(out_dir / REPORT_FILE, report, provenance)
    logger.info("yield report: %d clips, %d QC-accepted, %.1f clips per GPU-hour over the stages done on every clip %s "
                "(gate %s)", report.n_clips, report.n_qc_accepted, report.clips_per_gpu_hour, every_clip, gate_status)
    if not provenance["complete_pipeline"]:
        logger.warning("clips_per_gpu_hour is not end to end: stages not done on every clip: %s. A clip that stops "
                       "early counts at its partial cost, so the rate overstates end-to-end throughput.",
                       [stage for stage in ordered if stage not in every_clip])
    return report


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    apply_cpu_budget()
    apply_gpu_budget()
    try:
        root = Path(hydra.utils.get_original_cwd())
    except ValueError:                    # called with a composed config, outside a Hydra run (tests)
        root = Path.cwd()
    run_bench(cfg, root)
    code = exit_status(ClipManifest.load(_resolve(root, cfg.bench.out_dir) / MANIFEST_FILE))
    if code:
        raise SystemExit(code)


if __name__ == "__main__":
    main()
