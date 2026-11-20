"""The `prompt` stage worker: `mask_prompt.npz` from raw video, with no label.

`workers_video.py` states the gap it fills: "No stage writes the prompt yet", so today a caller supplies the
keyframe and mask from a dataset label. EK-100 and Ego4D have none, which is why S2 cannot start.

What this does, per window:
1. Ask the Hands23 detector (its own interpreter, `fastdo.prompt.hands23_client`) for hand records on a handful of
   candidate rows of `decode.npz`.
2. Pick the row and object box where a hand is most confidently HOLDING something (`fastdo.prompt.choose`).
3. Turn that box into a mask with SAM 2's IMAGE predictor, which takes an array, so no JPEG or crop is written --
   the only pixel cache stays `decode.npz` (Ego4D licence).
4. Write `mask_prompt.npz` in the schema the mask stage already reads, plus `prompt.json` for provenance.

**It refuses rather than guesses.** No confident hand-object contact, or an implausible mask, raises, so the window
is recorded as a stage failure and counted in the yield denominator. A prompt stage that always answers would turn
every unusable window into an accepted one, which is the number S2 exists to measure.

Licence: Hands23 code is MIT. Its training dataset is CC BY-NC 4.0. The public checkpoint instructions do not give a
separate licence; the project treats the optional detector as non-commercial. Those sources do not settle whether
that reaches data made with the detector (THIRD_PARTY_LICENSES.md, M10).
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np

from fastdo import hostpaths
from fastdo.engine.manifest import ClipManifest
from fastdo.engine.orchestrator import StageFn
from fastdo.engine.refusal import PROMPT_REFUSED_PREFIX
from fastdo.engine.worker_support import (
    DECODE_FILE,
    WorkerContext,
    read_decode_rows,
    read_frame_idx,
    require_artifact,
    save_npz,
)
from fastdo.prompt.choose import (
    GATES,
    PromptThresholds,
    candidate_rows,
    choose_prompt,
    gate_trace,
    mask_borders,
    mask_is_plausible,
)
from fastdo.prompt.hands23_client import Hands23Client
from fastdo.shape.sam2_variants import DEFAULT_SAM2_VARIANT, SAM2_VARIANTS

__all__ = ["PROMPT_FILE", "PROMPT_META_FILE", "PROMPT_REFUSAL_FILE", "PROMPT_REFUSED_PREFIX", "PromptRefused",
           "PromptStageConfig", "make_prompt_worker", "prompt_outcome_counts", "refusal_record", "with_prompt"]

logger = logging.getLogger(__name__)

PROMPT_FILE = "mask_prompt.npz"          # the schema workers_video already reads: keyframe_index, mask
PROMPT_META_FILE = "prompt.json"
PROMPT_REFUSAL_FILE = "prompt_refusal.json"   # written only with prompt_stage.record_refusals
DETECTOR_SCRIPT = Path(__file__).resolve().parents[1] / "prompt/hands23_detect.py"


def _hands23_default(var: str, rel: str) -> str:
    """`$var`, else `<hands23 root>/<rel>` (`fastdo.hostpaths.hands23_root`; run/conf/config.yaml uses the same keys)."""
    return str(hostpaths.env_path(var, hostpaths.hands23_root() / rel))
BoxMasker = Callable[[np.ndarray, list[float]], np.ndarray]


class PromptRefused(RuntimeError):
    """No window-level prompt could be made: no confident hand-object contact, or an implausible mask.

    The message starts with `PROMPT_REFUSED_PREFIX`, as `workers_gate.GateRejected` does:
    `run_clip` marks a raising stage `"failed"` with `str(exc)` and the manifest has no separate state, so a
    refusal and a genuine error are otherwise the same row. `prompt_outcome_counts` classifies on that prefix,
    and S2's Gate B criterion 5 -- the prompt refusal rate -- is not computable from artefacts without it.
    """


@dataclasses.dataclass(frozen=True)
class PromptStageConfig:
    """`prompt_stage` configuration.

    Attributes:
        enabled: Whether the engine registers the stage.
        python: Interpreter of the detector environment.
        weights: `model_hands23.pth`.
        candidate_stride: Run the detector on every n-th decode row (the ends are always included). 5 gives about
            20 candidates in a 10 s window at decode stride 3, measured at 1.39 s fp32.
        min_score: Lowest `interaction_score * p_held_like`; **null refuses to build**, as `qc_stage.rule_path` does,
            so no run can quietly use an unset threshold.
        hand_score, obj_score: Detection-score floors.
        area_frac: Allowed mask area as a fraction of the frame.
        sam2_variant: The SAM 2 size, the mask stage's default unless overridden.
        device: Torch device for both models.
        detect_batch: Candidate rows per Hands23 forward. 1 (default) is the one-row loop, byte-identical to
            before the option existed; >1 batches the backbone and RPN (the ROI heads stay per row, see
            `hands23_detect`). NOT exact: float-level box and score changes, SAM 2 masks that move, and occasional
            discrete flips in non-chosen records.
        detect_prefetch: CPU threads preparing candidate rows (BGR flip, resize) ahead of the Hands23 forward.
            0 (default) is the one-row loop; >0 is data movement only, the model sees the same tensors.
        record_refusals: On a refusal, write `prompt_refusal.json` (`refusal_record`): every
            candidate row's Hands23 records with the first gate each fails, and on the mask path the chosen record
            and its mask area. RECORD ONLY: no decision changes, and a built window's artifacts are untouched.
            False (default) writes nothing, so every artifact directory is byte-identical to before the option.
    """

    enabled: bool = False
    python: str = dataclasses.field(default_factory=lambda: _hands23_default("FASTDO_HANDS23_PYTHON",
                                                                             "env_hands23/bin/python"))
    weights: str = dataclasses.field(default_factory=lambda: _hands23_default("FASTDO_HANDS23_WEIGHTS",
                                                                              "weights/model_hands23.pth"))
    candidate_stride: int = 5
    min_score: float | None = None
    hand_score: float = 0.9
    obj_score: float = 0.9
    area_frac: tuple[float, float] = (0.002, 0.40)
    sam2_variant: str = DEFAULT_SAM2_VARIANT
    device: str = "cuda"
    detect_batch: int = 1
    detect_prefetch: int = 0
    record_refusals: bool = False

    def __post_init__(self) -> None:
        if self.candidate_stride < 1:
            raise ValueError(f"prompt_stage.candidate_stride must be >= 1, got {self.candidate_stride}")
        if isinstance(self.detect_batch, bool) or int(self.detect_batch) != self.detect_batch or self.detect_batch < 1:
            raise ValueError(f"prompt_stage.detect_batch must be an integer >= 1, got {self.detect_batch!r}")
        if (isinstance(self.detect_prefetch, bool) or int(self.detect_prefetch) != self.detect_prefetch
                or self.detect_prefetch < 0):
            raise ValueError(f"prompt_stage.detect_prefetch must be an integer >= 0, got {self.detect_prefetch!r}")
        if not isinstance(self.record_refusals, bool):
            raise ValueError(f"prompt_stage.record_refusals must be a bool, got {self.record_refusals!r}")  # noqa: TRY004
        if self.sam2_variant not in SAM2_VARIANTS:
            raise ValueError(f"prompt_stage.sam2_variant must be one of {sorted(SAM2_VARIANTS)}, "
                             f"got {self.sam2_variant!r}")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object] | None) -> PromptStageConfig:
        """Builds from a config mapping; an unknown key raises rather than being ignored."""
        if not mapping:
            return cls()
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted({str(k) for k in mapping} - known)
        if unknown:
            raise ValueError(f"prompt_stage has unknown keys {unknown}; known keys: {sorted(known)}")
        data = dict(mapping)
        if "area_frac" in data:
            data["area_frac"] = tuple(float(x) for x in data["area_frac"])  # type: ignore[arg-type]
        return cls(**data)  # type: ignore[arg-type]

    def thresholds(self) -> PromptThresholds:
        """The gates, refusing to build when `min_score` is unset (no accept-all default)."""
        if self.min_score is None:
            raise ValueError("prompt_stage.min_score is null: set it (from CALIB or a priori) before running the "
                             "prompt stage; there is no accept-all default")
        return PromptThresholds(min_score=float(self.min_score), hand_score=self.hand_score,
                                obj_score=self.obj_score, area_frac=self.area_frac)


def _sam2_box_masker(variant: str, device: str) -> BoxMasker:
    """SAM 2's image predictor, box-prompted. It takes an ARRAY, so nothing is written to disk.

    It must load through `load_sam2_image_predictor`, not `SAM2ImagePredictor.from_pretrained`: sam2 composes
    its config through Hydra's process-wide state, which every `run/pipeline` script already owns as a
    `@hydra.main` app. Calling `from_pretrained` directly worked standalone and failed on all 3 clips of the
    first end-to-end run with "Cannot find primary config 'configs/sam2.1/sam2.1_hiera_l.yaml'".
    """
    from fastdo.shape.masks_sam2 import load_sam2_image_predictor

    predictor = load_sam2_image_predictor(device=device, model_id=SAM2_VARIANTS[variant])

    def mask_of(image: np.ndarray, box: list[float]) -> np.ndarray:
        predictor.set_image(image)
        masks, _, _ = predictor.predict(box=np.asarray(box, dtype=np.float32)[None, :], multimask_output=False)
        return np.asarray(masks, dtype=bool).reshape(masks.shape[-2], masks.shape[-1])

    return mask_of


def refusal_record(clip_id: str, path: str, rows: list[int], per_row: Mapping[int, list[dict]],
                   thresholds: PromptThresholds, meta: Mapping[str, object], choice: object = None,
                   mask: np.ndarray | None = None) -> dict[str, object]:
    """The `prompt_refusal.json` payload: numbers only, like `prompt.json`.

    `path` is "no_contact" or "implausible_mask". `records` is `gate_trace` (every record, its row, and the first
    gate it fails; None = eligible), so which gate each no-contact window failed can be read, and any gate variant
    can be replayed on CPU without re-running the detector.
    """
    trace = gate_trace(per_row, thresholds)
    counts = {g: sum(1 for r in trace if r["first_failed_gate"] == g) for g in GATES}
    counts["eligible"] = sum(1 for r in trace if r["first_failed_gate"] is None)
    out: dict[str, object] = {
        "clip_id": clip_id, "path": path, "candidate_rows": rows,
        "n_records": len(trace), "first_failed_counts": counts, "records": trace,
        "thresholds": dataclasses.asdict(thresholds), "detector": dict(meta)}
    if choice is not None:
        out["choice"] = dataclasses.asdict(choice)  # type: ignore[call-overload]
    if mask is not None:
        out["mask_area_frac"] = float(np.asarray(mask, dtype=bool).mean())
        out["mask_borders"] = mask_borders(mask)
    return out


def make_prompt_worker(cfg: PromptStageConfig, client: Hands23Client | None = None,
                       masker: BoxMasker | None = None) -> StageFn:
    """The `prompt` stage. `client` and `masker` are injection points; the defaults are the real models."""
    thresholds = cfg.thresholds()
    detector = client or Hands23Client(python=cfg.python, script=DETECTOR_SCRIPT, weights=Path(cfg.weights),
                                       device=cfg.device, batch=int(cfg.detect_batch),
                                       prefetch=int(cfg.detect_prefetch))
    box_masker: BoxMasker | None = masker
    ctx = WorkerContext()

    def prompt_worker(clip_id: str, artifact_dir: Path) -> None:
        nonlocal box_masker
        ctx.budget()
        artifact_dir = Path(artifact_dir)
        decode_path = require_artifact(artifact_dir / DECODE_FILE, "decode")
        n_rows = int(read_frame_idx(artifact_dir).shape[0])     # the frames are read only for the chosen row
        rows = candidate_rows(n_rows, cfg.candidate_stride)
        refusal_path = artifact_dir / PROMPT_REFUSAL_FILE
        if cfg.record_refusals:
            refusal_path.unlink(missing_ok=True)                 # a rerun must not leave a stale refusal behind

        result = detector.detect(decode_path, rows)
        choice = choose_prompt(result.rows, thresholds)
        if choice is None:
            if cfg.record_refusals:
                write_refusal(refusal_path, refusal_record(clip_id, "no_contact", rows, result.rows, thresholds,
                                                           result.meta))
            raise PromptRefused(f"{PROMPT_REFUSED_PREFIX}: {clip_id}: no confident hand-object contact on rows {rows} "
                               f"(min_score {thresholds.min_score}); the window is counted as a failure")

        if box_masker is None:
            box_masker = _sam2_box_masker(cfg.sam2_variant, cfg.device)
        frames, _ = read_decode_rows(decode_path, [choice.row])
        mask = np.asarray(box_masker(frames[0], choice.box), dtype=bool)
        if mask.shape != frames.shape[1:3]:
            raise ValueError(f"{clip_id}: the prompt mask is {mask.shape}, the frames are {frames.shape[1:3]}")
        if not mask_is_plausible(mask, thresholds):
            if cfg.record_refusals:
                write_refusal(refusal_path, refusal_record(clip_id, "implausible_mask", rows, result.rows, thresholds,
                                                           result.meta, choice, mask))
            raise PromptRefused(f"{PROMPT_REFUSED_PREFIX}: {clip_id}: the box prompt gave an implausible mask "
                               f"({float(mask.mean()):.4f} of the frame); the window is counted as a failure")

        save_npz(artifact_dir / PROMPT_FILE, keyframe_index=np.array(choice.row, dtype=np.int64), mask=mask)
        (artifact_dir / PROMPT_META_FILE).write_text(json.dumps({
            "clip_id": clip_id, "source": "hands23+sam2", "candidate_rows": rows,
            "choice": dataclasses.asdict(choice), "mask_area_frac": float(mask.mean()),
            "thresholds": dataclasses.asdict(thresholds), "sam2_variant": cfg.sam2_variant,
            "detector": result.meta}, indent=1))
        logger.info("prompt %r: row %d of %d candidates, score %.3f, mask %.3f of the frame", clip_id, choice.row,
                    len(rows), choice.score, float(mask.mean()))

    return prompt_worker


def write_refusal(path: Path, payload: Mapping[str, object]) -> None:
    """`payload` as indented JSON via a tmp sibling and `os.replace` (a crash never leaves half a record)."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1))
    os.replace(tmp, path)


def with_prompt(stages: tuple[str, ...]) -> tuple[str, ...]:
    """`stages` with "prompt" inserted immediately before "mask" (idempotent).

    The mask stage consumes `mask_prompt.npz`, so the prompt must precede it and follow `decode`, which it reads.

    Raises:
        ValueError: If "mask" is absent, "decode" does not precede "mask", or "prompt" already sits elsewhere.
    """
    stages = tuple(stages)
    if "mask" not in stages:
        raise ValueError(f"with_prompt needs a 'mask' stage, got {stages}")
    at = stages.index("mask")
    if "decode" in stages and stages.index("decode") > at:
        raise ValueError(f"'decode' follows 'mask' in {stages}")
    if "prompt" in stages:
        if stages.count("prompt") == 1 and stages.index("prompt") == at - 1:
            return stages
        raise ValueError(f"'prompt' must precede 'mask' immediately (once), got {stages}")
    return stages[:at] + ("prompt",) + stages[at:]


def prompt_outcome_counts(manifest: ClipManifest) -> dict[str, int]:
    """`{"built", "refused", "error", "pending"}` over the manifest's prompt column.

    `done` built a prompt; `failed` with an error starting with `PROMPT_REFUSED_PREFIX` is a recorded refusal;
    any other `failed` is an unhandled error. The distinction is the whole point: S2's Gate A counts unhandled
    errors (a refusal is not one) and Gate B reads the refusal rate, and before the prefix existed the two were
    the same manifest row -- so the rate could only be scraped from log lines, where each refusal is logged
    twice (once by the batch, once by the clip) and is easy to double count.

    Mirrors `workers_gate.gate_outcome_counts`; the two are the same convention for two stages.
    """
    counts = dict.fromkeys(("built", "refused", "error", "pending"), 0)
    for clip_id in manifest.clip_ids():
        status = manifest.status(clip_id, "prompt")
        if status.state == "done":
            counts["built"] += 1
        elif status.state == "failed":
            counts["refused" if status.error.startswith(PROMPT_REFUSED_PREFIX) else "error"] += 1
        else:
            counts["pending"] += 1
    return counts
