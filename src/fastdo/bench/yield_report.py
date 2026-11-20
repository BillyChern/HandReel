"""Yield report: accepted-clip yield and clips per GPU-hour from a finished `ClipManifest`.

`compute_yield_report(manifest)` reads only the manifest's own columns, never a shared `Profiler`:

- `per_stage_seconds[stage]` sums the `{stage}_seconds` column over every clip, whatever its state. A failed stage's
  seconds are compute the engine spent (rejected and failed clips cost their front end: throughput budget, Section 2),
  and they are what the clip's own `profile.json` records for that stage.
- `clips_per_gpu_hour = 3600 / (total_seconds / n_clips)`: the reciprocal of the seconds spent per clip, `inf` when
  `total_seconds == 0` (`ProfileReport.clips_per_gpu_hour`'s convention). For one clip this is `3600 / total_seconds`,
  `ProfileReport`'s own formula; over `n` clips the plain `3600 / total_seconds` would count the whole manifest as one
  clip. Seconds are the orchestrator's per-stage wall clock on the one-GPU host, CPU stages included.
- A clip is gate-accepted when its `gate` stage is `"done"` (the orchestrator stops a clip at a failed stage). While
  the gate is unbuilt the caller passes `gate_built=False`: every clip counts as gate-accepted, the yield is
  QC-only, and the caller persists that flag (`GATE_UNBUILT`) next to the report.
- A clip is QC-accepted when it is gate-accepted, its `qc` stage is `"done"` and, when `qc_verdicts` is given, its
  verdict is True. The engine's `qc` worker marks the stage `"done"` for rejected clips too and records the decision
  in `qc.json` (`read_qc_verdicts`), so a real run passes those verdicts; a clip without a verdict is not accepted.
- `yield_fraction = n_qc_accepted / n_clips`, 0.0 for an empty manifest.

`write_yield_report` writes standard JSON: a non-finite float (an `inf` rate) is written as `null` and its JSON pointer
is listed in `non_finite_values` (`{"/clips_per_gpu_hour": "inf"}`); `read_yield_report` restores it.

The engine manifest module is imported lazily so `fastdo.bench` stays importable without the engine's worker stack.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastdo.bench.profiler import ProfileReport

if TYPE_CHECKING:
    from fastdo.engine.manifest import ClipManifest

__all__ = [
    "CAVEAT_KEY",
    "GATE_UNBUILT",
    "PROFILE_FILE",
    "QC_FILE",
    "YIELD_CAVEAT",
    "YieldReport",
    "clip_rate_agreement",
    "clip_total_seconds",
    "compute_yield_report",
    "read_qc_verdicts",
    "read_yield_report",
    "stage_order",
    "stage_state_counts",
    "write_yield_report",
]

logger = logging.getLogger(__name__)

GATE_UNBUILT = "unbuilt"
QC_FILE = "qc.json"              # `fastdo.engine.workers_qc.QC_FILE` (a test pins the two together)
PROFILE_FILE = "profile.json"    # written per clip by `fastdo.engine.orchestrator.run_clip`

# Emitted with EVERY yield report, as a reserved key, because the number it guards is the one a reader (or a
# paper) will quote. `yield_fraction` counts clips that passed the configured screen; calling that a rate of
# usable data assumes the screen discriminates, and the rule shipped on 2026-09-20 does not.
YIELD_CAVEAT = (
    "yield_fraction is the fraction of clips ACCEPTED BY THE CONFIGURED SCREEN. It is NOT a usable-data "
    "rate unless that screen's discrimination has been established OUT OF FOLD. The acceptance rule shipped "
    "on 2026-09-20 has an out-of-fold AUC of 0.4904 -- the 46th percentile of five PURE-NOISE features at "
    "the same n and k -- so with that rule this fraction carries no information about whether the accepted "
    "clips are good. Its conformal guarantee is unaffected: that is class-conditional coverage of the GOOD "
    "class and holds by construction, but it was never a claim about rejecting bad clips."
)
CAVEAT_KEY = "yield_fraction_caveat"


@dataclass(frozen=True)
class YieldReport:
    """Yield and throughput of one finished manifest (see the module docstring for each definition)."""

    n_clips: int
    n_gate_accepted: int
    n_qc_accepted: int
    clips_per_gpu_hour: float
    yield_fraction: float
    per_stage_seconds: dict[str, float]


def _rate(seconds_per_clip: float) -> float:
    return float("inf") if seconds_per_clip <= 0.0 else 3600.0 / seconds_per_clip


def stage_order(manifest: ClipManifest) -> list[str]:
    """`STAGE_NAMES` in pipeline order, then every other stage the manifest records, sorted.

    A clip's columns summed in this order are added in the orchestrator's execution order (for stages run in
    `STAGE_NAMES` order), so the sum equals the clip's `ProfileReport.total_seconds()` exactly: an unrun stage adds 0.0.
    """
    from fastdo.engine.manifest import STAGE_NAMES

    return [*STAGE_NAMES, *(stage for stage in manifest.stage_names() if stage not in STAGE_NAMES)]


def failed_before_qc(manifest: ClipManifest, clip_id: str) -> bool:
    """True when a stage that runs before qc is recorded `"failed"` for the clip.

    On a fresh run such a clip never reaches qc. On a RESUMED run qc can still read `"done"` from the previous
    attempt, so the yield must not count the clip on the qc column alone. Stages after qc in `STAGE_NAMES` do not
    undo a qc decision; every other recorded stage (the opt-in ones included) runs before qc.
    """
    from fastdo.engine.manifest import STAGE_NAMES

    after_qc = STAGE_NAMES[STAGE_NAMES.index("qc") + 1:]
    return any(manifest.status(clip_id, stage).state == "failed"
               for stage in stage_order(manifest) if stage != "qc" and stage not in after_qc)


def clip_total_seconds(manifest: ClipManifest, clip_id: str) -> float:
    """Sum of every `{stage}_seconds` column of one clip, in `stage_order`."""
    total = 0.0
    for stage in stage_order(manifest):
        total += manifest.status(clip_id, stage).seconds
    return total


def compute_yield_report(manifest: ClipManifest, *, gate_built: bool = True,
                         qc_verdicts: Mapping[str, bool] | None = None) -> YieldReport:
    """The `YieldReport` of a finished manifest (module docstring).

    Args:
        manifest: The manifest after `run_manifest`.
        gate_built: False while the interaction gate is unbuilt: every clip counts as gate-accepted.
        qc_verdicts: `clip_id -> accepted` from the clips' `qc.json`; None counts every `"done"` qc stage as accepted.
    """
    clip_ids = manifest.clip_ids()
    stages = stage_order(manifest)
    per_stage = dict.fromkeys(stages, 0.0)
    for clip_id in clip_ids:
        for stage in stages:
            per_stage[stage] += manifest.status(clip_id, stage).seconds
    total = 0.0
    for seconds in per_stage.values():
        total += seconds
    gate_ok = [c for c in clip_ids if not gate_built or manifest.status(c, "gate").state == "done"]
    qc_ok = [c for c in gate_ok if manifest.status(c, "qc").state == "done" and not failed_before_qc(manifest, c)
             and (qc_verdicts is None or bool(qc_verdicts.get(c, False)))]
    n = len(clip_ids)
    return YieldReport(n_clips=n, n_gate_accepted=len(gate_ok), n_qc_accepted=len(qc_ok),
                       clips_per_gpu_hour=_rate(total / n) if n else float("inf"),
                       yield_fraction=len(qc_ok) / n if n else 0.0, per_stage_seconds=per_stage)


def read_qc_verdicts(artifact_root: Path, clip_ids: Iterable[str]) -> dict[str, bool]:
    """`clip_id -> qc.json["accepted"]` for every clip whose artifact directory holds a `qc.json`."""
    verdicts: dict[str, bool] = {}
    for clip_id in clip_ids:
        path = Path(artifact_root) / clip_id / QC_FILE
        if path.is_file():
            verdicts[clip_id] = bool(json.loads(path.read_text())["accepted"])
    return verdicts


def stage_state_counts(manifest: ClipManifest) -> dict[str, dict[str, int]]:
    """Per stage (in `stage_order`), how many clips are `"done"`, `"failed"` and `"pending"`."""
    counts = {stage: {"done": 0, "failed": 0, "pending": 0} for stage in stage_order(manifest)}
    for clip_id in manifest.clip_ids():
        for stage, row in counts.items():
            row[manifest.status(clip_id, stage).state] += 1
    return counts


def clip_rate_agreement(manifest: ClipManifest, artifact_root: Path) -> tuple[float | None, int]:
    """`(largest |profile.json clips_per_gpu_hour() - 3600 / clip_total_seconds|, clips compared)` over the clips
    with a `profile.json`; two infinite rates agree.

    The difference is None, with a WARNING, when no clip was compared, so an empty cross-check never reads as perfect
    agreement. After a fresh run (no stage skipped as already done) every clip's profile holds exactly the stages its
    manifest columns record, so the difference is 0.0.
    """
    worst, compared = 0.0, 0
    for clip_id in manifest.clip_ids():
        path = Path(artifact_root) / clip_id / PROFILE_FILE
        if not path.is_file():
            logger.info("clip %r has no %s; left out of the clips-per-GPU-hour cross-check", clip_id, PROFILE_FILE)
            continue
        compared += 1
        from_profile = ProfileReport.from_json(path).clips_per_gpu_hour()
        from_manifest = _rate(clip_total_seconds(manifest, clip_id))
        if from_profile != from_manifest:
            worst = max(worst, abs(from_profile - from_manifest))
    if compared == 0:
        logger.warning("no clip under %s has a %s: the clips-per-GPU-hour cross-check compared nothing",
                       artifact_root, PROFILE_FILE)
        return None, 0
    return worst, compared


def _pointer_token(key: str) -> str:
    return key.replace("~", "~0").replace("/", "~1")


def _finite_json(value: Any, pointer: str, non_finite: dict[str, str]) -> Any:
    """`value` with every non-finite float replaced by None, its JSON pointer recorded in `non_finite`."""
    if isinstance(value, float) and not math.isfinite(value):
        non_finite[pointer] = repr(value)        # "inf", "-inf" or "nan"
        return None
    if isinstance(value, Mapping):
        return {k: _finite_json(v, f"{pointer}/{_pointer_token(str(k))}", non_finite) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(v, f"{pointer}/{i}", non_finite) for i, v in enumerate(value)]
    return value


def write_yield_report(path: Path, report: YieldReport, provenance: Mapping[str, Any] | None = None) -> None:
    """Writes the report's fields, then `provenance`, then `non_finite_values`, as indented standard JSON (tmp
    sibling, then `os.replace`; module docstring for non-finite floats).

    `CAVEAT_KEY` is written into every report: `yield_fraction` is the number a reader will quote, and it
    means "passed the configured screen", not "is usable". Emitting the caveat here rather than documenting
    it means no caller can produce a yield report without it.

    Raises:
        ValueError: If `provenance` repeats a `YieldReport` field or a reserved key (`non_finite_values`,
            `CAVEAT_KEY`).
    """
    payload: dict[str, Any] = dataclasses.asdict(report)
    extra = dict(provenance or {})
    clash = sorted(set(payload) & set(extra) | ({"non_finite_values", CAVEAT_KEY} & set(extra)))
    if clash:
        raise ValueError(f"provenance keys {clash} would overwrite YieldReport fields or reserved keys")
    payload.update(extra)
    non_finite: dict[str, str] = {}
    safe = _finite_json(payload, "", non_finite)
    safe["non_finite_values"] = non_finite
    safe[CAVEAT_KEY] = YIELD_CAVEAT
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(safe, indent=2, allow_nan=False))
    os.replace(tmp, path)


def read_yield_report(path: Path) -> dict[str, Any]:
    """The payload `write_yield_report` wrote, with every value listed in `non_finite_values` restored to its float."""
    payload: dict[str, Any] = json.loads(Path(path).read_text())
    for pointer, text in payload.get("non_finite_values", {}).items():
        *parents, last = [t.replace("~1", "/").replace("~0", "~") for t in pointer.split("/")[1:]]
        node: Any = payload
        for token in parents:
            node = node[int(token)] if isinstance(node, list) else node[token]
        if isinstance(node, list):
            node[int(last)] = float(text)
        else:
            node[last] = float(text)
    return payload
