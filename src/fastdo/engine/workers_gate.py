"""The `gate` stage: the interaction gate on the encoder's tokens; a rejected clip stops before any heavy stage.

Per clip, on CPU (`STAGE_GPU["gate"]` is False): read `encoder.npz` (`fastdo.gate.features.read_gate_tokens`), refuse a
crop or token width that differs from the bundle's, run `gate_features` (8 uniformly sampled rows, zero hand_conf)
and the bundle's model in eval mode, write `gate.json`, then raise `GateRejected` iff
`p_interact < threshold`. There is no accept-all default: the stage needs a gate bundle.

`GateRejected` is an ordinary `Exception` on purpose: `run_clip` marks the gate stage `"failed"` with its message and
stops the clip, so no downstream stage runs, and neither the orchestrator nor the manifest needs a new state.
The message starts with `GATE_REJECTED_PREFIX`, which is how `gate_outcome_counts` and the benchmark's S2 status
tell a rejection from an error. A rejected clip is retried on every `run_manifest` pass (failed rows count as pending),
at the cost of one gate call.

`gate.json` (version 1, written for accepted and rejected clips alike): `score` (= `p_interact`, the key
`workers_qc._gate_score` reads into `ClipRecord.gate_score`), `accepted`, `threshold`, `p_interact`, `p_rigid`,
`visibility`, `keyframe_scores` [8], `sampled_rows` [8], `sampled_frame_idx` [8], `keyframe_row` (the decode row with
the best keyframe score), `keyframe_frame_idx`, `crop`, `hand_conf_source`, `rigid_calibrated` (false: no source labels
rigidity) and `bundle` {`path`, `sha256`, `version`}.

`fastdo.gate` is imported inside the functions: it imports `fastdo.engine` submodules, and `fastdo.engine` imports this
module, so a module-level import would be circular.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from fastdo.engine.manifest import ClipManifest
from fastdo.engine.orchestrator import StageFn
from fastdo.engine.refusal import GATE_REJECTED_PREFIX
from fastdo.engine.worker_support import WorkerContext, require_artifact
from fastdo.engine.workers_qc import GATE_FILE, write_json

if TYPE_CHECKING:
    from fastdo.gate.bundle import GateBundle

__all__ = [
    "GATE_JSON_VERSION",
    "GATE_REJECTED_PREFIX",
    "GateRejected",
    "GateStageConfig",
    "gate_outcome_counts",
    "make_gate_worker",
    "read_gate_json",
]

logger = logging.getLogger(__name__)

GATE_JSON_VERSION = 1


class GateRejected(Exception):
    """The gate's accept decision for a clip: `p_interact` below the calibrated threshold."""

    def __init__(self, p_interact: float, threshold: float) -> None:
        super().__init__(f"{GATE_REJECTED_PREFIX}: p_interact {p_interact:.6f} < threshold {threshold:.6f}")
        self.p_interact = p_interact
        self.threshold = threshold


@dataclass(frozen=True)
class GateStageConfig:
    """The `gate_stage` config block: the gate bundle's path (None: the stage cannot be built)."""

    model_path: str | None = None

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> GateStageConfig:
        """From the `gate_stage` block (a dict or an OmegaConf node).

        Raises:
            ValueError: On an unknown key.
        """
        unknown = sorted({str(key) for key in mapping} - {"model_path"})
        if unknown:
            raise ValueError(f"gate_stage has unknown keys {unknown}; known keys: ['model_path']")
        model_path = mapping.get("model_path")
        return cls(model_path=None if model_path is None else str(model_path))


def make_gate_worker(cfg: GateStageConfig, bundle: GateBundle | None = None) -> StageFn:
    """The `gate` stage worker (module docstring). The bundle loads here, so a bad path fails before any clip.

    Args:
        cfg: The stage config; `model_path` names a bundle written by `run/pipeline/train_gate.py`.
        bundle: A loaded bundle to use instead of `cfg.model_path` (tests inject stub gates).

    Raises:
        ValueError: If neither `cfg.model_path` nor `bundle` is given (no accept-all default), or the file is not a
            valid gate bundle.
        FileNotFoundError: If `cfg.model_path` is absent.
    """
    from fastdo.gate.bundle import GATE_BUNDLE_VERSION, load_gate_bundle
    from fastdo.gate.features import gate_features, read_gate_tokens, sample_rows

    if bundle is None:
        if not cfg.model_path:
            raise ValueError("gate_stage.model_path is not set, and the gate stage has no accept-all default: fit and "
                             "save a gate bundle with run/pipeline/train_gate.py, then set gate_stage.model_path to it")
        bundle = load_gate_bundle(Path(cfg.model_path))
    gate = bundle
    model = gate.model.eval()
    threshold = float(gate.calibration.threshold)
    source = {"path": gate.path, "sha256": gate.sha256, "version": GATE_BUNDLE_VERSION}
    ctx = WorkerContext()
    logger.info("gate stage bundle %s (sha256 %s): threshold %.4f, crop %s", gate.path, gate.sha256[:12], threshold,
                gate.crop)

    def gate_worker(clip_id: str, artifact_dir: Path) -> None:
        ctx.budget()
        artifact_dir = Path(artifact_dir)
        tokens, frame_idx, crop = read_gate_tokens(artifact_dir)
        if crop != gate.crop:
            raise ValueError(f"{clip_id}: encoder.npz holds {crop!r} crops, but the gate bundle {gate.path} was "
                             f"calibrated on {gate.crop!r} crops: rerun the encoder stage with crop={gate.crop!r}")
        if int(tokens.shape[1]) != gate.config.token_dim:
            raise ValueError(f"{clip_id}: encoder tokens have width {int(tokens.shape[1])}, but the gate bundle "
                             f"{gate.path} expects {gate.config.token_dim}")
        rows = sample_rows(int(tokens.shape[0]))
        with torch.no_grad():
            out = model(gate_features(tokens)[None])
        p_interact = float(out.p_interact[0])
        keyframe_scores = [float(v) for v in out.keyframe_scores[0]]
        keyframe_row = int(rows[int(np.argmax(keyframe_scores))])
        accepted = p_interact >= threshold
        write_json(artifact_dir / GATE_FILE, {
            "version": GATE_JSON_VERSION, "score": p_interact, "accepted": accepted, "threshold": threshold,
            "p_interact": p_interact, "p_rigid": float(out.p_rigid[0]), "visibility": float(out.visibility[0]),
            "keyframe_scores": keyframe_scores, "sampled_rows": [int(r) for r in rows],
            "sampled_frame_idx": [int(frame_idx[r]) for r in rows], "keyframe_row": keyframe_row,
            "keyframe_frame_idx": int(frame_idx[keyframe_row]), "crop": crop, "hand_conf_source": gate.hand_conf_source,
            "rigid_calibrated": False, "bundle": source})
        logger.info("gate of %r: %s (p_interact %.4f, threshold %.4f)", clip_id,
                    "accepted" if accepted else "rejected", p_interact, threshold)
        if not accepted:
            raise GateRejected(p_interact, threshold)

    return gate_worker


def read_gate_json(artifact_dir: Path) -> dict[str, Any]:
    """`<artifact_dir>/gate.json` as written by the gate stage.

    Raises:
        FileNotFoundError: If the gate stage has not written it.
    """
    return dict(json.loads(require_artifact(Path(artifact_dir) / GATE_FILE, "gate").read_text()))


def gate_outcome_counts(manifest: ClipManifest) -> dict[str, int]:
    """`{"accepted", "rejected", "error", "pending"}` over the manifest's gate column: `done` is accepted, `failed` with
    an error starting with `GATE_REJECTED_PREFIX` is rejected, any other `failed` is an error."""
    counts = dict.fromkeys(("accepted", "rejected", "error", "pending"), 0)
    for clip_id in manifest.clip_ids():
        status = manifest.status(clip_id, "gate")
        if status.state == "done":
            counts["accepted"] += 1
        elif status.state == "failed":
            counts["rejected" if status.error.startswith(GATE_REJECTED_PREFIX) else "error"] += 1
        else:
            counts["pending"] += 1
    return counts
