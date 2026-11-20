"""Fit and save the qc stage's `AcceptanceRule` from DexYCB calibration clips run through the engine.

Protocol:

1. The engine is run on the calibration sequences with the GT YCB mesh as their shape and the deployed
   batched HaWoR hands (`run/pipeline/bench_throughput.py ... bench.dexycb_shape=template
   'bench.stages=[decode,mask,shape,intrinsics,hands,geometry,tracks,solve]' bench.artifact_root=<artifact_root>`).
   The sequence ids and their sha256 are fixed before any output. Every reserved set and decision set is
   excluded: all subject 07/08 sequences, the NEW 26 and the OLD held-out 9.
2. This script (CPU) reads each clip's artifacts: the qc stage's own features and the `pose_error_m` label, in the
   template's frame, against DexYCB's GT object poses (`fastdo.engine.qc_calibration`). It writes
   `acceptance_calibration.features_out` first, so a refused fit still leaves the data, then fits the rule and saves it
   with its provenance to `acceptance_calibration.out_path` (`fastdo.qc.rule_io`, version 1).

A fit without the conformal coverage guarantee is refused (`allow_unguaranteed: false`); alpha and tau are not loosened
to get one. An existing `out_path` is refused unless `overwrite`.

  cd ${FASTDO_ROOT} && uv run --extra dev python run/pipeline/calibrate_acceptance.py \
    paths.root=${FASTDO_ROOT} 'acceptance_calibration.dexycb_sequences=[<ids>]'
"""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from omegaconf import DictConfig

from fastdo.bench.bench_clips import dexycb_clips
from fastdo.bench.gpu_budget import apply_cpu_budget, apply_gpu_budget
from fastdo.engine.qc_calibration import (
    GtPosesFn,
    collect_calibration_clips,
    coverage_guaranteed,
    fit_rule_from_clips,
    write_calibration_features,
)
from fastdo.engine.workers_qc import FeaturesFn
from fastdo.io.datasets.dexycb import DexYCBReader
from fastdo.qc import AcceptanceRule, qc_features
from fastdo.qc.rule_io import save_acceptance_rule

logger = logging.getLogger(__name__)

SHAPE_SOURCE = "template (GT YCB mesh)"
LABEL = ("pose_error_m: mean per-frame translation error (m) of the solved poses in the GT YCB template's frame against "
         "DexYCB's GT object poses, over the solve rows whose GT pose is valid (confidence > 0)")


def _resolve(base: Path, path: Any) -> Path:
    p = Path(str(path))
    return p if p.is_absolute() else (base / p).resolve()


def _git_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=10,
                                check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    return commit if result.returncode == 0 and commit else None


def dexycb_gt_poses_fn(dexycb_root: Path, mano_pca: Path | None) -> GtPosesFn:
    """`sequence -> (object_pose (N,4,4) float64, confidence > 0)` from `DexYCBReader.load`."""
    reader = DexYCBReader(dexycb_root, mano_pca=mano_pca)

    def gt_poses(sequence: str) -> tuple[np.ndarray, np.ndarray]:
        record = reader.load(sequence)
        return np.asarray(record.object_pose, dtype=np.float64), np.asarray(record.confidence) > 0

    return gt_poses


def run_calibration(cfg: DictConfig, root: Path, gt_poses_fn: GtPosesFn | None = None,
                    features_fn: FeaturesFn = qc_features,
                    clip_table: Sequence[tuple[str, str]] | None = None) -> AcceptanceRule:
    """Collects the calibration clips, writes `features_out`, fits the rule and saves it with its provenance.

    Args:
        cfg: The root config (`acceptance_calibration`, `dexycb`, `qc_stage.laplace_std_threshold`).
        root: Base of relative paths.
        gt_poses_fn: GT per sequence; default `DexYCBReader(dexycb.root, dexycb.mano_pca).load`.
        features_fn: `qc_features`-compatible feature function (tests inject fakes).
        clip_table: `(clip_id, sequence)` pairs; default `dexycb_clips(dexycb.root, dexycb_sequences)`.

    Raises:
        ValueError: If `acceptance_calibration.dexycb_sequences` is empty without `clip_table`, or the fit is refused
            (class counts, or the coverage guarantee).
        FileExistsError: If `out_path` exists and `overwrite` is false.
    """
    cal = cfg.acceptance_calibration
    sequences = [str(seq) for seq in (cal.get("dexycb_sequences") or [])]
    if clip_table is None and not sequences:
        raise ValueError("acceptance_calibration.dexycb_sequences is empty: list the calibration sequence ids (fixed, "
                         "with their sha256 recorded, before any output)")
    out_path, features_out = _resolve(root, cal.out_path), _resolve(root, cal.features_out)
    if out_path.exists() and not bool(cal.overwrite):
        raise FileExistsError(f"{out_path} exists; set acceptance_calibration.overwrite=true to replace the saved rule")
    dexycb_root = _resolve(root, cfg.dexycb.root)
    if clip_table is None:
        clip_table = [(clip.clip_id, seq) for clip, seq in zip(dexycb_clips(dexycb_root, sequences), sequences,
                                                               strict=True)]
    table = [(str(clip_id), str(seq)) for clip_id, seq in clip_table]
    if gt_poses_fn is None:
        gt_poses_fn = dexycb_gt_poses_fn(dexycb_root, _resolve(root, cfg.dexycb.mano_pca))
    alpha, tau, fit_fraction, seed = float(cal.alpha), float(cal.tau), float(cal.fit_fraction), int(cal.seed)
    laplace_std_threshold = float(cfg.qc_stage.laplace_std_threshold)
    artifact_root = _resolve(root, cal.artifact_root)

    clips, skipped = collect_calibration_clips(artifact_root, table, gt_poses_fn, features_fn, laplace_std_threshold)
    write_calibration_features(features_out, clips, skipped, tau)
    n_good = sum(clip.pose_error_m < tau for clip in clips)
    logger.info("calibration: %d clips labelled (%d good, %d bad at tau %g m), %d skipped; features in %s", len(clips),
                n_good, len(clips) - n_good, tau, len(skipped), features_out)
    rule = fit_rule_from_clips(clips, alpha=alpha, tau=tau, fit_fraction=fit_fraction, seed=seed,
                               allow_unguaranteed=bool(cal.allow_unguaranteed))
    provenance: dict[str, Any] = {
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), "git_commit": _git_commit(root),
        "sequences": [seq for _, seq in table], "clip_ids": [clip_id for clip_id, _ in table], "n_clips": len(clips),
        "n_skipped": len(skipped), "skipped": skipped, "n_good": n_good, "n_bad": len(clips) - n_good, "alpha": alpha,
        "tau": tau, "fit_fraction": fit_fraction, "seed": seed, "laplace_std_threshold": laplace_std_threshold,
        "artifact_root": str(artifact_root), "coverage_guaranteed": coverage_guaranteed(rule.calibration),
        "shape_source": SHAPE_SOURCE, "label": LABEL, "features_file": str(features_out),
    }
    save_acceptance_rule(out_path, rule, provenance)
    logger.info("saved the acceptance rule to %s (threshold %.4f on %d good calibration clips, coverage guaranteed: %s)",
                out_path, rule.calibration.threshold, rule.calibration.n_calibration, provenance["coverage_guaranteed"])
    return rule


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    apply_cpu_budget()
    apply_gpu_budget()
    try:
        root = Path(hydra.utils.get_original_cwd())
    except ValueError:                    # called with a composed config, outside a Hydra run (tests)
        root = Path.cwd()
    run_calibration(cfg, root)


if __name__ == "__main__":
    main()
