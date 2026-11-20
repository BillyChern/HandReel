"""Calibration statistics of the solver's per-frame confidence against per-frame translation error.

A solve's confidence is `exp(-z / scale)` (`fastdo.solver.confidence`), so a result computed at any scale gives back
the scale-free statistic `z = -scale * ln(confidence)`; it is +inf where the confidence is 0.0 (no pointmap support,
or float32 underflow far past any threshold). The scale moves only the coverage: the Spearman correlation depends on
the ranking of `z` alone.

The confidence is a WITHIN-CLIP relative signal: each clip is normalized by its own noise floor, so its acceptance
statistic is the within-clip Spearman (`within_clip_spearman_median`); the pooled Spearman over clips is reported, not
gated.

Conventions:
* translation error: `|t_pred - t_gt|` per frame (metres), the per-frame term of `qc.pose_error_m` (the qc label);
* `spearman`: Spearman's rho between confidence and the NEGATED error, so 1.0 means confidence orders frames exactly
  by increasing error;
* `coverage`: the fraction of accurate frames (error < `error_threshold_m`) whose confidence exceeds `min_confidence`;
  `coverage_supported` the same over accurate frames with pointmap support (frames without it score 0.0 by
  definition, so a clip with occluded frames cannot reach full `coverage` at any scale);
* `false_confidence`: the fraction of inaccurate frames whose confidence exceeds `min_confidence`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import spearmanr

__all__ = ["ERROR_THRESHOLD_M", "MIN_CONFIDENCE", "TARGET_COVERAGE", "FrameSamples", "calibrate_scale",
           "calibration_summary", "confidence_at", "coverage", "frame_translation_errors", "merge_samples",
           "samples_from_eval_report", "spearman", "statistic_from_confidence", "within_clip_spearman_median"]

ERROR_THRESHOLD_M = 0.005
MIN_CONFIDENCE = 0.5
TARGET_COVERAGE = 0.9


@dataclass(frozen=True)
class FrameSamples:
    """Per-frame samples: the scale-free `statistic` (F,), `error_m` (F,), and each frame's `init_mode` and
    `sequence`."""

    statistic: np.ndarray
    error_m: np.ndarray
    init_mode: tuple[str, ...]
    sequence: tuple[str, ...]


def frame_translation_errors(pred_poses: np.ndarray, gt_poses: np.ndarray, gt_valid: np.ndarray) -> list[float | None]:
    """`|t_pred - t_gt|` (m) per frame of two (T,4,4) trajectories; None where `gt_valid` is False."""
    dist = np.linalg.norm(np.asarray(pred_poses, np.float64)[:, :3, 3] - np.asarray(gt_poses, np.float64)[:, :3, 3],
                          axis=1)
    return [float(d) if bool(v) else None for d, v in zip(dist, np.asarray(gt_valid, dtype=bool), strict=True)]


def statistic_from_confidence(confidence: np.ndarray, scale: float) -> np.ndarray:
    """`-scale * ln(confidence)`, +inf where the confidence is not positive."""
    c = np.asarray(confidence, dtype=np.float64)
    positive = c > 0.0
    return np.where(positive, -scale * np.log(np.where(positive, c, 1.0)), np.inf)


def confidence_at(statistic: np.ndarray, scale: float) -> np.ndarray:
    """`exp(-statistic / scale)`, 0.0 where the statistic is +inf."""
    z = np.asarray(statistic, dtype=np.float64)
    finite = np.isfinite(z)
    return np.where(finite, np.exp(-np.where(finite, z, 0.0) / scale), 0.0)


def spearman(confidence: np.ndarray, error_m: np.ndarray) -> float:
    """Spearman's rho between confidence and the negated error (ties averaged); NaN below two frames or for a
    constant input (no ranking to correlate)."""
    c, e = np.asarray(confidence, np.float64), np.asarray(error_m, np.float64)
    if c.size < 2 or bool(np.all(c == c[0])) or bool(np.all(e == e[0])):
        return math.nan
    return float(spearmanr(c, -e).statistic)


def coverage(confidence: np.ndarray, error_m: np.ndarray, error_threshold_m: float = ERROR_THRESHOLD_M,
             min_confidence: float = MIN_CONFIDENCE) -> float:
    """Fraction of frames with error < `error_threshold_m` whose confidence is > `min_confidence` (NaN if none)."""
    accurate = np.asarray(error_m, np.float64) < error_threshold_m
    if not accurate.any():
        return math.nan
    return float((np.asarray(confidence, np.float64)[accurate] > min_confidence).mean())


def calibrate_scale(statistic: np.ndarray, error_m: np.ndarray, error_threshold_m: float = ERROR_THRESHOLD_M,
                    min_confidence: float = MIN_CONFIDENCE, target_coverage: float = TARGET_COVERAGE) -> float:
    """The smallest scale on a two-significant-figure grid whose `coverage` is at least `target_coverage`.

    Raises:
        ValueError: no accurate frame, or too many accurate frames have no support (+inf) for any scale to reach it.
    """
    z = np.asarray(statistic, np.float64)
    accurate = np.sort(z[np.asarray(error_m, np.float64) < error_threshold_m])
    if accurate.size == 0:
        raise ValueError(f"no frame has a translation error below {error_threshold_m} m")
    needed = accurate[math.ceil(target_coverage * accurate.size - 1e-9) - 1]
    if not math.isfinite(needed):
        raise ValueError(f"coverage {target_coverage} is unreachable: too many accurate frames have no support")
    x = max(float(needed) / math.log(1.0 / min_confidence), 1e-6)   # exp(-needed / x) == min_confidence
    step = 10.0 ** (math.floor(math.log10(x)) - 1)
    scale = float(f"{(math.floor(x / step) + 1) * step:.12g}")
    while coverage(confidence_at(z, scale), error_m, error_threshold_m, min_confidence) < target_coverage:
        scale = float(f"{scale + step:.12g}")
    return scale


def _summary(z: np.ndarray, err: np.ndarray, scale: float, error_threshold_m: float, min_confidence: float
             ) -> dict[str, float | int | None]:
    conf = confidence_at(z, scale)
    inaccurate = err >= error_threshold_m

    def finite(x: float) -> float | None:
        return x if math.isfinite(x) else None

    supported = conf > 0.0
    return {"n_frames": int(z.size), "n_accurate": int((~inaccurate).sum()), "n_unsupported": int((~supported).sum()),
            "spearman": finite(spearman(conf, err)),
            "coverage": finite(coverage(conf, err, error_threshold_m, min_confidence)),
            "coverage_supported": finite(coverage(conf[supported], err[supported], error_threshold_m, min_confidence)),
            "false_confidence": finite(float((conf[inaccurate] > min_confidence).mean()) if inaccurate.any()
                                       else math.nan),
            "median_error_mm": finite(float(np.median(err) * 1e3)) if err.size else None,
            "p90_error_mm": finite(float(np.quantile(err, 0.9) * 1e3)) if err.size else None,
            "median_confidence": finite(float(np.median(conf))) if conf.size else None,
            "median_statistic": finite(float(np.median(z))) if z.size else None}


def calibration_summary(samples: FrameSamples, scale: float, error_threshold_m: float = ERROR_THRESHOLD_M,
                        min_confidence: float = MIN_CONFIDENCE) -> dict[str, Any]:
    """Statistics at `scale` over all frames, per init mode and per sequence."""
    out: dict[str, Any] = {"all": _summary(samples.statistic, samples.error_m, scale, error_threshold_m,
                                           min_confidence)}
    for key, labels in (("by_init_mode", samples.init_mode), ("by_sequence", samples.sequence)):
        names = np.asarray(labels)
        out[key] = {name: _summary(samples.statistic[names == name], samples.error_m[names == name], scale,
                                   error_threshold_m, min_confidence) for name in sorted(set(labels))}
    return out


def within_clip_spearman_median(summary: Mapping[str, Any]) -> float | None:
    """Median over `calibration_summary(...)["by_sequence"]` of each sequence's Spearman (its frames pooled over init
    modes; the inits solve each sequence near-identically), skipping sequences without a ranking (None)."""
    values = [float(v["spearman"]) for v in summary["by_sequence"].values() if v["spearman"] is not None]
    return float(np.median(values)) if values else None


def samples_from_eval_report(report: Mapping[str, Any]) -> FrameSamples:
    """Frames with a valid ground-truth pose from one `eval_solver_dexycb.py` report (its `init_mode`,
    `confidence_scale` and the rows' `frame_confidence` / `frame_translation_error_m`)."""
    scale, init_mode = float(report["confidence_scale"]), str(report["init_mode"])
    z: list[float] = []
    err: list[float] = []
    seqs: list[str] = []
    for row in report["rows"]:
        conf, errors = row["frame_confidence"], row["frame_translation_error_m"]
        if len(conf) != len(errors) or not conf:
            raise ValueError(f"{row['sequence']}: {len(conf)} confidences for {len(errors)} errors")
        stat = statistic_from_confidence(np.asarray(conf), scale)
        for k, e in enumerate(errors):
            if e is not None:
                z.append(float(stat[k]))
                err.append(float(e))
                seqs.append(str(row["sequence"]))
    return FrameSamples(statistic=np.asarray(z, np.float64), error_m=np.asarray(err, np.float64),
                        init_mode=(init_mode,) * len(z), sequence=tuple(seqs))


def merge_samples(parts: Sequence[FrameSamples]) -> FrameSamples:
    return FrameSamples(statistic=np.concatenate([p.statistic for p in parts]),
                        error_m=np.concatenate([p.error_m for p in parts]),
                        init_mode=tuple(m for p in parts for m in p.init_mode),
                        sequence=tuple(s for p in parts for s in p.sequence))
