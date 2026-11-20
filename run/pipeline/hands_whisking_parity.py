"""Whisking parity and speed run for the batched HaWoR hand stage, against DO AS I DO's shipped HaWoR output.

GPU (under the shared gpu.lock):
    FASTDO_HAWOR_ROOT=<baseline>/reconstruction/modules/HaWoR uv run python run/pipeline/hands_whisking_parity.py

`run_parity_calls` decodes `hands_parity.video` at `hands_parity.stride` into `<job_dir>/whisking_frames.npz` and makes
three driver calls: (a) the single clip; (b) `n_copies` copies in one call with `clips_per_group = n_copies`; (c) the
single clip with `temporal="sampled"`. It writes nothing outside `job_dir`. `main` then writes `hands_parity.report`
and appends a "Batched hands" section to `hands_parity.notes_path`.

`parity_metrics` returns `{"left"|"right": {"mean_mm", "p95_mm", "max_mm", "root_rel_mean_mm"}, "both": {"mean_mm",
"p95_mm", "max_mm"}, "n_eval_rows": int, "per_row": {"frame_idx", "left_mm", "right_mm"}}`. Evaluation rows are ours
(the reference has no infilled key): rows whose `frame_idx` lies in an inclusive range with both hands valid and
neither infilled.

Report top-level keys:
- `single`, `sampled`: `parity_metrics` of calls (a) and (c) over `both_detected_ranges`;
- `timing`: `load_seconds`, `warmup_seconds`, `compute_seconds` and the per-stage `seconds` of call (a);
- `copies`: `max_abs_joint_diff_mm`, `boxes_identical`, `valid_identical` and `infilled_identical` against call (a);
- `gap_rows`: the left rows at `gap_frame_idx`, with `infilled`, `infilled_frame_idx`, `per_row_mm`, `mean_mm`,
  `all_rows_valid`;
- `peak_cuda_gb`: per call (`single`, `copies`, `sampled`);
- `baseline_profile`, `baseline_note`: `BASELINE_PROFILE` and its caveat;
- `like_for_like_speedup`: `BASELINE_PROFILE["compute_seconds"]` over call (a)'s `seconds["total"]`.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from omegaconf import DictConfig

from fastdo.bench.gpu_budget import apply_cpu_budget, apply_gpu_budget
from fastdo.hands.hawor_batched import (
    HandsCallReport,
    HandsClipInput,
    hands_config_from_cfg,
    run_hands_batched,
)
from fastdo.io.video import decode_frames, probe_fps

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
SIDES = ("left", "right")

# One profiled pass (2026-09-13) of HaWoR's demo path on the whisking clip.
BASELINE_PROFILE: dict[str, float] = {
    "demo_wall_seconds": 184.7,
    "vis_render_seconds": 94.7,
    "imports_seconds": 4.63,
    "no_vis_wall_seconds": 56.79,
    "detect_track_seconds_excl_load": 9.00,
    "hawor_load_seconds": 7.81,
    "motion_inference_seconds": 10.85,
    "slam_mask_render_seconds": 21.81,
    "motion_remainder_seconds": 5.45,
    "infiller_seconds": 1.52,
    "infiller_forward_seconds": 0.03,
    "mano_save_seconds": 0.15,
    "compute_seconds": 26.98,
    "compute_seconds_remainder_as_slam": 21.53,
}
BASELINE_NOTE = ("The 26.98 s like-for-like baseline still includes about 0.8 s of infiller checkpoint load and the "
                 "first-call YOLO warm-up, so it overstates the baseline by about 1-2 s.")


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _rows_of(frame_idx: np.ndarray, wanted: np.ndarray, source: str) -> np.ndarray:
    position = {int(f): i for i, f in enumerate(np.asarray(frame_idx).reshape(-1))}
    missing = [int(f) for f in wanted if int(f) not in position]
    if missing:
        raise ValueError(f"{source} has no row for frame index {missing[0]}")
    return np.array([position[int(f)] for f in wanted], dtype=np.int64)


def parity_metrics(ours_npz: Path, reference_npz: Path, ranges: Sequence[Sequence[int]]) -> dict[str, Any]:
    """Camera-frame joint MPJPE (mm) of our hand npz against a HaWoR `all_hand_meshes.npz` (see the module docstring).

    Raises:
        ValueError: If no row qualifies, or the reference lacks one of the evaluation frames.
    """
    ours, ref = _load(ours_npz), _load(reference_npz)
    frame_idx = ours["frame_idx"].astype(np.int64)
    in_range = np.zeros(frame_idx.shape, dtype=bool)
    for lo, hi in ranges:
        in_range |= (frame_idx >= int(lo)) & (frame_idx <= int(hi))
    mask = in_range & ours["left_valid"] & ours["right_valid"] & ~ours["left_infilled"] & ~ours["right_infilled"]
    if not mask.any():
        raise ValueError(f"{ours_npz}: no evaluation rows in {list(ranges)}")
    ref_rows = _rows_of(ref.get("frame_idx", np.arange(ref["right_joints"].shape[0])), frame_idx[mask], str(reference_npz))
    metrics: dict[str, Any] = {}
    errors: dict[str, np.ndarray] = {}
    for side in SIDES:
        jo = ours[f"{side}_joints"][mask].astype(np.float64)
        jr = ref[f"{side}_joints"][ref_rows].astype(np.float64)
        err = np.linalg.norm(jo - jr, axis=-1).mean(axis=-1) * 1000.0
        root_rel = np.linalg.norm((jo - jo[:, :1]) - (jr - jr[:, :1]), axis=-1).mean(axis=-1) * 1000.0
        metrics[side] = {"mean_mm": float(err.mean()), "p95_mm": float(np.percentile(err, 95)),
                         "max_mm": float(err.max()), "root_rel_mean_mm": float(root_rel.mean())}
        errors[side] = err
    both = np.concatenate([errors["left"], errors["right"]])
    metrics["both"] = {"mean_mm": float(both.mean()), "p95_mm": float(np.percentile(both, 95)), "max_mm": float(both.max())}
    metrics["n_eval_rows"] = int(mask.sum())
    metrics["per_row"] = {"frame_idx": frame_idx[mask].tolist(), "left_mm": errors["left"].tolist(),
                          "right_mm": errors["right"].tolist()}
    return metrics


def decode_clip_frames(video: Path, stride: int, out: Path) -> Path:
    """Decodes `video` at `stride` (ffmpeg) into an uncompressed frames npz: `frames`, `frame_idx`, `fps`."""
    frames, idx = decode_frames(Path(video), stride=stride, backend="ffmpeg")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        np.savez(f, frames=np.ascontiguousarray(frames.numpy(), dtype=np.uint8), frame_idx=np.asarray(idx, np.int64),
                 fps=np.array(probe_fps(Path(video)) / stride, dtype=np.float64))
    return out


def _require_done(report: HandsCallReport, name: str) -> None:
    failed = [clip for clip in report.clips if clip.status != "done"]
    if failed:
        raise RuntimeError(f"parity call {name!r}: clip {failed[0].clip_id} failed: {failed[0].error}")


def run_parity_calls(cfg: DictConfig, job_dir: Path) -> dict[str, Any]:
    """The three driver calls of the parity run (see the module docstring); every file lands under `job_dir`."""
    job_dir, p = Path(job_dir), cfg.hands_parity
    frames_npz = decode_clip_frames(Path(p.video), int(p.stride), job_dir / "whisking_frames.npz")
    hands_cfg, focal, n_copies = hands_config_from_cfg(cfg), float(p.focal_px), int(p.n_copies)

    single_npz = job_dir / "single" / "whisking_hands.npz"
    single = run_hands_batched([HandsClipInput("whisking", frames_npz, focal, single_npz)], hands_cfg, job_dir / "single")
    _require_done(single, "single")
    copies_npz = [job_dir / "copies" / f"copy{i}_hands.npz" for i in range(n_copies)]
    copies = run_hands_batched([HandsClipInput(f"copy{i}", frames_npz, focal, path) for i, path in enumerate(copies_npz)],
                               replace(hands_cfg, clips_per_group=n_copies), job_dir / "copies")
    _require_done(copies, "copies")
    sampled_npz = job_dir / "sampled" / "whisking_hands.npz"
    sampled = run_hands_batched([HandsClipInput("whisking", frames_npz, focal, sampled_npz)],
                                replace(hands_cfg, temporal="sampled"), job_dir / "sampled")
    _require_done(sampled, "sampled")
    return {"frames_npz": frames_npz, "single_report": single, "single_npz": single_npz, "copies_report": copies,
            "copies_npz": copies_npz, "sampled_report": sampled, "sampled_npz": sampled_npz}


def copies_comparison(single_npz: Path, copies_npz: Sequence[Path]) -> dict[str, Any]:
    """Each copy against the single call: boxes, valid and infilled bit-identical; max abs joint difference (mm)."""
    ref = _load(single_npz)
    worst, same = 0.0, {"box": True, "valid": True, "infilled": True}
    for path in copies_npz:
        other = _load(path)
        for side in SIDES:
            for key in same:
                same[key] &= bool(np.array_equal(ref[f"{side}_{key}"], other[f"{side}_{key}"]))
            rows = ref[f"{side}_valid"]
            if rows.any():
                diff = other[f"{side}_joints"][rows].astype(np.float64) - ref[f"{side}_joints"][rows]
                worst = max(worst, float(np.abs(diff).max()) * 1000.0)
    return {"max_abs_joint_diff_mm": worst, "boxes_identical": same["box"], "valid_identical": same["valid"],
            "infilled_identical": same["infilled"]}


def gap_rows(single_npz: Path, reference_npz: Path, gap_frame_idx: Sequence[int]) -> dict[str, Any]:
    """Our left-hand rows at HaWoR's infilled frames: flags and mean joint error (mm) against HaWoR's infill."""
    ours, ref = _load(single_npz), _load(reference_npz)
    wanted = np.asarray(list(gap_frame_idx), dtype=np.int64)
    rows = _rows_of(ours["frame_idx"], wanted, str(single_npz))
    ref_rows = _rows_of(ref.get("frame_idx", np.arange(ref["left_joints"].shape[0])), wanted, str(reference_npz))
    err = np.linalg.norm(ours["left_joints"][rows].astype(np.float64) - ref["left_joints"][ref_rows], axis=-1)
    per_row = err.mean(axis=-1) * 1000.0
    return {"frame_idx": wanted.tolist(), "infilled": ours["left_infilled"][rows].tolist(),
            "infilled_frame_idx": ours["frame_idx"][ours["left_infilled"]].tolist(), "per_row_mm": per_row.tolist(),
            "mean_mm": float(per_row.mean()),
            "all_rows_valid": bool(ours["left_valid"].all() and ours["right_valid"].all())}


def build_report(cfg: DictConfig, calls: dict[str, Any]) -> dict[str, Any]:
    p = cfg.hands_parity
    ranges = [list(r) for r in p.both_detected_ranges]
    single: HandsCallReport = calls["single_report"]
    seconds = dict(single.clips[0].seconds)
    return {
        "single": parity_metrics(calls["single_npz"], Path(p.reference_npz), ranges),
        "sampled": parity_metrics(calls["sampled_npz"], Path(p.reference_npz), ranges),
        "timing": {"load_seconds": single.load_seconds, "warmup_seconds": single.warmup_seconds,
                   "compute_seconds": single.compute_seconds, "seconds": seconds},
        "copies": copies_comparison(calls["single_npz"], calls["copies_npz"]),
        "gap_rows": gap_rows(calls["single_npz"], Path(p.reference_npz), list(p.gap_frame_idx)),
        "peak_cuda_gb": {name: calls[f"{name}_report"].peak_cuda_gb for name in ("single", "copies", "sampled")},
        "baseline_profile": dict(BASELINE_PROFILE),
        "baseline_note": BASELINE_NOTE,
        "like_for_like_speedup": BASELINE_PROFILE["compute_seconds"] / max(float(seconds["total"]), 1e-9),
    }


def notes_section(report: dict[str, Any]) -> str:
    s, t, c, g = report["single"], report["timing"], report["copies"], report["gap_rows"]
    stages = ", ".join(f"{k} {v:.3f}" for k, v in t["seconds"].items() if k != "total")
    stamp = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    return "\n".join([
        "", f"## Batched hands: whisking parity and speed ({stamp})", "",
        (f"- Single call, {s['n_eval_rows']} both-detected rows: both-hands MPJPE mean {s['both']['mean_mm']:.2f} mm "
         f"(left {s['left']['mean_mm']:.2f}, right {s['right']['mean_mm']:.2f}), p95 {s['both']['p95_mm']:.2f} mm, "
         f"max {s['both']['max_mm']:.2f} mm; root-relative {s['left']['root_rel_mean_mm']:.2f} / "
         f"{s['right']['root_rel_mean_mm']:.2f} mm."),
        (f"- Speed: {t['seconds']['total']:.3f} s per clip excluding load ({stages}); load {t['load_seconds']:.2f} s, "
         f"warm-up {t['warmup_seconds']:.2f} s, call compute {t['compute_seconds']:.3f} s; peak CUDA "
         f"{report['peak_cuda_gb']['single']:.2f} GB. Like-for-like baseline {BASELINE_PROFILE['compute_seconds']} s: "
         f"{report['like_for_like_speedup']:.1f}x."),
        (f"- Copies in one call: max abs joint difference {c['max_abs_joint_diff_mm']:.5f} mm, boxes identical "
         f"{c['boxes_identical']}, peak {report['peak_cuda_gb']['copies']:.2f} GB."),
        (f"- Left gap rows {g['frame_idx']}: infilled {g['infilled']}, {g['mean_mm']:.2f} mm mean against HaWoR's "
         "infiller output."),
        f"- Sampled-temporal ablation: both-hands mean {report['sampled']['both']['mean_mm']:.2f} mm.",
        f"- {BASELINE_NOTE}", ""])


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    apply_cpu_budget()
    apply_gpu_budget()
    calls = run_parity_calls(cfg, Path(cfg.hands_parity.job_dir))
    report = build_report(cfg, calls)
    out = Path(cfg.hands_parity.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    notes = Path(cfg.hands_parity.notes_path)
    notes = notes if notes.is_absolute() else REPO_ROOT / notes
    with notes.open("a") as f:
        f.write(notes_section(report))
    logger.info("wrote %s and appended %s", out, notes)


if __name__ == "__main__":
    main()
