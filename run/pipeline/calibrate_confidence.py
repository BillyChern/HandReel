"""Calibrate `SolverConfig.confidence_scale` on DexYCB oracle runs; report the same statistics on synthetic clips.

Reads `eval_solver_dexycb.py` reports (`dexycb.mode=oracle`, one per init mode in
`confidence_calibration.init_modes`, all over the same sequences), turns each frame's confidence back into the
scale-free statistic, and finds the smallest two-significant-figure scale at which frames with translation error
< 5 mm score > 0.5 on at least 90 % of them (`fastdo.bench.confidence_calibration`). The DexYCB frames alone bind the
scale. The synthetic box fixture of
tests/solver/test_solve_synthetic.py (solved here, 5 cm / 15 deg perturbed init, `confidence_calibration.synthetic`)
is reported at that scale, never binding: `scale_synthetic` records the scale it alone would need, which its
occluded frames (confidence 0.0 by definition) push up. Shipping the larger of the two gave
1.6 on the CPU run at mass floor 0.7, which is not the accepted 1.4. At the shipped scale it reports
Spearman(confidence, -error), coverage (also over supported frames) and false confidence over all frames, per init
mode and per sequence, for both sets, plus the median within-clip Spearman over the DexYCB sequences (the
confidence is a within-clip relative signal). The JSON goes to `confidence_calibration.out`;
tests/bench/test_confidence_calibration.py checks the thresholds against it and `SolverConfig`'s defaults. Promote a
run to `outputs/eval/confidence_calibration.json` only if its `confidence_scale` equals
`SolverConfig().confidence_scale`; write it elsewhere first.

Example (CPU; drop CUDA_VISIBLE_DEVICES= for the GPU):
  for init in pnp gt gt_perturbed; do
    CUDA_VISIBLE_DEVICES= uv run python run/pipeline/eval_solver_dexycb.py paths.data=${FASTDO_ROOT}/data \
        paths.outputs=${FASTDO_ROOT}/outputs/confidence_calibration/$init dexycb.max_sequences=3 \
        dexycb.init=$init dexycb.protocol_metrics=false
  done
  O=${FASTDO_ROOT}/outputs/confidence_calibration
  CUDA_VISIBLE_DEVICES= uv run python run/pipeline/calibrate_confidence.py \
      paths.outputs=${FASTDO_ROOT}/outputs \
      "confidence_calibration.reports=[$O/pnp/eval/dexycb_oracle.json,$O/gt/eval/dexycb_oracle.json,\
$O/gt_perturbed/eval/dexycb_oracle.json]"
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from fastdo.bench.confidence_calibration import (
    FrameSamples,
    calibrate_scale,
    calibration_summary,
    frame_translation_errors,
    merge_samples,
    samples_from_eval_report,
    statistic_from_confidence,
    within_clip_spearman_median,
)
from fastdo.bench.gpu_budget import apply_cpu_budget, apply_gpu_budget
from fastdo.geometry.lie import se3_exp
from fastdo.solver import (
    MeshSamples,
    SolverConfig,
    SyntheticConfig,
    TrajectoryObservations,
    make_synthetic_clip,
    solve_trajectory,
)

logger = logging.getLogger(__name__)


def check_reports(reports: list[dict[str, Any]], init_modes: list[str]) -> None:
    """Raises `ValueError` unless the reports are oracle runs, one per init mode of `init_modes`, over one sequence set,
    all at one confidence mass floor."""
    modes = [str(r.get("init_mode")) for r in reports]
    if sorted(modes) != sorted(init_modes):
        raise ValueError(f"need one report per init mode {sorted(init_modes)}, got {modes}")
    if any(r.get("mode") != "oracle" for r in reports):
        raise ValueError(f"calibration uses oracle runs only, got modes {[r.get('mode') for r in reports]}")
    seq_sets = {tuple(sorted(row["sequence"] for row in r["rows"])) for r in reports}
    if len(seq_sets) != 1:
        raise ValueError(f"reports cover different sequences: {sorted(seq_sets)}")
    floors = {float(r["confidence_mass_floor"]) for r in reports}
    if len(floors) != 1:
        raise ValueError(f"reports were solved at different confidence mass floors: {sorted(floors)}")


def _perturbed(poses: torch.Tensor, seed: int) -> torch.Tensor:
    """The 5 cm / 15 deg left perturbation of tests/solver/test_solve_synthetic.py's `_perturbed_init`."""
    g = torch.Generator().manual_seed(seed)
    xi = torch.zeros(poses.shape[0], 6)
    xi[:, :3] = 0.05 * torch.nn.functional.normalize(torch.randn(poses.shape[0], 3, generator=g), dim=1)
    xi[:, 3:] = math.radians(15.0) * torch.nn.functional.normalize(torch.randn(poses.shape[0], 3, generator=g), dim=1)
    return se3_exp(xi.double()).float() @ poses


def synthetic_samples(occlusions: list[float], seeds: list[int], device: str) -> FrameSamples:
    """Solves the box fixture of `test_recovers_trajectory_visibility` (shipped solver defaults at n_obs 512 /
    n_mesh 1024) per occlusion and clip seed and returns its per-frame samples."""
    parts = []
    for occlusion in occlusions:
        for seed in seeds:
            clip = make_synthetic_clip(SyntheticConfig(n_frames=30, image_hw=(240, 320), n_surface=1024, n_tracks=48,
                                                       occlusion=occlusion, depth_noise=0.002, shape="box"), seed=seed)
            obs = TrajectoryObservations(intrinsics=clip.intrinsics, masks=clip.masks, hand_masks=clip.hand_masks,
                                         pointmaps=clip.pointmaps, tracks=clip.tracks, track_vis=clip.track_vis,
                                         anchors=clip.anchors)
            cfg = SolverConfig(n_obs=512, n_mesh=1024, device=device)
            sol = solve_trajectory(MeshSamples(clip.surface_points, clip.surface_normals), obs,
                                   _perturbed(clip.poses_gt, seed=1), clip.scale_gt * 1.2, cfg)
            t = int(clip.poses_gt.shape[0])
            errors = frame_translation_errors(sol.poses.numpy(), clip.poses_gt.numpy(), np.ones(t, dtype=bool))
            parts.append(FrameSamples(statistic=statistic_from_confidence(sol.confidence.numpy(), cfg.confidence_scale),
                                      error_m=np.asarray(errors, np.float64), init_mode=("perturbed_5cm_15deg",) * t,
                                      sequence=(f"box_occ{occlusion}_seed{seed}",) * t))
    return merge_samples(parts)


def _git_commit(cwd: Path) -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    apply_cpu_budget()
    apply_gpu_budget()
    cc = cfg.confidence_calibration
    orig_cwd = Path(hydra.utils.get_original_cwd())
    paths = [p if p.is_absolute() else (orig_cwd / p).resolve() for p in (Path(str(x)) for x in cc.reports)]
    reports = [json.loads(p.read_text()) for p in paths]
    init_modes = [str(m) for m in cc.init_modes]
    check_reports(reports, init_modes)
    thr, min_conf, cov = float(cc.error_threshold_m), float(cc.min_confidence), float(cc.target_coverage)
    samples = merge_samples([samples_from_eval_report(r) for r in reports])
    scale_dexycb = calibrate_scale(samples.statistic, samples.error_m, thr, min_conf, cov)
    syn_samples: FrameSamples | None = None
    scale_synthetic: float | None = None
    if bool(cc.synthetic):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        syn_samples = synthetic_samples([float(o) for o in cc.synthetic_occlusions],
                                        [int(s) for s in cc.synthetic_seeds], device)
        scale_synthetic = calibrate_scale(syn_samples.statistic, syn_samples.error_m, thr, min_conf, cov)
    # The DexYCB coverage gate alone binds the scale; the synthetic fixture is reported at it (module docstring).
    scale = scale_dexycb
    dexycb = calibration_summary(samples, scale, thr, min_conf)
    synthetic = None if syn_samples is None else calibration_summary(syn_samples, scale, thr, min_conf)
    out = {"created_utc": datetime.now(UTC).isoformat(timespec="seconds"), "git_commit": _git_commit(orig_cwd),
           "confidence_scale": scale, "scale_dexycb": scale_dexycb, "scale_synthetic": scale_synthetic,
           "binding": "dexycb", "synthetic_binds": False,
           "dexycb_within_clip_spearman_median": within_clip_spearman_median(dexycb),
           "confidence_mass_floor": float(reports[0]["confidence_mass_floor"]),
           "report_confidence_scale": sorted({float(r["confidence_scale"]) for r in reports}),
           "mode": "oracle", "init_modes": sorted(init_modes),
           "sequences": sorted({row["sequence"] for row in reports[0]["rows"]}),
           "reports": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths],
           "error_threshold_m": thr, "min_confidence": min_conf, "target_coverage": float(cc.target_coverage),
           "spearman_convention": "Spearman rho between confidence and the negated per-frame translation error",
           "dexycb": dexycb, "synthetic": synthetic,
           "synthetic_fixture": {"occlusions": list(cc.synthetic_occlusions), "seeds": list(cc.synthetic_seeds),
                                 "init": "5 cm / 15 deg perturbed, scale x1.2", "n_obs": 512, "n_mesh": 1024}}
    out_path = Path(str(cc.out))
    out_path = out_path if out_path.is_absolute() else (orig_cwd / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, allow_nan=False))
    for name, block in (("dexycb", dexycb), ("synthetic", synthetic)):
        if block is not None:
            logger.info("%s at scale %.3g: %s", name, scale, block["all"])
    logger.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
