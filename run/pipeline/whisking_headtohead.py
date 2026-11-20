"""Whisking head-to-head: our real front end + M1 solver vs. the DO AS I DO baseline layout.

Runs on the demo the baseline ships (`third_party/do-as-i-do/reconstruction/whisking`): decodes
`whisking.mp4` at 10 fps, reuses the keyframe SAM 3D mesh, propagates SAM 2.1 masks and tracks
with CoTracker3 (both reused from `whisking.adapters_dir` when the adapter validation
already cached them for the same frames), loads HaWoR hands, runs MoGe-2 with a HaWoR-anchored
metric scale, then `pnp_initialize` + `solve_trajectory` at 10 fps, Lie-interpolates the result to
the baseline's 138 frames, and compares against `read_baseline_layout` (called exactly once).

The demo ships the keyframe mesh but not SAM 3's keyframe mask. Our trajectory starts from the shape
stage's own output, never from the baseline's optimized answer: SAM 3D's pre-optimization layout at the
reference frame 125 (`read_keyframe_layout`: raw pose, `mesh_scale_original`) places the mesh at our
nearest 10 fps keyframe (126, one source frame later), prompts SAM 2 and anchors the track queries, and
seeds the solver's scale. Cached SAM 2 masks / CoTracker tracks, when reused, were prompted from
the baseline's keyframe-126 silhouette instead; the details JSON records how well our prompt agrees with
that cached keyframe mask. `solver_seconds` covers everything after the front end: query anchoring, PnP
init + gating, the solve and the interpolation to 138 frames. Writes
`<out_dir>/headtohead_report.json` (six keys), `<out_dir>/headtohead_details.json`, and a
"Whisking head-to-head" section in `whisking.notes_path`.

Example:
  uv run python run/pipeline/whisking_headtohead.py paths.outputs=${FASTDO_ROOT}/outputs
"""

from __future__ import annotations

import json
import logging
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import hydra
import numpy as np
import torch
import trimesh
from omegaconf import DictConfig
from torch import Tensor

from fastdo.bench import Profiler
from fastdo.bench.gpu_budget import apply_cpu_budget, apply_gpu_budget
from fastdo.bench.headtohead import (
    axis_tilt_errors,
    interpolate_poses,
    mask_iou,
    principal_axis,
    render_silhouette,
    rotation_errors,
    translation_errors,
)
from fastdo.geometry.moge_model import run_moge
from fastdo.geometry.moge_pointmaps import (
    apply_metric_scale,
    clip_median_scale,
    hand_anchored_scale,
)
from fastdo.hands.hawor_npz import load_hawor_npz
from fastdo.io.video import decode_frames, sample_frame_indices
from fastdo.shape.masks_sam2 import propagate_masks_sam2
from fastdo.solver import (
    MeshSamples,
    SolverConfig,
    SyntheticConfig,
    TrajectoryObservations,
    anchors_from_pose,
    gate_init_poses,
    make_synthetic_clip,
    pnp_initialize,
    solve_trajectory,
)
from fastdo.teacher.layout_reader import read_baseline_layout, read_keyframe_layout
from fastdo.tracks.cotracker import (
    sample_query_points_in_mask,
    track_points_cotracker,
    tracks_in_masks,
)

logger = logging.getLogger(__name__)

REPORT_KEYS = ("mean_translation_error_m", "mean_rotation_error_rad", "reprojection_iou_ours",
               "reprojection_iou_baseline", "solver_seconds", "frontend_seconds")
_N_MESH = 2048
_N_ANCHOR_SURFACE = 16384
_QUERY_SEED = 0
_MASKS_CACHE = "masks_sam2_stride3.npz"
_TRACKS_CACHE = "tracks_cotracker_stride3.npz"
_CACHED_STAGE_NAMES = {"sam2": "masks_sam2/stride3_10fps", "cotracker": "tracks_cotracker/stride3_10fps"}
_NOTES_HEADING = "## Whisking head-to-head"


def _resolve(root: Path, base: Path, p: str) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return (base / path) if (base / path).exists() else (root / path).resolve()


def _sample_surface(mesh: trimesh.Trimesh, n: int, seed: int) -> tuple[Tensor, Tensor]:
    points, face_idx = trimesh.sample.sample_surface(mesh, n, seed=seed)
    pts = torch.as_tensor(np.asarray(points), dtype=torch.float32)
    nrm = torch.as_tensor(np.asarray(mesh.face_normals)[face_idx], dtype=torch.float32)
    if float((nrm * (pts - pts.mean(0))).sum(-1).mean()) < 0:  # outward normals for visible_samples
        nrm = -nrm
    return pts, nrm


def _posed(vertices: np.ndarray, pose: Tensor, scale: float) -> np.ndarray:
    P = pose.double().numpy()
    return scale * vertices @ P[:3, :3].T + P[:3, 3]


def _load_cache(path: Path, idx: list[int], keyframe_index: int | None = None) -> dict[str, np.ndarray] | None:
    """A cached adapter npz, or None when absent or computed for different frames / keyframe."""
    if not path.is_file():
        return None
    with np.load(path) as data:
        arrays = {k: np.asarray(data[k]) for k in data.files}
    if arrays["frame_idx"].tolist() != idx or (keyframe_index is not None
                                               and int(arrays["keyframe_index"]) != keyframe_index):
        logger.warning("%s was computed for other frames/keyframe; recomputing", path)
        return None
    return arrays


def _cached_seconds(adapters_dir: Path) -> dict[str, float]:
    """Measured stage seconds of the run that produced the adapter caches (profile_report.json)."""
    path = adapters_dir / "profile_report.json"
    if not path.is_file():
        return {}
    stages = {s["name"]: float(s["seconds"]) for s in json.loads(path.read_text())["stages"]}
    return {kind: stages[name] for kind, name in _CACHED_STAGE_NAMES.items() if name in stages}


def _hand_masks(hand_npz: Path, idx: list[int], K: Tensor, hw: tuple[int, int]) -> Tensor:
    """HaWoR hand-mesh silhouettes (both hands) at frames `idx`: the solver's hand occluder masks."""
    out = np.zeros((len(idx), *hw), dtype=bool)
    with np.load(hand_npz) as data:
        for side in ("left", "right"):
            if f"{side}_vertices" not in data.files:
                continue
            verts, faces = np.asarray(data[f"{side}_vertices"]), np.asarray(data[f"{side}_faces"])
            valid = np.asarray(data[f"{side}_valid"]) if f"{side}_valid" in data.files else np.ones(len(verts), bool)
            for i, f in enumerate(idx):
                if valid[f] and np.isfinite(verts[f]).all():
                    out[i] |= render_silhouette(verts[f].astype(np.float64), faces, K, hw)
    return torch.from_numpy(out)


def _warm_up_solver(cfg: SolverConfig) -> None:
    """One tiny synthetic solve, so `solver_seconds` measures the steady-state per-clip cost rather than
    the process's first-CUDA-solve context and kernel start-up (reported separately as `warmup/solver`)."""
    clip = make_synthetic_clip(SyntheticConfig(n_frames=3, image_hw=(120, 160), n_surface=512, n_tracks=16), seed=0)
    obs = TrajectoryObservations(intrinsics=clip.intrinsics, masks=clip.masks, hand_masks=clip.hand_masks,
                                 pointmaps=clip.pointmaps, tracks=clip.tracks, track_vis=clip.track_vis,
                                 anchors=clip.anchors)
    solve_trajectory(MeshSamples(points=clip.surface_points, normals=clip.surface_normals), obs, clip.poses_gt,
                     clip.scale_gt, cfg)


def run_headtohead(cfg: DictConfig, root: Path) -> dict[str, float]:
    """Runs the head-to-head and writes the report, details and notes section; returns the report."""
    w = cfg.whisking
    demo = _resolve(root, root, str(w.demo_root))
    video, mesh_path = _resolve(root, demo, str(w.video)), _resolve(root, demo, str(w.mesh_path))
    hand_npz, layout_json = _resolve(root, demo, str(w.hand_npz)), _resolve(root, demo, str(w.layout_json))
    adapters_dir, out_dir = _resolve(root, root, str(w.adapters_dir)), _resolve(root, root, str(w.out_dir))
    notes_path = _resolve(root, root, str(w.notes_path))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_full, stride = int(w.n_frames_full), int(w.stride)
    profiler = Profiler(clip_id="whisking", system="fastdo_m1", n_frames=n_full, fps=29.97,
                        sync_cuda=device == "cuda")

    base_poses, mesh_scale, K = read_baseline_layout(layout_json)
    shape_pose, shape_scale = read_keyframe_layout(layout_json, int(w.keyframe_frame_idx))
    with profiler.stage("frontend/decode"):
        frames, idx = decode_frames(video, stride=stride, max_frames=len(sample_frame_indices(n_full, stride)))
    hw = (int(frames.shape[1]), int(frames.shape[2]))
    k = int(np.argmin(np.abs(np.asarray(idx) - int(w.keyframe_frame_idx))))
    mesh = cast("trimesh.Trimesh", trimesh.load(str(mesh_path), process=False, force="mesh"))
    verts, faces = np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64)
    points, normals = _sample_surface(mesh, _N_MESH, seed=int(cfg.seed))
    dense_points, dense_normals = _sample_surface(mesh, _N_ANCHOR_SURFACE, seed=int(cfg.seed) + 2)

    cached_seconds = _cached_seconds(adapters_dir)
    reused: list[str] = []
    prompt = torch.from_numpy(render_silhouette(_posed(verts, shape_pose, shape_scale), faces, K, hw))
    masks_cache = _load_cache(adapters_dir / _MASKS_CACHE, idx)
    if masks_cache is not None:
        masks = torch.from_numpy(masks_cache["masks"]).to(torch.bool)
        reused.append("sam2")
    else:
        with profiler.stage("frontend/sam2"):
            masks = propagate_masks_sam2(frames, k, prompt, device=device).to(torch.bool)
    tracks_cache = _load_cache(adapters_dir / _TRACKS_CACHE, idx, keyframe_index=k)
    if tracks_cache is not None:
        tracks, track_vis = torch.from_numpy(tracks_cache["tracks"]), torch.from_numpy(tracks_cache["vis"])
        reused.append("cotracker")
    else:
        query_uv = sample_query_points_in_mask(prompt, int(w.n_track_queries),
                                               generator=torch.Generator().manual_seed(_QUERY_SEED))
        with profiler.stage("frontend/cotracker"):
            tracks, track_vis = track_points_cotracker(frames, k, query_uv, device=device)

    with profiler.stage("frontend/hands"):
        hand = load_hawor_npz(hand_npz, side=str(w.hand_side))
        hand_masks = _hand_masks(hand_npz, idx, K, hw)
    with profiler.stage("frontend/moge"):
        raw = run_moge(frames, device=device, intrinsics=K)
    with profiler.stage("frontend/metric_scale"):
        joints = torch.from_numpy(hand.joints[idx])
        joint_valid = torch.from_numpy(hand.valid[idx])[:, None].expand(-1, joints.shape[1])
        pointmaps = apply_metric_scale(raw, clip_median_scale(hand_anchored_scale(raw, joints, K, joint_valid)))
    del raw

    solver_cfg = SolverConfig(device=device, seed=int(cfg.seed))
    if device == "cuda" and bool(w.solver_warmup):
        with profiler.stage("warmup/solver"):
            _warm_up_solver(solver_cfg)
    with profiler.stage("solver/anchors"):
        anchors, anchor_valid = anchors_from_pose(dense_points, dense_normals, shape_pose, shape_scale, K, hw,
                                                  tracks[k])
        obs = TrajectoryObservations(intrinsics=K, masks=masks, hand_masks=hand_masks, pointmaps=pointmaps,
                                     tracks=tracks, track_vis=track_vis.to(torch.bool)
                                     & tracks_in_masks(masks, tracks) & anchor_valid[None], anchors=anchors)
    with profiler.stage("solver/pnp_init"):
        try:
            init_poses, n_inliers = pnp_initialize(anchors, anchor_valid, obs.tracks, obs.track_vis, K,
                                                   scale=shape_scale)
        except ValueError:
            logger.warning("pnp_initialize solved no frame; seeding every frame from the keyframe pose")
            init_poses = shape_pose[None].expand(len(idx), 4, 4).clone()
            n_inliers = torch.zeros(len(idx), dtype=torch.int64)
        init_poses, pnp_accepted = gate_init_poses(init_poses, n_inliers, points, shape_scale, obs, k, shape_pose)
    with profiler.stage("solver/solve"):
        sol = solve_trajectory(MeshSamples(points=points, normals=normals), obs, init_poses, shape_scale, solver_cfg)
    with profiler.stage("solver/interpolate"):
        ours_full = interpolate_poses(sol.poses, idx, n_full)

    iou_ours = [mask_iou(render_silhouette(_posed(verts, sol.poses[i], sol.scale), faces, K, hw), masks[i].numpy())
                for i in range(len(idx))]
    iou_base = [mask_iou(render_silhouette(_posed(verts, base_poses[f], mesh_scale), faces, K, hw), masks[i].numpy())
                for i, f in enumerate(idx)]
    stages = {s.name: s.seconds for s in profiler.report().stages}
    reused_seconds = {kind: cached_seconds.get(kind, math.nan) for kind in reused}
    frontend_measured = sum(v for name, v in stages.items() if name.startswith("frontend/"))
    report = {
        "mean_translation_error_m": float(translation_errors(ours_full, base_poses[:n_full]).mean()),
        "mean_rotation_error_rad": float(rotation_errors(ours_full, base_poses[:n_full]).mean()),
        "reprojection_iou_ours": float(np.nanmean(iou_ours)),
        "reprojection_iou_baseline": float(np.nanmean(iou_base)),
        "solver_seconds": sum(v for name, v in stages.items() if name.startswith("solver/")),
        "frontend_seconds": frontend_measured + sum(v for v in reused_seconds.values() if math.isfinite(v)),
    }
    details: dict[str, Any] = {
        "stage_seconds_this_run": stages, "reused_adapter_caches": reused_seconds, "device": device,
        "n_frames_solved": len(idx), "keyframe_frame_idx": idx[k], "solved_scale": sol.scale,
        "init_scale": shape_scale, "init_keyframe_pose": "SAM 3D pre-optimization layout, reference frame "
        f"{int(w.keyframe_frame_idx)} (read_keyframe_layout)", "baseline_mesh_scale": mesh_scale,
        "keyframe_prompt_iou_vs_sam2_mask": mask_iou(prompt.numpy(), masks[k].numpy()),
        "init_keyframe_translation_error_m": float(translation_errors(shape_pose[None], base_poses[idx[k]][None])[0]),
        "solver_iterations": sol.iterations, "solver_warmup": bool(w.solver_warmup),
        "tracks_visible_fraction": float(obs.track_vis.float().mean()),
        "anchored_queries": int(anchor_valid.sum()), "pnp_frames_accepted": int(pnp_accepted.sum()),
        "mean_axis_tilt_error_rad": float(axis_tilt_errors(ours_full, base_poses[:n_full], principal_axis(verts)).mean()),
        "translation_error_m_at_solved_frames": translation_errors(sol.poses, base_poses[idx]).tolist(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "headtohead_report.json").write_text(json.dumps(report, indent=2))
    (out_dir / "headtohead_details.json").write_text(json.dumps(details, indent=2))
    _write_notes(notes_path, report, details)
    logger.info("head-to-head report: %s", json.dumps(report))
    return report


def _write_notes(path: Path, report: dict[str, float], details: dict[str, Any]) -> None:
    """Writes (or replaces) the "Whisking head-to-head" section of the notes file at `path`."""
    reused = ", ".join(f"{k} {v:.2f} s" for k, v in details["reused_adapter_caches"].items()) or "none"
    stage_rows = "\n".join(f"| {name} | {sec:.2f} |" for name, sec in details["stage_seconds_this_run"].items())
    warmup = ("after a one-off synthetic warm-up solve (listed below, not counted)" if details["solver_warmup"]
              else "with no warm-up (the process's first solve, CUDA solver start-up included)")
    section = (
        f"{_NOTES_HEADING} ({datetime.now(UTC).date().isoformat()})\n\n"
        "`run/pipeline/whisking_headtohead.py`: our real front end (SAM 2.1 masks, CoTracker3 tracks, "
        "HaWoR hands, MoGe-2 through the layout's K with a HaWoR-anchored metric scale) and the M1 solver at "
        f"10 fps ({details['n_frames_solved']} frames, keyframe {details['keyframe_frame_idx']}), Lie-interpolated "
        "to 138 frames, against the baseline's `layout_camera_frame_optimized.json`.\n\n"
        f"- Initialization: {details['init_keyframe_pose']} with its scale {details['init_scale']:.4f}, not the "
        f"baseline's optimized keyframe pose or its optimized mesh scale {details['baseline_mesh_scale']:.4f}; "
        f"the solver ends at scale {details['solved_scale']:.4f}. Keyframe init vs. baseline keyframe: "
        f"{details['init_keyframe_translation_error_m']:.4f} m.\n"
        "- Yardstick caveat: both IoUs are measured against the SAM 2 masks that our solver's silhouette term "
        "fits, so `reprojection_iou_ours` > `reprojection_iou_baseline` is expected by construction and is not "
        "evidence of better tracking. The reused masks were prompted from the baseline's keyframe silhouette "
        f"(our SAM 3D prompt vs. that keyframe mask: IoU {details['keyframe_prompt_iou_vs_sam2_mask']:.3f}).\n"
        f"- `solver_seconds` is query anchoring + PnP init + gating + solve + interpolation, {warmup}.\n\n"
        "| metric | value |\n|---|---|\n"
        + "\n".join(f"| `{key}` | {report[key]:.4f} |" for key in REPORT_KEYS)
        + f"\n\nAdapter caches reused (the seconds measured when they were built are included in `frontend_seconds`): "
        f"{reused}. Rotation difference excluding spin about the whisk's long axis (axis tilt): "
        f"{details['mean_axis_tilt_error_rad']:.4f} rad.\n\n| stage (this run) | seconds |\n|---|---|\n{stage_rows}\n"
    )
    text = path.read_text() if path.is_file() else ""
    start = text.find(_NOTES_HEADING)
    if start >= 0:
        end = text.find("\n## ", start + len(_NOTES_HEADING))
        text = text[:start] + section + (text[end + 1:] if end >= 0 else "")
    else:
        text = text.rstrip("\n") + ("\n\n" if text else "") + section
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    apply_cpu_budget()
    apply_gpu_budget()
    run_headtohead(cfg, Path(hydra.utils.get_original_cwd()))


if __name__ == "__main__":
    main()
