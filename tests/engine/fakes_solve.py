"""Synthetic front-end artifacts and solver fakes for the solve-stage tests (no model, no GPU).

`write_synthetic_solve_inputs` writes a `SyntheticClip`'s observations as the engine stage artifacts the solve stage
reads; `synthetic_front_end_stage_fns` wraps the same writers as fake stage workers for a manifest run. The shape stage
is the real template worker on the clip's metric template (`scale_gt * mesh_vertices`), so the clip's `poses_gt` are
the template's raw-frame poses at metric scale 1.0.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from fastdo.engine.orchestrator import StageFn
from fastdo.engine.worker_support import save_npz
from fastdo.engine.workers_hands import write_hands_artifact
from fastdo.engine.workers_shape import make_template_shape_worker
from fastdo.engine.workers_solve import RegistrationConfig, SolveStageConfig
from fastdo.geometry.intrinsics import write_intrinsics
from fastdo.qc import QcFeatures
from fastdo.solver.api import MeshSamples, SolverConfig, TrajectoryObservations, TrajectorySolution
from fastdo.solver.synthetic import SyntheticClip, SyntheticConfig
from tests.engine.fakes import make_hawor_npz, random_frames, write_decode

SMALL_CLIP = SyntheticConfig(n_frames=6, image_hw=(120, 160), focal=150.0, shape="box", n_surface=512, n_tracks=24,
                             depth_noise=0.001)
SMALL_SOLVE = SolveStageConfig(device="cpu", seed=0, registration=RegistrationConfig(n_obs=128, n_mesh=256, n_iters=10),
                               n_anchor_surface=4096, solver={"iters_fine": 3})
Writer = Callable[[Path, SyntheticClip, int], None]


def _rows(clip: SyntheticClip, stride: int) -> np.ndarray:
    return stride * np.arange(int(clip.masks.shape[0]), dtype=np.int64)


def write_template_obj(path: Path, clip: SyntheticClip) -> Path:
    """The clip's metric template `scale_gt * mesh_vertices` as OBJ text at full float precision."""
    vertices = (clip.scale_gt * clip.mesh_vertices.double()).tolist()
    lines = [f"v {x!r} {y!r} {z!r}" for x, y, z in vertices]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in clip.mesh_faces.tolist()]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def write_decode_artifact(artifact_dir: Path, clip: SyntheticClip, stride: int = 3) -> None:
    rows = _rows(clip, stride)
    h, w = clip.image_hw
    write_decode(artifact_dir, random_frames(len(rows), h, w), frame_idx=rows.tolist(), fps=10.0, stride=stride)


def write_mask_artifact(artifact_dir: Path, clip: SyntheticClip, stride: int = 3) -> None:
    save_npz(artifact_dir / "mask.npz", masks=clip.masks.numpy().astype(bool), frame_idx=_rows(clip, stride),
             keyframe_index=np.array(0, dtype=np.int64))


def write_geometry_artifact(artifact_dir: Path, clip: SyntheticClip, stride: int = 3) -> None:
    t = int(clip.masks.shape[0])
    save_npz(artifact_dir / "geometry.npz", pointmaps=clip.pointmaps.numpy().astype(np.float32),
             frame_idx=_rows(clip, stride), scale=np.array(1.0, dtype=np.float32),
             frame_scale=np.ones(t, dtype=np.float32))


def write_tracks_artifact(artifact_dir: Path, clip: SyntheticClip, stride: int = 3) -> None:
    tracks = np.nan_to_num(clip.tracks.numpy().astype(np.float32), nan=0.0)
    save_npz(artifact_dir / "tracks.npz", tracks=tracks, vis=clip.track_vis.numpy().astype(bool),
             query_uv=tracks[0].copy(), frame_idx=_rows(clip, stride), keyframe_index=np.array(0, dtype=np.int64))


def write_intrinsics_artifact(artifact_dir: Path, clip: SyntheticClip, stride: int = 3) -> None:
    write_intrinsics(artifact_dir, clip.intrinsics.numpy(), "given")


def write_hands_stage_artifact(artifact_dir: Path, clip: SyntheticClip, stride: int = 3) -> None:
    hawor = make_hawor_npz(artifact_dir / "all_hand_meshes.npz", stride * int(clip.masks.shape[0]))
    write_hands_artifact(hawor, artifact_dir)


FRONT_END: dict[str, Writer] = {
    "decode": write_decode_artifact, "mask": write_mask_artifact, "intrinsics": write_intrinsics_artifact,
    "hands": write_hands_stage_artifact, "geometry": write_geometry_artifact, "tracks": write_tracks_artifact,
}


def write_synthetic_solve_inputs(artifact_dir: Path, clip: SyntheticClip, stride: int = 3, sdf_resolution: int = 16,
                                 n_samples: int = 512) -> None:
    """Every artifact the solve stage reads (and the qc stage's hands), for one clip named `artifact_dir.name`."""
    artifact_dir = Path(artifact_dir)
    for write in FRONT_END.values():
        write(artifact_dir, clip, stride)
    template = write_template_obj(artifact_dir / "template.obj", clip)
    make_template_shape_worker({artifact_dir.name: template}, sdf_resolution=sdf_resolution,
                               n_samples=n_samples)(artifact_dir.name, artifact_dir)


def synthetic_front_end_stage_fns(clips: Mapping[str, SyntheticClip], template_root: Path,
                                  stride: int = 3) -> dict[str, StageFn]:
    """Fake decode, mask, intrinsics, hands, geometry and tracks workers writing `clips[clip_id]`'s artifacts, and the
    real template shape worker (`sdf_resolution` 16, `n_samples` 512) on templates written under `template_root`."""
    templates = {clip_id: write_template_obj(Path(template_root) / f"{clip_id}.obj", clip)
                 for clip_id, clip in clips.items()}

    def stage(write: Writer) -> StageFn:
        def worker(clip_id: str, artifact_dir: Path) -> None:
            write(Path(artifact_dir), clips[clip_id], stride)

        return worker

    fns = {name: stage(write) for name, write in FRONT_END.items()}
    fns["shape"] = make_template_shape_worker(templates, sdf_resolution=16, n_samples=512)
    return fns


def fake_solution(init_poses: Tensor, init_scale: float) -> TrajectorySolution:
    """A finite `TrajectorySolution` at the initialization (covariance `1e-6 * eye(6)` per frame, as in `fakes_qc`)."""
    t = int(init_poses.shape[0])
    return TrajectorySolution(poses=init_poses.clone(), scale=float(init_scale),
                              covariance=1e-6 * torch.eye(6).expand(t, 6, 6).clone(), per_frame_cost=torch.zeros(t),
                              confidence=torch.ones(t), iterations=0, seconds=0.0)


class RecordingSolve:
    """An injected `solve_fn`: records each call's arguments (and `"solve_fn"` in `order`), runs `on_call`, and returns
    `fake_solution`, with the field `nan_field` set to NaN when given."""

    def __init__(self, order: list[str] | None = None, on_call: Callable[[], None] | None = None,
                 nan_field: str | None = None) -> None:
        self.calls: list[tuple[MeshSamples, TrajectoryObservations, Tensor, float, SolverConfig]] = []
        self.order, self.on_call, self.nan_field = order, on_call, nan_field

    def __call__(self, mesh: MeshSamples, obs: TrajectoryObservations, init_poses: Tensor, init_scale: float,
                 cfg: SolverConfig) -> TrajectorySolution:
        self.calls.append((mesh, obs, init_poses, init_scale, cfg))
        if self.order is not None:
            self.order.append("solve_fn")
        if self.on_call is not None:
            self.on_call()
        solution = fake_solution(init_poses, init_scale)
        if self.nan_field is None:
            return solution
        value = getattr(solution, self.nan_field)
        nan = torch.full_like(value, float("nan")) if isinstance(value, Tensor) else float("nan")
        return replace(solution, **{self.nan_field: nan})


def write_template_meta(clip: Path, scale: float = 1.0, center: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> None:
    """A `shape.json` of a template shape with the given normalization (for artifacts written by `fakes_qc`)."""
    (clip / "shape.json").write_text(json.dumps({"backend": "template", "normalization_scale": scale,
                                                 "normalization_center": list(center)}))


def shifted_gt(solution: TrajectorySolution, shift: float, stride: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """GT poses over `stride * T` source rows, all valid: the solved poses shifted by `shift` m in x at the solve rows."""
    t = int(solution.poses.shape[0])
    gt = np.tile(np.eye(4), (stride * t, 1, 1))
    rows = stride * np.arange(t)
    gt[rows] = solution.poses.double().numpy()
    gt[rows, 0, 3] += shift
    return gt, np.ones(stride * t, dtype=bool)


def monotone_features(errors: np.ndarray, seed: int = 0) -> list[QcFeatures]:
    """QC features as deterministic monotone functions of each pose error `e`, plus N(0, 0.02) IoU noise (rng `seed`):
    `iou = 1 - 10e + noise`, `std_ok = 1 - 10e`, `ot = 0.01 + e`, `pen = 0.1e`, `inlier = 1 - 10e`."""
    noise = np.random.default_rng(seed).normal(0.0, 0.02, size=len(errors))
    return [QcFeatures(reprojection_iou=float(1.0 - 10.0 * e + n), laplace_std_ok_frac=float(1.0 - 10.0 * e),
                       ot_residual_rms_frac=float(0.01 + e), penetration_depth_m=float(0.1 * e),
                       track_inlier_ratio=float(1.0 - 10.0 * e)) for e, n in zip(errors, noise, strict=True)]
