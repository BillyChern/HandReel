"""24-row synthetic clip writer for the driftqc stage tests (no model, no GPU, no data file).

`write_driftqc_inputs` writes decode, mask, geometry (with `frame_scale`) and tracks for a 24-frame `SMALL_CLIP`. With
`contraction_from = t`, rows >= t of `tracks` are moved 50 % toward their row centroid: a fixture input that makes the
frozen SG statistic flag those rows, not a constant. E4/E5 extend the same fixture (N = 24 is owned here).
"""

from __future__ import annotations

from dataclasses import replace
from functools import cache
from pathlib import Path

import numpy as np

from fastdo.engine.worker_support import save_npz
from fastdo.engine.workers_shape import make_template_shape_worker
from fastdo.solver.synthetic import SyntheticClip, make_synthetic_clip
from tests.engine.fakes_solve import (
    SMALL_CLIP,
    write_decode_artifact,
    write_geometry_artifact,
    write_hands_stage_artifact,
    write_intrinsics_artifact,
    write_mask_artifact,
    write_template_obj,
)

__all__ = ["DRIFTQC_CLIP", "N_ROWS", "contracted_tracks", "driftqc_clip", "write_driftqc_inputs",
           "write_driftqc_solve_inputs"]

N_ROWS = 24
DRIFTQC_CLIP = replace(SMALL_CLIP, n_frames=N_ROWS)
CONTRACTION = 0.5  # fraction of the way each contracted track moves toward its row centroid (fixture input)


@cache
def driftqc_clip(seed: int = 0) -> SyntheticClip:
    """`make_synthetic_clip(DRIFTQC_CLIP, seed)`, built once per seed (treat as read-only)."""
    return make_synthetic_clip(DRIFTQC_CLIP, seed=seed)


def contracted_tracks(clip: SyntheticClip, contraction_from: int | None) -> np.ndarray:
    """(N,J,2) float32 tracks as the tracks fake writes them, rows >= contraction_from moved toward their centroid."""
    tracks = np.nan_to_num(clip.tracks.numpy().astype(np.float32), nan=0.0)
    if contraction_from is not None:
        rows = tracks[contraction_from:]
        centroid = rows.mean(axis=1, keepdims=True)
        tracks[contraction_from:] = centroid + (1.0 - CONTRACTION) * (rows - centroid)
    return tracks


def write_driftqc_inputs(artifact_dir: Path, clip: SyntheticClip, contraction_from: int | None, *, stride: int = 3,
                         keyframe_index: int = 0, frame_scale: np.ndarray | None = None,
                         write_frame_scale: bool = True) -> None:
    """decode.npz, mask.npz, geometry.npz and tracks.npz for `clip` (the driftqc stage's inputs).

    Args:
        artifact_dir: The clip's artifact directory (created).
        clip: A synthetic clip, normally `driftqc_clip(0)` (24 rows).
        contraction_from: First contracted track row, or None for a healthy clip.
        stride: Decode stride (frame_idx = stride * row).
        keyframe_index: The keyframe index written to mask.npz and tracks.npz (U13 uses 5).
        frame_scale: Per-row frame_scale to write instead of ones (U14 perturbs it).
        write_frame_scale: False writes geometry.npz without a frame_scale key.
    """
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    write_decode_artifact(artifact_dir, clip, stride)
    write_mask_artifact(artifact_dir, clip, stride)
    write_geometry_artifact(artifact_dir, clip, stride)
    idx = stride * np.arange(int(clip.masks.shape[0]), dtype=np.int64)
    kf = np.array(keyframe_index, dtype=np.int64)
    if keyframe_index != 0:
        mask = dict(np.load(artifact_dir / "mask.npz", allow_pickle=False))
        save_npz(artifact_dir / "mask.npz", **{**mask, "keyframe_index": kf})
    if frame_scale is not None or not write_frame_scale:
        geometry = dict(np.load(artifact_dir / "geometry.npz", allow_pickle=False))
        geometry.pop("frame_scale")
        if write_frame_scale:
            geometry["frame_scale"] = np.asarray(frame_scale, dtype=np.float32)
        save_npz(artifact_dir / "geometry.npz", **geometry)
    tracks = contracted_tracks(clip, contraction_from)
    save_npz(artifact_dir / "tracks.npz", tracks=tracks, vis=clip.track_vis.numpy().astype(bool),
             query_uv=tracks[0].copy(), frame_idx=idx, keyframe_index=kf)


def write_driftqc_solve_inputs(artifact_dir: Path, clip: SyntheticClip, contraction_from: int | None,
                               stride: int = 3) -> Path:
    """`write_driftqc_inputs` plus intrinsics, hands and the template shape stage (sdf 16, 512 samples): every
    artifact the solve and qc stages read, on the 24-row clip (E4/E5; `write_synthetic_solve_inputs` adapted)."""
    artifact_dir = Path(artifact_dir)
    write_driftqc_inputs(artifact_dir, clip, contraction_from, stride=stride)
    write_intrinsics_artifact(artifact_dir, clip, stride)
    write_hands_stage_artifact(artifact_dir, clip, stride)
    template = write_template_obj(artifact_dir / "template.obj", clip)
    make_template_shape_worker({artifact_dir.name: template}, sdf_resolution=16,
                               n_samples=512)(artifact_dir.name, artifact_dir)
    return artifact_dir
