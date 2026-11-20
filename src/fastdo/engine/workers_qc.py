"""The `qc` stage worker: `qc.json` and, for an accepted clip, its `ClipRecord`.

`qc` runs after `solve` (`fastdo.engine.manifest.STAGE_NAMES`). Every per-row input has exactly the rows of
`decode.npz`; a mismatch raises ValueError naming the artifact. Inputs:

- `solve.npz` (`write_solve_artifact`, `read_solve_artifact`; the solve stage writes it): the `TrajectorySolution`
  fields `poses` (T,4,4), `scale`, `covariance` (T,d,d), `per_frame_cost` (T,), `confidence` (T,), `iterations`,
  `seconds`, plus `frame_idx` (T,) int64 and the tracks' final IRLS weights `track_weight` (T,Q); with the solver's
  `anisotropic_scale` on (and only then, so a flag-off artifact is unchanged byte for byte) also
  `shape_log_stretch` (3,), `shape_axes` (3,3) and `shape_centroid` (3,) float64 (`fastdo.solver.aniso`);
- `mask.npz` (`masks`), `intrinsics.npz` (`K`), the shape stage's `mesh.obj`, `samples.npz` and `sdf.npz`, and
  `hands.npz` (the `HandTrack` fields and `{side}_vertices` of both sides);
- optional `geometry.npz` keys `cam_to_world` (T,4,4) and `gravity_cam` (3,), a camera estimate of the geometry stage
  (it writes neither today); without them `fastdo.qc.export.DEFAULT_GRAVITY_CAM` and the static
  `gravity_aligned_cam_to_world`, recorded as `gravity_source` "default" / `camera_source` "gravity_aligned_static";
- optional `gate.json` `{"score": float}`, the gate stage's score (NaN without it).

`features_fn` (`fastdo.qc.qc_features`) computes the five `QcFeatures`. Its hand vertices are both sides'
`{side}_vertices` concatenated, with rows whose `valid` is False set to NaN (the penetration feature ignores non-finite
vertices; a side with no valid row is left out). The object diameter is `|scale|` times the bounding-box diagonal of the
canonical samples. `accept(rule, features)` decides. `qc.json` records `accepted`, `confidence` (`1 - score`, in
[0, 1]), the score and threshold, the features, the diameter and the camera sources. An accepted clip's
`clip_record.npz` is written before `qc.json`, so an accepted `qc.json` always has its record; a rejected clip's stale
record is removed. A hand side with no valid row is exported as None. Timestamps are `frame_idx / source_fps`, with
`source_fps` = decode `fps` times the decode `stride` (`decode_timestamps`).

`read_qc_inputs` + `compute_qc_features` are the worker's reads and feature call, so QC calibration computes exactly
these features.

driftqc: with `driftqc_enabled` True, `read_qc_inputs` resolves the clip's `valid_range.json` prefix once and applies it
to the decode rows and timestamps, and through the single slicing helper `fastdo.engine.valid_range.slice_artifact` to
`mask.npz`, `hands.npz` and the geometry camera keys, before the decode-row checks (`solve.npz` already has the prefix
rows); ACCEPT (range [0, N-1]) takes the unmodified path. With it False (the default) a leftover `driftqc.json` or
`valid_range.json` is a ValueError and nothing is ever sliced. `make_qc_worker_from_config` loads a saved rule
(`fastdo.qc.rule_io`) at construction, with no accept-all default; `qc.json["rule"]` records its path, sha256 and
version (null for `make_qc_worker` called directly).
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from fastdo.engine.orchestrator import StageFn
from fastdo.engine.valid_range import RowRange, resolve_valid_range, slice_artifact
from fastdo.engine.worker_support import (
    DECODE_FILE,
    SOLVE_FILE,
    load_npz,
    require_artifact,
    save_npz,
)
from fastdo.engine.workers_contact import CONTACT_FILE, read_contact_artifact
from fastdo.engine.workers_geometry import GEOMETRY_FILE
from fastdo.engine.workers_hands import HANDS_FILE, SIDES, decode_frame_size
from fastdo.engine.workers_shape import MESH_FILE, read_mesh_samples, read_sdf_grid
from fastdo.engine.workers_video import MASK_FILE
from fastdo.geometry.intrinsics import read_intrinsics
from fastdo.hands.hand_camera import require_hands_camera
from fastdo.io.clip_record import HandTrack
from fastdo.qc.acceptance import AcceptanceRule, accept
from fastdo.qc.export import DEFAULT_GRAVITY_CAM, export_clip_record, gravity_aligned_cam_to_world
from fastdo.qc.features import QcFeatures, qc_features
from fastdo.qc.interaction import InteractionScreenConfig, interaction_features, screen_passes
from fastdo.qc.world_motion import WorldMotionScreenConfig, world_motion_features, world_screen_decision
from fastdo.qc.rule_io import ACCEPTANCE_RULE_VERSION, load_acceptance_rule_and_sha256
from fastdo.solver.api import MeshSamples, TrajectorySolution
from fastdo.solver.sdf import SdfGrid

__all__ = [
    "CLIP_RECORD_FILE",
    "GATE_FILE",
    "QC_FILE",
    "SOLVE_FILE",
    "CameraFrame",
    "FeaturesFn",
    "HandsArtifact",
    "QcInputs",
    "QcStageConfig",
    "SolveArtifact",
    "check_decode_rows",
    "compute_qc_features",
    "decode_timestamps",
    "make_qc_worker",
    "make_qc_worker_from_config",
    "read_camera_frame",
    "read_hands_artifact",
    "read_qc_inputs",
    "read_solve_artifact",
    "write_json",
    "write_solve_artifact",
]

logger = logging.getLogger(__name__)

QC_FILE = "qc.json"
CLIP_RECORD_FILE = "clip_record.npz"
GATE_FILE = "gate.json"
FeaturesFn = Callable[..., QcFeatures]


@dataclass(frozen=True)
class SolveArtifact:
    solution: TrajectorySolution
    frame_idx: np.ndarray     # (T,) int64
    track_weight: Tensor      # (T,Q) final track IRLS weights


@dataclass(frozen=True)
class HandsArtifact:
    tracks: dict[str, HandTrack]       # per side, the hands.npz rows as a HandTrack
    vertices: dict[str, np.ndarray]    # per side, (T,778,3) float32 camera-frame MANO vertices


@dataclass(frozen=True)
class CameraFrame:
    cam_to_world: np.ndarray    # (T,4,4) float32
    gravity_cam: np.ndarray     # (3,) float32
    camera_source: str          # "geometry" | "gravity_aligned_static"
    gravity_source: str         # "geometry" | "default"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Writes `payload` as indented JSON via a tmp sibling and `os.replace`."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


def check_decode_rows(frame_idx: np.ndarray, decode_idx: np.ndarray, artifact: str) -> None:
    """Raises ValueError unless an artifact's `frame_idx` equals the decode rows'."""
    idx = np.asarray(frame_idx, dtype=np.int64).reshape(-1)
    if not np.array_equal(idx, np.asarray(decode_idx, dtype=np.int64).reshape(-1)):
        raise ValueError(f"{artifact} rows are not the decode rows: frame_idx {idx[:6].tolist()} "
                         f"({idx.size} rows) vs decode {np.asarray(decode_idx)[:6].tolist()} ({np.size(decode_idx)} rows)")


_SHAPE_FIELDS = ("shape_log_stretch", "shape_axes", "shape_centroid")   # anisotropic solves only


def write_solve_artifact(artifact_dir: Path, solution: TrajectorySolution, frame_idx: np.ndarray,
                         track_weight: Tensor) -> None:
    """Writes `<artifact_dir>/solve.npz` (module docstring); arrays keep their dtypes, scalars are 0-d."""
    t = int(solution.poses.shape[0])
    idx = np.asarray(frame_idx, dtype=np.int64).reshape(-1)
    weight = track_weight.detach().cpu()
    if idx.shape[0] != t or weight.shape[0] != t:
        raise ValueError(f"solve.npz needs {t} rows everywhere: frame_idx {idx.shape[0]}, track_weight {weight.shape[0]}")
    # the shape fields exist only on an anisotropic solve; a flag-off artifact gets exactly the historical keys.
    shape = {name: getattr(solution, name).detach().cpu().double().numpy() for name in _SHAPE_FIELDS
             if getattr(solution, name) is not None}
    save_npz(Path(artifact_dir) / SOLVE_FILE, poses=solution.poses.detach().cpu().numpy(),
             scale=np.array(solution.scale, dtype=np.float64), covariance=solution.covariance.detach().cpu().numpy(),
             per_frame_cost=solution.per_frame_cost.detach().cpu().numpy(),
             confidence=solution.confidence.detach().cpu().numpy(), iterations=np.array(solution.iterations, np.int64),
             seconds=np.array(solution.seconds, dtype=np.float64), frame_idx=idx, track_weight=weight.numpy(), **shape)


def read_solve_artifact(artifact_dir: Path) -> SolveArtifact:
    """`<artifact_dir>/solve.npz` as a `SolveArtifact`."""
    data = load_npz(require_artifact(Path(artifact_dir) / SOLVE_FILE, "solve"))
    shape = {name: torch.from_numpy(data[name]) for name in _SHAPE_FIELDS if name in data}
    solution = TrajectorySolution(poses=torch.from_numpy(data["poses"]), scale=float(data["scale"]),
                                  covariance=torch.from_numpy(data["covariance"]),
                                  per_frame_cost=torch.from_numpy(data["per_frame_cost"]),
                                  confidence=torch.from_numpy(data["confidence"]), iterations=int(data["iterations"]),
                                  seconds=float(data["seconds"]), shape_log_stretch=shape.get("shape_log_stretch"),
                                  shape_axes=shape.get("shape_axes"), shape_centroid=shape.get("shape_centroid"))
    return SolveArtifact(solution, data["frame_idx"].astype(np.int64), torch.from_numpy(data["track_weight"]))


def read_hands_artifact(artifact_dir: Path, decode_idx: np.ndarray, valid_range: RowRange | None = None,
                        check_camera: bool = False) -> HandsArtifact:
    """Both sides of `<artifact_dir>/hands.npz`, whose rows must be the decode rows (`decode_idx`, already the prefix
    rows when `valid_range` is given: the rows are then sliced through `slice_artifact` first). With `check_camera`, the
    hand outputs must also be in `intrinsics.npz`'s camera (`fastdo.hands.hand_camera.require_hands_camera`)."""
    data = load_npz(require_artifact(Path(artifact_dir) / HANDS_FILE, "hands"))
    if valid_range is not None:
        data = slice_artifact(data, HANDS_FILE, valid_range)
    check_decode_rows(data["frame_idx"], decode_idx, HANDS_FILE)
    if check_camera:
        width, height = decode_frame_size(artifact_dir)
        require_hands_camera(data, read_intrinsics(artifact_dir)[0], width, height, HANDS_FILE)
    tracks = {side: HandTrack(params=data[f"{side}_params"], betas=data[f"{side}_betas"], trans=data[f"{side}_trans"],
                              joints=data[f"{side}_joints"], valid=data[f"{side}_valid"].astype(bool))
              for side in SIDES}
    return HandsArtifact(tracks=tracks, vertices={side: data[f"{side}_vertices"] for side in SIDES})


def decode_timestamps(artifact_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """`(frame_idx (N,) int64, timestamps (N,) float64 seconds)` of the decode rows.

    Normally `frame_idx / (fps * stride)`. `stride` is `decode.npz`'s own (the decode stage writes it); an archive
    without one takes its rows' uniform step.

    **The timebase rule's exception.** "If any row has |pts - n / fps_v| > 1 / fps_v, the window is flagged
    `timebase_mismatch`. **Its labels are then computed from the pts.** It stays in every analysis." So a flagged
    window returns its RECORDED pts here, not the arithmetic -- the flag exists precisely because the arithmetic
    disagrees with the video, and continuing to label from the arithmetic would use the quantity the flag
    disputes. The derived value stays the default for unflagged windows because it is exactly uniform where pts
    carry container jitter; the rule chooses derived normally and measured on disagreement, and that is implemented
    literally rather than improved.

    An archive with no `timebase_mismatch` member is an unflagged window (the check ran before the flag existed,
    or the whole-file decode path wrote it): derived, as before. The even-spacing validation runs either way --
    unevenly spaced rows are a stride defect, not a timebase one, and must stay loud.

    Raises:
        ValueError: If the rows are not evenly spaced by the stride (the source rate is then ambiguous), a one-row
            archive has no `stride` (its source rate cannot be recovered), the rate is not positive, or a flagged
            window's `pts` do not match its rows.
    """
    path = require_artifact(Path(artifact_dir) / DECODE_FILE, "decode")
    with np.load(path, allow_pickle=False) as data:
        frame_idx, fps = np.asarray(data["frame_idx"], dtype=np.int64), float(data["fps"])
        stored = int(data["stride"]) if "stride" in data.files else None
        flagged = bool(int(data["timebase_mismatch"])) if "timebase_mismatch" in data.files else False
        pts = np.asarray(data["pts"], dtype=np.float64) if "pts" in data.files else None
    steps = np.unique(np.diff(frame_idx))
    if steps.size > 1 or (stored is not None and steps.size == 1 and int(steps[0]) != stored):
        raise ValueError(f"{path}: decode rows are not evenly spaced by the stride (steps {steps[:4].tolist()}, stride "
                         f"{stored}); source rate unknown")
    if stored is None and steps.size == 0:
        raise ValueError(f"{path}: a one-row decode without `stride` has no recoverable source rate")
    source_fps = fps * (stored if stored is not None else int(steps[0]))
    if not (math.isfinite(source_fps) and source_fps > 0.0):
        raise ValueError(f"{path}: source frame rate must be positive, got {source_fps}")
    if flagged:
        # the timebase rule: "Its labels are then computed from the pts."
        if pts is None or pts.shape != frame_idx.shape:
            raise ValueError(f"{path}: window is flagged `timebase_mismatch` but its `pts` are "
                             f"{'absent' if pts is None else f'{pts.shape}, not {frame_idx.shape}'}; "
                             f"a flagged window is labelled from its pts, so there is nothing to label it with")
        logger.info("%s: timebase_mismatch -- labelling %d rows from the recorded pts",
                    Path(artifact_dir).name, frame_idx.size)
        return frame_idx, pts
    return frame_idx, frame_idx / source_fps


def read_camera_frame(artifact_dir: Path, decode_idx: np.ndarray, valid_range: RowRange | None = None) -> CameraFrame:
    """The clip's camera-to-world transforms and up vector (module docstring for the sources and fallbacks). With
    `valid_range`, `decode_idx` is the prefix rows and the geometry keys are sliced through `slice_artifact` first."""
    path = Path(artifact_dir) / GEOMETRY_FILE
    found: dict[str, np.ndarray] = {}
    if path.is_file():
        with np.load(path, allow_pickle=False) as data:
            found = {key: np.asarray(data[key]) for key in ("cam_to_world", "gravity_cam", "frame_idx") if key in data.files}
    if valid_range is not None:
        found = slice_artifact(found, GEOMETRY_FILE, valid_range)
    if "gravity_cam" in found:
        gravity, gravity_source = found["gravity_cam"].astype(np.float32).reshape(3), "geometry"
    else:
        gravity, gravity_source = DEFAULT_GRAVITY_CAM.copy(), "default"
    if "cam_to_world" in found:
        check_decode_rows(found.get("frame_idx", np.empty(0, np.int64)), decode_idx, GEOMETRY_FILE)
        return CameraFrame(found["cam_to_world"].astype(np.float32), gravity, "geometry", gravity_source)
    static = gravity_aligned_cam_to_world(gravity, int(np.size(decode_idx))).numpy().astype(np.float32)
    return CameraFrame(static, gravity, "gravity_aligned_static", gravity_source)


def _penetration_vertices(hands: HandsArtifact) -> Tensor | None:
    blocks = []
    for side in SIDES:
        valid = hands.tracks[side].valid
        if not valid.any():
            continue
        verts = torch.from_numpy(hands.vertices[side].astype(np.float64))
        verts[torch.from_numpy(~valid)] = float("nan")
        blocks.append(verts)
    return torch.cat(blocks, dim=1) if blocks else None


def _gate_score(artifact_dir: Path) -> float:
    path = artifact_dir / GATE_FILE
    if not path.is_file():
        logger.info("%s has no %s; exporting gate_score NaN", artifact_dir, GATE_FILE)
        return math.nan
    return float(json.loads(path.read_text())["score"])


@dataclass(frozen=True)
class QcInputs:
    """Everything the qc stage reads for one clip (`read_qc_inputs`); every per-row input has the decode rows."""

    frame_idx: np.ndarray       # (T,) int64 decode rows
    timestamps: np.ndarray      # (T,) float64 seconds (`decode_timestamps`)
    solve: SolveArtifact
    hands: HandsArtifact
    masks: Tensor               # (T,H,W) bool
    mesh: MeshSamples
    sdf: SdfGrid
    mesh_path: Path
    K: Tensor                   # (3,3) float32
    diameter: float             # |scale| times the canonical samples' bounding-box diagonal, metres
    valid_range: RowRange | None = None   # the driftqc prefix applied, or None (identity path)
    poses_source: str = "solve"           # "contact" when the contact stage's corrected poses were used
    contact_shift_mm: np.ndarray | None = None   # (T,) the applied along-ray shifts, or None


def read_qc_inputs(artifact_dir: Path, driftqc_enabled: bool = False) -> QcInputs:
    """The qc stage's reads and row checks (module docstring). `resolve_valid_range(artifact_dir, decode rows,
    driftqc_enabled)` runs once, right after the decode rows are read: None leaves the body unchanged; a `RowRange`
    slices the decode rows, timestamps, `mask.npz` and `hands.npz` (`slice_artifact`) before the R7 checks. The sliced
    arrays are views and are never written to.

    Raises:
        FileNotFoundError: If an artifact is missing.
        ValueError: If `solve.npz`, `hands.npz` or `mask.npz` does not have the decode rows; if the
            `hands.npz` outputs are not in `intrinsics.npz`'s camera; with
            `driftqc_enabled` False if driftqc artifacts are present, and with it True if `valid_range.json` is absent
            or does not match its `driftqc.json` and the decode rows (`resolve_valid_range`).
    """
    artifact_dir = Path(artifact_dir)
    frame_idx, timestamps = decode_timestamps(artifact_dir)
    rng = resolve_valid_range(artifact_dir, frame_idx, driftqc_enabled)
    if rng is not None:
        frame_idx, timestamps = frame_idx[rng.as_slice()], timestamps[rng.as_slice()]
    solve = read_solve_artifact(artifact_dir)
    check_decode_rows(solve.frame_idx, frame_idx, SOLVE_FILE)
    # The contact stage corrects along-ray placement AFTER the solve, so qc -- and the ClipRecord it exports --
    # must score the corrected trajectory. `read_contact_artifact` returns None unless contact.npz was computed from
    # THIS solve.npz, so a stale correction is never applied silently.
    poses_source, contact_shift = "solve", None
    contact = read_contact_artifact(artifact_dir)
    if contact is not None:
        if rng is not None:
            contact = slice_artifact(contact, CONTACT_FILE, rng)
        check_decode_rows(np.asarray(contact["frame_idx"], dtype=np.int64), frame_idx, CONTACT_FILE)
        solve = dataclasses.replace(solve, solution=dataclasses.replace(
            solve.solution, poses=torch.from_numpy(np.asarray(contact["poses"], dtype=np.float32))))
        poses_source, contact_shift = "contact", np.asarray(contact["shift_mm"])
    hands = read_hands_artifact(artifact_dir, frame_idx, rng, check_camera=True)
    mask_data = load_npz(require_artifact(artifact_dir / MASK_FILE, "mask"))
    if rng is not None:
        mask_data = slice_artifact(mask_data, MASK_FILE, rng)
    check_decode_rows(mask_data["frame_idx"], frame_idx, MASK_FILE)
    mesh, sdf = read_mesh_samples(artifact_dir), read_sdf_grid(artifact_dir)
    mesh_path = require_artifact(artifact_dir / MESH_FILE, "shape")
    K = torch.from_numpy(read_intrinsics(artifact_dir)[0])
    points = mesh.points.to(torch.float64)
    diameter = abs(float(solve.solution.scale)) * float((points.max(dim=0).values - points.min(dim=0).values).norm())
    return QcInputs(frame_idx=frame_idx, timestamps=timestamps, solve=solve, hands=hands,
                    masks=torch.from_numpy(mask_data["masks"].astype(bool)), mesh=mesh, sdf=sdf, mesh_path=mesh_path,
                    K=K, diameter=diameter, valid_range=rng, poses_source=poses_source,
                    contact_shift_mm=contact_shift)


def compute_qc_features(inputs: QcInputs, features_fn: FeaturesFn = qc_features,
                        laplace_std_threshold: float = 0.01) -> QcFeatures:
    """`features_fn` on a clip's qc inputs, with exactly the arguments the qc stage passes."""
    sol = inputs.solve.solution
    return features_fn(inputs.masks, sol.poses, sol.scale, inputs.mesh, sol.covariance, sol.per_frame_cost, inputs.K,
                       inputs.diameter, _penetration_vertices(inputs.hands), inputs.sdf, inputs.solve.track_weight,
                       laplace_std_threshold=laplace_std_threshold)


def min_camera_z(poses: np.ndarray, scale: float, mesh_points: np.ndarray) -> float:
    """Smallest camera-frame depth the solve puts any mesh point at, over all frames.

    The emitted camera-frame cloud is `p = scale * R @ y + t`, so this is `min_k min_i p_z`. A value at
    or below zero means the solve placed geometry BEHIND the camera -- physically impossible output that
    the track term's `clamp_min(1e-6)` then turns into residuals around 3.7e+06 px.

    Measured across the whole corpus -- 572 emitted solves, 13 run directories, 38 meshes:

    * **12 of 572 (2.1%)** put at least one point behind the camera, and the same 12 reach the clamp:
      nothing lands between 0 and 1e-6, so once a point crosses the plane it is already deep inside it.
    * The danger zone is **MILD collapse**, recovered scale 1e-4 to 1e-2: **10 of 10**, median min z
      **-0.0485 m**. Total collapse below 1e-20 is SAFE (0 of 46, median +0.6633 m) because the object
      shrinks in place and `p -> t`. So **min z is the discriminator and recovered scale is not** -- a
      scale gate would have to fire on a narrow interior band while passing values a thousand times
      smaller.
    * Healthy solves span **0.0831 m to 2.2774 m**, so a refusal at `min z <= 0` has an enormous margin:
      the nearest healthy solve sits **83,095x above the 1e-6 clamp**.

    Concentrated in MIXED arms (7% where tracks or init are fixed alone); the matched production arm
    `sam2/mogeclipgt/cotracker/real` is **0 of 26** -- which is 0 with a rule-of-three upper bound near
    11%, and unmeasured on internet video.
    """
    poses = np.asarray(poses, dtype=np.float64)
    points = np.asarray(mesh_points, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[-2:] != (4, 4):
        raise ValueError(f"poses must be (T, 4, 4), got {poses.shape}")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"mesh points must be (N, 3), got {points.shape}")
    if points.shape[0] == 0 or poses.shape[0] == 0:
        raise ValueError("min camera z needs at least one pose and one mesh point")
    # only the z row of each rotation is needed, so this never materialises the (T, N, 3) cloud
    z_rows = poses[:, 2, :3]                                  # (T, 3)
    z_of_points = float(scale) * (points @ z_rows.T) + poses[:, 2, 3]      # (N, T)
    return float(z_of_points.min())


def interaction_block(inputs: QcInputs, screen: InteractionScreenConfig) -> dict[str, Any]:
    """`qc.json["interaction"]` for an ENABLED `qc_stage.interaction_screen`: the window's
    `fastdo.qc.interaction.InteractionFeatures` on the trajectory qc scores, and whether it passes the screen."""
    sol = inputs.solve.solution
    tracks = inputs.hands.tracks
    feats, _ = interaction_features(sol.poses.detach().cpu().numpy(), float(sol.scale),
                                    inputs.mesh.points.detach().cpu().numpy(),
                                    {side: np.asarray(tracks[side].joints) for side in SIDES},
                                    {side: np.asarray(tracks[side].valid, dtype=bool) for side in SIDES},
                                    touch_gate_mm=screen.touch_gate_mm)
    return {"feature": screen.feature, "min_value": screen.min_value, "touch_gate_mm": screen.touch_gate_mm,
            "features": dataclasses.asdict(feats), "passed": screen_passes(feats, screen)}


def world_motion_block(artifact_dir: Path, inputs: QcInputs, screen: WorldMotionScreenConfig) -> dict[str, Any]:
    """`qc.json["world_motion"]` for an ENABLED `qc_stage.world_motion_screen`: the window's world-frame
    object motion (`fastdo.qc.world_motion`) from the solve and `ego_motion.npz`, and the screen's decision.

    Raises:
        FileNotFoundError: If `ego_motion.npz` is missing (enable `geometry_stage.ego_motion`).
        ValueError: If its rows are not the qc rows.
    """
    from fastdo.geometry.ego_motion import EGO_MOTION_FILE, load_ego_motion

    path = Path(artifact_dir) / EGO_MOTION_FILE
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found: qc_stage.world_motion_screen needs geometry_stage.ego_motion")
    ego = load_ego_motion(path)
    rows = np.asarray(ego["frame_idx"])
    c2w, valid = np.asarray(ego["cam_to_world"]), np.asarray(ego["valid"], dtype=bool)
    if inputs.valid_range is not None:           # driftqc's inclusive prefix [0, stop]
        end = inputs.valid_range.stop + 1
        rows, c2w, valid = rows[:end], c2w[:end], valid[:end]
    check_decode_rows(rows, inputs.frame_idx, EGO_MOTION_FILE)
    sol = inputs.solve.solution
    tracks = inputs.hands.tracks
    feats, _ = world_motion_features(sol.poses.detach().cpu().numpy(), float(sol.scale),
                                     inputs.mesh.points.detach().cpu().numpy(),
                                     sol.confidence.detach().cpu().numpy(), c2w, valid, bool(ego["summary"]["ok"]),
                                     {side: np.asarray(tracks[side].joints) for side in SIDES},
                                     {side: np.asarray(tracks[side].valid, dtype=bool) for side in SIDES},
                                     step_rows=screen.step_rows, move_gate_mm=screen.move_gate_mm)
    keep, reason = world_screen_decision(feats, screen)
    return {"feature": screen.feature, "min_value": screen.min_value, "features": dataclasses.asdict(feats),
            "passed": keep, "reason": reason, "ego_summary": ego["summary"]}


def make_qc_worker(rule: AcceptanceRule, features_fn: FeaturesFn = qc_features, laplace_std_threshold: float = 0.01,
                   rule_source: Mapping[str, Any] | None = None, *, driftqc_enabled: bool = False,
                   reject_behind_camera: bool = False,
                   interaction_screen: InteractionScreenConfig | None = None,
                   world_motion_screen: WorldMotionScreenConfig | None = None) -> StageFn:
    """The `qc` stage worker (module docstring).

    Args:
        rule: The fitted acceptance rule (`fastdo.qc.fit_acceptance_rule`).
        features_fn: `qc_features`-compatible feature function (tests inject fakes).
        laplace_std_threshold: Translation-std threshold (m) of the Laplace feature.
        rule_source: Where `rule` came from, recorded as `qc.json["rule"]` (null when None).
        driftqc_enabled: `driftqc_stage.enabled` (default False: never slices; leftover driftqc artifacts raise).
        reject_behind_camera: Reject a clip whose solve puts mesh points at or behind the camera plane
            (`min_camera_z`). Default False, so no run's acceptance changes: `min_camera_z` and
            `behind_camera` are RECORDED either way, which is what was missing. Turning it on rejected
            12 of 572 solves across the corpus and 0 of 26 in the matched production arm.
        interaction_screen: Opt-in (`qc_stage.interaction_screen`): reject a window whose hands never touch the
            solved object (`fastdo.qc.interaction`). None or disabled (the default) computes nothing and writes no
            `interaction` key, so `qc.json` is byte-identical to before the option. The rule
            `touch_frac >= 0.26` FAILED on Ego4D (driving keeps touch_frac ~1).
        world_motion_screen: Opt-in (`qc_stage.world_motion_screen`): reject a window whose solved object does
            not move relative to the static scene (`fastdo.qc.world_motion`; needs `ego_motion.npz` from
            `geometry_stage.ego_motion`). None or disabled (the default) reads nothing and writes no `world_motion`
            key: `qc.json` is byte-identical to before the option.
    """
    source = None if rule_source is None else dict(rule_source)
    screen = interaction_screen if interaction_screen is not None and interaction_screen.enabled else None
    world = world_motion_screen if world_motion_screen is not None and world_motion_screen.enabled else None

    def qc_worker(clip_id: str, artifact_dir: Path) -> None:
        artifact_dir = Path(artifact_dir)
        inputs = read_qc_inputs(artifact_dir, driftqc_enabled)
        features = compute_qc_features(inputs, features_fn, laplace_std_threshold)
        accepted, confidence = accept(rule, features)
        # 2.1% of emitted solves put mesh points BEHIND the camera, and nothing checked. The
        # value is recorded unconditionally so it reaches the report; rejecting on it is opt-in, because
        # QC acceptance feeds the yield number and changing it is a protocol change, not a fix.
        depth = min_camera_z(inputs.solve.solution.poses.detach().cpu().numpy(),
                             inputs.solve.solution.scale,
                             inputs.mesh.points.detach().cpu().numpy())
        behind_camera = depth <= 0.0
        if behind_camera and reject_behind_camera:
            accepted = False
        interaction = interaction_block(inputs, screen) if screen is not None else None
        if interaction is not None and not interaction["passed"]:
            accepted = False
        world_motion = world_motion_block(artifact_dir, inputs, world) if world is not None else None
        if world_motion is not None and not world_motion["passed"]:
            accepted = False
        camera = read_camera_frame(artifact_dir, inputs.frame_idx, inputs.valid_range)
        record_path = artifact_dir / CLIP_RECORD_FILE
        if accepted:
            tracks = {side: track if track.valid.any() else None for side, track in inputs.hands.tracks.items()}
            tmp = record_path.with_name(record_path.name + ".tmp")
            export_clip_record(clip_id, inputs.solve.solution, str(inputs.mesh_path), inputs.K,
                               torch.from_numpy(camera.cam_to_world), torch.from_numpy(camera.gravity_cam),
                               tracks["left"], tracks["right"], _gate_score(artifact_dir), inputs.frame_idx,
                               inputs.timestamps, path=tmp)
            os.replace(tmp, record_path)
        else:
            record_path.unlink(missing_ok=True)
        payload: dict[str, Any] = {
            "clip_id": clip_id, "accepted": bool(accepted), "confidence": float(confidence),
            "score": float(1.0 - confidence), "threshold": float(rule.calibration.threshold),
            "alpha": float(rule.calibration.alpha), "tau_m": float(rule.tau), "features": dataclasses.asdict(features),
            "diameter_m": inputs.diameter, "n_frames": int(inputs.frame_idx.size), "camera_source": camera.camera_source,
            "gravity_source": camera.gravity_source, "record": CLIP_RECORD_FILE if accepted else None, "rule": source,
            # healthy solves span 0.0831-2.2774 m; <= 0 is physically impossible output
            "min_camera_z": depth, "behind_camera": bool(behind_camera),
            "rejected_for_behind_camera": bool(behind_camera and reject_behind_camera),
            # which trajectory was scored and exported: the solver's, or the contact stage's corrected one
            "poses_source": inputs.poses_source,
            "contact_shift_mm_median": (float(np.median(np.abs(inputs.contact_shift_mm)))
                                        if inputs.contact_shift_mm is not None else None)}
        if interaction is not None:          # only when that screen is enabled: the default qc.json is unchanged
            payload["interaction"] = interaction
            payload["rejected_for_no_interaction"] = not interaction["passed"]
        if world_motion is not None:         # only when that screen is enabled: the default qc.json is unchanged
            payload["world_motion"] = world_motion
            payload["rejected_for_no_world_motion"] = not world_motion["passed"]
        write_json(artifact_dir / QC_FILE, payload)
        # Print the score that is ACTUALLY compared, not just the confidence. The rule accepts iff
        # `score <= threshold` with `score = 1 - P(good)`, while `confidence` is `P(good)` -- so logging the
        # confidence beside a score threshold invites reading 0.458 < 0.6908 as a contradiction of "accepted".
        logger.info("qc of %r: %s (score %.4f %s threshold %.4f; confidence %.3f)", clip_id,
                    "accepted" if accepted else "rejected", 1.0 - confidence,
                    "<=" if accepted else ">", rule.calibration.threshold, confidence)

    return qc_worker


@dataclass(frozen=True)
class QcStageConfig:
    """The `qc_stage` config block: the saved rule's path (None: the stage cannot be built) and the Laplace threshold."""

    rule_path: str | None = None
    laplace_std_threshold: float = 0.01
    interaction_screen: InteractionScreenConfig = dataclasses.field(default_factory=InteractionScreenConfig)
    world_motion_screen: WorldMotionScreenConfig = dataclasses.field(default_factory=WorldMotionScreenConfig)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> QcStageConfig:
        """From the `qc_stage` block (a dict or an OmegaConf node).

        Raises:
            ValueError: On an unknown key (here or in `interaction_screen`).
        """
        known = ["interaction_screen", "laplace_std_threshold", "rule_path", "world_motion_screen"]
        unknown = sorted({str(key) for key in mapping} - set(known))
        if unknown:
            raise ValueError(f"qc_stage has unknown keys {unknown}; known keys: {known}")
        rule_path = mapping.get("rule_path")
        screen = mapping.get("interaction_screen")
        world = mapping.get("world_motion_screen")
        return cls(rule_path=None if rule_path is None else str(rule_path),
                   laplace_std_threshold=float(mapping.get("laplace_std_threshold", 0.01)),
                   interaction_screen=InteractionScreenConfig.from_mapping(
                       None if screen is None else {str(k): screen[k] for k in screen}),
                   world_motion_screen=WorldMotionScreenConfig.from_mapping(
                       None if world is None else {str(k): world[k] for k in world}))


def make_qc_worker_from_config(cfg: QcStageConfig, features_fn: FeaturesFn = qc_features, *,
                               driftqc_enabled: bool = False) -> StageFn:
    """The `qc` stage worker of the rule saved at `cfg.rule_path`, loaded and checked at construction, before any clip.

    There is no accept-all default. The file is read once: `qc.json["rule"]` records `{"path":
    <resolved path>, "sha256": <sha256 of the bytes the rule was parsed from>, "version": 1}`. `driftqc_enabled` is passed
    to `make_qc_worker`.

    Raises:
        ValueError: If `cfg.rule_path` is None or empty, or the file is not a valid version-1 rule.
        FileNotFoundError: If the rule file is absent.
    """
    if not cfg.rule_path:
        raise ValueError("qc_stage.rule_path is not set, and the qc stage has no accept-all default: "
                         "fit and save an AcceptanceRule with run/pipeline/calibrate_acceptance.py, then set "
                         "qc_stage.rule_path to its file")
    path = Path(cfg.rule_path).resolve()
    rule, sha256 = load_acceptance_rule_and_sha256(path)     # one read: the recorded hash is of the parsed bytes
    logger.info("qc stage rule %s (sha256 %s): threshold %.4f", path, sha256[:12], rule.calibration.threshold)
    source = {"path": str(path), "sha256": sha256, "version": ACCEPTANCE_RULE_VERSION}
    return make_qc_worker(rule, features_fn, cfg.laplace_std_threshold, rule_source=source,
                          driftqc_enabled=driftqc_enabled, interaction_screen=cfg.interaction_screen,
                          world_motion_screen=cfg.world_motion_screen)
