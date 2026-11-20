"""Reference sets and the curriculum of the tracking policy (design Sections 8-9).

Object-level split (design Section 9: held-out DexYCB objects, not held-out sequences). `object_split(objects,
n_held_out, seed)` picks the held-out objects deterministically. `build_reference_set(dexycb_root, objects, max_seqs,
cfg)` walks the DexYCB sequences in reader order, reads each sequence's `meta.yml` for its grasped object and keeps
the sequence when that object is in `objects` and, by `cfg.split`, outside (`"train"`) or inside (`"held_out"`)
`cfg.held_out_objects`. A held-out object therefore never enters a training set, whatever `objects` holds. One camera
view is used per sequence (`cfg.serial_index`): the views film the same motion, so more would duplicate it. Each
`ReferenceClip` carries its sequence id and grasped object, the keys of per-object reporting; `build_reference_set`
returns the references alone.

References. `cfg.reference_fn` turns one DexYCB `ClipRecord` into a `PolicyReference` at the policy rate, or declines
it with None (logged, never kept, not counted by `max_seqs`). The DexYCB function is `fastdo.policy.dexycb_refs.
make_reference_fn` (per-object Sharpa scenes, GT MANO hands, the retarget stage); `build_reference_set` refuses
to run without a function. Before a record loads, `cfg.hand_side` keeps only sequences whose `meta.yml` hand is that side
and `cfg.skip_reason(sequence id, object, side)` may decline a sequence with a reason (e.g. its hands were never built),
so declined sequences cost no label reads. The default reader never lists `cfg.exclude_subjects`. Each `ReferenceClip`
carries its hand side, which with its object names its scene. The shipped whisking demo (`whisking_policy_reference`)
is one clip of one object, named by `scene_object_name` from its scene's `objects/<name>/` meshes.

Curriculum (design Section 8): the randomization strength ramps linearly from 0 to 1 x Sigma over `strength_updates`
PPO updates and the tracked clip length from `start_seconds` to the full clip over `length_updates`, re-applied every
`stage_updates` (`curriculum_stage`). `curriculum_reference` realizes a stage on one reference: its covariance scaled by
strength^2 (the environment draws the Section 6 perturbation from it when `EnvConfig.randomize`) and its first frames.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from fastdo.io.clip_record import ClipRecord
from fastdo.io.datasets.dexycb import DexYCBReader, load_dexycb_yaml, ycb_object_name
from fastdo.policy.reference import PolicyReference, resample_reference

logger = logging.getLogger(__name__)

__all__ = ["CurriculumConfig", "ReferenceClip", "ReferenceFn", "ReferenceSetConfig", "SkipReason", "build_reference_clips",
           "build_reference_set", "curriculum_reference", "curriculum_stage", "grasped_object", "object_split",
           "scene_object_name", "whisking_policy_reference"]

ReferenceFn = Callable[[ClipRecord], PolicyReference | None]
SkipReason = Callable[[str, str, str | None], str | None]
_SPLITS = ("train", "held_out")
_SIDES = ("left", "right")


class _Reader(Protocol):
    root: Path

    def sequences(self) -> list[str]: ...

    def load(self, seq: str) -> ClipRecord: ...


@dataclass(frozen=True)
class ReferenceClip:
    """One reference with its provenance: `clip_id` (the DexYCB sequence and view, or the demo's id) and the
    grasped object's name, the key of per-object reporting."""

    clip_id: str
    object_name: str
    ref: PolicyReference
    hand_side: str | None = None


@dataclass(frozen=True)
class ReferenceSetConfig:
    """Which side of the object split to build and how (module docstring)."""

    held_out_objects: tuple[str, ...] = ()
    split: str = "train"
    reference_fn: ReferenceFn | None = None
    mano_pca: str | None = None
    serial_index: int = 0
    hand_side: str | None = None
    skip_reason: SkipReason | None = None
    exclude_subjects: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.split not in _SPLITS:
            raise ValueError(f"split must be one of {_SPLITS}, got {self.split!r}")
        if self.hand_side is not None and self.hand_side not in _SIDES:
            raise ValueError(f"hand_side must be one of {_SIDES} or None, got {self.hand_side!r}")
        if self.serial_index < 0:
            raise ValueError(f"serial_index must be >= 0, got {self.serial_index}")


@dataclass(frozen=True)
class CurriculumConfig:
    """Randomization-strength and clip-length schedules over PPO updates (module docstring)."""

    strength_updates: int = 500
    start_seconds: float = 1.0
    end_seconds: float = 10.0
    length_updates: int = 1000
    stage_updates: int = 50

    def __post_init__(self) -> None:
        if min(self.strength_updates, self.length_updates, self.stage_updates) < 1:
            raise ValueError(f"curriculum updates must be >= 1, got {self}")
        if not 0.0 < self.start_seconds <= self.end_seconds:
            raise ValueError(f"need 0 < start_seconds <= end_seconds, got {self.start_seconds}, {self.end_seconds}")


def grasped_object(record: ClipRecord) -> str:
    """The YCB model name of a DexYCB clip's grasped object (its mesh directory, e.g. `006_mustard_bottle`)."""
    return Path(record.object_mesh_path).parent.name


def object_split(objects: Sequence[str], n_held_out: int, seed: int) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """`(train, held_out)`: `n_held_out` distinct objects drawn with `seed`, both sorted; the same inputs always give
    the same split.

    Raises:
        ValueError: On duplicate objects or `n_held_out` outside [0, len(objects)).
    """
    names = sorted(objects)
    if len(set(names)) != len(names) or not 0 <= n_held_out < len(names):
        raise ValueError(f"need distinct objects and 0 <= n_held_out < {len(names)}, got {n_held_out}")
    chosen = np.random.default_rng(seed).permutation(len(names))[:n_held_out]
    held_out = tuple(sorted(names[i] for i in chosen))
    return tuple(name for name in names if name not in held_out), held_out


def _sequence_meta(root: Path, seq: str) -> tuple[str, str | None]:
    """`(grasped object, hand side or None)` from the sequence's `meta.yml`."""
    subject, sequence, _ = seq.split("/")
    meta = load_dexycb_yaml(root / subject / sequence / "meta.yml")
    sides = list(meta.get("mano_sides") or [])
    return ycb_object_name(int(list(meta["ycb_ids"])[int(meta["ycb_grasp_ind"])])), str(sides[0]) if sides else None


def build_reference_set(dexycb_root: str | Path, objects: Sequence[str], max_seqs: int, cfg: ReferenceSetConfig,
                        reader: _Reader | None = None) -> list[PolicyReference]:
    """The references of `build_reference_clips` (same arguments and errors)."""
    return [clip.ref for clip in build_reference_clips(dexycb_root, objects, max_seqs, cfg, reader)]


def build_reference_clips(dexycb_root: str | Path, objects: Sequence[str], max_seqs: int, cfg: ReferenceSetConfig,
                          reader: _Reader | None = None) -> list[ReferenceClip]:
    """Clips of up to `max_seqs` DexYCB sequences on `cfg.split`'s side of the object split (module docstring;
    `max_seqs <= 0` keeps all), each with its sequence id and grasped object. `reader` defaults to
    `DexYCBReader(dexycb_root, cfg.mano_pca)`.

    Raises:
        ValueError: Without `cfg.reference_fn`, or when the split leaves no allowed object.
    """
    if cfg.reference_fn is None:
        raise ValueError("build_reference_clips needs cfg.reference_fn (DexYCB clip -> PolicyReference): the kinematic "
                         "reference needs a Sharpa scene with the clip's object and its MANO vertices")
    held = set(cfg.held_out_objects)
    allowed = {o for o in objects if (o in held) == (cfg.split == "held_out")}
    if not allowed:
        raise ValueError(f"no {cfg.split} object among {sorted(objects)} (held out: {sorted(held)})")
    source = reader if reader is not None else DexYCBReader(Path(dexycb_root), Path(cfg.mano_pca) if cfg.mano_pca
                                                            else None, exclude_subjects=cfg.exclude_subjects)
    views: dict[str, list[str]] = {}
    for seq in source.sequences():
        views.setdefault(seq.rsplit("/", 1)[0], []).append(seq)
    refs: list[ReferenceClip] = []
    for sequence, serials in views.items():
        if 0 < max_seqs <= len(refs):
            break
        if cfg.serial_index >= len(serials):
            continue
        seq = serials[cfg.serial_index]
        name, side = _sequence_meta(source.root, seq)
        if name not in allowed or (cfg.hand_side is not None and side != cfg.hand_side):
            continue
        reason = cfg.skip_reason(seq, name, side) if cfg.skip_reason is not None else None
        if reason is not None:
            logger.info("%s (%s, %s hand) skipped before loading: %s", seq, name, side, reason)
            continue
        record = source.load(seq)
        if grasped_object(record) != name:
            raise ValueError(f"{seq}: meta.yml grasps {name} but the record's mesh is {record.object_mesh_path}")
        ref = cfg.reference_fn(record)
        if ref is None:
            logger.warning("%s (%s, %s hand): the reference function declined the clip; not kept", seq, name, side)
            continue
        refs.append(ReferenceClip(seq, name, ref, side))
        logger.debug("reference %d: %s (%s, %s hand)", len(refs), sequence, name, side)
    logger.info("%s reference set: %d sequences of %d allowed objects", cfg.split, len(refs), len(allowed))
    return refs


def scene_object_name(scene_xml: str | Path) -> str:
    """The object of a baseline retargeting scene: `<name>` of its `objects/<name>/...` mesh files (`whisking` for the
    whisking demo).

    Raises:
        ValueError: If the scene's meshes belong to no object or to more than one.
    """
    from xml.etree import ElementTree

    names: set[str] = set()
    for mesh in ElementTree.parse(scene_xml).iter("mesh"):
        parts = Path(mesh.get("file", "")).parts
        if len(parts) > 2 and parts[0] == "objects":
            names.add(parts[1])
    if len(names) != 1:
        raise ValueError(f"{scene_xml}: expected the meshes of one object under objects/, found {sorted(names)}")
    return names.pop()


def whisking_policy_reference(npz: str | Path, hz: float = 50.0, n_frames: int | None = None, pos_std: float = 1e-3,
                              rot_std: float = 1e-2, log_scale_std: float = 1e-2) -> PolicyReference:
    """The shipped whisking kinematic reference (`trajectory_kinematic.npz`: `qpos (T, 35)` with the object free joint
    at 28:35 and its own `frequency`) as a policy reference at `hz`: object poses from the free joint, an empty contact
    plan (the demo ships no C2 contact labels) and a synthetic diagonal posterior (`pos_std` m, `rot_std` rad,
    `log_scale_std`) on every frame, since the shipped reference has no M1 solve. `n_frames` crops the source clip."""
    from fastdo.geometry.lie import quat_wxyz_to_matrix
    from fastdo.policy.scene import N_LABELS
    from fastdo.retarget.kinematic import KinematicTrajectory

    with np.load(npz) as data:
        qpos, qvel, frequency = data["qpos"], data["qvel"], float(data["frequency"])
    if n_frames is not None:
        qpos, qvel = qpos[:n_frames], qvel[:n_frames]
    q = torch.as_tensor(qpos, dtype=torch.float64)
    T = int(q.shape[0])
    pose = torch.eye(4, dtype=torch.float64).repeat(T, 1, 1)
    pose[:, :3, :3], pose[:, :3, 3] = quat_wxyz_to_matrix(q[:, 31:35]), q[:, 28:31]
    variances = torch.tensor([pos_std**2] * 3 + [rot_std**2] * 3 + [log_scale_std**2], dtype=torch.float64)
    return resample_reference(KinematicTrajectory(qpos=qpos, qvel=qvel, frequency=frequency), pose,
                              torch.zeros(T, N_LABELS, dtype=torch.bool), torch.diag(variances).repeat(T, 1, 1), hz=hz)


def curriculum_stage(update: int, cfg: CurriculumConfig) -> tuple[float, float]:
    """`(strength, seconds)` of the stage holding `update`, both evaluated at the stage's first update: the length
    ramps linearly from `start_seconds` to `end_seconds` and is `inf` (every clip in full) once the ramp is over."""
    start = (max(update, 0) // cfg.stage_updates) * cfg.stage_updates
    strength = min(1.0, start / cfg.strength_updates)
    if start >= cfg.length_updates:
        return strength, math.inf
    fraction = start / cfg.length_updates
    return strength, cfg.start_seconds + fraction * (cfg.end_seconds - cfg.start_seconds)


def curriculum_reference(ref: PolicyReference, strength: float, seconds: float) -> PolicyReference:
    """`ref` cropped to its first `round(seconds hz) + 1` frames (at least 2; all of them for `inf`) with its covariance
    scaled by `strength^2`.

    Raises:
        ValueError: If `strength` lies outside [0, 1] or `seconds` is not positive.
    """
    if not 0.0 <= strength <= 1.0 or not seconds > 0.0:
        raise ValueError(f"need strength in [0, 1] and seconds > 0, got {strength}, {seconds}")
    frames = ref.num_frames if math.isinf(seconds) else min(ref.num_frames, max(2, round(seconds * ref.hz) + 1))
    cov = None if ref.covariance is None else ref.covariance[:frames] * strength**2
    return dataclasses.replace(ref, q_ref=ref.q_ref[:frames], obj_pose=ref.obj_pose[:frames],
                               contact_plan=ref.contact_plan[:frames], covariance=cov)
