"""DexYCB policy references: `make_reference_fn(scenes_root, hands_root, cfg)`.

One DexYCB `ClipRecord` (GT object poses and MANO hand on every source frame, camera frame) becomes one
`PolicyReference` at the policy rate, through the engine's retarget stage:

1. Skip, logged and recorded in `records`, never kept, when any of these holds:
   - no validated scene exists for the clip's object and hand side (`run/pipeline/build_ycb_scenes.py`);
   - the hands were not built, failed validation, or were built for another camera or hand
     (`run/pipeline/build_dexycb_hands.py`);
   - no MANO model is configured for the side (`retarget_stage.mano_pkl_path`);
   - the GT object pose is missing on any row (every row is a spline knot).
2. World frame: the DexYCB extrinsics frame (`fastdo.io.datasets.dexycb_world`, table plane z = 0, z up), not the
   engine's static-camera fallback.
3. Engine rows: every `engine_stride`-th source row from the first, the engine's 10 fps decode rows at stride 3. The
   reference spans the first to the last engine row at the source rate.
4. The composed `fastdo.qc.export.run_retarget_stage` on the engine rows:
   - its prepare half: world conversion; C2 splines of the object and wrist poses; hand vertices lerped in the wrist
     frame; invalid hand rows masked; the IK-site wrist target by default;
   - its solve half: one `retarget_kinematic` call at the full source rate. Robot joints are solved at that rate, never
     spline-upsampled.
   The prepare half is deterministic, so step 5 reads the reference-rate object poses, hand vertices and frame mask from
   `prepare_retarget_stage` on the same arguments.
5. The `PolicyReference`, via `resample_reference` to `hz`, holds:
   - the solve's joints and the upsampled world object poses;
   - the contact plan: per finger label, any MANO vertex in the object's contact band (`human_contact_measure` with the
     solver's sigma and tau), on frames whose hand input is evidence;
   - a synthetic diagonal posterior (`pos_std`, `rot_std`, `log_scale_std`), since GT has no M1 solve.
6. Clearance (reported, not gated on 1 mm): `primitive_clearance_stats` of the solve, per clip in
   `records`. `clearance_report` gives, per object and hand side, the minimum clearance and the fraction of frames
   clearing 0.5 mm.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from fastdo.io.clip_record import ClipRecord, HandTrack
from fastdo.io.datasets.dexycb_world import (
    WORLD_FRAME,
    dexycb_cam_to_world,
    select_rows,
    with_world_frame,
)
from fastdo.policy.data import grasped_object
from fastdo.policy.dexycb_inputs import (
    SIDES,
    ObjectGeometry,
    hands_skip_reason,
    load_hand_vertices,
    object_geometry,
    read_scene_record,
    scene_skip_reason,
)
from fastdo.policy.reference import PolicyReference, resample_reference
from fastdo.qc.export import prepare_retarget_stage, retarget_hand_side, run_retarget_stage
from fastdo.qc.reference_rate import ReferenceInputs
from fastdo.retarget.clearance import primitive_clearance_stats, robot_contact_frames
from fastdo.retarget.contact import human_contact_measure
from fastdo.retarget.kinematic import KinematicTrajectory, RetargetConfig
from fastdo.retarget.labels import FINGER_LABEL_NAMES, mano_finger_labels
from fastdo.retarget.mj_model import RobotScene
from fastdo.retarget.wrist_target import DEFAULT_WRIST_TARGET, WRIST_TARGETS

logger = logging.getLogger(__name__)

__all__ = ["DexYCBRefConfig", "DexYCBReferenceFn", "clearance_report", "make_reference_fn"]

_N_LABELS = len(FINGER_LABEL_NAMES)


@dataclass(frozen=True)
class DexYCBRefConfig:
    """Settings of `make_reference_fn` (module docstring). MANO models per side come from `retarget_stage.mano_pkl_path`
    ("" = not configured); `retarget` holds the kinematic solver's weights (default: the merged defaults)."""

    dexycb_root: str
    mano_right_pkl: str
    mano_left_pkl: str = ""
    hz: float = 50.0
    engine_stride: int = 3
    wrist_target: str = DEFAULT_WRIST_TARGET
    retarget: RetargetConfig = field(default_factory=RetargetConfig)
    pos_std: float = 1e-3
    rot_std: float = 1e-2
    log_scale_std: float = 1e-2
    sdf_resolution: int = 64
    n_surface: int = 1024
    surface_seed: int = 0
    clearance: bool = True

    def __post_init__(self) -> None:
        if self.engine_stride < 1 or not self.hz > 0.0:
            raise ValueError(f"need engine_stride >= 1 and hz > 0, got {self.engine_stride}, {self.hz}")
        if self.wrist_target not in WRIST_TARGETS:
            raise ValueError(f"wrist_target must be one of {WRIST_TARGETS}, got {self.wrist_target!r}")

    def mano_pkl(self, side: str) -> str:
        return {"right": self.mano_right_pkl, "left": self.mano_left_pkl}[side]


def _source_rate(record: ClipRecord) -> float:
    """Source frames per second over the record's rows, as `fastdo.engine.workers_retarget.source_frequency`."""
    return float(record.frame_idx[-1] - record.frame_idx[0]) / float(record.timestamps[-1] - record.timestamps[0])


class DexYCBReferenceFn:
    """The function `make_reference_fn` returns (module docstring).

    - `fn(record) -> PolicyReference | None`.
    - `fn.skip_reason(clip_id, object, side)`: the checks that need no record.
    - `fn.records`: one entry per call, kept or skipped, with the reason, sizes, solve seconds and clearance.
    """

    def __init__(self, scenes_root: str | Path, hands_root: str | Path, cfg: DexYCBRefConfig) -> None:
        self.scenes_root, self.hands_root, self.cfg = Path(scenes_root), Path(hands_root), cfg
        self.records: list[dict[str, Any]] = []
        self._scenes: dict[str, RobotScene] = {}
        self._geometry: dict[str, ObjectGeometry] = {}
        self._labels: dict[str, Tensor] = {}

    def skip_reason(self, clip_id: str, object_name: str, side: str | None) -> str | None:
        """Why the clip `"<subject>/<sequence>/<serial>"` of `object_name` held by `side` cannot become a reference
        (scene, hands, MANO model), or None."""
        if side not in SIDES:
            return f"the clip has no MANO hand side ({side!r})"
        reason = scene_skip_reason(read_scene_record(self.scenes_root, object_name, side), object_name, side)
        reason = reason or hands_skip_reason(self.hands_root, clip_id, side)
        if reason is None and not self.cfg.mano_pkl(side):
            reason = f"no MANO model configured for the {side} hand (retarget_stage.mano_pkl_path.{side})"
        return reason

    def __call__(self, record: ClipRecord) -> PolicyReference | None:
        name = grasped_object(record)
        side, track = retarget_hand_side(record, None)
        entry: dict[str, Any] = {"clip_id": record.clip_id, "object": name, "side": side}
        reason = self.skip_reason(record.clip_id, name, side if track is not None else None)
        missing = np.flatnonzero(np.asarray(record.confidence) <= 0.0)
        if reason is None and missing.size:
            reason = f"object pose missing on rows {missing[:8].tolist()} (every row is a spline knot)"
        if reason is not None or track is None:
            self._skip(entry, reason or "no hand track")
            return None
        return self._reference(record, name, side, track, entry)

    def _skip(self, entry: dict[str, Any], reason: str) -> None:
        self.records.append({**entry, "status": "skipped", "reason": reason})
        logger.warning("DexYCB reference %s (%s, %s hand) skipped, not kept: %s", entry["clip_id"], entry["object"],
                       entry["side"], reason)

    def _reference(self, record: ClipRecord, name: str, side: str, track: HandTrack,
                   entry: dict[str, Any]) -> PolicyReference | None:
        cfg = self.cfg
        world = with_world_frame(record, dexycb_cam_to_world(cfg.dexycb_root, record.clip_id))
        sequence = "/".join(record.clip_id.split("/")[:2])
        verts = load_hand_vertices(self.hands_root, sequence, side, np.asarray(track.valid, dtype=bool))
        rows = np.arange(0, record.n_frames, cfg.engine_stride)
        engine, engine_verts = select_rows(world, rows), verts[torch.from_numpy(rows)]
        scene, geometry, labels = self._scene(name, side), self._object(record.object_mesh_path), self._hand_labels(side)
        frequency, start = _source_rate(engine), time.perf_counter()
        kin = run_retarget_stage(engine, engine_verts, labels, scene, geometry.sdf, geometry.surface, geometry.normals,
                                 frequency, cfg.retarget, side=side, wrist_target=cfg.wrist_target)
        seconds = time.perf_counter() - start
        inputs = None if kin is None else prepare_retarget_stage(engine, engine_verts, side=side,
                                                                   wrist_target=cfg.wrist_target)
        if kin is None or inputs is None:
            self._skip(entry, "run_retarget_stage skipped the clip: no valid hand row on the engine rows")
            return None
        plan = self._contact_plan(inputs, labels, geometry, record.object_scale)
        variances = torch.tensor([cfg.pos_std**2] * 3 + [cfg.rot_std**2] * 3 + [cfg.log_scale_std**2],
                                 dtype=torch.float64)
        ref = resample_reference(kin, inputs.obj_poses, plan, torch.diag(variances).repeat(inputs.n_frames, 1, 1),
                                 hz=cfg.hz, log_scale=math.log(record.object_scale))
        clearance = self._clearance(kin, scene, geometry, record.object_scale) if cfg.clearance else None
        self.records.append({**entry, "status": "kept", "n_rows": int(rows.size), "n_frames": inputs.n_frames,
                             "n_policy_frames": ref.num_frames, "frequency": frequency, "hz": cfg.hz,
                             "masked_rows": inputs.masked_rows, "masked_frames": inputs.masked_frames,
                             "planned_contact_frames": int(plan.any(dim=1).sum()), "world_frame": WORLD_FRAME,
                             "wrist_target": cfg.wrist_target, "scene_xml": str(scene.xml_path),
                             "retarget_stage_seconds": seconds, "clearance": clearance})
        logger.info("DexYCB reference %s (%s, %s hand): %d engine rows -> %d frames at %.4g Hz -> %d at %g Hz in %.2f s; "
                    "clearance min %s mm, %s of frames >= 0.5 mm", record.clip_id, name, side, rows.size,
                    inputs.n_frames, frequency, ref.num_frames, cfg.hz, seconds,
                    None if clearance is None else f"{clearance['frame_min_mm_min']:.3f}",
                    None if clearance is None else f"{clearance['frac_frames_clear_0p5mm']:.3f}")
        return ref

    def _scene(self, name: str, side: str) -> RobotScene:
        record = read_scene_record(self.scenes_root, name, side)
        assert record is not None                           # checked by skip_reason
        xml = str(record["scene_xml"])
        if xml not in self._scenes:
            self._scenes[xml] = RobotScene.from_xml(xml)
        return self._scenes[xml]

    def _object(self, mesh_path: str) -> ObjectGeometry:
        if mesh_path not in self._geometry:
            cfg = self.cfg
            self._geometry[mesh_path] = object_geometry(mesh_path, cfg.sdf_resolution, cfg.n_surface, cfg.surface_seed)
        return self._geometry[mesh_path]

    def _hand_labels(self, side: str) -> Tensor:
        if side not in self._labels:
            self._labels[side] = mano_finger_labels(self.cfg.mano_pkl(side))
        return self._labels[side]

    def _contact_plan(self, inputs: ReferenceInputs, labels: Tensor, geometry: ObjectGeometry, scale: float) -> Tensor:
        """`(N, L)` bool: per finger label, any MANO vertex of that label in the object's contact band, on the frames
        whose hand input is evidence (`frame_valid`)."""
        retarget = self.cfg.retarget
        measure = human_contact_measure(inputs.hand_verts, inputs.obj_poses, scale, labels, geometry.sdf,
                                        geometry.surface, sigma=retarget.sigma, tau=retarget.tau)
        touching = measure.weights > 0.0
        by_label = torch.stack([touching[:, labels.long() == label].any(dim=1) for label in range(_N_LABELS)], dim=1)
        return by_label & inputs.frame_valid.to(torch.bool)[:, None]

    def _clearance(self, kin: KinematicTrajectory, scene: RobotScene, geometry: ObjectGeometry,
                   scale: float) -> dict[str, float | int]:
        retarget = self.cfg.retarget
        frames = robot_contact_frames(kin.qpos, scene, geometry.sdf, geometry.surface, scale, retarget.n_per_link,
                                      retarget.seed)
        return primitive_clearance_stats(kin.qpos, scene, geometry.sdf, scale, frames, retarget.box_face_tol).as_dict()


def make_reference_fn(scenes_root: str | Path, hands_root: str | Path, cfg: DexYCBRefConfig) -> DexYCBReferenceFn:
    """The DexYCB reference function (module docstring) over the scene records under `scenes_root` and the GT hands
    under `hands_root` (`fastdo.policy.dexycb_inputs` layout)."""
    return DexYCBReferenceFn(scenes_root, hands_root, cfg)


def clearance_report(records: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per object and hand side, keyed `<object>/<side>`: the object and side, kept clips, skipped clips, pooled frames,
    the minimum clearance over its clips (mm) and the frame-weighted fractions of frames clearing 0.5 mm and
    penetrating; None where no kept clip carries clearance. Reported only: references are never gated on it."""
    totals: dict[str, dict[str, Any]] = {}
    for record in records:
        name, side = str(record["object"]), str(record.get("side"))
        row = totals.setdefault(f"{name}/{side}", {"object": name, "side": side, "n_clips": 0, "n_skipped": 0,
                                                   "n_frames": 0, "min_clearance_mm": None, "clear": 0.0,
                                                   "penetrating": 0.0})
        if record.get("status") != "kept":
            row["n_skipped"] += 1
            continue
        clearance = record.get("clearance")
        row["n_clips"] += 1
        if not clearance:
            continue
        n = int(clearance["n_frames"])
        row["n_frames"] += n
        low = float(clearance["frame_min_mm_min"])
        row["min_clearance_mm"] = low if row["min_clearance_mm"] is None else min(row["min_clearance_mm"], low)
        row["clear"] += float(clearance["frac_frames_clear_0p5mm"]) * n
        row["penetrating"] += float(clearance["frac_frames_penetrating"]) * n
    report: dict[str, dict[str, Any]] = {}
    for key, row in totals.items():
        n = row["n_frames"]
        report[key] = {"object": row["object"], "side": row["side"], "n_clips": row["n_clips"],
                       "n_skipped": row["n_skipped"], "n_frames": n,
                       "min_clearance_mm": row["min_clearance_mm"],
                       "frac_frames_clear_0p5mm": row["clear"] / n if n else None,
                       "frac_frames_penetrating": row["penetrating"] / n if n else None}
    return report
