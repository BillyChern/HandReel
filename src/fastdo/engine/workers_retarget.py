"""The `retarget` stage worker: the clip's full-rate kinematic reference, `trajectory_kinematic.npz`.

`retarget` runs after `qc` and only on a clip whose `qc.json` says `accepted`; otherwise it writes `retarget.json` with
`status` "skipped" and removes a stale reference. For an accepted clip it reads `clip_record.npz`; the hand side's
camera-frame MANO vertices from `hands.npz` (`{side}_vertices`: `HandTrack` carries no vertices, so they never come
from the `ClipRecord`); the per-vertex finger labels `mano_finger_labels(cfg.mano_pkl_path(side))`; and the shape
stage's `sdf.npz` and `samples.npz` (`surface`, `surface_normals`). It then runs `fastdo.qc.export.run_retarget_stage`'s
two parts: `prepare_retarget_stage` among the input checks, and the solve (`stage_fn`, `solve_retarget_stage` by
default) on the worker's `RobotScene`, which loads once, on the first accepted clip.

Wrist target. `cfg.wrist_target` selects the wrist prior's target (`fastdo.retarget.wrist_target`):
`ik_site` (the default) is the baseline IK-site target, validated in physics; `mcp` is
the MCP geometric frame `human_wrist_poses`, which has not passed physics.

Grid and frequency. The reference has one frame per source frame between the first and last decode row; nothing
resamples that grid. Its frequency is `cfg.reference_frequency`, else the source video rate from the record's
timestamps. A configured frequency only relabels the grid: it scales `qvel` and playback speed by its ratio to the
source rate (the baseline labels the 29.97 fps whisking frames 50 Hz), so one more than 1% off the source rate is logged
as a WARNING.

`retarget.json` records `status`, the hand `side`, the `wrist_target`, `n_rows` (engine rows), `n_reference_frames`,
`masked_rows` (hand rows with `valid` False), `masked_frames` (reference frames without contacts or wrist prior),
`frequency`, `source_frequency` (null for a one-row clip), the solve seconds and `retarget_fallback_needed`. Every
failure writes `status` "failed" with `error` and `error_type`, removes a stale reference and re-raises, so the
orchestrator marks the manifest's `retarget` stage "failed". Failures come in two kinds:

- Unusable inputs or configuration, found before the solve. `retarget_fallback_needed` stays False: the MPC fallback
  cannot repair a data or contract bug. These are: a missing artifact; rows that are not the decode rows;
  `{side}_vertices` that are not `(T,V,3)` for the record's `T` rows with one label per vertex, or that are non-finite
  on a valid row; `hands.npz` outputs that are not in `intrinsics.npz`'s camera (`require_hands_camera`); a
  missing MANO pkl; anything `prepare_retarget_stage` raises (a non-finite camera or object pose,
  non-finite joints or global orient on a valid row, an error while composing the reference-rate inputs); a scene that
  does not load.
- The kinematic solve: `stage_fn` raises, or returns a trajectory whose `qpos` or `qvel` is not finite.
  `retarget_fallback_needed` is True. The baseline MPC fallback that marker asks for is outside this stage.

Hand side and scene. `side_from_prompt` False (the default) keeps the configured `side` rule: the configured
side, else right when the record has a right hand, else left. With `side_from_prompt` True and `side` null, the
retargeted hand is the INTERACTING one: the prompt stage's `prompt.json` `choice.side` (the hand of the chosen
hand-object box); without a prompt (DexYCB: the bench prompts from the GT label map, no `prompt.json`) the hand the
geometry stage anchored the clip on (`geometry.npz` `hand_sides`, when it is one side); else the rule. Measured:
the rule retargets the wrong hand on 25 of 48 Ego4D windows. `retarget.json` records the `side_source` (`config`,
`prompt`, `geometry` or `rule`). A left hand is solved in `scene_xml_left` (the right in `scene_xml`); a left hand
with no left scene configured is an unusable configuration (ValueError, never a left hand on the right robot).
Each side's scene loads once, on its first clip.

Skip reasons: `retarget_skip_reason` names why `prepare_retarget_stage` skips (mean tracker confidence not positive,
no valid hand row, or no row with both a valid hand and a supported object).

driftqc: with `driftqc_enabled` True, the `hands.npz` rows are sliced to the clip's `valid_range.json` prefix through
the single slicing helper `fastdo.engine.valid_range.slice_artifact` before the decode-row check against
`record.frame_idx` (the qc stage already wrote the record on the prefix). With it False (the default) a leftover
`driftqc.json` or `valid_range.json` is an unusable-input failure (ValueError) and nothing is ever sliced.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from fastdo.engine.orchestrator import StageFn
from fastdo.engine.valid_range import resolve_valid_range, slice_artifact
from fastdo.engine.worker_support import WorkerContext, load_npz, read_frame_idx, require_artifact
from fastdo.engine.workers_geometry import GEOMETRY_FILE
from fastdo.engine.workers_hands import HANDS_FILE, SIDES, decode_frame_size
from fastdo.engine.workers_prompt import PROMPT_META_FILE
from fastdo.engine.workers_qc import CLIP_RECORD_FILE, QC_FILE, check_decode_rows, write_json
from fastdo.engine.workers_shape import read_mesh_samples, read_sdf_grid
from fastdo.geometry.intrinsics import read_intrinsics
from fastdo.hands.hand_camera import require_hands_camera
from fastdo.io.clip_record import ClipRecord
from fastdo.qc.export import (
    prepare_retarget_stage,
    retarget_hand_side,
    retarget_row_valid,
    retarget_skip_reason,
    solve_retarget_stage,
)
from fastdo.qc.reference_rate import reference_frame_valid, reference_key_frames
from fastdo.retarget.kinematic import KinematicTrajectory, RetargetConfig
from fastdo.retarget.labels import mano_finger_labels
from fastdo.retarget.mj_model import RobotScene
from fastdo.retarget.wrist_target import DEFAULT_WRIST_TARGET, WRIST_TARGETS

__all__ = ["RETARGET_FILE", "SIDE_SOURCES", "TRAJECTORY_FILE", "RetargetStageConfig", "interacting_side",
           "make_retarget_worker", "source_frequency"]

logger = logging.getLogger(__name__)

RETARGET_FILE = "retarget.json"
TRAJECTORY_FILE = "trajectory_kinematic.npz"
FREQUENCY_REL_TOL = 0.01
SIDE_SOURCES = ("config", "prompt", "geometry", "rule")
StageRunner = Callable[..., KinematicTrajectory]


@dataclass(frozen=True)
class RetargetStageConfig:
    """Settings of the `retarget` stage (Hydra block `retarget_stage` in `run/conf/config.yaml`).

    `scene_xml`: the right-hand robot scene the reference is solved in; `scene_xml_left`: the left-hand one ("" = not
    configured). `mano_right_pkl` / `mano_left_pkl`: per-side MANO model for the finger labels ("" = not configured).
    `side`: hand to retarget (None: right when present, else left, or the interacting hand with `side_from_prompt`;
    module docstring). `wrist_target`: `ik_site` (default) or `mcp` (module docstring). `reference_frequency`: Hz
    label of the source-frame grid (None: the source video rate); it does not resample. `retarget`: solver weights
    and schedule.
    """

    scene_xml: str
    mano_right_pkl: str
    mano_left_pkl: str = ""
    side: str | None = None
    wrist_target: str = DEFAULT_WRIST_TARGET
    reference_frequency: float | None = None
    retarget: RetargetConfig = field(default_factory=RetargetConfig)
    scene_xml_left: str = ""
    side_from_prompt: bool = False

    def __post_init__(self) -> None:
        if self.side is not None and self.side not in SIDES:
            raise ValueError(f"side must be one of {SIDES} or None, got {self.side!r}")
        if self.wrist_target not in WRIST_TARGETS:
            raise ValueError(f"wrist_target must be one of {WRIST_TARGETS}, got {self.wrist_target!r}")
        if self.reference_frequency is not None and not self.reference_frequency > 0.0:
            raise ValueError(f"reference_frequency must be positive, got {self.reference_frequency}")
        if not isinstance(self.side_from_prompt, bool):
            raise ValueError(f"side_from_prompt must be a bool, got {self.side_from_prompt!r}")  # noqa: TRY004
        if self.side_from_prompt and self.side is not None:
            raise ValueError(f"side_from_prompt needs side null (a configured side {self.side!r} always wins)")

    def scene_xml_for(self, side: str) -> str:
        """The configured robot scene of `side`; ValueError for a left hand without `scene_xml_left`."""
        path = {"right": self.scene_xml, "left": self.scene_xml_left}[side]
        if not path:
            raise ValueError(f"no robot scene configured for the {side} hand (retarget_stage.scene_xml_left): a {side} "
                             "hand is never solved on the other side's robot")
        return path

    def mano_pkl_path(self, side: str) -> str:
        """The configured MANO pkl of `side`; ValueError when that side has none."""
        path = {"right": self.mano_right_pkl, "left": self.mano_left_pkl}[side]
        if not path:
            raise ValueError(f"no MANO pkl configured for the {side} hand (retarget_stage.mano_pkl_path.{side})")
        return path

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any]) -> RetargetStageConfig:
        """From the `retarget_stage` config block (a dict or an OmegaConf node)."""
        paths = cfg.get("mano_pkl_path") or {}
        frequency = cfg.get("reference_frequency")
        return cls(scene_xml=str(cfg["scene_xml"]), mano_right_pkl=str(paths.get("right") or ""),
                   mano_left_pkl=str(paths.get("left") or ""), side=cfg.get("side"),
                   wrist_target=str(cfg.get("wrist_target") or DEFAULT_WRIST_TARGET),
                   reference_frequency=None if frequency is None else float(frequency),
                   retarget=RetargetConfig(**dict(cfg.get("retarget") or {})),
                   scene_xml_left=str(cfg.get("scene_xml_left") or ""),
                   side_from_prompt=cfg.get("side_from_prompt", False))


def interacting_side(artifact_dir: Path) -> tuple[str | None, str]:
    """`(side, source)` of the clip's interacting hand: `prompt.json`'s `choice.side` (`prompt`), else the single side
    of `geometry.npz` `hand_sides` (`geometry`), else `(None, "rule")`.

    Raises:
        ValueError: If `prompt.json` names a side outside `SIDES`.
    """
    prompt = Path(artifact_dir) / PROMPT_META_FILE
    if prompt.is_file():
        side = (json.loads(prompt.read_text()).get("choice") or {}).get("side")
        if side is not None:
            if side not in SIDES:
                raise ValueError(f"{PROMPT_META_FILE} choice.side must be one of {SIDES}, got {side!r}")
            return str(side), "prompt"
    geometry = Path(artifact_dir) / GEOMETRY_FILE
    if geometry.is_file():
        with np.load(geometry, allow_pickle=False) as data:
            sides = [str(x) for x in np.atleast_1d(data["hand_sides"])] if "hand_sides" in data.files else []
        if len(sides) == 1 and sides[0] in SIDES:
            return sides[0], "geometry"
    return None, "rule"


def source_frequency(record: ClipRecord) -> float:
    """Source frames per second from `record.frame_idx` and `record.timestamps` (first to last row).

    Raises:
        ValueError: With fewer than 2 rows, or a non-positive rate.
    """
    if record.n_frames < 2:
        raise ValueError("a one-row clip has no source rate; set retarget_stage.reference_frequency")
    rate = float(record.frame_idx[-1] - record.frame_idx[0]) / float(record.timestamps[-1] - record.timestamps[0])
    if not (np.isfinite(rate) and rate > 0.0):
        raise ValueError(f"clip {record.clip_id!r}: source rate must be positive, got {rate}")
    return rate


def _stage_frequency(cfg: RetargetStageConfig, record: ClipRecord) -> tuple[float, float | None]:
    """`(frequency, source rate or None for a one-row clip)`; warns when a configured label is off the source rate."""
    if cfg.reference_frequency is None:
        rate = source_frequency(record)
        return rate, rate
    label = cfg.reference_frequency
    source = source_frequency(record) if record.n_frames >= 2 else None
    if source is not None and abs(label / source - 1.0) > FREQUENCY_REL_TOL:
        logger.warning("clip %r: retarget_stage.reference_frequency %.4g Hz is not the source rate %.4g Hz. It relabels "
                       "the one-frame-per-source-frame grid without resampling, so qvel and playback speed scale by "
                       "%.4gx", record.clip_id, label, source, label / source)
    return label, source


def _hand_vertices(hands: Mapping[str, np.ndarray], side: str, valid: np.ndarray, n_labels: int) -> Tensor:
    """`hands.npz` `{side}_vertices` as a tensor, after the contract checks of the module docstring (ValueError)."""
    key = f"{side}_vertices"
    if key not in hands:
        raise ValueError(f"{HANDS_FILE} has no {key!r}")
    verts = np.asarray(hands[key])
    expected = (int(valid.shape[0]), n_labels, 3)
    if verts.shape != expected:
        raise ValueError(f"{HANDS_FILE} {key} must be {expected} (the record's rows and one finger label per "
                         f"vertex), got {verts.shape}")
    bad_rows = np.flatnonzero(valid & ~np.isfinite(verts).all(axis=(1, 2)))
    if bad_rows.size:
        raise ValueError(f"{HANDS_FILE} {key} is non-finite on valid rows {bad_rows[:6].tolist()}")
    return torch.from_numpy(verts)


def _require_finite(trajectory: KinematicTrajectory) -> None:
    """Raises FloatingPointError when the solved reference's `qpos` or `qvel` has a non-finite entry."""
    for name in ("qpos", "qvel"):
        values = np.asarray(getattr(trajectory, name))
        n_bad = int(values.size - np.isfinite(values).sum())
        if n_bad:
            raise FloatingPointError(f"kinematic reference {name} has {n_bad} non-finite entries")


def make_retarget_worker(cfg: RetargetStageConfig, scene: RobotScene | None = None,
                         stage_fn: StageRunner = solve_retarget_stage,
                         label_fn: Callable[[str], Tensor] | None = None, *, driftqc_enabled: bool = False) -> StageFn:
    """The `retarget` stage worker (module docstring); nothing loads until the first accepted clip.

    Args:
        cfg: Stage settings.
        scene: An already-built scene instead of `RobotScene.from_xml(cfg.scene_xml)`.
        stage_fn: `solve_retarget_stage`-compatible callable, the only step that can set `retarget_fallback_needed`
            (tests inject fakes).
        label_fn: MANO pkl path -> `(778,)` finger labels; None uses `mano_finger_labels`.
        driftqc_enabled: `driftqc_stage.enabled` (default False: never slices; leftover driftqc artifacts raise).
    """
    contexts: dict[str, WorkerContext] = {}
    labels: dict[str, Tensor] = {}

    def robot_scene(side: str) -> RobotScene:
        if side not in contexts:
            path = cfg.scene_xml_for(side) if scene is None else ""
            contexts[side] = WorkerContext(loader=lambda: RobotScene.from_xml(path), model=scene)
        return contexts[side].model()

    def hand_labels(side: str) -> Tensor:
        if side not in labels:
            path = cfg.mano_pkl_path(side)
            labels[side] = label_fn(path) if label_fn is not None else mano_finger_labels(path)
        return labels[side]

    def retarget_worker(clip_id: str, artifact_dir: Path) -> None:
        artifact_dir = Path(artifact_dir)
        trajectory_path = artifact_dir / TRAJECTORY_FILE
        report: dict[str, Any] = {"clip_id": clip_id, "retarget_fallback_needed": False}

        def write_report(status: str, **fields: Any) -> None:
            if status != "done":
                trajectory_path.unlink(missing_ok=True)
            write_json(artifact_dir / RETARGET_FILE, {**report, "status": status, **fields})

        def fail(exc: Exception, fallback: bool, **fields: Any) -> None:
            write_report("failed", retarget_fallback_needed=fallback, error=f"{type(exc).__name__}: {exc}",
                         error_type=type(exc).__name__, **fields)

        def skip(reason: str) -> None:
            write_report("skipped", reason=reason)
            logger.info("retarget of %r skipped: %s", clip_id, reason)

        try:
            qc = json.loads(require_artifact(artifact_dir / QC_FILE, "qc").read_text())
            if not qc.get("accepted", False):
                skip("rejected by qc")
                return
            record = ClipRecord.load(require_artifact(artifact_dir / CLIP_RECORD_FILE, "qc"))
            requested, side_source = cfg.side, "config" if cfg.side is not None else "rule"
            if cfg.side_from_prompt:
                requested, side_source = interacting_side(artifact_dir)
            side, track = retarget_hand_side(record, requested)
            report["side_source"] = side_source
            if track is None or not bool(track.valid.any()):
                skip(f"no valid {side} hand row (side from {side_source})")
                return
            hands = load_npz(require_artifact(artifact_dir / HANDS_FILE, "hands"))
            rng = resolve_valid_range(artifact_dir, read_frame_idx(artifact_dir), driftqc_enabled)
            if rng is not None:
                hands = slice_artifact(hands, HANDS_FILE, rng)
            check_decode_rows(hands["frame_idx"], record.frame_idx, HANDS_FILE)
            width, height = decode_frame_size(artifact_dir)
            require_hands_camera(hands, read_intrinsics(artifact_dir)[0], width, height, HANDS_FILE)
            keys = reference_key_frames(record.frame_idx)
            row_valid = retarget_row_valid(record, track.valid)      # hand valid AND object observed
            frame_valid = reference_frame_valid(keys, row_valid)
            frequency, source = _stage_frequency(cfg, record)
            report |= {"side": side, "side_source": side_source, "wrist_target": cfg.wrist_target, "n_rows": record.n_frames,
                       "n_reference_frames": int(keys[-1]) + 1, "masked_rows": int((~row_valid).sum()),
                       "masked_frames": int((~frame_valid).sum()), "frequency": frequency, "source_frequency": source}
            labels_side = hand_labels(side)
            verts = _hand_vertices(hands, side, np.asarray(track.valid, dtype=bool), int(labels_side.shape[0]))
            inputs = prepare_retarget_stage(record, verts, side=side, wrist_target=cfg.wrist_target)
            if inputs is None:
                skip(retarget_skip_reason(record, side) or "prepare_retarget_stage returned no inputs")
                return
            samples, sdf = read_mesh_samples(artifact_dir), read_sdf_grid(artifact_dir)
            robot = robot_scene(side)
        except Exception as exc:
            fail(exc, fallback=False)
            logger.exception("retarget inputs of %r are unusable; failed without retarget_fallback_needed", clip_id)
            raise
        t0 = time.perf_counter()
        try:
            trajectory = stage_fn(record, inputs, labels_side, robot, sdf, samples.points.to(torch.float64),
                                  samples.normals.to(torch.float64), frequency, cfg.retarget)
            _require_finite(trajectory)
        except Exception as exc:
            fail(exc, fallback=True, seconds=time.perf_counter() - t0)
            logger.exception("kinematic retargeting failed for %r; marked retarget_fallback_needed", clip_id)
            raise
        seconds = time.perf_counter() - t0
        tmp = artifact_dir / "trajectory_kinematic.tmp.npz"
        trajectory.save(tmp)
        os.replace(tmp, trajectory_path)
        write_report("done", trajectory=TRAJECTORY_FILE, seconds=seconds)
        logger.info("retarget of %r: %d reference frames at %.4g Hz in %.2f s (masked %d rows, %d frames)", clip_id,
                    report["n_reference_frames"], frequency, seconds, report["masked_rows"], report["masked_frames"])

    return retarget_worker
