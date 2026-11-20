"""The `contact` stage worker: the hand-contact along-ray correction, applied after `solve`.

Why a separate stage rather than a change inside `solve`: the correction uses the HANDS, which the solver does not
see, and keeping it separate leaves `solve.npz` exactly as the solver wrote it, so the uncorrected trajectory stays
on disk as provenance and the correction can be audited or switched off without re-solving.

Inputs (all already written by earlier stages):
- `solve.npz`: `poses` (T,4,4), `scale`, `frame_idx` (T,);
- `hands.npz`: `{side}_joints` (T,21,3) in the intrinsics camera and `{side}_valid` (T,), both sides;
- `samples.npz`: the shape stage's `points` (M,3), the object surface in its own frame.

Output `contact.npz` (uncompressed, tmp file then `os.replace`):
- `poses` (T,4,4) float64 corrected, `shift_mm` (T,) float64 (0 where no correction), `frame_idx` (T,) int64;
- `side` 0-d str (the hand used, or "" when no frame had one), `n_corrected` 0-d int64;
- `solve_sha256` 0-d str: the `solve.npz` this was computed from, so a STALE artifact is detectable rather than
  silently consumed downstream;
- `constants` 0-d str: the JSON of the constants actually applied.
- only when a ruler option is set (`bias_free` or `touch_gate_mm`, both off by default, so the default artifact is
  byte-for-byte the default ruler's): `min_cost_mm` (T,) float64 (the profile minimum, NaN where none was computed),
  `gated` (T,) bool (frames the touch gate left uncorrected), `touch_gate_mm` 0-d float64 (NaN = no gate) and
  `bias_free` 0-d bool.

The hand side is chosen per clip by the smaller median joint-to-surface distance at the solved pose
(`hands: auto`). Measured on all 87 GT-labelled clips, that recovers the known MANO side **87/87**, with per-frame
agreement median 1.000, so `auto` inherits the default ruler's held-out validation. `left` or
`right` forces a side.

A clip with no valid hand on any frame is NOT a failure: the stage writes zero shifts and `side` "", and the
trajectory is the solver's. That keeps the correction from turning a hand-detection gap into a lost clip.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from fastdo.contact import ContactConstants, contact_correction, load_constants
from fastdo.engine.orchestrator import StageFn
from fastdo.engine.source_domain import refuse_declared_egocentric
from fastdo.engine.worker_support import (
    SOLVE_FILE,
    WorkerContext,
    load_npz,
    require_artifact,
    save_npz,
)
from fastdo.engine.workers_hands import HANDS_FILE
from fastdo.engine.workers_shape import SAMPLES_FILE

__all__ = ["CONTACT_FILE", "ContactStageConfig", "choose_side", "make_contact_worker",
           "read_contact_artifact", "with_contact"]

logger = logging.getLogger(__name__)

CONTACT_FILE = "contact.npz"
SIDES: tuple[str, ...] = ("left", "right")
DEFAULT_CONSTANTS = Path(__file__).resolve().parents[1] / "contact/constants_production.json"


@dataclasses.dataclass(frozen=True)
class ContactStageConfig:
    """`contact_stage` configuration.

    Attributes:
        enabled: Whether the engine registers the stage at all.
        cell: Which calibration the constants file supplies, `"prod"` (the shipped hand-anchored depth) or
            `"oraclescale"` (a diagnostic level; the correction gains almost nothing there).
        hands: `"auto"`, `"left"` or `"right"`.
        constants: Path to a constants JSON, or None/"" for the package's production file (the config's `null`).
        hands_source: Which calibration row to use, `"hawor"` (production) or `"gt"`.
        grid_mm: `(lo, hi, step)` of the slide grid; the default design is (-300, 300, 2).
        bias_free: Opt-in: zero the constants' intercept and prior mean (`ContactConstants.bias_free`),
            which encode DexYCB's pre-contact solve bias. Off = the default ruler.
        touch_gate_mm: Opt-in: correct only frames whose minimum contact cost is <= this many mm
            (`fastdo.contact.contact_correction`). None = every frame with a valid hand.
        geometric: Opt-in: the geometric ruler (`ContactConstants.geometric`: a = 1, b = mu = 0; sig and
            tau from the constants file). It implies `bias_free`. Off = the default ruler.
    """

    enabled: bool = False
    cell: str = "prod"
    hands: str = "auto"
    constants: str | None = None      # None means DEFAULT_CONSTANTS; the Hydra config writes `null` for that
    hands_source: str = "hawor"
    grid_mm: tuple[float, float, float] = (-300.0, 300.0, 2.0)
    bias_free: bool = False
    touch_gate_mm: float | None = None
    geometric: bool = False

    def __post_init__(self) -> None:
        if self.cell not in ("prod", "oraclescale"):
            raise ValueError(f"contact_stage.cell must be prod or oraclescale, got {self.cell!r}")
        if self.hands not in ("auto", *SIDES):
            raise ValueError(f"contact_stage.hands must be auto, left or right, got {self.hands!r}")
        if self.hands_source not in ("hawor", "gt"):
            raise ValueError(f"contact_stage.hands_source must be hawor or gt, got {self.hands_source!r}")
        lo, hi, step = self.grid_mm
        if not (lo < hi and step > 0):
            raise ValueError(f"contact_stage.grid_mm must be (lo < hi, step > 0), got {self.grid_mm}")
        if not isinstance(self.bias_free, bool):     # ValueError like every other config check here
            raise ValueError(f"contact_stage.bias_free must be a bool, got {self.bias_free!r}")  # noqa: TRY004
        if not isinstance(self.geometric, bool):
            raise ValueError(f"contact_stage.geometric must be a bool, got {self.geometric!r}")  # noqa: TRY004
        gate = self.touch_gate_mm
        if gate is not None and not (np.isfinite(gate) and gate > 0):
            raise ValueError(f"contact_stage.touch_gate_mm must be null or a positive number, got {gate!r}")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object] | None) -> ContactStageConfig:
        """Builds from a config mapping; an unknown key raises rather than being ignored."""
        if not mapping:
            return cls()
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted({str(k) for k in mapping} - known)
        if unknown:
            raise ValueError(f"contact_stage has unknown keys {unknown}; known keys: {sorted(known)}")
        data = dict(mapping)
        if "grid_mm" in data:
            data["grid_mm"] = tuple(float(x) for x in data["grid_mm"])  # type: ignore[arg-type]
        if data.get("touch_gate_mm") is not None:
            data["touch_gate_mm"] = float(data["touch_gate_mm"])  # type: ignore[arg-type]
        return cls(**data)  # type: ignore[arg-type]

    def grid(self) -> np.ndarray:
        lo, hi, step = self.grid_mm
        return np.arange(lo, hi + step / 2, step, dtype=float)

    def constants_path(self) -> Path:
        """The constants file to read: `constants`, or the package's production file when it is null/empty."""
        return Path(self.constants) if self.constants else DEFAULT_CONSTANTS

    def load(self) -> ContactConstants:
        constants = load_constants(self.constants_path(), self.cell, self.hands_source)
        if self.geometric:
            return constants.geometric()
        return constants.bias_free() if self.bias_free else constants

    def is_m88_default_ruler(self) -> bool:
        """True when no ruler option is set, so the artifact must stay the default ruler's byte for byte."""
        return not self.bias_free and self.touch_gate_mm is None and not self.geometric


def choose_side(surface_by_frame, joints: dict[str, np.ndarray], valid: dict[str, np.ndarray]) -> str:
    """The side whose median joint-to-surface distance is smaller over the frames where it is valid.

    Args:
        surface_by_frame: Callable `t -> (M,3)` object surface in the camera frame at frame `t`.
        joints: `side -> (T,J,3)` joints in the camera frame.
        valid: `side -> (T,)` bool.

    Returns:
        `"left"`, `"right"`, or `""` when no side has a usable frame.
    """
    med: dict[str, float] = {}
    for side in SIDES:
        d = []
        for t in np.flatnonzero(np.asarray(valid[side], dtype=bool)):
            j = np.asarray(joints[side], dtype=float)[t]
            j = j[np.isfinite(j).all(axis=1)]
            if j.size:
                d.append(float(np.median(cKDTree(surface_by_frame(int(t))).query(j, k=1)[0])))
        if d:
            med[side] = float(np.median(d))
    if not med:
        return ""
    return min(med, key=med.__getitem__)


def make_contact_worker(cfg: ContactStageConfig, *, allow_egocentric: bool = False) -> StageFn:
    """The `contact` stage: `contact.npz` from `solve.npz`, `hands.npz` and `samples.npz`.

    A clip whose `source_domain.json` declares it EGOCENTRIC is refused (`DeclaredEgocentric`, nothing written)
    unless `allow_egocentric`, which only the source profile's study override sets
    (`fastdo.engine.source_profile.make_profiled_contact_worker`): the egocentric profile keeps contact OFF.
    """
    constants = cfg.load()
    grid = cfg.grid()

    def contact_worker(clip_id: str, artifact_dir: Path) -> None:
        ctx.budget()
        artifact_dir = Path(artifact_dir)
        if not allow_egocentric:
            refuse_declared_egocentric(artifact_dir, clip_id, "the contact stage (along-ray correction)")
        solve_path = require_artifact(artifact_dir / SOLVE_FILE, "solve")
        solve = load_npz(solve_path)
        poses, scale = np.asarray(solve["poses"], dtype=float), float(solve["scale"])
        frame_idx = np.asarray(solve["frame_idx"], dtype=np.int64)
        mesh = np.asarray(load_npz(require_artifact(artifact_dir / SAMPLES_FILE, "shape"))["points"], dtype=float)
        hands = load_npz(require_artifact(artifact_dir / HANDS_FILE, "hands"))
        joints = {s: np.asarray(hands[f"{s}_joints"], dtype=float) for s in SIDES}
        valid = {s: np.asarray(hands[f"{s}_valid"]).astype(bool) for s in SIDES}
        for s in SIDES:
            if joints[s].shape[0] != poses.shape[0] or valid[s].shape[0] != poses.shape[0]:
                raise ValueError(f"{clip_id}: hands.npz {s} has {joints[s].shape[0]} rows, solve.npz has "
                                 f"{poses.shape[0]}")

        centroid = mesh.mean(axis=0)

        def surface_at(t: int) -> np.ndarray:
            return scale * (mesh @ poses[t, :3, :3].T) + poses[t, :3, 3]

        side = cfg.hands if cfg.hands in SIDES else choose_side(surface_at, joints, valid)
        n = poses.shape[0]
        if side:
            res = contact_correction(poses, scale, mesh, joints[side], valid[side], constants, grid,
                                     touch_gate_mm=cfg.touch_gate_mm)
            out, shift, min_cost, gated = res.poses, res.shift_mm, res.min_cost_mm, res.gated
        else:
            out, shift = poses.copy(), np.zeros(n, dtype=float)
            min_cost, gated = np.full(n, np.nan), np.zeros(n, dtype=bool)
        extra: dict[str, np.ndarray] = {}
        if not cfg.is_m88_default_ruler():
            extra = {"min_cost_mm": min_cost, "gated": gated, "bias_free": np.array(cfg.bias_free),
                     "touch_gate_mm": np.array(np.nan if cfg.touch_gate_mm is None else cfg.touch_gate_mm)}
            if cfg.geometric:
                extra["geometric"] = np.array(True)
        save_npz(artifact_dir / CONTACT_FILE, poses=out, shift_mm=shift, frame_idx=frame_idx,
                 side=np.array(side), n_corrected=np.array(int((shift != 0).sum()), dtype=np.int64),
                 solve_sha256=np.array(_sha256(solve_path)),
                 constants=np.array(json.dumps(dataclasses.asdict(constants), sort_keys=True)),
                 centroid_obj=centroid, **extra)
        logger.info("contact %r: side %r corrected %d/%d frames, median shift %.1f mm", clip_id, side or "none",
                    int((shift != 0).sum()), shift.size,
                    float(np.median(shift[shift != 0])) if (shift != 0).any() else 0.0)

    ctx = WorkerContext()
    return contact_worker


def read_contact_artifact(artifact_dir: Path) -> dict[str, np.ndarray] | None:
    """`contact.npz` if it exists AND was computed from the current `solve.npz`; otherwise None.

    A stale artifact (a rerun solve, a copied clip directory) must never be consumed silently, so the check is the
    solve file's own hash, not its timestamp.
    """
    artifact_dir = Path(artifact_dir)
    path = artifact_dir / CONTACT_FILE
    if not path.is_file():
        return None
    data = load_npz(path)
    solve_path = artifact_dir / SOLVE_FILE
    if not solve_path.is_file():
        return None
    if str(data["solve_sha256"]) != _sha256(solve_path):
        logger.warning("%s is stale (its solve_sha256 does not match %s); ignoring it", path, solve_path)
        return None
    return {k: data[k] for k in data}


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def with_contact(stages: tuple[str, ...]) -> tuple[str, ...]:
    """`stages` with "contact" inserted immediately after "solve" (idempotent).

    Raises:
        ValueError: If "solve" is absent, "qc" precedes "solve", or "contact" already sits elsewhere.
    """
    stages = tuple(stages)
    if "solve" not in stages:
        raise ValueError(f"with_contact needs a 'solve' stage, got {stages}")
    at = stages.index("solve")
    if "qc" in stages and stages.index("qc") < at:
        raise ValueError(f"'qc' precedes 'solve' in {stages}")
    if "contact" in stages:
        if stages.count("contact") == 1 and stages.index("contact") == at + 1:
            return stages
        raise ValueError(f"'contact' must follow 'solve' immediately (once), got {stages}")
    return stages[:at + 1] + ("contact",) + stages[at + 1:]
