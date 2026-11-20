"""The solve stage's configuration: `SolveStageConfig` and its registration and init-gate blocks.

Every option is a named argument of a merged function (`register_keyframe_pose`, `anchors_from_pose`, `pnp_initialize`,
`gate_init_poses`) or a merged `SolverConfig` field. `fix_scale` accepts only false. Solver features
that are not merged (the calibrated WNOA prior, the silhouette term, multi-keyframe anchoring, hand-relative anchoring)
have no key. `fastdo.engine.workers_solve` re-exports every name here.

`depth_anchor_gate` (default None = today's solve stage, byte for byte): the certificate-gated re-solve with the depth
anchor, `DepthAnchorGateConfig` (its mechanism: `fastdo.engine.depth_anchor_gate`).

`per_frame_reanchor` (default None = no re-anchor, byte for byte): the per-frame re-anchor "G" after the
production solve and the gate, `PerFrameReanchorConfig` (its mechanism: `fastdo.engine.per_frame_reanchor`).

`branch_select` (default None = today's stage, byte for byte): branch-robust keyframe registration, the
production solve from each of the registration's top distinct hypotheses and a label-free choice among them,
`BranchSelectConfig` (its mechanism: `fastdo.engine.branch_select`).

`per_frame_reanchor_impl` ("reference" by default = today's code): how G computes its per-frame observations,
"reference" or "vectorized" (`fastdo.engine.per_frame_reanchor.REANCHOR_IMPLS`); both give the same bytes, and neither
solve.json nor solve.npz records the choice. Only read when `per_frame_reanchor` is set.

`fixed_scale` (default None = today's stage, byte for byte): the fixed-scale re-solve, `FixedScaleConfig`. The window's
scale is read from a per-window JSON file that the instance-size fusion post-pass writes
(`fastdo.engine.instance_scale`); the registration starts at it and the LM holds it (`SolverConfig.fix_scale`). The
top-level `fix_scale` (hold the REGISTRATION's own scale) stays refused.

`contact_ray_smoother` (default None = today's stage, byte for byte): the contact-gated along-ray log-depth
smoother applied in `finish_solve` after G, `ContactRaySmootherConfig` (its mechanism:
`fastdo.engine.contact_ray_smoother`).

`grasp_ray_constant` (default None = today's stage, byte for byte): ONE along-ray log factor per clip from the
grasp-onset contact rows, applied in `finish_solve` at the smoother's place, `GraspRayConstantConfig` (its mechanism:
`fastdo.engine.grasp_ray_constant`). At most one of the two may be set.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from fastdo.engine.contact_ray_smoother import ContactRaySmootherConfig, smoother_from_mapping
from fastdo.engine.grasp_ray_constant import GraspRayConstantConfig, grasp_constant_from_mapping
from fastdo.solver.api import SolverConfig

__all__ = ["BRANCH_SELECTORS", "REANCHOR_IMPLS", "SCALE_SOURCES", "BranchSelectConfig", "ContactRaySmootherConfig",
           "DepthAnchorGateConfig", "FixedScaleConfig", "GraspRayConstantConfig",
           "InitGateConfig", "PerFrameReanchorConfig", "RegistrationConfig", "SolveStageConfig"]

SCALE_SOURCES: tuple[str, ...] = ("auto", "metric_template", "extent", "lateral")
REANCHOR_IMPLS: tuple[str, ...] = ("reference", "vectorized")   # fastdo.engine.per_frame_reanchor re-exports it
BRANCH_SELECTORS: tuple[str, ...] = ("reg_score", "mf_score", "sil_iou", "sil_cov", "medoid")   # fastdo.engine.branch_select
_TOP_LEVEL_KEYS = ("device", "seed", "fix_scale")
_SOLVER_FIELDS = tuple(f.name for f in dataclasses.fields(SolverConfig) if f.name not in ("device", "seed", "fix_scale"))
_FIX_SCALE_REFUSAL = (
    "solve_stage.fix_scale=true is refused: whether to hold the REGISTRATION's own scale for metric templates is "
    "undecided, and no verdict wires it. A per-window scale supplied from outside (the fused instance size of "
    "fastdo.engine.instance_scale) is solve_stage.fixed_scale.")


@dataclass(frozen=True)
class RegistrationConfig:
    """`register_keyframe_pose` arguments, and where its initial scale comes from (`registration_init_scale`)."""

    scale_source: str = "auto"                       # auto | metric_template | extent | lateral
    estimate_scale: bool = False                     # rigid
    scale_range: tuple[float, float] = (0.5, 2.0)
    n_obs: int = 512
    n_mesh: int = 1024
    n_iters: int = 50


@dataclass(frozen=True)
class InitGateConfig:
    """`gate_init_poses` arguments."""

    min_inliers: int = 6
    max_offset: float = 1.0
    min_points: int = 20


@dataclass(frozen=True)
class DepthAnchorGateConfig:
    """Certificate-gated depth anchor.

    After the production solve, the clip's certificate c = |median_t d_t| (d_t = log z_centroid(t) - log z_obs(t), the
    depth-anchor term's own quantities, `fastdo.engine.depth_anchor_gate.depth_certificate`) is computed; c >=
    `threshold` re-solves the clip from the SAME initialization with `SolverConfig.lam_depth_anchor = lam` and
    `depth_anchor_band = band` and keeps the re-solve. The defaults: threshold 0.25 (good lab clips have |median d| <=
    0.065, the offset tail 0.36-0.74), lam 500, band 0.2 (the depth anchor's default band).

    Raises:
        ValueError: Unless `threshold`, `lam` and `band` are finite and > 0.
    """

    threshold: float = 0.25
    lam: float = 500.0
    band: float = 0.2

    def __post_init__(self) -> None:
        for name in ("threshold", "lam", "band"):
            value = float(getattr(self, name))
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"solve_stage.depth_anchor_gate.{name} must be finite and > 0, got "
                                 f"{getattr(self, name)!r}")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class PerFrameReanchorConfig:
    """Per-frame re-anchor "G".

    A clip whose production certificate |median_t d_t| >= `threshold` has every frame with |d_t| > `band` or
    a mask-centroid departure q_t > `perp_k` mask radii placed at its pointmap depth on the mask-centroid ray; any
    other clip has only its dead frames (< `vis_min` visible tracks) pulled to the band edge and onto the mask ray
    (`fastdo.engine.per_frame_reanchor`).

    Raises:
        ValueError: Unless `threshold`, `band` and `perp_k` are finite and > 0 and `vis_min` is an integer >= 1.
    """

    threshold: float = 0.25
    band: float = 0.2
    perp_k: float = 1.0
    vis_min: int = 5
    # safety options (None = G without them, byte for byte; optional keys of the config block):
    #   occluded_area_frac  a frame whose mask area is below this fraction of the clip's median mask area (an occluded
    #                       or truncated object) may only be moved TOWARD the camera, never away (its masked pointmap
    #                       bleeds onto the background behind the object, so its depth is an upper bound)
    #   max_log_shift       cap on any frame's along-ray move |log z_new / z_solved| (the point slides along its target
    #                       ray back to the cap): bounds what one frame's observation can do
    occluded_area_frac: float | None = None
    max_log_shift: float | None = None
    # dead-frame option (None = the branch as above, byte for byte; an optional key of the config block):
    #   dead_frame_branch   False turns the UNFLAGGED (dead-frame) branch off: a clip whose certificate is below
    #                       `threshold` keeps its solve; flagged clips are re-anchored as before. True = on (as None,
    #                       but recorded in solve.json)
    dead_frame_branch: bool | None = None

    def __post_init__(self) -> None:
        for name in ("threshold", "band", "perp_k"):
            value = float(getattr(self, name))
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"solve_stage.per_frame_reanchor.{name} must be finite and > 0, got "
                                 f"{getattr(self, name)!r}")
            object.__setattr__(self, name, value)
        vis_min = self.vis_min
        if isinstance(vis_min, bool) or int(vis_min) != vis_min or int(vis_min) < 1:
            raise ValueError(f"solve_stage.per_frame_reanchor.vis_min must be an integer >= 1, got {vis_min!r}")
        object.__setattr__(self, "vis_min", int(vis_min))
        for name, upper in (("occluded_area_frac", 1.0), ("max_log_shift", math.inf)):
            raw = getattr(self, name)
            if raw is None:
                continue
            value = float(raw) if not isinstance(raw, bool) else math.nan
            if not (math.isfinite(value) and 0.0 < value <= upper):
                raise ValueError(f"solve_stage.per_frame_reanchor.{name} must be null or finite in (0, {upper}], "
                                 f"got {raw!r}")
            object.__setattr__(self, name, value)
        if self.dead_frame_branch is not None and not isinstance(self.dead_frame_branch, bool):
            raise ValueError("solve_stage.per_frame_reanchor.dead_frame_branch must be null, true or false, got "
                             f"{self.dead_frame_branch!r}")

    def safety_options(self) -> dict[str, float | bool]:
        """The safety options that are set (empty for the plain G, whose solve.json block then stays as it was)."""
        return {name: getattr(self, name) for name in SAFETY_OPTIONS if getattr(self, name) is not None}


SAFETY_OPTIONS = ("occluded_area_frac", "max_log_shift",   # PerFrameReanchorConfig's optional keys
                  "dead_frame_branch")


@dataclass(frozen=True)
class FixedScaleConfig:
    """Fixed-scale re-solve at a per-window scale read from a file.

    `prepare_solve` reads `<artifact_dir>/<file>` (`fastdo.engine.instance_scale.read_fixed_scale`: a JSON object
    whose `scale` is finite and > 0), registers the keyframe at that scale (source "fixed") and solves with
    `SolverConfig.fix_scale`, so the written `solve.npz` scale is the file's. A missing file is an error.

    Raises:
        ValueError: Unless `file` is a non-empty plain file name (no directory part).
    """

    file: str = "instance_scale.json"

    def __post_init__(self) -> None:
        name = str(self.file)
        if not name or Path(name).name != name or name in (".", ".."):
            raise ValueError(f"solve_stage.fixed_scale.file must be a plain file name, got {self.file!r}")
        object.__setattr__(self, "file", name)


@dataclass(frozen=True)
class BranchSelectConfig:
    """Branch-robust registration (the keyframe registration's rotation-hypothesis choice is a knife edge).

    The registration is run at every (seed + o, init scale x m) for o in `seed_offsets`, m in `scale_mults` (the
    production run (0, 1.0) first); each run keeps its `top_k` best DISTINCT hypotheses
    (`keyframe.register_keyframe_hypotheses`, `min_rot_deg` / `min_shift`). Every hypothesis gets the production
    anchors, PnP, init gate, solve and depth-anchor gate; `selector` (one of `BRANCH_SELECTORS`, lower is better, ties
    keep the earlier candidate, so the production one) picks the solution that is kept (`fastdo.engine.branch_select`),
    with a production-anchored `margin`: another candidate is kept only if its value is below (1 - margin) x the
    production candidate's (selector values are >= 0; margin 0 = the plain argmin). One run with `top_k` 1 is today's
    stage
    exactly.

    Raises:
        ValueError: On `top_k < 1`, `seed_offsets` without 0 or with repeats, `scale_mults` without 1.0, with repeats
            or a value <= 0, an unknown `selector`, `min_rot_deg` / `min_shift` not finite and > 0, or `margin` not in
            [0, 1).
    """

    top_k: int = 1
    seed_offsets: tuple[int, ...] = (0,)
    scale_mults: tuple[float, ...] = (1.0,)
    selector: str = "reg_score"
    min_rot_deg: float = 10.0
    min_shift: float = 0.005
    margin: float = 0.0

    def __post_init__(self) -> None:
        margin = float(self.margin)
        if not (math.isfinite(margin) and 0.0 <= margin < 1.0):
            raise ValueError(f"solve_stage.branch_select.margin must be in [0, 1), got {self.margin!r}")
        object.__setattr__(self, "margin", margin)
        top_k = self.top_k
        if isinstance(top_k, bool) or int(top_k) != top_k or int(top_k) < 1:
            raise ValueError(f"solve_stage.branch_select.top_k must be an integer >= 1, got {top_k!r}")
        object.__setattr__(self, "top_k", int(top_k))
        offsets = tuple(int(o) for o in self.seed_offsets)
        if 0 not in offsets or len(set(offsets)) != len(offsets):
            raise ValueError(f"solve_stage.branch_select.seed_offsets must hold 0 once and no repeats, got {offsets}")
        mults = tuple(float(m) for m in self.scale_mults)
        if 1.0 not in mults or len(set(mults)) != len(mults) or not all(math.isfinite(m) and m > 0 for m in mults):
            raise ValueError(f"solve_stage.branch_select.scale_mults must hold 1.0 once, no repeats, all > 0, got "
                             f"{mults}")
        if self.selector not in BRANCH_SELECTORS:
            raise ValueError(f"solve_stage.branch_select.selector must be one of {BRANCH_SELECTORS}, got "
                             f"{self.selector!r}")
        for name in ("min_rot_deg", "min_shift"):
            value = float(getattr(self, name))
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"solve_stage.branch_select.{name} must be finite and > 0, got {getattr(self, name)!r}")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "seed_offsets", offsets)
        object.__setattr__(self, "scale_mults", mults)

    def runs(self) -> tuple[tuple[int, float], ...]:
        """The registration runs (seed offset, scale multiplier), the production run (0, 1.0) first, then the rest in
        (seed_offsets, scale_mults) order."""
        rest = tuple((o, m) for o in self.seed_offsets for m in self.scale_mults if (o, m) != (0, 1.0))
        return ((0, 1.0), *rest)


@dataclass(frozen=True)
class SolveStageConfig:
    """The `solve_stage` config block and its defaults.

    `device` is used as given (no CPU fallback); `seed` seeds the registration, the dense anchor samples (`seed + 2`)
    and the solver. `solver` overrides `SolverConfig` fields other than `device` and `seed` (`solver_config`).

    Raises:
        ValueError: On `fix_scale` true; an unknown `registration.scale_source`; `registration.scale_range` not
            `0 < lo < hi`; `n_anchor_surface < 1`, `anchor_max_px <= 0` or `pnp_min_inliers < 1`; a `solver` key
            that is not a `SolverConfig` field (the message lists the valid ones); or a `solver` key `device`, `seed`
            or `fix_scale` (the message points to the top-level key); a `depth_anchor_gate` together with a
            `solver.lam_depth_anchor` > 0 (the gate's first solve must be the production solve, without the anchor);
            a `fixed_scale` together with `registration.estimate_scale` (the scale is given, not fitted).
    """

    device: str = "cuda"
    seed: int = 42
    fix_scale: bool = False
    registration: RegistrationConfig = field(default_factory=RegistrationConfig)
    n_anchor_surface: int = 16384
    anchor_max_px: float = 2.0
    pnp_min_inliers: int = 6
    gate: InitGateConfig = field(default_factory=InitGateConfig)
    solver: Mapping[str, Any] = field(default_factory=dict)
    depth_anchor_gate: DepthAnchorGateConfig | None = None    # None = no gate (today's stage)
    per_frame_reanchor: PerFrameReanchorConfig | None = None  # None = no re-anchor (today's stage)
    fixed_scale: FixedScaleConfig | None = None                # None = the solved scale (today's stage)
    branch_select: BranchSelectConfig | None = None            # None = one registration hypothesis (today's)
    per_frame_reanchor_impl: str = "reference"                 # G's observations; "vectorized" = same bytes
    contact_ray_smoother: ContactRaySmootherConfig | None = None  # None = no smoother (today's stage)
    grasp_ray_constant: GraspRayConstantConfig | None = None      # None = no constant (today's stage)

    def __post_init__(self) -> None:
        if self.fix_scale:
            raise ValueError(_FIX_SCALE_REFUSAL)
        if self.contact_ray_smoother is not None and self.grasp_ray_constant is not None:
            raise ValueError("solve_stage: set at most one of contact_ray_smoother and grasp_ray_constant; "
                             "both move the object along the ray")
        if self.per_frame_reanchor_impl not in REANCHOR_IMPLS:
            raise ValueError(f"solve_stage.per_frame_reanchor_impl must be one of {REANCHOR_IMPLS}, got "
                             f"{self.per_frame_reanchor_impl!r}")
        registration = self.registration
        if registration.scale_source not in SCALE_SOURCES:
            raise ValueError(f"registration.scale_source must be one of {SCALE_SOURCES}, got "
                             f"{registration.scale_source!r}")
        lo, hi = (float(bound) for bound in registration.scale_range)
        if not 0.0 < lo < hi:
            raise ValueError(f"registration.scale_range must satisfy 0 < lo < hi, got {tuple(registration.scale_range)}")
        if self.n_anchor_surface < 1 or not self.anchor_max_px > 0.0 or self.pnp_min_inliers < 1:
            raise ValueError(f"need n_anchor_surface >= 1, anchor_max_px > 0 and pnp_min_inliers >= 1, got "
                             f"{self.n_anchor_surface}, {self.anchor_max_px} and {self.pnp_min_inliers}")
        solver = {str(key): value for key, value in dict(self.solver).items()}
        for key in _TOP_LEVEL_KEYS:
            if key in solver:
                raise ValueError(f"solve_stage.solver.{key} is not a solver override: set the top-level "
                                 f"solve_stage.{key} instead")
        unknown = sorted(set(solver) - set(_SOLVER_FIELDS))
        if unknown:
            raise ValueError(f"solve_stage.solver keys {unknown} are not SolverConfig fields (solver features that are "
                             f"not merged have no key); valid fields: {list(_SOLVER_FIELDS)}")
        if self.depth_anchor_gate is not None and float(solver.get("lam_depth_anchor", 0.0)) > 0.0:
            raise ValueError("solve_stage.depth_anchor_gate needs the production first solve, but solve_stage.solver."
                             f"lam_depth_anchor={solver['lam_depth_anchor']!r} already builds the anchor into it: set "
                             "one or the other")
        if self.fixed_scale is not None and registration.estimate_scale:
            raise ValueError("solve_stage.fixed_scale gives the window's scale, so registration.estimate_scale must be "
                             "false (a similarity registration would move it before the solve holds it)")
        object.__setattr__(self, "solver", MappingProxyType(solver))

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> SolveStageConfig:
        """From the `solve_stage` block (a dict or an OmegaConf node); `scale_range` lists become tuples.

        Raises:
            ValueError: On an unknown key in any block, a `depth_anchor_gate` block that is not null or lacks one of
                threshold / lam / band, and on anything `__post_init__` rejects.
        """
        data = _block(mapping, cls, "solve_stage")
        registration = _block(data.pop("registration", None) or {}, RegistrationConfig, "solve_stage.registration")
        if "scale_range" in registration:
            registration["scale_range"] = tuple(float(bound) for bound in registration["scale_range"])
        gate = _block(data.pop("gate", None) or {}, InitGateConfig, "solve_stage.gate")
        solver = _block(data.pop("solver", None) or {}, None, "solve_stage.solver")
        depth_anchor_gate = _anchor_gate(data.pop("depth_anchor_gate", None))
        per_frame_reanchor = _reanchor(data.pop("per_frame_reanchor", None))
        fixed_scale = _fixed_scale(data.pop("fixed_scale", None))
        branch_select = _branch_select(data.pop("branch_select", None))
        smoother = smoother_from_mapping(data.pop("contact_ray_smoother", None))
        grasp = grasp_constant_from_mapping(data.pop("grasp_ray_constant", None))
        return cls(registration=RegistrationConfig(**registration), gate=InitGateConfig(**gate), solver=solver,
                   depth_anchor_gate=depth_anchor_gate, per_frame_reanchor=per_frame_reanchor,
                   fixed_scale=fixed_scale, branch_select=branch_select, contact_ray_smoother=smoother,
                   grasp_ray_constant=grasp, **data)

    def solver_config(self) -> SolverConfig:
        """`SolverConfig(device=self.device, seed=self.seed, **self.solver)`, with `fix_scale=True` only when
        `fixed_scale` is set (otherwise the field keeps its default and the config is today's)."""
        extra = {} if self.fixed_scale is None else {"fix_scale": True}
        return SolverConfig(device=self.device, seed=self.seed, **self.solver, **extra)


def _anchor_gate(block: Any) -> DepthAnchorGateConfig | None:
    """`solve_stage.depth_anchor_gate`: null (no gate), or a block spelling all of threshold, lam and band (so no
    gate constant is ever inherited silently)."""
    if block is None:
        return None
    if not isinstance(block, Mapping):     # a ValueError, as every other solve_stage refusal
        raise ValueError(  # noqa: TRY004
            f"solve_stage.depth_anchor_gate must be null or {{threshold, lam, band}}, got {block!r}")
    data = _block(block, DepthAnchorGateConfig, "solve_stage.depth_anchor_gate")
    missing = sorted({f.name for f in dataclasses.fields(DepthAnchorGateConfig)} - set(data))
    if missing:
        raise ValueError(f"solve_stage.depth_anchor_gate is missing {missing}: spell threshold, lam and band")
    return DepthAnchorGateConfig(**data)


def _reanchor(block: Any) -> PerFrameReanchorConfig | None:
    """`solve_stage.per_frame_reanchor`: null (off), or a block spelling all of threshold, band, perp_k and vis_min
    (so no constant is ever inherited silently), optionally with the safety options `occluded_area_frac` and
    `max_log_shift` (absent = null = the plain G)."""
    if block is None:
        return None
    if not isinstance(block, Mapping):     # a ValueError, as every other solve_stage refusal
        raise ValueError(  # noqa: TRY004
            f"solve_stage.per_frame_reanchor must be null or {{threshold, band, perp_k, vis_min}}, got {block!r}")
    data = _block(block, PerFrameReanchorConfig, "solve_stage.per_frame_reanchor")
    missing = sorted({f.name for f in dataclasses.fields(PerFrameReanchorConfig)} - set(SAFETY_OPTIONS) - set(data))
    if missing:
        raise ValueError(f"solve_stage.per_frame_reanchor is missing {missing}: spell threshold, band, perp_k and "
                         "vis_min")
    return PerFrameReanchorConfig(**data)


def _fixed_scale(block: Any) -> FixedScaleConfig | None:
    """`solve_stage.fixed_scale`: null (the solved scale), or `{file: <name>}`."""
    if block is None:
        return None
    if not isinstance(block, Mapping):     # a ValueError, as every other solve_stage refusal
        raise ValueError(f"solve_stage.fixed_scale must be null or {{file}}, got {block!r}")  # noqa: TRY004
    data = _block(block, FixedScaleConfig, "solve_stage.fixed_scale")
    if "file" not in data:
        raise ValueError("solve_stage.fixed_scale is missing ['file']: spell the per-window file name")
    return FixedScaleConfig(**data)


def _branch_select(block: Any) -> BranchSelectConfig | None:
    """`solve_stage.branch_select`: null (today's single hypothesis), or a block spelling every field (so no
    constant is ever inherited silently); list values become tuples."""
    if block is None:
        return None
    if not isinstance(block, Mapping):     # a ValueError, as every other solve_stage refusal
        raise ValueError(f"solve_stage.branch_select must be null or a mapping, got {block!r}")  # noqa: TRY004
    data = _block(block, BranchSelectConfig, "solve_stage.branch_select")
    missing = sorted({f.name for f in dataclasses.fields(BranchSelectConfig)} - set(data))
    if missing:
        raise ValueError(f"solve_stage.branch_select is missing {missing}: spell every field")
    for name in ("seed_offsets", "scale_mults"):
        data[name] = tuple(data[name])
    return BranchSelectConfig(**data)


def _block(mapping: Mapping[str, Any], cls: type[Any] | None, name: str) -> dict[str, Any]:
    """`mapping` as a plain dict (an OmegaConf node's interpolations resolve on access); unknown keys of `cls` raise."""
    data = {str(key): mapping[key] for key in mapping}
    if cls is not None:
        known = [f.name for f in dataclasses.fields(cls)]
        unknown = sorted(set(data) - set(known))
        if unknown:
            raise ValueError(f"{name} has unknown keys {unknown}; known keys: {known}")
    return data
