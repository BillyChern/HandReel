"""Task placement and adaptation search for the G1 right arm: where the task goes on the table, and the
only adaptation allowed when the wrist cannot be oriented as the human's was.

**What is chosen** (`PlacementChoice`), and why each choice is neutral to the task or a stated assumption:
- object start (x, y) on the table and a yaw of the WHOLE task about the vertical (`yaw_deg`): the hand path, the
  object path and their relation are moved rigidly, gravity untouched (the robot meets the task from another side);
- the table height `table_z` within `table_z_range` (0.70-1.00 m: a sitting desk to a standing workbench; the G1
  shoulder is at 1.08 m, and its palm frame reaches at most 0.435 m from it (median 0.335 m over 20,000 uniform
  samples of the command box); the lab sets its table to the package's height);
- "up", ONLY when the reconstruction never estimated gravity (qc `gravity_source` "default": up is then the camera's
  -y at row 0, wrong by the wearer's head pitch, which no stored quantity identifies): a head pitch `pitch_deg`, or
  the normal of a stable hull face the object rests on at its start pose (`up_face`, `fastdo.robot.rest.stable_ups`).
  Each is an ASSUMPTION, reported as such; neither = the record's own up;
- a grasp rotation about a symmetry axis of the object (`fastdo.robot.symmetry`), only at angles the mesh passes.
A uniform rescale of the hand path is never considered.

**Approach and retreat** (`ReachTask.contact_span`, `approach_m`): before the human's first contact and after the
last one the human hand is farther than the contact threshold from the object, and the task rows there are only
those on which the reconstructed object was MEASURED static (`kinematic_rows.static_window`: within 2 cm / 10 deg of
its contact pose; rows where it moved are dropped and refilled by synthetic rows). The robot replaces the human's
hand path there by a straight vertical approach onto the first-contact wrist pose from `approach_m` (8 cm) above it,
and a vertical retreat of the same height from the last-contact pose (smoothstep in time, wrist orientation held,
fingers as the human's). It is reported with the measured object motion. On the
C1 window the human hand was 15-19 cm BELOW the bottle's support plane in the second before the grasp at every
head pitch from -10 to 80 deg (hand depth along the camera ray, or a support the hand reached under), which no robot
can follow above a table.

**Table consistency** (`table_depths`), per row: how deep the human hand (its 21 joints, moved rigidly with the wrist
target) lies below the table top where it is over the table, and how deep the reference object lies below it on rows
whose object CONFIDENCE is at least `object_conf_min` (0.1, i.e. a solve residual of about 3.2 noise floors: the
solver's `confidence` is exp(-z / 1.4), well-converged rows read 0.5-0.56; below it the reconstruction is not
trusted to say where the object is). A row deeper than `hand_table_tol` (2 cm) / `object_table_tol` (1 cm) is an
INFEASIBLE row with violation depth / 1 cm, folded into the objective like a reach failure. The object footprint must
lie on the table (`table_margin` inside its edges): a candidate that overhangs is never evaluated.

**Stable start** (`rest_inside`): the lab puts the object on the table at its start pose, and the replay settles it
there for 0.5 s before the robot moves, so that pose must REST: the centre of mass (of the object's convex hull,
which is also what MuJoCo collides) must lie at least `rest_margin` (5 mm) inside the support polygon of the hull
points within `rest_support_tol` (3 mm) of the lowest one. Otherwise the violation (how far it is from that, in cm)
is one more entry of the objective's margin vector. The pose depends on the assumed head pitch (gravity) and on the
reconstructed orientation; on the first C5 export (pitch 30 deg) the half sandwich stood on an edge at 47 deg
of roll and fell 0.84 m to the floor during the settle, before the robot moved.

**Objective.** The minimum over ALL task rows of the per-row margin `fastdo.robot.reach_eval.row_margins` (reach and
orientation residual, joint-limit margin, distance to singularity, collision clearance) and of the table-consistency
violation, extended LEXIMIN: candidates are compared on their sorted per-row margins (quantised to
`LEXIMIN_QUANTUM`), worst row first, then the second worst, and so on; ties left by that go to the mean margin. Plain
maximin was blind on the C1 window: a reference-object row that lies below the table at every placement (it
depends on the assumed pitch only) set the same minimum for every candidate. A path with any row out of reach (> 1 cm or
> 0.1 rad) or through the table scores below every feasible path; among infeasible paths, the smaller worst
violation wins, so the search always returns the least-bad placement and the package says what fails.

**Search** (`search_placement`):
1. Coarse: every (pitch, yaw, symmetry angle on a `sym_step_deg` grid) with the task's wrist centroid put at each
   of `anchor_points` (the table height clipped to its range, the object footprint required on the table), scored on
   every `coarse_stride`-th task row with the collision-aware IK (`collider`; without it the first C1 search
   put the hand 9-19 cm in front of the torso, where the body then blocked it: 65 of 70 rows out of reach).
2. Local: pattern search from the `top_coarse` best distinct rotations over (x, y, table_z, yaw, symmetry angle),
   each step half the previous, down to `min_step_xy`, same rows; the starts advance together, one batch of
   neighbours per round.
3. Final: the `top_final` best distinct candidates on EVERY task row with the collision-aware IK (robot body and
   table); the best by the objective is the choice.
Every evaluated candidate is returned (`SearchResult.records`) and written into the package.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import Executor
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np

from fastdo.robot.arm_ik import ArmIkConfig
from fastdo.robot.kinematic_rows import robot_frame_targets
from fastdo.robot.placement import Placement, footprint_inside, pitched_up, place_on_table
from fastdo.robot.reach_eval import (
    IkJob,
    IkOutcome,
    MarginConfig,
    RetryPool,
    evaluate_jobs,
    row_margins,
)
from fastdo.robot.rest import StableUp, support_inside
from fastdo.robot.symmetry import SymmetryAxis, grasp_rotation, rotate_grasp

logger = logging.getLogger(__name__)

LEXIMIN_QUANTUM = 0.1   # margin units (1 mm / 0.01 rad / 1 mm of table depth): rows closer than this tie

__all__ = ["PlacementChoice", "ReachSearchConfig", "ReachTask", "SearchResult", "approach_retreat", "choice_targets",
           "physical_check", "rest_inside", "rest_violation", "score_outcome", "search_placement", "table_depths",
           "table_violation"]


@dataclass(frozen=True)
class ReachTask:
    wrist_world: np.ndarray       # (T,4,4) video world
    object_world: np.ndarray      # (T,4,4) video world (metric object frame -> world)
    joints_world: np.ndarray      # (T,21,3) human right-hand joints, video world
    anchor: int
    rows: np.ndarray              # (T,) bool task rows
    times: np.ndarray             # (T,)
    hand_q: np.ndarray            # (T,7)
    mesh_vertices: np.ndarray
    scale: float
    cam_to_world0: np.ndarray     # (4,4)
    up_record: np.ndarray         # (3,) the record's up in the video world
    rest_check: bool
    symmetries: tuple[SymmetryAxis, ...] = ()
    object_conf: np.ndarray | None = None          # (T,) solver confidence of the object pose (None: all trusted)
    contact_span: tuple[int, int] | None = None    # first / last human contact row (None: no approach synthesis)
    approach_m: float = 0.0                        # approach / retreat height (0: the human's own hand path)
    hull_vertices: np.ndarray | None = None        # (H,3) convex hull of the object, metric object frame (None: no
    hull_com: np.ndarray | None = None             #   rest check); (3,) its volume centroid
    stable_ups: tuple[StableUp, ...] = ()          # gravity candidates from resting faces (`fastdo.robot.rest`)
    approach_synthesised: bool = False             # rows before the first contact are a planned approach:
                                                   #   only the retreat is replaced by `approach_retreat`


@dataclass(frozen=True)
class PlacementChoice:
    table_xy: tuple[float, float]
    yaw_deg: float = 0.0
    table_z: float = 0.80
    pitch_deg: float | None = None
    sym_axis: int | None = None
    sym_deg: float = 0.0
    up_face: int | None = None          # index into `ReachTask.stable_ups` (overrides `pitch_deg`)

    def to_dict(self) -> dict:
        out = asdict(self)
        out["table_xy"] = [round(float(v), 4) for v in self.table_xy]
        return out


@dataclass(frozen=True)
class ReachSearchConfig:
    pitch_deg: tuple[float | None, ...] = (None, 15.0, 30.0, 45.0, 60.0)
    yaw_deg: tuple[float, ...] = tuple(float(v) for v in range(-180, 180, 30))
    sym_step_deg: int = 60
    table_z_range: tuple[float, float] = (0.70, 1.00)
    anchor_points: tuple[tuple[float, float, float], ...] = ((0.26, -0.20, 0.90), (0.32, -0.26, 0.95),
                                                             (0.30, -0.10, 0.85))
    coarse_stride: int = 4
    top_coarse: int = 6
    top_final: int = 3
    step_xy: float = 0.04
    step_z: float = 0.04
    step_yaw: float = 15.0
    min_step_xy: float = 0.01
    max_rounds: int = 10
    collider: bool = True                 # collision-aware IK (body, table) in the coarse and local stages too
    table_margin: float = 0.02
    object_table_tol: float = 0.01
    hand_table_tol: float = 0.02
    object_conf_min: float = 0.1
    rest_support_tol: float = 0.003       # hull points this close to the lowest one support the object
    rest_margin: float = 0.005            # the centre of mass must lie this far inside the support polygon
    margin: MarginConfig = field(default_factory=MarginConfig)
    ik: ArmIkConfig = field(default_factory=ArmIkConfig)


@dataclass(frozen=True)
class SearchResult:
    choice: PlacementChoice
    score: float
    records: list[dict]
    final: dict                   # the chosen candidate's full-row statistics


def _up(task: ReachTask, ch: PlacementChoice) -> np.ndarray:
    if ch.up_face is not None:
        return task.stable_ups[ch.up_face].up_world
    return task.up_record if ch.pitch_deg is None else pitched_up(task.cam_to_world0, ch.pitch_deg)


def choice_targets(task: ReachTask, ch: PlacementChoice) -> tuple[Placement, np.ndarray, np.ndarray]:
    """`(placement, wrist targets (T,4,4), reference object poses (T,4,4))` in the robot frame for one choice."""
    pl = place_on_table(task.object_world[task.anchor], task.mesh_vertices, task.scale, _up(task, ch),
                        task.cam_to_world0, ch.table_xy, ch.table_z, yaw_offset=float(np.radians(ch.yaw_deg)))
    wrist_r, obj_r = robot_frame_targets(pl.T_rv, pl.object_start, task.wrist_world, task.object_world, task.anchor)
    if ch.sym_axis is not None and ch.sym_deg % 360.0 != 0.0:
        ax = task.symmetries[ch.sym_axis]
        wrist_r = rotate_grasp(wrist_r, obj_r, grasp_rotation(ax.axis, ax.centre, np.radians(ch.sym_deg)))
    if task.contact_span is not None and task.approach_m > 0.0:
        wrist_r = approach_retreat(wrist_r, task.times, task.contact_span, task.approach_m,
                                   approach=not task.approach_synthesised)
    return pl, wrist_r, obj_r


def approach_retreat(wrist: np.ndarray, times: np.ndarray, span: tuple[int, int], height: float,
                     approach: bool = True) -> np.ndarray:
    """`(T,4,4)` wrist targets with the rows before `span[0]` replaced by a vertical approach onto row `span[0]` from
    `height` above it (unless `approach` is False: a synthesised approach is kept) and the rows after
    `span[1]` by a vertical retreat from row `span[1]` (module docstring)."""
    out = wrist.copy()
    c0, c1 = span
    t = np.asarray(times, dtype=np.float64)

    def smooth(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, 0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)

    if c0 > 0 and approach:
        frac = smooth((t[c0] - t[:c0]) / max(t[c0] - t[0], 1e-9))
        out[:c0] = wrist[c0]
        out[:c0, 2, 3] += height * frac
    if c1 < len(t) - 1:
        frac = smooth((t[c1 + 1:] - t[c1]) / max(t[-1] - t[c1], 1e-9))
        out[c1 + 1:] = wrist[c1]
        out[c1 + 1:, 2, 3] += height * frac
    return out


def table_depths(task: ReachTask, pl: Placement, wrist_r: np.ndarray, obj_r: np.ndarray,
                 table_x: tuple[float, float], table_y: tuple[float, float], cfg: ReachSearchConfig
                 ) -> tuple[np.ndarray, np.ndarray]:
    """`(hand depth (T,), object depth (T,))` below the table top in m (>= 0; module docstring, "Table
    consistency"); the object depth is 0 on rows below `object_conf_min`. The object's lowest point is read on EVERY
    vertex of its convex hull (`task.hull_vertices`; without one, every mesh vertex), never on a subsample."""
    hull_pts = (np.asarray(task.hull_vertices, dtype=np.float64) if task.hull_vertices is not None
                else task.scale * np.asarray(task.mesh_vertices, dtype=np.float64))
    z_obj = (np.einsum("tij,vj->tvi", obj_r[:, :3, :3], hull_pts)[..., 2] + obj_r[:, None, 2, 3]).min(axis=1)
    obj_d = np.maximum(0.0, pl.table_z - z_obj)
    if task.object_conf is not None:
        obj_d = np.where(np.asarray(task.object_conf) >= cfg.object_conf_min, obj_d, 0.0)
    move = np.einsum("tij,tjk->tik", wrist_r, np.linalg.inv(task.wrist_world))
    j = np.einsum("tij,tvj->tvi", move[:, :3, :3], task.joints_world) + move[:, None, :3, 3]
    over = (j[..., 0] >= table_x[0]) & (j[..., 0] <= table_x[1]) & (j[..., 1] >= table_y[0]) & (j[..., 1] <= table_y[1])
    hand_d = np.where(over, np.maximum(0.0, pl.table_z - j[..., 2]), 0.0).max(axis=1)
    return hand_d, obj_d


def rest_inside(task: ReachTask, pl: Placement, cfg: ReachSearchConfig) -> float | None:
    """`rest.support_inside` at the placed start pose (module docstring, "Stable start"); None without a hull."""
    if task.hull_vertices is None or task.hull_com is None:
        return None
    return support_inside(task.hull_vertices, task.hull_com, pl.object_start, cfg.rest_support_tol)


def physical_check(task: ReachTask, pl: Placement, wrist_r: np.ndarray, obj_r: np.ndarray,
                   table_x: tuple[float, float], table_y: tuple[float, float], cfg: ReachSearchConfig) -> dict:
    """`{"ok", "on_table", "object_below_table_m", "hand_below_table_m", "table_bad_rows", "rest_com_inside_m",
    "rest_stable"}` over the task rows: `ok` = the footprint is on the table (the only hard requirement); the depths
    are the worst rows'; `rest_*` the stable-start check (`rest_inside`; None without a hull)."""
    rows = np.asarray(task.rows, dtype=bool)
    on_table = footprint_inside(pl.object_start, task.mesh_vertices, task.scale, table_x, table_y, cfg.table_margin)
    hand_d, obj_d = table_depths(task, pl, wrist_r, obj_r, table_x, table_y, cfg)
    bad = rows & ((hand_d > cfg.hand_table_tol) | (obj_d > cfg.object_table_tol))
    inside = rest_inside(task, pl, cfg)
    return {"ok": bool(on_table), "on_table": bool(on_table), "object_below_table_m": float(obj_d[rows].max()),
            "hand_below_table_m": float(hand_d[rows].max()), "table_bad_rows": int(bad.sum()),
            "rest_com_inside_m": inside, "rest_stable": None if inside is None else bool(inside >= cfg.rest_margin)}


def rest_violation(task: ReachTask, pl: Placement, cfg: ReachSearchConfig) -> float | None:
    """The unstable-start violation in units of 1 cm (0 when stable, None when unchecked: no hull)."""
    inside = rest_inside(task, pl, cfg)
    return None if inside is None else max(0.0, cfg.rest_margin - inside) / 0.01


def table_violation(task: ReachTask, pl: Placement, wrist_r: np.ndarray, obj_r: np.ndarray,
                    table_x: tuple[float, float], table_y: tuple[float, float], cfg: ReachSearchConfig) -> np.ndarray:
    """`(T,)` table-consistency violation in units of 1 cm (0 on consistent rows)."""
    hand_d, obj_d = table_depths(task, pl, wrist_r, obj_r, table_x, table_y, cfg)
    v = np.maximum(np.where(hand_d > cfg.hand_table_tol, hand_d, 0.0), np.where(obj_d > cfg.object_table_tol, obj_d, 0.0))
    return v / 0.01


def score_outcome(out: IkOutcome, cfg: MarginConfig, table_violation: np.ndarray | None = None,
                  rest: float | None = None) -> dict:
    """Objective of the module docstring plus the worst residuals; `table_violation` (R,) per row in cm units;
    `rest` the unstable-start violation (cm units), one more entry of the margin vector (+1 when stable)."""
    m = row_margins(out.pos_err, out.rot_err, out.jl_margin, out.sigma_min, out.clearance, cfg)
    if table_violation is not None:
        tv = np.asarray(table_violation, dtype=np.float64)
        m = np.where(tv > 0.0, np.minimum(m, -np.maximum(tv, 1.0)), m)
    n_rows = int(m.size)
    if rest is not None:
        m = np.append(m, 1.0 if rest <= 0.0 else -max(rest, 1.0))
    return {"score": float(m.min()), "mean_margin": float(m.mean()), "infeasible_rows": int((m[:n_rows] < 0).sum()),
            "rest_violation": rest,
            "leximin": tuple(float(v) for v in np.sort(np.round(m / LEXIMIN_QUANTUM) * LEXIMIN_QUANTUM)),
            "rows": n_rows, "pos_err_max_m": float(out.pos_err.max()), "rot_err_max_rad": float(out.rot_err.max()),
            "jl_margin_min_rad": float(out.jl_margin.min()), "sigma_min_min": float(out.sigma_min.min()),
            "clearance_min_m": float(np.nanmin(out.clearance)) if np.isfinite(out.clearance).any() else None}


class _Evaluator:
    def __init__(self, task: ReachTask, cfg: ReachSearchConfig, table_x: tuple[float, float],
                 table_y: tuple[float, float], pool: RetryPool | Executor | None) -> None:
        self.task, self.cfg, self.tx, self.ty, self.pool = task, cfg, table_x, table_y, pool
        self.records: list[dict] = []
        self.cache: dict[tuple, dict] = {}

    def key(self, ch: PlacementChoice, full: bool) -> tuple:
        return (round(ch.table_xy[0], 4), round(ch.table_xy[1], 4), round(ch.yaw_deg % 360.0, 3), round(ch.table_z, 4),
                ch.pitch_deg, ch.up_face, ch.sym_axis, round(ch.sym_deg % 360.0, 3), full)

    def run(self, choices: list[PlacementChoice], stage: str, full: bool = False) -> list[dict]:
        rows_all = np.flatnonzero(self.task.rows)
        rows = rows_all if full else rows_all[:: max(self.cfg.coarse_stride, 1)]
        jobs: list[IkJob] = []
        pending: list[dict[str, Any]] = []
        out: list[dict[str, Any]] = []
        tviol: list[np.ndarray] = []
        rests: list[float | None] = []
        for ch in choices:
            k = self.key(ch, full)
            if k in self.cache:
                out.append(self.cache[k])
                continue
            pl, wrist_r, obj_r = choice_targets(self.task, ch)
            phys = physical_check(self.task, pl, wrist_r, obj_r, self.tx, self.ty, self.cfg)
            rec: dict[str, Any] = {"stage": stage, "choice": ch.to_dict(), "physical": phys}
            self.cache[k] = rec
            out.append(rec)
            if not phys["ok"]:
                rec.update(score=-1e9, mean_margin=-1e9, leximin=(-1e9,))
                self.records.append(rec)
                continue
            tviol.append(table_violation(self.task, pl, wrist_r, obj_r, self.tx, self.ty, self.cfg)[rows])
            rests.append(rest_violation(self.task, pl, self.cfg))
            jobs.append(IkJob(key=len(jobs), targets=wrist_r[rows], times=self.task.times[rows],
                              hand_q=self.task.hand_q[rows], table_z=ch.table_z if (full or self.cfg.collider) else None,
                              ik=self.cfg.ik, table_x=self.tx, table_y=self.ty))
            pending.append(rec)
        for rec, res, tv, rv in zip(pending, evaluate_jobs(jobs, self.pool), tviol, rests, strict=True):
            rec.update(score_outcome(res, self.cfg.margin, tv, rv))
            if full:
                rec["outcome"] = res
            self.records.append(rec)
        return out


def _key(rec: dict) -> tuple:
    return (rec["leximin"], rec["mean_margin"])


def _rank(recs: list[dict]) -> list[dict]:
    return sorted(recs, key=_key, reverse=True)


def _choice(rec: dict) -> PlacementChoice:
    c = dict(rec["choice"])
    c["table_xy"] = tuple(c["table_xy"])
    return PlacementChoice(**c)


def _centred(task: ReachTask, base: PlacementChoice, point: tuple[float, float, float],
             zr: tuple[float, float]) -> PlacementChoice:
    """`base` moved so the task rows' mean wrist position lands on `point` (table height clipped to `zr`)."""
    _, wrist_r, _ = choice_targets(task, base)
    c = wrist_r[task.rows, :3, 3].mean(axis=0)
    z = float(np.clip(base.table_z + point[2] - c[2], *zr))
    return replace(base, table_xy=(base.table_xy[0] + point[0] - c[0], base.table_xy[1] + point[1] - c[1]),
                   table_z=round(z, 3))


def _neighbours(ch: PlacementChoice, sxy: float, sz: float, syaw: float, task: ReachTask,
                cfg: ReachSearchConfig) -> list[PlacementChoice]:
    x, y = ch.table_xy
    out = [replace(ch, table_xy=(x + d, y)) for d in (-sxy, sxy)] + [replace(ch, table_xy=(x, y + d)) for d in (-sxy, sxy)]
    out += [replace(ch, table_z=round(float(z), 3)) for z in (ch.table_z - sz, ch.table_z + sz)
            if cfg.table_z_range[0] - 1e-9 <= z <= cfg.table_z_range[1] + 1e-9]
    out += [replace(ch, yaw_deg=ch.yaw_deg + d) for d in (-syaw, syaw)]
    if ch.sym_axis is not None:
        allowed = set(task.symmetries[ch.sym_axis].allowed_deg)
        for d in (-syaw, syaw):
            deg = (ch.sym_deg + d) % 360.0
            if abs(deg - round(deg)) < 1e-6 and round(deg) % 360 in allowed:
                out.append(replace(ch, sym_deg=float(round(deg) % 360)))
    return out


def search_placement(task: ReachTask, cfg: ReachSearchConfig, table_x: tuple[float, float],
                     table_y: tuple[float, float], pool: RetryPool | Executor | None = None,
                     allow_pitch: bool = True) -> SearchResult:
    """The three-stage search of the module docstring. Raises ValueError when no candidate keeps the object on the
    table."""
    ev = _Evaluator(task, cfg, table_x, table_y, pool)
    ups: list[tuple[float | None, int | None]] = [(p, None) for p in (cfg.pitch_deg if allow_pitch else (None,))]
    if allow_pitch:
        ups += [(None, k) for k in range(len(task.stable_ups))]
    syms: list[tuple[int | None, float]] = [(None, 0.0)]
    for ax in task.symmetries:
        syms += [(ax.index, float(d)) for d in ax.allowed_deg if d and d % cfg.sym_step_deg == 0]
    coarse = []
    for pitch, face in ups:
        for yaw in cfg.yaw_deg:
            for axis, deg in syms:
                base = PlacementChoice(table_xy=(0.0, 0.0), yaw_deg=float(yaw), table_z=0.80, pitch_deg=pitch,
                                       sym_axis=axis, sym_deg=deg, up_face=face)
                coarse += [_centred(task, base, p, cfg.table_z_range) for p in cfg.anchor_points]
    t0 = time.perf_counter()
    ranked = [r for r in _rank(ev.run(coarse, "coarse")) if r["physical"]["ok"]]
    logger.info("placement search: coarse %d candidates, %d physically consistent, best score %s (%.0f s)",
                len(coarse), len(ranked), f"{ranked[0]['score']:.3f}" if ranked else "-", time.perf_counter() - t0)
    if not ranked:
        raise ValueError("no placement keeps the object footprint on the table")
    starts, seen = [], set()
    for r in ranked:                               # distinct rotations
        sig = (r["choice"]["pitch_deg"], r["choice"]["up_face"], r["choice"]["yaw_deg"], r["choice"]["sym_axis"],
               r["choice"]["sym_deg"])
        if sig not in seen:
            seen.add(sig)
            starts.append(r)
        if len(starts) >= cfg.top_coarse:
            break
    # pattern search, all starts advanced together (one batch of neighbours per round)
    state: list[dict[str, Any]] = [{"best": r, "start": r, "sxy": cfg.step_xy, "sz": cfg.step_z,
                                    "syaw": cfg.step_yaw, "done": False} for r in starts]
    for _ in range(cfg.max_rounds * 4):
        live = [st for st in state if not st["done"]]
        if not live:
            break
        batches = [_neighbours(_choice(st["best"]), st["sxy"], st["sz"], st["syaw"], task, cfg) for st in live]
        results = ev.run([c for b in batches for c in b], "local")
        k = 0
        for st, b in zip(live, batches, strict=True):
            cand = _rank(results[k:k + len(b)])
            k += len(b)
            best = st["best"]
            if cand and _key(cand[0]) > _key(best):
                st["best"] = cand[0]
            elif st["sxy"] <= cfg.min_step_xy + 1e-12:
                st["done"] = True
            else:
                st["sxy"], st["sz"], st["syaw"] = st["sxy"] / 2, st["sz"] / 2, st["syaw"] / 2
    local = [st["best"] for st in state]
    for st in state:
        logger.info("placement search: local from %s -> score %.3f %s", st["start"]["choice"], st["best"]["score"],
                    st["best"]["choice"])
    logger.info("placement search: local done (%.0f s)", time.perf_counter() - t0)
    finals = [_choice(r) for r in _rank(local)[: cfg.top_final]]
    done = _rank(ev.run(finals, "final", full=True))
    best = done[0]
    logger.info("placement search: final (all rows, collider) scores %s (%.0f s)",
                [round(r["score"], 3) for r in done], time.perf_counter() - t0)
    final = {k: v for k, v in best.items() if k not in ("outcome", "leximin")}
    records = [{k: v for k, v in r.items() if k not in ("outcome", "leximin")} for r in ev.records]
    return SearchResult(choice=_choice(best), score=float(best["score"]), records=records, final=final)
