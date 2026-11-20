"""Collision-checked moves between a home pose and the trajectory.

The run script (`fastdo.robot.g1_sdk_replay`) moves the arms from the pose it measures at the start (HOME) to the
trajectory's first command (the HOLD pose: right arm and hand at the trajectory start, left arm parked, waist 0) and,
after the trajectory, back to HOME. Unitree's own references do these moves as straight joint-space interpolations at a
capped joint speed (xr_teleoperate `ctrl_dual_arm_go_home`, the SDK arm7 example's move to the zero posture). In front
of a table a straight line is not safe: from the official zero posture the early drafts swept 46-66 mm into the table
and 12-32 mm into the object. Here each move is PLANNED offline in the package's own MuJoCo scene
and stored in `trajectory.npz`; the run script follows the stored polyline (`g1_sdk_replay.Polyline`, still a
joint-space interpolation at `--hold-speed` between waypoints) and refuses a HOME it was not planned from.

**Homes** (`TransferConfig.homes`, 17 joints each: right arm 7, left arm 7, waist 3): `zeros`, the official arm_sdk
home (all arm joints 0; xr_teleoperate's go-home target and the arm7 example's zero posture: forearms horizontal,
pointing forward), and `arms_hanging`, our choice (right arm `RIGHT_ARM_HANGING`, the mirror of
`g1_constants.LEFT_ARM_PARK`, forearms down). The pose Unitree's standing controller holds the arms at is NOT in any
official file (UNVERIFIED); the lab can add its measured pose with `export_g1_package.py --home-q`.

**Planner** (`plan_to_hold`, `plan_go_home`): TO_HOLD = the right hand shapes to its start pose at HOME, then the left
arm parks (right arm at HOME), then the right arm goes to the hold pose (left parked); GO_HOME = the right arm goes home
from the trajectory end (left parked, hand holding its last command), then the left arm. Each arm segment is the
straight line when it is collision-free, else an RRT-Connect path (seeded, deterministic) in that arm's 7 joints,
shortcut greedily. The table is checked as a SOLID block from the floor to its top (the simulation models only the
4 cm top; legs, apron or a frame under it are unknown), so no planned path passes under the table top. **Table
tolerance** (`table_tol`, 1 cm; the README's placement instruction): the block is GROWN by it on its top and on every
edge, and every object pose is also checked raised and lowered by it (the object stands on the table), so the
endpoint relaxation below never takes a path closer than the tolerance to a table the lab may set up
(C2's hold pose sat 4.9 mm from the nominal table, and a table 1 cm higher touched it). A configuration
is valid when every robot geom is at least `d_safe` (2 cm) from that table and from
the object (at each of the given object poses: the start pose for TO_HOLD; the simulated end pose and the reference
end pose for GO_HOME) and no two robot geoms penetrate more than `self_tol` (pairs within the right hand are not
counted: finger-finger contact does not depend on the arm path, and the arm IK skips them too); next to an endpoint that is itself closer,
the segment's threshold drops to that endpoint's clearance minus 1 mm, and an endpoint closer than 1 mm fails the
plan. Edges are checked every `resolution` rad (L-inf). The run script starts from the MEASURED pose, within
`home_tol` of HOME: `tolerance_box_ok` checks that `tol_samples` corners and the 28 axis points of that box (14 arm
joints) and the straight moves from them to HOME are valid too (for GO_HOME the move from the vertex before HOME
to them: the run script's last commanded segment ends at the measured pose). The waist is not modelled (the official model welds it
at 0); a HOME within `home_tol` of the waist at 0 is accepted.

**Simulation** (`check_transfers`): the run script itself (`g1_sdk_replay.Replayer`) driven through
`fastdo.robot.replay.simulate_run` on the MuJoCo scene, from each planned HOME: the whole sequence, and operator-stop
drills (`drills`: a stop half way through TO_HOLD and at 25/50/75% of the playback, after which the script retraces
its own commands back to HOME), and the whole sequence again on the table at both edges of the tolerance
(`table_tolerance_runs`: top +tol with every edge out by tol, and top -tol; the object raised / lowered with it). Per
phase: table / object / self-collision depths, table contact force, robot-object contact, tracking. The run-script
options the simulation used (hold speed, speed scale, feed-forward) and the Dex3 hand pose it started from
(`start_hand`, 0) are recorded: `--execute` accepts no other options and refuses a measured hand more than
`g1_sdk_replay.START_HAND_TOL` from that start. `TransferConfig.speed_scale` (the export's
`--speed-scale`) is the playback scale the run, the drills and the table-tolerance runs are simulated at.
`g1_sdk_replay.SIM_GATE` reads them from `sim_stats.json` (`transfers`). The trajectory replay itself
(`fastdo.robot.replay`, the gate's object-motion criteria) is simulated at the trajectory's own timing, not at the
speed scale; the object criteria of the run itself, simulated at the speed scale (`object_criteria` of `run`), are
recorded too, and the gate holds them to the same thresholds.

**Tolerance runs** (`fastdo.robot.tolerance_runs`): the planner's clearance to the object
is reduced by the object's placement envelope (+-1 cm x / y, +-5 deg yaw) and the run is simulated with the
object at each of the 8 corners (`placement_runs`); and the run is simulated from the measured states the run script
accepts (`state_runs`: the 28 axis extremes and `state_corners` seeded corners of the arm box, the Dex3 hand at
+-`g1_sdk_replay.START_HAND_TOL`). Every simulated run records its run options (`replay.run_options`), and the report
records `--max-track-err` / `--hold-timeout` (the run script accepts only these or lower values under `--execute`).
"""

from __future__ import annotations

import copy
import itertools
import logging
from dataclasses import asdict, dataclass, field

import mujoco
import numpy as np

from fastdo.robot import g1_constants as C
from fastdo.robot import g1_sdk_replay as S
from fastdo.robot.g1_model import G1Upper
from fastdo.robot.replay import ReplayScene, perturbed_table_scene, simulate_run
from fastdo.robot.tolerance_runs import (
    OBJECT_XY_TOL_M,
    OBJECT_YAW_TOL_DEG,
    object_criteria,
    object_envelope_m,
    placement_pose,
    state_starts,
)
from fastdo.robot.trajectory import (
    RobotTrajectory,
    TrajectoryLimits,
    left_park_torque,
)

logger = logging.getLogger(__name__)

__all__ = ["HOMES", "RIGHT_ARM_HANGING", "TABLE_TOL_M", "PlannedPath", "TransferChecker", "TransferConfig",
           "TransferReport", "arm_gravity", "check_transfers", "plan_fields", "plan_go_home", "plan_to_hold",
           "z_shifted"]

# Mirror of LEFT_ARM_PARK (shoulder roll -0.4 abducts the RIGHT arm away from the hip; elbow 1.57 hangs the forearm).
RIGHT_ARM_HANGING: tuple[float, ...] = (0.0, -0.4, 0.0, 1.57, 0.0, 0.0, 0.0)
HOMES: tuple[tuple[str, tuple[float, ...]], ...] = (
    ("zeros", (0.0,) * 17),
    ("arms_hanging", RIGHT_ARM_HANGING + tuple(C.LEFT_ARM_PARK) + (0.0, 0.0, 0.0)),
)
_ARM = slice(0, 14)
TABLE_TOL_M = 0.01      # the package's table placement tolerance (height and every edge, README "Object placement")


@dataclass(frozen=True)
class TransferConfig:
    homes: tuple[tuple[str, tuple[float, ...]], ...] = HOMES
    home_tol: float = 0.08              # rad: the run script refuses a measured pose farther than this from a HOME
    d_safe: float = 0.02                # m: robot-table / robot-object clearance of the planned path
    self_tol: float = 0.0005            # m: robot-robot penetration allowed on the planned path
    resolution: float = 0.01            # rad: edge check step (L-inf)
    rrt_step: float = 0.15
    rrt_iters: int = 4000
    shortcut_iters: int = 200
    seed: int = 0
    tol_samples: int = 48
    table_tol: float = TABLE_TOL_M      # m: plan against the table grown by this (top and edges), simulate +-it
    object_xy_tol: float = OBJECT_XY_TOL_M          # m: object placement tolerance in x / y ...
    object_yaw_tol_deg: float = OBJECT_YAW_TOL_DEG  # deg: ... and in yaw; planned (envelope) and simulated (corners)
    state_corners: int = S.STATE_CORNERS_MIN        # seeded arm-box corners simulated per home (+ the 28 axis extremes)
    state_edge: float = 0.999           # the measured-state runs start this fraction of each tolerance out
    hold_speed: float = 0.3             # rad/s: the run script's default --hold-speed (the simulated speed)
    speed_scale: float = 1.0            # the run script's --speed-scale the run is simulated at
    drills: tuple[tuple[str, float], ...] = (("to_hold", 0.5), ("play", 0.25), ("play", 0.5), ("play", 0.75))
    enabled: bool = True


@dataclass(frozen=True)
class PlannedPath:
    ok: bool
    reason: str | None
    vertices: np.ndarray                # (N,21): right arm 7, left arm 7, right hand 7
    methods: tuple[str, ...]            # per arm segment: "direct" / "rrt" / "none" (no motion)
    min_table_m: float                  # along the path, sampled every `resolution` (capped at d_safe)
    min_object_m: float
    max_self_m: float
    length_rad: float                   # L-inf path length
    tolerance_box_ok: bool

    def to_dict(self) -> dict:
        out = {k: v for k, v in asdict(self).items() if k != "vertices"}
        out["methods"] = list(self.methods)
        out["waypoints"] = len(self.vertices)
        return out


def arm_gravity(g1: G1Upper, right: np.ndarray, left: np.ndarray, hand: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`((N,7), (N,7))` gravity torques of the right and the left arm at `(N,7)` joints (fixed base, zero velocity)."""
    m = g1.model
    d = mujoco.MjData(m)
    d.qpos[:] = g1.default_qpos()
    left_dof = m.jnt_dofadr[[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in C.LEFT_ARM_JOINTS]]
    res = np.zeros(m.nv)
    tr, tl = np.zeros((len(right), 7)), np.zeros((len(right), 7))
    for k in range(len(right)):
        d.qpos[g1.right_arm_qpos], d.qpos[g1.left_arm_qpos], d.qpos[g1.right_hand_qpos] = right[k], left[k], hand[k]
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_rne(m, d, 0, res)
        tr[k], tl[k] = res[g1.right_arm_dof], res[left_dof]
    return tr, tl


class TransferChecker:
    """Clearances of a whole-robot configuration in the replay scene (module docstring, "Planner")."""

    def __init__(self, scene: ReplayScene, objects: list[np.ndarray], d_safe: float, table_tol: float = 0.0,
                 object_tol: tuple[float, float] = (0.0, 0.0)) -> None:
        self.scene, self.g1, self.d_safe, self.table_tol = scene, scene.g1, d_safe, table_tol
        self.objects = [np.asarray(p, dtype=np.float64) for p in objects]
        # The clearance to each object pose is reduced by how far the object may be placed off it
        self.envelope = [object_envelope_m(scene, p, *object_tol) for p in self.objects]
        self.model = copy.deepcopy(scene.g1.model)
        m = self.model
        self.object_geoms = m.geom_bodyid == scene.object_body
        g, tb = scene.table_geom, m.geom_bodyid[scene.table_geom]   # the table as a solid block down to the floor,
        top = float(m.body_pos[tb][2]) + float(m.geom_pos[g][2]) + float(m.geom_size[g][2]) + table_tol
        m.geom_size[g][0] += table_tol                              # grown by the placement tolerance
        m.geom_size[g][1] += table_tol
        m.geom_size[g][2] = top / 2.0
        m.body_pos[tb][2] = top / 2.0 - float(m.geom_pos[g][2])    # the body moves (its geom shares its frame)
        m.geom_aabb[g] = [0.0, 0.0, 0.0, *m.geom_size[g]]
        m.geom_rbound[g] = float(np.linalg.norm(m.geom_size[g]))
        m.geom_margin[scene.table_geom] = d_safe
        m.geom_margin[self.object_geoms] = d_safe + max(self.envelope, default=0.0)
        self.data = mujoco.MjData(m)
        self.data.qpos[:] = self.g1.default_qpos()
        self.calls = 0
        hand = np.zeros(m.nbody, dtype=bool)                  # the right hand subtree (`arm_ik.IkCollider` alike)
        root = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
        hand[root] = True
        for b in range(root + 1, m.nbody):
            hand[b] = hand[m.body_parentid[b]]
        self.hand_geom = hand[m.geom_bodyid]

    def measure(self, x: np.ndarray) -> tuple[float, float, float]:
        """`(table clearance, object clearance, self-penetration)` in m at `x` (21,): clearances are signed distances
        (negative = penetration) capped at `d_safe`, the object's the minimum over its poses, each reduced by that
        pose's placement envelope (`tolerance_runs.object_envelope_m`)."""
        m, d, g1, sc = self.model, self.data, self.g1, self.scene
        d.qpos[g1.right_arm_qpos], d.qpos[g1.left_arm_qpos], d.qpos[g1.right_hand_qpos] = x[:7], x[7:14], x[14:21]
        table = obj = self.d_safe
        self_d = 0.0
        rob, body = sc.robot_geoms, m.geom_bodyid
        oq = sc.object_qpos
        for pose, env in zip(self.objects, self.envelope, strict=True):
            d.qpos[oq:oq + 3] = pose[:3, 3]
            mujoco.mju_mat2Quat(d.qpos[oq + 3:oq + 7], pose[:3, :3].ravel())
            mujoco.mj_kinematics(m, d)
            mujoco.mj_collision(m, d)
            self.calls += 1
            for c in d.contact[: d.ncon]:
                g_1, g_2 = int(c.geom[0]), int(c.geom[1])
                dist = float(c.dist)
                if rob[g_1] and rob[g_2]:
                    if not (self.hand_geom[g_1] and self.hand_geom[g_2]):
                        self_d = max(self_d, -dist)
                elif (rob[g_1] and g_2 == sc.table_geom) or (rob[g_2] and g_1 == sc.table_geom):
                    table = min(table, dist)
                elif (rob[g_1] and body[g_2] == sc.object_body) or (rob[g_2] and body[g_1] == sc.object_body):
                    obj = min(obj, dist - env)
        return table, obj, self_d


@dataclass
class _Limits:
    table: float
    obj: float
    self_: float

    def ok(self, meas: tuple[float, float, float]) -> bool:
        return meas[0] >= self.table and meas[1] >= self.obj and meas[2] <= self.self_


class _Segments:
    """Edge checks and the per-segment thresholds (module docstring)."""

    def __init__(self, chk: TransferChecker, cfg: TransferConfig) -> None:
        self.chk, self.cfg = chk, cfg

    def limits(self, *ends: np.ndarray) -> tuple[_Limits | None, str | None]:
        meas = [self.chk.measure(e) for e in ends]
        t = min(self.cfg.d_safe, min(mm[0] for mm in meas) - 0.001)
        o = min(self.cfg.d_safe, min(mm[1] for mm in meas) - 0.001)
        s = max(self.cfg.self_tol, max(mm[2] for mm in meas))
        grown = f" grown by the {self.chk.table_tol * 100:g} cm placement tolerance" if self.chk.table_tol else ""
        for name, mm in zip(("start", "end"), (meas[0], meas[-1]), strict=True):
            if mm[0] < 0.001:
                return None, f"the {name} pose is {mm[0] * 1000:.1f} mm from the table{grown} (needs >= 1 mm)"
            if mm[1] < 0.001:
                return None, f"the {name} pose is {mm[1] * 1000:.1f} mm from the object (needs >= 1 mm)"
            if mm[2] > 0.001:
                return None, f"the {name} pose self-collides {mm[2] * 1000:.1f} mm"
        return _Limits(t, o, s), None

    def edge(self, a: np.ndarray, b: np.ndarray, lim: _Limits) -> bool:
        n = max(int(np.ceil(float(np.abs(b - a).max()) / self.cfg.resolution - 1e-9)), 1)
        return all(lim.ok(self.chk.measure(a + (b - a) * (i / n))) for i in range(1, n + 1))

    def sweep(self, verts: np.ndarray) -> tuple[float, float, float]:
        """Worst clearances along the polyline `verts`, sampled every `resolution`."""
        worst = [np.inf, np.inf, 0.0]
        pts = [verts[0]]
        for a, b in itertools.pairwise(verts):
            n = max(int(np.ceil(float(np.abs(b - a).max()) / self.cfg.resolution - 1e-9)), 1)
            pts += [a + (b - a) * (i / n) for i in range(1, n + 1)]
        for p in pts:
            t, o, s = self.chk.measure(p)
            worst = [min(worst[0], t), min(worst[1], o), max(worst[2], s)]
        return float(worst[0]), float(worst[1]), float(worst[2])


def _rrt_connect(edge, a: np.ndarray, b: np.ndarray, lo: np.ndarray, hi: np.ndarray, rng: np.random.Generator,
                 step: float, iters: int) -> list[np.ndarray] | None:
    """RRT-Connect between `a` and `b` (7-D) with `edge(p, q) -> bool`; the path from `a` to `b`, or None."""
    trees = [([a], [-1]), ([b], [-1])]

    def nearest(tree: tuple[list, list], x: np.ndarray) -> int:
        return int(np.argmin(np.linalg.norm(np.asarray(tree[0]) - x, axis=1)))

    def toward(p: np.ndarray, x: np.ndarray) -> np.ndarray:
        gap = float(np.abs(x - p).max())
        return x.copy() if gap <= step else p + (x - p) * (step / gap)

    def trace(tree: tuple[list, list], i: int) -> list[np.ndarray]:
        out = []
        while i >= 0:
            out.append(tree[0][i])
            i = tree[1][i]
        return out[::-1]

    for it in range(iters):
        ta, tb = trees[it % 2], trees[(it + 1) % 2]
        x = rng.uniform(lo, hi)
        i = nearest(ta, x)
        new = toward(ta[0][i], x)
        if not edge(ta[0][i], new):
            continue
        ta[0].append(new)
        ta[1].append(i)
        ia = len(ta[0]) - 1
        j = nearest(tb, new)                          # connect: grow tb toward `new` until blocked or there
        while True:
            nxt = toward(tb[0][j], new)
            if not edge(tb[0][j], nxt):
                break
            tb[0].append(nxt)
            tb[1].append(j)
            j = len(tb[0]) - 1
            if float(np.abs(nxt - new).max()) < 1e-12:
                pa, pb = trace(ta, ia), trace(tb, j)[::-1][1:]
                path = pa + pb
                return path if it % 2 == 0 else path[::-1]
    return None


def _shortcut(path: list[np.ndarray], edge, rng: np.random.Generator, iters: int) -> list[np.ndarray]:
    path = list(path)
    for _ in range(iters):
        if len(path) <= 2:
            break
        i, j = sorted(rng.choice(len(path), 2, replace=False))
        if j - i > 1 and edge(path[i], path[j]):
            path = path[: i + 1] + path[j:]
    out, i = [path[0]], 0
    while i < len(path) - 1:                          # greedy: farthest directly reachable vertex
        j = len(path) - 1
        while j > i + 1 and not edge(path[i], path[j]):
            j -= 1
        out.append(path[j])
        i = j
    return out


def _arm_segment(seg: _Segments, x0: np.ndarray, q_goal: np.ndarray, cols: slice, lo: np.ndarray, hi: np.ndarray,
                 rng: np.random.Generator) -> tuple[list[np.ndarray] | None, str, str | None]:
    """Move the 7 joints `cols` of `x0` (21,) to `q_goal`: `(vertices after x0, method, failure)`."""
    x1 = x0.copy()
    x1[cols] = q_goal
    if float(np.abs(x1 - x0).max()) < 1e-12:
        return [], "none", None
    lim, why = seg.limits(x0, x1)
    if lim is None:
        return None, "none", why
    if seg.edge(x0, x1, lim):
        return [x1], "direct", None

    def full(q: np.ndarray) -> np.ndarray:
        x = x0.copy()
        x[cols] = q
        return x

    def edge(p: np.ndarray, q: np.ndarray) -> bool:
        return seg.edge(full(p), full(q), lim)

    path = _rrt_connect(edge, x0[cols].copy(), np.asarray(q_goal, dtype=np.float64), lo, hi, rng,
                        seg.cfg.rrt_step, seg.cfg.rrt_iters)
    if path is None:
        return None, "rrt", f"no collision-free path found ({seg.cfg.rrt_iters} RRT-Connect iterations)"
    path = _shortcut(path, edge, rng, seg.cfg.shortcut_iters)
    return [full(q) for q in path[1:]], "rrt", None


def _bounds(g1: G1Upper, limits: TrajectoryLimits) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    right = limits.arm_bounds()
    m = g1.model
    left = m.jnt_range[[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in C.LEFT_ARM_JOINTS]]
    return right[:, 0], right[:, 1], left[:, 0] + 0.05, left[:, 1] - 0.05


def _box_ok(seg: _Segments, home: np.ndarray, hand_home: np.ndarray, target: np.ndarray, cfg: TransferConfig,
            rng: np.random.Generator, lim: _Limits, last_lim: _Limits | None = None) -> bool:
    """The corners and axis points of the `home_tol` box around `home` (14 arm joints, hand `hand_home`) and the
    straight moves from them to `target` (21,) are valid (module docstring). With `last_lim` (GO_HOME), `target` is
    the vertex BEFORE the home and the move is target -> box point (the run script replaces the home by the measured
    pose, so its last commanded segment ends there), checked against `last_lim`."""
    signs = rng.choice([-1.0, 1.0], size=(cfg.tol_samples, 14))
    axes = np.concatenate([np.eye(14), -np.eye(14)])
    for off in np.concatenate([signs, axes]):
        x = np.concatenate([home[_ARM] + cfg.home_tol * off, hand_home])
        if not lim.ok(seg.chk.measure(x)):
            return False
        if not (seg.edge(target, x, last_lim) if last_lim is not None else seg.edge(x, target, lim)):
            return False
    return True


def _finish(seg: _Segments, verts: list[np.ndarray], methods: list[str], box_ok: bool) -> PlannedPath:
    v = np.asarray(verts)
    t, o, s = seg.sweep(v)
    length = float(np.abs(np.diff(v, axis=0)).max(axis=1).sum()) if len(v) > 1 else 0.0
    ok = box_ok
    return PlannedPath(ok=ok, reason=None if ok else "a pose within the home tolerance has no valid straight move to "
                       "the home path", vertices=v, methods=tuple(methods), min_table_m=t, min_object_m=o,
                       max_self_m=s, length_rad=length, tolerance_box_ok=box_ok)


def _failed(reason: str, verts: list[np.ndarray]) -> PlannedPath:
    return PlannedPath(ok=False, reason=reason, vertices=np.asarray(verts), methods=(), min_table_m=float("nan"),
                       min_object_m=float("nan"), max_self_m=float("nan"), length_rad=float("nan"),
                       tolerance_box_ok=False)


def plan_to_hold(chk: TransferChecker, home: np.ndarray, hold: np.ndarray, g1: G1Upper, limits: TrajectoryLimits,
                 cfg: TransferConfig) -> PlannedPath:
    """HOME (17,) -> HOLD (21,: right, left, hand); the hand starts at 0 (module docstring, "Planner")."""
    seg = _Segments(chk, cfg)
    rng = np.random.default_rng(cfg.seed)
    rlo, rhi, llo, lhi = _bounds(g1, limits)
    x0 = np.concatenate([home[_ARM], np.zeros(7)])
    x1 = np.concatenate([home[_ARM], hold[14:21]])
    lim, why = seg.limits(x0, x1)
    if lim is None:
        return _failed(f"hand at HOME: {why}", [x0])
    if not seg.edge(x0, x1, lim):
        return _failed("the hand cannot shape at HOME without contact", [x0])
    verts, methods = [x0, x1], []
    for cols, goal, lo, hi in ((slice(7, 14), hold[7:14], llo, lhi), (slice(0, 7), hold[0:7], rlo, rhi)):
        more, method, why = _arm_segment(seg, verts[-1], goal, cols, lo, hi, rng)
        methods.append(method)
        if more is None:
            return _failed(f"{'left' if cols.start else 'right'} arm to the hold pose: {why}", verts)
        verts += more
    box_lim, _ = seg.limits(x0, verts[1])
    box = box_lim is not None and _box_ok(seg, home, np.zeros(7), verts[1], cfg, rng, box_lim)
    return _finish(seg, verts, methods, box)


def plan_go_home(chk: TransferChecker, end: np.ndarray, home: np.ndarray, g1: G1Upper, limits: TrajectoryLimits,
                 cfg: TransferConfig) -> PlannedPath:
    """END (21,) -> HOME (17,), the hand holding its last command (module docstring, "Planner")."""
    seg = _Segments(chk, cfg)
    rng = np.random.default_rng(cfg.seed + 1)
    rlo, rhi, llo, lhi = _bounds(g1, limits)
    verts, methods = [np.asarray(end, dtype=np.float64)], []
    for cols, goal, lo, hi in ((slice(0, 7), home[0:7], rlo, rhi), (slice(7, 14), home[7:14], llo, lhi)):
        more, method, why = _arm_segment(seg, verts[-1], goal, cols, lo, hi, rng)
        methods.append(method)
        if more is None:
            return _failed(f"{'left' if cols.start else 'right'} arm home: {why}", verts)
        verts += more
    last = verts[-1]
    box_lim, _ = seg.limits(last, last)
    prev = verts[-2] if len(verts) > 1 else last
    last_lim, _ = seg.limits(prev, last)
    box = box_lim is not None and last_lim is not None and _box_ok(seg, home, last[14:21], prev, cfg, rng, box_lim,
                                                                   last_lim)
    return _finish(seg, verts, methods, box)


def _cmd_rows(verts: np.ndarray, g1: G1Upper) -> np.ndarray:
    """`(N,38)` `g1_sdk_replay.Cmd` vectors of `(N,21)` vertices: waist 0, gravity feed-forward of both arms."""
    tr, tl = arm_gravity(g1, verts[:, 0:7], verts[:, 7:14], verts[:, 14:21])
    return np.concatenate([verts[:, 0:14], np.zeros((len(verts), 3)), tr, tl, verts[:, 14:21]], axis=1)


def plan_fields(traj: RobotTrajectory, limits: TrajectoryLimits, g1: G1Upper,
                paths: list[tuple[np.ndarray, PlannedPath, PlannedPath]], home_tol: float) -> dict[str, np.ndarray]:
    """Every `g1_sdk_replay.Plan` field (float64 arrays, as `Plan.load` reads them) for `paths` =
    [(home (17,), to_hold, go_home)] of the homes that go into the package."""
    left_tau = left_park_torque(g1)
    hold = np.concatenate([traj.arm_q[0], C.LEFT_ARM_PARK, np.zeros(3), traj.arm_tau_ff[0], left_tau, traj.hand_q[0]])
    end = np.concatenate([traj.arm_q[-1], C.LEFT_ARM_PARK, np.zeros(3), traj.arm_tau_ff[-1], left_tau,
                          traj.hand_q[-1]])
    to_hold, go_home, homes = [], [], []
    for home, th, gh in paths:
        rows = _cmd_rows(th.vertices, g1)
        rows[0, 14:17] = home[14:17]
        rows[-1] = hold                                   # bit-exact trajectory start (validate() compares)
        to_hold.append(rows)
        rows = _cmd_rows(gh.vertices, g1)
        rows[0] = end
        rows[-1, 14:17] = home[14:17]
        go_home.append(rows)
        homes.append(np.asarray(home, dtype=np.float64))

    def index(parts: list[np.ndarray]) -> np.ndarray:
        return np.concatenate([[0], np.cumsum([len(p) for p in parts])]).astype(np.float64)

    return {
        "arm_t": traj.arm_t, "arm_q": traj.arm_q, "arm_tau_ff": traj.arm_tau_ff, "hand_t": traj.hand_t,
        "hand_q": traj.hand_q, "left_park": np.asarray(C.LEFT_ARM_PARK, dtype=np.float64), "left_tau_ff": left_tau,
        "arm_limits": limits.arm_bounds(), "hand_limits": limits.hand_bounds(),
        "arm_vmax": np.asarray(limits.arm_vmax, dtype=np.float64),
        "hand_vmax": np.asarray(limits.hand_vmax, dtype=np.float64),
        "home_q": np.asarray(homes, dtype=np.float64).reshape(-1, 17), "home_tol": np.asarray(home_tol, dtype=np.float64),
        "to_hold_path": np.concatenate(to_hold) if to_hold else np.zeros((0, S.CMD_SIZE)),
        "to_hold_index": index(to_hold),
        "go_home_path": np.concatenate(go_home) if go_home else np.zeros((0, S.CMD_SIZE)),
        "go_home_index": index(go_home),
    }


@dataclass(frozen=True)
class TransferReport:
    home_tol: float
    hold_speed: float
    homes: list[dict] = field(default_factory=list)     # per candidate home: name, home_q, plans, run, drills
    fields: dict[str, np.ndarray] = field(default_factory=dict)   # the `Plan` fields written into trajectory.npz
    send_tau_ff: bool = True            # the run-script options the simulation used (the gate allows only these)
    speed_scale: float = 1.0
    table_tol: float = 0.0              # m: the table placement tolerance planned and simulated
    start_hand: tuple[float, ...] = (0.0,) * 7   # rad: the Dex3 pose every simulated run starts from
    max_track_err: float = S.MAX_TRACK_ERR       # rad / s: the run script's --max-track-err and --hold-timeout the
    hold_timeout_s: float = 120.0                #   runs used (--execute accepts only these or lower values)
    object_xy_tol: float = 0.0          # m / deg: the object placement tolerance planned and simulated
    object_yaw_tol_deg: float = 0.0
    object_envelope_m: float = 0.0      # m: the largest clearance reduction the planner applied for it
    state_corners: int = 0              # seeded arm-box corners simulated per home
    state_edge: float = 0.0

    def to_dict(self) -> dict:
        return {"home_tol_rad": self.home_tol, "hold_speed_rad_s": self.hold_speed, "send_tau_ff": self.send_tau_ff,
                "speed_scale": self.speed_scale, "table_tol_m": self.table_tol, "start_hand": list(self.start_hand),
                "max_track_err_rad": self.max_track_err, "hold_timeout_s": self.hold_timeout_s,
                "object_xy_tol_m": self.object_xy_tol, "object_yaw_tol_deg": self.object_yaw_tol_deg,
                "object_envelope_m": self.object_envelope_m, "state_corners": self.state_corners,
                "state_edge": self.state_edge, "homes": self.homes}


def _pose(qpos7: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, np.asarray(qpos7[3:7], dtype=np.float64))
    out[:3, :3], out[:3, 3] = mat.reshape(3, 3), qpos7[:3]
    return out


def z_shifted(poses: list[np.ndarray], dz: float) -> list[np.ndarray]:
    """Each pose, and with `dz` > 0 also raised and lowered by `dz` (an object standing on a table within the
    placement tolerance)."""
    out = [np.asarray(p, dtype=np.float64) for p in poses]
    if dz <= 0.0:
        return out
    for p in list(out):
        for s in (dz, -dz):
            q = p.copy()
            q[2, 3] += s
            out.append(q)
    return out


TABLE_CASES = (("table_high", 1.0, True), ("table_low", -1.0, False))   # (name, dz sign, edges grown)


def check_transfers(scene: ReplayScene, traj: RobotTrajectory, limits: TrajectoryLimits, g1: G1Upper,
                    object_start: np.ndarray, object_end_ref: np.ndarray, object_end_sim: np.ndarray,
                    cfg: TransferConfig, object_ref_start: np.ndarray | None = None) -> TransferReport:
    """Plan TO_HOLD / GO_HOME from every candidate home, then simulate the run script from each planned one (module
    docstring). `object_end_sim`: the trajectory replay's final object qpos (7,) or pose (4,4); `object_ref_start` /
    `object_end_ref`: the reference object pose at the first and last reference row (the run's object criteria;
    default start: `object_start`)."""
    if not 0.0 < cfg.speed_scale <= 1.0:
        raise ValueError(f"speed_scale must be in (0, 1], got {cfg.speed_scale}")
    end_sim = _pose(object_end_sim) if np.asarray(object_end_sim).shape == (7,) else np.asarray(object_end_sim)
    ref_start = np.asarray(object_start if object_ref_start is None else object_ref_start, dtype=np.float64)
    tol = cfg.table_tol
    otol = (cfg.object_xy_tol, cfg.object_yaw_tol_deg)
    chk_start = TransferChecker(scene, z_shifted([object_start], tol), cfg.d_safe, tol, otol)
    chk_end = TransferChecker(scene, z_shifted([end_sim, object_end_ref], tol), cfg.d_safe, tol, otol)
    hold = np.concatenate([traj.arm_q[0], C.LEFT_ARM_PARK, traj.hand_q[0]])
    end = np.concatenate([traj.arm_q[-1], C.LEFT_ARM_PARK, traj.hand_q[-1]])
    records, planned = [], []
    for name, q in cfg.homes:
        home = np.asarray(q, dtype=np.float64)
        th = plan_to_hold(chk_start, home, hold, g1, limits, cfg)
        gh = plan_go_home(chk_end, end, home, g1, limits, cfg)
        rec = {"name": name, "home_q": home.tolist(), "to_hold_plan": th.to_dict(), "go_home_plan": gh.to_dict(),
               "in_package": bool(th.ok and gh.ok), "index": None, "run": None, "drills": []}
        logger.info("transfer %s: to_hold %s (%s), go_home %s (%s)", name, th.ok, th.reason, gh.ok, gh.reason)
        if rec["in_package"]:
            rec["index"] = len(planned)
            planned.append((home, th, gh))
        records.append(rec)
    fields = plan_fields(traj, limits, g1, planned, cfg.home_tol)
    plan = S.Plan(**{k: fields[k] for k in S.Plan.__dataclass_fields__})
    hand0 = np.zeros(7)                                   # the Dex3 start the runs are simulated from (Q6)
    run_cfg = S.Config(hold_speed=cfg.hold_speed, speed_scale=cfg.speed_scale, start_hand=tuple(hand0.tolist()))
    table_z = scene.cfg.table_z
    m = g1.model
    left_rng = m.jnt_range[[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in C.LEFT_ARM_JOINTS]]
    arm_rng = np.vstack([limits.arm_bounds(), left_rng])   # the run script refuses a right arm off its command limits
    for rec in records:
        if rec["index"] is None:
            continue
        k = rec["index"]
        rec["run"] = simulate_run(scene, plan, k, object_start, run_cfg, hand0=hand0)
        rec["run"]["object_criteria"] = object_criteria(rec["run"], ref_start, object_end_ref, table_z)
        rec["drills"] = [simulate_run(scene, plan, k, object_start, run_cfg, stop=stop, hand0=hand0)
                         for stop in cfg.drills]
        rec["table_tolerance_runs"] = []
        for name, sign, grow in (TABLE_CASES if tol > 0.0 else ()):
            start = np.asarray(object_start, dtype=np.float64).copy()
            start[2, 3] += sign * tol
            sc = perturbed_table_scene(scene, sign * tol, tol if grow else 0.0)
            rec["table_tolerance_runs"].append({"case": name, "dz_m": sign * tol, "grow_m": tol if grow else 0.0,
                                                **simulate_run(sc, plan, k, start, run_cfg, hand0=hand0)})
        rec["placement_runs"] = []                        # every corner of the object placement box
        for case, (sx, sy, sa) in (S.PLACEMENT_CASES.items() if otol[0] > 0.0 or otol[1] > 0.0 else ()):
            dx, dy, da = sx * otol[0], sy * otol[0], sa * otol[1]
            pose = placement_pose(object_start, dx, dy, da)
            run = simulate_run(scene, plan, k, pose, run_cfg, hand0=hand0)
            run["object_criteria"] = object_criteria(run, ref_start, object_end_ref, table_z)
            rec["placement_runs"].append({"case": case, "dx_m": dx, "dy_m": dy, "dyaw_deg": da,
                                          "object_start": pose.tolist(), **run})
        rec["state_runs"] = []                            # the measured states --execute accepts
        for st in state_starts(planned[k][0], hand0, cfg.home_tol, arm_rng, g1.right_hand_range, cfg.state_corners,
                               cfg.state_edge, cfg.seed + 2):
            pin = st["name"].startswith("hand")         # the script must MEASURE the hand at the edge
            run = simulate_run(scene, plan, k, object_start, run_cfg, hand0=st["hand0"], start_q=st["start_q"],
                               pin_hand=pin)
            rec["state_runs"].append({"name": st["name"], "home_tol": cfg.home_tol, "edge": cfg.state_edge,
                                      "hand_pinned_at_start": pin,
                                      "arm_offset": st["arm_offset"].tolist(), "hand_offset": st["hand_offset"].tolist(),
                                      "clipped": st["clipped"], **run})
    return TransferReport(home_tol=cfg.home_tol, hold_speed=cfg.hold_speed, homes=records, fields=fields,
                          send_tau_ff=run_cfg.send_tau_ff, speed_scale=run_cfg.speed_scale, table_tol=tol,
                          start_hand=tuple(hand0.tolist()), max_track_err=run_cfg.max_track_err,
                          hold_timeout_s=run_cfg.hold_timeout_s, object_xy_tol=otol[0], object_yaw_tol_deg=otol[1],
                          object_envelope_m=max(chk_start.envelope + chk_end.envelope, default=0.0),
                          state_corners=cfg.state_corners, state_edge=cfg.state_edge)
