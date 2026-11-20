"""Simulated runs at the edges of what the lab may set up and what `--execute` accepts.

`fastdo.robot.transfer.check_transfers` simulates the run script (`g1_sdk_replay.Replayer`, through
`fastdo.robot.replay.simulate_run`) from every planned home; this module builds the extra starts it simulates and the
object criteria it records. `g1_sdk_replay.SIM_GATE` reads all of them from `sim_stats.json` ("transfers").

- **Object placement** (+-1 cm in x and y, +-5 deg in yaw, planned and simulated like the table):
  `placement_pose` puts the object start at a corner of that box (`g1_sdk_replay.PLACEMENT_CASES`, 8 corners; yaw about
  the vertical through the object origin). The planner reduces its clearance to the object by `object_envelope_m`, a
  bound on how far any point of the object's mesh moves under such an offset (sqrt(2) xy + 2 R sin(yaw / 2), R the
  largest horizontal distance of a mesh vertex from that vertical; the convex hull MuJoCo collides is inside the
  vertices), so a planned path keeps `d_safe` from the object anywhere inside the tolerance (the analogue of planning
  against the grown table). Each corner is then simulated, and the gate holds every corner to the table-tolerance
  thresholds and to the object criteria against the NOMINAL reference end (the robot replays open loop).
- **Object criteria at the run's speed scale**: `object_criteria` reads the object's motion in the
  simulated run itself (at the package's `--speed-scale`), with the replay's definitions: displacement over the
  playback, final error to the reference end at the playback's end, and a fall (more than 5 cm below the table top at
  the end of the playback or of the run).
- **Measured states**: `state_starts` gives the starts the run script accepts at the edge of its
  tolerances: every axis extreme and `corners` seeded corners of the 14-joint arm box of `home_tol`, and the Dex3 hand
  at +-`START_HAND_TOL` on every joint, each `edge` (0.999) of the tolerance out (the run script refuses beyond it)
  and clipped to the joint's range (a clipped joint is recorded). The waist stays at 0: the simulation welds it, and
  its 0.01 rad run-time tolerance is bounded by a lever argument (<= 4.2 mm at the wrist), not simulated.
"""

from __future__ import annotations

import mujoco
import numpy as np

from fastdo.robot import g1_sdk_replay as S
from fastdo.robot.replay import ReplayScene

__all__ = ["FALL_M", "OBJECT_XY_TOL_M", "OBJECT_YAW_TOL_DEG", "object_criteria", "object_envelope_m",
           "object_vertices_world", "placement_pose", "state_starts"]

OBJECT_XY_TOL_M = S.OBJECT_XY_TOL_MIN_M          # the package's object placement tolerance
OBJECT_YAW_TOL_DEG = S.OBJECT_YAW_TOL_MIN_DEG
FALL_M = 0.05                                     # the replay's fall: more than this below the table top


def placement_pose(start: np.ndarray, dx: float, dy: float, dyaw_deg: float) -> np.ndarray:
    """`start` (4,4) moved by `dx`, `dy` (m) and turned by `dyaw_deg` about the vertical through its origin."""
    out = np.asarray(start, dtype=np.float64).copy()
    a = np.radians(dyaw_deg)
    rz = np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])
    out[:3, :3] = rz @ out[:3, :3]
    out[0, 3] += dx
    out[1, 3] += dy
    return out


def object_vertices_world(scene: ReplayScene, pose: np.ndarray) -> np.ndarray:
    """`(V,3)` vertices of the object's collision meshes with the object body at `pose` (4,4). A non-mesh geom
    contributes the 8 corners of the cube of its bounding radius (a bound, never used by the replay scene)."""
    m = scene.g1.model
    pose = np.asarray(pose, dtype=np.float64)
    out = []
    for g in np.flatnonzero(m.geom_bodyid == scene.object_body):
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, m.geom_quat[g])
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            local = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(np.float64)
        else:
            r = float(m.geom_rbound[g])
            local = r * np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=np.float64)
        body = local @ rot.reshape(3, 3).T + m.geom_pos[g]
        out.append(body @ pose[:3, :3].T + pose[:3, 3])
    return np.concatenate(out) if out else np.zeros((0, 3))


def object_envelope_m(scene: ReplayScene, pose: np.ndarray, xy_tol: float, yaw_tol_deg: float) -> float:
    """Largest distance (m) a point of the object at `pose` moves when the object is placed anywhere within +-`xy_tol`
    in x and y and +-`yaw_tol_deg` about the vertical through its origin (module docstring); 0 without a tolerance."""
    if xy_tol <= 0.0 and yaw_tol_deg <= 0.0:
        return 0.0
    v = object_vertices_world(scene, pose)
    radius = float(np.linalg.norm(v[:, :2] - np.asarray(pose)[:2, 3], axis=1).max()) if len(v) else 0.0
    return float(np.sqrt(2.0) * max(xy_tol, 0.0) + 2.0 * radius * np.sin(np.radians(max(yaw_tol_deg, 0.0)) / 2.0))


def object_criteria(sim: dict, ref_start: np.ndarray, ref_end: np.ndarray, table_z: float) -> dict:
    """The replay's object criteria on a simulated run's playback (module docstring): `{"reference_displacement_m",
    "displacement_m", "final_error_m", "fell"}`, None where the run never played."""
    ob = sim.get("object") or {}
    p0, p1, pe = ob.get("play_start_m"), ob.get("play_end_m"), ob.get("run_end_m")
    ref = float(np.linalg.norm(np.asarray(ref_end)[:3, 3] - np.asarray(ref_start)[:3, 3]))
    if p0 is None or p1 is None:
        return {"reference_displacement_m": ref, "displacement_m": None, "final_error_m": None, "fell": None}
    p0, p1 = np.asarray(p0, dtype=np.float64), np.asarray(p1, dtype=np.float64)
    low = min(float(p1[2]), float(pe[2]) if pe is not None else float(p1[2]))
    return {"reference_displacement_m": ref, "displacement_m": float(np.linalg.norm(p1 - p0)),
            "final_error_m": float(np.linalg.norm(p1 - np.asarray(ref_end)[:3, 3])),
            "fell": bool(low < table_z - FALL_M)}


def _clip(base: np.ndarray, off: np.ndarray, rng: np.ndarray, kind: str) -> tuple[np.ndarray, list[str]]:
    """`base + off` clipped to the `(n,2)` ranges `rng`: `(offset actually applied, clipped joints as "<kind> <k>")`."""
    want = base + off
    got = np.clip(want, rng[:, 0], rng[:, 1])
    clipped = [f"{kind} {k}" for k in np.flatnonzero(np.abs(got - want) > 1e-12)]
    return got - base, clipped


def state_starts(home: np.ndarray, hand0: np.ndarray, home_tol: float, arm_range: np.ndarray,
                 hand_range: np.ndarray, corners: int, edge: float, seed: int) -> list[dict]:
    """The measured-state starts of one home (module docstring), named as `g1_sdk_replay.state_run_names(corners)`:
    `{"name", "start_q" (17,), "hand0" (7,), "arm_offset" (14,), "hand_offset" (7,), "clipped"}`. `arm_range` (14,2):
    the right arm's command limits and the left arm's joint ranges; `hand_range` (7,2): the Dex3 joint ranges."""
    home = np.asarray(home, dtype=np.float64)
    hand0 = np.asarray(hand0, dtype=np.float64)
    arm0 = home[:14]
    offsets: list[tuple[str, np.ndarray, np.ndarray]] = []
    for j in range(14):
        for sign, tag in ((1.0, "+"), (-1.0, "-")):
            off = np.zeros(14)
            off[j] = sign * edge * home_tol
            offsets.append((f"arm_axis_{j:02d}{tag}", off, np.zeros(7)))
    signs = np.random.default_rng(seed).choice([-1.0, 1.0], size=(corners, 14))
    offsets += [(f"arm_corner_{i}", edge * home_tol * signs[i], np.zeros(7)) for i in range(corners)]
    offsets += [(f"hand{tag}", np.zeros(14), sign * edge * S.START_HAND_TOL * np.ones(7))
                for sign, tag in ((1.0, "+"), (-1.0, "-"))]
    out = []
    for name, arm_off, hand_off in offsets:
        a, ca = _clip(arm0, arm_off, np.asarray(arm_range, dtype=np.float64), "arm")
        h, ch = _clip(hand0, hand_off, np.asarray(hand_range, dtype=np.float64), "hand")
        out.append({"name": name, "start_q": np.concatenate([arm0 + a, home[14:17]]), "hand0": hand0 + h,
                    "arm_offset": a, "hand_offset": h, "clipped": ca + ch})
    assert [o["name"] for o in out] == S.state_run_names(corners)
    return out
