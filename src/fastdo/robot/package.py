"""Self-contained robot package writer: what the lab copies to the robot PC and runs.

Folder layout (`write_package`):
- `README.md`            what the package is, how to place the object, how to run it, what was and was NOT tested;
- `trajectory.npz`       every stream the run script needs (`g1_sdk_replay.Plan` fields: the trajectory, the checked
                          homes and the collision-checked paths between them and the trajectory, `fastdo.robot.transfer`)
                          plus velocities, joint names;
- `arm_250hz.csv`        right arm: t, q (7), dq (7), tau_ff (7), `RIGHT_ARM_JOINTS` order;
- `hand_100hz.csv`       right Dex3-1: t, q (7), the enum order assumed by this package (`RIGHT_HAND_JOINTS`);
- `sim_stats.json`       the kinematic-path summary: IK and retargeting residuals, reach margins, every adaptation,
                          MuJoCo replay metrics, the simulated runs of the run script from each checked home
                          (`transfers`), and the sha256 of `trajectory.npz` they were computed on (the run script's
                          execution gate);
- `placement_search.json` every placement candidate the search evaluated (choice, physical checks, score), when the
                          placement was searched;
- `package.json`         provenance (clip, source directory, fastdo commit, pinned Unitree sources), the object start
                          pose in the robot frame and the execution gate (`sim_gate`: thresholds, verdict, failures);
- `object_mesh.obj`      the reconstructed object mesh in METRES in its own frame (`package.json` places it);
- `run_g1_replay.py`     a verbatim copy of `fastdo.robot.g1_sdk_replay` (dry-run by default);
- `NOTICE.md`            the official sources the numbers come from and their licences.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from fastdo.robot import g1_constants as C
from fastdo.robot import g1_sdk_replay
from fastdo.robot.g1_model import UNITREE_ROS_SHA, UNITREE_ROS_XML, UNITREE_ROS_XML_SHA256
from fastdo.robot.kinematic_path import KinematicPathResult, Reconstruction
from fastdo.robot.trajectory import TrajectoryLimits
from fastdo.robot.transfer import TABLE_TOL_M

__all__ = ["DRAFT_MARKER", "PACKAGE_FILES", "TABLE_PLACEMENT_TOL_M", "load_run_script", "sim_gate", "stray_files",
           "write_package"]

TABLE_PLACEMENT_TOL_M = TABLE_TOL_M   # default; a package states the tolerance its moves were planned and simulated at
# Every package is written as a DRAFT: only an independent safety review that returns
# safe_to_relay = true, plus the lab operator's sign-off on the motion, may clear it; nothing here clears it.
DRAFT_MARKER = ("DRAFT: NOT lab-ready. Not reviewed for the lab: nothing may run on a robot until an independent "
                "safety review returns safe_to_relay = true AND the lab operator has signed off on the motion.")
PACKAGE_FILES = frozenset({"README.md", "trajectory.npz", "arm_250hz.csv", "hand_100hz.csv", "sim_stats.json",
                           "package.json", "object_mesh.obj", "run_g1_replay.py", "NOTICE.md", "placement_search.json"})

_SOURCES = (
    ("unitree_ros", "github.com/unitreerobotics/unitree_ros", UNITREE_ROS_SHA, "BSD-3-Clause",
     f"model `{UNITREE_ROS_XML}` (sha256 {UNITREE_ROS_XML_SHA256}), joint/velocity/torque limits (URDF)"),
    ("xr_teleoperate", "github.com/unitreerobotics/xr_teleoperate", "817fb00c63cde15e5f24a0f8fa08e1e33ed89d3b",
     "Apache-2.0", ("arm/hand gains and rates, rt/arm_sdk weight blending, the Dex3 enum order assumed by this package "
                    "(its mapping to the physical fingers is not verified), DexPilot config, "
                    "Dex3 retargeting frames")),
    ("unitree_sdk2_python", "github.com/unitreerobotics/unitree_sdk2_python", "814556d15970dd2ecf1c9984e845ca02ab07e206",
     "BSD-3-Clause", "DDS API, LowCmd_/HandCmd_ message layout, CRC, arm7 arm_sdk example (joint indices, ramp)"),
    ("dex-retargeting (Unitree fork)", "github.com/silencht/dex-retargeting", "d7753d38c9ff11f80bafea6cd168351fd3db9b0e",
     "MIT", "DexPilot objective (re-implemented, not copied)"),
)


def _git_commit() -> str:
    root = Path(__file__).resolve().parents[3]
    try:
        head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True, timeout=30).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--", "src", "run"],
                               capture_output=True, text=True, check=True, timeout=30).stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return "UNKNOWN"
    return head + ("+dirty" if dirty else "")


def _write_csv(path: Path, header: list[str], rows: np.ndarray) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow([f"{v:.6f}" for v in row])


def _json(x: object) -> object:
    if isinstance(x, dict):
        return {str(k): _json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json(v) for v in x]
    if isinstance(x, np.ndarray):
        return _json(x.tolist())
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def stray_files(package: Path) -> list[str]:
    """Names in the package folder that `write_package` does not write (e.g. a `__pycache__` left by importing the run
    script): a package is relayed with none."""
    return sorted(p.name for p in Path(package).iterdir() if p.name not in PACKAGE_FILES)


def load_run_script(package: Path, name: str = "run_g1_replay_pkg") -> ModuleType:
    """Import a package's `run_g1_replay.py` WITHOUT writing `__pycache__` into the package (a
    check that imported the run script left a stray `__pycache__` in every draft)."""
    path = Path(package) / "run_g1_replay.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    before = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        sys.modules[name] = module
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = before
    return module


def sim_gate(package: Path) -> dict:
    """The run script's execution gate on a written package: `{"thresholds", "execute_allowed", "failures",
    "homes_allowed", "home_failures"}` (`g1_sdk_replay.gate_report`)."""
    rep = g1_sdk_replay.gate_report(package)
    return {"thresholds": dict(g1_sdk_replay.SIM_GATE), "notes": dict(g1_sdk_replay.SIM_GATE_NOTES),
            "execute_allowed": not rep["failures"], "failures": rep["failures"], "homes_allowed": rep["homes_allowed"],
            "home_failures": rep["home_failures"], "simulated_run_options": rep["simulated"]}


def _gate_readme(gate: dict) -> str:
    g = gate["thresholds"]
    verdict = ("**This package passes: `--execute` is allowed.**" if gate["execute_allowed"] else
               "**`--execute` REFUSES this package:**\n" + "\n".join(f"- {f}" for f in gate["failures"]))
    return f"""## Execution gate
`--execute` REFUSES a package unless its simulation (below; computed on this exact `trajectory.npz`, whose sha256 is
in `sim_stats.json`) meets every registered threshold (`SIM_GATE` in `run_g1_replay.py`, set by its safety
reviews). The trajectory replay:
- no robot-table penetration deeper than {g['table_max_depth_m'] * 1000:.1f} mm and no robot-table normal force above
  {g['table_max_force_n']:g} N (the simulated depth under-reports a forceful contact; the {g['table_max_force_n']:g} N
  cap is {_note(gate, 'table_max_force_n')});
- no self-collision deeper than {g['self_collision_max_depth_m'] * 1000:.1f} mm;
- {g['limit_violation_ticks']} joint-limit violation ticks;
- right-arm tracking max below {g['arm_tracking_max_rad']:g} rad (the runtime abort is 0.3 rad);
- the reference moves the object at least {g['min_reference_displacement_m'] * 100:.1f} cm, the simulated object moves
  at least {g['min_move_fraction']:g} x the reference, ends within max({g['final_error_floor_m'] * 100:.1f} cm,
  {g['max_final_error_fraction']:g} x the reference) of the reference end, and does not fall;
- the object rests where it is placed: while it settles before the robot moves it moves at most
  {g['settle_drift_max_m'] * 1000:.1f} mm and turns at most {g['settle_rot_max_deg']:g} deg.
AND, for at least one checked home ("Moves to the hold pose and home" below), the run script itself simulated from
that home: outside the playback no robot-table contact ({g['transfer_table_contact_steps']} physics steps), no
self-collision deeper than {g['transfer_self_collision_max_depth_m'] * 1000:.1f} mm, no robot-object contact
({g['transfer_object_contact_steps']} steps), tracking below {g['transfer_arm_tracking_max_rad']:g} rad; in every
operator-stop drill the way home stays within {g['stop_table_max_depth_m'] * 1000:.1f} mm of table and
{g['stop_self_collision_max_depth_m'] * 1000:.1f} mm of self contact and {g['stop_table_max_force_n']:g} N of table
force; and the same run on a table at both edges of the placement tolerance ("Object placement": top raised by it with
every edge moved out by it, and top lowered by it, the object standing on it) meets the thresholds above outside and
inside the playback; the run itself, simulated at the package's playback speed scale, moves the object as the
reference does (the object thresholds above, read on its playback); the same run with the object placed at
each of the 8 corners of its placement tolerance (+-x, +-y, +-yaw; "Object placement") meets the table-tolerance
thresholds and the object thresholds against the nominal reference; and the same run from the measured states the
script accepts (every axis extreme and {g['state_corner_runs_min']} seeded corners of the arm box around the home, and
the Dex3 hand at +-{g['start_hand_tol_rad']:g} rad on every joint, each at least {g['state_edge_min']:g} of its tolerance
out) meets the table-tolerance thresholds. Every simulated run records the run options it used, and they must be the
package's. `--execute` then starts only from a home that passed, and only with the run options the simulation used:
{_options(gate)}. The dry-run reports the verdict too (`sim_gate` in its JSON).

{verdict}
"""


def _note(gate: dict, key: str) -> str:
    return (gate.get("notes") or {}).get(key) or "unlabelled"


def _options(gate: dict) -> str:
    o = gate.get("simulated_run_options") or {}
    ff = {True: "on", False: "off"}.get(o.get("send_tau_ff"), "unknown")
    hs = "unknown" if o.get("hold_speed") is None else f"{o['hold_speed']:g} rad/s"
    sc = "unknown" if o.get("speed_scale") is None else f"{o['speed_scale']:g}"
    mte = "unknown" if o.get("max_track_err") is None else f"{o['max_track_err']:g} rad"
    hto = "unknown" if o.get("hold_timeout_s") is None else f"{o['hold_timeout_s']:g} s"
    return (f"`--hold-speed` {hs}, `--speed-scale` {sc} (the default), gravity feed-forward {ff} (no `--no-tau-ff`), "
            f"the waist commanded to 0 (no `--no-waist`), `--max-track-err` at most {mte} and `--hold-timeout` at most "
            f"{hto} (the simulated defaults)")


def _start_hand(pkg: dict) -> str:
    hand = (pkg["sim_gate"].get("simulated_run_options") or {}).get("start_hand")
    return "unknown (not recorded: --execute refuses)" if hand is None else str([round(v, 3) for v in hand])


def _speed_readme(pkg: dict) -> str:
    sc = (pkg["sim_gate"].get("simulated_run_options") or {}).get("speed_scale")
    if sc is None:
        return "The simulated playback speed scale is not recorded: `--execute` refuses the package."
    first = (" This is a **bring-up package**: the first lab run of a motion plays at 0.5, and a "
             "1.0 package is exported only after a clean 0.5 run on the robot." if sc < 1.0 else
             " Export a 1.0 package only after a clean 0.5 run of the same motion on the robot.")
    return (f"**Playback speed scale {sc:g}** (`--speed-scale`; the default is this simulated value, any other is "
            f"refused with `--execute`): the trajectory below plays {1.0 / sc:.2f}x slower than its own timing." + first)


def _mm(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 1000:.1f}"


def _obj_tol(pkg: dict) -> tuple[float, float]:
    o = pkg.get("object_placement_tolerance") or {}
    return float(o.get("xy_m", float("nan"))), float(o.get("yaw_deg", float("nan")))


def _readme(result: KinematicPathResult, pkg: dict, stats: dict) -> str:
    obj = pkg["object_start_robot"]
    tbl = pkg["table"]
    rp = stats["replay"]
    pos = ", ".join(f"{v:.3f}" for v in obj["position_m"])
    rpy = ", ".join(f"{v:.1f}" for v in obj["rpy_deg"])
    failed = ", ".join(rp["failed_criteria"]) or "none"
    warn = "\n".join(f"- {w}" for w in stats["warnings"]) or "- none"
    adapt = "\n".join(f"- {a}" for a in stats["adaptations"]) or "- none"
    rm = stats["reach_margin"]
    tc = stats.get("table_check") or {}
    return f"""# G1 + Dex3-1 replay package: `{result.clip_id}`

**{DRAFT_MARKER}**

Kinematic baseline: Unitree's DexPilot-style retargeting of the right hand plus damped-least-squares
wrist IK of an in-the-wild clip reconstructed by our pipeline. No physics was used to produce the motion.

**NOT TESTED ON HARDWARE.** The run script has only run against its own simulated I/O (the default dry-run). The trajectory
has only been replayed in MuJoCo on the official Unitree model with a fixed base. Treat the first run as a hardware
bring-up: e-stop in hand, robot standing in Regular mode, nobody in the arm's reach.

## Warnings (from the export; read before running)
{warn}

{_gate_readme(pkg["sim_gate"])}
{_transfers_readme(stats, pkg)}
## Robot
G1 29-DoF with 4010 wrist motors (`mode_machine` 5 or 6; the script refuses any other value) and a right Dex3-1.
Upper body only; the base must not move. Left arm: parked at `{list(C.LEFT_ARM_PARK)}` rad. Left hand: never
commanded. Waist: all three waist entries are sent in every `rt/arm_sdk` message (as Unitree's arm7 SDK example),
commanded to 0 (with `mode_machine` 6, the lock-waist variant, only the yaw goes to 0 and roll / pitch are held at their
measured start values). The simulation welds the waist at 0, so the script REFUSES to start (nothing sent) when any
measured waist joint (roll, pitch, yaw) is more than {pkg['sim_gate']['thresholds']['waist_tol_rad']:g} rad from 0, and
`--execute` refuses `--no-waist` (a 0.08 rad waist offset moved an earlier draft's wrist 28-34 mm).
**Right hand (Dex3-1) at the start:** the run was simulated from the hand at
{_start_hand(pkg)} rad (the enum order assumed by this package; its mapping to the physical index and middle fingers
is NOT verified on hardware); with `--execute` the script REFUSES to start (nothing sent) when any measured
Dex3 joint is more than {pkg['sim_gate']['thresholds']['start_hand_tol_rad']:g} rad from it, and prints the measured
values: pose the hand there and retry.

**Measured start state:** the run was also simulated from the measured states the script accepts: every axis extreme
and seeded corners of the {pkg['home_tol_rad']:g} rad arm box around the home, and the Dex3 hand at
+-{pkg['sim_gate']['thresholds']['start_hand_tol_rad']:g} rad on every joint (joints at their range limit stay at it).
The waist is NOT simulated off 0 (the model welds it): its {pkg['sim_gate']['thresholds']['waist_tol_rad']:g} rad
tolerance rests on a lever argument, not a simulation (0.01 rad moves the wrist at most 4.2 mm on the largest
lever of the draft packages, 0.421 m; not recomputed for this package).

{_start_gravity_readme(pkg)}
## Adaptations (everything that is not the human's recorded motion)
{adapt}

## Object placement (robot frame)
Robot frame: origin on the floor below the pelvis origin, x forward, y left, z up (pelvis origin at z = 0.793 m when
standing). **Table top at z = {pkg['table_z']:.3f} m** (set the lab table to this height,
+-{tbl['placement_tolerance_m'] * 100:.1f} cm, or re-export).
**Table footprint** (the rectangle the simulation and the execution gate assume; place the table within
+-{tbl['placement_tolerance_m'] * 100:.1f} cm on every edge): near edge (facing the robot) at x = {tbl['near_edge_x_m']:.3f} m, far
edge at x = {tbl['far_edge_x_m']:.3f} m, right edge at y = {tbl['right_edge_y_m']:.3f} m, left edge at
y = {tbl['left_edge_y_m']:.3f} m; top {tbl['thickness_m'] * 100:.1f} cm thick. Nothing may be closer to the robot than the near
edge, stick out of the rectangle, or rise above the top (other than the object). The moves to the hold pose and home
are planned as if the table were solid down to the floor, so legs and frame may be anywhere under the rectangle; the
trajectory replay simulates the top only. `run_g1_replay.py --execute` refuses a package whose stated table differs
from the simulated one. Put
the object (mesh `object_mesh.obj`, metres) at position [{pos}] m, roll/pitch/yaw [{rpy}] deg (extrinsic xyz). Object
extents (mesh frame): {pkg['object_extent_m']} m.
**Object placement tolerance: +-{_obj_tol(pkg)[0] * 100:.1f} cm in x and y, +-{_obj_tol(pkg)[1]:g} deg in yaw** (about
the vertical through the object's origin), and its height moves with the table
(+-{tbl['placement_tolerance_m'] * 100:.1f} cm). The moves to the hold pose and home keep >= 2 cm from the object
anywhere inside that tolerance (the planner reduces its clearance by
{_mm(pkg.get('object_placement_tolerance', {}).get('planner_envelope_m'))} mm, the farthest a point of the object moves
under it), and the whole run was simulated with the object at each of the 8 corners (+-x, +-y, +-yaw; the execution
gate above). An offset beyond the tolerance is untested: re-export instead.
For another measured start pose, re-export with the same choices and the measured position:
`{pkg['reexport_command']}` (replace X Y / Z by the measured values).

## Run
```
python run_g1_replay.py                              # dry-run: validate + simulated I/O (default)
python run_g1_replay.py --execute --iface <nic>      # robot; asks 'yes' to connect and 'go' at the hold pose
```
{_speed_readme(pkg)}
Sequence (one control loop publishes `rt/arm_sdk` every 4 ms and the hand every 10 ms in EVERY phase, also while
waiting for 'go'): refuse unless the arms and waist are at a checked home (above); ramp the weight 0 -> 1 holding the
measured pose (HOME); follow that home's checked path to the hold pose (trajectory start, left park, waist 0, right
hand at its start) at <= 0.3 rad/s on every joint; wait for 'go' (`--hold-timeout`, default 120 s, without an answer
= stop); play the arm at 250 Hz and the hand at 100 Hz; follow the checked path back to HOME at <= 0.3 rad/s (the
hand keeps its last command); ramp the weight to 0 over 2 s.
Aborts on stale `rt/lowstate` or Dex3 state (> 0.1 s), a mode_machine change, right-arm tracking error > 0.3 rad
(`--max-track-err`), any arm or hand command outside the limits or above the velocity limits stored in
`trajectory.npz`, a left-arm/waist command above 0.5 rad/s, or any exception. An abort holds the MEASURED pose (the last
command when the state is stale) and ramps the weight to 0 over 2 s, publishing every tick.
Operator stop: Ctrl-C (or SIGTERM, or any answer other than 'go') RETRACES the commands sent so far back to HOME
(hand included, <= 0.3 rad/s), then ramps the weight down; a second Ctrl-C during the way home stops there and ramps
the weight down. An ABORT (or a second Ctrl-C) releases `rt/arm_sdk` where the arm is: Unitree's standing controller
then takes the arms back to ITS own arm pose over the 2 s ramp, along a path nothing here checks (as with
xr_teleoperate). Keep the e-stop in hand regardless.
With `--execute` only the run options the package simulated are accepted ({_options(pkg['sim_gate'])}); another
`--hold-speed`, `--speed-scale`, `--no-tau-ff` or `--no-waist` is refused before anything connects (they change the
dry-run only; re-export to simulate another speed scale with the export's `--speed-scale`).

## Streams
- `arm_250hz.csv`: {len(result.trajectory.arm_t)} samples, {result.trajectory.duration:.2f} s; joints {list(C.RIGHT_ARM_JOINTS)}.
- `hand_100hz.csv`: {len(result.trajectory.hand_t)} samples; columns in the enum order assumed by this
  package, {list(C.RIGHT_HAND_JOINTS)}.
  The upstream sources disagree on the right-hand index and middle order, and the mapping of these columns to the
  physical fingers is NOT verified on hardware: establish it on the robot, with an independent physical finger label,
  before any hand command.
- Played {result.trajectory.time_scale:.2f}x slower than the video (one uniform time scale from the velocity and acceleration limits; see `package.json`).

## Reach (collision-aware IK of the exported path, every used row)
Wrist position residual median {stats['ik_pos_err_m'].get('median', float('nan')) * 1000:.1f} mm, max
{stats['ik_pos_err_m'].get('max', float('nan')) * 1000:.1f} mm; rotation residual median
{stats['ik_rot_err_rad'].get('median', float('nan')):.3f} rad, max {stats['ik_rot_err_rad'].get('max', float('nan')):.3f} rad.
Reach margin (`fastdo.robot.reach_eval`; < 0 = a row out of reach by > 1 cm or > 0.1 rad, or through the table):
min {rm['min']:.2f}, {rm['infeasible_rows']} of {stats['used_rows']} rows infeasible (rows and causes in
`sim_stats.json` `table_check.infeasible`). Table consistency of the placed task: the human hand reaches
{tc.get('hand_below_table_m', float('nan')) * 100:.1f} cm and the trusted reference object
{tc.get('object_below_table_m', float('nan')) * 100:.1f} cm below the table top where over it
({tc.get('table_bad_rows', 0)} rows beyond tolerance). Stable start (centre of mass at least 5 mm inside the support
polygon of the object's lowest hull points): {tc.get('rest_stable')} (inside by
{_mm(tc.get('rest_com_inside_m'))} mm); the replay's settle moved the object
{rp['settle_drift'] * 100:.1f} cm before the robot started.

## Simulation (MuJoCo, CPU, official model, fixed base)
Arm tracking RMS {rp['arm_rms']:.4f} rad, max {rp['arm_max']:.4f} rad; hand RMS {rp['hand_rms']:.4f} rad.
Self-collision steps {rp['self_collision_steps']} (max depth {rp['self_collision_max_depth'] * 1000:.1f} mm); table
max depth {rp['table_max_depth'] * 1000:.1f} mm; joint-limit violation ticks {rp['limit_violation_ticks']}.
Robot-object contact in {rp['object_contact_steps']} physics steps (2 ms). Object moved
{rp['object_displacement'] * 100:.1f} cm (reference {rp['reference_displacement'] * 100:.1f} cm, from the contact-onset
anchor, clip row {stats['anchor_clip_row']}), final error {rp['object_final_error'] * 100:.1f} cm. **Success: {rp['success']}** (failed: {failed}).
Object mass in sim is a placeholder ({pkg['object_mass_kg']} kg); the object is collided as its convex hull.

## Known gaps
- Torso sway of Unitree's standing controller is not modelled (the sim welds the pelvis).
- Dex3 finger dynamics (armature, friction) are not in any official file; sim fingers use the official kp/kd only.
- `rt/dex3/right/state` vs `rt/lf/dex3/right/state`: the SDK C++ example uses the latter; UNVERIFIED which the robot
  publishes. If the script refuses with "no robot state after 5 s", re-run with
  `--hand-state-topic rt/lf/dex3/right/state`.

See `NOTICE.md` for the official sources and licences, `package.json` for provenance.
"""


def _start_gravity_readme(pkg: dict) -> str:
    """Where the object's start pose and the up direction come from."""
    sp, gr = pkg.get("start_pose") or {}, pkg.get("gravity") or {}
    if "kind" in sp:
        snap = sp.get("snap") or {}
        how = (f"SNAPPED onto the accepted support plane: face {snap.get('face_index')} (landing probability "
               f"{snap.get('face_probability')}), turned {snap.get('rot_deg')} deg, centroid moved {snap.get('disp_m')} m "
               f"(caps {sp.get('caps')})") if sp["kind"] == "snapped" else "consolidated over the rest rows, NOT snapped"
        start = (f"the start-pose sidecar `{sp['path']}` (sha256 {sp['sha256'][:12]}, source "
                 f"{sp['source']}; {how}; ruled rest verdict p7_ruled_pass {sp.get('p7_ruled_pass')} under "
                 f"{sp.get('amendment')}). It replaces the contact-onset row's pose (source frame {sp['c0_frame_idx']}; "
                 f"the start pose is {sp['correction_shift_m'] * 100:.1f} cm / {sp['correction_rot_deg']:.1f} deg from "
                 "it); every row from the onset on is moved by the same rigid correction, so the grasp keeps its "
                 "hand-object relation")
    else:
        start = sp.get("source", "unknown")
    pl = gr.get("support_plane")
    if gr.get("source") == "support_plane":
        up = (f"the ESTIMATED support plane `{pl['path']}` (sha256 {pl['sha256'][:12]}; {pl['rows']} rest rows, "
              f"normal sd {pl['normal_sd_deg']} deg; accepted under {pl['rule']}); the head pitch and the resting "
              "face were NOT searched")
    else:
        up = gr.get("source", "unknown") + ("" if pl is None else
                                            f"; support plane `{pl['path']}` REFUSED: {'; '.join(pl['refusal'])}")
    return f"""## Start pose and gravity
- **Object start pose:** {start}.
- **Up (gravity):** {up}.
"""


def _transfers_readme(stats: dict, pkg: dict) -> str:
    """The README section on the checked homes and the moves between them and the trajectory."""
    tr = stats.get("transfers") or {"homes": [], "home_tol_rad": float("nan")}
    allowed = set(pkg["sim_gate"].get("homes_allowed", []))
    lines = []
    for rec in tr["homes"]:
        th, gh = rec["to_hold_plan"], rec["go_home_plan"]

        def plan_cell(pl: dict) -> str:
            if not pl["ok"]:
                return f"NO: {pl['reason']}"
            return (f"{pl['waypoints']} waypoints ({'/'.join(pl['methods'])}), clearance table "
                    f"{_mm(pl['min_table_m'])} / object {_mm(pl['min_object_m'])} mm")
        run = rec.get("run")
        if run is None:
            sim = "not simulated"
        else:
            ph = run["phases"]
            table = sum(v["table_contact_steps"] for k, v in ph.items() if k != "play")
            touch = sum(v["object_contact_steps"] for k, v in ph.items() if k != "play")
            sim = f"{run['status']}; outside the playback: table contact {table}, object contact {touch} steps"
            if rec.get("drills"):
                sim += "; stops: " + ", ".join(f"{d['stop']} {d['status']}" for d in rec["drills"])
            for tr_run in rec.get("table_tolerance_runs") or []:
                tph = tr_run["phases"]
                steps = sum(v["table_contact_steps"] for k, v in tph.items() if k != "play")
                sim += (f"; {tr_run['case']} ({tr_run['dz_m'] * 100:+.1f} cm): {tr_run['status']}, table contact "
                        f"{steps} steps outside the playback")
            for key, what in (("placement_runs", "object placement corners"), ("state_runs", "measured states")):
                runs = rec.get(key) or []
                done = sum(r.get("status") == "completed" for r in runs)
                sim += f"; {what}: {done} of {len(runs)} completed"
        ok = "PASSES" if rec.get("index") is not None and rec["index"] in allowed else "refused"
        q = ", ".join(f"{v:.2f}" for v in rec["home_q"])
        lines.append(f"| `{rec['name']}` | [{q}] | {plan_cell(th)} | {plan_cell(gh)} | {sim} | {ok} |")
    table = "\n".join(lines) or "| (none) | | | | | |"
    fails = "\n".join(f"- {f}" for f in pkg["sim_gate"].get("home_failures", [])) or "- none"
    return f"""## Moves to the hold pose and home (checked)
The run script starts only when the measured arms and waist are within {tr['home_tol_rad']:g} rad (every joint) of a
CHECKED home that passes the execution gate, and moves only along the paths planned and simulated for that home
(`fastdo.robot.transfer`, waypoints in `trajectory.npz`): at HOME the right hand shapes, the left arm parks, then
the right arm goes to the trajectory start; after the playback the right arm, then the left arm, go back along the
checked path; after an operator stop the script retraces its own commands back to HOME. Paths are planned in this
package's scene at >= 2 cm from the table and the object (object at its start pose; at its simulated and reference
end poses for the way home; the table grown by the placement tolerance on its top and every edge, the object
raised and lowered by it), with every corner of the {tr['home_tol_rad']:g} rad box around the home checked too; then
the run script itself is simulated from the home, with operator stops half way through TO_HOLD and at 25 / 50 / 75 % of
the playback, and again on a table at both edges of the placement tolerance. `zeros` is the official arm_sdk home (xr_teleoperate's go-home target, the SDK arm7 example's
zero posture); `arms_hanging` is our choice. The arm pose Unitree's standing controller holds is in no official file
(UNVERIFIED): if the script refuses with "the measured arms / waist are ... from the nearest checked home", its message
lists the 17 measured joints; re-export with `--home-q` and those values (the export plans and simulates that home).

| Home | Joints (right arm 7, left arm 7, waist 3; rad) | Path to the hold pose | Path home | Simulated run | Gate |
|---|---|---|---|---|---|
{table}

Why homes are refused:
{fails}
"""


def _reexport_command(result: KinematicPathResult, recon: Reconstruction) -> str:
    """The export command that reproduces this package's choices at a measured object position X Y / table Z."""
    ch = result.choice
    parts = ["python run/robot/export_g1_package.py", f"'{recon.source_dir}'", "--out <dir>"]
    if recon.ego is not None:
        parts.append(f"--ego-motion '{recon.ego.path}'")
    if result.first_row:
        parts.append(f"--first-row {result.first_row}")
    if not result.trim_to_contact:
        parts.append("--no-trim")
    parts += ["--object-xy X Y --table-z Z", f"--yaw-offset-deg {ch.yaw_deg:g}"]
    if ch.pitch_deg is not None:
        parts.append(f"--camera-pitch-deg {ch.pitch_deg:g}")
    if ch.up_face is not None:
        parts.append(f"--up-face {ch.up_face}")
    if ch.sym_axis is not None and ch.sym_deg % 360.0:
        parts.append(f"--sym-axis {ch.sym_axis} --sym-deg {ch.sym_deg:g}")
    if result.start_pose is not None:
        parts.append(f"--start-pose '{result.start_pose['path']}'")
    if result.gravity is not None and result.gravity.get("support_plane") is not None:
        parts.append(f"--support-plane '{result.gravity['support_plane']['path']}'")
    for rec in (result.transfers.homes if result.transfers is not None else []):
        if rec["name"] == "measured":
            parts.append("--home-q " + " ".join(f"{v:g}" for v in rec["home_q"]))
    if result.transfers is not None and result.transfers.speed_scale != 1.0:
        parts.append(f"--speed-scale {result.transfers.speed_scale:g}")
    if result.approach is not None:
        parts.append("--approach palm")
    if result.closure is not None:
        parts.append(f"--grasp-closure --closure-target-m {result.closure['target_m']:g}")
    return " ".join(parts)


def write_package(out_dir: str | Path, result: KinematicPathResult, recon: Reconstruction,
                  limits: TrajectoryLimits | None = None, object_mass: float = 0.2) -> Path:
    """Write the package folder (module docstring) and return its path. Refuses to overwrite a non-empty folder."""
    limits = TrajectoryLimits() if limits is None else limits
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} exists and is not empty")
    if result.transfers is None:
        raise ValueError("the kinematic path ran without the transfer check (transfer.enabled False): a package "
                         "needs the checked moves to the hold pose and home")
    fields = result.transfers.fields
    if not (np.array_equal(fields["arm_limits"], limits.arm_bounds())
            and np.array_equal(fields["arm_vmax"], np.asarray(limits.arm_vmax, dtype=np.float64))
            and np.array_equal(fields["hand_vmax"], np.asarray(limits.hand_vmax, dtype=np.float64))):
        raise ValueError("the package limits differ from the ones the kinematic path (and its simulation) used")
    out.mkdir(parents=True, exist_ok=True)
    traj = result.trajectory
    np.savez(out / "trajectory.npz", **fields, arm_dq=traj.arm_dq, hand_dq=traj.hand_dq,
             arm_joints=np.asarray(C.RIGHT_ARM_JOINTS), hand_joints=np.asarray(C.RIGHT_HAND_JOINTS))
    arm_cols = list(C.RIGHT_ARM_JOINTS)
    _write_csv(out / "arm_250hz.csv", ["t"] + [f"q_{j}" for j in arm_cols] + [f"dq_{j}" for j in arm_cols]
               + [f"tau_ff_{j}" for j in arm_cols],
               np.column_stack([traj.arm_t, traj.arm_q, traj.arm_dq, traj.arm_tau_ff]))
    _write_csv(out / "hand_100hz.csv", ["t"] + [f"q_{j}" for j in C.RIGHT_HAND_JOINTS],
               np.column_stack([traj.hand_t, traj.hand_q]))
    stats = _json(result.summary())
    assert isinstance(stats, dict)
    stats["trajectory_sha256"] = g1_sdk_replay.file_sha256(out / "trajectory.npz")   # the gate checks it
    (out / "sim_stats.json").write_text(json.dumps(stats, indent=1))
    rec = recon.record
    mesh = trimesh.Trimesh(recon.mesh_vertices * rec.object_scale, recon.mesh_faces, process=False)
    mesh.export(out / "object_mesh.obj")
    start = result.placement.object_start
    pkg = {
        "lab_status": DRAFT_MARKER,
        "clip_id": rec.clip_id, "source_dir": str(recon.source_dir), "created_utc": datetime.now(UTC).isoformat(),
        "fastdo_commit": _git_commit(),
        "unitree_ros": {"sha": UNITREE_ROS_SHA, "model": UNITREE_ROS_XML, "model_sha256": UNITREE_ROS_XML_SHA256},
        "robot": "G1 29-DoF (mode_machine 5/6, 4010 wrists) + right Dex3-1; fixed base; left arm parked",
        "table_z": result.placement.table_z, "table_xy": list(result.placement.table_xy),
        "table": {**(result.table or {}), "placement_tolerance_m": result.transfers.table_tol,
                  "frame": "robot: origin on the floor below the pelvis origin, x forward, y left, z up"},
        "object_placement_tolerance": {"xy_m": result.transfers.object_xy_tol,
                                       "yaw_deg": result.transfers.object_yaw_tol_deg,
                                       "planner_envelope_m": result.transfers.object_envelope_m,
                                       "yaw_axis": "the vertical through the object's origin"},
        "object_start_robot": {"position_m": start[:3, 3].tolist(),
                               "quat_wxyz": np.roll(Rotation.from_matrix(start[:3, :3]).as_quat(), 1).tolist(),
                               "rpy_deg": Rotation.from_matrix(start[:3, :3]).as_euler("xyz", degrees=True).tolist()},
        "object_extent_m": np.round(np.ptp(np.asarray(mesh.vertices), axis=0), 4).tolist(),
        "object_scale_m_per_unit": rec.object_scale, "object_mass_kg": object_mass,
        "object_mass_note": "ASSUMED, not weighed: the simulation (S3) runs at this mass until the lab weighs the "
                            "object; the package is then re-simulated at the weighed mass",
        "approach": result.approach if result.approach is not None else
        {"mode": "vertical", "height_m": "approach_m of the export config"},
        "grasp_closure": result.closure,
        "T_robot_from_video_world": result.placement.T_rv.tolist(), "yaw_rad": result.placement.yaw,
        "time_scale": traj.time_scale, "hold_s": traj.hold_s, "warnings": list(result.warnings),
        "camera_source": result.camera_source, "gravity_source": result.gravity_source, "up_source": result.up_source,
        "anchor_row": result.anchor_row, "row_range": list(result.row_range), "choice": result.choice.to_dict(),
        "adaptations": list(result.adaptations), "ego_motion": result.ego,
        "start_pose": result.start_pose if result.start_pose is not None else
        {"source": "the contact-onset row's reconstructed pose (no start-pose sidecar)"},
        "gravity": result.gravity,
        "reexport_command": _reexport_command(result, recon),
        "limits": {"arm_vmax": list(limits.arm_vmax), "arm_amax": list(limits.arm_amax),
                   "hand_vmax": list(limits.hand_vmax), "hand_amax": list(limits.hand_amax),
                   "arm_bounds": limits.arm_bounds().tolist(), "hand_bounds": limits.hand_bounds().tolist()},
        "rates_hz": {"arm": 1.0 / float(traj.arm_t[1] - traj.arm_t[0]),
                     "hand": 1.0 / float(traj.hand_t[1] - traj.hand_t[0])},
    }
    (out / "package.json").write_text(json.dumps(_json(pkg), indent=1))      # the gate reads its "table"
    pkg["sim_gate"] = sim_gate(out)
    allowed = set(pkg["sim_gate"]["homes_allowed"])
    pkg["home_tol_rad"] = result.transfers.home_tol
    pkg["homes"] = [{"name": r["name"], "home_q": r["home_q"], "in_package": r["in_package"], "index": r["index"],
                     "passes_gate": r["index"] is not None and r["index"] in allowed}
                    for r in result.transfers.homes]
    (out / "package.json").write_text(json.dumps(_json(pkg), indent=1))
    if result.search is not None:
        (out / "placement_search.json").write_text(json.dumps(_json(result.search.records), indent=1))
    shutil.copyfile(g1_sdk_replay.__file__, out / "run_g1_replay.py")
    notice = ["# Sources and licences", "",
              "The numbers in this package (model, limits, gains, topics, message layout) come from these official",
              "sources at the pinned commits. No third-party source file is copied into the package.", ""]
    notice += [f"- **{n}** ({url} @ `{sha[:8]}`, {lic}): {what}" for n, url, sha, lic, what in _SOURCES]
    (out / "NOTICE.md").write_text("\n".join(notice) + "\n")
    (out / "README.md").write_text(_readme(result, _json(pkg), stats))  # type: ignore[arg-type]
    return out
