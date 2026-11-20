#!/usr/bin/env python3
"""Open-loop replay of an exported robot package on a Unitree G1 (29-DoF, mode_machine 5/6) + right Dex3-1.

THIS SCRIPT HAS NEVER RUN ON HARDWARE. It was exercised only against its own simulated I/O (the default, without
`--execute`).
It is copied verbatim into every package by `fastdo.robot.package` and must not import `fastdo` (one self-contained
file the lab copies to the robot PC); it needs numpy, and `unitree_sdk2_python` only with `--execute`.

Interface (official SDK, DDS; see the package README for the sources):
- arms: `rt/arm_sdk` (`unitree_hg` `LowCmd_`, CRC32), blend weight in `motor_cmd[29].q` -- the robot stays in
  Unitree's own standing controller (Regular mode); every arm and waist entry is enabled (mode 1) with its gains;
- right hand: `rt/dex3/right/cmd` (`HandCmd_`, 7 motors, the enum order assumed by this package: thumb0-2, index0-1,
  middle0-1; its mapping to the physical index and middle fingers is not verified on hardware);
- state: `rt/lowstate` (`LowState_`), `rt/dex3/right/state` (`HandState_`).

Threading, as the official references: ONE control loop publishes `rt/arm_sdk` every tick at the plan's arm rate
(250 Hz, xr_teleoperate `G1_29_ArmController._ctrl_motor_state`; the SDK arm7 example's `RecurrentThread`) and the hand
at 100 Hz (xr_teleoperate `Dex3_1_Controller`), in EVERY phase, including while the operator is asked for 'go'. With
`--execute` the loop runs in its own (non-daemon) thread and the main thread only talks to the operator; the dry-run
runs the same loop in the main thread on a virtual clock.

Phases: refuse unless fresh state, mode_machine 5/6, the right arm inside its limits, the measured waist (roll, pitch,
yaw) within `WAIST_TOL` (0.01 rad) of the simulated 0 (the simulation welds the waist at 0 and the
planner's tolerance box has no waist; with `mode_machine` 6 roll and pitch are held where they are measured), the
measured arms / waist within `home_tol` (stored in the package, at most `HOME_TOL_MAX`) of one of the package's
CHECKED homes (`home_q`; with `--execute`, only a home whose moves pass `SIM_GATE`) and, with `--execute`, the measured
Dex3 joints within `START_HAND_TOL` (0.1 rad) of the hand pose the package simulated the run from (the
refusal prints the measured values, so the lab can pose the hand and retry) -> RAMP_UP the weight 0 -> 1 holding the measured pose
(HOME) -> TO_HOLD: follow the package's collision-checked path from that home (`to_hold_path`: the right hand shapes,
the left arm parks, then the right arm moves to the trajectory start; waypoints planned and simulated offline by
`fastdo.robot.transfer`) to the hold pose (trajectory start for the right arm and hand, park pose for the left arm,
waist 0) -> WAIT_GO for the operator (publishing the hold pose; no answer within `--hold-timeout` = operator stop) ->
PLAY (`--speed-scale` < 1 slows it; the control rate is unchanged) -> GO_HOME: after a completed playback, the
package's collision-checked path back to HOME (`go_home_path`, the hand holding its last command); after an operator
stop, the commands sent since TO_HOLD began, RETRACED in reverse (hand included) back to HOME -> RELEASE: the weight
ramps to 0 over `ramp_s`, publishing every tick. Every TO_HOLD / GO_HOME command lies on its polyline and no joint
steps faster than `--hold-speed` (`Polyline`); between waypoints the motion is the straight joint-space interpolation
of Unitree's references (xr_teleoperate `ctrl_dual_arm_go_home`, the SDK arm7 example).

Safety, every tick before RELEASE (any failure = ABORT): `rt/lowstate` or the Dex3 state older than
`Config.state_timeout` (0.1 s); the tick started more than `OVERRUN_PERIODS` (3) control periods late; mode_machine changed; a right-arm command outside the joint limits or stepping faster than the per-joint velocity limit
stored in the package; a left-arm or waist command stepping faster than `OTHER_VMAX`; a hand command outside its limits
or faster than its velocity limit; a right-arm joint more than `--max-track-err` rad from the previous command (from
TO_HOLD on); any exception while computing or publishing. ABORT = RELEASE at the MEASURED pose (arm and hand) when the
state is fresh, else at the LAST COMMAND; RELEASE never reads state, so a stale-state abort still ramps the weight down
tick by tick. Operator stop (Ctrl-C / SIGINT, SIGTERM, an answer other than 'go', the hold timeout): during TO_HOLD,
WAIT_GO or PLAY -> GO_HOME then RELEASE (as xr_teleoperate's main loop `finally`); during RAMP_UP or GO_HOME (e.g. a
second Ctrl-C) -> RELEASE at the last command; during RELEASE -> ignored (the ramp always finishes).

Timing: a tick that starts more than half a period late resynchronises the schedule to one period after its own start
(the playback slows down, it never catches up in a burst); the run result reports the tick statistics (`ticks`).

`--execute` also refuses unless the package's own simulation (`sim_stats.json`, computed on THIS `trajectory.npz`)
passes `SIM_GATE`: the trajectory replay (no table penetration > 5 mm, no robot-table force > 20 N, no self-collision
> 1 mm, no joint-limit violation, right-arm tracking max < 0.15 rad, the object moved as in the reference) AND, for at
least one checked home, this script's own run simulated in MuJoCo from that home (`transfers`: no table contact and no
robot-object contact outside the playback, no self-collision, tracking < 0.15 rad; operator-stop drills whose retrace
stays within the playback's table / self-collision thresholds; the same run on a table at both edges of the stated
placement tolerance, `table_tolerance_runs`, planned and simulated at >= `SIM_GATE["table_tol_min_m"]`). The script
then accepts only the homes that passed, and only the run options the simulation used (`--hold-speed`, `--speed-scale`,
the gravity feed-forward: `condition_failures`; `--no-tau-ff` had let the arm onto the table in
TO_HOLD from a checked home). `--no-waist` is never simulated, so `--execute` refuses it.
`--speed-scale` defaults to the value the package simulated (`transfers` "speed_scale"; a task's first package is
simulated at 0.5). The trajectory replay's object must rest where it is placed
(settle <= 5 mm, <= 2 deg); the run simulated at the package's speed scale must itself move the object as the reference
does; the run is also simulated with the object at each corner of its placement tolerance (+-1 cm x / y, +-5 deg
yaw) and from the measured states this script accepts (arm-box axis extremes and seeded corners, the Dex3 hand at
+-`START_HAND_TOL`), all held to the gate; every simulated run records its run options and they must be the
package's; `--max-track-err` and `--hold-timeout` are accepted under `--execute` only at or below the simulated values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import signal
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

ARM_SDK_TOPIC = "rt/arm_sdk"
LOWSTATE_TOPIC = "rt/lowstate"
HAND_CMD_TOPIC = "rt/dex3/right/cmd"
HAND_STATE_TOPIC = "rt/dex3/right/state"
WEIGHT_INDEX = 29
MODE_MACHINES = (5, 6)
LEFT_ARM_DDS = tuple(range(15, 22))
RIGHT_ARM_DDS = tuple(range(22, 29))
WAIST_DDS = (12, 13, 14)
ARM_KP = (80.0, 80.0, 80.0, 80.0, 40.0, 40.0, 40.0)
ARM_KD = (3.0, 3.0, 3.0, 3.0, 1.5, 1.5, 1.5)
WAIST_KP, WAIST_KD = 300.0, 3.0
HAND_KP, HAND_KD = 1.5, 0.2
HAND_RATE_HZ = 100.0
OTHER_VMAX = 0.5          # rad/s: the left arm and the waist only ever move at --hold-speed (<= 0.5)
TRACKED_PHASES = ("to_hold", "wait_go", "play", "go_home")
OVERRUN_PERIODS = 3       # a tick starting later than this many control periods aborts
MAX_TRACK_ERR = 0.3       # rad: --max-track-err may only tighten the runtime tracking abort
HOME_TOL_MAX = 0.15       # rad: the largest home tolerance a package may declare
WAIST_TOL = 0.01          # rad: measured waist vs the simulated 0 (the waist is welded in the simulation)
START_HAND_TOL = 0.1      # rad: measured Dex3 joints vs the simulated start hand under --execute
TABLE_TOL_MIN_M = 0.01    # m: the smallest table placement tolerance a package may be planned and simulated at
OBJECT_XY_TOL_MIN_M = 0.01      # m: object placement tolerance in x and y the moves must be planned and simulated at
OBJECT_YAW_TOL_MIN_DEG = 5.0    # deg: ... and in yaw
# The object placement corners: case -> signs of (dx, dy, dyaw); each is simulated, all must pass.
PLACEMENT_CASES = {f"x{'+' if sx > 0 else '-'}y{'+' if sy > 0 else '-'}yaw{'+' if sa > 0 else '-'}": (sx, sy, sa)
                   for sx in (1, -1) for sy in (1, -1) for sa in (1, -1)}
STATE_CORNERS_MIN = 4     # seeded corners of the home_tol arm box simulated per home (a choice; + 28 axis extremes)
STATE_EDGE_MIN = 0.99     # the measured-state runs start at least this fraction of each tolerance from the home / hand
SETTLE_DRIFT_MAX_M = 0.005      # the object's settle on the table: translation
SETTLE_ROT_MAX_DEG = 2.0        # ... and rotation
# A command as one vector (`Cmd.vec`): right arm 7, left arm 7, waist 3, right / left feed-forward 7 + 7, right hand 7.
CMD_SIZE = 38
RIGHT, LEFT, WAIST, TAU, HAND = slice(0, 7), slice(7, 14), slice(14, 17), slice(17, 31), slice(31, 38)
POS = np.r_[0:17, 31:38]  # the position entries (the feed-forward torques are not positions)
ARMS_HAND = np.r_[0:14, 31:38]  # arms and hand (the waist is the measured / commanded one at run time)
TRANSFER_PHASES = ("ramp_up", "to_hold", "wait_go", "go_home", "release")
TABLE_KEYS = ("near_edge_x_m", "far_edge_x_m", "right_edge_y_m", "left_edge_y_m", "top_z_m", "thickness_m")
# N: robot-table normal force cap of the playback, the stop drills and the table-tolerance runs (a chosen value; in
# one draft clip's scene a stop drill's light touches read 1.3-1.5 N, an earlier straight TO_HOLD through the table
# 66 N, that clip's own aborting playback 123 N; the simulated contact depth stays below 2 mm in all three)
TABLE_FORCE_MAX_N = 20.0
# Labels of gate values that are choices, not sourced numbers; package.json and the README carry them.
SIM_GATE_NOTES = {
    "table_max_force_n": ("chosen, not sourced: calibrated on one draft clip only (stop-drill touches 1.3-1.5 N; an "
                          "earlier straight TO_HOLD through the table 66 N, that clip's aborting playback 123 N)"),
    "state_corner_runs_min": ("chosen, not sourced: 4 seeded corners of the 14-joint arm box per home, "
                              "beside all 28 axis extremes; the kinematic box check samples 48 corners"),
    "placement_cases": ("set tolerance (+-1 cm x / y, +-5 deg yaw); gating the object criteria at the corners "
                        "(not only contacts) is a design choice"),
}


def _close(a: np.ndarray, b: np.ndarray, tol: float = 1e-9) -> bool:
    return bool(np.abs(np.asarray(a) - np.asarray(b)).max() <= tol)


def state_run_names(corners: int) -> list[str]:
    """Names of the measured-state runs of one home: the 28 axis extremes of the 14-joint arm box
    (`arm_axis_<joint>+` / `-`, joints in `home_q` order), `corners` seeded corners (`arm_corner_<i>`) and the Dex3 start
    at +- `START_HAND_TOL` on every joint (`hand+`, `hand-`)."""
    return ([f"arm_axis_{j:02d}{s}" for j in range(14) for s in "+-"] + [f"arm_corner_{i}" for i in range(corners)]
            + ["hand+", "hand-"])

# Execution gate: `--execute` refuses unless the package's simulated replay of
# THIS trajectory meets every threshold. Mirrored in the package README and `package.json` ("sim_gate").
SIM_GATE = {
    "table_max_depth_m": 0.005,            # robot-table penetration, max over the replay
    "table_max_force_n": TABLE_FORCE_MAX_N,  # robot-table normal force (summed over contacts), max over the replay
    "self_collision_max_depth_m": 0.001,   # robot-robot penetration (contact, not collision, below 1 mm)
    "limit_violation_ticks": 0,            # ticks with a right arm/hand joint > 0.01 rad outside its range
    "arm_tracking_max_rad": 0.15,          # strict: max |q - q_cmd| of the right arm (half the runtime abort, 0.3)
    "min_reference_displacement_m": 0.02,  # the reference itself must move the object
    "min_move_fraction": 0.5,              # sim object displacement >= this x reference displacement
    "final_error_floor_m": 0.03,           # final object error <= max(floor, fraction x reference displacement)
    "max_final_error_fraction": 0.5,
    # this script's own run, simulated from a checked home (`transfers`)
    "transfer_table_contact_steps": 0,     # outside the playback (ramps, TO_HOLD, WAIT_GO, GO_HOME): no table contact
    "transfer_self_collision_max_depth_m": 0.001,
    "transfer_object_contact_steps": 0,    # ... and the robot never touches the object outside the playback
    "transfer_arm_tracking_max_rad": 0.15,
    "stop_table_max_depth_m": 0.005,       # operator-stop drills: the retrace runs through the playback's own poses
    "stop_table_max_force_n": TABLE_FORCE_MAX_N,
    # the run simulated again on a table at the edges of the placement tolerance (`transfers` "table_tolerance_runs":
    # top +tol with every edge out by tol, and top -tol): the transfer thresholds outside the playback, the replay's
    # table thresholds (depth and force) inside it
    "table_tolerance_cases": ["table_high", "table_low"],
    "stop_self_collision_max_depth_m": 0.001,
    "stop_arm_tracking_max_rad": 0.15,
    "home_tol_max_rad": HOME_TOL_MAX,
    "table_tol_min_m": TABLE_TOL_MIN_M,    # the tolerance runs must be planned / simulated at least this far out
    "waist_tol_rad": WAIST_TOL,            # run time: measured waist vs 0
    "start_hand_tol_rad": START_HAND_TOL,  # run time, --execute: measured Dex3 vs the simulated start hand
    # the object must rest where it is placed (the trajectory replay's settle)
    "settle_drift_max_m": SETTLE_DRIFT_MAX_M,
    "settle_rot_max_deg": SETTLE_ROT_MAX_DEG,
    # the run simulated at the package's speed scale must move the object as the reference
    # (the replay's object thresholds above, read on the run's playback)
    "run_object_criteria": True,
    # the object placed at every corner of +-xy / +-yaw (`placement_runs`): the table-tolerance
    # thresholds outside and inside the playback, and the object thresholds against the nominal reference end
    "object_xy_tol_min_m": OBJECT_XY_TOL_MIN_M,
    "object_yaw_tol_min_deg": OBJECT_YAW_TOL_MIN_DEG,
    "placement_cases": list(PLACEMENT_CASES),
    # runs from the measured states --execute accepts (`state_runs`: the arm-box axis
    # extremes and seeded corners, the Dex3 start at +-start_hand_tol), the table-tolerance thresholds
    "state_corner_runs_min": STATE_CORNERS_MIN,
    "state_edge_min": STATE_EDGE_MIN,
    "state_rest_max_frac": 0.1,           # ... while the joints a run does not test stay within this fraction of it
}


class Abort(RuntimeError):
    """A safety check failed: the replayer releases arm_sdk (module docstring)."""


@dataclass
class Plan:
    arm_t: np.ndarray
    arm_q: np.ndarray
    arm_tau_ff: np.ndarray
    hand_t: np.ndarray
    hand_q: np.ndarray
    left_park: np.ndarray
    left_tau_ff: np.ndarray
    arm_limits: np.ndarray
    hand_limits: np.ndarray
    arm_vmax: np.ndarray
    hand_vmax: np.ndarray
    home_q: np.ndarray        # (H,17) checked homes: right arm 7, left arm 7, waist 3
    home_tol: np.ndarray      # () rad: the measured pose must be this close to a home (max over the 17 joints)
    to_hold_path: np.ndarray  # (N,38) `Cmd.vec` waypoints of every home, concatenated: home -> hold
    to_hold_index: np.ndarray  # (H+1,) row offsets of each home's path in `to_hold_path`
    go_home_path: np.ndarray  # (M,38) trajectory end -> home, per home
    go_home_index: np.ndarray

    @classmethod
    def load(cls, path: Path) -> Plan:
        """Raises ValueError when a field is missing (a package written before the checked transfers)."""
        with np.load(path) as z:
            missing = [k for k in cls.__dataclass_fields__ if k not in z.files]
            if missing:
                raise ValueError(f"{Path(path).name} has no {', '.join(missing)}: it was written before the "
                                 "collision-checked moves to the hold pose and home; re-export the package")
            return cls(**{k: np.asarray(z[k], dtype=np.float64) for k in cls.__dataclass_fields__})

    def to_hold_rows(self, k: int) -> np.ndarray:
        i = self.to_hold_index.astype(np.int64)
        return self.to_hold_path[i[k]:i[k + 1]]

    def go_home_rows(self, k: int) -> np.ndarray:
        i = self.go_home_index.astype(np.int64)
        return self.go_home_path[i[k]:i[k + 1]]

    def hold_vec(self) -> np.ndarray:
        """The trajectory's first command (waist 0) as a `Cmd.vec`."""
        return np.concatenate([self.arm_q[0], self.left_park, np.zeros(3), self.arm_tau_ff[0], self.left_tau_ff,
                               self.hand_q[0]])

    def end_vec(self) -> np.ndarray:
        return np.concatenate([self.arm_q[-1], self.left_park, np.zeros(3), self.arm_tau_ff[-1], self.left_tau_ff,
                               self.hand_q[-1]])

    def _validate_transfers(self) -> list[str]:
        h, n = self.home_q, len(self.home_q)
        if h.ndim != 2 or h.shape[1] != 17 or n == 0 or not np.isfinite(h).all():
            return ["no checked home (home_q): the package has no collision-checked move to the hold pose"]
        if self.home_tol.size != 1 or not 0.0 < float(self.home_tol) <= HOME_TOL_MAX:
            return [f"home_tol must be in (0, {HOME_TOL_MAX}] rad"]
        errs = []
        hold, end = self.hold_vec(), self.end_vec()
        for name, path, idx, rows in (("to_hold", self.to_hold_path, self.to_hold_index, self.to_hold_rows),
                                      ("go_home", self.go_home_path, self.go_home_index, self.go_home_rows)):
            if path.ndim != 2 or path.shape[1] != CMD_SIZE or not np.isfinite(path).all():
                errs.append(f"{name}_path: bad shape {path.shape} or non-finite values")
                continue
            if idx.shape != (n + 1,) or (idx != np.round(idx)).any() or idx[0] != 0 or idx[-1] != len(path) \
                    or (np.diff(idx) < 2).any():
                errs.append(f"{name}_index: not {n} consecutive paths of >= 2 rows")
                continue
            if ((path[:, RIGHT] < self.arm_limits[:, 0] - 1e-9) | (path[:, RIGHT] > self.arm_limits[:, 1] + 1e-9)).any() \
                    or ((path[:, HAND] < self.hand_limits[:, 0] - 1e-9)
                        | (path[:, HAND] > self.hand_limits[:, 1] + 1e-9)).any():
                errs.append(f"{name}_path: positions outside the joint limits")
            if np.abs(path[:, TAU]).max() > 25.0:
                errs.append(f"{name}_path: feed-forward torque above the 25 N m arm limit")
            for k in range(n):
                r = rows(k)
                if name == "to_hold":         # starts AT the home (waist included), ends at the trajectory start
                    joined = _close(r[0, :17], h[k]) and _close(r[-1, ARMS_HAND], hold[ARMS_HAND])
                else:                         # starts at the trajectory end, ends AT the home
                    joined = _close(r[0, ARMS_HAND], end[ARMS_HAND]) and _close(r[-1, :17], h[k])
                if not joined:
                    errs.append(f"{name} path of home #{k} does not join " +
                                ("that home and the trajectory start" if name == "to_hold" else
                                 "the trajectory end and that home"))
        return errs

    def validate(self) -> list[str]:
        errs = []
        for name, t, q, lim, vmax, n in (("arm", self.arm_t, self.arm_q, self.arm_limits, self.arm_vmax, 7),
                                         ("hand", self.hand_t, self.hand_q, self.hand_limits, self.hand_vmax, 7)):
            if q.ndim != 2 or q.shape[1] != n or len(t) != len(q) or len(t) < 2:
                errs.append(f"{name}: bad shapes {q.shape} / {t.shape}")
                continue
            if not (np.isfinite(q).all() and np.isfinite(t).all()):
                errs.append(f"{name}: non-finite values")
                continue
            dt = np.diff(t)
            if (dt <= 0).any() or np.ptp(dt) > 1e-6:
                errs.append(f"{name}: time is not uniform and increasing")
                continue
            if ((q < lim[:, 0] - 1e-9) | (q > lim[:, 1] + 1e-9)).any():
                errs.append(f"{name}: positions outside the joint limits")
            vel = np.abs(np.diff(q, axis=0)) / dt[0]
            if (vel > vmax * (1 + 1e-3)).any():
                errs.append(f"{name}: velocity {float((vel / vmax).max()):.3f}x the limit")
        if not errs and not 1.0 / 500.0 - 1e-9 <= self.arm_t[1] - self.arm_t[0] <= 1.0 / 100.0 + 1e-9:
            errs.append("arm: the control period must be between 2 and 10 ms")
        if abs(self.arm_t[-1] - self.hand_t[-1]) > 0.02:
            errs.append("arm and hand streams have different durations")
        if not np.isfinite(self.arm_tau_ff).all() or np.abs(self.arm_tau_ff).max() > 25.0:
            errs.append("feed-forward torque is non-finite or above the 25 N m arm limit")
        if not errs:
            errs += self._validate_transfers()
        return errs


@dataclass
class State:
    stamp: float             # receive time of rt/lowstate
    mode_machine: int
    q: np.ndarray            # (35,) body motors
    hand_q: np.ndarray       # (7,)
    hand_stamp: float        # receive time of the Dex3 state


@dataclass
class Config:
    hold_speed: float = 0.3
    ramp_s: float = 2.0
    max_track_err: float = 0.3
    state_timeout: float = 0.1
    send_tau_ff: bool = True
    hold_waist: bool = True
    speed_scale: float = 1.0
    hold_timeout_s: float = 120.0
    homes: tuple[int, ...] | None = None    # indices of `Plan.home_q` the run may start from (None: all; dry-run)
    start_hand: tuple[float, ...] | None = None  # the simulated Dex3 start; None: not checked (unit runs)


@dataclass(frozen=True)
class Cmd:
    """One control-tick command: arms, waist (all 3 entries), feed-forward torques, right hand."""
    right: np.ndarray
    left: np.ndarray
    waist: np.ndarray
    tau_r: np.ndarray
    tau_l: np.ndarray
    hand: np.ndarray

    def parts(self) -> tuple[np.ndarray, ...]:
        return self.right, self.left, self.waist, self.tau_r, self.tau_l, self.hand

    def vec(self) -> np.ndarray:
        """`(CMD_SIZE,)`: right, left, waist, tau_r, tau_l, hand."""
        return np.concatenate(self.parts())

    @classmethod
    def from_vec(cls, v: np.ndarray) -> Cmd:
        v = np.asarray(v, dtype=np.float64)
        return cls(v[RIGHT].copy(), v[LEFT].copy(), v[WAIST].copy(), v[17:24].copy(), v[24:31].copy(), v[HAND].copy())


class Polyline:
    """A joint-space polyline of `Cmd.vec` points followed at a constant L-inf speed: tick k (1..n) commands the point
    at arc length k/n of the whole length (positions only, `POS`), n = ceil(length / (speed * dt)). Every command lies
    on the polyline, no joint steps more than speed * dt per tick (also across a vertex), and tick n is the last point."""

    def __init__(self, points: np.ndarray, speed: float, dt: float) -> None:
        pts = np.asarray(points, dtype=np.float64)
        seg = np.abs(np.diff(pts[:, POS], axis=0)).max(axis=1) if len(pts) > 1 else np.zeros(0)
        moving = seg > 1e-12
        self.points = pts[np.concatenate([[True], moving])]
        if len(self.points) > 1 and not moving[-1]:
            self.points[-1] = pts[-1]                  # keep the exact last point (its feed-forward torque)
        self.cum = np.concatenate([[0.0], np.cumsum(seg[moving])])
        self.length = float(self.cum[-1])
        self.n = max(int(np.ceil(self.length / (speed * dt) - 1e-9)), 1)

    def at(self, k: int) -> np.ndarray:
        if len(self.points) == 1:
            return self.points[0].copy()
        s = min(k / self.n, 1.0) * self.length
        i = int(np.clip(np.searchsorted(self.cum, s, side="right") - 1, 0, len(self.points) - 2))
        w = min(max((s - self.cum[i]) / (self.cum[i + 1] - self.cum[i]), 0.0), 1.0)
        return (1.0 - w) * self.points[i] + w * self.points[i + 1]


class SimulatedIO:
    """Dry-run robot: measured joints follow the commands with a first-order lag on a virtual clock (no sleeping)."""

    def __init__(self, q0: np.ndarray | None = None, hand0: np.ndarray | None = None, lag_s: float = 0.03,
                 mode_machine: int = 5, fault: dict | None = None) -> None:
        self.t = 0.0
        self.q = np.zeros(35) if q0 is None else q0.copy()
        self.hand_q = np.zeros(7) if hand0 is None else hand0.copy()
        self.lag = lag_s
        self.mode_machine = mode_machine
        self.fault = fault or {}
        self.weight = 0.0
        self.sent_arm = 0
        self.sent_hand = 0

    def now(self) -> float:
        return self.t

    def sleep_until(self, t: float) -> None:
        self.t = max(self.t, t)
        time.sleep(0)                      # yield to the operator thread when the loop is threaded

    def state(self) -> State:
        return State(self.t, self.mode_machine, self.q.copy(), self.hand_q.copy(), self.t)

    def send_arm(self, q: dict[int, float], kp: dict[int, float], kd: dict[int, float], tau: dict[int, float],
                 weight: float, dt: float) -> None:
        self.weight = weight
        self.sent_arm += 1
        a = min(1.0, dt / self.lag) * weight
        for i, target in q.items():
            if i == self.fault.get("stuck_joint") and self.t >= self.fault.get("after_s", 0.0):
                continue
            self.q[i] += a * (target - self.q[i])

    def send_hand(self, q: np.ndarray, dt: float) -> None:
        self.sent_hand += 1
        self.hand_q += min(1.0, dt / 0.05) * (q - self.hand_q)


class SdkIO:
    """unitree_sdk2_python transport (imported only here, only with --execute)."""

    def __init__(self, iface: str, domain: int, hand_state_topic: str = HAND_STATE_TOPIC) -> None:
        from unitree_sdk2py.core.channel import (
            ChannelFactoryInitialize,
            ChannelPublisher,
            ChannelSubscriber,
        )
        from unitree_sdk2py.idl.default import (
            unitree_hg_msg_dds__HandCmd_,
            unitree_hg_msg_dds__LowCmd_,
        )
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_, LowCmd_, LowState_
        from unitree_sdk2py.utils.crc import CRC
        ChannelFactoryInitialize(domain, iface)
        self.crc = CRC()
        self.low = unitree_hg_msg_dds__LowCmd_()
        self.hand = unitree_hg_msg_dds__HandCmd_()
        self.arm_pub = ChannelPublisher(ARM_SDK_TOPIC, LowCmd_)
        self.arm_pub.Init()
        self.hand_pub = ChannelPublisher(HAND_CMD_TOPIC, HandCmd_)
        self.hand_pub.Init()
        self._low_state: Any = None
        self._hand_state: Any = None
        self._low_stamp = self._hand_stamp = -1.0
        self.low_sub = ChannelSubscriber(LOWSTATE_TOPIC, LowState_)
        self.low_sub.Init(self._on_low, 10)
        self.hand_sub = ChannelSubscriber(hand_state_topic, HandState_)
        self.hand_sub.Init(self._on_hand, 10)

    def _on_low(self, msg: Any) -> None:
        self._low_state = msg
        self._low_stamp = time.perf_counter()

    def _on_hand(self, msg: Any) -> None:
        self._hand_state = msg
        self._hand_stamp = time.perf_counter()

    def now(self) -> float:
        return time.perf_counter()

    def sleep_until(self, t: float) -> None:
        delay = t - time.perf_counter()
        if delay > 0:
            time.sleep(delay)

    def state(self) -> State | None:
        low, hand = self._low_state, self._hand_state
        if low is None or hand is None:
            return None
        q = np.array([low.motor_state[i].q for i in range(35)])
        hq = np.array([hand.motor_state[i].q for i in range(7)])
        return State(self._low_stamp, int(low.mode_machine), q, hq, self._hand_stamp)

    def send_arm(self, q: dict[int, float], kp: dict[int, float], kd: dict[int, float], tau: dict[int, float],
                 weight: float, dt: float) -> None:
        low = self._low_state
        self.low.mode_pr = 0                                # as xr_teleoperate: PR ankle mode, echo mode_machine
        self.low.mode_machine = int(low.mode_machine) if low is not None else 0
        for i, target in q.items():
            m = self.low.motor_cmd[i]
            m.mode = 1                                      # enable, as xr_teleoperate's arm controller
            m.q, m.dq, m.tau, m.kp, m.kd = float(target), 0.0, float(tau.get(i, 0.0)), float(kp[i]), float(kd[i])
        self.low.motor_cmd[WEIGHT_INDEX].q = float(weight)
        self.low.crc = self.crc.Crc(self.low)
        self.arm_pub.Write(self.low)

    def send_hand(self, q: np.ndarray, dt: float) -> None:
        for i in range(7):
            m = self.hand.motor_cmd[i]
            m.mode = (i & 0x0F) | (0x01 << 4)          # id | status=enable (xr_teleoperate _RIS_Mode)
            m.q, m.dq, m.tau, m.kp, m.kd = float(q[i]), 0.0, 0.0, HAND_KP, HAND_KD
        self.hand_pub.Write(self.hand)


def _within(q: np.ndarray, lim: np.ndarray) -> bool:
    return bool(((q >= lim[:, 0] - 1e-9) & (q <= lim[:, 1] + 1e-9)).all())


def _interp(t: np.ndarray, q: np.ndarray, at: float) -> np.ndarray:
    """Rows `q` at uniform times `t`, linearly interpolated at `at` (clipped to the plan)."""
    f = (min(max(at, float(t[0])), float(t[-1])) - float(t[0])) / float(t[1] - t[0])
    i = min(int(np.floor(f + 1e-9)), len(q) - 1)
    w = f - i
    if i >= len(q) - 1 or w < 1e-9:
        return q[i].copy()
    return (1.0 - w) * q[i] + w * q[i + 1]


class Replayer:
    """The control loop (module docstring). `run()`: the loop in the calling thread (dry-run). `run(confirm)`: the
    loop in a control thread, the operator in the calling thread (`confirm()` returns for 'go'; raising anything,
    e.g. Abort, EOFError on a closed stdin, or KeyboardInterrupt, is an operator stop)."""

    def __init__(self, plan: Plan, io: Any, cfg: Config) -> None:
        self.plan, self.io, self.cfg = plan, io, cfg
        self.dt = float(plan.arm_t[1] - plan.arm_t[0])
        self.hand_dt = 1.0 / HAND_RATE_HZ
        self.ramp_n = max(round(cfg.ramp_s / self.dt), 1)
        self.weight = 0.0
        self.mode_machine = 0
        self.phase = "init"
        self.k = 0                      # tick within the phase
        self.n = 0                      # tick since the start
        self.t_next = 0.0
        self.hand_slot = 0              # next hand slot (multiples of hand_dt)
        self.last_hand: tuple[int, np.ndarray] | None = None
        self._sent = False              # an arm message went out this tick
        self.status: str | None = None
        self.reason: str | None = None
        self.phases: dict[str, str] = {}
        self._go = threading.Event()
        self._at_hold = threading.Event()
        self._finished = threading.Event()
        self._lock = threading.RLock()  # also taken by the signal handler, possibly inside request_stop
        self._stop_requests = 0         # pending, consumed by the control loop
        self.stop_count = 0             # all requests so far
        self._stop_reason: str | None = None
        self._signal_ki = False
        self.abort_phase: str | None = None
        self._tick_n = 0                # tick statistics (module docstring, "Timing")
        self._late_ticks = 0
        self._max_late = 0.0
        self._max_gap = 0.0
        self._last_start: float | None = None
        self.home_k: int | None = None  # index of the checked home the run started from
        self.trail: list[np.ndarray] = []   # `Cmd.vec` of every command sent in TO_HOLD and PLAY (the retrace)
        self.retraced = False
        self.path: Polyline | None = None

    # ---- operator side (any thread) -------------------------------------------------------------------------------
    @property
    def stop_requested(self) -> bool:
        return self._stop_reason is not None

    def request_stop(self, reason: str) -> None:
        """Operator stop (module docstring); each call is one request, so a second one escalates."""
        with self._lock:
            self._stop_requests += 1
            self.stop_count += 1
            if self._stop_reason is None:
                self._stop_reason = reason

    def _on_signal(self, signum: int, frame: Any, raise_ki: bool) -> None:
        self.request_stop(f"operator interrupt ({signal.Signals(signum).name})")
        if raise_ki:                    # wake the operator thread out of input() / join(); already counted
            self._signal_ki = True
            raise KeyboardInterrupt

    # ---- control loop -----------------------------------------------------------------------------------------------
    def _start(self) -> dict | None:
        """Wait for state and check the robot: None when the loop may start, else the "refused" result."""
        t0 = self.io.now()
        while self.io.state() is None:
            self.io.sleep_until(self.io.now() + 0.05)
            if self.io.now() - t0 > 5.0:
                return {"status": "refused", "reason": "no robot state after 5 s", "phases": {}}
        st = self.io.state()
        now = self.io.now()
        if now - st.stamp > self.cfg.state_timeout or now - st.hand_stamp > self.cfg.state_timeout:
            return {"status": "refused", "reason": "no fresh rt/lowstate or rt/dex3 hand state", "phases": {}}
        if st.mode_machine not in MODE_MACHINES:
            return {"status": "refused", "reason": f"mode_machine {st.mode_machine} is not one of {MODE_MACHINES} "
                    "(G1 29-DoF + Dex3-1 with 4010 wrists)", "phases": {}}
        right = st.q[list(RIGHT_ARM_DDS)]
        if not _within(right, self.plan.arm_limits):
            return {"status": "refused", "reason": "the measured right arm is outside the command limits",
                    "phases": {}}
        measured = np.concatenate([right, st.q[list(LEFT_ARM_DDS)], st.q[list(WAIST_DDS)]])
        home_k = self._pick_home(measured)
        if isinstance(home_k, str):
            return {"status": "refused", "reason": home_k, "phases": {}}
        hand_why = self._hand_refusal(st.hand_q)
        if hand_why is not None:
            return {"status": "refused", "reason": hand_why, "phases": {}}
        self.home_k = home_k
        self.mode_machine = st.mode_machine
        hand = np.clip(st.hand_q, self.plan.hand_limits[:, 0], self.plan.hand_limits[:, 1])
        tau = self.plan.to_hold_rows(home_k)[0, TAU]          # gravity feed-forward at the checked home
        self.home = Cmd(right.copy(), measured[LEFT].copy(), measured[WAIST].copy(), tau[:7].copy(), tau[7:].copy(),
                        hand)
        self.cmd = self.home            # the last PUBLISHED command from here on
        self.t_next = self.io.now()
        self._enter("ramp_up")
        return None

    def _hand_refusal(self, hand_q: np.ndarray) -> str | None:
        """Why the measured Dex3 joints are refused (more than `START_HAND_TOL` from the simulated start
        hand), or None; nothing is checked without `Config.start_hand`."""
        if self.cfg.start_hand is None:
            return None
        want = np.asarray(self.cfg.start_hand, dtype=np.float64)
        hand = np.asarray(hand_q, dtype=np.float64)
        err = float(np.abs(hand - want).max()) if want.shape == hand.shape and np.isfinite(hand).all() else np.inf
        if err <= START_HAND_TOL:
            return None
        return (f"the measured Dex3 hand is {err:.3f} rad from the simulated start hand (tolerance {START_HAND_TOL:g} "
                f"rad); measured = {np.round(hand, 3).tolist()}, simulated start = {np.round(want, 3).tolist()} "
                "(enum order assumed by this package, thumb0-2, index0-1, middle0-1; its mapping to the physical index and "
                "middle fingers is not verified on hardware): pose the hand there and retry")

    def _pick_home(self, measured: np.ndarray) -> int | str:
        """The allowed checked home nearest to the measured right/left/waist (17,) within `home_tol`, else why not.
        A measured waist more than `WAIST_TOL` from the simulated 0 is refused first."""
        p = self.plan
        waist = np.asarray(measured[WAIST], dtype=np.float64)
        werr = float(np.abs(waist).max()) if np.isfinite(waist).all() else np.inf
        if not werr <= WAIST_TOL:
            return (f"the measured waist (roll, pitch, yaw) = {np.round(waist, 4).tolist()} rad is {werr:.4f} rad from "
                    f"the simulated 0 (tolerance {WAIST_TOL:g} rad): the simulation welds the waist at 0, so a waist "
                    "offset moves the arm along a path nothing checked; zero the waist and retry")
        allowed = range(len(p.home_q)) if self.cfg.homes is None else self.cfg.homes
        errs = {k: float(np.abs(measured - p.home_q[k]).max()) for k in allowed}
        tol = float(p.home_tol)
        if errs:
            k = min(errs, key=errs.__getitem__)
            if errs[k] <= tol:
                return k
            why = f"are {errs[k]:.3f} rad from the nearest checked home (#{k})"
        else:
            why = "match no checked home: none passes the execution gate"
        return (f"the measured arms / waist {why}, tolerance {tol:g} rad; measured right arm, left arm, waist = "
                f"{np.round(measured, 3).tolist()}: put the arms at a checked home (README) or re-export the package "
                "with --home-q and these 17 values")

    def _enter(self, phase: str, retrace: bool = False) -> None:
        self.phase, self.k = phase, 0
        if phase == "to_hold":
            rows = self.plan.to_hold_rows(self.home_k).copy()
            rows[0] = self.home.vec()                       # the measured home (within home_tol of the checked one)
            rows[1:, WAIST] = self._waist_target()
            self.path = Polyline(rows, self.cfg.hold_speed, self.dt)
        elif phase == "go_home":
            self.retraced = retrace
            if retrace:                                     # the commands sent so far, in reverse, then HOME
                rows = np.vstack([self.cmd.vec()[None], *[v[None] for v in reversed(self.trail)],
                                  self.home.vec()[None]])
            else:                                           # the checked path from the trajectory end
                rows = self.plan.go_home_rows(self.home_k).copy()
                rows[0] = self.cmd.vec()
                rows[1:-1, WAIST] = self._waist_target()
                rows[-1, :17] = self.home.vec()[:17]
                rows[-1, TAU] = self.home.vec()[TAU]
                rows[:, HAND] = self.cmd.hand               # the hand holds its last command
            self.path = Polyline(rows, self.cfg.hold_speed, self.dt)
        elif phase == "release":
            self.w0 = self.weight
        if phase in ("wait_go", "done"):
            self._at_hold.set()

    def _waist_target(self) -> np.ndarray:
        """Commanded waist joints at 0 (mode_machine 6: yaw only, roll/pitch are locked); the others, and all three
        with --no-waist, at their measured start value."""
        w = self.home.waist.copy()
        if self.cfg.hold_waist:
            w[: 3 if self.mode_machine == 5 else 1] = 0.0
        return w

    def _finish(self, status: str, reason: str | None) -> None:
        if self.status is None or status == "aborted":
            self.status, self.reason = status, reason

    def _handle_requests(self) -> None:
        with self._lock:
            pending, self._stop_requests = self._stop_requests, 0
            reason = self._stop_reason or "operator stop"
        if pending and self.phase in ("to_hold", "wait_go", "play"):
            if self.phase == "play":
                self.phases["play"] = f"interrupted at {self.k * self.dt:.2f} s"
            self._finish("stopped", reason)
            self._enter("go_home", retrace=True)
            pending -= 1
        if pending and self.phase in ("ramp_up", "go_home"):
            self.phases[self.phase] = f"interrupted at {self.k * self.dt:.2f} s"
            self._finish("stopped", reason)
            self._enter("release")

    def _fresh_state(self) -> State:
        st = self.io.state()
        now = self.io.now()
        if st is None or now - st.stamp > self.cfg.state_timeout:
            raise Abort(f"no fresh rt/lowstate (older than {self.cfg.state_timeout} s)")
        if now - st.hand_stamp > self.cfg.state_timeout:
            raise Abort(f"no fresh rt/dex3 hand state (older than {self.cfg.state_timeout} s)")
        if st.mode_machine != self.mode_machine:
            raise Abort(f"mode_machine changed from {self.mode_machine} to {st.mode_machine}")
        return st

    def _next(self) -> tuple[Cmd, bool]:
        """`(command of this tick, phase finished after it)`; sets the weight during the ramp."""
        k, p = self.k, self.plan
        if self.phase == "ramp_up":
            self.weight = min(1.0, (k + 1) / self.ramp_n)
            return self.home, k + 1 >= self.ramp_n
        if self.phase in ("to_hold", "go_home"):
            return Cmd.from_vec(self.path.at(k + 1)), k + 1 >= self.path.n
        if self.phase == "wait_go":
            if not self._go.is_set() and k * self.dt >= self.cfg.hold_timeout_s and not self.stop_requested:
                self.request_stop(f"no 'go' within {self.cfg.hold_timeout_s:g} s at the hold pose")
            return self.cmd, self._go.is_set()
        if self.phase == "play":
            at = min(k * self.dt * self.cfg.speed_scale, float(p.arm_t[-1]))
            cmd = Cmd(_interp(p.arm_t, p.arm_q, at), p.left_park.copy(), self._waist_target(),
                      _interp(p.arm_t, p.arm_tau_ff, at), p.left_tau_ff.copy(), _interp(p.hand_t, p.hand_q, at))
            return cmd, at >= float(p.arm_t[-1]) - 1e-12
        raise RuntimeError(f"no command in phase {self.phase}")

    def _check(self, new: Cmd, st: State, hand_due: bool) -> None:
        p, prev, dt = self.plan, self.cmd, self.dt
        if not _within(new.right, p.arm_limits):
            raise Abort(f"right-arm command outside the limits: {np.round(new.right, 4).tolist()}")
        step = np.abs(new.right - prev.right) / dt
        if (step > p.arm_vmax * (1 + 1e-3) + 1e-9).any():
            raise Abort(f"right-arm command step {step.max():.3f} rad/s above the velocity limit")
        other = max(np.abs(new.left - prev.left).max(), np.abs(new.waist - prev.waist).max()) / dt
        if other > OTHER_VMAX * (1 + 1e-6):
            raise Abort(f"left arm / waist command step {other:.3f} rad/s above {OTHER_VMAX} rad/s")
        if self.phase in TRACKED_PHASES:
            err = np.abs(st.q[list(RIGHT_ARM_DDS)] - prev.right)
            if err.max() > self.cfg.max_track_err:
                raise Abort(f"right-arm tracking error {err.max():.3f} rad > {self.cfg.max_track_err}")
        if hand_due:
            if not _within(new.hand, p.hand_limits):
                raise Abort(f"hand command outside the limits: {np.round(new.hand, 4).tolist()}")
            if self.last_hand is not None:
                n0, q0 = self.last_hand
                hstep = np.abs(new.hand - q0) / ((self.n - n0) * dt)
                if (hstep > p.hand_vmax * (1 + 1e-3) + 1e-9).any():
                    raise Abort(f"hand command step {hstep.max():.3f} rad/s above the velocity limit")

    def _publish(self, cmd: Cmd, hand_due: bool) -> None:
        q = {**dict(zip(RIGHT_ARM_DDS, cmd.right, strict=True)), **dict(zip(LEFT_ARM_DDS, cmd.left, strict=True)),
             **dict(zip(WAIST_DDS, cmd.waist, strict=True))}
        kp = {**dict(zip(RIGHT_ARM_DDS + LEFT_ARM_DDS, ARM_KP + ARM_KP, strict=True)),
              **dict.fromkeys(WAIST_DDS, WAIST_KP)}
        kd = {**dict(zip(RIGHT_ARM_DDS + LEFT_ARM_DDS, ARM_KD + ARM_KD, strict=True)),
              **dict.fromkeys(WAIST_DDS, WAIST_KD)}
        tau = {}
        if self.cfg.send_tau_ff:
            tau = {**dict(zip(RIGHT_ARM_DDS, cmd.tau_r, strict=True)), **dict(zip(LEFT_ARM_DDS, cmd.tau_l, strict=True))}
        self.cmd = cmd
        self.io.send_arm(q, kp, kd, tau, self.weight, self.dt)
        self._sent = True
        if self.phase in ("to_hold", "play"):
            self.trail.append(cmd.vec())
        if hand_due:
            self.last_hand = (self.n, cmd.hand.copy())
            self.io.send_hand(cmd.hand, self.hand_dt)

    def _hand_due(self) -> bool:
        return self.n * self.dt >= self.hand_slot * self.hand_dt - 1e-9

    def _abort(self, reason: str) -> None:
        """Release at the measured pose when the state is fresh, else at the last command."""
        self.phases["abort"] = reason
        self.abort_phase = self.phase
        self._finish("aborted", reason)
        hold = self.cmd
        try:
            st = self.io.state()
            now = self.io.now()
            if st is not None and now - st.stamp <= self.cfg.state_timeout:
                lim, hlim = self.plan.arm_limits, self.plan.hand_limits
                hold = replace(hold, right=np.clip(st.q[list(RIGHT_ARM_DDS)], lim[:, 0], lim[:, 1]),
                               left=st.q[list(LEFT_ARM_DDS)].copy(), waist=st.q[list(WAIST_DDS)].copy())
                if now - st.hand_stamp <= self.cfg.state_timeout:
                    hold = replace(hold, hand=np.clip(st.hand_q, hlim[:, 0], hlim[:, 1]))
        except Exception as exc:        # noqa: BLE001 - the release must not depend on the state
            self.phases["abort_state_error"] = f"{type(exc).__name__}: {exc}"
        self.cmd = hold
        self._enter("release")

    def _release_tick(self) -> None:
        self.weight = max(0.0, self.w0 - (self.k + 1) / self.ramp_n)
        try:
            self._publish(self.cmd, self._hand_due())
        except Exception as exc:        # noqa: BLE001 - keep ramping; the next tick publishes again
            self.phases["release_errors"] = f"{type(exc).__name__}: {exc}"
        self.k += 1
        if self.weight <= 0.0:
            self.phases.setdefault("release", "ok")
            self._enter("done")

    def tick(self) -> bool:
        """One control tick: checks, then publish (module docstring). False once released."""
        if self.phase == "done":
            return False
        self._sent = False
        start = self.io.now()
        late = start - self.t_next
        self._tick_stats(start, late)
        if self.phase != "release":
            try:
                if late > OVERRUN_PERIODS * self.dt:
                    raise Abort(f"control loop overran: tick started {late * 1000:.1f} ms late (> {OVERRUN_PERIODS} "
                                f"periods of {self.dt * 1000:.0f} ms)")
                self._handle_requests()
                if self.phase != "release":
                    st = self._fresh_state()
                    phase = self.phase
                    cmd, finished = self._next()
                    hand_due = self._hand_due()
                    self._check(cmd, st, hand_due)
                    self._publish(cmd, hand_due)
                    self.k += 1
                    if finished:
                        self._advance(phase)
            except Abort as exc:
                self._abort(str(exc))
            except Exception as exc:    # noqa: BLE001 - any failure while commanding = abort and release
                self._abort(f"{type(exc).__name__}: {exc}")
        if self.phase == "release" and not self._sent:       # one arm message per tick
            self._release_tick()
        if self._hand_due():
            self.hand_slot = int(np.floor(self.n * self.dt / self.hand_dt + 1e-9)) + 1
        self.n += 1
        self.t_next += self.dt
        if start > self.t_next - self.dt / 2:      # more than half a period late: resynchronise, never burst
            self.t_next = start + self.dt
        return self.phase != "done"

    def _tick_stats(self, start: float, late: float) -> None:
        self._tick_n += 1
        self._late_ticks += int(late > self.dt)
        self._max_late = max(self._max_late, late)
        if self._last_start is not None:
            self._max_gap = max(self._max_gap, start - self._last_start)
        self._last_start = start

    def tick_stats(self) -> dict:
        """`{"ticks", "late_ticks" (> 1 period late), "max_late_ms", "max_gap_ms"}` (gap between tick starts)."""
        return {"ticks": self._tick_n, "late_ticks": self._late_ticks, "max_late_ms": round(self._max_late * 1000, 3),
                "max_gap_ms": round(self._max_gap * 1000, 3)}

    def _advance(self, phase: str) -> None:
        nxt = {"ramp_up": "to_hold", "to_hold": "wait_go", "wait_go": "play", "play": "go_home", "go_home": "release"}
        self.phases[phase] = "ok" if phase == "ramp_up" else f"{self.k * self.dt:.2f} s"
        if phase == "play":
            self._finish("completed", None)
        self._enter(nxt[phase])

    def _loop(self) -> None:
        if self.n == 0:                 # the schedule starts with the first tick, not at `_start` (thread start-up)
            self.t_next = self.io.now()
        try:
            while self.tick():
                self.io.sleep_until(self.t_next)
        finally:
            self._finished.set()

    def run(self, confirm: Callable[[], None] | None = None) -> dict:
        """The whole sequence -> {"status": "completed" | "stopped" | "aborted" | "refused", "reason", "phases"};
        "refused" = nothing was sent. Called from the main thread, SIGINT and SIGTERM are operator stops."""
        refused = self._start()
        if refused is not None:
            return refused
        threaded = confirm is not None
        previous = {}
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGINT, signal.SIGTERM):
                previous[sig] = signal.signal(sig, lambda s, f: self._on_signal(s, f, threaded))
        try:
            if confirm is not None:
                self._run_threaded(confirm)
            else:
                self._go.set()
                while self.phase != "done":
                    try:
                        self._loop()
                    except KeyboardInterrupt:          # only reachable without our handler (not the main thread)
                        self.request_stop("operator interrupt (KeyboardInterrupt)")
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
        out: dict = {"status": self.status or "aborted", "phases": dict(self.phases), "ticks": self.tick_stats()}
        if self.reason is not None:
            out["reason"] = self.reason
        if self.abort_phase is not None:
            out["abort_phase"] = self.abort_phase
        return out

    def _run_threaded(self, confirm: Callable[[], None]) -> None:
        """The operator side. It waits on `_finished`, never in `Thread.join`: a KeyboardInterrupt inside `join()`
        marks a RUNNING thread as stopped in CPython 3.12 (`_wait_for_tstate_lock` releases its tstate lock), so
        `is_alive()` turns False and the interpreter would exit mid-playback without waiting for it."""
        worker = threading.Thread(target=self._loop, name="arm_sdk_control", daemon=False)
        stage = "start"
        while not self._finished.is_set():
            try:
                if stage == "start":
                    stage = "hold"
                    try:
                        worker.start()
                    except RuntimeError as exc:        # nothing was published: the loop never ran
                        self.status, self.reason = "refused", f"the control thread did not start: {exc}"
                        self._finished.set()
                        return
                elif stage == "hold":
                    if self._at_hold.wait(0.02):
                        stage = "confirm"
                elif stage == "confirm":
                    stage = "wait"
                    if self.phase == "wait_go":
                        try:
                            confirm()
                        except Exception as exc:       # noqa: BLE001 - 'no go', a closed stdin (EOFError), ...
                            self.request_stop(f"operator: {type(exc).__name__}: {exc}")
                        else:
                            self._go.set()
                else:
                    self._finished.wait(0.05)
            except KeyboardInterrupt:
                stage = "wait"
                if self._signal_ki:
                    self._signal_ki = False
                else:
                    self.request_stop("operator interrupt (KeyboardInterrupt)")
        worker.join(1.0)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _num(d: Any, key: str) -> float:
    v = d.get(key) if isinstance(d, dict) else None
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else float("nan")


def _replay_failures(stats: dict) -> list[str]:
    """`SIM_GATE` on the trajectory replay (`sim_stats.json` "replay"). `not <=` / `not >=`: NaN (missing) fails."""
    rp = stats.get("replay")
    if not isinstance(rp, dict):
        return ["sim_stats.json has no replay metrics"]
    g, out = SIM_GATE, []
    table, self_d, lim = _num(rp, "table_max_depth"), _num(rp, "self_collision_max_depth"), \
        _num(rp, "limit_violation_ticks")
    arm_max, ref = _num(rp, "arm_max"), _num(rp, "reference_displacement")
    disp, err = _num(rp, "object_displacement"), _num(rp, "object_final_error")
    if not table <= g["table_max_depth_m"]:
        out.append(f"table penetration {table * 1000:.1f} mm > {g['table_max_depth_m'] * 1000:.1f} mm")
    force = _num(rp, "table_max_force")
    if not force <= g["table_max_force_n"]:
        out.append(f"table contact force {force:.1f} N > {g['table_max_force_n']:g} N")
    if not self_d <= g["self_collision_max_depth_m"]:
        out.append(f"self-collision {self_d * 1000:.1f} mm > {g['self_collision_max_depth_m'] * 1000:.1f} mm")
    if not lim <= g["limit_violation_ticks"]:
        out.append(f"joint-limit violations in {lim:g} ticks")
    if not arm_max < g["arm_tracking_max_rad"]:
        out.append(f"arm tracking max {arm_max:.3f} rad >= {g['arm_tracking_max_rad']} rad")
    if not ref >= g["min_reference_displacement_m"]:
        out.append(f"the reference moves the object only {ref * 100:.1f} cm")
    if not disp >= g["min_move_fraction"] * ref:
        out.append(f"object moved {disp * 100:.1f} cm < {g['min_move_fraction']} x reference {ref * 100:.1f} cm")
    if not err <= max(g["final_error_floor_m"], g["max_final_error_fraction"] * ref):
        out.append(f"object final error {err * 100:.1f} cm")
    if rp.get("object_fell") is not False:
        out.append("the object fell (or object_fell is missing)")
    drift, rot = _num(rp, "settle_drift"), _num(rp, "settle_rot_deg")
    if not drift <= g["settle_drift_max_m"]:
        out.append(f"the object does not rest where it is placed: it moved {drift * 1000:.1f} mm while settling "
                   f"(> {g['settle_drift_max_m'] * 1000:.1f} mm)")
    if not rot <= g["settle_rot_max_deg"]:
        out.append(f"the object does not rest where it is placed: it turned {rot:.2f} deg while settling "
                   f"(> {g['settle_rot_max_deg']:g} deg)")
    return out


def _phase_failures(name: str, ph: Any, table: tuple[str, float], self_max: float, contact_max: float | None,
                    track_max: float | None, force_max: float | None = None) -> list[str]:
    """`table` = ("steps", max robot-table contact steps) or ("depth", max penetration in m); `force_max` also caps
    the robot-table normal force (N)."""
    if not isinstance(ph, dict):
        return [f"{name}: no simulated metrics"]
    out = []
    self_d = _num(ph, "self_collision_max_depth")
    if table[0] == "steps":
        steps = _num(ph, "table_contact_steps")
        if not steps <= table[1]:
            out.append(f"{name}: the robot touches the table in {steps:g} physics steps "
                       f"(deepest {_num(ph, 'table_max_depth') * 1000:.1f} mm)")
    else:
        depth = _num(ph, "table_max_depth")
        if not depth <= table[1]:
            out.append(f"{name}: table contact {depth * 1000:.1f} mm > {table[1] * 1000:.1f} mm")
    if force_max is not None:
        force = _num(ph, "table_max_force")
        if not force <= force_max:
            out.append(f"{name}: table contact force {force:.1f} N > {force_max:g} N")
    if not self_d <= self_max:
        out.append(f"{name}: self-collision {self_d * 1000:.1f} mm > {self_max * 1000:.1f} mm")
    contact = _num(ph, "object_contact_steps")
    if contact_max is not None and not contact <= contact_max:
        out.append(f"{name}: the robot touches the object in {contact:g} physics steps")
    track = _num(ph, "arm_tracking_max")
    if track_max is not None and not track < track_max:
        out.append(f"{name}: arm tracking max {track:.3f} rad >= {track_max} rad")
    return out


TOLERANCE_CASES = {"table_high": (1.0, True), "table_low": (-1.0, False)}   # case: (sign of dz, edges grown by tol)


def _table_like_failures(label: str, run: Any) -> list[str]:
    """The table-tolerance thresholds on one simulated run: it completed; outside the playback no table or object
    contact, no self-collision, tracking in TO_HOLD / GO_HOME; inside it the replay's table depth and force,
    self-collision and tracking."""
    g = SIM_GATE
    if not isinstance(run, dict) or run.get("status") != "completed":
        status, reason = (run.get("status"), run.get("reason")) if isinstance(run, dict) else (None, None)
        return [f"{label}: the simulated run did not complete (status {status}, {reason})"]
    tg = (("steps", g["transfer_table_contact_steps"]), g["transfer_self_collision_max_depth_m"],
          g["transfer_object_contact_steps"])
    phases = run.get("phases", {})
    out = []
    for ph in TRANSFER_PHASES:
        track = g["transfer_arm_tracking_max_rad"] if ph in ("to_hold", "go_home") else None
        out += [f"{label}: {f}" for f in _phase_failures(ph, phases.get(ph), *tg, track)]
    out += [f"{label}: {f}" for f in _phase_failures("play", phases.get("play"), ("depth", g["table_max_depth_m"]),
                                                     g["self_collision_max_depth_m"], None,
                                                     g["arm_tracking_max_rad"], g["table_max_force_n"])]
    return out


def _tolerance_failures(rec: dict, table_tol: float) -> list[str]:
    """`SIM_GATE` on the runs simulated at the edges of the table placement tolerance (`table_tolerance_cases`): each
    case at its own edge (table_high: top +tol, every edge out by tol; table_low: top -tol, edges unchanged)."""
    g = SIM_GATE
    if not table_tol >= g["table_tol_min_m"] - 1e-12:
        return [(f"the table placement tolerance {table_tol * 100:.1f} cm is below the "
                 f"{g['table_tol_min_m'] * 100:.1f} cm the moves must be planned and simulated at")]
    runs = rec.get("table_tolerance_runs")
    by_case = {r.get("case"): r for r in runs if isinstance(r, dict)} if isinstance(runs, list) else {}
    out = []
    for case in g["table_tolerance_cases"]:
        run = by_case.get(case)
        label = f"table at the tolerance edge ({case})"
        if run is None:
            out.append(f"{label}: not simulated")
            continue
        sign, grown = TOLERANCE_CASES[case]
        if not abs(_num(run, "dz_m") - sign * table_tol) <= 1e-9:
            out.append(f"{label}: simulated with the top {_num(run, 'dz_m') * 100:+.1f} cm, not "
                       f"{sign * table_tol * 100:+.1f} cm")
        if not abs(_num(run, "grow_m") - (table_tol if grown else 0.0)) <= 1e-9:
            out.append(f"{label}: simulated with the edges out by {_num(run, 'grow_m') * 100:.1f} cm, not "
                       f"{(table_tol if grown else 0.0) * 100:.1f} cm")
        out += _table_like_failures(label, run)
    return out


def _object_failures(label: str, crit: Any) -> list[str]:
    """The replay's object thresholds on a simulated run's playback (`object_criteria`)."""
    g = SIM_GATE
    if not isinstance(crit, dict):
        return [f"{label}: the object's motion in the simulated run is not recorded"]
    ref, disp, err = _num(crit, "reference_displacement_m"), _num(crit, "displacement_m"), _num(crit, "final_error_m")
    out = []
    if not ref >= g["min_reference_displacement_m"]:
        out.append(f"{label}: the reference moves the object only {ref * 100:.1f} cm")
    if not disp >= g["min_move_fraction"] * ref:
        out.append(f"{label}: object moved {disp * 100:.1f} cm < {g['min_move_fraction']} x reference "
                   f"{ref * 100:.1f} cm")
    if not err <= max(g["final_error_floor_m"], g["max_final_error_fraction"] * ref):
        out.append(f"{label}: object final error {err * 100:.1f} cm")
    if crit.get("fell") is not False:
        out.append(f"{label}: the object fell (or fell is missing)")
    return out


def _options_failures(label: str, run: Any, want: dict) -> list[str]:
    """Each simulated run must record the package's run options (a drill simulated at speed scale
    1 in a 0.5 package)."""
    got = run.get("run_options") if isinstance(run, dict) else None
    if not isinstance(got, dict):
        return [f"{label}: the run options it was simulated with are not recorded"]
    out = []
    for key, value in want.items():
        v = got.get(key)
        if isinstance(value, bool):
            same = v is value
        else:
            same = (value is not None and isinstance(v, (int, float)) and not isinstance(v, bool)
                    and abs(float(v) - float(value)) <= 1e-9)
        if not same:
            out.append(f"{label}: simulated with {key} {v}, the package records {value}")
    return out


def _started_failures(label: str, run: Any, index: int | None) -> list[str]:
    if index is None or not isinstance(run, dict) or run.get("status") != "completed":
        return []
    started = run.get("started_home")
    return [] if started == index else [f"{label}: the run script started from home #{started}, not #{index}"]


def _placement_failures(rec: dict, xy_tol: float, yaw_tol: float, index: int | None) -> list[str]:
    """`SIM_GATE` on the runs with the object at every corner of its placement tolerance
    (`placement_runs`): the table-tolerance thresholds and the object criteria against the nominal reference."""
    g = SIM_GATE
    if not (xy_tol >= g["object_xy_tol_min_m"] - 1e-12 and yaw_tol >= g["object_yaw_tol_min_deg"] - 1e-9):
        return [(f"the object placement tolerance +-{xy_tol * 100:.1f} cm / +-{yaw_tol:g} deg is below the "
                 f"+-{g['object_xy_tol_min_m'] * 100:.1f} cm / +-{g['object_yaw_tol_min_deg']:g} deg the moves must "
                 "be planned and simulated at")]
    runs = rec.get("placement_runs")
    by_case = {r.get("case"): r for r in runs if isinstance(r, dict)} if isinstance(runs, list) else {}
    out = []
    for case in g["placement_cases"]:
        run = by_case.get(case)
        label = f"object at the placement corner {case}"
        if run is None:
            out.append(f"{label}: not simulated")
            continue
        sx, sy, sa = PLACEMENT_CASES[case]
        want = (sx * xy_tol, sy * xy_tol, sa * yaw_tol)
        got = (_num(run, "dx_m"), _num(run, "dy_m"), _num(run, "dyaw_deg"))
        if not all(abs(a - b) <= 1e-9 for a, b in zip(got, want, strict=True)):
            out.append(f"{label}: simulated at dx, dy, dyaw = {got}, not {want}")
        out += _table_like_failures(label, run) + _started_failures(label, run, index)
        if run.get("status") == "completed":
            out += _object_failures(label, run.get("object_criteria"))
    return out


def _offset_ok(off: np.ndarray, k: int, sign: float, tol: float, clipped: set[str], kind: str) -> bool:
    return bool(sign * off[k] >= SIM_GATE["state_edge_min"] * tol - 1e-12 or f"{kind} {k}" in clipped)


def _measured_offsets(run: dict, home_q: np.ndarray, start_hand: Any) -> tuple[np.ndarray, np.ndarray] | None:
    """`(arm (14,), hand (7,))`: the start the run script MEASURED in the simulated run, relative to the home and the
    simulated start hand, or None when not recorded."""
    ms = run.get("measured_start")
    if not isinstance(ms, dict) or not isinstance(start_hand, list) or len(start_hand) != 7:
        return None
    q, hand = np.asarray(ms.get("q17", []), dtype=np.float64), np.asarray(ms.get("hand", []), dtype=np.float64)
    if q.shape != (17,) or hand.shape != (7,) or not (np.isfinite(q).all() and np.isfinite(hand).all()):
        return None
    return q[:14] - np.asarray(home_q, dtype=np.float64)[:14], hand - np.asarray(start_hand, dtype=np.float64)


def _state_failures(rec: dict, home_tol: float, index: int | None, home_q: np.ndarray | None = None,
                    start_hand: Any = None) -> list[str]:
    """`SIM_GATE` on the runs from the measured states `--execute` accepts (`state_runs`): every
    axis extreme and at least `state_corner_runs_min` seeded corners of the arm box, and the Dex3 start at
    +-START_HAND_TOL, each MEASURED by the run script at least `state_edge_min` of its tolerance out (a joint at its
    range limit is recorded as clipped; the joints a run does not test stay within `state_rest_max_frac` of the arm
    tolerance: holding the hand at its edge while the object settles moves the arm < 1 mrad), each meeting the
    table-tolerance thresholds."""
    g = SIM_GATE
    spare = g["state_rest_max_frac"]
    runs = rec.get("state_runs")
    by_name = {r.get("name"): r for r in runs if isinstance(r, dict)} if isinstance(runs, list) else {}
    corners = sum(1 for n in by_name if isinstance(n, str) and n.startswith("arm_corner_"))
    need = state_run_names(max(corners, g["state_corner_runs_min"]))
    out = [f"measured state {n}: not simulated" for n in need if n not in by_name]
    for name in need:
        run = by_name.get(name)
        if run is None:
            continue
        label = f"measured state {name}"
        clipped = {str(c) for c in run.get("clipped", [])} if isinstance(run.get("clipped"), list) else set()
        offs = None if home_q is None else _measured_offsets(run, home_q, start_hand)
        if offs is None or not abs(_num(run, "home_tol") - home_tol) <= 1e-9:
            out.append(f"{label}: its measured start or home tolerance are not recorded as simulated")
            continue
        arm, hand = offs
        if name.startswith("arm_axis_"):
            j, sign = int(name[9:11]), (1.0 if name.endswith("+") else -1.0)
            ok = _offset_ok(arm, j, sign, home_tol, clipped, "arm") and \
                float(np.abs(np.delete(arm, j)).max()) <= spare * home_tol
        elif name.startswith("arm_corner_"):
            ok = all(_offset_ok(arm, k, float(np.sign(arm[k]) or 1.0), home_tol, clipped, "arm") for k in range(14))
        else:
            sign = 1.0 if name == "hand+" else -1.0
            ok = all(_offset_ok(hand, k, sign, START_HAND_TOL, clipped, "hand") for k in range(7)) and \
                float(np.abs(arm).max()) <= spare * home_tol
        if not ok:
            out.append(f"{label}: not started at the edge of the tolerance it tests (>= {g['state_edge_min']:g} of it)")
        out += _table_like_failures(label, run) + _started_failures(label, run, index)
    return out


def _run_options_wanted(tr: dict) -> dict:
    """The run options every simulated run of the package must record (`_options_failures`)."""
    sim = simulated_conditions({"transfers": tr})
    return {"hold_speed": sim["hold_speed"], "speed_scale": sim["speed_scale"], "send_tau_ff": sim["send_tau_ff"],
            "hold_waist": True, "max_track_err": sim["max_track_err"], "hold_timeout_s": sim["hold_timeout_s"]}


def _home_failures(rec: Any, home_q: np.ndarray, table_tol: float = 0.0, tr: dict | None = None,
                   index: int | None = None, home_tol: float | None = None) -> list[str]:
    """`SIM_GATE` on one checked home's simulated run, operator-stop drills, table-tolerance runs and the
    run's object criteria, the object placement corners, the measured-state runs and every run's options
    (`sim_stats.json` "transfers" `tr`)."""
    g = SIM_GATE
    tr = tr if isinstance(tr, dict) else {}
    if not isinstance(rec, dict):
        return ["no simulated run"]
    hq = rec.get("home_q")
    if not isinstance(hq, list) or len(hq) != 17 or not _close(np.asarray(hq, dtype=np.float64), home_q):
        return ["its sim_stats.json home differs from trajectory.npz"]
    out = [f"{key}: not planned" for key in ("to_hold_plan", "go_home_plan")
           if not isinstance(rec.get(key), dict) or rec[key].get("ok") is not True]
    tg = (("steps", g["transfer_table_contact_steps"]), g["transfer_self_collision_max_depth_m"],
          g["transfer_object_contact_steps"])
    run = rec.get("run")
    if not isinstance(run, dict) or run.get("status") != "completed":
        status = run.get("status") if isinstance(run, dict) else None
        out.append(f"the simulated run did not complete (status {status})")
    else:
        for ph in TRANSFER_PHASES:
            track = g["transfer_arm_tracking_max_rad"] if ph in ("to_hold", "go_home") else None
            out += _phase_failures(ph, run.get("phases", {}).get(ph), *tg, track)
        sc = _num(tr, "speed_scale")
        out += _object_failures(f"the simulated run (speed scale {sc:g})", run.get("object_criteria"))
    drills = rec.get("drills")
    if not isinstance(drills, list) or not drills:
        return out + ["no operator-stop drill"]
    for dr in drills:
        label = f"stop drill {dr.get('stop') if isinstance(dr, dict) else '?'}"
        if not isinstance(dr, dict) or dr.get("status") != "stopped":
            out.append(f"{label}: did not stop and go home (status {dr.get('status') if isinstance(dr, dict) else None})")
            continue
        before_play = str(dr.get("stop", "")).startswith("to_hold")
        lim = tg if before_play else (("depth", g["stop_table_max_depth_m"]), g["stop_self_collision_max_depth_m"], None)
        force = None if before_play else g["stop_table_max_force_n"]
        for ph in ("go_home", "release"):
            track = g["stop_arm_tracking_max_rad"] if ph == "go_home" else None
            out += [f"{label}: {f}" for f in _phase_failures(ph, dr.get("phases", {}).get(ph), *lim, track, force)]
    out += _tolerance_failures(rec, table_tol)
    out += _placement_failures(rec, _num(tr, "object_xy_tol_m"), _num(tr, "object_yaw_tol_deg"), index)
    tol = _num(tr, "home_tol_rad") if home_tol is None else float(home_tol)
    out += _state_failures(rec, tol, index, home_q, tr.get("start_hand"))
    want = _run_options_wanted(tr)
    named = [("the simulated run", run)] + [(f"stop drill {d.get('stop')}", d) for d in drills if isinstance(d, dict)]
    for key, label in (("table_tolerance_runs", "table run"), ("placement_runs", "placement run"),
                       ("state_runs", "measured-state run")):
        named += [(f"{label} {r.get('case', r.get('name'))}", r) for r in rec.get(key) or [] if isinstance(r, dict)]
    for label, r in named:
        out += _options_failures(label, r, want)
    return out


def transfer_gate(stats: dict, home_q: np.ndarray, home_tol: float | None = None) -> tuple[list[int], list[str]]:
    """`(indices of the trajectory.npz homes whose simulated runs pass SIM_GATE, why the others do not)`. `home_tol`:
    the package's (trajectory.npz); the measured-state runs must have been simulated at it."""
    tr = stats.get("transfers")
    if not isinstance(tr, dict) or not isinstance(tr.get("homes"), list):
        return [], ["sim_stats.json has no simulation of the moves to the hold pose and home"]
    by_index = {r["index"]: r for r in tr["homes"] if isinstance(r, dict) and isinstance(r.get("index"), int)}
    table_tol = _num(tr, "table_tol_m")
    if not table_tol >= 0.0:
        return [], ["sim_stats.json does not record the table placement tolerance the moves were simulated at"]
    allowed, fails = [], []
    for k in range(len(home_q)):
        rec = by_index.get(k)
        name = rec.get("name") if isinstance(rec, dict) else "?"
        why = _home_failures(rec, home_q[k], table_tol, tr, k, home_tol)
        if why:
            fails += [f"home #{k} '{name}': {w}" for w in why]
        else:
            allowed.append(k)
    for rec in tr["homes"]:
        if isinstance(rec, dict) and rec.get("index") is None:
            why = [str(rec[key].get("reason")) for key in ("to_hold_plan", "go_home_plan")
                   if isinstance(rec.get(key), dict) and not rec[key].get("ok")]
            fails.append(f"home '{rec.get('name')}' is not in the package, no checked path: {'; '.join(why)}")
    return allowed, fails


def _table_failures(package: Path, stats: dict) -> list[str]:
    """The table the lab is told to set up (package.json "table") must be the one simulated (sim_stats.json)."""
    try:
        told = json.loads((package / "package.json").read_text()).get("table")
    except (OSError, ValueError) as exc:
        return [f"cannot read package.json: {exc}"]
    sim = stats.get("table")
    if not isinstance(told, dict) or not isinstance(sim, dict):
        return [("package.json or sim_stats.json does not state the table rectangle, so the lab table cannot be "
                 "matched to the simulated one")]
    diff = [k for k in TABLE_KEYS if not abs(_num(told, k) - _num(sim, k)) <= 1e-6]
    if diff:
        return [(f"the table in package.json differs from the simulated table ({', '.join(diff)}): re-export "
                 "instead of editing the placement")]
    tr = stats.get("transfers")
    if not isinstance(tr, dict):
        return []                     # no simulation of the moves at all: `transfer_gate` refuses it
    tol_sim = _num(tr, "table_tol_m")
    if not abs(_num(told, "placement_tolerance_m") - tol_sim) <= 1e-9:
        return [(f"package.json allows the table +-{_num(told, 'placement_tolerance_m') * 100:.1f} cm from the "
                 f"simulated one, but the moves were planned and simulated at +-{tol_sim * 100:.1f} cm")]
    try:                              # the object placement tolerance the lab is told
        obj = json.loads((package / "package.json").read_text()).get("object_placement_tolerance")
    except (OSError, ValueError) as exc:
        return [f"cannot read package.json: {exc}"]
    xy, yaw = _num(obj, "xy_m"), _num(obj, "yaw_deg")
    if not (abs(xy - _num(tr, "object_xy_tol_m")) <= 1e-9 and abs(yaw - _num(tr, "object_yaw_tol_deg")) <= 1e-9):
        return [(f"package.json allows the object +-{xy * 100:.1f} cm / +-{yaw:g} deg from its stated pose, but the "
                 f"moves were planned and simulated at +-{_num(tr, 'object_xy_tol_m') * 100:.1f} cm / "
                 f"+-{_num(tr, 'object_yaw_tol_deg'):g} deg")]
    return []


def simulated_conditions(stats: dict) -> dict:
    """The run-script options the package's simulation of the run used (`sim_stats.json` "transfers"): `{"hold_speed",
    "send_tau_ff", "speed_scale"}`, a missing one as None. `--execute` runs only under exactly these
    (`condition_failures`)."""
    tr = stats.get("transfers") if isinstance(stats, dict) else None
    tr = tr if isinstance(tr, dict) else {}
    hs, sc, ff, hand = tr.get("hold_speed_rad_s"), tr.get("speed_scale"), tr.get("send_tau_ff"), tr.get("start_hand")
    num = (int, float)
    ok_hand = (isinstance(hand, list) and len(hand) == 7
               and all(isinstance(v, num) and not isinstance(v, bool) and np.isfinite(v) for v in hand))

    def number(v: Any) -> float | None:
        return float(v) if isinstance(v, num) and not isinstance(v, bool) and np.isfinite(v) else None

    return {"hold_speed": number(hs), "send_tau_ff": ff if isinstance(ff, bool) else None, "speed_scale": number(sc),
            "start_hand": [float(v) for v in hand] if ok_hand else None,
            "max_track_err": number(tr.get("max_track_err_rad")), "hold_timeout_s": number(tr.get("hold_timeout_s"))}


def condition_failures(sim: dict, cfg: Config) -> list[str]:
    """Why `--execute` must refuse these options: each run-script option the simulation depends on must be the value
    the package simulated (`--no-tau-ff` and a faster `--hold-speed` were accepted
    unsimulated, and from a checked home the arm then touched the table in TO_HOLD and GO_HOME)."""
    out = []
    for key, flag, value in (("send_tau_ff", "--no-tau-ff", cfg.send_tau_ff), ("hold_speed", "--hold-speed",
                                                                                cfg.hold_speed),
                             ("speed_scale", "--speed-scale", cfg.speed_scale)):
        want = sim.get(key)
        if want is None:
            out.append(f"sim_stats.json does not record the simulated {key}, so {flag} cannot be matched to it")
        elif (value != want) if isinstance(want, bool) else abs(float(value) - float(want)) > 1e-9:
            shown = (lambda v: ("on" if v else "off") if isinstance(v, bool) else f"{v:g}")
            out.append(f"{flag}: the package simulated {key} {shown(want)}, this run asks for {shown(value)}; only "
                       "the simulated value is gated (re-export to simulate another)")
    if not cfg.hold_waist:
        out.append("--no-waist: the simulation welds the waist at 0 and never ran with the waist held at a measured "
                   "pose, so --execute refuses it")
    if sim.get("start_hand") is None:
        out.append("sim_stats.json does not record the Dex3 hand pose the run was simulated from, so the measured "
                   "hand cannot be checked against it")
    for key, flag, value in (("max_track_err", "--max-track-err", cfg.max_track_err),
                             ("hold_timeout_s", "--hold-timeout", cfg.hold_timeout_s)):
        want = sim.get(key)                  # accepted only at or below the simulated value
        if want is None:
            out.append(f"sim_stats.json does not record the simulated {key}, so {flag} cannot be matched to it")
        elif not float(value) <= float(want) + 1e-9:
            out.append(f"{flag}: the package simulated {key} {want:g}, this run asks for {float(value):g}; only values "
                       "at or below the simulated one are accepted")
    return out


def gate_report(package: Path) -> dict:
    """`{"failures", "homes_allowed", "home_failures", "simulated"}`: `failures` empty = `--execute` may run, only
    from the checked homes in `homes_allowed` and only under the `simulated` run options (`condition_failures`);
    `home_failures` say why the other homes are refused (`SIM_GATE`)."""
    try:
        stats = json.loads((package / "sim_stats.json").read_text())
        sha = file_sha256(package / "trajectory.npz")
        plan = Plan.load(package / "trajectory.npz")
    except (OSError, ValueError, KeyError) as exc:
        return {"failures": [f"cannot read sim_stats.json / trajectory.npz: {exc}"], "homes_allowed": [],
                "home_failures": [], "simulated": simulated_conditions({})}
    out = []
    if stats.get("trajectory_sha256") != sha:
        out.append("sim_stats.json was not computed on this trajectory.npz (sha256 missing or different)")
    out += _table_failures(package, stats)
    out += _replay_failures(stats)
    tr = stats.get("transfers")
    htol = float(plan.home_tol) if plan.home_tol.size == 1 else float("nan")
    if isinstance(tr, dict) and not abs(_num(tr, "home_tol_rad") - htol) <= 1e-12:
        out.append("sim_stats.json simulated another home tolerance than trajectory.npz stores")
    allowed, home_fail = transfer_gate(stats, plan.home_q, htol)
    if not allowed:
        out.append("no checked home passes the gate on the moves to the hold pose and home: "
                   + ("; ".join(home_fail) if home_fail else "the package has no checked home"))
    return {"failures": out, "homes_allowed": allowed, "home_failures": home_fail,
            "simulated": simulated_conditions(stats)}


def sim_gate_failures(package: Path) -> list[str]:
    """Why `--execute` must refuse this package (`SIM_GATE`, `gate_report`); [] = it may run."""
    return gate_report(package)["failures"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--package", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--execute", action="store_true", help="talk to the robot (default: dry-run only)")
    ap.add_argument("--iface", default=None, help="network interface of the robot (required with --execute)")
    ap.add_argument("--domain", type=int, default=0)
    ap.add_argument("--hold-speed", type=float, default=0.3, help="rad/s to the hold pose and home")
    ap.add_argument("--hold-timeout", type=float, default=120.0,
                    help="s at the hold pose without 'go' -> go home (--execute: at most the simulated value)")
    ap.add_argument("--max-track-err", type=float, default=MAX_TRACK_ERR,
                    help=f"abort above this right-arm error (rad, at most {MAX_TRACK_ERR}; --execute: at most the "
                    "simulated value)")
    ap.add_argument("--speed-scale", type=float, default=None,
                    help="<= 1 slows the playback (default: the value the package simulated)")
    ap.add_argument("--no-tau-ff", action="store_true", help="do not send the gravity feed-forward torque")
    ap.add_argument("--no-waist", action="store_true",
                    help="hold the waist at its measured start pose instead of 0 (it stays in the message)")
    ap.add_argument("--hand-state-topic", default=HAND_STATE_TOPIC,
                    help="Dex3 state topic (UNVERIFIED: the SDK C++ example subscribes rt/lf/dex3/right/state)")
    args = ap.parse_args(argv)
    if not args.hand_state_topic.startswith("rt/"):
        print("--hand-state-topic must be a DDS topic starting with 'rt/'", file=sys.stderr)
        return 2
    if (args.speed_scale is not None and not 0.0 < args.speed_scale <= 1.0) or not 0.0 < args.hold_speed <= OTHER_VMAX \
            or not args.hold_timeout > 0 \
            or not 0.0 < args.max_track_err <= MAX_TRACK_ERR:
        print(f"speed-scale must be in (0, 1], hold-speed in (0, {OTHER_VMAX}] rad/s, hold-timeout > 0 s and "
              f"max-track-err in (0, {MAX_TRACK_ERR}] rad", file=sys.stderr)
        return 2
    if args.execute and args.iface is None:
        print("--execute needs --iface", file=sys.stderr)
        return 2
    try:
        plan = Plan.load(args.package / "trajectory.npz")
    except (OSError, ValueError) as exc:
        print(f"REFUSED, cannot load the trajectory: {exc}", file=sys.stderr)
        return 2
    errs = plan.validate()
    if errs:
        print("REFUSED, the trajectory is invalid:\n  " + "\n  ".join(errs), file=sys.stderr)
        return 2
    report = gate_report(args.package)
    gate = report["failures"]
    sim = report["simulated"]
    scale = args.speed_scale if args.speed_scale is not None else (sim["speed_scale"] or 1.0)
    if not 0.0 < scale <= 1.0:
        print(f"REFUSED: the package's simulated speed scale {scale:g} is not in (0, 1]", file=sys.stderr)
        return 2
    hand0 = None if sim["start_hand"] is None else tuple(sim["start_hand"])
    cfg = Config(hold_speed=args.hold_speed, max_track_err=args.max_track_err, speed_scale=scale,
                 send_tau_ff=not args.no_tau_ff, hold_waist=not args.no_waist, hold_timeout_s=args.hold_timeout,
                 start_hand=hand0)
    if not args.execute:
        k0 = report["homes_allowed"][0] if report["homes_allowed"] else 0
        q0 = np.zeros(35)
        q0[list(RIGHT_ARM_DDS + LEFT_ARM_DDS + WAIST_DDS)] = plan.home_q[k0]
        result = Replayer(plan, SimulatedIO(q0=q0, hand0=np.zeros(7) if hand0 is None else np.asarray(hand0)),
                          cfg).run()
        gate = gate + condition_failures(report["simulated"], cfg)
        print(json.dumps({"mode": "dry-run (simulated I/O, no robot contacted)", "home": k0, **result,
                          "sim_gate": {"execute_allowed": not gate, "failures": gate,
                                       "simulated_run_options": report["simulated"],
                                       "homes_allowed": report["homes_allowed"],
                                       "home_failures": report["home_failures"], "thresholds": SIM_GATE}},
                         indent=1))
        return 0 if result["status"] == "completed" else 3
    gate = gate + condition_failures(report["simulated"], cfg)
    if gate:
        print("REFUSED: the package's own simulation fails the execution gate (README, 'Execution gate'):\n  "
              + "\n  ".join(gate), file=sys.stderr)
        return 4
    cfg = replace(cfg, homes=tuple(report["homes_allowed"]))
    print("EXECUTE: the robot will move. Robot standing in Regular mode, e-stop in hand, workspace clear.")
    try:
        if input("type 'yes' to connect: ").strip() != "yes":
            return 1
    except (KeyboardInterrupt, EOFError):
        return 1
    io = SdkIO(args.iface, args.domain, args.hand_state_topic)

    def _confirm() -> None:
        if input("at the hold pose; type 'go' to play the trajectory (anything else / Ctrl-C: go home): ").strip() \
                != "go":
            raise Abort("did not type 'go'")

    result = Replayer(plan, io, cfg).run(confirm=_confirm)
    print(json.dumps(result, indent=1))
    return 0 if result["status"] == "completed" else 3


if __name__ == "__main__":
    sys.exit(main())
