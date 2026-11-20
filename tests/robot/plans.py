"""Run-script plans for the robot-path tests: give a `g1_sdk_replay.Plan` a checked home with straight paths (the unit
tests' stand-in for the planned, collision-checked paths of `fastdo.robot.transfer`)."""

from __future__ import annotations

import dataclasses

import numpy as np

from fastdo.robot import g1_sdk_replay as S

BODY17 = list(S.RIGHT_ARM_DDS + S.LEFT_ARM_DDS + S.WAIST_DDS)     # (35,) body vector -> right, left, waist (17,)


def with_home(plan: S.Plan, *bodies: np.ndarray, tol: float = 0.08,
              via: list[np.ndarray] | None = None) -> S.Plan:
    """`plan` whose checked homes are the right arm / left arm / waist of each `(35,)` body vector, each with the path
    home -> (`via` waypoints, `(17,)` right/left/waist each) -> trajectory start and the straight path trajectory end ->
    home. Path rows: zero feed-forward torque except the trajectory's own start / end rows."""
    hold, end = plan.hold_vec(), plan.end_vec()
    homes, to_hold, go_home = [], [], []
    for body in bodies:
        home = np.asarray(body, dtype=np.float64)[BODY17]
        first = np.concatenate([home, np.zeros(14), hold[S.HAND]])
        mids = [np.concatenate([np.asarray(v, dtype=np.float64), np.zeros(14), hold[S.HAND]]) for v in via or []]
        to_hold.append(np.vstack([first, *mids, hold]))
        go_home.append(np.vstack([end, np.concatenate([home, np.zeros(14), end[S.HAND]])]))
        homes.append(home)

    def index(parts: list[np.ndarray]) -> np.ndarray:
        return np.concatenate([[0], np.cumsum([len(p) for p in parts])]).astype(np.float64)

    return dataclasses.replace(plan, home_q=np.asarray(homes), home_tol=np.asarray(tol, dtype=np.float64),
                               to_hold_path=np.vstack(to_hold), to_hold_index=index(to_hold),
                               go_home_path=np.vstack(go_home), go_home_index=index(go_home))


def body(plan: S.Plan, home: int = 0) -> np.ndarray:
    """`(35,)` body vector at the plan's checked home `home`."""
    q = np.zeros(35)
    q[BODY17] = plan.home_q[home]
    return q
