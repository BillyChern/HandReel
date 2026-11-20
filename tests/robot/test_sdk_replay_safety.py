"""Safety behaviour of the package run script `fastdo.robot.g1_sdk_replay`, on its simulated I/O.

The lab runs the script on a real G1; these tests pin what it must do when something goes wrong:
- every abort ramps the `rt/arm_sdk` weight down, PUBLISHING every tick, also when the abort is stale state (the
  release must not read state);
- Ctrl-C (SIGINT) and SIGTERM, at the 'go' prompt or during playback, go home and release (xr_teleoperate's
  `ctrl_dual_arm_go_home` in its main loop's `finally`);
- arm_sdk keeps being published while the script waits for the operator (one control thread at 250 Hz, as
  xr_teleoperate `_ctrl_motor_state` and the SDK's arm7 example `RecurrentThread`);
- each per-tick safety check aborts on its own fault, and the faulty command is never sent.

Plans are built here directly (no MuJoCo): a 1 s smooth right-arm motion at 250 Hz, a hand motion at 100 Hz, and a
checked home with straight paths to the trajectory and back (`tests.robot.plans.with_home`; the run script refuses a
measured pose that is not at a checked home).
"""

from __future__ import annotations

import _thread
import dataclasses
import signal
import time
from collections.abc import Callable
from types import SimpleNamespace

import numpy as np
import pytest

from fastdo.robot import g1_constants as C
from fastdo.robot import g1_sdk_replay as S
from fastdo.robot.trajectory import TrajectoryLimits
from tests.robot.plans import with_home

RIGHT = list(S.RIGHT_ARM_DDS)
LEFT = list(S.LEFT_ARM_DDS)
DT = 0.004


def make_plan(seconds: float = 1.0) -> S.Plan:
    lim = TrajectoryLimits()
    n = round(seconds / DT) + 1
    s = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, n))
    q0 = np.array([-0.6, -0.3, 0.0, 0.3, 0.0, 0.0, 0.0])
    q1 = np.array([-0.8, -0.5, 0.2, 0.2, 0.3, 0.2, 0.0])
    m = round(seconds / 0.01) + 1
    hb = lim.hand_bounds()
    h0 = hb.mean(axis=1)
    sh = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, m))
    hand_q = h0 + sh[:, None] * (0.2 * (hb[:, 1] - h0))
    empty = np.zeros((0, S.CMD_SIZE))
    plan = S.Plan(arm_t=np.arange(n) * DT, arm_q=q0 + s[:, None] * (q1 - q0), arm_tau_ff=np.zeros((n, 7)),
                  hand_t=np.arange(m) * 0.01, hand_q=hand_q, left_park=np.asarray(C.LEFT_ARM_PARK),
                  left_tau_ff=np.zeros(7), arm_limits=lim.arm_bounds(), hand_limits=hb,
                  arm_vmax=np.asarray(lim.arm_vmax), hand_vmax=np.asarray(lim.hand_vmax), home_q=np.zeros((0, 17)),
                  home_tol=np.asarray(0.08), to_hold_path=empty, to_hold_index=np.zeros(1), go_home_path=empty,
                  go_home_index=np.zeros(1))
    plan = with_home(plan, body_at_hold(plan))                      # checked home = the hold pose
    assert plan.validate() == []
    return plan


def body_at_hold(plan: S.Plan, right_offset: float = 0.0) -> np.ndarray:
    q = np.zeros(35)
    q[RIGHT] = plan.arm_q[0] + right_offset
    q[LEFT] = plan.left_park
    return q


class LoggedIO(S.SimulatedIO):
    """Records every arm_sdk / hand message; `hooks` = [(virtual time, callable(io))] fired once, in order."""

    def __init__(self, *args, hooks: list | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.arm_log: list[tuple[float, float, dict, dict]] = []
        self.hand_log: list[tuple[float, np.ndarray]] = []
        self.hooks = list(hooks or [])

    def send_arm(self, q, kp, kd, tau, weight, dt) -> None:
        self.arm_log.append((self.t, float(weight), dict(q), dict(kp)))
        super().send_arm(q, kp, kd, tau, weight, dt)

    def send_hand(self, q, dt) -> None:
        self.hand_log.append((self.t, np.array(q, dtype=np.float64)))
        super().send_hand(q, dt)

    def sleep_until(self, t: float) -> None:
        super().sleep_until(t)
        if self.t > 600.0:                         # a leaked control thread must not spin forever (non-daemon)
            raise RuntimeError("virtual clock ran past 600 s")
        while self.hooks and self.t >= self.hooks[0][0]:
            self.hooks.pop(0)[1](self)

    def weights(self) -> np.ndarray:
        return np.array([e[1] for e in self.arm_log])

    def right(self) -> np.ndarray:
        return np.array([[e[2][i] for i in RIGHT] for e in self.arm_log])


class StaleAfter(LoggedIO):
    """`rt/lowstate` (or, with `hand=True`, the Dex3 state) stops arriving at virtual time `t_stale`."""

    def __init__(self, t_stale: float, *args, hand: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.t_stale, self.hand = t_stale, hand

    def state(self) -> S.State:
        st = super().state()
        if self.t >= self.t_stale:
            st = dataclasses.replace(st, **{("hand_stamp" if self.hand else "stamp"): self.t_stale})
        return st


def interrupt(send: Callable[[], None]) -> Callable[[LoggedIO], None]:
    """Hook: deliver an interrupt, then hold the (virtual-clock) control loop until the replayer registered it, as
    a real-time loop would be at most a tick ahead."""
    def hook(io: LoggedIO) -> None:
        before = io.replayer.stop_count
        send()
        deadline = time.monotonic() + 3.0
        while io.replayer.stop_count == before and time.monotonic() < deadline:
            time.sleep(0.001)
    return hook


def replay(plan: S.Plan, io: LoggedIO, cfg: S.Config, confirm: Callable[[], None] | None = None) -> dict:
    rp = S.Replayer(plan, io, cfg)
    io.replayer = rp
    return rp.run(confirm=confirm)


def release_start(w: np.ndarray) -> int:
    """Index of the first arm message whose weight is below the previous one (the release ramp)."""
    down = np.flatnonzero(np.diff(w) < -1e-12)
    assert down.size, "the weight never ramps down"
    return int(down[0]) + 1


def assert_smooth_release(io: LoggedIO, cfg: S.Config) -> int:
    """The weight ends at 0, never rises again once it falls, and falls at most dt/ramp_s per message, so the
    release takes ramp_s * (weight at the start of it) of published ticks. Returns the release start index."""
    w = io.weights()
    k = release_start(w)
    assert w[-1] == 0.0 and io.weight == 0.0
    assert (np.diff(w[k - 1:]) <= 1e-12).all()
    assert np.abs(np.diff(w)).max() <= DT / cfg.ramp_s * (1 + 1e-6)
    assert len(w) - k >= round(w[k - 1] * cfg.ramp_s / DT) - 1
    return k


class _Guard(Exception):
    """Raised by the tests' own signal handlers: a signal the run script failed to handle fails the test instead of
    aborting the pytest session (SIGINT) or killing it (SIGTERM)."""


@pytest.fixture(autouse=True)
def sigint_guard():
    def guard(signum, frame):
        raise _Guard(f"signal {signum} reached the test's handler")

    previous = signal.signal(signal.SIGINT, guard)
    try:
        yield guard
    finally:
        signal.signal(signal.SIGINT, previous)


# --- an abort on stale state still ramps the weight down, publishing every tick ------------------------

def test_stale_state_abort_ramps_the_weight_down_holding_the_last_command() -> None:
    plan = make_plan()
    io = StaleAfter(2.5, q0=body_at_hold(plan), hand0=plan.hand_q[0])      # stale in the middle of the playback
    cfg = S.Config()
    res = S.Replayer(plan, io, cfg).run()
    assert res["status"] == "aborted" and "lowstate" in res["reason"]
    k = assert_smooth_release(io, cfg)
    right = io.right()
    assert np.abs(right[k:] - right[k - 1]).max() == 0.0      # no fresh state: hold the last command, never jump
    assert io.arm_log[k][0] < 2.5 + cfg.state_timeout + 2 * DT  # released within a tick of the timeout


def test_stale_hand_state_aborts() -> None:
    plan = make_plan()
    io = StaleAfter(2.5, q0=body_at_hold(plan), hand0=plan.hand_q[0], hand=True)
    cfg = S.Config()
    res = S.Replayer(plan, io, cfg).run()
    assert res["status"] == "aborted" and "dex3" in res["reason"]
    assert_smooth_release(io, cfg)


# --- Ctrl-C / SIGTERM go home and release ----------------------------------------------------------------

def test_ctrl_c_at_the_go_prompt_goes_home_and_releases() -> None:
    plan = make_plan()
    q0 = body_at_hold(plan, right_offset=0.15)                    # the hold pose is 0.15 rad away on every joint
    plan = with_home(plan, q0)
    io = LoggedIO(q0=q0, hand0=plan.hand_q[0])
    cfg = S.Config(hold_timeout_s=1e6)

    def confirm() -> None:
        raise KeyboardInterrupt

    try:
        res = S.Replayer(plan, io, cfg).run(confirm=confirm)
    except KeyboardInterrupt:
        pytest.fail("KeyboardInterrupt escaped run(): the arm was left at weight 1")
    assert res["status"] == "stopped" and "operator" in res["reason"]
    assert_smooth_release(io, cfg)
    right = io.right()
    assert np.abs(right[-1] - q0[RIGHT]).max() < 1e-9                # home = the pose the robot had at the start
    assert np.abs(right - plan.arm_q[0]).max(axis=1).min() < 1e-9    # it was at the hold pose before going home
    steps = np.abs(np.diff(right, axis=0)).max(axis=1)
    assert steps.max() <= cfg.hold_speed * DT * (1 + 1e-6)           # home at the hold speed, no jump


@pytest.mark.parametrize("threaded", [False, True])
def test_sigint_during_playback_goes_home_and_releases(threaded: bool, sigint_guard) -> None:
    """SIGINT delivered to the main thread (`_thread.interrupt_main`, what Ctrl-C does) half-way through the
    playback, in the dry-run loop (`run()`) and with the control thread (`run(confirm)`, the execute path)."""
    plan = make_plan()
    q0 = body_at_hold(plan, right_offset=0.1)
    plan = with_home(plan, q0)
    io = LoggedIO(q0=q0, hand0=plan.hand_q[0], hooks=[(3.0, interrupt(_thread.interrupt_main))])
    cfg = S.Config()
    res = replay(plan, io, cfg, (lambda: None) if threaded else None)
    assert res["status"] == "stopped" and "interrupt" in res["reason"]
    assert "play" not in res["phases"] or "interrupted" in res["phases"]["play"]
    assert_smooth_release(io, cfg)
    assert np.abs(io.right()[-1] - q0[RIGHT]).max() < 1e-9
    assert signal.getsignal(signal.SIGINT) is sigint_guard            # the handler is restored


def test_sigterm_during_playback_goes_home_and_releases() -> None:
    """SIGTERM (e.g. `kill`, a closing terminal) is handled like Ctrl-C. The test's own handler raises `_Guard`, so a
    regression fails the test instead of killing pytest."""
    def guard(signum, frame):
        raise _Guard

    previous = signal.signal(signal.SIGTERM, guard)
    try:
        plan = make_plan()
        q0 = body_at_hold(plan, right_offset=0.1)
        plan = with_home(plan, q0)
        io = LoggedIO(q0=q0, hand0=plan.hand_q[0], hooks=[(3.0, interrupt(lambda: signal.raise_signal(signal.SIGTERM)))])
        cfg = S.Config()
        res = replay(plan, io, cfg, lambda: None)
        assert res["status"] == "stopped"
        assert_smooth_release(io, cfg)
        assert np.abs(io.right()[-1] - q0[RIGHT]).max() < 1e-9
        assert signal.getsignal(signal.SIGTERM) is guard              # the handler is restored
    finally:
        signal.signal(signal.SIGTERM, previous)


def test_second_ctrl_c_releases_in_place() -> None:
    """Escalation: a second Ctrl-C during the go-home skips the rest of it and releases at the last command."""
    plan = make_plan()
    q0 = body_at_hold(plan, right_offset=0.3)
    plan = with_home(plan, q0)
    hooks = [(3.0, interrupt(_thread.interrupt_main)), (3.5, interrupt(_thread.interrupt_main))]
    io = LoggedIO(q0=q0, hand0=plan.hand_q[0], hooks=hooks)
    cfg = S.Config()
    res = replay(plan, io, cfg)
    assert res["status"] == "stopped"
    k = assert_smooth_release(io, cfg)
    right = io.right()
    assert np.abs(right[k:] - right[k - 1]).max() == 0.0
    assert np.abs(right[-1] - q0[RIGHT]).max() > 0.05                # stopped before reaching home


@pytest.mark.parametrize("error", [EOFError, OSError, ValueError])
def test_any_error_at_the_go_prompt_goes_home_and_releases(error: type[Exception]) -> None:
    """stdin closed (EOFError) or any other failure of the prompt is an operator stop, not an exception that leaves
    the control thread holding the hold pose until the hold timeout."""
    plan = make_plan()
    q0 = body_at_hold(plan, right_offset=0.1)
    plan = with_home(plan, q0)
    io = LoggedIO(q0=q0, hand0=plan.hand_q[0])
    cfg = S.Config(hold_timeout_s=1e6)

    def confirm() -> None:
        raise error("prompt failed")

    res = S.Replayer(plan, io, cfg).run(confirm=confirm)
    assert res["status"] == "stopped" and "prompt failed" in res["reason"]
    assert_smooth_release(io, cfg)
    assert np.abs(io.right()[-1] - q0[RIGHT]).max() < 1e-9


# --- arm_sdk is published while waiting for 'go' ----------------------------------------------------------

def test_arm_sdk_is_published_while_waiting_for_go() -> None:
    plan = make_plan()
    io = LoggedIO(q0=body_at_hold(plan), hand0=plan.hand_q[0])
    cfg = S.Config(hold_timeout_s=1e6)
    gained: list[int] = []

    def confirm() -> None:
        n0 = len(io.arm_log)
        deadline = time.monotonic() + 3.0
        while len(io.arm_log) < n0 + 500 and time.monotonic() < deadline:
            time.sleep(0.01)
        gained.append(len(io.arm_log) - n0)

    res = S.Replayer(plan, io, cfg).run(confirm=confirm)
    assert res["status"] == "completed"
    assert gained and gained[0] >= 500                                # >= 2 s of 250 Hz messages while blocked
    t = np.array([e[0] for e in io.arm_log])
    assert np.allclose(np.diff(t), DT)                                # one message per 4 ms tick throughout
    th = np.array([e[0] for e in io.hand_log])
    assert np.diff(th).max() <= 0.012 + 1e-9                          # the hand at ~100 Hz throughout


def test_no_go_within_the_hold_timeout_goes_home() -> None:
    """The operator never answers: after `hold_timeout_s` at the hold pose the robot goes home and releases on its
    own; a 'go' typed after that is ignored."""
    plan = make_plan()
    q0 = body_at_hold(plan, right_offset=0.1)
    plan = with_home(plan, q0)
    io = LoggedIO(q0=q0, hand0=plan.hand_q[0])
    cfg = S.Config(hold_timeout_s=1.0)

    def confirm() -> None:                     # blocks until the control thread has released, then says 'go'
        deadline = time.monotonic() + 5.0
        while io.weight > 0.0 and time.monotonic() < deadline:
            time.sleep(0.01)

    res = S.Replayer(plan, io, cfg).run(confirm=confirm)
    assert res["status"] == "stopped" and "no 'go'" in res["reason"]
    assert_smooth_release(io, cfg)
    assert np.abs(io.right()[-1] - q0[RIGHT]).max() < 1e-9
    assert np.abs(io.right() - plan.arm_q[-1]).max(axis=1).min() > 0.05      # the trajectory was never played


# --- each per-tick check aborts, and the faulty command is never sent ------------------------------------

def _sent_before_release(io: LoggedIO) -> np.ndarray:
    return io.right()[: release_start(io.weights())]


def test_velocity_abort_never_sends_the_jump() -> None:
    plan = make_plan()
    plan.arm_q[150:, 1] += 0.1                                        # 25 rad/s in one tick (validate() would refuse)
    io = LoggedIO(q0=body_at_hold(plan), hand0=plan.hand_q[0])
    cfg = S.Config()
    res = S.Replayer(plan, io, cfg).run()
    assert res["status"] == "aborted" and "velocity" in res["reason"]
    sent = _sent_before_release(io)
    assert (np.abs(np.diff(sent, axis=0)) <= plan.arm_vmax * DT * (1 + 1e-3) + 1e-12).all()
    assert_smooth_release(io, cfg)


def test_limit_abort_never_sends_a_command_past_the_limit() -> None:
    """Joint 0 ramps slowly (0.07 rad over the second) through its upper command limit."""
    plan = make_plan()
    hi = plan.arm_limits[0, 1]
    plan.arm_q[:, 0] = np.linspace(hi - 0.02, hi + 0.05, len(plan.arm_q))
    plan = with_home(plan, body_at_hold(plan))                      # the hold pose moved with arm_q[0]
    io = LoggedIO(q0=body_at_hold(plan), hand0=plan.hand_q[0])
    cfg = S.Config()
    res = S.Replayer(plan, io, cfg).run()
    assert res["status"] == "aborted" and "outside the limits" in res["reason"]
    assert _sent_before_release(io)[:, 0].max() <= hi + 1e-12
    assert_smooth_release(io, cfg)


def test_hand_velocity_and_limit_aborts() -> None:
    cfg = S.Config()
    plan = make_plan()
    plan.hand_q[50:, 1] += 0.5                                        # 50 rad/s on one 10 ms hand row
    io = LoggedIO(q0=body_at_hold(plan), hand0=plan.hand_q[0])
    res = S.Replayer(plan, io, cfg).run()
    assert res["status"] == "aborted" and "hand command step" in res["reason"]
    hq = np.array([e[1] for e in io.hand_log])
    assert np.abs(np.diff(hq[:, 1])).max() < 0.5 * 0.9
    assert_smooth_release(io, cfg)

    plan = make_plan()
    hi = plan.hand_limits[2, 1]
    plan.hand_q[:, 2] = np.linspace(hi - 0.02, hi + 0.05, len(plan.hand_q))
    plan = with_home(plan, body_at_hold(plan))
    io = LoggedIO(q0=body_at_hold(plan), hand0=plan.hand_q[0])
    res = S.Replayer(plan, io, cfg).run()
    assert res["status"] == "aborted" and "hand command outside the limits" in res["reason"]
    assert max(e[1][2] for e in io.hand_log) <= hi + 1e-12
    assert_smooth_release(io, cfg)


def test_left_arm_and_waist_step_cap_aborts() -> None:
    """A hold speed above the 0.5 rad/s cap of the non-trajectory joints (the CLI refuses it; `Config` does not):
    the left arm, 1.57 rad from its park pose, must not be moved at 2 rad/s."""
    plan = make_plan()
    q0 = body_at_hold(plan)
    q0[LEFT] = 0.0
    plan = with_home(plan, q0)
    io = LoggedIO(q0=q0, hand0=plan.hand_q[0])
    cfg = S.Config(hold_speed=2.0)
    res = S.Replayer(plan, io, cfg).run()
    assert res["status"] == "aborted" and "left arm / waist" in res["reason"]
    left = np.array([[e[2][i] for i in LEFT] for e in io.arm_log])
    assert np.abs(np.diff(left, axis=0)).max() <= S.OTHER_VMAX * DT * (1 + 1e-6)
    assert_smooth_release(io, cfg)


def test_mode_machine_change_aborts() -> None:
    plan = make_plan()

    def switch(io: LoggedIO) -> None:
        io.mode_machine = 4

    io = LoggedIO(q0=body_at_hold(plan), hand0=plan.hand_q[0], hooks=[(2.5, switch)])
    cfg = S.Config()
    res = S.Replayer(plan, io, cfg).run()
    assert res["status"] == "aborted" and "mode_machine" in res["reason"]
    assert_smooth_release(io, cfg)


def test_tracking_abort_holds_the_measured_pose() -> None:
    """A right-arm joint stuck 0.5 rad from the hold pose: abort on tracking, then hold the MEASURED pose (stop
    pushing) while the weight ramps down."""
    plan = make_plan()
    q0 = body_at_hold(plan)
    q0[RIGHT[0]] -= 0.5
    plan = with_home(plan, q0)
    io = LoggedIO(q0=q0, hand0=plan.hand_q[0], fault={"stuck_joint": RIGHT[0], "after_s": 0.0})
    cfg = S.Config()
    res = S.Replayer(plan, io, cfg).run()
    assert res["status"] == "aborted" and "tracking error" in res["reason"]
    k = assert_smooth_release(io, cfg)
    assert io.right()[k, 0] == pytest.approx(q0[RIGHT[0]], abs=1e-12)


def test_unexpected_error_while_publishing_still_releases() -> None:
    """A transport exception mid-run (e.g. a DDS write error) is an abort; the release keeps trying every tick."""
    plan = make_plan()

    class Flaky(LoggedIO):
        fails = 0

        def send_arm(self, q, kp, kd, tau, weight, dt) -> None:
            if 2.5 <= self.t < 2.5 + 3 * DT:
                Flaky.fails += 1
                raise OSError("dds write failed")
            super().send_arm(q, kp, kd, tau, weight, dt)

    io = Flaky(q0=body_at_hold(plan), hand0=plan.hand_q[0])
    cfg = S.Config()
    res = S.Replayer(plan, io, cfg).run()
    assert res["status"] == "aborted" and "dds write failed" in res["reason"]
    assert Flaky.fails >= 3                                           # the failing release ticks were retried
    w = io.weights()
    k = release_start(w)
    assert w[-1] == 0.0 and io.weight == 0.0 and (np.diff(w[k - 1:]) <= 1e-12).all()
    assert len(w) - k >= round(cfg.ramp_s / DT) - 3                   # every other release tick was published


# --- --no-waist still commands the waist, at its measured pose ---------------------------------------------

def test_no_waist_holds_the_measured_waist_in_every_message() -> None:
    """(Dry-run only: `--execute` refuses `--no-waist`, `test_sim_gate`.) A measured waist inside the 0.01 rad start
    tolerance is held where it is in every message."""
    plan = make_plan()
    q0 = body_at_hold(plan)
    q0[list(S.WAIST_DDS)] = [0.008, -0.006, 0.004]
    plan = with_home(plan, q0)
    io = LoggedIO(q0=q0, hand0=plan.hand_q[0])
    res = S.Replayer(plan, io, S.Config(hold_waist=False)).run()
    assert res["status"] == "completed"
    for _, _, q, kp in io.arm_log:
        assert [q[i] for i in S.WAIST_DDS] == pytest.approx([0.008, -0.006, 0.004], abs=1e-12)
        assert all(kp[i] == S.WAIST_KP for i in S.WAIST_DDS)


# --- a measured waist off the simulated 0 is refused -------------------

@pytest.mark.parametrize("mode_machine", [5, 6])
@pytest.mark.parametrize(("joint", "offset", "status"), [
    (1, 0.05, "refused"),        # pitch 0.05 rad: the reference case (inside the 0.08 rad home tolerance)
    (0, -0.011, "refused"),      # roll just outside 0.01 rad
    (2, 0.02, "refused"),        # yaw
    (1, 0.0099, "completed"),    # inside
])
def test_a_measured_waist_off_zero_is_refused_before_sending(mode_machine: int, joint: int, offset: float,
                                                             status: str) -> None:
    """`_pick_home` refuses a measured waist joint more than `WAIST_TOL` (0.01 rad) from the simulated 0, with zero
    messages, also when the checked home itself carries that waist (mutation: waist compared with the home, not 0) and
    under mode_machine 6 (which would hold roll / pitch where measured); the reason names the measured waist."""
    plan = make_plan()
    q0 = body_at_hold(plan)
    q0[S.WAIST_DDS[joint]] = offset
    for home in (body_at_hold(plan), q0):
        p = with_home(plan, home)
        io = LoggedIO(q0=q0, hand0=plan.hand_q[0], mode_machine=mode_machine)
        res = replay(p, io, S.Config())
        assert res["status"] == status, (home[list(S.WAIST_DDS)], res)
        if status == "refused":
            assert "measured waist" in res["reason"] and f"{round(offset, 4)}" in res["reason"], res["reason"]
            assert io.sent_arm == 0 and io.sent_hand == 0


# --- the Dex3 hand must start where the run was simulated from ---------------------------------

@pytest.mark.parametrize(("offset", "status"), [(0.15, "refused"), (-0.101, "refused"), (0.09, "completed")])
def test_a_measured_hand_off_the_simulated_start_is_refused_before_sending(offset: float, status: str) -> None:
    """With `Config.start_hand` (what `--execute` passes: the hand the package simulated the run from), a measured
    Dex3 joint more than 0.1 rad from it is refused with zero messages and the measured values printed (mutation:
    the hand never checked); within 0.1 rad the run goes ahead."""
    plan = make_plan()
    start = plan.hand_q[0].copy()
    hand0 = start.copy()
    hand0[2] += offset
    io = LoggedIO(q0=body_at_hold(plan), hand0=hand0)
    res = replay(plan, io, S.Config(start_hand=tuple(start.tolist())))
    assert res["status"] == status, res
    if status == "refused":
        assert "Dex3" in res["reason"] and str(np.round(hand0, 3).tolist()) in res["reason"], res["reason"]
        assert io.sent_arm == 0 and io.sent_hand == 0
    io = LoggedIO(q0=body_at_hold(plan), hand0=hand0)
    assert replay(plan, io, S.Config())["status"] == "completed"              # unchecked without a start hand


def test_sdk_message_enables_every_commanded_motor() -> None:
    """`SdkIO.send_arm` on a stub message: every commanded index (arms AND waist) gets mode 1 with its gains, the
    weight goes in motor 29, mode_machine is echoed and the CRC is set (no SDK needed)."""
    io = S.SdkIO.__new__(S.SdkIO)
    io.low = SimpleNamespace(motor_cmd=[SimpleNamespace(mode=0, q=0.0, dq=0.0, tau=0.0, kp=0.0, kd=0.0)
                                        for _ in range(35)], mode_pr=None, mode_machine=None, crc=None)
    io.crc = SimpleNamespace(Crc=lambda msg: 1234)
    written = []
    io.arm_pub = SimpleNamespace(Write=written.append)
    io._low_state = SimpleNamespace(mode_machine=5)
    idx = list(S.RIGHT_ARM_DDS + S.LEFT_ARM_DDS + S.WAIST_DDS)
    q = {i: 0.1 * k for k, i in enumerate(idx)}
    kp = {i: S.WAIST_KP if i in S.WAIST_DDS else 80.0 for i in idx}
    kd = {i: 3.0 for i in idx}
    io.send_arm(q, kp, kd, {}, 0.7, DT)
    assert len(written) == 1 and io.low.crc == 1234 and io.low.mode_machine == 5
    for i in idx:
        m = io.low.motor_cmd[i]
        assert (m.mode, m.q, m.kp) == (1, pytest.approx(q[i]), kp[i])
    assert io.low.motor_cmd[S.WEIGHT_INDEX].q == pytest.approx(0.7)


# --- mutants R2 (tracking only in 'play'), R6 (no start refusal of a right arm outside
# the limits), R7 (waist dropped from the OTHER_VMAX step cap) survived the suite -----------------------------------

class StuckInPhase(LoggedIO):
    """Right-arm joint `joint` stops following its command from the first message sent in `phase` on."""

    def __init__(self, phase: str, joint: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.stuck_phase, self.stuck = phase, joint

    def send_arm(self, q, kp, kd, tau, weight, dt) -> None:
        if not self.fault and self.replayer.phase == self.stuck_phase:
            self.fault = {"stuck_joint": self.stuck, "after_s": 0.0}
        super().send_arm(q, kp, kd, tau, weight, dt)


def test_stuck_joint_during_to_hold_aborts_in_to_hold() -> None:
    """The hold pose is 0.5 rad away on joint 0, which sticks as soon as TO_HOLD starts: the tracking check must
    abort IN TO_HOLD (mutant R2, tracking checked only in 'play', reaches the hold pose and aborts in play)."""
    plan = make_plan()
    q0 = body_at_hold(plan)
    q0[RIGHT[0]] -= 0.5
    plan = with_home(plan, q0)
    io = StuckInPhase("to_hold", RIGHT[0], q0=q0, hand0=plan.hand_q[0])
    cfg = S.Config()
    res = replay(plan, io, cfg)
    assert res["status"] == "aborted" and "tracking error" in res["reason"]
    assert "to_hold" not in res["phases"] and "wait_go" not in res["phases"] and "play" not in res["phases"]
    assert_smooth_release(io, cfg)


def test_stuck_joint_during_go_home_aborts_in_go_home() -> None:
    """Home is 0.5 rad from the trajectory end on joint 0, which sticks when GO_HOME starts: abort IN GO_HOME
    (mutant R2 lets the way home finish with the joint stuck)."""
    plan = make_plan()
    q0 = body_at_hold(plan, right_offset=0.5)
    plan = with_home(plan, q0)
    io = StuckInPhase("go_home", RIGHT[0], q0=q0, hand0=plan.hand_q[0])
    cfg = S.Config()
    res = replay(plan, io, cfg)
    assert res["status"] == "aborted" and "tracking error" in res["reason"]
    assert res["phases"]["play"] and "go_home" not in res["phases"]
    assert_smooth_release(io, cfg)


def test_measured_right_arm_outside_the_limits_is_refused_before_sending() -> None:
    """Mutant R6: `_start` must refuse a measured right arm outside the command limits; nothing is published."""
    plan = make_plan()
    q0 = body_at_hold(plan)
    q0[RIGHT[0]] = plan.arm_limits[0, 1] + 0.01
    plan = with_home(plan, q0)                                      # at a checked home: only the limit check refuses
    io = LoggedIO(q0=q0, hand0=plan.hand_q[0])
    res = replay(plan, io, S.Config())
    assert res["status"] == "refused" and "outside the command limits" in res["reason"]
    assert io.sent_arm == 0 and io.sent_hand == 0


def test_waist_only_step_above_the_cap_aborts() -> None:
    """Mutant R7: only the waist moves (0.0099 rad of yaw to 0; a larger start offset is refused, see above) at a hold
    speed of 2 rad/s, 4x the OTHER_VMAX cap of the non-trajectory joints: the step check must abort, and no waist step
    above the cap is sent."""
    plan = make_plan()
    q0 = body_at_hold(plan)
    q0[list(S.WAIST_DDS)] = [0.0099, 0.0, 0.0]
    plan = with_home(plan, q0)
    io = LoggedIO(q0=q0, hand0=plan.hand_q[0])
    cfg = S.Config(hold_speed=2.0)
    res = replay(plan, io, cfg)
    assert res["status"] == "aborted" and "left arm / waist" in res["reason"] and "to_hold" not in res["phases"]
    waist = np.array([[e[2][i] for i in S.WAIST_DDS] for e in io.arm_log])
    assert np.abs(np.diff(waist, axis=0)).max() <= S.OTHER_VMAX * DT * (1 + 1e-6)
    assert_smooth_release(io, cfg)


# --- control-loop timing (mutant R4: no resync when late), the --max-track-err cap, the control thread
# start inside the guarded block ------------------------------------------------------------------------------------

def _oversleep(seconds: float) -> Callable[[LoggedIO], None]:
    """Hook: the control loop wakes up `seconds` late once (a slow publish, a GC pause, a loaded PC)."""
    def hook(io: LoggedIO) -> None:
        io.t += seconds
    return hook


def test_a_late_tick_resynchronises_without_a_burst() -> None:
    """Woken 10 ms (2.5 periods) late in the playback: no catch-up burst (consecutive arm messages at least half a
    period apart), no abort (below the 3-period overrun), and the late tick is in the run's tick statistics."""
    plan = make_plan()
    io = LoggedIO(q0=body_at_hold(plan), hand0=plan.hand_q[0], hooks=[(3.0, _oversleep(0.010))])
    res = replay(plan, io, S.Config())
    assert res["status"] == "completed"
    t = np.array([e[0] for e in io.arm_log])
    assert np.diff(t).min() >= DT / 2 - 1e-9
    assert res["ticks"]["late_ticks"] >= 1 and res["ticks"]["max_late_ms"] == pytest.approx(10.0, abs=0.01)
    assert res["ticks"]["max_gap_ms"] == pytest.approx(14.0, abs=0.01)


def test_a_tick_overrun_beyond_three_periods_aborts() -> None:
    plan = make_plan()
    io = LoggedIO(q0=body_at_hold(plan), hand0=plan.hand_q[0], hooks=[(3.0, _oversleep(0.020))])
    cfg = S.Config()
    res = replay(plan, io, cfg)
    assert res["status"] == "aborted" and "overran" in res["reason"] and res["abort_phase"] == "play"
    assert res["ticks"]["max_late_ms"] == pytest.approx(20.0, abs=0.01)
    assert_smooth_release(io, cfg)


def test_max_track_err_is_capped(built) -> None:
    assert S.main(["--package", str(built[3]), "--max-track-err", "0.31"]) == 2
    assert S.main(["--package", str(built[3]), "--max-track-err", "0"]) == 2
    assert S.main(["--package", str(built[3]), "--max-track-err", "0.3"]) == 0


def test_a_control_thread_that_cannot_start_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """`Thread.start` failing (e.g. no thread left) must not escape `run()`: nothing was published, so the run is
    refused."""
    plan = make_plan()
    io = LoggedIO(q0=body_at_hold(plan), hand0=plan.hand_q[0])

    def no_thread(self) -> None:
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(S.threading.Thread, "start", no_thread)
    res = S.Replayer(plan, io, S.Config()).run(confirm=lambda: None)
    assert res["status"] == "refused" and "control thread" in res["reason"]
    assert io.sent_arm == 0 and io.sent_hand == 0


def test_docstring_names_only_existing_flags() -> None:
    import re
    flags = set(re.findall(r"`(--[a-z-]+)`", S.__doc__))
    parser_flags = {"--package", "--execute", "--iface", "--domain", "--hold-speed", "--hold-timeout",
                    "--max-track-err", "--speed-scale", "--no-tau-ff", "--no-waist", "--hand-state-topic"}
    assert flags and flags <= parser_flags, flags - parser_flags


# --- the run starts only at a checked home, follows the checked paths, and retraces on a stop ------

@pytest.mark.parametrize(("offset", "status"), [(0.10, "refused"), (0.07, "completed")])
def test_a_pose_off_every_checked_home_is_refused_before_sending(offset: float, status: str) -> None:
    """The plan's checked home is the hold pose with a 0.08 rad tolerance: the left elbow 0.10 rad off is refused with
    nothing sent (the message lists the 17 measured joints for a re-export); 0.07 rad off runs."""
    plan = make_plan()
    q0 = body_at_hold(plan)
    q0[LEFT[3]] += offset
    io = LoggedIO(q0=q0, hand0=plan.hand_q[0])
    res = replay(plan, io, S.Config())
    assert res["status"] == status
    if status == "refused":
        assert "from the nearest checked home" in res["reason"] and "--home-q" in res["reason"]
        assert io.sent_arm == 0 and io.sent_hand == 0


def test_only_the_allowed_homes_are_used() -> None:
    """`Config.homes` (what --execute passes: the homes that passed the gate) restricts the start: at home #0 with
    only #1 allowed the run is refused; with #0 allowed it runs; with none, refused."""
    plan = make_plan()
    q_a, q_b = body_at_hold(plan), body_at_hold(plan, right_offset=0.3)
    plan = with_home(plan, q_a, q_b)
    for homes, status in (((1,), "refused"), ((0,), "completed"), ((), "refused")):
        io = LoggedIO(q0=q_a, hand0=plan.hand_q[0])
        res = replay(plan, io, S.Config(homes=homes))
        assert res["status"] == status, homes
        assert (io.sent_arm == 0) == (status == "refused")
    assert "none passes" in replay(plan, LoggedIO(q0=q_a, hand0=plan.hand_q[0]), S.Config(homes=()))["reason"]


def test_to_hold_follows_the_checked_waypoints() -> None:
    """A checked path with two waypoints between the home and the hold pose: TO_HOLD passes within one tick's step of
    each, in order, and no joint steps faster than the hold speed (a straight line missed them)."""
    plan = make_plan()
    q0 = body_at_hold(plan, right_offset=0.2)
    home17, hold17 = q0[list(S.RIGHT_ARM_DDS + S.LEFT_ARM_DDS + S.WAIST_DDS)], body_at_hold(plan)[
        list(S.RIGHT_ARM_DDS + S.LEFT_ARM_DDS + S.WAIST_DDS)]
    via = [home17 + np.r_[np.zeros(3), 0.4, np.zeros(13)], hold17 + np.r_[np.zeros(3), 0.4, np.zeros(13)]]
    plan = with_home(plan, q0, via=via)
    io = LoggedIO(q0=q0, hand0=plan.hand_q[0])
    cfg = S.Config()
    rp = S.Replayer(plan, io, cfg)
    io.replayer = rp
    phases: list[str] = []
    orig = io.send_arm

    def send(q, kp, kd, tau, weight, dt) -> None:
        phases.append(rp.phase)
        orig(q, kp, kd, tau, weight, dt)

    io.send_arm = send
    assert rp.run()["status"] == "completed"
    right = io.right()[np.asarray(phases) == "to_hold"]
    step = cfg.hold_speed * DT
    assert np.abs(np.diff(right, axis=0)).max() <= step * (1 + 1e-6)
    hits = [int(np.argmin(np.abs(right - v[:7]).max(axis=1))) for v in via]
    assert all(np.abs(right[h] - v[:7]).max() <= step + 1e-9 for h, v in zip(hits, via, strict=True))
    assert hits[0] < hits[1]


def test_an_operator_stop_retraces_the_commands_sent() -> None:
    """Ctrl-C half-way through the playback: the way home runs back along the commands already sent (the playback
    and TO_HOLD, reversed; each command within one sent step of them, hand included) and ends at HOME, never along a
    new straight line; no joint faster than the hold speed."""
    plan = make_plan()
    q0 = body_at_hold(plan, right_offset=0.1)
    plan = with_home(plan, q0)
    io = LoggedIO(q0=q0, hand0=plan.hand_q[0], hooks=[(3.0, interrupt(_thread.interrupt_main))])
    cfg = S.Config()
    rp = S.Replayer(plan, io, cfg)
    io.replayer = rp
    phases: list[str] = []
    orig = io.send_arm

    def send(q, kp, kd, tau, weight, dt) -> None:
        phases.append(rp.phase)
        orig(q, kp, kd, tau, weight, dt)

    io.send_arm = send
    res = rp.run()
    assert res["status"] == "stopped" and "interrupted" in res["phases"]["play"] and rp.retraced
    ph = np.asarray(phases)
    right = io.right()
    sent = right[(ph == "to_hold") | (ph == "play")]
    back = right[ph == "go_home"]
    gap = np.abs(np.diff(sent, axis=0)).max()
    dist = np.abs(back[:, None, :] - sent[None, :, :]).max(axis=2).min(axis=1)
    assert dist.max() <= gap + 1e-9
    assert np.abs(back[-1] - q0[RIGHT]).max() < 1e-12
    assert np.abs(np.diff(back, axis=0)).max() <= cfg.hold_speed * DT * (1 + 1e-6)
