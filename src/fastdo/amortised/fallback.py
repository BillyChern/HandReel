"""(d) capped MPC fallback, warm-started from the best candidate.

Trigger: a clip with no certified success after (b) training and (c) pass@S. Warm start: u^(0) = ctrl_ref + U_best,
U_best = the knot residuals of the candidate with the lowest normalised SR_MT margin (`fastdo.amortised.select`'s
ONE score), written as a warm-start file (`write_warm_start`: knot rows + residuals) and injected at the FAST
executor's initial plan by `_baseline_physics_driver.py --warm-start` (`fastdo.physics.clip.setup`: the plan the first
rollout optimises, and every control row the receding horizon appends, is ctrl_ref + the interpolated residual;
`fastdo.amortised.knots.residual_rows`). A zero residual reproduces the unflagged run exactly.

Caps (code): iterations per control step <= `ITERATION_CAP` (8; `--iteration-cap`); GPU-s per clip <= c_cap =
`CAP_FRACTION` (0.25) x the fresh FAST median, enforced by `--hold-cap-s` (the run stops at the first control-step
boundary past the cap, so a hold overshoots by at most one control step; the unit driver's per-hold timeout,
`fastdo.amortised.unit.drive`, is the hard backstop); corpus budget B_fb = `BUDGET_FRACTION` (0.05) x N x
c_cap, spent in order of the margin (`FallbackBudget`, a JSON ledger written atomically after every hold, so a resumed
corpus run never re-spends). Each hold is charged its ACTUAL lock-held wall (`FallbackBudget.record_hold` with the
`fastdo.amortised.speed_ledger.Hold` of the unit; a hold killed by the driver's timeout is charged to the driver's reap
time, `unit.drive`), never the cap: `--hold-cap-s` bounds the control loop only, so setup and the last control step can
take a hold past c_cap. A hold is admitted only if the spent GPU-s plus a RESERVE fits in B_fb, the reserve being the
larger of one full cap and the largest hold charged so far (so an observed overshoot is budgeted for); any remaining
overshoot of B_fb is reported (`summary()["overshoot_s"]`).

Whole-hold bound: the WHOLE hold is bounded and charged. `--hold-cap-s` keeps bounding the
control loop; the unit driver's per-hold timeout is c_cap + p95(setup) (`HoldBound.timeout_s`), with setup measured
on the FIRST `SETUP_SAMPLES` holds and then frozen (the bound does not move with later holds). A setup sample is
measured from the LAUNCH (where the driver's timeout starts): for an MPC hold, `record(row, loop_s=...)` with the
run's own `fast_timings["loop_s"]` (`_baseline_physics_driver.py`'s arm_meta) takes launch-to-release minus the
control loop, i.e. everything the hold does outside the capped loop (imports, scene build, graph capture, the epilogue
that saves the run), so the bound covers the whole hold; for work without a loop timing, the work's
`ctx.setup_done()` stamp (`unit.drive`'s row `setup_s` = t_setup - t_launch). With a blocking
gpu.lock, "the launch" is the row's EFFECTIVE launch `t_launch_eff` (`launch_of`: the launch shifted by the unit's
blocking lock wait, which is where the driver's timeout now starts), so a wait for E1's turn is never setup. A hold
KILLED by its timeout (`rc == "timeout"`) gives NO sample, stamp or not: its stamp covers only the part of setup
that ran before the stamp, so a frozen sample from it would shrink the bound. With `loop_only=True` (the MPC holds:
the stamp is taken before the driver's scene build and graph capture, so it does not mark the end of setup) only a
FINISHED hold (rc 0) with a loop time gives a sample; the stamp is never used. Until those samples exist
the driver's backstop (`unit.UNIT_MAX_S` + 120 s) applies, and those holds are the setup measurement. p95 is
numpy's linear-interpolation percentile of the frozen samples. Every hold's overshoot is recorded: its lock-held wall
beyond c_cap and beyond the timeout it ran under (a killed hold is charged to the driver's reap time, as `unit.drive`
does). The loop itself overshoots c_cap by up to one control step (`--hold-cap-s`), which the formula does not cover:
such a hold can be killed in its epilogue; it is charged and recorded, and a capped run cannot reach the clip end.
Two time bases, kept apart: `over_cap_s` is the lock-held wall (the CHARGED quantity) beyond c_cap; `over_timeout_s`
is launch-to-release beyond the timeout, because the driver's timeout counts from the launch (a hold killed exactly at
its timeout reads 0 there, whatever its lock wait).

UNCAPPED holds (the H2/H2+ runs: plain FAST runs with no `--hold-cap-s`) are not fallback holds and
have no c_cap. They use a RECORD-ONLY bound, `HoldBound(path, cap_s=None)`: `timeout_s()` is ALWAYS the driver's
backstop (never c_cap + p95), `record` still takes their setup samples (the first `n_setup`, then frozen), and no
over-cap is computed. c_cap needs the fresh FAST median, which is measured, not assumed: the first capped bound is made
by `capped(path, cap_s)`, which carries the record-only bound's frozen setup samples over unchanged (setup is the same
scene build, graph capture and epilogue whether or not the loop is capped).

c_cap's SOURCE: "the fresh FAST median" is the median of H2's OWN FAST walls (the block of uncapped FAST holds:
lock-held, same hardware, measured), and c_cap = `CAP_FRACTION` x that median is FROZEN from H2 (a committed-before-use
record carrying its sha256) before any capped hold runs. A c_cap travels as a `CCap` with its status: "FROZEN" (from
that record) or "ESTIMATE" (before H2 exists: `estimate_c_cap`, the stored-estimate figure, 0.25 x 894 GPU-s stock
Ego4D median / FAST's 1.67x = 133.8 s, range 117.6-159.6 s at 1.9x-1.4x; also written c*). A capped hold never runs on
an ESTIMATE: `HoldBound.capped` refuses anything but a FROZEN `CCap` (a bare number included), and the capped bound
carries the `CCap` it was made from (`HoldBound.ccap`) so the hold launcher can check it. Killed H2 holds are
RIGHT-CENSORED at their charged wall (a FAST run that hit the backstop took at least that long):
`fast_median_from_walls` takes the median with them at their charged walls and accepts it only if it is IDENTIFIED, i.e.
unchanged when every censored wall is replaced by +infinity (fewer than half censored, all above the median); otherwise
it refuses.

Acceptance (`accept`): only CERTIFIED output ships, and a fallback never replaces a clip's certified candidate. The
worst single clip's fallback GPU-s is reported (`FallbackBudget.summary`).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fastdo.amortised.knots import residual_rows

__all__ = ["BUDGET_FRACTION", "CAP_FRACTION", "CCap", "ITERATION_CAP", "SETUP_SAMPLES", "FallbackBudget",
           "FallbackCaps", "HoldBound", "accept", "c_cap", "corpus_budget", "driver_args", "estimate_c_cap",
           "fast_median_from_walls", "launch_of", "load_warm_start", "p95", "warm_start_rows", "write_warm_start"]

ITERATION_CAP = 8
CAP_FRACTION = 0.25
BUDGET_FRACTION = 0.05
SETUP_SAMPLES = 5                     # setup measured on the first holds (implementation choice: the first 5)
BACKSTOP_S = 900.0 + 120.0            # `unit.UNIT_MAX_S` + 120 s: `unit.drive`'s default per-hold timeout


STOCK_EGO4D_MEDIAN_S = 894.0           # DO AS I DO's stock executor, Ego4D median
FAST_RATIO = (1.67, 1.4, 1.9)           # FAST over stock: conservative ratio of sums and its range


def c_cap(fast_median_gpu_s: float) -> float:
    return CAP_FRACTION * float(fast_median_gpu_s)


@dataclass(frozen=True)
class CCap:
    """A c_cap with its provenance (module docstring): status "FROZEN" (from H2's committed record) or
    "ESTIMATE" (the stored-estimate figure, before H2 exists)."""

    c_cap_s: float
    status: str
    source: str
    fast_median_s: float | None = None
    record_sha256: str | None = None
    range_s: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if self.status not in ("FROZEN", "ESTIMATE"):
            raise ValueError(f"c_cap status {self.status!r}")
        if not self.c_cap_s >= 0.0:
            raise ValueError("c_cap must be >= 0 s")


def estimate_c_cap() -> CCap:
    """The stored-estimate c_cap, labelled ESTIMATE (module docstring); never runs a capped hold."""
    med, lo, hi = (STOCK_EGO4D_MEDIAN_S / r for r in FAST_RATIO)
    return CCap(c_cap(med), "ESTIMATE", "0.25 x 894 GPU-s stock Ego4D median / FAST "
                "1.67x (1.4-1.9x)", fast_median_s=med, range_s=(c_cap(hi), c_cap(lo)))


def fast_median_from_walls(walls: list[tuple[float, bool]]) -> float:
    """The median of H2's lock-held FAST walls, `walls` = [(wall_s, censored)] (module docstring): a censored wall
    (killed hold) is a lower bound; the median must not depend on the censored values, else ValueError."""
    if not walls:
        raise ValueError("no H2 walls: c_cap cannot be frozen before H2 ran")
    x = np.asarray([float(w) for w, _ in walls], np.float64)
    cen = np.asarray([bool(c) for _, c in walls])
    med = float(np.median(x))
    if cen.any() and float(np.median(np.where(cen, np.inf, x))) != med:
        raise ValueError(f"the median of the H2 walls is not identified: {int(cen.sum())} of {len(x)} holds were "
                         "killed (right-censored) and they move it")
    return med


def corpus_budget(n_clips: int, cap_s: float) -> float:
    return BUDGET_FRACTION * int(n_clips) * float(cap_s)


@dataclass(frozen=True)
class FallbackCaps:
    hold_cap_s: float
    iteration_cap: int = ITERATION_CAP

    def __post_init__(self) -> None:
        if not 1 <= self.iteration_cap <= ITERATION_CAP:
            raise ValueError(f"iteration cap {self.iteration_cap} outside [1, {ITERATION_CAP}]")
        if not self.hold_cap_s >= 0.0:
            raise ValueError("hold cap must be >= 0 s")


def write_warm_start(path: Path, knots: np.ndarray, U: np.ndarray, meta: dict[str, Any] | None = None) -> Path:
    """The warm-start file: knot rows (K,) int64, residuals (K, R) float64, provenance JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, knots=np.asarray(knots, np.int64), U=np.asarray(U, np.float64),
             meta=np.frombuffer(json.dumps(meta or {}).encode(), np.uint8))
    tmp.replace(path)
    return path


def load_warm_start(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(Path(path)) as z:
        return np.asarray(z["knots"], np.int64), np.asarray(z["U"], np.float64)


def warm_start_rows(ctrl_ref: np.ndarray, knots: np.ndarray, U: np.ndarray) -> np.ndarray:
    """ctrl_ref (n, nu) + the interpolated residual on its first U.shape[1] columns (float64)."""
    out = np.asarray(ctrl_ref, np.float64).copy()
    out[:, :U.shape[1]] += residual_rows(knots, U, out.shape[0])
    return out


def driver_args(caps: FallbackCaps, warm_start: Path) -> list[str]:
    """The `_baseline_physics_driver.py` flags of one fallback hold (the caps are always passed)."""
    return ["--executor", "fast", "--warm-start", str(warm_start), "--iteration-cap", str(caps.iteration_cap),
            "--hold-cap-s", repr(float(caps.hold_cap_s))]


def launch_of(row: dict[str, Any]) -> float:
    """A drive row's launch for every "from the launch" quantity: `t_launch_eff` (the launch shifted by the unit's
    blocking wait for gpu.lock; `unit` module docstring) when the row has it, else `t_launch`."""
    t = row.get("t_launch_eff")
    return float(row["t_launch"] if t is None else t)


def p95(xs: list[float]) -> float:
    """The 95th percentile (numpy's linear interpolation) of the setup samples."""
    return float(np.percentile(np.asarray(xs, np.float64), 95))


class HoldBound:
    """R3's whole-hold bound (module docstring). `path` holds {cap_s, n_setup, setup_s: [...], holds: [...]}; pass
    `timeout_s` as `unit.drive(hold_timeout_s=...)` and `record` as its `on_hold`. `cap_s=None` is the RECORD-ONLY bound
    of uncapped holds (module docstring): its timeout is always the backstop."""

    def __init__(self, path: Path, cap_s: float | None, n_setup: int = SETUP_SAMPLES,
                 backstop_s: float = BACKSTOP_S, setup_s: list[float] | None = None,
                 setup_from: str | None = None) -> None:
        self.path, self.n_setup, self.backstop_s = Path(path), int(n_setup), float(backstop_s)
        self.cap_s = None if cap_s is None else float(cap_s)
        self.setup_s: list[float] = [float(x) for x in (setup_s or [])][:self.n_setup]
        self.setup_from = setup_from
        self.ccap: CCap | None = None          # the FROZEN c_cap a capped bound was made from (`capped`)
        self.holds: list[dict[str, Any]] = []
        if self.path.is_file():
            st = json.loads(self.path.read_text())
            if (st["cap_s"], st["n_setup"]) != (self.cap_s, self.n_setup):
                raise ValueError("hold-bound ledger differs from this run's cap or sample count")
            if setup_s is not None and [float(x) for x in st["setup_s"]][:len(self.setup_s)] != self.setup_s:
                raise ValueError("hold-bound ledger's setup samples differ from the ones carried over")
            self.setup_s, self.holds = list(st["setup_s"]), list(st["holds"])
            self.setup_from = st.get("setup_from", self.setup_from)

    @property
    def record_only(self) -> bool:
        return self.cap_s is None

    @property
    def measured(self) -> bool:
        return len(self.setup_s) >= self.n_setup

    def timeout_s(self) -> float:
        """c_cap + p95(setup) once the first `n_setup` holds measured setup; the driver's backstop before, and ALWAYS
        for a record-only bound (uncapped holds)."""
        if self.record_only or not self.measured:
            return self.backstop_s
        return self.cap_s + p95(self.setup_s[:self.n_setup])

    def capped(self, path: Path, cap: CCap) -> HoldBound:
        """The capped bound for fallback holds, with THIS bound's frozen setup samples carried over (module
        docstring). Refused until the samples exist, so a capped hold never runs under a bound with no setup term,
        and refused for anything but a FROZEN `CCap` (never an ESTIMATE, never a bare number)."""
        if not isinstance(cap, CCap):
            raise ValueError(f"a capped bound needs a FROZEN CCap, not {type(cap).__name__}")
        if cap.status != "FROZEN":
            raise ValueError(f"c_cap is an {cap.status} ({cap.c_cap_s:.1f} s): a capped hold never runs on it; "
                             "freeze c_cap from H2's walls first")
        if not self.measured:
            raise ValueError(f"only {len(self.setup_s)} of {self.n_setup} setup samples: setup is not measured yet")
        b = HoldBound(path, cap.c_cap_s, n_setup=self.n_setup, backstop_s=self.backstop_s,
                      setup_s=self.setup_s[:self.n_setup], setup_from=str(self.path))
        b.ccap = cap
        return b

    @staticmethod
    def setup_sample(row: dict[str, Any], loop_s: float | None = None, loop_only: bool = False) -> float | None:
        """A hold's setup from the launch (module docstring): launch-to-release minus `loop_s` when the run's loop
        time is given, else the row's `setup_s` stamp (None if neither exists). A killed hold gives None; with
        `loop_only`, only a finished hold (rc 0) with a loop time gives a sample (never the stamp)."""
        if row.get("rc") == "timeout":
            return None
        if loop_only and (row.get("rc") != 0 or loop_s is None):
            return None
        if loop_s is not None:
            return float(row["t_release"]) - launch_of(row) - float(loop_s)
        return None if row.get("setup_s") is None else float(row["setup_s"])

    def record(self, row: dict[str, Any], loop_s: float | None = None, loop_only: bool = False) -> None:
        """One `unit.drive` history row: a setup sample (`setup_sample`) while fewer than `n_setup` exist (then
        frozen), and the hold's overshoot beyond c_cap (lock-held wall, the charged quantity; None when record-only)
        and beyond the timeout it ran under (from the LAUNCH, where the timeout counts from). Launches that never took
        the lock are skipped; a killed hold gives no setup sample, whatever it stamped (`setup_sample`)."""
        if row.get("t_lock") is None or row.get("t_release") is None:
            return
        wall = float(row["t_release"]) - float(row["t_lock"])
        since_launch = float(row["t_release"]) - launch_of(row)
        sample = self.setup_sample(row, loop_s, loop_only)
        if sample is not None and not self.measured:
            self.setup_s.append(sample)
        to = row.get("timeout_s")
        self.holds.append({"name": row.get("name"), "rc": row.get("rc"), "wall_s": wall,
                           "launch_to_release_s": since_launch, "setup_s": sample,
                           "timeout_s": to, "release_source": row.get("release_source"),
                           "killed": row.get("rc") == "timeout",
                           "over_cap_s": None if self.record_only else max(0.0, wall - self.cap_s),
                           "over_timeout_s": None if to is None else max(0.0, since_launch - float(to))})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps({"cap_s": self.cap_s, "n_setup": self.n_setup, "setup_s": self.setup_s,
                                   "setup_from": self.setup_from, "holds": self.holds}, indent=1, default=str))
        tmp.replace(self.path)

    def summary(self) -> dict[str, Any]:
        over = [h["over_cap_s"] for h in self.holds if h["over_cap_s"] is not None]
        return {"cap_s": self.cap_s, "record_only": self.record_only, "setup_from": self.setup_from,
                "setup_samples": self.setup_s[:self.n_setup], "measured": self.measured,
                "p95_setup_s": p95(self.setup_s[:self.n_setup]) if self.measured else None,
                "timeout_s": self.timeout_s(), "holds": len(self.holds),
                "killed": sum(h["killed"] for h in self.holds), "over_cap_holds": sum(x > 0 for x in over),
                "overshoot_max_s": max(over, default=0.0), "overshoot_total_s": float(sum(over)),
                "over_timeout_max_s": max((h["over_timeout_s"] for h in self.holds
                                           if h["over_timeout_s"] is not None), default=0.0)}


def accept(certified_candidate: Any | None, fallback_certified: bool) -> str:
    """What ships for a clip (module docstring): 'kept_certified', 'fallback' or 'none'."""
    if certified_candidate is not None:
        return "kept_certified"
    if fallback_certified:
        return "fallback"
    return "none"


class FallbackBudget:
    """The corpus fallback ledger (module docstring). `path` holds {cap_s, budget_s, spent_s, holds: [...]}."""

    def __init__(self, path: Path, n_clips: int, cap_s: float) -> None:
        self.path = Path(path)
        self.cap_s = float(cap_s)
        self.budget_s = corpus_budget(n_clips, cap_s)
        self.holds: list[dict[str, Any]] = []
        if self.path.is_file():
            st = json.loads(self.path.read_text())
            if (st["cap_s"], st["budget_s"]) != (self.cap_s, self.budget_s):
                raise ValueError("ledger caps differ from this run's")
            self.holds = st["holds"]

    @property
    def spent_s(self) -> float:
        return float(sum(h["gpu_s"] for h in self.holds))

    def done(self) -> set[str]:
        return {h["task"] for h in self.holds}

    def order(self, candidates: list[tuple[str, float]]) -> list[str]:
        """Clips still to try, by margin (lowest first; ties by name)."""
        done = self.done()
        return [t for t, _ in sorted(candidates, key=lambda x: (x[1], x[0])) if t not in done]

    def reserve_s(self) -> float:
        """What one more hold may cost: max(one full cap, the largest hold charged so far) (module docstring)."""
        return max([self.cap_s] + [float(h["gpu_s"]) for h in self.holds])

    def admit(self) -> bool:
        """One more hold fits only if spent + the reserve stays within B_fb."""
        return self.spent_s + self.reserve_s() <= self.budget_s + 1e-9

    def record_hold(self, task: str, hold: Any, capped: bool, certified: bool) -> None:
        """Charge a hold's ACTUAL lock-held wall (`speed_ledger.Hold.hold_wall_s`), whatever its cap."""
        self.record(task, float(hold.hold_wall_s), capped, certified, hold=str(hold.name),
                    release_source=str(getattr(hold, "release_source", "unit")))

    def record(self, task: str, gpu_s: float, capped: bool, certified: bool, **extra: Any) -> None:
        self.holds.append({"task": task, "gpu_s": float(gpu_s), "capped": bool(capped), "certified": bool(certified),
                           **extra})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps({"cap_s": self.cap_s, "budget_s": self.budget_s, "holds": self.holds}, indent=1))
        tmp.replace(self.path)

    def summary(self) -> dict[str, Any]:
        worst = max(self.holds, key=lambda h: h["gpu_s"]) if self.holds else None
        return {"cap_s": self.cap_s, "budget_s": self.budget_s, "spent_s": self.spent_s, "holds": len(self.holds),
                "overshoot_s": max(0.0, self.spent_s - self.budget_s),
                "over_cap_holds": sum(h["gpu_s"] > self.cap_s for h in self.holds),
                "certified": sum(h["certified"] for h in self.holds), "capped": sum(h["capped"] for h in self.holds),
                "worst_clip": worst}
