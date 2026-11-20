# Copyright 2026 The HandReel authors. SPDX-License-Identifier: Apache-2.0
"""The null audit: run a success test on null trajectories and report which nulls pass.

A success test is any callable that scores an EXECUTED object trajectory ((n, 7) poses, see `poses`) and returns a
bool, an object with a boolean `passed` attribute, or a mapping with a "passed" key. The audit never inspects the
test; it only feeds it trajectories that a hand-borne grasp did not have to produce.

Per clip the audit records each null's outcome and the clip's Z0 flag: Z0 = the static-at-start null passes the
test. Across clips the headline is each null's pass rate, and that rate is ALWAYS reported beside the Z0 base rate
(how many of the audited clips are Z0) and split into its Z0 and Z1 strata. A null-pass rate without the base rate
cannot be read: on a corpus where most clips start and end on a support, a pose-only test passes the static null on
most clips no matter which method is scored.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

import numpy as np

from . import nulls as N
from .poses import as_pose7

__all__ = ["KINEMATIC_NULLS", "SIMULATED_NULLS", "STANDARD_NULLS", "AuditReport", "ClipAudit", "NullOutcome",
           "NullRate", "Rollout", "Simulator", "audit_clip", "summarize", "verdict", "wilson95"]

KINEMATIC_NULLS = ("static_at_start", "time_shifted", "reversed")
SIMULATED_NULLS = ("hand_absent", "plan_then_freeze")
STANDARD_NULLS = KINEMATIC_NULLS + SIMULATED_NULLS
PASS, FAIL, NOT_RUN, INVALID = "pass", "fail", "not_run", "invalid"


@dataclass(frozen=True)
class Rollout:
    """One simulator execution.

    object_traj: (n, 7) or (n, 4, 4) executed object poses, one per executed row (n may be < T if the run stopped).
    hand_object_force: (n,) total hand-object normal force per row in N, or None if the simulator cannot report it.
        The hand-absent null is only VERIFIED when this is given, has one entry per trajectory row, and is 0 on
        every row >= release_row. A ledger of another length makes the run INVALID.
    status: free text from the simulator (e.g. "clip_end", "drop"). "invalid" (non-finite state, solver warning)
        makes the null INVALID before any scoring; an early stop such as "drop" is scored as executed, unless
        the run stopped before the release row: then no post-release row ran and both simulated nulls are INVALID.
    """

    object_traj: np.ndarray
    hand_object_force: np.ndarray | None = None
    status: str = "complete"


class Simulator(Protocol):
    """Executes controls (T, nu) from the clip's start state. With `hand_object_contact=False` the simulator must
    remove every hand-object contact force (the hand-absent null); it should still report the force it measured."""

    def __call__(self, controls: np.ndarray, *, hand_object_contact: bool = True) -> Rollout: ...


SuccessTest = Callable[[np.ndarray], Any]


def verdict(result: Any) -> bool:
    """A success test's return value as a bool (bool, `.passed`, or mapping["passed"]). A wrapped verdict must
    itself be a bool: the string "false" and the number 0.5 are both truthy, and neither is a verdict."""
    if isinstance(result, (bool, np.bool_)):
        return bool(result)
    if isinstance(result, Mapping) or hasattr(result, "passed"):
        v = result["passed"] if isinstance(result, Mapping) else result.passed
        if not isinstance(v, (bool, np.bool_)):
            raise TypeError(f"success test's `passed` is {type(v).__name__}; expected a bool")
        return bool(v)
    raise TypeError(f"success test returned {type(result).__name__}; expected bool, .passed or mapping['passed']")


def _detail(result: Any) -> dict[str, Any]:
    if isinstance(result, Mapping):
        return {k: v for k, v in result.items() if isinstance(v, (bool, int, float, str))}
    if hasattr(result, "__dataclass_fields__"):
        return {k: v for k, v in asdict(result).items() if isinstance(v, (bool, int, float, str))}
    return {}


@dataclass(frozen=True)
class NullOutcome:
    """One null on one clip. status: pass | fail | not_run (inputs missing) | invalid (the null was not a null)."""

    null: str
    status: str
    note: str = ""
    verified_n0: bool | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == PASS


@dataclass(frozen=True)
class ClipAudit:
    """All nulls on one clip. `z0` = the static-at-start null passes; `reference_passes` = the positive control (the
    reference itself, scored as if executed) passes. When it fails, check the row count, the pose validity and the
    evidence the test requires before reading the nulls: four valid rows correctly fail a five-row requirement."""

    clip_id: str
    z0: bool
    reference_passes: bool
    outcomes: dict[str, NullOutcome]

    def passing(self) -> list[str]:
        return [k for k, o in self.outcomes.items() if o.passed]


def _score(name: str, test: SuccessTest, traj: np.ndarray, note: str = "", verified: bool | None = None) -> NullOutcome:
    res = test(as_pose7(traj))
    return NullOutcome(name, PASS if verdict(res) else FAIL, note, verified, _detail(res))


def _invalid_rollout(name: str, ro: Rollout) -> NullOutcome | None:
    """INVALID before any scoring when the simulator itself reports an invalid run (non-finite state, solver
    warning): such rows are not a trajectory of the null, so they are never scored."""
    if ro.status == "invalid":
        return NullOutcome(name, INVALID, "the simulator reported an invalid run", None, {"sim_status": ro.status})
    return None


def _hand_absent(test: SuccessTest, sim: Simulator, controls: np.ndarray, release_row: int | None) -> NullOutcome:
    ro = sim(controls, hand_object_contact=False)
    bad = _invalid_rollout("hand_absent", ro)
    if bad is not None:
        return NullOutcome(bad.null, bad.status, bad.note, False, bad.detail)
    # The run must REACH the release whether or not it reports a force: a rollout that stops before the release
    # row has no post-release rows, and scoring its prefix would be a pass of something that never happened.
    r_rel = release_row or 0
    if not 0 <= r_rel < len(ro.object_traj):
        return NullOutcome("hand_absent", INVALID, f"release row {r_rel} outside the run's rows "
                           f"[0, {len(ro.object_traj)}): N0 cannot be checked", False, {"sim_status": ro.status})
    if ro.hand_object_force is None:
        out = _score("hand_absent", test, ro.object_traj, "N0 unverified: the simulator reported no hand-object force")
        return NullOutcome(out.null, out.status, out.note, False, {**out.detail, "sim_status": ro.status})
    # The ledger is one total normal force in N per row. Its shape and dtype are checked on the array as reported,
    # before any conversion: a scalar has no rows, and a complex, Boolean or non-numeric array is not a force.
    raw = np.asarray(ro.hand_object_force)
    not_a_ledger = NullOutcome("hand_absent", INVALID, "force ledger is not one finite, non-negative real value per "
                               "row: N0 cannot be checked", False, {"sim_status": ro.status})
    if raw.ndim != 1 or raw.dtype.kind not in "fiu":
        return not_a_ledger
    if len(raw) != len(ro.object_traj):
        return NullOutcome("hand_absent", INVALID,
                           f"force ledger has {len(raw)} rows, trajectory {len(ro.object_traj)}",
                           False, {"sim_status": ro.status})
    # finite, sign and exact-zero checks read the array as reported: a wider float type is not narrowed first, so a
    # tiny nonzero force cannot round to zero
    if not bool(np.all(np.isfinite(raw))) or bool(np.any(raw < 0)):
        return not_a_ledger
    f = raw
    r0 = release_row or 0
    if not 0 <= r0 < len(f):
        return NullOutcome("hand_absent", INVALID, f"release row {r0} outside the run's rows [0, {len(f)}): N0 cannot "
                           "be checked", False, {"sim_status": ro.status})
    tail = f[r0:]
    leaks = int(np.count_nonzero(tail != 0.0))       # exactly zero on every post-release row, or the null leaks
    if leaks:
        return NullOutcome("hand_absent", INVALID, f"N0 violated: hand-object force on {leaks} post-release rows",
                           False, {"sim_status": ro.status, "leak_rows": leaks})
    out = _score("hand_absent", test, ro.object_traj, verified=True)
    return NullOutcome(out.null, out.status, out.note, True, {**out.detail, "sim_status": ro.status})


def audit_clip(reference: np.ndarray, success_test: SuccessTest, *, clip_id: str = "",
               simulator: Simulator | None = None, controls: np.ndarray | None = None,
               release_row: int | None = None, absent_controls: np.ndarray | None = None,
               shift_rows: int | None = None, nulls: Sequence[str] = STANDARD_NULLS) -> ClipAudit:
    """Run `success_test` on each requested null of one clip.

    Args:
        reference: (T, 7) or (T, 4, 4) reference object trajectory on the rows the test scores.
        success_test: callable scoring an executed (n, 7) object trajectory (module docstring).
        clip_id: label carried into the report.
        simulator: optional; needed for hand_absent and plan_then_freeze (otherwise they are `not_run`).
        controls: (T, nu) stored controls of the clip; needed for both simulated nulls.
        release_row: W, the first post-release row; needed for plan_then_freeze, and bounds the N0 force check.
        absent_controls: controls for the hand-absent run (e.g. `nulls.move_hand_away(controls, ...)`); default
            `controls`.
        shift_rows: delay of the time-shifted null; default a quarter of the post-release rows (at least 1).
        nulls: which nulls to run; static_at_start always runs because it defines Z0.
    """
    ref = as_pose7(reference)
    if release_row is not None and not 0 <= release_row < ref.shape[0]:
        raise ValueError(f"release_row {release_row} outside [0, {ref.shape[0]})")
    unknown = set(nulls) - set(STANDARD_NULLS)
    if unknown:
        raise ValueError(f"unknown nulls {sorted(unknown)}; choose from {STANDARD_NULLS}")
    out: dict[str, NullOutcome] = {"static_at_start": _score("static_at_start", success_test, N.static_at_start(ref))}
    if "time_shifted" in nulls:
        k = shift_rows if shift_rows is not None else max(1, (ref.shape[0] - (release_row or 0)) // 4)
        out["time_shifted"] = _score("time_shifted", success_test, N.time_shifted(ref, k), f"shift {k} rows")
    if "reversed" in nulls:
        out["reversed"] = _score("reversed", success_test, N.reversed_reference(ref))
    if "hand_absent" in nulls:
        u = absent_controls if absent_controls is not None else controls
        if simulator is None or u is None:
            out["hand_absent"] = NullOutcome("hand_absent", NOT_RUN, "needs a simulator and controls")
        else:
            out["hand_absent"] = _hand_absent(success_test, simulator, u, release_row)
    if "plan_then_freeze" in nulls:
        if simulator is None or controls is None or release_row is None:
            out["plan_then_freeze"] = NullOutcome("plan_then_freeze", NOT_RUN,
                                                  "needs a simulator, controls and release_row")
        elif release_row == 0:
            # the release is the first row: no earlier command exists to freeze, so N1 is undefined for this
            # clip. It is reported as not run with its reason; the other nulls keep their outcomes.
            out["plan_then_freeze"] = NullOutcome("plan_then_freeze", NOT_RUN,
                                                  "release at row 0: no command before the release to freeze")
        elif release_row >= len(controls):
            # the controls end at the release: no row exists on which the frozen command would be sent
            out["plan_then_freeze"] = NullOutcome("plan_then_freeze", NOT_RUN,
                                                  f"controls end at row {len(controls)}: no row at or after the "
                                                  f"release (row {release_row}) to freeze")
        else:
            ro = simulator(N.plan_then_freeze_controls(controls, release_row), hand_object_contact=True)
            bad = _invalid_rollout("plan_then_freeze", ro)
            n_run = len(as_pose7(ro.object_traj)) if bad is None else 0
            if bad is not None:
                out["plan_then_freeze"] = bad
            elif n_run <= release_row:
                # The run stopped before any frozen command executed (the same reach rule as the hand-absent
                # null). A prefix-scoring test can pass these rows, but they are the plan, not the freeze: the
                # prefix verdict is kept in the detail and the null itself is unevaluated.
                o = _score("plan_then_freeze", success_test, ro.object_traj)
                out["plan_then_freeze"] = NullOutcome(
                    "plan_then_freeze", INVALID, f"release row {release_row} outside the run's rows [0, {n_run}): "
                    "no frozen command executed, N1 is unevaluated", None,
                    {"sim_status": ro.status, "executed_rows": n_run, "prefix_passed": o.passed})
            else:
                o = _score("plan_then_freeze", success_test, ro.object_traj)
                out["plan_then_freeze"] = NullOutcome(o.null, o.status, o.note, None,
                                                      {**o.detail, "sim_status": ro.status})
    return ClipAudit(clip_id=clip_id, z0=out["static_at_start"].passed,
                     reference_passes=verdict(success_test(ref)), outcomes=out)


def wilson95(k: int, n: int) -> tuple[float, float] | None:
    """Wilson 95% interval for k of n (None when n = 0)."""
    if n == 0:
        return None
    z, p = 1.959963984540054, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


@dataclass(frozen=True)
class NullRate:
    """One null across clips, with the Z0 base rate it must be quoted beside."""

    null: str
    passed: int
    tested: int
    z0_passed: int
    z0_tested: int
    z1_passed: int
    z1_tested: int
    invalid: int
    not_run: int
    z0_clips: int
    clips: int
    unverified: int = 0          # passes whose hand-absent run was not verified as N0 (no force ledger)

    def sentence(self) -> str:
        if self.tested == 0:
            return (f"{self.null}: not tested on any clip ({self.not_run} not run, {self.invalid} invalid); "
                    f"Z0 base rate {self.z0_clips}/{self.clips}")
        unv = f", {self.unverified} of which N0-unverified" if self.unverified else ""
        s = (f"{self.null}: passes on {self.passed}/{self.tested} tested clips{unv} "
             f"(Z0 {self.z0_passed}/{self.z0_tested}, Z1 {self.z1_passed}/{self.z1_tested}); "
             f"Z0 base rate {self.z0_clips}/{self.clips} clips (clips where the static null passes)")
        if self.invalid or self.not_run:
            s += f"; {self.invalid} invalid, {self.not_run} not run"
        return s


@dataclass(frozen=True)
class AuditReport:
    clips: int
    z0_clips: int
    rates: dict[str, NullRate]
    reference_rejected: list[str]

    def headline(self) -> str:
        lines = [f"Z0 base rate: {self.z0_clips}/{self.clips} clips pass the static-at-start null."]
        lines += [r.sentence() for r in self.rates.values() if r.null != "static_at_start"]
        if self.reference_rejected:
            more = "..." if len(self.reference_rejected) > 5 else ""
            lines.append(f"WARNING: the test rejects its own reference on {len(self.reference_rejected)} clips "
                         f"({', '.join(self.reference_rejected[:5])}{more})")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        rates = {}
        for k, r in self.rates.items():
            rates[k] = {**asdict(r), "wilson95": wilson95(r.passed, r.tested),
                        "z0_wilson95": wilson95(r.z0_passed, r.z0_tested), "sentence": r.sentence()}
        return {"clips": self.clips, "z0_clips": self.z0_clips, "z0_base_rate": f"{self.z0_clips}/{self.clips}",
                "rates": rates, "reference_rejected": list(self.reference_rejected)}


def summarize(audits: Iterable[ClipAudit]) -> AuditReport:
    """Null-pass rates across clips, each with the Z0 base rate and its Z0 / Z1 split."""
    a = list(audits)
    z0 = sum(c.z0 for c in a)
    names: list[str] = []
    for c in a:
        names += [k for k in c.outcomes if k not in names]
    rates = {}
    for k in names:
        oc = [(c.z0, c.outcomes[k]) for c in a if k in c.outcomes]
        tested = [(z, o) for z, o in oc if o.status in (PASS, FAIL)]
        rates[k] = NullRate(
            null=k, passed=sum(o.passed for _, o in tested), tested=len(tested),
            z0_passed=sum(o.passed for z, o in tested if z), z0_tested=sum(1 for z, _ in tested if z),
            z1_passed=sum(o.passed for z, o in tested if not z), z1_tested=sum(1 for z, _ in tested if not z),
            invalid=sum(o.status == INVALID for _, o in oc), not_run=sum(o.status == NOT_RUN for _, o in oc),
            z0_clips=z0, clips=len(a),
            unverified=sum(o.passed and o.verified_n0 is False for _, o in tested))
    return AuditReport(clips=len(a), z0_clips=z0, rates=rates,
                       reference_rejected=[c.clip_id for c in a if not c.reference_passes])
