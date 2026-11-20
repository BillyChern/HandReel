# Copyright 2026 The HandReel authors. SPDX-License-Identifier: Apache-2.0
"""Mutant check: every plausible wrong implementation of the test, the nulls or the report fails the KA suite."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from _ka import Impl, run_ka

import nullaudit as na
from nullaudit import audit as A
from nullaudit.poses import as_pose7


def _errors(ex: np.ndarray, ref: np.ndarray, start: int = 0, pad_zero: bool = False, signed: bool = False):
    ex, ref = as_pose7(ex), as_pose7(ref)
    n = len(ex)
    if n <= len(ref):
        ref = ref[:n]
    else:
        tail = np.zeros((n - len(ref), 7)) if pad_zero else np.repeat(ref[-1:], n - len(ref), 0)
        ref = np.concatenate([ref, tail])
    pos = np.linalg.norm(ex[start:, :3] - ref[start:, :3], axis=1).mean()
    if signed:                                       # no shortest path: q and -q differ
        d = np.clip(np.sum(ex[:, 3:] * ref[:, 3:], 1), -1, 1)
        rot = (2 * np.arccos(d))[start:].mean()
    else:
        rot = na.quat_angle(ex[start:, 3:], ref[start:, 3:]).mean()
    return float(pos), float(rot)


def _mk(*, strict=True, skip_warmup=False, need_full=True, use_rot=True, pad_zero=False, signed=False,
        check_norm=True):
    def make(reference, required_rows, warmup_rows, pos_m=0.1, rot_rad=0.5):
        def test(ex):
            ex7 = as_pose7(ex)
            pos, rot = _errors(ex7, reference, warmup_rows if skip_warmup else 0, pad_zero, signed)
            full = (not need_full) or required_rows is None or len(ex7) >= required_rows
            if check_norm:
                full = full and bool((np.linalg.norm(ex7[:, 3:], axis=1) > 0).all())
            lt = (lambda a, b: a < b) if strict else (lambda a, b: a <= b)
            ok = full and lt(pos, pos_m) and (lt(rot, rot_rad) if use_rot else True)
            return na.ObjectTestResult(bool(ok), pos, rot, len(ex7), bool(full))
        return test
    return make


def _summarize_invalid_as_pass(audits):
    fixed = []
    for c in audits:
        oc = {k: (dataclasses.replace(o, status="pass") if o.status == "invalid" else o) for k, o in c.outcomes.items()}
        fixed.append(dataclasses.replace(c, outcomes=oc))
    return A.summarize(fixed)


def _summarize_z0_among_tested(audits):
    rep = A.summarize(audits)                        # base rate computed over the tested clips of each null
    rates = {k: dataclasses.replace(r, z0_clips=r.z0_tested, clips=r.tested) for k, r in rep.rates.items()}
    return dataclasses.replace(rep, rates=rates)


class _StatusOk:
    """Wraps a simulator and reports every run as complete (a mutant that ignores the simulator's status)."""

    def __init__(self, sim):
        self.sim = sim

    def __call__(self, controls, *, hand_object_contact=True):
        ro = self.sim(controls, hand_object_contact=hand_object_contact)
        return A.Rollout(ro.object_traj, ro.hand_object_force, "clip_end")


class _PadLedger:
    """Wraps a simulator and zero-pads a short force ledger to the trajectory length (a mutant without F2's check)."""

    def __init__(self, sim):
        self.sim = sim

    def __call__(self, controls, *, hand_object_contact=True):
        ro = self.sim(controls, hand_object_contact=hand_object_contact)
        f = ro.hand_object_force
        if f is not None and len(f) < len(ro.object_traj):
            f = np.concatenate([f, np.zeros(len(ro.object_traj) - len(f))])
        return A.Rollout(ro.object_traj, f, ro.status)


def _wrap_sim(wrapper):
    def audit_clip(*args, simulator=None, **kw):
        return A.audit_clip(*args, simulator=None if simulator is None else wrapper(simulator), **kw)
    return audit_clip


def _summarize_unverified_pooled(audits):
    rep = A.summarize(audits)
    return dataclasses.replace(rep, rates={k: dataclasses.replace(r, unverified=0) for k, r in rep.rates.items()})


MUTANTS = {
    "M1_threshold_le": Impl(make_test=_mk(strict=False)),
    "M2_warmup_excluded": Impl(make_test=_mk(skip_warmup=True)),
    "M3_prefix_scored": Impl(make_test=_mk(need_full=False)),
    "M4_rotation_ignored": Impl(make_test=_mk(use_rot=False)),
    "M5_pad_with_zeros": Impl(make_test=_mk(pad_zero=True)),
    "M6_signed_quat_angle": Impl(make_test=_mk(signed=True),
                                 quat_angle=lambda q, r: 2 * np.arccos(np.clip(np.sum(q * r, 1), -1, 1))),
    "M7_static_at_last_pose": Impl(static_at_start=lambda ref, rows=None: np.repeat(as_pose7(ref)[-1:], len(ref), 0)),
    "M8_freeze_at_release_row": Impl(plan_then_freeze_controls=lambda u, W: np.concatenate(
        [u[:W], np.repeat(u[W:W + 1], len(u) - W, 0)])),
    "M9_invalid_counted_as_pass": Impl(summarize=_summarize_invalid_as_pass),
    "M10_base_rate_over_tested": Impl(summarize=_summarize_z0_among_tested),
    "M11_invalid_rollout_scored": Impl(audit_clip=_wrap_sim(_StatusOk)),
    "M12_short_ledger_accepted": Impl(audit_clip=_wrap_sim(_PadLedger)),
    "M13_unverified_pooled": Impl(summarize=_summarize_unverified_pooled),
    "M14_zero_quaternion_accepted": Impl(make_test=_mk(check_norm=False)),
}


@pytest.mark.parametrize("name", sorted(MUTANTS))
def test_mutant_is_killed(name: str) -> None:
    killed_by = run_ka(MUTANTS[name])
    print(f"{name}: killed by {killed_by}")
    assert killed_by, f"mutant {name} survived the known-answer suite"
