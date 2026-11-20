# Copyright 2026 The HandReel authors. SPDX-License-Identifier: Apache-2.0
"""Known-answer suite on synthetic clips, parameterised by the implementation under test.

`run_ka(Impl())` must return no failures; every mutant in test_mutants.py must fail at least one check. Each check
has a closed-form answer.

Synthetic clip S(n, W, v, w): W welded warm-up rows at the start pose r_0 = (origin, identity), then for
k = 1 .. n - W the reference moves k * v along x and turns k * w about z. The static null's errors are
    E_pos = v (n - W)(n - W + 1) / (2 n),   E_rot = w (n - W)(n - W + 1) / (2 n),
i.e. the post-release mean (n - W + 1) v / 2 diluted by the factor (1 - W / n) (Proposition 1 in the README).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

import nullaudit as na
from nullaudit import audit as A
from nullaudit import nulls as N
from nullaudit.poses import quat_angle as _quat_angle


def synthetic_clip(n: int, W: int, v: float, w: float) -> np.ndarray:
    k = np.concatenate([np.zeros(W), np.arange(1, n - W + 1)]).astype(np.float64)
    ang = k * w
    ref = np.zeros((n, 7))
    ref[:, 0] = k * v
    ref[:, 3] = np.cos(ang / 2)
    ref[:, 6] = np.sin(ang / 2)
    return ref


def closed_form(n: int, W: int, x: float) -> float:
    return x * (n - W) * (n - W + 1) / (2 * n)


def to_matrices(p7: np.ndarray) -> np.ndarray:
    w, x, y, z = p7[:, 3], p7[:, 4], p7[:, 5], p7[:, 6]
    R = np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], 1)
    M = np.zeros((len(p7), 4, 4))
    M[:, :3, :3], M[:, :3, 3], M[:, 3, 3] = R, p7[:, :3], 1.0
    return M


def _default_test(reference: np.ndarray, required_rows: int | None, warmup_rows: int, pos_m: float = 0.1,
                  rot_rad: float = 0.5) -> Callable[[np.ndarray], Any]:
    return na.DoAsIDoObjectTest(reference, required_rows=required_rows, pos_m=pos_m, rot_rad=rot_rad)


@dataclass
class Impl:
    """The pieces a mutant may replace. `make_test(reference, required_rows, warmup_rows, pos_m, rot_rad)`."""

    make_test: Callable[..., Callable[[np.ndarray], Any]] = _default_test
    static_at_start: Callable[..., np.ndarray] = N.static_at_start
    plan_then_freeze_controls: Callable[..., np.ndarray] = N.plan_then_freeze_controls
    quat_angle: Callable[..., np.ndarray] = _quat_angle
    summarize: Callable[..., Any] = A.summarize
    audit_clip: Callable[..., Any] = A.audit_clip


class FakeSim:
    """A simulator whose hand-absent run leaves the object at its start pose. Faults: `leak` adds post-release hand
    force; `status` is reported for every run (e.g. "invalid"); `force` = "short" drops the ledger's last 10 rows,
    "none" reports no ledger."""

    def __init__(self, reference: np.ndarray, release_row: int, leak: bool = False, status: str = "clip_end",
                 force: str = "full") -> None:
        self.ref, self.W, self.leak, self.status, self.force = reference, release_row, leak, status, force

    def _f(self, f: np.ndarray) -> np.ndarray | None:
        return None if self.force == "none" else (f[:-10] if self.force == "short" else f)

    def __call__(self, controls: np.ndarray, *, hand_object_contact: bool = True) -> A.Rollout:
        n = len(controls)
        f = np.zeros(n)
        if self.leak:
            f[self.W + 5] = 3.0
        if hand_object_contact:                       # plan-then-freeze: the object keeps its last pre-release pose
            traj = np.repeat(self.ref[self.W - 1:self.W], n, 0)
            return A.Rollout(traj, self._f(f + 1.0), self.status)
        return A.Rollout(np.repeat(self.ref[:1], n, 0), self._f(f), self.status)


def run_ka(impl: Impl) -> list[str]:
    """Names of the failed known-answer checks (empty = all pass)."""
    fail: list[str] = []

    def check(name: str, ok: bool) -> None:
        if not ok:
            fail.append(name)

    # K1 static null on a Z0 clip: n = 1000, W = 400, post-release means 0.15 m / 0.6 rad -> 0.09 / 0.36, pass
    n, W = 1000, 400
    v, w = 0.3 / 601, 1.2 / 601
    ref = synthetic_clip(n, W, v, w)
    test = impl.make_test(ref, n, W)
    st = impl.static_at_start(ref)
    r = test(st)
    check("K1_static_pos", abs(r.pos_err_m - closed_form(n, W, v)) < 1e-12 and abs(closed_form(n, W, v) - 0.09) < 1e-12)
    check("K1_static_rot", abs(r.rot_err_rad - closed_form(n, W, w)) < 1e-12)
    check("K1_static_passes", bool(r.passed))
    post = float(np.linalg.norm(ref[W:, :3] - ref[0, :3], axis=1).mean())
    check("K1_prop1_identity", (post < 0.1 / (1 - W / n)) == bool(r.passed) and abs(post - 0.15) < 1e-12)
    a = impl.audit_clip(ref, test, release_row=W, nulls=na.KINEMATIC_NULLS)
    check("K1_z0", a.z0 is True and a.reference_passes)

    # K2 the same post-release motion with no warm-up: the first row is already k = 1, so the static null's errors
    # are the means of (k - 1) v and (k - 1) w = 599 v / 2 = 0.1495 m and 0.598 rad -> fails, Z1 (dilution, P4)
    ref2 = synthetic_clip(n - W, 0, v, w)
    t2 = impl.make_test(ref2, n - W, 0)
    r2 = t2(impl.static_at_start(ref2))
    check("K2_no_warmup_fails", (not r2.passed) and abs(r2.pos_err_m - 599 * v / 2) < 1e-12
          and abs(r2.rot_err_rad - 599 * w / 2) < 1e-12)
    check("K2_z1", impl.audit_clip(ref2, t2, nulls=na.KINEMATIC_NULLS).z0 is False)

    # K3 (T, 4, 4) input gives the (T, 7) answer
    r3 = test(to_matrices(st))
    check("K3_matrix_input", abs(r3.pos_err_m - r.pos_err_m) < 1e-12 and abs(r3.rot_err_rad - r.rot_err_rad) < 1e-9)

    # K4 strict thresholds: an error exactly at the threshold fails (dyadic 0.125 so the mean is exact)
    flat = np.zeros((64, 7))
    flat[:, 3] = 1.0
    t4 = impl.make_test(flat, 64, 0, pos_m=0.125)
    off = flat.copy()
    off[:, 0] = 0.125
    half = flat.copy()
    half[:, 0] = 0.0625
    check("K4_strict_lt", (not t4(off).passed) and bool(t4(half).passed))

    # K5 prefix scoring: a run that stops 10 rows after release passes on its prefix, fails if T rows are required
    short = st[:W + 10]
    check("K5_prefix_needs_full", (not test(short).passed) and bool(impl.make_test(ref, None, W)(short).passed))

    # K6 rotation is scored: exact positions, constant 0.6 rad yaw offset -> E_rot = 0.6, fails
    rot = flat.copy()
    rot[:, 3], rot[:, 6] = np.cos(0.3), np.sin(0.3)
    r6 = impl.make_test(flat, 64, 0)(rot)
    check("K6_rotation_scored", (not r6.passed) and abs(r6.rot_err_rad - 0.6) < 1e-12)

    # K7 q and -q are the same rotation
    neg = ref.copy()
    neg[:, 3:] *= -1.0
    r7 = test(neg)
    check("K7_sign_invariant", bool(r7.passed) and r7.rot_err_rad < 1e-6)
    check("K7_quat_angle", float(np.max(impl.quat_angle(neg[:, 3:], ref[:, 3:]))) < 1e-6)

    # K8 a run longer than the reference is scored against the reference's last row repeated
    p0 = flat.copy()
    p0[:, 0] = 1.0
    r8 = impl.make_test(p0, 64, 0)(np.repeat(p0[:1], 96, 0))
    check("K8_pad_last_row", bool(r8.passed) and r8.pos_err_m < 1e-12)

    # K9 plan-then-freeze holds u_{W-1} from row W on
    u = np.arange(50 * 3, dtype=float).reshape(50, 3)
    f = impl.plan_then_freeze_controls(u, 20)
    check("K9_freeze_rows", np.array_equal(f[:20], u[:20]) and bool((f[20:] == u[19]).all()))

    # K10 the report: rates beside the Z0 base rate; a leaking hand-absent run is invalid, never a pass
    ref_b = synthetic_clip(n, W, v * 0.5, w * 0.5)
    clips = [("z0_a", ref, False), ("z1", ref2, False), ("z0_leak", ref_b, True)]
    audits = []
    for cid, rr, leak in clips:
        Wc = W if len(rr) == n else 0
        Wsim = Wc if Wc else 1
        ctrl = np.zeros((len(rr), 2))
        audits.append(impl.audit_clip(rr, impl.make_test(rr, len(rr), Wc), clip_id=cid,
                                      simulator=FakeSim(rr, Wsim, leak),
                                      controls=ctrl, release_row=Wsim))
    rep = impl.summarize(audits)
    ha = rep.rates["hand_absent"]
    check("K10_base_rate", rep.z0_clips == 2 and rep.clips == 3 and "Z0 base rate 2/3" in ha.sentence())
    check("K10_hand_absent_counts", (ha.passed, ha.tested, ha.invalid, ha.z0_passed, ha.z0_tested) == (1, 2, 1, 1, 1))
    check("K10_headline_has_base_rate", all("Z0 base rate" in s for s in rep.headline().splitlines()[1:]))
    d = rep.to_dict()
    check("K10_dict_has_base_rate", all(v["z0_clips"] == 2 and v["clips"] == 3 for v in d["rates"].values()))

    # K12 a rollout the simulator reports "invalid" is INVALID for both simulated nulls, never scored (its rows would
    # pass: the fake keeps the object at the start pose of a Z0 clip)
    ctrl = np.zeros((n, 2))
    a12 = impl.audit_clip(ref, test, simulator=FakeSim(ref, W, status="invalid"), controls=ctrl, release_row=W)
    check("K12_invalid_rollout", a12.outcomes["hand_absent"].status == "invalid"
          and a12.outcomes["plan_then_freeze"].status == "invalid")

    # K13 a force ledger shorter than the trajectory cannot verify N0: INVALID, never verified_n0
    o13 = impl.audit_clip(ref, test, simulator=FakeSim(ref, W, force="short"), controls=ctrl, release_row=W,
                          nulls=("hand_absent",)).outcomes["hand_absent"]
    check("K13_ledger_length", o13.status == "invalid" and o13.verified_n0 is not True)

    # K14 an unverified hand-absent pass is counted and labelled, not pooled silently
    a14 = [impl.audit_clip(ref, test, clip_id="u", simulator=FakeSim(ref, W, force="none"), controls=ctrl,
                           release_row=W, nulls=("hand_absent",)),
           impl.audit_clip(ref_b, impl.make_test(ref_b, n, W), clip_id="v", simulator=FakeSim(ref_b, W),
                           controls=ctrl, release_row=W, nulls=("hand_absent",))]
    h14 = impl.summarize(a14).rates["hand_absent"]
    check("K14_unverified_counted", (h14.passed, h14.tested, getattr(h14, "unverified", None)) == (2, 2, 1)
          and "1 of which N0-unverified" in h14.sentence())

    # K15 a zero-norm quaternion row is treated like a non-finite row
    zq = ref.copy()
    zq[W + 3, 3:] = 0.0
    check("K15_zero_quaternion", not test(zq).passed and not impl.make_test(ref, None, W)(zq).passed)

    # K11 their loader's grid: 3 frames at 0.02 s on a 0.005 s grid, 2 warm-up rows, 20 rows
    g = na.reference_on_executor_grid(np.array([[0.0], [1.0], [3.0]]), 0.02, 0.005, 2, 20)
    expect = np.concatenate([[0.0, 0.0], np.interp(np.linspace(0, 2, 12), [0, 1, 2], [0.0, 1.0, 3.0]), [3.0] * 6])
    check("K11_grid", g.shape == (20, 1) and g.dtype == np.float32 and np.allclose(g[:, 0], expect, atol=1e-6))
    return fail
