# Copyright 2026 The HandReel authors. SPDX-License-Identifier: Apache-2.0
"""Grip census: how hard the hand squeezes the object, as a multiple of the object's weight.

A port of the census loop of our contact studies. At every `stride`-th post-release row (rows W, W + stride, ...)
the executed state (qpos, qvel, ctrl) is set on a fresh `MjData` with the run's gravity and the warm-up welds off,
`mj_forward` re-solves the contacts, and the normal components (`mj_contactForce`, entry 0) of every active contact
between an object geom and a LEDGER geom are summed: the hand mask plus every other dynamic geom
(`nullaudit.mujoco_adapter.contact_ledger_geoms`), the same set the N0 check of the contact-removed null uses. The
part from geoms outside the hand mask is reported separately (`other_force_n_p50`). The per-run statistic is the
median over the sampled rows of the total divided by the object's weight m |g|.

Stated approximation: the forward re-solve at a stored state is a weight, not the force of the executed step; a
contact-sensitivity study measured the in-step force at 0.990-1.013 of it on 53 OakInk2 runs. Every number is a
property of the simulator's contact model (solref, solimp, condim, friction), not of the grip a task needs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .errors import InputError

__all__ = ["F_RATED_N", "GripCensus", "check_object_mask", "grip_census", "object_weight"]

F_RATED_N = 20.0      # a per-contact force above this counts as over a dexterous hand's rating


@dataclass(frozen=True)
class GripCensus:
    """Per-run census. Ratios are ledger normal force on the object / object weight; None when no row was sampled.
    `replay_status`: "executed" (stored rows) or the replay's end status; `truncated_at`: the row at which a
    non-finite state ended the sampling (None if none did)."""

    weight_n: float
    rows_sampled: int
    grip_over_weight_p50: float | None
    grip_over_weight_p90: float | None
    force_n_p50: float | None
    contacts_p50: float | None
    max_contact_n_p50: float | None
    rows_any_contact_over_rating: float | None
    grip_tracked_p50: float | None = None
    frac_rows_tracked: float | None = None
    other_force_n_p50: float | None = None
    replay_status: str = "executed"
    truncated_at: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def object_weight(model: Any, object_geom: np.ndarray, gravity: np.ndarray) -> float:
    """m |g| of the body that carries the object geoms."""
    body = int(model.geom_bodyid[np.flatnonzero(object_geom)[0]])
    return float(model.body_mass[body] * np.linalg.norm(gravity))


def _mask(x: Any, n: int) -> np.ndarray:
    a = np.asarray(x)
    if a.dtype == bool:
        if a.shape != (n,):
            raise InputError(f"geom mask has shape {a.shape}, expected ({n},)")
        return a
    ids = a.astype(int).ravel()
    if ids.size and (ids.min() < 0 or ids.max() >= n):
        raise InputError(f"geom ids {ids.tolist()[:8]} outside [0, {n})")
    m = np.zeros(n, bool)
    m[ids] = True
    return m


def _free_joint(model: Any) -> tuple[int, int]:
    """(body, qpos address) of the scene's only free joint."""
    import mujoco

    free = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    if len(free) != 1:
        raise InputError(f"expected exactly one free joint (the object), found {len(free)}")
    return int(model.jnt_bodyid[free[0]]), int(model.jnt_qposadr[free[0]])


def check_object_mask(model: Any, object_geom: Any, hand_geom: Any) -> np.ndarray:
    """The object mask as a bool array, refused unless it is consistent with the scene: every masked geom is on the
    free joint's body, and every geom of that body left out of the mask has no contact path (explicit `<pair>` or
    default contype/conaffinity collision) to a ledger geom, so leaving it out cannot hide hand force. (The studies'
    masks leave out the zero-contype support markers some scenes attach to the object body.)"""
    from nullaudit.mujoco_adapter import contact_ledger_geoms

    obj = _mask(object_geom, model.ngeom)
    body, _ = _free_joint(model)
    free = np.asarray(model.geom_bodyid == body)
    if (obj & ~free).any() or not obj.any():
        raise InputError("the object mask is empty or has geoms off the free joint's body")
    left = free & ~obj
    if left.any():
        led = contact_ledger_geoms(model, _mask(hand_geom, model.ngeom), free)
        ct, ca = model.geom_contype, model.geom_conaffinity
        for g in np.flatnonzero(left):
            paired = any((int(model.pair_geom1[p]) == g and led[model.pair_geom2[p]])
                         or (int(model.pair_geom2[p]) == g and led[model.pair_geom1[p]]) for p in range(model.npair))
            if paired or any((ct[g] & ca[h]) or (ct[h] & ca[g]) for h in np.flatnonzero(led)):
                raise InputError(f"geom {g} of the free joint's body is outside the object mask but can touch the "
                                 "hand or another dynamic body: the object mask is inconsistent")
    return obj


def grip_census(model: Any, qpos: np.ndarray, qvel: np.ndarray, ctrl: np.ndarray, *, hand_geom: Any,
                object_geom: Any, release_row: int, gravity: np.ndarray | None = None, stride: int = 5,
                ref_pos: np.ndarray | None = None, rated_n: float = F_RATED_N, track_m: float = 0.1,
                weld_eq_ids: Sequence[int] = (), replay_status: str = "executed") -> GripCensus:
    """The grip census of one executed run.

    Args:
        model: the scene's MjModel (its gravity is set for the re-solve and restored afterwards).
        qpos, qvel, ctrl: (n, nq), (n, nv), (n, nu) executed post-step states and the controls applied on each row.
        hand_geom, object_geom: (ngeom,) bool masks or geom-id sequences. The object mask must be consistent with
            the free joint's body (`check_object_mask`).
        release_row: W, the first post-release row; sampling starts there.
        gravity: post-release gravity (default: the model's).
        stride: sample every `stride`-th row (the studies used 5).
        ref_pos: optional (n, 3) reference object positions; adds the median over rows whose object position error
            is < `track_m` (the "tracked" force).
        rated_n: per-contact rating for `rows_any_contact_over_rating`.
        weld_eq_ids: the warm-up welds, switched off on every sampled (post-release) row.
        replay_status: carried into the result ("executed" for stored rows).

    A non-finite state ends the sampling and is reported in `truncated_at`.
    """
    import mujoco
    from nullaudit.mujoco_adapter import contact_ledger_geoms

    if int(stride) != stride or stride < 1:
        raise InputError(f"stride {stride!r} must be an integer >= 1")
    hand = _mask(hand_geom, model.ngeom)
    obj = check_object_mask(model, object_geom, hand)
    _, qadr = _free_joint(model)
    if not hand.any() or (hand & obj).any():
        raise InputError("the hand mask is empty or overlaps the object")
    led = contact_ledger_geoms(model, hand, obj)
    g = np.array(model.opt.gravity if gravity is None else gravity, dtype=np.float64)
    weight = object_weight(model, obj, g)
    qpos, qvel, ctrl = (np.asarray(x, np.float64) for x in (qpos, qvel, ctrl))
    d = mujoco.MjData(model)
    f6 = np.zeros(6)
    tot, other, ncon, fmax, rows = [], [], [], [], []
    truncated = None
    g_model = np.array(model.opt.gravity)
    try:
        model.opt.gravity[:] = g
        for r in range(int(release_row), qpos.shape[0], int(stride)):
            if not (np.isfinite(qpos[r]).all() and np.isfinite(qvel[r]).all()):
                truncated = r
                break
            d.qpos[:], d.qvel[:], d.ctrl[:] = qpos[r], qvel[r], ctrl[r]
            for e in weld_eq_ids:
                d.eq_active[int(e)] = 0
            mujoco.mj_forward(model, d)
            fs, fo = [], 0.0
            for i in range(d.ncon):
                c = d.contact[i]
                g1, g2 = int(c.geom1), int(c.geom2)
                if obj[g1] == obj[g2] or not led[g2 if obj[g1] else g1]:
                    continue
                mujoco.mj_contactForce(model, d, i, f6)
                fs.append(float(f6[0]))
                if not hand[g2 if obj[g1] else g1]:
                    fo += float(f6[0])
            tot.append(sum(fs))
            other.append(fo)
            ncon.append(len(fs))
            fmax.append(max(fs) if fs else 0.0)
            rows.append(r)
    finally:
        model.opt.gravity[:] = g_model
    if not tot:
        return GripCensus(weight, 0, None, None, None, None, None, None, replay_status=replay_status,
                          truncated_at=truncated)
    t = np.asarray(tot)
    tracked = frac = None
    if ref_pos is not None:
        idx = np.asarray(rows, np.int64)
        held = np.linalg.norm(qpos[idx, qadr:qadr + 3] - np.asarray(ref_pos)[idx], axis=1) < track_m
        tracked = float(np.median(t[held] / weight)) if held.any() else None
        frac = float(held.mean())
    return GripCensus(weight_n=weight, rows_sampled=len(tot),
                      grip_over_weight_p50=float(np.median(t / weight)),
                      grip_over_weight_p90=float(np.percentile(t / weight, 90)),
                      force_n_p50=float(np.median(t)), contacts_p50=float(np.median(ncon)),
                      max_contact_n_p50=float(np.median(fmax)),
                      rows_any_contact_over_rating=float(np.mean(np.asarray(fmax) > rated_n)),
                      grip_tracked_p50=tracked, frac_rows_tracked=frac,
                      other_force_n_p50=float(np.median(other)), replay_status=replay_status,
                      truncated_at=truncated)
