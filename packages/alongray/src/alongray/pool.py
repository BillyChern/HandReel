# Copyright 2026 The alongray authors
# SPDX-License-Identifier: Apache-2.0
"""Pooling per-unit splits into a corpus summary, the cluster bootstrap, and splits from stored moments.

The pooled share weights units by their mean square: ``S = sum_u gauge_part_u^2 / sum_u e_par_u^2``. Medians are over
units. The RMS over units of each part also splits exactly: ``rms_u(e_par)^2 = rms_u(gauge)^2 + rms_u(off_orbit)^2``.

The bootstrap resamples CLUSTERS (videos, sequences) with replacement within one group. Clusters are the sorted
distinct labels; each draw is ONE call ``pick = rng.integers(0, K, K)`` on a numpy Generator (K = number of
clusters), and the drawn units are those of clusters ``pick[0], pick[1], ...``, each cluster's units in their original
order. The interval is ``numpy.percentile`` (default linear interpolation) at ``(100 - 100 level) / 2`` and
``(100 + 100 level) / 2``. An analysis that follows the same order with the same generator gets the same draws.
With fewer than two distinct clusters there is no between-cluster information: the interval is the string
:data:`NOT_ESTIMABLE`, never a zero-width interval (a one-cluster call still makes its draws, so that a generator
shared with later groups advances exactly as before).

``python -m alongray.pool ROWS.jsonl`` (also ``alongray-pool``) pools rows that each carry ONE of three forms. Here
``a_r`` is the prediction's RANGE along the true ray, ``a_r = <c_pred, c_gt> / |c_gt|``; it is NOT the along-ray
error ``a = a_r - Z`` of :mod:`alongray.core`.

1. ``"moments"``: ``pp = <a_r^2>``, ``pz = <a_r Z>``, ``zz = <Z^2>``, ``n``, optionally ``z1 = <Z>`` and
   ``a1 = <a_r>`` (which add the shift reading).
2. Per-frame ``"z"`` (true range, every value > 0) and ``"a_r"`` lists. The key ``"a"`` is accepted as a deprecated
   alias of ``"a_r"`` and warns (FutureWarning).
3. ``"split"``: an :class:`alongray.core.EParSplit` as a dict in metres (e.g. ``dataclasses.asdict``); the fields
   ``e_par, gauge_part, off_orbit_part, lam, z_rms, oracle_floor, n_frames`` are required and must be finite.

A row may also be ``"status": "failed"`` or ``"excluded"`` (with a ``"reason"``) and carry no form; every group
reports its assigned, scored, failed, excluded and missing counts (:func:`pool_rows`).
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import sys
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np

from alongray import strict_json
from alongray.core import MIN_ROWS, EParSplit, e_par_split

__all__ = ["MOMENT_RTOL", "NOT_ESTIMABLE", "SPLIT_RTOL", "cluster_bootstrap", "main", "pool_rows", "pool_splits",
           "pooled_shares", "split_from_moments", "split_of_row"]

_LENGTHS = ("e_par", "gauge_part", "off_orbit_part", "oracle_floor", "shift_part")
_REQUIRED = ("e_par", "gauge_part", "off_orbit_part", "lam", "z_rms", "oracle_floor", "n_frames")
# Relative tolerance of the Cauchy-Schwarz checks on moments: float rounding can push an exact zero (a pure scale's
# off-orbit square) slightly negative; anything beyond this is refused as moments of no real trajectory.
MOMENT_RTOL = 1e-9
# Relative tolerance of the identities a stored split must satisfy (e_par^2 = gauge^2 + off^2, gauge = |lam - 1| z_rms,
# oracle_floor = off / lam). Stored splits written by this package agree to rounding; a rounded copy may need more.
SPLIT_RTOL = 1e-6
# What cluster_bootstrap returns, in place of an interval, with fewer than two distinct clusters.
NOT_ESTIMABLE = "NOT_ESTIMABLE"


class UndefinedStatistic(ValueError):
    """A statistic that does not exist for the given units (a share of zero total error). In a bootstrap this is an
    undefined DRAW, counted and left out of the percentiles; it is never a reason to drop the point estimate."""


def _count(x: Any, name: str) -> int:
    """A frame count: a positive integer (an integral float is accepted, a bool is not)."""
    ok = (isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))) or (
        isinstance(x, float) and math.isfinite(x) and x.is_integer())
    if not ok or int(x) < 1:
        raise ValueError(f"{name} = {x!r} must be a positive integer")
    return int(x)


def _check_moments(pp: float, pz: float, zz: float, z1: float, a1: float) -> None:
    """Refuse moments that no real trajectory has (module constant MOMENT_RTOL)."""
    if not all(math.isfinite(v) for v in (pp, pz, zz)):
        raise ValueError(f"moments pp, pz, zz must be finite, got {pp!r}, {pz!r}, {zz!r}")
    if not zz > 0:
        raise ValueError(f"zz = {zz!r} must be > 0")
    if pp < 0:
        raise ValueError(f"pp = {pp!r} is a mean square and cannot be negative")
    if pz * pz > pp * zz * (1 + MOMENT_RTOL):
        raise ValueError(f"inconsistent moments: pz^2 = {pz * pz!r} exceeds pp zz = {pp * zz!r} (Cauchy-Schwarz); "
                         "the off-orbit square would be negative")
    if math.isfinite(z1) and not (z1 > 0 and z1 * z1 <= zz * (1 + MOMENT_RTOL)):
        raise ValueError(f"z1 = {z1!r} must be a positive mean range with z1^2 <= zz (Cauchy-Schwarz)")
    if math.isfinite(a1) and a1 * a1 > pp * (1 + MOMENT_RTOL):
        raise ValueError(f"a1 = {a1!r}: a1^2 exceeds pp (Cauchy-Schwarz)")


def split_from_moments(pp: float, pz: float, zz: float, n: int, z1: float = math.nan,
                       a1: float = math.nan) -> EParSplit:
    """The split from per-unit moments (``lam = pz / zz``, ``e_par^2 = pp - 2 pz + zz``,
    ``gauge^2 = (lam - 1)^2 zz``, ``off^2 = pp - pz^2 / zz``).

    The shift reading needs ``z1 = <Z>`` and ``a1 = <a_r>``: ``rbar = a1 - z1`` is the mean error, ``shift_part =
    |rbar|``, ``collinearity = z1^2 / zz``, and ``shift_scale_part`` solves the 2 x 2 Gram system of the fit of the
    error on ``span{1, Z}``: with ``rz = pz - zz`` (``= <a Z>``) and ``det = zz - z1^2``,
    ``b0 = (zz rbar - z1 rz) / det``, ``b1 = (rz - z1 rbar) / det``, ``shift_scale_part^2 = b0 rbar + b1 rz``; when
    ``det <= 1e-15 zz`` (constant range) the two directions coincide and it is ``max(gauge^2, rbar^2)``.

    Moments that no real trajectory has are refused, not clamped: ``pz^2 > pp zz``, ``z1^2 > zz`` or ``a1^2 > pp``
    beyond the relative tolerance ``MOMENT_RTOL`` (Cauchy-Schwarz), a negative ``pp``, a non-positive ``z1``, or a
    shift reading outside ``[0, e_par]``. Within the tolerance a rounded negative square is set to zero.

    Raises:
        ValueError: non-finite moments, ``zz`` not positive, ``n`` not a positive integer, inconsistent moments, or
            ``lam`` not finite and positive.
    """
    n = _count(n, "n")
    _check_moments(pp, pz, zz, z1, a1)
    lam = pz / zz
    if not (math.isfinite(lam) and lam > 0):
        raise ValueError(f"lam = {lam!r} is not finite and positive")
    scale = pp + zz
    e2 = pp - 2.0 * pz + zz                    # >= 0 whenever Cauchy-Schwarz holds, up to rounding
    off2 = pp - pz * pz / zz
    e2, off2 = max(e2, 0.0), max(off2, 0.0)    # only rounding below zero is left after _check_moments
    off = math.sqrt(off2)
    g2 = (lam - 1.0) ** 2 * zz
    shift = both = coll = math.nan
    if math.isfinite(z1) and math.isfinite(a1):
        rbar, rz, det = a1 - z1, pz - zz, zz - z1 * z1
        if rbar * rbar > e2 + MOMENT_RTOL * scale:
            raise ValueError("inconsistent moments: the squared mean error (a1 - z1)^2 exceeds e_par^2")
        if det > 1e-15 * zz:
            b0 = (zz * rbar - z1 * rz) / det
            b1 = (rz - z1 * rbar) / det
            b2 = b0 * rbar + b1 * rz
            if not (-MOMENT_RTOL * scale <= b2 <= e2 + MOMENT_RTOL * scale):
                raise ValueError("inconsistent moments: the shift-and-scale fit explains a share of e_par^2 outside "
                                 "[0, 1]")
            b2 = min(max(b2, 0.0), e2)
        else:
            b2 = max(g2, rbar * rbar)
        shift, both, coll = abs(rbar), math.sqrt(b2), z1 * z1 / zz
    return EParSplit(e_par=math.sqrt(e2), gauge_part=math.sqrt(g2), off_orbit_part=off, lam=lam,
                     z_rms=math.sqrt(zz), oracle_floor=off / lam, n_frames=int(n), shift_part=shift,
                     shift_scale_part=both, collinearity=coll, low_n=int(n) < MIN_ROWS)


def _check_stored_split(sp: dict[str, Any], rtol: float, uid: object) -> None:
    """A stored split must be one the definitions produce: positive n and lam, non-negative lengths, and the three
    identities to ``rtol``."""
    _count(sp["n_frames"], f"row {uid!r}: n_frames")
    lam, z, e, g, o, f = (float(sp[k]) for k in ("lam", "z_rms", "e_par", "gauge_part", "off_orbit_part",
                                                 "oracle_floor"))
    if not lam > 0:
        raise ValueError(f"row {uid!r}: lam = {lam!r} must be positive")
    if not z > 0:
        raise ValueError(f"row {uid!r}: z_rms = {z!r} must be positive")
    if min(e, g, o, f) < 0:
        raise ValueError(f"row {uid!r}: e_par, gauge_part, off_orbit_part and oracle_floor must be non-negative")
    tiny = 1e-30
    if abs(e * e - g * g - o * o) > rtol * max(e * e, g * g + o * o, tiny):
        raise ValueError(f"row {uid!r}: the identity e_par^2 = gauge_part^2 + off_orbit_part^2 fails "
                         f"({e * e!r} against {g * g + o * o!r})")
    gz = abs(lam - 1.0) * z
    if abs(g - gz) > rtol * max(g, gz) + tiny:
        raise ValueError(f"row {uid!r}: gauge_part = {g!r} is not |lam - 1| z_rms = {gz!r}")
    if abs(f - o / lam) > rtol * max(f, o / lam) + tiny:
        raise ValueError(f"row {uid!r}: oracle_floor = {f!r} is not off_orbit_part / lam = {o / lam!r}")


def split_of_row(r: dict[str, Any], rtol: float = SPLIT_RTOL) -> EParSplit:
    """A unit's split from one row (module docstring for the three forms).

    Moments go through :func:`split_from_moments` (inconsistent moments are refused). A stored split must have a
    positive integer ``n_frames``, positive ``lam`` and ``z_rms``, non-negative lengths, and satisfy
    ``e_par^2 = gauge_part^2 + off_orbit_part^2``, ``gauge_part = |lam - 1| z_rms`` and
    ``oracle_floor = off_orbit_part / lam`` to the relative tolerance ``rtol``.

    Raises:
        ValueError: the row carries none of the three forms, or its form is inconsistent.
    """
    if "moments" in r:
        m = r["moments"]
        return split_from_moments(m["pp"], m["pz"], m["zz"], m["n"], m.get("z1", math.nan), m.get("a1", math.nan))
    if "z" in r and ("a_r" in r or "a" in r):
        if "a_r" not in r:
            warnings.warn("row key 'a' is deprecated: use 'a_r' (the predicted range along the true ray, not the "
                          "error)", FutureWarning, stacklevel=2)
        z = np.asarray(r["z"], np.float64)
        a = np.asarray(r["a_r"] if "a_r" in r else r["a"], np.float64)
        if z.ndim != 1 or z.shape != a.shape:
            raise ValueError(f"row {r.get('unit_id')!r}: z and a_r must be equal-length lists")
        if not (np.isfinite(z).all() and (z > 0).all()):
            raise ValueError(f"row {r.get('unit_id')!r}: every true range z must be finite and > 0")
        e = np.zeros((len(z), 3))
        e[:, 2] = 1.0            # points on one ray at ranges a_r and z: the along-ray reading needs nothing else
        return e_par_split(a[:, None] * e, z[:, None] * e)
    if "split" in r:
        sp = r["split"]
        if not isinstance(sp, dict):
            raise ValueError(f"row {r.get('unit_id')!r}: 'split' must be an object")
        missing = [k for k in _REQUIRED if k not in sp]
        if missing:
            raise ValueError(f"row {r.get('unit_id')!r}: split lacks {missing}")
        bad = [k for k in _REQUIRED if not (isinstance(sp[k], (int, float)) and math.isfinite(sp[k]))]
        if bad:
            raise ValueError(f"row {r.get('unit_id')!r}: split fields {bad} are not finite numbers")
        _check_stored_split(sp, rtol, r.get("unit_id"))
        kw = {f.name: sp[f.name] for f in fields(EParSplit) if f.name in sp}
        kw["n_frames"] = int(sp["n_frames"])
        kw.setdefault("low_n", kw["n_frames"] < MIN_ROWS)
        return EParSplit(**kw)
    raise ValueError(f"row {r.get('unit_id')!r} has none of moments / z,a / split")


def pooled_shares(splits: Sequence[EParSplit]) -> dict[str, float]:
    """Mean-square-weighted shares of ``e_par^2``: gauge, off-orbit, one constant shift, joint shift-and-scale, and
    ``pure_shift_null`` (the share the gauge part would carry if each unit's error were one constant shift of its
    size: ``sum e^2 c / sum e^2``). The last three need each unit's first moments (z1, a1); if any unit lacks
    them, those three are NaN here and ``pool_splits`` reports them as undefined by name (``FIRST_MOMENT_SHARES``).

    Raises:
        ValueError: every unit has zero error.
    """
    e2 = sum(s.e_par ** 2 for s in splits)
    if not e2 > 0:
        raise UndefinedStatistic("every unit has zero error; shares are undefined")
    return {"gauge": sum(s.gauge_part ** 2 for s in splits) / e2,
            "off_orbit": sum(s.off_orbit_part ** 2 for s in splits) / e2,
            "shift": sum(s.shift_part ** 2 for s in splits) / e2,
            "shift_scale": sum(s.shift_scale_part ** 2 for s in splits) / e2,
            "pure_shift_null": sum(s.e_par ** 2 * s.collinearity for s in splits) / e2}


FIRST_MOMENT_SHARES = ("shift", "shift_scale", "pure_shift_null")


def _median(x: Sequence[float]) -> float:
    v = [float(a) for a in x if math.isfinite(a)]
    return float(np.median(v)) if v else math.nan


def pool_splits(splits: Sequence[EParSplit], unit_scale: float = 1.0) -> dict[str, Any]:
    """Medians, means and RMS over units of each part (lengths x ``unit_scale``), the pooled shares, the median
    per-unit gauge share and collinearity, and ``n_low_n``. When every unit has zero error, ``pooled_share`` is
    None with a ``pooled_share_reason``; the zero lengths and the counts are still returned.

    Raises:
        ValueError: no unit.
    """
    if not splits:
        raise ValueError("no unit to pool")
    out: dict[str, Any] = {"n_units": len(splits), "n_low_n": sum(bool(s.low_n) for s in splits)}
    for k in _LENGTHS:
        v = [getattr(s, k) * unit_scale for s in splits]
        fin = np.asarray([x for x in v if math.isfinite(x)], np.float64)
        out[f"median_{k}"] = _median(v)
        out[f"mean_{k}"] = float(fin.mean()) if fin.size else math.nan
        out[f"rms_{k}"] = float(np.sqrt(np.mean(fin * fin))) if fin.size else math.nan
    out["median_abs_log_lam"] = _median([abs(math.log(s.lam)) for s in splits])
    out["median_unit_gauge_share"] = _median([s.gauge_part ** 2 / s.e_par ** 2 if s.e_par > 0 else math.nan
                                              for s in splits])
    out["median_collinearity"] = _median([s.collinearity for s in splits])
    try:
        share: dict[str, float | None] = dict(pooled_shares(splits))
        bare = sum(not math.isfinite(s.shift_part) for s in splits)
        undefined = {k: f"{bare} of {len(splits)} units carry no first moments (z1, a1); this share needs them "
                        "from every unit" for k in FIRST_MOMENT_SHARES if not math.isfinite(share[k])}
        if undefined:
            # A3 #10777: a component that does not exist for the cohort is named with its reason, never a bare null
            share.update(dict.fromkeys(undefined))
            out["pooled_share_undefined"] = undefined
        out["pooled_share"] = share
    except UndefinedStatistic:
        # every unit has zero error: the lengths above (all zero) and the counts are defined and are kept; only
        # the share of a zero total does not exist
        out["pooled_share"] = None
        out["pooled_share_reason"] = "every unit has zero error; a share of a zero total is undefined"
    return out


def cluster_bootstrap(stat: Callable[[list[int]], float | dict[str, float]], clusters: Sequence[str],
                      n_boot: int = 10_000, rng: np.random.Generator | int = 0,
                      level: float = 0.95, info: dict[str, int] | None = None
                      ) -> dict[str, tuple[float, float]] | str:
    """Percentile interval of ``stat(unit indices)`` under resampling of clusters (module docstring).

    Returns ``{key: (lo, hi)}`` (key ``"value"`` for a scalar ``stat``), or :data:`NOT_ESTIMABLE` when there are
    fewer than two distinct clusters or when no draw has a defined statistic.

    A draw on which ``stat`` raises :class:`UndefinedStatistic` (for a pooled share: every drawn cluster has zero
    error) is an undefined draw. It still consumes its random numbers, it is counted in ``info["undefined_draws"]``
    when ``info`` is given, and the percentiles are taken over the defined draws only, so the interval is then
    conditional on the statistic being defined. A non-finite value of one key of a dict statistic is undefined for
    that key alone: it is counted in ``info["undefined_by_key"]`` and left out of that key's percentiles, and a
    key with no finite draw has no entry in the result. The caller's point estimate is not touched.

    Args:
        stat: maps a list of unit indices (with repeats) to a number or a dict of numbers.
        clusters: cluster label of every unit.
        n_boot: draws.
        rng: a numpy Generator (shared across groups to reproduce a sequential analysis) or an integer seed.
        level: central coverage.
    """
    gen = np.random.default_rng(rng) if isinstance(rng, int) else rng
    labels = sorted(set(clusters))
    if not labels:
        return NOT_ESTIMABLE
    by = {c: [i for i, x in enumerate(clusters) if x == c] for c in labels}
    draws: dict[str, list[float]] = {}
    by_key: dict[str, int] = {}
    undefined = 0
    for _ in range(n_boot):
        pick = gen.integers(0, len(labels), len(labels))
        try:
            v = stat([i for j in pick for i in by[labels[j]]])
        except UndefinedStatistic:
            undefined += 1
            continue
        for k, x in (v.items() if isinstance(v, dict) else (("value", v),)):
            if math.isfinite(float(x)):
                draws.setdefault(k, []).append(float(x))
            else:                                           # undefined for this key on this draw: counted, never
                by_key[k] = by_key.get(k, 0) + 1            # passed to a percentile
    if info is not None:
        info.update(n_boot=n_boot, undefined_draws=undefined, undefined_by_key=by_key)
    if len(labels) < 2 or not draws:
        return NOT_ESTIMABLE                                # the draws above only advance a shared generator
    lo, hi = (100 - 100 * level) / 2, (100 + 100 * level) / 2
    return {k: (float(np.percentile(v, lo)), float(np.percentile(v, hi))) for k, v in draws.items()}


def _shares_of(splits: list[EParSplit], idx: list[int]) -> dict[str, float]:
    return pooled_shares([splits[i] for i in idx])


STATUSES = ("scored", "failed", "excluded")
FAILURE_POLICIES = ("report", "refuse")


def _cohort(name: str, rows: list[dict[str, Any]], manifest: set[str] | None, policy: str) -> dict[str, Any]:
    """Denominators of one group, after checking each row's status; refuses under policy "refuse"."""
    ids = [str(r["unit_id"]) for r in rows if "unit_id" in r]
    dup = sorted({u for u in ids if ids.count(u) > 1})
    if dup:
        raise ValueError(f"group {name!r}: unit(s) {dup} listed twice")
    reasons: dict[str, int] = {}
    for r in rows:
        st = r.get("status", "scored")
        if st not in STATUSES:
            raise ValueError(f"row {r.get('unit_id')!r}: status {st!r} is not one of {STATUSES}")
        has_form = any(k in r for k in ("moments", "split", "z"))
        if st != "scored" and has_form:
            raise ValueError(f"row {r.get('unit_id')!r}: a {st} unit carries a split; mark it scored or drop the data")
        if st == "excluded":
            reason = r.get("reason")
            if not (isinstance(reason, str) and reason.strip()):
                raise ValueError(f"row {r.get('unit_id')!r}: an excluded unit needs a 'reason'")
            reasons[reason] = reasons.get(reason, 0) + 1
    missing = sorted(manifest - set(ids)) if manifest is not None else []
    n = {k: sum(r.get("status", "scored") == k for r in rows) for k in STATUSES}
    out = {"n_assigned": len(rows) + len(missing), "n_scored": n["scored"], "n_failed": n["failed"],
           "n_excluded": n["excluded"], "n_missing": len(missing), "excluded_reasons": reasons,
           "failure_policy": policy, "assignment": "manifest" if manifest is not None else "observed rows only"}
    out["cohort"] = (f"scored units only: {n['scored']} of {out['n_assigned']} assigned ({n['failed']} failed, "
                     f"{n['excluded']} excluded, {len(missing)} missing); failed, excluded and missing units are not "
                     "in any statistic")
    if policy == "refuse" and (n["failed"] or missing):
        raise ValueError(f"group {name!r}: {n['failed']} failed and {len(missing)} missing unit(s) "
                         f"{missing[:5]} under failure_policy='refuse'")
    return out


def pool_rows(rows: list[dict[str, Any]], group_key: str | None = None, cluster_key: str | None = None,
              n_boot: int = 0, seed: int = 0, unit_scale: float = 1e3, *,
              assigned: Sequence[str] | Mapping[str, str] | None = None,
              failure_policy: str = "report") -> dict[str, Any]:
    """``{group: pool_splits(...)}`` over each group's scored units, with the group's cohort denominators and, when
    ``n_boot > 0``, ``pooled_share_ci95`` from one Generator seeded once and consumed group by group in sorted
    group order (the interval, like every statistic, is over the scored units). A group with fewer than two
    distinct clusters gets ``pooled_share_ci95 = NOT_ESTIMABLE`` and a ``pooled_share_ci95_reason``.

    A row's ``status`` is ``"scored"`` (the default: it carries one of the three forms), ``"failed"`` (assigned, but
    the method delivered nothing to score) or ``"excluded"`` (left out by a stated rule; it needs a ``"reason"``).
    Each group reports ``n_assigned``, ``n_scored``, ``n_failed``, ``n_excluded``, ``n_missing``,
    ``excluded_reasons`` and a ``cohort`` sentence.

    Args:
        assigned: the assigned unit ids (a sequence, without ``group_key``) or ``{unit_id: group}``. An assigned
            unit without a row counts as missing; a row whose unit is not assigned, or whose group differs from
            the manifest's, is refused. Unit ids are compared as strings. Rows need
            ``unit_id`` when it is given.
        failure_policy: ``"report"`` pools the scored units and reports the denominators beside them;
            ``"refuse"`` refuses any group with a failed or missing unit, so that its statistics describe every
            assigned unit not excluded by a stated reason.

    Raises:
        ValueError: ``n_boot > 0`` without ``cluster_key``, an unknown policy, a malformed or refused cohort, or a
            row that fails :func:`split_of_row`.
    """
    if n_boot > 0 and not cluster_key:
        raise ValueError("the bootstrap resamples clusters: give cluster_key")
    if failure_policy not in FAILURE_POLICIES:
        raise ValueError(f"failure_policy must be one of {FAILURE_POLICIES}, got {failure_policy!r}")
    manifest: dict[str, str] | None = None
    if assigned is not None:
        names = [str(u) for u in assigned]                   # a mapping gives its keys
        twice = sorted({u for u in names if names.count(u) > 1})
        if twice:                                            # 1 and "1", or a list that repeats an id
            raise ValueError(f"assigned unit(s) {twice[:5]} are listed more than once: one assignment per unit")
        if isinstance(assigned, Mapping):
            manifest = {str(u): str(g) for u, g in assigned.items()}
        elif group_key:
            raise ValueError("with group_key, give assigned as {unit_id: group}")
        else:
            manifest = {str(u): "all" for u in assigned}
        if any("unit_id" not in r for r in rows):
            raise ValueError("with an assigned manifest every row needs a unit_id")
        outside = sorted({str(r["unit_id"]) for r in rows} - set(manifest))
        if outside:
            raise ValueError(f"unit(s) {outside[:5]} are not in the assigned manifest")
        wrong = sorted(str(r["unit_id"]) for r in rows
                       if (str(r.get(group_key, "all")) if group_key else "all") != manifest[str(r["unit_id"])])
        if wrong:
            raise ValueError(f"unit(s) {wrong[:5]} are in a different group than the assigned manifest says")
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(str(r.get(group_key, "all")) if group_key else "all", []).append(r)
    for g in set(manifest.values()) if manifest else ():
        groups.setdefault(g, [])
    gen = np.random.default_rng(seed)
    out: dict[str, Any] = {}
    for name in sorted(groups):
        rs = groups[name]
        assigned_here = {u for u, g in manifest.items() if g == name} if manifest is not None else None
        cohort = _cohort(name, rs, assigned_here, failure_policy)
        scored = [r for r in rs if r.get("status", "scored") == "scored"]
        if not scored:
            out[name] = {"n_units": 0, **cohort}
            continue
        splits = [split_of_row(r) for r in scored]
        res = {**pool_splits(splits, unit_scale), **cohort}
        if n_boot > 0 and cluster_key:
            cl = [str(r[cluster_key]) for r in scored]
            res["n_clusters"] = len(set(cl))
            info: dict[str, int] = {}
            res["pooled_share_ci95"] = cluster_bootstrap(functools.partial(_shares_of, splits), cl, n_boot, gen,
                                                         info=info)
            n_undef = info["undefined_draws"]
            if n_undef:
                res["pooled_share_ci95_undefined_draws"] = n_undef
            point = ("no point estimate exists either (pooled_share is null)" if res["pooled_share"] is None else
                     "the point estimate pooled_share stands without an interval")
            if res["pooled_share_ci95"] == NOT_ESTIMABLE:
                res["pooled_share_ci95_reason"] = (
                    f"fewer than 2 distinct clusters: no between-cluster information; {point}"
                    if res["n_clusters"] < 2 else
                    f"all {n_boot} draws had zero error in every drawn cluster; {point}")
            else:
                gone = res.get("pooled_share_undefined", {})
                if gone:                # an interval from the draws that happen to be complete would describe a
                    for k in gone:      # quantity that does not exist for this cohort
                        res["pooled_share_ci95"].pop(k, None)
                    res["pooled_share_ci95_component_reason"] = {
                        k: f"the full-cohort value is undefined ({why}); no interval is given" for k, why in gone.items()}
            if res["pooled_share_ci95"] != NOT_ESTIMABLE and n_undef:
                res["pooled_share_ci95_reason"] = (
                    f"{n_undef} of {n_boot} draws had zero error in every drawn cluster and no share; the interval "
                    "is over the other draws and is conditional on a defined share")
        out[name] = res
    return out


def _jsonable(obj: object) -> object:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _read_manifest(path: Path, grouped: bool) -> list[str] | dict[str, str]:
    """`unit_id` or `unit_id group` per line; with --group-key every line must name its group."""
    pairs = []
    for n, line in enumerate(path.read_text().splitlines(), 1):
        f = line.split("#", 1)[0].split()
        if not f:
            continue
        if len(f) != (2 if grouped else 1):
            raise ValueError(f"{path.name} line {n}: expected 'unit_id group' with --group-key, else 'unit_id'")
        pairs.append((n, f))
    first: dict[str, int] = {}
    for n, f in pairs:                                       # one line per assigned unit: a repeat is refused
        if f[0] in first:
            raise ValueError(f"{path.name} line {n}: unit {f[0]!r} is already assigned on line {first[f[0]]}")
        first[f[0]] = n
    return {f[0]: f[1] for _, f in pairs} if grouped else [f[0] for _, f in pairs]


def main(argv: Sequence[str] | None = None) -> int:
    """Console entry point: print the pooled summary as JSON; exit 2 on a usage or input error."""
    ap = argparse.ArgumentParser(prog="alongray-pool", description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("rows", type=Path, help="JSONL rows, or a JSON file with --rows-key")
    ap.add_argument("--rows-key", help="read a JSON object and take this key's list")
    ap.add_argument("--group-key")
    ap.add_argument("--cluster-key")
    ap.add_argument("--boot", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--unit-scale", type=float, default=1e3, help="length multiplier (1e3: metres -> mm)")
    ap.add_argument("--assigned", type=Path,
                    help="manifest of assigned units, one 'unit_id [group]' per line ('#' starts a comment); an "
                         "assigned unit with no row counts as missing")
    ap.add_argument("--failure-policy", choices=FAILURE_POLICIES, default="report",
                    help="report: pool the scored units and print the denominators; refuse: exit 2 when a group has "
                         "a failed or missing unit")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    if a.boot > 0 and not a.cluster_key:
        ap.error("--boot needs --cluster-key (the bootstrap resamples clusters)")
    try:
        text = a.rows.read_text()
        if a.rows_key:
            doc = strict_json.loads(text, str(a.rows))
            if not isinstance(doc, dict):
                ap.error(f"--rows-key needs a JSON object at the top level, got {type(doc).__name__}")
            rows = doc[a.rows_key]
        else:
            rows = [strict_json.loads(x, f"{a.rows} line {ln}") for ln, x in enumerate(text.splitlines(), 1)
                    if x.strip()]
        if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
            raise ValueError("rows must be a list of JSON objects")
        assigned = _read_manifest(a.assigned, bool(a.group_key)) if a.assigned is not None else None
        res = _jsonable(pool_rows(rows, a.group_key, a.cluster_key, a.boot, a.seed, a.unit_scale, assigned=assigned,
                                  failure_policy=a.failure_policy))
    except (ValueError, KeyError, TypeError, OSError) as e:
        print(f"alongray-pool: error: {e}", file=sys.stderr)
        return 2
    if a.out is not None:
        a.out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
