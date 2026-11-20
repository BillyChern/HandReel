# Copyright 2026 The alongray authors
# SPDX-License-Identifier: Apache-2.0
"""Report blocks of :mod:`alongray.coverage`: supplied-score aggregates, counts-only checks, reported-cohort overlap
and the human-readable summary. Nothing here scores an output."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from alongray.coverage_io import Method, Protocol
from alongray.pool import cluster_bootstrap

__all__ = ["aggregate_scores", "counts_only_block", "format_summary", "reported_overlap"]


def _means(vals: list[float], wts: list[int], seqs: list[str], idx: list[int]) -> dict[str, float]:
    """Both weightings on the rows `idx` of the units `vals` (a bootstrap draw repeats rows).

    The frame-weighted mean counts a row as often as it was drawn. The sequence-weighted mean must do the same for
    a sequence, and a draw resamples clusters, which need not be sequences: a sequence can lie inside one cluster or
    be shared by several (two objects filmed in one sequence). So each sequence counts by the drawn share of its
    frames, `sum(drawn frames of s) / sum(frames of s over all units)`, and its value is the frame-weighted mean of
    its drawn rows. On the full data every share is 1 and this is the plain mean over sequences; a sequence whose
    one cluster was drawn k times counts k times; a sequence none of whose rows was drawn does not count.
    """
    v, w = np.array([vals[i] for i in idx]), np.array([wts[i] for i in idx], float)
    total: dict[str, float] = {}
    for q, x in zip(seqs, wts, strict=True):
        total[q] = total.get(q, 0.0) + float(x)
    drawn: dict[str, list[int]] = {}
    for j, i in enumerate(idx):
        drawn.setdefault(seqs[i], []).append(j)
    share = np.array([w[g].sum() / total[q] for q, g in drawn.items()])
    per = np.array([(v[g] * w[g]).sum() / w[g].sum() for g in drawn.values()])
    return {"frame_weighted": float((v * w).sum() / w.sum()),
            "sequence_weighted": float((share * per).sum() / share.sum())}


def aggregate_scores(p: Protocol, a: Method, b: Method, common: Mapping[str, list[int]], n_boot: int = 0,
                     seed: int = 0) -> tuple[dict[str, Any], list[str]]:
    """Re-aggregate the methods' OWN per-unit scores on the common units, both weightings labelled.

    For each metric both methods are aggregated on THE SAME units: the common units for which both supplied a
    finite value and a positive ``n_frames``. A metric key that one row omits is a missing value of that unit; the
    metric is still aggregated on the units that have it. A common unit with a score row from one method only is
    excluded by name like any other missing value. When no unit qualifies, the metric is reported as
    unavailable with its excluded units. A unit with a missing or non-finite value from either method is left out
    of both aggregates,
    counted in ``n_excluded_not_finite_in_both``, listed in ``excluded_units`` and flagged, so that two aggregates
    with equal ``n_units`` never rest on different units.

    Each unit is weighted by the method's declared ``n_frames``. A unit whose ``n_frames`` differs from its number of
    common frames was scored on other frames than the common IDs; it is counted and flagged, never corrected. With
    ``n_boot > 0`` a percentile interval resamples the protocol's cluster (sequence or object); with fewer than two
    distinct clusters ``ci95_cluster_bootstrap`` is the string ``NOT_ESTIMABLE``.
    """
    if not (a.scores and b.scores):
        return {}, []
    spec = {u.unit_id: u for u in p.units}
    flags, out = [], {}
    for m in (a, b):
        off = [u for u in common if u in m.scores and m.scores[u].get("n_frames") != len(common[u])]
        miss = [u for u in common if u not in m.scores]
        if off:
            flags.append(f"SCORES_NOT_ON_COMMON_FRAMES: {m.name}: {len(off)} common units were scored on a different "
                         "number of frames than their common IDs")
        if miss:
            flags.append(f"SCORES_MISSING: {m.name}: {len(miss)} common units have no supplied score")
    # EVERY common unit is a candidate, whether or not both methods supplied a score row for it: a unit with a row
    # from one method only is excluded BY NAME, never dropped silently, and a metric that only such a row carries
    # is still reported (as unavailable). Keys are discovered across every supplied common-unit row; a key absent
    # from a row is a missing value of that unit.
    units = list(common)
    keys = set().union(*[set(m.scores[u]) - {"n_frames"} for m in (a, b) for u in units if u in m.scores])

    def finite(m: Method, u: str, k: str) -> bool:
        v = m.scores[u].get(k) if u in m.scores else None
        return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)

    def weighted(m: Method, u: str) -> bool:           # a unit of zero frames has no weight to contribute
        return u in m.scores and int(m.scores[u]["n_frames"]) > 0

    for k in sorted(keys):
        ok = [u for u in units if finite(a, u, k) and finite(b, u, k) and weighted(a, u) and weighted(b, u)]
        excluded = sorted(set(units) - set(ok))
        if excluded:
            flags.append(f"SCORES_NOT_FINITE_FOR_BOTH: {k}: {len(excluded)} common units lack a score row, a finite "
                         "value or a positive frame count from at least one method and are excluded from both "
                         "aggregates")
        if not ok:
            # no estimate exists; the excluded identities are still reported, for both methods
            for m in (a, b):
                out.setdefault(k, {})[m.name] = {
                    "frame_weighted": None, "sequence_weighted": None, "n_units": 0,
                    "n_excluded_not_finite_in_both": len(excluded), "excluded_units": excluded,
                    "unavailable": "no common unit has a finite value and a positive frame count from both methods"}
            continue
        for m in (a, b):
            vals = [float(m.scores[u][k]) for u in ok]
            wts = [int(m.scores[u]["n_frames"]) for u in ok]
            seqs = [spec[u].sequence for u in ok]
            clusters = [getattr(spec[u], p.cluster) for u in ok]
            row: dict[str, Any] = dict(_means(vals, wts, seqs, list(range(len(ok)))))
            row.update(n_units=len(ok), n_excluded_not_finite_in_both=len(excluded), excluded_units=excluded,
                       n_sequences=len(set(seqs)), cluster=p.cluster,
                       n_clusters=len(set(clusters)), declared_aggregation=m.conventions.get("aggregation"),
                       protocol_aggregation=p.conventions.get("aggregation"))
            if n_boot > 0:
                ci = cluster_bootstrap(lambda idx, v=vals, w=wts, q=seqs: _means(v, w, q, idx), clusters, n_boot, seed)
                row["ci95_cluster_bootstrap"] = ci if isinstance(ci, str) else {w: list(v) for w, v in ci.items()}
                if isinstance(ci, str):
                    row["ci95_cluster_bootstrap_reason"] = ("fewer than 2 distinct clusters: no between-cluster "
                                                            "information")
            out.setdefault(k, {})[m.name] = row
    return out, flags


def counts_only_block(p: Protocol, a: Method, b: Method) -> dict[str, Any]:
    """Consistency checks on published counts and the derived only-A / only-B / neither counts.

    Raises:
        ValueError: missing or inconsistent counts.
    """
    c = dict(p.published_counts or {})
    for k in ("analysed", "both"):
        if not isinstance(c.get(k), int):
            raise ValueError(f"published_counts.{k} is required for a counts-only comparison")  # noqa: TRY004
    da, db, n, both = a.counts["delivered"], b.counts["delivered"], c["analysed"], c["both"]
    checks = {}
    if all(isinstance(c.get(k), int) for k in ("assigned", "run", "not_run")):
        checks["run + not_run == assigned"] = c["run"] + c["not_run"] == c["assigned"]
    if isinstance(c.get("run"), int):
        checks["analysed == run - removed"] = n == c["run"] - int(c.get("removed", 0))
    checks["delivered <= analysed (each)"] = da <= n and db <= n
    checks["max(0, dA + dB - analysed) <= both <= min(dA, dB)"] = max(0, da + db - n) <= both <= min(da, db)
    bad = [k for k, v in checks.items() if not v]
    if bad:
        raise ValueError(f"inconsistent published counts: {bad}")
    return {"published": c, "delivered": {a.name: da, b.name: db}, "checks": checks,
            "derived": {f"{a.name}_only": da - both, f"{b.name}_only": db - both, "neither": n - da - db + both}}


def reported_overlap(p: Protocol, reported: Method, got: Mapping[str, set[int]]) -> dict[str, Any]:
    """How many of the other method's delivered annotated frames lie in the reported method's declared cohort."""
    ev = reported.reported.get("evaluated_frames")
    base = {"source": reported.reported.get("source"), "cohort": reported.reported.get("cohort"),
            "note": "reported numbers are aggregates over the reported cohort; they are not values on these frames"}
    if not isinstance(ev, dict):
        return {**base, "status": "no evaluated frame list declared"}
    spec = {u.unit_id: u for u in p.units}
    hit = {u: {f for f in fs if f in set(ev.get(spec[u].sequence, []))} for u, fs in got.items()}
    hit = {u: v for u, v in hit.items() if v}
    return {**base, "status": "declared cohort frames counted", "units": len(hit),
            "frames": sum(len(v) for v in hit.values()), "sequences": len({spec[u].sequence for u in hit}),
            "cohort_frames_declared": sum(len(v) for v in ev.values())}


def format_summary(rep: Mapping[str, Any]) -> str:
    """A human-readable rendering of a report."""
    out = [f"protocol {rep['protocol']}: {' vs '.join(rep['methods'])}", f"VERDICT {rep['verdict']}"]
    out += [f"  reason: {r}" for r in rep["reasons"]]
    for m, stages in rep.get("flow", {}).items():
        out.append(f"flow {m}:")
        for s in stages:
            drops = "; ".join(f"{k} {v}" for k, v in s["dropped_units"].items() if v)
            out.append(f"  {s['stage']:<20} units {s['units']:>5}  frames {s['frames']:>6} ({s['frames_are']})"
                       + (f"  dropped: {drops}" if drops else ""))
    if "common_ids" in rep:
        c = rep["common_ids"]
        out.append("common IDs: " + ", ".join(f"{k} {v}" for k, v in c.items() if k != "ids"))
    if "counts" in rep:
        out.append(f"counts: published {rep['counts']['published']}; delivered {rep['counts']['delivered']}; "
                   f"derived {rep['counts']['derived']}")
    if "reported_cohort_overlap" in rep:
        out.append(f"reported cohort: {rep['reported_cohort_overlap']}")
    out.append("conventions:")
    for k, v in rep["conventions"].items():
        out.append(f"  {k:<22} {v['status']:<15} " + ", ".join(f"{n}={v[n]!r}" for n in ["protocol", *rep['methods']]))
    for pair, chk in rep.get("checks", {}).items():
        out.append(f"check {pair} ({chk.get('label')}): "
                   + ", ".join(f"{k} {v}" for k, v in chk.items() if k not in ("label", "origin_offset_mm")))
    if "reference" in rep:
        r = rep["reference"]
        out.append(f"reference {r['name']}: delivers {r['delivery_share_of_annotated_frames']:.2f} of the annotated "
                   "frames; flow " + ", ".join(f"{s['stage']} {s['units']}/{s['frames']}" for s in r["flow"]))
    for pair, chk in rep.get("reference_checks", {}).items():
        out.append(f"reference check {pair} {chk['result']} on {chk['frames_checked']} common frames"
                   + (f" ({', '.join(chk['flags'])})" if chk["flags"] else "") + " [values withheld]")
    out.append("matching: " + ", ".join(f"{k} {v}" for k, v in rep["matching"].items()))
    out += [f"FLAG {f}" for f in rep["flags"]]
    out += [f"INFO {f}" for f in rep.get("info", [])]
    out += [f"speed: {rep['speed']['status']}", f"missing outputs: {rep['missing_outputs']}"]
    return "\n".join(out)
