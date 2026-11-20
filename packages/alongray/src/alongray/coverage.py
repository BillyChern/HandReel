# Copyright 2026 The alongray authors
# SPDX-License-Identifier: Apache-2.0
"""``alongray-coverage PROTOCOL A B``: a coverage and protocol report for two output folders, before any comparison.

It answers, for two methods' existing outputs on one frozen protocol (formats: :mod:`alongray.coverage_io`):

- the flow of units and frames, stage by stage with the reason for every drop: assigned, annotated (scorable) in the
  protocol, declared in the manifest, delivered, declared source identity (the manifest's sequence, camera and
  object equal the protocol's; the physical content of a file is not checked), on the protocol's frame grid,
  delivered on annotated frames, and the exact common (unit, frame) IDs of the two methods;
- the conventions (units, frame, camera axes, frame-index base, clock, object origin, scale, symmetry, registration,
  alignment-fit support, aggregation) side by side, converted where declared, flagged where unknown or mismatched;
- empirical convention checks on the common frames (:mod:`alongray.conventions`). Their numbers between the two
  methods are labelled screening statistics, not scores. With ``--reference``, each method is screened against the
  reference on the same common IDs and only PASS / FLAGGED and the flag codes are reported, never a value: a
  per-method number against ground truth would be an accuracy reading. The reference's own flow and flags are
  reported too;
- informational lines (``info``; they never change the verdict): DELIVERY_DIFFERS for any non-delivered status or
  any difference in delivery share, NONFINITE_ROWS for rows left out as non-finite;
- a REFUSAL when modalities or provenance are incompatible, NOT_PAIRABLE for reported-only numbers or an empty
  intersection, COUNTS_ONLY for published counts without IDs;
- never a speed ratio: timing fields in the manifests are listed and ignored.

It scores nothing and changes no scorer. Optional per-unit ``scores`` that the methods supply are only re-aggregated
on the common units, labelled frame-weighted or sequence-weighted, with the number of resampling clusters. Delivery
over all assigned units is reported apart from anything on the common IDs, and nothing is inferred about the quality
of missing outputs: the direction of any bias they cause is unknown.

Exit status: 0 COMPARABLE, 1 COMPARABLE_WITH_FLAGS, 2 malformed input, usage error or any internal error (never 1),
3 NOT_PAIRABLE or COUNTS_ONLY, 4 REFUSED.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from alongray.conventions import UNIT_TO_M, Track, to_canonical
from alongray.coverage_checks import (
    SCREEN_LABEL,
    delivery_info,
    grid_offset_check,
    nonfinite_info,
    pair,
    pair_shift_check,
    reference_checks,
)
from alongray.coverage_io import (
    CONVENTION_KEYS,
    CONVERTED_KEYS,
    Method,
    Protocol,
    UnitSpec,
    load_method,
    load_protocol,
    load_unit_arrays,
)
from alongray.coverage_report import aggregate_scores, counts_only_block, format_summary, reported_overlap

__all__ = ["EXIT", "compare", "compare_folders", "main"]

EXIT = {"COMPARABLE": 0, "COMPARABLE_WITH_FLAGS": 1, "NOT_PAIRABLE": 3, "COUNTS_ONLY": 3, "REFUSED": 4}
ASYMMETRY_TOL = 0.1
_KNOWN = {"length_unit": set(UNIT_TO_M), "frame": {"camera", "world"}, "camera_axes": {"opencv", "opengl"},
          "frame_index_base": {0, 1}}
NO_BIAS_CLAIM = ("Missing outputs are counted, never imputed; nothing is inferred about their quality and the "
                 "direction of any bias they cause is unknown.")
SPEED = ("NOT_INFERRED: output files carry no matched timing protocol (hardware, boundaries, cold/warm state, "
         "population), so no speed ratio is computed from them.")


RESERVED_NAMES = ("protocol", "status")       # column names of the report's convention table


def _name_refusals(ms: Sequence[Method]) -> list[str]:
    """The report keys its tables by method name. Two inputs with one name (two folders with the same basename, or
    equal `method` fields) would overwrite each other's inputs, counts and conventions, and a name that equals a
    column of the convention table would overwrite that column. Both are refused before any comparison."""
    names = [m.name for m in ms]
    out = [f"method name {n!r} is given by {names.count(n)} inputs: give each manifest a distinct `method`"
           for n in sorted({n for n in names if names.count(n) > 1})]
    out += [f"method name {n!r} is a column of the report: give the manifest another `method`"
            for n in sorted(set(names) & set(RESERVED_NAMES))]
    return out


def _refusals(p: Protocol, ms: Sequence[Method]) -> list[str]:
    out = []
    for m in ms:
        for k in ("dataset", "split"):
            if m.provenance.get(k) is None:
                out.append(f"{m.name}: provenance.{k} undeclared")
            elif m.provenance.get(k) != p.provenance.get(k):
                out.append(f"{m.name}: provenance.{k} {m.provenance.get(k)!r} != protocol {p.provenance.get(k)!r}")
        g = m.provenance.get("gt_release")
        if g is not None and p.provenance.get("gt_release") is not None and g != p.provenance["gt_release"]:
            out.append(f"{m.name}: gt_release {g!r} != protocol {p.provenance['gt_release']!r}")
        if m.modality.get("input") != p.modality.get("input"):
            out.append(f"{m.name}: input modality {m.modality.get('input')!r} != protocol "
                       f"{p.modality.get('input')!r}")
        missing = set(p.modality.get("compare", [])) - set(m.modality.get("outputs", []))
        if m.outputs == "per_frame" and missing:
            out.append(f"{m.name}: does not output {sorted(missing)}")
    return out


def _duplicate_keys(p: Protocol) -> tuple[dict[str, int], list[str]]:
    keyed = Counter((u.sequence, u.camera, u.object, f) for u in p.units for f in u.frames)
    physical = {(s, c, f) for (s, c, _, f) in keyed}
    dup = {k: n for k, n in keyed.items() if n > 1}
    info = {"keyed_frames": sum(keyed.values()), "distinct_keys": len(keyed), "distinct_physical_frames": len(physical),
            "duplicated_keys": len(dup), "extra_copies": sum(n - 1 for n in dup.values())}
    if dup and p.overlap_policy == "refuse":
        return info, [(f"protocol: {len(dup)} (sequence, camera, object, frame) keys occur in more than one unit "
                       "(overlapping windows); declare overlap_policy 'keep_with_multiplicity' or remove the overlap")]
    return info, []


def _conventions(p: Protocol, a: Method, b: Method) -> tuple[dict[str, Any], list[str]]:
    table, flags = {}, []
    rot = "object_rotation" in p.modality.get("compare", [])
    for k in CONVENTION_KEYS:
        pv, av, bv = p.conventions.get(k), a.conventions.get(k), b.conventions.get(k)
        if k == "symmetry" and not rot:
            st = "NOT_APPLICABLE"
        elif av is None or bv is None or (k in _KNOWN and not {av, bv} <= _KNOWN[k]):
            st = "UNKNOWN"
        elif k in CONVERTED_KEYS:
            st = "MATCH" if av == bv == pv else "CONVERTED"
        else:
            st = "MATCH" if av == bv and pv in (None, av) else "MISMATCH"
        table[k] = {"protocol": pv, a.name: av, b.name: bv, "status": st}
        if st in ("UNKNOWN", "MISMATCH"):
            flags.append(f"CONVENTION_{st}: {k} (protocol {pv!r}, {a.name} {av!r}, {b.name} {bv!r})")
    return table, flags


def _tracks(p: Protocol, m: Method) -> dict[str, Track]:
    out = {}
    compared = [k for k in p.modality.get("compare", []) if k in ("object_position", "object_rotation")]
    for u in p.units:
        rec = m.units.get(u.unit_id)
        if rec is None or rec.get("status") != "delivered":
            continue
        arr = load_unit_arrays(m, u.unit_id)
        if arr is None:
            out[u.unit_id] = Track(np.zeros(0, np.int64), None, None, ("delivered without a unit file",))
            continue
        t = to_canonical(arr, m.conventions, int(p.conventions.get("frame_index_base", 0)), p.frame_rate)
        missing = [k for k in compared if k not in arr]
        if missing:  # a delivered unit without a compared array is never silently position- or rotation-free
            t = dataclasses.replace(t, notes=(*t.notes, f"COMPARED_ARRAY_MISSING: {', '.join(missing)}"))
        out[u.unit_id] = t
    return out


def _flow(p: Protocol, m: Method, tracks: dict[str, Track]) -> tuple[list[dict[str, Any]], dict[str, set[int]],
                                                                     list[str]]:
    """Stage-by-stage flow with drop reasons; returns stages, delivered annotated frames per unit, flags."""
    stages: list[dict[str, Any]] = []
    flags: list[str] = []

    def stage(name: str, units: list[UnitSpec], dropped: Counter, frames: int, what: str) -> None:
        stages.append({"stage": name, "units": len(units), "frames": frames, "frames_are": what,
                       "dropped_units": dict(sorted(dropped.items()))})

    cur = list(p.units)
    stage("assigned", cur, Counter(), sum(len(u.frames) for u in cur), "assigned keyed frames")
    keep = [u for u in cur if u.scorable]
    stage("annotated", keep, Counter({"no annotated frame in the protocol": len(cur) - len(keep)}),
          sum(len(u.scorable) for u in keep), "annotated keyed frames of these units")
    cur, keep, drop = keep, [], Counter()
    for u in cur:
        rec = m.units.get(u.unit_id)
        if rec is None:
            drop["absent from the manifest (no status)"] += 1
        elif rec["status"] != "delivered":
            drop[f"{rec['status']}: {rec.get('reason', 'no reason given')}"] += 1
        else:
            keep.append(u)
    if any(k.startswith("absent") for k in drop):
        flags.append(f"UNDECLARED_UNITS: {m.name}: {drop['absent from the manifest (no status)']} assigned units have "
                     "no status (an inner join would drop them silently)")
    stage("delivered", keep, drop, sum(len(u.scorable) for u in keep), "annotated keyed frames of these units")
    cur, keep, drop = keep, [], Counter()
    for u in cur:
        rec = m.units[u.unit_id]
        bad = [k for k in ("sequence", "camera", "object") if rec.get(k) != getattr(u, k)]
        if bad:
            drop[f"declared source identity differs or is missing: {', '.join(bad)}"] += 1
        else:
            keep.append(u)
    if drop:
        flags.append(f"SOURCE_MISMATCH: {m.name}: {sum(drop.values())} delivered units do not declare the protocol's "
                     "sequence, camera and object (declared source identity; an ID match as text is not enough, and "
                     "the physical content of a file is not checked)")
    stage("declared_source_identity", keep, drop, sum(len(u.scorable) for u in keep),
          "annotated keyed frames of these units")
    cur, keep, drop, got = keep, [], Counter(), {}
    off, nonfinite, notes = 0, 0, Counter()
    for u in cur:
        t = tracks[u.unit_id]
        grid = set(u.frames)
        ids = {int(f) for f in t.frame_ids}
        off += len(ids - grid)
        nonfinite += t.nonfinite
        notes.update(n.split(":")[0] for n in t.notes)
        if ids & grid:
            keep.append(u)
            got[u.unit_id] = ids & grid
        else:
            drop["delivered, no finite frame on the protocol grid"] += 1
    stage("on_grid", keep, drop, sum(len(v) for v in got.values()), "delivered keyed frames on the grid")
    cur, keep, drop = keep, [], Counter()
    for u in cur:
        got[u.unit_id] &= u.scorable
        (keep.append(u) if got[u.unit_id] else drop.update(["delivered frames all unannotated"]))
    got = {k: v for k, v in got.items() if v}
    stage("delivered_annotated", keep, drop, sum(len(v) for v in got.values()), "delivered annotated keyed frames")
    stages[-1].update(off_grid_frames=off, nonfinite_frames=nonfinite)
    for n, c in sorted(notes.items()):
        if n.isupper() or n.startswith(("CAMERA", "CLOCK", "ROTATION")):
            flags.append(f"{n}: {m.name}: {c} units")
    flags += grid_offset_check(p, m, tracks)
    return stages, got, flags


def _common(got_a: dict[str, set[int]], got_b: dict[str, set[int]]) -> dict[str, list[int]]:
    return {u: sorted(got_a[u] & got_b[u]) for u in sorted(set(got_a) & set(got_b)) if got_a[u] & got_b[u]}


def compare(p: Protocol, a: Method, b: Method, reference: Method | None = None, include_ids: bool = False,
            n_boot: int = 0, seed: int = 0) -> dict[str, Any]:
    """The coverage and protocol report (a JSON-ready dict); see the module docstring."""
    named = _name_refusals([a, b] + ([reference] if reference else []))
    if named:
        # Nothing keyed by method name is built for ambiguous names: a second method of the same name would
        # overwrite the first one's inputs, timing fields and conventions. The names are given by position.
        return {"protocol": p.name, "methods": [a.name, b.name], "cluster": p.cluster, "verdict": "REFUSED",
                "reasons": named, "flags": [], "info": []}
    rep: dict[str, Any] = {"protocol": p.name, "methods": [a.name, b.name], "cluster": p.cluster,
                           "speed": {"status": SPEED, "ignored_timing_fields": {a.name: list(a.timing_fields),
                                                                                 b.name: list(b.timing_fields)}},
                           "missing_outputs": NO_BIAS_CLAIM}
    kinds = {a.outputs, b.outputs}
    if "per_frame" in kinds and not p.units:
        raise ValueError("per-frame outputs need a protocol with units; this protocol has published_counts only")
    refused = _refusals(p, [a, b] + ([reference] if reference else []))
    dup, dup_refusal = _duplicate_keys(p) if p.units else ({}, [])
    refused += dup_refusal
    rep["protocol_keys"] = dup
    conv, flags = _conventions(p, a, b)
    rep["conventions"] = conv
    rep["inputs"] = {k: {a.name: a.inputs.get(k), b.name: b.inputs.get(k)} for k in sorted({*a.inputs, *b.inputs})}
    flags += [f"INPUT_DIFFERS: {k}" for k, v in rep["inputs"].items() if v[a.name] != v[b.name]]
    info: list[str] = []
    if refused:
        verdict, why = "REFUSED", refused
    elif "reported_only" in kinds:
        verdict, why = "NOT_PAIRABLE", ["reported-only, not pairable: no per-frame outputs exist for "
                                        + ", ".join(m.name for m in (a, b) if m.outputs == "reported_only")]
    elif kinds == {"counts_only"}:
        verdict, why = "COUNTS_ONLY", [("published counts only: the intersection is taken from the record and "
                                        "checked for consistency, not recomputed from IDs")]
        rep["counts"] = counts_only_block(p, a, b)
        if a.counts["delivered"] != b.counts["delivered"]:
            info.append(f"DELIVERY_DIFFERS: {a.name} delivers {a.counts['delivered']} and {b.name} "
                        f"{b.counts['delivered']} units (published counts)")
    elif "counts_only" in kinds:
        verdict, why = "NOT_PAIRABLE", ["one side has counts only: no common IDs can be formed"]
    else:
        verdict, why = None, []
    if verdict in (None, "NOT_PAIRABLE") and "per_frame" in kinds and p.units:
        per = {}
        for m in (a, b):
            if m.outputs != "per_frame":
                continue
            t = _tracks(p, m)
            stages, got, fl = _flow(p, m, t)
            per[m.name] = (t, got)
            flags += fl
            info += nonfinite_info(m.name, stages)
            rep.setdefault("flow", {})[m.name] = stages
        if len(per) == 1:
            info += delivery_info(p, [m for m in (a, b) if m.outputs == "per_frame"])
        if len(per) == 1 and "reported_only" in kinds:
            other = a if a.outputs == "reported_only" else b
            rep["reported_cohort_overlap"] = reported_overlap(p, other, next(iter(per.values()))[1])
    if verdict is None:
        (ta, ga), (tb, gb) = per[a.name], per[b.name]
        common = _common(ga, gb)
        spec = {u.unit_id: u for u in p.units}
        nf = sum(len(v) for v in common.values())
        rep["common_ids"] = {
            "units": len(common), "frames": nf,
            "sequences": len({spec[u].sequence for u in common}), "objects": len({spec[u].object for u in common}),
            "cameras": len({(spec[u].sequence, spec[u].camera) for u in common}),
            "clusters": len({getattr(spec[u], p.cluster) for u in common}),
            "distinct_physical_frames": len({(spec[u].sequence, spec[u].camera, f) for u, v in common.items()
                                             for f in v}),
            f"{a.name}_only_frames": sum(len(v) for v in ga.values()) - nf,
            f"{b.name}_only_frames": sum(len(v) for v in gb.values()) - nf}
        ann = sum(len(u.scorable) for u in p.units)
        sh = {m: sum(len(v) for v in g.values()) / ann for m, g in ((a.name, ga), (b.name, gb))}
        rep["delivery_share_of_annotated_frames"] = sh
        info += delivery_info(p, [a, b], sh)
        if abs(sh[a.name] - sh[b.name]) > ASYMMETRY_TOL:
            flags.append(f"COVERAGE_ASYMMETRIC: {a.name} delivers {sh[a.name]:.2f} and {b.name} {sh[b.name]:.2f} of "
                         "the annotated frames; common-ID numbers describe the overlap only")
        if include_ids:
            rep["common_ids"]["ids"] = {u: v for u, v in common.items()}
        summ, fl = pair(p, ta, tb, common)
        rep["checks"] = {f"{a.name}~{b.name}": {**summ, "label": SCREEN_LABEL}}
        flags += [f"{x.split(':')[0]}: {a.name}~{b.name}:{x.split(':', 1)[1]}" for x in fl]
        flags += pair_shift_check(a.name, b.name, ta, tb, common)
        if reference is not None and nf:
            tr = _tracks(p, reference)
            stages, gr, rfl = _flow(p, reference, tr)
            ann_r = sum(len(v) for v in gr.values()) / ann
            rep["reference"] = {"name": reference.name, "flow": stages, "delivery_share_of_annotated_frames": ann_r,
                                "flags": rfl}
            flags += rfl
            info += nonfinite_info(reference.name, stages)
            rep["reference_checks"], fl = reference_checks(p, reference.name, tr, gr, {a.name: ta, b.name: tb},
                                                           common)
            flags += fl
        agg, fl = aggregate_scores(p, a, b, common, n_boot, seed)
        if agg:
            rep["aggregates"] = agg
        flags += fl
        verdict = "NOT_PAIRABLE" if nf == 0 else ("COMPARABLE_WITH_FLAGS" if flags else "COMPARABLE")
        why = ["no common (unit, frame) IDs"] if nf == 0 else []
    rep["matching"] = _matching(rep, p, a, b, flags)
    rep.update(verdict=verdict, reasons=why, flags=flags, info=info)
    return rep


def _matching(rep: dict[str, Any], p: Protocol, a: Method, b: Method, flags: list[str]) -> dict[str, str]:
    """Item-by-item: do the two outputs share the source frame, GT, clock, units, camera, object, symmetry and
    alignment-fit support? VERIFIED, MISMATCH or UNVERIFIED (never assumed)."""
    conv = rep["conventions"]

    def st(*keys: str) -> str:
        s = {conv[k]["status"] for k in keys}
        return "MISMATCH" if "MISMATCH" in s else "UNVERIFIED" if "UNKNOWN" in s else "VERIFIED"

    gts = {m.provenance.get("gt_release") for m in (a, b)} | {p.provenance.get("gt_release")}
    has = lambda code: any(f.startswith(code) for f in flags)
    both_flows = len(rep.get("flow", {})) == 2
    return {"source_frame": "MISMATCH" if has("SOURCE_MISMATCH") or has("FRAME_ID_OFFSET") else
            ("VERIFIED" if both_flows else "UNVERIFIED"),
            "ground_truth": "UNVERIFIED" if None in gts else ("VERIFIED" if len(gts) == 1 else "MISMATCH"),
            "clock": "MISMATCH" if has("CLOCK_MISMATCH") else st("clock", "frame_index_base"),
            "units": "MISMATCH" if has("UNIT_OR_SCALE") else st("length_unit", "scale"),
            "camera": "MISMATCH" if has("CAMERA") or has("FRAME_OR_TRANSFORM") else st("frame", "camera_axes"),
            "object": "MISMATCH" if has("ORIGIN_OFFSET") or has("ROTATION_DISAGREEMENT") else st("object_origin"),
            "symmetry": "MISMATCH" if has("SYMMETRY") else {"MATCH": "VERIFIED", "UNKNOWN": "UNVERIFIED"}.get(
                conv["symmetry"]["status"], conv["symmetry"]["status"]),
            "alignment_fit_support": st("registration", "alignment_fit_support")}


def compare_folders(protocol: str | Path, a: str | Path, b: str | Path, reference: str | Path | None = None,
                    **kw: Any) -> dict[str, Any]:
    return compare(load_protocol(protocol), load_method(a), load_method(b),
                   None if reference is None else load_method(reference), **kw)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="alongray-coverage", description=__doc__.split("\n\n")[0])
    ap.add_argument("protocol", nargs="?")
    ap.add_argument("a", nargs="?")
    ap.add_argument("b", nargs="?")
    ap.add_argument("--reference", help="a ground-truth folder in the method format: pass/fail screens of each "
                    "method on the common IDs (no value is reported) and the reference's own coverage")
    ap.add_argument("--ids", action="store_true", help="include common (unit, frame) IDs (default: counts only)")
    ap.add_argument("--boot", type=int, default=0, help="cluster-bootstrap draws for supplied-score aggregates")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", help="write the JSON report here")
    ap.add_argument("--summary", action="store_true", help="print a human-readable summary instead of JSON")
    ap.add_argument("--write-known-answers", metavar="DIR", help="write the synthetic known-answer cases and exit")
    args = ap.parse_args(argv)
    try:
        if args.write_known_answers:
            from alongray.coverage_fixtures import write_known_answers
            print(json.dumps(write_known_answers(args.write_known_answers), indent=1))
            return 0
        if not (args.protocol and args.a and args.b) or args.boot < 0:
            ap.print_usage(sys.stderr)
            return 2
        rep = compare_folders(args.protocol, args.a, args.b, args.reference, include_ids=args.ids,
                              n_boot=args.boot, seed=args.seed)
        text = json.dumps(rep, indent=1, default=lambda x: None if isinstance(x, float) and math.isnan(x) else str(x))
        if args.out:
            Path(args.out).write_text(text + "\n")
        out = format_summary(rep) if args.summary else text
    except (ValueError, OSError) as e:
        print(f"alongray-coverage: {e}", file=sys.stderr)
        return 2
    except Exception as e:  # noqa: BLE001 (any other failure is an input or internal error: exit 2, never 1)
        print(f"alongray-coverage: internal error, no verdict ({type(e).__name__}: {e})", file=sys.stderr)
        return 2
    print(out)
    return EXIT[rep["verdict"]]


if __name__ == "__main__":
    raise SystemExit(main())
