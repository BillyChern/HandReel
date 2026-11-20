# Copyright 2026 The alongray authors
# SPDX-License-Identifier: Apache-2.0
"""Checks of :mod:`alongray.coverage` that read tracks: pairing on the common IDs, the 0- vs 1-based frame-ID
screens, the reference checks, and the informational delivery and non-finite-row lines.

Nothing here scores a method. The pairwise numbers between the two methods are screening statistics on the common
IDs, labelled as such. A reference check reports PASS or FLAGGED and the flag codes only: the values it computes are
withheld, because a method-against-ground-truth number on a method's frames is an accuracy reading.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from alongray.conventions import Track, pair_residuals, summarise_pair
from alongray.coverage_io import Method, Protocol

__all__ = ["OFFSET_AGREE", "REF_LABEL", "REF_SHIFT_GAIN", "SCREEN_LABEL", "delivery_info", "grid_offset_check",
           "nonfinite_info", "pair", "pair_shift_check", "reference_checks"]

SCREEN_LABEL = "screening statistic, not a score"
REF_LABEL = ("pass/fail screen of each method against the reference on the common IDs only; the values are withheld, "
             "so this is not a per-method accuracy")
SHIFTS = (-2, -1, 1, 2)
OFFSET_LOOSE = 0.95  # the original rule: under 0.95 as given, at least 0.95 after a shift
OFFSET_AGREE = 0.99  # the stride-1 rule: a shift reaching 0.99 and beating the ids as given
REF_SHIFT_GAIN = 0.5  # a shift is chosen against the reference only if it at least halves the discrepancy
REF_SHIFT_SUPPORT = 0.5  # ... and keeps at least half of the unshifted pairs
_STAT_OF = {"object_position": "median_scale_ratio", "object_rotation": "median_rotation_deg"}


def grid_offset_check(p: Protocol, m: Method, tracks: Mapping[str, Track]) -> list[str]:
    """Share of delivered ids on the protocol grid as given and after shifts of -2..2 frames."""
    grids = {u.unit_id: set(u.frames) for u in p.units}
    ids = [(uid, int(f)) for uid, t in tracks.items() for f in t.frame_ids]
    if not ids:
        return []
    share = {k: sum((f + k) in grids[uid] for uid, f in ids) / len(ids) for k in (0, *SHIFTS)}
    best = max(SHIFTS, key=lambda k: (share[k], -abs(k)))
    if share[0] < OFFSET_LOOSE <= share[best] or (share[best] >= OFFSET_AGREE and share[best] > share[0]):
        return [(f"FRAME_ID_OFFSET_SUSPECT: {m.name}: {share[0]:.2f} of delivered ids lie on the protocol grid as "
                 f"given, {share[best]:.2f} after shifting by {best:+d} (0- vs 1-based index?)")]
    return []


def pair(p: Protocol, ta: Mapping[str, Track], tb: Mapping[str, Track], common: Mapping[str, Sequence[int]]
         ) -> tuple[dict[str, Any], list[str]]:
    """Screening residuals of two tracks on the given (unit, frame) IDs, pooled; see :mod:`alongray.conventions`."""
    res: dict[str, Any] = {"offset_by_object": {}, "camera_offset_by_object": {}}
    spec = {u.unit_id: u for u in p.units}
    for uid, frames in common.items():
        a, b = ta[uid], tb[uid]
        ia = [int(np.flatnonzero(a.frame_ids == f)[0]) for f in frames]
        ib = [int(np.flatnonzero(b.frame_ids == f)[0]) for f in frames]
        r = pair_residuals(None if a.position is None else a.position[ia], None if b.position is None else
                           b.position[ib], None if a.rotation is None else a.rotation[ia],
                           None if b.rotation is None else b.rotation[ib], p.symmetry.get(spec[uid].object))
        for k, v in r.items():
            if k in ("offset_obj", "offset_cam"):
                key = "offset_by_object" if k == "offset_obj" else "camera_offset_by_object"
                res[key].setdefault(spec[uid].object, []).append(v)
            else:
                res.setdefault(k, []).append(v)
    return summarise_pair(res, "object_rotation" in p.modality.get("compare", []))


def _shift(label: str, how: str, tr: Mapping[str, Track], tm: Mapping[str, Track],
           common: Mapping[str, Sequence[int]]) -> list[str]:
    """The frame shift k at which ``tm``'s position on frame f best matches ``tr``'s on frame f + k. Flagged only
    when k != 0 at least halves the discrepancy and keeps half of the unshifted pairs."""
    disc, n = {}, {}
    for k in (0, *SHIFTS):
        diffs, norms = [], []
        for uid, frames in common.items():
            r, t = tr.get(uid), tm[uid]
            if r is None or r.position is None or t.position is None:
                continue
            ri = {int(f): i for i, f in enumerate(r.frame_ids)}
            ti = {int(f): i for i, f in enumerate(t.frame_ids)}
            for f in frames:
                if f + k in ri:
                    diffs.append(float(np.linalg.norm(t.position[ti[f]] - r.position[ri[f + k]])))
                    norms.append(float(np.linalg.norm(r.position[ri[f + k]])))
        n[k] = len(diffs)
        if diffs:
            disc[k] = float(np.median(diffs)) / max(float(np.median(norms)), 1e-300)
    if not n[0]:
        return []
    best = min((k for k in disc if n[k] >= REF_SHIFT_SUPPORT * n[0]), key=lambda k: (disc[k], abs(k)))
    if best != 0 and disc[best] < REF_SHIFT_GAIN * disc[0]:
        return [(f"FRAME_ID_OFFSET_SUSPECT: {label}: {how} the positions agree best at shift {best:+d} (0- vs "
                 "1-based index?; value withheld)")]
    return []


def pair_shift_check(a: str, b: str, ta: Mapping[str, Track], tb: Mapping[str, Track],
                     common: Mapping[str, Sequence[int]]) -> list[str]:
    """The between-method shift screen: needs no reference, sees shifts whose ids all lie on the grid."""
    return _shift(f"{a}~{b}", f"{b}'s frame f against {a}'s frame f + k:", ta, tb, common)


def reference_checks(p: Protocol, ref: str, tr: Mapping[str, Track], got_ref: Mapping[str, set[int]],
                     tracks: Mapping[str, Mapping[str, Track]], common: Mapping[str, Sequence[int]]
                     ) -> tuple[dict[str, Any], list[str]]:
    """PASS/FLAGGED per method against the reference, on the common IDs that the reference also covers."""
    on = {u: [f for f in fr if f in got_ref.get(u, set())] for u, fr in common.items()}
    on = {u: v for u, v in on.items() if v}
    out, flags = {}, []
    stats = [_STAT_OF[k] for k in p.modality.get("compare", []) if k in _STAT_OF] or list(_STAT_OF.values())
    for m, tm in tracks.items():
        summ, fl = pair(p, tr, tm, on)
        shift = _shift(f"{ref}~{m}", "against the reference", tr, tm, common)
        found = sorted({x.split(":")[0] for x in fl + shift})
        checked = any(k in summ for k in stats)  # a statistic of a COMPARED quantity was computed
        n = sum(len(v) for v in on.values()) if checked else 0
        result = "FLAGGED" if found else ("PASS" if n else "NOT_CHECKED")
        out[f"{ref}~{m}"] = {"result": result, "flags": found, "frames_checked": n, "label": REF_LABEL}
        if result == "NOT_CHECKED":
            flags.append(f"REFERENCE_NOT_CHECKED: {ref}~{m}: the reference has no finite compared value on any "
                         "common ID, so nothing was screened")
        flags += [f"{c}: {ref}~{m}: screening threshold exceeded on the common IDs (value withheld: reference "
                  "checks report pass/fail only)" for c in sorted({x.split(":")[0] for x in fl})]
        flags += shift
    return out, flags


def delivery_info(p: Protocol, methods: Sequence[Method], shares: Mapping[str, float] | None = None) -> list[str]:
    """Informational DELIVERY_DIFFERS lines: any non-delivered status, and any difference in delivery share."""
    out = []
    for m in methods:
        c = Counter("absent from the manifest" if m.units.get(u.unit_id) is None else m.units[u.unit_id]["status"]
                    for u in p.units)
        c.pop("delivered", None)
        if c:
            out.append(f"DELIVERY_DIFFERS: {m.name}: " + ", ".join(f"{n} {s}" for s, n in sorted(c.items()))
                       + f" of {len(p.units)} assigned units")
    if shares and len(set(shares.values())) > 1:
        out.append("DELIVERY_DIFFERS: " + " and ".join(f"{k} delivers {v:.3f}" for k, v in shares.items())
                   + " of the annotated frames")
    return out


def nonfinite_info(name: str, stages: Sequence[Mapping[str, Any]]) -> list[str]:
    n = int(stages[-1].get("nonfinite_frames", 0)) if stages else 0
    return [f"NONFINITE_ROWS: {name}: {n} delivered rows are non-finite and left out"] if n else []
