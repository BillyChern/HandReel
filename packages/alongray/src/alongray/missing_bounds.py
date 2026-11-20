# Copyright 2026 The alongray authors. SPDX-License-Identifier: Apache-2.0
"""Missing-outcome sensitivity card: finite-cohort bounds for a binary endpoint when some outcomes are unknown.

For a frozen common set of N assigned, ground-truth-scorable units and a binary endpoint, each method has s known
passes and u genuinely unknown outcomes. Over every way of completing the unknowns, its pass fraction lies in
[s/N, (s + u)/N], and the difference of two methods' pass fractions in

    [(s_A - s_B - u_B) / N, (s_A - s_B + u_A) / N],

and both intervals are sharp (each endpoint is reached by one completion). These are finite-cohort sensitivity
bounds: not confidence intervals, not bounds on a continuous error, and not rankings. They can be uninformatively
wide. Known delivery failures count as the frozen protocol says (``delivery_failure_counts_as``, which must be
declared and must be ``"fail"``); they are never turned into unknowns. Partial-identification precedent: J. L.
Horowitz and C. F. Manski, "Identification and estimation of statistical functionals using incomplete data",
Journal of Econometrics (2006), https://www.sciencedirect.com/science/article/pii/S030440760500062X .

Inputs (JSON objects):

- protocol: ``unit_definition``, ``endpoint``, ``delivery_failure_counts_as`` and ``units``, a list of
  ``{"id": ..., "gt_scorable": true|false}``; ids must be unique;
- each method: ``method``, ``unit_definition`` and ``endpoint`` (equal to the protocol's), and ``statuses``, one of
  ``pass``, ``fail``, ``unknown`` or ``delivery_failure`` for exactly the GT-scorable units.

Command line: ``python -m alongray.missing_bounds PROTOCOL.json A.json B.json`` prints the card (exit 2 on bad input).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Any

from alongray import strict_json

__all__ = ["DELIVERY_RULES", "LABEL", "MIN_UNITS_TO_APPLY", "REFERENCE", "STATUSES", "main", "sensitivity_card"]

STATUSES = ("pass", "fail", "unknown", "delivery_failure")
DELIVERY_RULES = ("fail",)
MIN_UNITS_TO_APPLY = 20
LABEL = ("finite-cohort sensitivity bounds over every binary completion of the unknown outcomes: not confidence "
         "intervals, not bounds on a continuous error, not rankings")
REFERENCE = ("Horowitz and Manski, Identification and estimation of statistical functionals using incomplete data, "
             "Journal of Econometrics 2006, https://www.sciencedirect.com/science/article/pii/S030440760500062X")


def _units(protocol: dict[str, Any]) -> tuple[list[str], list[str]]:
    units = protocol.get("units")
    if not isinstance(units, list):
        raise ValueError("protocol.units must be a list of {id, gt_scorable}")
    ids = [u.get("id") for u in units]
    if not all(isinstance(i, str) and i for i in ids):
        raise ValueError("every protocol unit needs a non-empty string id")
    if len(set(ids)) != len(ids):
        raise ValueError(f"protocol units have duplicate ids {sorted({i for i in ids if ids.count(i) > 1})[:5]}")
    if not all(isinstance(u.get("gt_scorable"), bool) for u in units):
        raise ValueError("every protocol unit needs a boolean gt_scorable")
    return [u["id"] for u in units if u["gt_scorable"]], [u["id"] for u in units if not u["gt_scorable"]]


def _counts(m: dict[str, Any], protocol: dict[str, Any], scorable: list[str], excluded: list[str]) -> dict[str, Any]:
    name = m.get("method")
    for k in ("unit_definition", "endpoint"):
        if m.get(k) != protocol[k]:
            raise ValueError(f"method {name!r}: {k} {m.get(k)!r} differs from the protocol's {protocol[k]!r}")
    st = m.get("statuses")
    if not isinstance(st, dict):
        raise ValueError(f"method {name!r}: statuses must be an object")
    off = sorted(set(st) & set(excluded))
    if off:
        raise ValueError(f"method {name!r}: units {off[:5]} are not GT-scorable and cannot carry a status")
    if set(st) != set(scorable):
        raise ValueError(f"method {name!r}: statuses must cover exactly the GT-scorable units (missing "
                         f"{sorted(set(scorable) - set(st))[:5]}, extra {sorted(set(st) - set(scorable))[:5]})")
    bad = sorted(i for i, v in st.items() if v not in STATUSES)
    if bad:
        raise ValueError(f"method {name!r}: undefined status for {bad[:5]}; allowed {STATUSES}")
    v = list(st.values())
    s, u, dfail = v.count("pass"), v.count("unknown"), v.count("delivery_failure")
    n = len(scorable)
    return {"s": s, "u": u, "known_fail": v.count("fail") + dfail, "delivery_failures": dfail,
            "pass_lower_numerator": s, "pass_upper_numerator": s + u, "pass_lower": s / n, "pass_upper": (s + u) / n}


def sensitivity_card(protocol: dict[str, Any], a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """The card for methods ``a`` and ``b`` on the protocol's frozen units (module docstring).

    Raises:
        ValueError: duplicate unit ids, an undeclared or unsupported delivery-failure rule, a unit definition or
            endpoint that differs from the protocol's, statuses that do not cover exactly the GT-scorable units, an
            undefined status, no GT-scorable unit, or the same method twice.
    """
    if not isinstance(protocol, dict):
        raise ValueError("protocol must be an object")
    for k in ("unit_definition", "endpoint"):
        if not (isinstance(protocol.get(k), str) and protocol[k].strip()):
            raise ValueError(f"protocol.{k} must be non-empty text")
    rule = protocol.get("delivery_failure_counts_as")
    if rule not in DELIVERY_RULES:
        raise ValueError(f"protocol.delivery_failure_counts_as must be declared as one of {DELIVERY_RULES}, got "
                         f"{rule!r}: a known delivery failure is never an unknown outcome")
    scorable, excluded = _units(protocol)
    if not scorable:
        raise ValueError("the protocol has no GT-scorable unit")
    if a.get("method") == b.get("method"):
        raise ValueError(f"the same method {a.get('method')!r} twice")
    ca, cb = _counts(a, protocol, scorable, excluded), _counts(b, protocol, scorable, excluded)
    n = len(scorable)
    lo = ca["s"] - cb["s"] - cb["u"]
    hi = ca["s"] - cb["s"] + ca["u"]
    reasons = []
    if n < MIN_UNITS_TO_APPLY:
        reasons.append(f"fewer than {MIN_UNITS_TO_APPLY} common GT-scorable units")
    if ca["u"] + cb["u"] == 0:
        reasons.append("no genuine unknown outcome")
    return {"label": LABEL, "reference": REFERENCE, "endpoint": protocol["endpoint"],
            "unit_definition": protocol["unit_definition"], "delivery_failure_counts_as": rule,
            "n_assigned": n + len(excluded), "N": n, "gt_not_scorable": excluded,
            "methods": {a["method"]: ca, b["method"]: cb}, "a": a["method"], "b": b["method"],
            "difference_a_minus_b": {"lower_numerator": lo, "upper_numerator": hi, "lower": lo / n, "upper": hi / n},
            "application_gate": {"status": "NOT_ESTIMABLE" if reasons else "ESTIMABLE", "reasons": reasons}}


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="alongray.missing_bounds", description=LABEL)
    p.add_argument("protocol")
    p.add_argument("a")
    p.add_argument("b")
    args = p.parse_args(argv)
    try:
        docs = []
        for f in (args.protocol, args.a, args.b):
            with open(f) as fh:
                docs.append(strict_json.load(fh, f))
        card = sensitivity_card(*docs)
    except (OSError, ValueError, TypeError, AttributeError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(json.dumps(card, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
