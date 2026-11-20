# Copyright 2026 The alongray authors. SPDX-License-Identifier: Apache-2.0
"""The missing-outcome sensitivity card: finite-cohort bounds checked against EVERY binary completion for N <= 4.
The tests write files only under pytest's tmp_path."""

from __future__ import annotations

import itertools
import json
from fractions import Fraction

import pytest

from alongray import missing_bounds as MB

STATUSES = ("pass", "fail", "unknown")


def _protocol(ids, scorable=None, rule="fail"):
    scorable = scorable or {}
    return {"unit_definition": "one window of 30 frames", "endpoint": "centre error < 50 mm",
            "delivery_failure_counts_as": rule,
            "units": [{"id": i, "gt_scorable": scorable.get(i, True)} for i in ids]}


def _method(name, statuses, unit_definition="one window of 30 frames", endpoint="centre error < 50 mm"):
    return {"method": name, "unit_definition": unit_definition, "endpoint": endpoint, "statuses": dict(statuses)}


def _completions(st):
    """Every binary completion of one method's statuses: the pass counts it can reach."""
    s = sum(x == "pass" for x in st)
    u = sum(x == "unknown" for x in st)
    return [s + sum(bits) for bits in itertools.product((0, 1), repeat=u)]


@pytest.mark.parametrize("N", [1, 2, 3, 4])
def test_bounds_equal_the_extremes_over_every_binary_completion(N):
    ids = [f"u{i}" for i in range(N)]
    p = _protocol(ids)
    for sa in itertools.product(STATUSES, repeat=N):
        ca = _completions(sa)
        for sb in itertools.product(STATUSES, repeat=N):
            cb = _completions(sb)
            card = MB.sensitivity_card(p, _method("A", zip(ids, sa)), _method("B", zip(ids, sb)))
            assert card["N"] == N
            a, b = card["methods"]["A"], card["methods"]["B"]
            assert (a["pass_lower_numerator"], a["pass_upper_numerator"]) == (min(ca), max(ca))
            assert (b["pass_lower_numerator"], b["pass_upper_numerator"]) == (min(cb), max(cb))
            diffs = [x - y for x in ca for y in cb]
            d = card["difference_a_minus_b"]
            assert (d["lower_numerator"], d["upper_numerator"]) == (min(diffs), max(diffs))
            assert Fraction(d["lower_numerator"], N) == Fraction(min(diffs), N)
            assert d["lower"] == min(diffs) / N and d["upper"] == max(diffs) / N
            assert a["pass_lower"] == min(ca) / N and a["pass_upper"] == max(ca) / N


def test_known_delivery_failures_count_as_the_protocol_says_and_never_widen_the_bounds():
    ids = ["u0", "u1", "u2", "u3"]
    p = _protocol(ids)
    a = _method("A", {"u0": "pass", "u1": "delivery_failure", "u2": "unknown", "u3": "fail"})
    b = _method("B", {"u0": "pass", "u1": "pass", "u2": "pass", "u3": "pass"})
    c = MB.sensitivity_card(p, a, b)
    ma = c["methods"]["A"]
    assert (ma["s"], ma["u"], ma["known_fail"], ma["delivery_failures"]) == (1, 1, 2, 1)
    assert (ma["pass_lower"], ma["pass_upper"]) == (0.25, 0.5)
    assert (c["difference_a_minus_b"]["lower"], c["difference_a_minus_b"]["upper"]) == (-0.75, -0.5)
    with pytest.raises(ValueError, match="delivery_failure_counts_as"):
        MB.sensitivity_card(_protocol(ids, rule="unknown"), a, b)
    q = _protocol(ids)
    del q["delivery_failure_counts_as"]
    with pytest.raises(ValueError, match="delivery_failure_counts_as"):
        MB.sensitivity_card(q, a, b)


def test_non_scorable_ground_truth_is_disclosed_and_kept_out_of_N():
    ids = ["u0", "u1", "u2"]
    p = _protocol(ids, scorable={"u2": False})
    a = _method("A", {"u0": "pass", "u1": "unknown"})
    b = _method("B", {"u0": "fail", "u1": "pass"})
    c = MB.sensitivity_card(p, a, b)
    assert c["N"] == 2 and c["n_assigned"] == 3 and c["gt_not_scorable"] == ["u2"]
    with pytest.raises(ValueError, match="not GT-scorable"):
        MB.sensitivity_card(p, _method("A", {"u0": "pass", "u1": "pass", "u2": "pass"}), b)


def test_refusals():
    ids = ["u0", "u1"]
    good = _method("B", {"u0": "pass", "u1": "fail"})
    with pytest.raises(ValueError, match="duplicate"):
        MB.sensitivity_card(_protocol(["u0", "u0"]), good, good)
    with pytest.raises(ValueError, match="status"):
        MB.sensitivity_card(_protocol(ids), _method("A", {"u0": "pass", "u1": "maybe"}), good)
    with pytest.raises(ValueError, match="units"):
        MB.sensitivity_card(_protocol(ids), _method("A", {"u0": "pass"}), good)
    with pytest.raises(ValueError, match="units"):
        MB.sensitivity_card(_protocol(ids), _method("A", {"u0": "pass", "u1": "fail", "u9": "fail"}), good)
    with pytest.raises(ValueError, match="unit_definition"):
        MB.sensitivity_card(_protocol(ids), _method("A", {"u0": "pass", "u1": "fail"}, unit_definition="a clip"),
                            good)
    with pytest.raises(ValueError, match="endpoint"):
        MB.sensitivity_card(_protocol(ids), _method("A", {"u0": "pass", "u1": "fail"}, endpoint="F-10 > 0.5"), good)
    with pytest.raises(ValueError, match="no GT-scorable"):
        MB.sensitivity_card(_protocol(ids, scorable={"u0": False, "u1": False}), _method("A", {}), _method("B", {}))
    with pytest.raises(ValueError, match="same method"):
        MB.sensitivity_card(_protocol(ids), good, good)


def test_labels_and_application_gate():
    ids = [f"u{i}" for i in range(4)]
    c = MB.sensitivity_card(_protocol(ids), _method("A", dict.fromkeys(ids, "pass")),
                            _method("B", dict.fromkeys(ids, "unknown")))
    assert "not confidence intervals" in c["label"] and "not rankings" in c["label"]
    assert "Horowitz" in c["reference"] and "S030440760500062X" in c["reference"]
    assert c["application_gate"]["status"] == "NOT_ESTIMABLE"              # N < 20
    ids20 = [f"u{i}" for i in range(20)]
    full = MB.sensitivity_card(_protocol(ids20), _method("A", dict.fromkeys(ids20, "pass")),
                               _method("B", dict.fromkeys(ids20, "fail")))
    assert full["application_gate"]["status"] == "NOT_ESTIMABLE"           # no genuine unknown
    st = dict.fromkeys(ids20, "pass")
    st["u3"] = "unknown"
    ok = MB.sensitivity_card(_protocol(ids20), _method("A", st), _method("B", dict.fromkeys(ids20, "fail")))
    assert ok["application_gate"]["status"] == "ESTIMABLE"
    assert ok["difference_a_minus_b"]["lower"] == pytest.approx(19 / 20) and ok["difference_a_minus_b"]["upper"] == 1.0


def test_command_line(tmp_path, capsys):
    ids = ["u0", "u1"]
    files = {}
    for name, obj in (("p", _protocol(ids)), ("a", _method("A", {"u0": "pass", "u1": "unknown"})),
                      ("b", _method("B", {"u0": "fail", "u1": "fail"}))):
        files[name] = tmp_path / f"{name}.json"
        files[name].write_text(json.dumps(obj))
    assert MB.main([str(files["p"]), str(files["a"]), str(files["b"])]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["difference_a_minus_b"]["lower"] == 0.5 and out["difference_a_minus_b"]["upper"] == 1.0
    files["b"].write_text("{")
    assert MB.main([str(files["p"]), str(files["a"]), str(files["b"])]) == 2

