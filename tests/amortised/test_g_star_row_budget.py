"""g* takes ROWS only, typed.

`g_star` accepts only a typed row budget (`prefix_trace.Rows`, which `c_star_rows(r_H0, c_cap)` returns); a plain
float raises. The reader's other budget arguments are typed too: `p_at` takes `Rows` or `GpuS` (diagnostic) and reads
the matching column, `curve` / `first_certified` take the unit as a type, and a budget cannot be re-labelled from one
unit to the other. The case this guards: a PrefixTrace read with c_cap in GPU-s passed positionally
(g_star(R, clips, 30.0)) silently gave p = 0 at every grid point; it now raises."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fastdo.amortised import prefix_trace as T

REFS = [T.TraceRef(f"c{i}", f"v{i}", f"{i:064x}") for i in range(10)]
CLIPS = [r.task for r in REFS]


def _trace(tmp: Path) -> list[dict]:
    """2,000 rows per GPU-s after a 500 GPU-s setup; c0 certified at 1,000 family rows (100 per reference), c1 at
    4,000 (400 per reference)."""
    clock = {"g": 500.0}
    tr = T.PrefixTrace(tmp / "trace.jsonl", REFS, family="F", gpu_s=lambda: clock["g"])
    for rows, clip in ((1000, 0), (2000, None), (4000, 1), (8000, None)):
        clock["g"] = 500.0 + rows / 10.0
        if clip is not None:
            tr.certificate(clip, rows=rows, certified=True)
        tr.checkpoint(rows=rows)
    tr.close(rows=8000)
    return T.read(tmp / "trace.jsonl")


@pytest.mark.parametrize("bad", [100.0, 100, np.float64(100.0), np.int64(100), "100", None, True],
                         ids=["float", "int", "np_float", "np_int", "str", "none", "bool"])
def test_g_star_refuses_a_bare_number(tmp_path, bad) -> None:
    rows = _trace(tmp_path)
    with pytest.raises(TypeError, match="typed"):
        T.g_star(rows, CLIPS, bad)


def test_g_star_takes_rows_only_and_c_star_rows_returns_rows(tmp_path) -> None:
    rows = _trace(tmp_path)
    c = T.c_star_rows(r_h0=2.0, c_star_s=50.0)
    assert type(c) is T.Rows and c == 100.0
    got = T.g_star(rows, CLIPS, c)
    assert got["g_star"] == 1 and got["p"] == {1: 0.1, 2: 0.1, 4: 0.2, 8: 0.2} and got["unit"] == "rows"
    assert type(got["c_star"]) is float and json.loads(json.dumps(got))["c_star"] == 100.0
    assert T.g_star(rows, CLIPS, T.Rows(100.0)) == got
    with pytest.raises(TypeError, match="Rows"):
        T.g_star(rows, CLIPS, T.GpuS(100.0))                            # GPU-s are never the g* unit
    with pytest.raises(TypeError):
        T.c_star_rows(2.0, T.Rows(50.0))                                # c* already in rows: no double conversion
    with pytest.raises(TypeError, match="typed"):
        T.gpu_s_beside(rows, 100.0)


def test_c_cap_in_gpu_s_passed_positionally_raises(tmp_path) -> None:
    rows = _trace(tmp_path)
    with pytest.raises(TypeError, match="c_star_rows"):
        T.g_star(rows, CLIPS, 30.0)


def test_p_at_reads_the_column_of_its_budgets_type_and_refuses_a_bare_number(tmp_path) -> None:
    rows = _trace(tmp_path)
    assert T.p_at(rows, CLIPS, T.Rows(100.0)) == 0.1 and T.p_at(rows, CLIPS, T.Rows(99.9)) == 0.0
    assert T.p_at(rows, CLIPS, T.GpuS(70.0)) == 0.1 and T.p_at(rows, CLIPS, T.GpuS(59.9)) == 0.0
    for bad in (100.0, 100, np.float64(100.0)):
        with pytest.raises(TypeError, match="typed"):
            T.p_at(rows, CLIPS, bad)


def test_curve_and_first_certified_take_a_unit_type_and_curve_points_are_typed(tmp_path) -> None:
    rows = _trace(tmp_path)
    pts = T.curve(rows, CLIPS)
    assert all(type(p["c"]) is T.Rows for p in pts) and pts[-1]["certified"] == 2
    assert all(type(p["c"]) is T.GpuS for p in T.curve(rows, CLIPS, unit=T.GpuS))
    assert json.loads(json.dumps(pts))[0]["c"] == float(pts[0]["c"])
    assert T.first_certified(rows, CLIPS) == {"c0": 100.0, "c1": 400.0}
    for bad in ("rows", "gpu_s", None, 1.0, float):
        with pytest.raises(TypeError, match="Rows or GpuS"):
            T.curve(rows, CLIPS, unit=bad)
        with pytest.raises(TypeError, match="Rows or GpuS"):
            T.first_certified(rows, CLIPS, bad)


def test_a_budget_cannot_change_unit() -> None:
    with pytest.raises(TypeError, match="cannot change unit"):
        T.Rows(T.GpuS(1.0))
    with pytest.raises(TypeError, match="cannot change unit"):
        T.GpuS(T.Rows(1.0))
    for bad in (True, "100", None):
        with pytest.raises(TypeError, match="real number"):
            T.Rows(bad)
    for bad in (float("nan"), float("inf"), -1.0):
        with pytest.raises(ValueError, match="finite and non-negative"):
            T.GpuS(bad)
    assert type(T.Rows(np.int64(4))) is T.Rows and T.Rows(np.float64(2.5)) == 2.5
    assert type(T.Rows(T.Rows(2.0))) is T.Rows and repr(T.Rows(2.0)) == "Rows(2.0)"
    assert T.Rows(3) * 2 == 6.0 and type(T.Rows(3) * 2) is float       # arithmetic drops the type: re-wrap
