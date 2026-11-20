"""Tracker-only PREFIX TRACE: certification events recorded against the cumulative training budget.

WRITER (`PrefixTrace`): an APPEND-ONLY JSONL file written while a tracker family trains. Rows (`kind`):
- "header": once per writer opened (a resumed run appends another; every header of one file must agree): the
  family, the run-level certificate source, and the family's references `refs` (task = clip id, video uid,
  reference sha256), whose count `n_refs` is the per-reference divisor;
- "certificate": one per certificate EVENT (a closed-loop CPU certificate of a stored candidate, certified or not):
  the clip id, video uid, reference sha256, the certificate `source`, `certified`, and the budget at the event;
- "checkpoint": one per budget checkpoint (every `every` vectorised steps of the trainer, and a final one at
  `close`): the budget only.
The budget of a row is the family's CUMULATIVE consumption when the row is written: executor `rows` and GPU-s
(`gpu_s`, from the driver's clock: lock-held GPU-s of earlier units + this unit's since its lock), and the
PER-REFERENCE budget = cumulative / `n_refs` (`per_ref_rows`, `per_ref_gpu_s`): a per-reference budget rho has
family total n_k x rho, and the tracker's per-clip budget is its cap over its clips. A certificate's budget INCLUDES
the rows of the step that produced it (the trainer writes it after that step's rows are counted;
`fastdo.amortised.trainer`). Each row also carries `clip_rows`, the rows executed on that clip so far (diagnostic,
never used by the reader).

Sources (`SOURCES`): "tracker" (an alpha = 0, row-0 tracker candidate certified closed loop: the only counted
source), "pass@S" (flow prior), "MPC", "fallback", "H2", "H2+" and "smoke" (a smoke-clip unit). A trainer writer
labels its certificates with its run-level source ("tracker" for a tracker run, "smoke" for a smoke run).

READER: `p_at(rows, clips, c)` = the fraction of the clip set `clips` (e.g. the 24 Ego4D MT-informative clips in 9
videos) certified by the TRACKER ALONE within per-reference budget c: a clip counts iff some certificate row with
source "tracker" and certified True names it at a per-reference budget <= c (a clip certified twice counts once, at
its FIRST, i.e. smallest, budget). Every other source is ignored. The trace's MEASURED range ends at its last
measured point: the largest budget of any checkpoint OR certificate row (a certificate row is a measured point: every
certificate up to it was written before it, append-only). So a unit killed before `close` (its hold timeout, the
NVML exit 3, a crash) loses no certificate written after its last periodic checkpoint. c beyond that point is
UNMEASURED (None): the prefix there was never run. `curve` gives p at every measured point (checkpoints and
certificate events; its last point is the final tracker-certified count, whether or not the unit closed the trace),
and `g_star` reads off the smallest g in {1, 2, 4, 8} with p(g x c*) >= 0.10.

UNIT of g*: ROWS. rho* = g* x r_H0 x c* is a
per-reference ROW budget, so the reader's registered unit is executor ROWS: `curve` and
`first_certified` default to per-reference rows (`Rows`), `p_at` reads the unit of its typed budget, and `g_star`
decides ONLY in rows, with c* given in rows (`c_star_rows(r_H0, c*)` = Rows(r_H0 x c*), c* = the fallback's c_cap
in GPU-s). The GPU-s are reported BESIDE it, never decision-bearing: per grid point, the per-reference GPU-s the
family had spent at the first measured point whose per-reference rows reach g x c* (`gpu_s_beside`; None when the
trace has no GPU-s clock or that budget was not reached). Unit setup, warm-up and certification are in the GPU-s
and not in the rows, which is why the two differ. `p_at` with a `GpuS` budget stays available for diagnostics.

TYPED BUDGETS: a budget is never a bare number.
A per-reference ROW budget is a `Rows` (`c_star_rows` returns one), a per-reference GPU-s budget a `GpuS`
(diagnostics only). `p_at` takes either and reads the matching column; `g_star` and `gpu_s_beside` take ONLY `Rows`;
`curve` and `first_certified` take the unit as the type (`Rows`, the default, or `GpuS`) and `curve`'s points carry
typed budgets. A bare float, int or numpy scalar, a unit string, or a GpuS where rows are required raises TypeError,
so c* in GPU-s can no longer be read as rows by accident (e.g. g_star(R, clips, 30.0)).
"""

from __future__ import annotations

import datetime as dt
import json
import math
import numbers
import os
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Self

__all__ = ["COUNTED", "GRID", "SOURCES", "GpuS", "PrefixTrace", "Rows", "TraceRef", "c_star_rows", "curve",
           "first_certified", "g_star", "gpu_s_beside", "p_at", "read"]

SOURCES = ("tracker", "pass@S", "MPC", "fallback", "H2", "H2+", "smoke")
COUNTED = "tracker"
GRID = (1, 2, 4, 8)
THRESHOLD = 0.10


class _Budget(float):
    """A typed per-reference budget (module docstring, TYPED BUDGETS). Refuses the OTHER budget type, so a value
    cannot be re-labelled from GPU-s to rows (or back) by wrapping it, and anything that is not a finite, non-negative
    real number (a bool, a string, None, NaN)."""

    __slots__ = ()

    def __new__(cls, value: Any) -> Self:
        if isinstance(value, _Budget) and not isinstance(value, cls):
            raise TypeError(f"{cls.__name__} from {type(value).__name__} {value!r}: a budget cannot change unit")
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise TypeError(f"{cls.__name__} from {type(value).__name__} {value!r}: a budget is a real number")
        if not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"{cls.__name__}({value!r}): a budget is finite and non-negative")
        return super().__new__(cls, value)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({float(self)!r})"


class Rows(_Budget):
    """A per-reference executor ROW budget (the registered unit of g*)."""

    __slots__ = ()


class GpuS(_Budget):
    """A per-reference GPU-s budget (diagnostics; never decision-bearing for g*)."""

    __slots__ = ()


UNITS = {Rows: "per_ref_rows", GpuS: "per_ref_gpu_s"}


@dataclass(frozen=True)
class TraceRef:
    """One reference of the family: its clip id (task), Ego4D video uid (or source id), reference sha256."""

    task: str
    video_uid: str
    ref_sha256: str


def _utc() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class PrefixTrace:
    """The append-only writer (module docstring). `gpu_s()` returns the family's cumulative GPU-s (None on a CPU
    run: the GPU-s columns are then None and the reader refuses a `GpuS` budget)."""

    def __init__(self, path: Path, refs: Sequence[TraceRef], *, family: str, source: str = COUNTED,
                 gpu_s: Callable[[], float | None] | None = None, every: int = 1) -> None:
        if source not in SOURCES:
            raise ValueError(f"certificate source {source!r} not in {SOURCES}")
        if not refs or len({r.task for r in refs}) != len(refs):
            raise ValueError("a trace needs >= 1 reference with unique clip ids")
        if every < 1:
            raise ValueError("checkpoint cadence must be >= 1 step")
        self.path, self.refs, self.family, self.source = Path(path), list(refs), family, source
        self.gpu_s, self.every = gpu_s, int(every)
        self.n_refs = len(self.refs)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write({"kind": "header", "family": family, "source": source, "n_refs": self.n_refs,
                     "refs": [asdict(r) for r in self.refs]})

    def _write(self, row: dict[str, Any]) -> dict[str, Any]:
        row = {"utc": _utc(), **row}
        with self.path.open("a") as fh:
            fh.write(json.dumps(row, sort_keys=True, default=float) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        return row

    def budget(self, rows: int) -> dict[str, Any]:
        g = None if self.gpu_s is None else self.gpu_s()
        return {"rows": int(rows), "gpu_s": None if g is None else float(g), "n_refs": self.n_refs,
                "per_ref_rows": int(rows) / self.n_refs, "per_ref_gpu_s": None if g is None else float(g) / self.n_refs}

    def certificate(self, clip: int, *, rows: int, certified: bool, source: str | None = None, step: int | None = None,
                    episode: int | None = None, clip_rows: int | None = None, sha: str | None = None) -> dict[str, Any]:
        """One certificate event of clip index `clip` at cumulative executor `rows` (module docstring)."""
        src = self.source if source is None else source
        if src not in SOURCES:
            raise ValueError(f"certificate source {src!r} not in {SOURCES}")
        r = self.refs[int(clip)]
        return self._write({"kind": "certificate", "clip": r.task, "video_uid": r.video_uid,
                            "ref_sha256": r.ref_sha256, "source": src, "certified": bool(certified), "step": step,
                            "episode": episode, "clip_rows": clip_rows, "cert_sha": sha, **self.budget(rows)})

    def checkpoint(self, *, rows: int, step: int | None = None, final: bool = False) -> dict[str, Any]:
        return self._write({"kind": "checkpoint", "step": step, "final": bool(final), **self.budget(rows)})

    def close(self, *, rows: int, step: int | None = None) -> dict[str, Any]:
        """The final checkpoint: the trace's last point."""
        return self.checkpoint(rows=rows, step=step, final=True)


# ------------------------------------------------------------------------------------------------ reader
def read(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    heads = [r for r in rows if r["kind"] == "header"]
    if not heads:
        raise ValueError(f"{path}: no header row")
    key = lambda h: (h["family"], h["n_refs"], json.dumps(h["refs"], sort_keys=True))  # noqa: E731
    if len({key(h) for h in heads}) != 1:
        raise ValueError(f"{path}: headers disagree (one trace file per family)")
    return rows


def _col(unit: Any) -> str:
    """The column of a unit TYPE (`Rows` or `GpuS`); anything else (a unit string, None, a value) raises TypeError."""
    if not (isinstance(unit, type) and unit in UNITS):
        raise TypeError(f"unit {unit!r}: pass the type Rows or GpuS")
    return UNITS[unit]


def _typed(c: Any, *allowed: type) -> type:
    """The unit type of budget `c`, refused (TypeError) unless it is an instance of one of `allowed`."""
    for t in allowed:
        if isinstance(c, t):
            return t
    names = " or ".join(t.__name__ for t in allowed)
    raise TypeError(f"a budget must be typed ({names}), not a bare {type(c).__name__} {c!r} ("
                    "c* in rows comes from c_star_rows(r_H0, c_cap))")


def _clip_set(rows: list[dict[str, Any]], clips: Iterable[str]) -> list[str]:
    cs = sorted(set(clips))
    known = {r["task"] for r in next(r for r in rows if r["kind"] == "header")["refs"]}
    if not cs:
        raise ValueError("an empty clip set")
    missing = [c for c in cs if c not in known]
    if missing:
        raise ValueError(f"clips not among the family's references: {missing}")
    return cs


def first_certified(rows: list[dict[str, Any]], clips: Iterable[str], unit: type = Rows) -> dict[str, float]:
    """Per clip of the set: the smallest per-reference budget (in `unit`: Rows or GpuS) at which a TRACKER
    certificate certified it."""
    col, cs = _col(unit), set(_clip_set(rows, clips))
    out: dict[str, float] = {}
    for r in rows:
        if r["kind"] != "certificate" or r["source"] != COUNTED or not r["certified"] or r["clip"] not in cs:
            continue
        if r[col] is None:
            raise ValueError(f"a certificate row without {col}: the trace has no {unit} budget")
        out[r["clip"]] = min(out.get(r["clip"], float("inf")), float(r[col]))
    return out


def _points(rows: list[dict[str, Any]], col: str) -> list[float]:
    """Every measured budget point: checkpoint AND certificate rows (module docstring)."""
    return [float(r[col]) for r in rows if r["kind"] in ("checkpoint", "certificate") and r.get(col) is not None]


def _last(rows: list[dict[str, Any]], col: str) -> float | None:
    pts = _points(rows, col)
    return max(pts) if pts else None


def p_at(rows: list[dict[str, Any]], clips: Iterable[str], c: Rows | GpuS) -> float | None:
    """p(c): the fraction of `clips` certified by the tracker alone within per-reference budget c, a TYPED budget
    (`Rows`, or `GpuS` for diagnostics; a bare number raises TypeError); None if c lies beyond the trace's last
    measured point (unmeasured)."""
    unit = _typed(c, Rows, GpuS)
    cs = _clip_set(rows, clips)
    last = _last(rows, _col(unit))
    if last is None or float(c) > last:
        return None
    first = first_certified(rows, cs, unit)
    return sum(1 for b in first.values() if b <= float(c)) / len(cs)


def curve(rows: list[dict[str, Any]], clips: Iterable[str], unit: type = Rows) -> list[dict[str, Any]]:
    """p at every measured point (checkpoints and certificate events), in budget order, each point's budget `c` typed
    (`unit`: Rows, the default, or GpuS); the last point is the final tracker-certified count."""
    cs = _clip_set(rows, clips)
    col = _col(unit)
    first = first_certified(rows, cs, unit)
    pts = sorted(set(_points(rows, col)))
    return [{"c": unit(c), "certified": sum(1 for b in first.values() if b <= c),
             "p": sum(1 for b in first.values() if b <= c) / len(cs)} for c in pts]


def c_star_rows(r_h0: float, c_star_s: float | GpuS) -> Rows:
    """c* in ROWS, a typed `Rows`: r_H0 (rows per GPU-s, H0's measured rate) x c* (the fallback's c_cap, GPU-s).
    Refuses a c* already typed as Rows (it would be converted twice), and any r_H0 or c* that
    is not a real number (a bool, a string, None) or not finite and strictly positive: '200' was coerced by float()
    and r_H0 = 0 gave c* = 0 rows, so g* read p(0) = 0 at every grid point without an error."""
    if isinstance(c_star_s, Rows) or isinstance(r_h0, _Budget):
        raise TypeError("c_star_rows takes r_H0 (rows per GPU-s) and c* in GPU-s, not a budget already in rows")
    for name, v in (("r_H0", r_h0), ("c*", c_star_s)):
        if isinstance(v, bool) or not isinstance(v, numbers.Real):
            raise TypeError(f"c_star_rows: {name} must be a real number, not {type(v).__name__} {v!r}")
        if not math.isfinite(float(v)) or float(v) <= 0:
            raise ValueError(f"c_star_rows: {name} = {v!r} must be finite and strictly positive")
    return Rows(float(r_h0) * float(c_star_s))


def gpu_s_beside(rows: list[dict[str, Any]], c_rows: Rows) -> float | None:
    """The per-reference GPU-s spent at the first measured point whose per-reference rows reach `c_rows` (a `Rows`;
    module docstring, "UNIT of g*"): reported beside the row reading, never decision-bearing."""
    _typed(c_rows, Rows)
    pts = sorted((float(r["per_ref_rows"]), r.get("per_ref_gpu_s")) for r in rows
                 if r["kind"] in ("checkpoint", "certificate"))
    hit = [g for c, g in pts if c >= float(c_rows)]
    return None if not hit or hit[0] is None else float(hit[0])


def g_star(rows: list[dict[str, Any]], clips: Iterable[str], c_star: Rows, *, grid: Sequence[int] = GRID,
           threshold: float = THRESHOLD) -> dict[str, Any]:
    """The smallest g in `grid` with p(g x c*) >= `threshold`, read in ROWS: `c_star` is a TYPED
    per-reference row budget, `Rows` (`c_star_rows(r_H0, c_cap)`); a bare number or a GpuS raises TypeError. Each grid
    point's p (None = unmeasured: the trace stopped before g x c*) and, beside it, the per-reference GPU-s at that row
    budget (`gpu_s_beside`)."""
    _typed(c_star, Rows)
    pts = {int(g): p_at(rows, clips, Rows(g * float(c_star))) for g in grid}
    hit = [g for g in sorted(pts) if pts[g] is not None and pts[g] >= threshold]
    return {"g_star": hit[0] if hit else None, "p": pts, "c_star": float(c_star), "unit": "rows",
            "gpu_s_beside": {int(g): gpu_s_beside(rows, Rows(g * float(c_star))) for g in grid},
            "threshold": threshold, "n_clips": len(set(clips))}
