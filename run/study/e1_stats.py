"""E1 statistics: paired median log-ratio, stratified clustered bootstrap, Holm on calibrated thresholds.

Pure numpy / scipy; no data access. `summarise(rows, thresholds)` is the whole analysis for one scoring pass:

  rows        one dict per ASSIGNED unit: {unit_id, corpus ("H" | "D"), cluster, stratum, run (bool),
              ours: {delivered, e_mm}, theirs: {delivered, e_mm}}. A unit with run False is NOT RUN (budget
              stop): excluded from ITT and complete-case alike, counted in the assigned total.
  thresholds  size-calibrated one-sided p thresholds per corpus, {"H": {"0.025": a, "0.05": b}, "D": {...}}
              (`load_thresholds` reads the committed e1_power size JSON).

ITT (primary): d(u) = log(e_O / e_T) with the pre-declared bad values: only ours undelivered +BAD, only theirs
undelivered -BAD, both undelivered 0. Complete-case: units both methods delivered. The estimate is the MEDIAN of d
(never the ratio of medians). One-sided p = (1 + #{M* >= log 1.15}) / (B + 1) over B cluster-bootstrap medians
(H: instance clusters resampled within category; D: sequences; numpy default_rng(1)). Holm over the two corpora: corpus c
passes if p_c <= a_c(0.025), or p_c <= a_c(0.05) and the other corpus has p <= its a(0.025). An INCOMPLETE corpus
(< 80% of its assigned units RUN) gets no verdict and enters Holm with p = 1. INFERIOR: the one-sided 95% LOWER bound
(5th percentile of M*) exceeds log 1.15. LOWER ERROR (hierarchical, NON-INFERIOR only): the calibrated upper bound at
the level the corpus passed (the 1 - a_c(level) percentile of M*) is below 0. Sensitivity: the exact cluster sign test
(per-cluster median of d against log 1.15, ties dropped, one-sided binomial).
"""
from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest

MARGIN = math.log(1.15)
BAD = 1e6
B_DEFAULT = 10_000
SEED = 1
MIN_RUN_FRACTION = 0.8
GOOD_MM, TAIL_MM = 50.0, 150.0
NOMINAL = {"0.025": 0.025, "0.05": 0.05}
POWER_KEYS = {"hoi": "H", "dex": "D"}


def paired_d(e_o: Sequence[float], e_t: Sequence[float], ok_o: Sequence[bool], ok_t: Sequence[bool]) -> np.ndarray:
    """ITT paired log-ratios with the pre-declared bad values (module docstring)."""
    e_o, e_t = np.asarray(e_o, np.float64), np.asarray(e_t, np.float64)
    ok_o, ok_t = np.asarray(ok_o, bool), np.asarray(ok_t, bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.log(e_o / e_t)
    d = np.where(ok_o & ok_t, d, 0.0)
    d = np.where(~ok_o & ok_t, BAD, d)
    d = np.where(ok_o & ~ok_t, -BAD, d)
    if not np.isfinite(d).all():
        raise ValueError("a delivered unit has a non-finite or non-positive error")
    return d


def _codes(labels: Sequence[Any]) -> tuple[np.ndarray, list[Any]]:
    uniq = sorted({str(x) for x in labels})
    return np.asarray([uniq.index(str(x)) for x in labels], dtype=np.int64), uniq


def bootstrap_medians(d: np.ndarray, clusters: Sequence[Any], strata: Sequence[Any] | None, b: int = B_DEFAULT,
                      seed: int = SEED) -> np.ndarray:
    """B stratified cluster-bootstrap medians of `d` (exact weighted medians, e1_power._boot_median_p's method)."""
    d = np.asarray(d, np.float64)
    cl, cl_names = _codes(clusters)
    if strata is None:
        st_of_cluster = np.zeros(len(cl_names), dtype=np.int64)
    else:
        st_of: dict[str, str] = {}
        for c, s in zip(clusters, strata, strict=True):
            if st_of.setdefault(str(c), str(s)) != str(s):
                raise ValueError(f"cluster {c} spans strata")
        st_of_cluster, _ = _codes([st_of[c] for c in cl_names])
    rng = np.random.default_rng(seed)
    counts = np.zeros((b, len(cl_names)), dtype=np.int64)
    for s in np.unique(st_of_cluster):
        idx = np.flatnonzero(st_of_cluster == s)
        counts[:, idx] = rng.multinomial(idx.size, np.full(idx.size, 1.0 / idx.size), size=b)
    order = np.argsort(d, kind="stable")
    w = counts[:, cl[order]]
    cum = np.cumsum(w, axis=1)
    tot = cum[:, -1]
    lo, hi = (tot + 1) // 2, tot // 2 + 1
    ds = d[order]
    return 0.5 * (ds[(cum >= lo[:, None]).argmax(1)] + ds[(cum >= hi[:, None]).argmax(1)])


def corpus_test(d: np.ndarray, clusters: Sequence[Any], strata: Sequence[Any] | None, b: int = B_DEFAULT,
                seed: int = SEED) -> dict[str, Any]:
    """Median, one-sided p against log 1.15, percentile bounds and the bootstrap medians of one corpus."""
    d = np.asarray(d, np.float64)
    if d.size == 0:
        return {"n": 0, "median": None, "p": 1.0, "meds": np.zeros(0)}
    meds = bootstrap_medians(d, clusters, strata, b, seed)
    return {"n": int(d.size), "n_clusters": len(set(map(str, clusters))), "median": float(np.median(d)),
            "ratio": float(math.exp(np.median(d))) if abs(np.median(d)) < 700 else None,
            "p": (1 + int((meds >= MARGIN).sum())) / (b + 1), "ub95": float(np.quantile(meds, 0.95)),
            "lb95_one_sided": float(np.quantile(meds, 0.05)),
            "ci95": [float(np.quantile(meds, 0.025)), float(np.quantile(meds, 0.975))], "meds": meds}


def holm(p: dict[str, float], thr: dict[str, dict[str, float]]) -> dict[str, str | None]:
    """Per corpus the Holm level it passes at ("0.025" | "0.05") or None (two hypotheses, calibrated thresholds)."""
    first = {c: p[c] <= thr[c]["0.025"] for c in p}
    out: dict[str, str | None] = {}
    for c, pc in p.items():
        others = [o for o in p if o != c]
        if first[c]:
            out[c] = "0.025"
        elif pc <= thr[c]["0.05"] and all(first[o] for o in others):
            out[c] = "0.05"
        else:
            out[c] = None
    return out


def cluster_sign_test(d: Sequence[float], clusters: Sequence[Any]) -> dict[str, Any]:
    """Exact cluster sign test: per-cluster median d vs log 1.15; favourable = below; ties dropped; one-sided."""
    d = np.asarray(d, np.float64)
    by: dict[str, list[float]] = {}
    for x, c in zip(d, clusters, strict=True):
        by.setdefault(str(c), []).append(float(x))
    med = np.asarray([np.median(v) for v in by.values()])
    fav, unf = int((med < MARGIN).sum()), int((med > MARGIN).sum())
    n = fav + unf
    p = float(binomtest(fav, n, 0.5, alternative="greater").pvalue) if n else 1.0
    return {"clusters": len(med), "favourable": fav, "unfavourable": unf, "ties": len(med) - n, "p": p}


def _fraction_ci(ind: np.ndarray, clusters: Sequence[Any], strata: Sequence[Any] | None, b: int,
                 seed: int) -> dict[str, Any]:
    """Count and cluster-bootstrap 95% CI of the fraction of units with `ind` true."""
    cl, names = _codes(clusters)
    rng = np.random.default_rng(seed)
    if strata is None:
        groups = [np.arange(len(names))]
    else:
        st = {str(c): str(s) for c, s in zip(clusters, strata, strict=True)}
        groups = [np.asarray([i for i, n in enumerate(names) if st[n] == s]) for s in sorted(set(st.values()))]
    counts = np.zeros((b, len(names)), dtype=np.int64)
    for g in groups:
        counts[:, g] = rng.multinomial(g.size, np.full(g.size, 1.0 / g.size), size=b)
    w = counts[:, cl]
    frac = (w * ind[None, :]).sum(1) / np.maximum(w.sum(1), 1)
    return {"count": int(ind.sum()), "n": int(ind.size), "fraction": float(ind.mean()) if ind.size else None,
            "ci95": [float(np.quantile(frac, 0.025)), float(np.quantile(frac, 0.975))]}


def load_thresholds(size_json: Path) -> dict[str, dict[str, float]]:
    """The calibrated thresholds of an e1_power `--size-only` result, keyed by corpus H / D."""
    thr = json.loads(Path(size_json).read_text())["thresholds"]
    return {POWER_KEYS[k]: {lv: float(v) for lv, v in t.items()} for k, t in thr.items()}


def summarise(rows: Sequence[dict[str, Any]], thresholds: dict[str, dict[str, float]], b: int = B_DEFAULT,
              seed: int = SEED) -> dict[str, Any]:
    """The full analysis (module docstring). Returns {"corpora": {c: {...}}, "holm": {...}}."""
    out: dict[str, Any] = {"margin": MARGIN, "B": b, "seed": seed, "thresholds": thresholds, "corpora": {}}
    tests: dict[str, dict[str, Any]] = {}
    for c in sorted({r["corpus"] for r in rows}):
        assigned = [r for r in rows if r["corpus"] == c]
        run = [r for r in assigned if r.get("run", True)]
        ok_o = np.asarray([bool(r["ours"]["delivered"]) for r in run], bool)
        ok_t = np.asarray([bool(r["theirs"]["delivered"]) for r in run], bool)
        e_o = np.asarray([r["ours"].get("e_mm") if r["ours"]["delivered"] else np.nan for r in run], np.float64)
        e_t = np.asarray([r["theirs"].get("e_mm") if r["theirs"]["delivered"] else np.nan for r in run], np.float64)
        cl = [r["cluster"] for r in run]
        st = None if c == "D" or all(r.get("stratum") is None for r in run) else [r["stratum"] for r in run]
        d_itt = paired_d(e_o, e_t, ok_o, ok_t)
        both = ok_o & ok_t
        itt = corpus_test(d_itt, cl, st, b, seed)
        cc = corpus_test(d_itt[both], [x for x, k in zip(cl, both) if k],
                         None if st is None else [x for x, k in zip(st, both) if k], b, seed)
        run_frac = len(run) / len(assigned) if assigned else 0.0
        tests[c] = itt
        cc_cl = [x for x, k in zip(cl, both) if k]
        cc_st = None if st is None else [x for x, k in zip(st, both) if k]
        counts: dict[str, Any] = {}
        for name, ind in (("good_ours", e_o[both] <= GOOD_MM), ("good_theirs", e_t[both] <= GOOD_MM),
                          ("tail_ours", e_o[both] > TAIL_MM), ("tail_theirs", e_t[both] > TAIL_MM),
                          ("wins_ours", e_o[both] < e_t[both])):
            counts[name] = _fraction_ci(np.asarray(ind, bool), cc_cl, cc_st, b, seed) if both.any() else None
        out["corpora"][c] = {
            "n_assigned": len(assigned), "n_run": len(run), "n_not_run": len(assigned) - len(run),
            "run_fraction": run_frac, "incomplete": run_frac < MIN_RUN_FRACTION,
            "delivered_ours": int(ok_o.sum()), "delivered_theirs": int(ok_t.sum()), "n_both": int(both.sum()),
            "itt": {k: v for k, v in itt.items() if k != "meds"},
            "complete_case": {k: v for k, v in cc.items() if k != "meds"},
            "sign_test_itt": cluster_sign_test(d_itt, cl) if len(run) else None,
            "sign_test_cc": cluster_sign_test(d_itt[both], cc_cl) if both.any() else None,
            "counts_complete_case": counts}
    p = {c: (1.0 if out["corpora"][c]["incomplete"] else tests[c]["p"]) for c in tests}
    thr = {c: thresholds.get(c, NOMINAL) for c in tests}
    level = holm(p, thr)
    out["holm"] = {"p": p, "pass_level": level}
    for c, rec in out["corpora"].items():
        t = tests[c]
        if rec["incomplete"]:
            verdict = "INCOMPLETE"
        elif level[c] is not None:
            verdict = "NON-INFERIOR"
        elif t["n"] and t["lb95_one_sided"] > MARGIN:
            verdict = "INFERIOR"
        else:
            verdict = "INCONCLUSIVE"
        rec["verdict"] = verdict
        rec["calibrated_ub"] = float(np.quantile(t["meds"], 1 - thr[c]["0.05"])) if t["n"] else None
        rec["lower_error"] = bool(verdict == "NON-INFERIOR"
                                  and float(np.quantile(t["meds"], 1 - thr[c][level[c]])) < 0)
    return out


if __name__ == "__main__":
    import sys

    print("usage: not a command. A library: `from run.study.e1_stats import load_thresholds, summarise` "
          "(see the module docstring).")
    raise SystemExit(0 if sys.argv[1:] in ([], ["-h"], ["--help"]) else 2)
