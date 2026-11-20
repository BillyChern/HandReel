# Copyright 2026 The alongray authors
# SPDX-License-Identifier: Apache-2.0
"""Pooling, the cluster bootstrap and the three row forms (alongray.pool)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from alongray import (
    cluster_bootstrap,
    e_par_split,
    pool_rows,
    pool_splits,
    pooled_shares,
    split_from_moments,
    split_of_row,
)
from alongray.pool import main


def _traj(n: int = 15, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.normal(0, 0.05, n), rng.normal(0, 0.05, n), rng.uniform(0.45, 0.75, n)])


def _units(k: int = 6) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(0)
    out = []
    for i in range(k):
        g = _traj(seed=i)
        out.append((rng.uniform(0.8, 1.2) * g + rng.normal(scale=0.01, size=g.shape), g))
    return out


def _moments(p: np.ndarray, g: np.ndarray) -> dict[str, float]:
    z = np.linalg.norm(g, axis=1)
    a = np.einsum("ij,ij->i", p, g) / z
    return {"pp": float(np.mean(a * a)), "pz": float(np.mean(a * z)), "zz": float(np.mean(z * z)), "n": len(z),
            "z1": float(np.mean(z)), "a1": float(np.mean(a))}


def test_moments_equal_the_per_frame_split() -> None:
    for p, g in _units():
        a, b = e_par_split(p, g), split_from_moments(**_moments(p, g))
        for k in ("e_par", "gauge_part", "off_orbit_part", "lam", "z_rms", "oracle_floor", "shift_part",
                  "shift_scale_part", "collinearity"):
            assert getattr(b, k) == pytest.approx(getattr(a, k), rel=1e-9, abs=1e-13), k


def test_moments_constant_range_and_pooled_shift_scale_share() -> None:
    g = np.tile([[0.0, 0.0, 0.7]], (10, 1))
    a = e_par_split(1.1 * g, g)
    b = split_from_moments(**_moments(1.1 * g, g))
    assert b.shift_scale_part == pytest.approx(a.shift_scale_part, rel=1e-9)
    sh = pool_splits([split_from_moments(**_moments(p, g)) for p, g in _units()])["pooled_share"]["shift_scale"]
    assert np.isfinite(sh) and 0.0 < sh <= 1.0 + 1e-12


def test_three_row_forms_pool_identically(tmp_path: Path) -> None:
    rows = []
    for i, (p, g) in enumerate(_units()):
        m = _moments(p, g)
        z = np.linalg.norm(g, axis=1)
        a = np.einsum("ij,ij->i", p, g) / z
        sp = e_par_split(p, g)
        rows += [{"g": "m", "cl": str(i // 2), "moments": m}, {"g": "z", "cl": str(i // 2), "z": z.tolist(),
                                                                "a_r": a.tolist()},
                 {"g": "s", "cl": str(i // 2), "split": {k: getattr(sp, k) for k in sp.__dataclass_fields__}}]
    res = pool_rows(rows, "g", "cl", n_boot=50, seed=0)
    for k in ("median_gauge_part", "median_off_orbit_part", "rms_e_par", "median_shift_part"):
        assert res["m"][k] == pytest.approx(res["z"][k], rel=1e-9) == res["s"][k]
    assert res["m"]["n_clusters"] == 3
    f = tmp_path / "rows.jsonl"
    f.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert main([str(f), "--group-key", "g", "--cluster-key", "cl", "--boot", "20", "--out", str(tmp_path / "o")]) == 0
    with pytest.raises(SystemExit) as e:
        main([str(f), "--boot", "10"])
    assert e.value.code == 2
    with pytest.raises(ValueError):
        pool_rows(rows, n_boot=5)
    with pytest.raises(ValueError):
        split_of_row({"unit_id": "x"})


def test_identities_low_n_and_bootstrap() -> None:
    sp = [e_par_split(p, g) for p, g in _units(8)]
    with pytest.warns(UserWarning):
        sp.append(e_par_split(*[x[:1] for x in _units(1)[0]]))
    p = pool_splits(sp)
    assert p["n_low_n"] == 1
    assert p["pooled_share"]["gauge"] + p["pooled_share"]["off_orbit"] == pytest.approx(1.0, rel=1e-12)
    assert p["rms_e_par"] ** 2 == pytest.approx(p["rms_gauge_part"] ** 2 + p["rms_off_orbit_part"] ** 2, rel=1e-12)
    cl = [str(i % 3) for i in range(len(sp))]

    def stat(ix: list[int]) -> float:
        return pooled_shares([sp[i] for i in ix])["gauge"]

    a = cluster_bootstrap(stat, cl, 200, 0)
    assert a == cluster_bootstrap(stat, cl, 200, np.random.default_rng(0))
    # the documented draw order and percentile, written out by hand
    gen, labels = np.random.default_rng(0), sorted(set(cl))
    by = {c: [i for i, x in enumerate(cl) if x == c] for c in labels}
    draws = [stat([i for j in gen.integers(0, 3, 3) for i in by[labels[j]]]) for _ in range(200)]
    assert a["value"] == (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))
    with pytest.raises(ValueError):
        pool_splits([])


def test_percentile_level_is_exact() -> None:
    """Continuous statistic, one unit per cluster: the interval is numpy.percentile at exactly 2.5 and 97.5."""
    x = np.random.default_rng(4).normal(size=40)
    labels = [f"{i:02d}" for i in range(40)]
    got = cluster_bootstrap(lambda ix: float(np.mean(x[ix])), labels, 500, 7)["value"]
    gen = np.random.default_rng(7)
    draws = [float(np.mean(x[gen.integers(0, 40, 40)])) for _ in range(500)]
    assert got == (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))
    got90 = cluster_bootstrap(lambda ix: float(np.mean(x[ix])), labels, 500, 7, level=0.9)["value"]
    assert got90 == (float(np.percentile(draws, 5.0)), float(np.percentile(draws, 95.0)))


def test_deprecated_a_alias_and_bad_rows(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    p, g = _units(1)[0]
    z = np.linalg.norm(g, axis=1)
    a = np.einsum("ij,ij->i", p, g) / z
    with pytest.warns(FutureWarning, match="a_r"):
        old = split_of_row({"z": z.tolist(), "a": a.tolist()})
    assert old == split_of_row({"z": z.tolist(), "a_r": a.tolist()})
    with pytest.raises(ValueError, match="> 0"):
        split_of_row({"z": [0.5, -0.1, 0.6], "a_r": [0.5, 0.5, 0.5]})
    with pytest.raises(ValueError, match="lacks"):
        split_of_row({"split": {"e_par": 0.1}})
    with pytest.raises(ValueError, match="not finite"):
        split_of_row({"split": {"e_par": "x", "gauge_part": 0.1, "off_orbit_part": 0.1, "lam": 1.0, "z_rms": 0.6,
                                "oracle_floor": 0.1, "n_frames": 5}})
    bad = tmp_path / "bad.jsonl"
    for line, msg in (('{"split": {"e_par": null}}', "alongray-pool: error"),
                      ('{"moments": {"pp": "x", "pz": 1, "zz": 1, "n": 3}}', "alongray-pool: error"),
                      ('[1, 2]', "alongray-pool: error")):
        bad.write_text(line + "\n")
        assert main([str(bad)]) == 2
        assert msg in capsys.readouterr().err
    lst = tmp_path / "list.json"
    lst.write_text("[]")
    with pytest.raises(SystemExit) as e:
        main([str(lst), "--rows-key", "rows"])
    assert e.value.code == 2
    assert "top level" in capsys.readouterr().err


def test_an_undefined_draw_does_not_remove_the_point_estimate(tmp_path: Path) -> None:
    """Cluster A has zero error and cluster B has error: the pooled share exists. A draw of A alone has no share; it
    is counted and left out of the percentiles, and the point estimate stays."""
    from alongray.pool import NOT_ESTIMABLE, UndefinedStatistic, _read_manifest, pool_rows

    rng = np.random.default_rng(3)
    g = rng.normal(size=(12, 3)) + np.array([0.0, 0.0, 0.6])
    zero = e_par_split(g, g)                                          # exact prediction: zero error
    some = e_par_split(1.1 * g, g)
    with pytest.raises(UndefinedStatistic):
        pooled_shares([zero, zero])
    assert pooled_shares([zero, some])["gauge"] == pytest.approx(1.0)
    splits, info = [zero, some], {}
    ci = cluster_bootstrap(lambda ix: pooled_shares([splits[i] for i in ix]), ["A", "B"], 200, 0, info=info)
    assert 0 < info["undefined_draws"] < 200 and info["n_boot"] == 200            # the A-only draws
    assert ci["gauge"] == (pytest.approx(1.0), pytest.approx(1.0))
    none: dict[str, int] = {}
    assert cluster_bootstrap(lambda ix: pooled_shares([zero for _ in ix]), ["A", "B"], 50, 0, info=none) == NOT_ESTIMABLE
    assert none["undefined_draws"] == 50
    with pytest.raises(KeyError):                                     # any other error still propagates
        cluster_bootstrap(lambda ix: {}["x"], ["A", "B"], 5, 0)
    z = np.linalg.norm(g, axis=1)                                     # the range rows: z and a_r per frame
    rows = [{"unit_id": u, "cluster": c, "z": z.tolist(), "a_r": (k * z).tolist()}
            for u, c, k in (("u1", "A", 1.0), ("u2", "B", 1.1))]
    out = pool_rows(rows, cluster_key="cluster", n_boot=200)
    res = next(iter(out.values()))
    assert res["pooled_share"]["gauge"] == pytest.approx(1.0) and res["pooled_share_ci95_undefined_draws"] > 0
    assert "conditional on a defined share" in res["pooled_share_ci95_reason"]
    m = tmp_path / "m.txt"
    m.write_text("u1\nu2\nu1\n")
    with pytest.raises(ValueError, match="already assigned on line 1"):
        _read_manifest(m, grouped=False)
    m.write_text("u1 g1\n")
    with pytest.raises(ValueError, match="expected"):                 # a second column without --group-key
        _read_manifest(m, grouped=False)
    m.write_text("u1 g1\nu1 g2\n")
    with pytest.raises(ValueError, match="already assigned"):         # a conflicting second assignment
        _read_manifest(m, grouped=True)
    m.write_text("u1 g1\nu2 g1\n")
    assert _read_manifest(m, grouped=True) == {"u1": "g1", "u2": "g1"}


def test_an_all_zero_cohort_keeps_its_lengths_and_counts() -> None:
    """Every unit exact: the share of a zero total does not exist, but the zero lengths and the counts do."""
    from alongray.pool import NOT_ESTIMABLE, pool_rows

    rng = np.random.default_rng(5)
    g = rng.normal(size=(12, 3)) + np.array([0.0, 0.0, 0.6])
    z = np.linalg.norm(g, axis=1)
    rows = [{"unit_id": u, "cluster": c, "z": z.tolist(), "a_r": z.tolist()} for u, c in (("u1", "A"), ("u2", "B"))]
    res = next(iter(pool_rows(rows, cluster_key="cluster", n_boot=50).values()))
    assert res["pooled_share"] is None and "zero error" in res["pooled_share_reason"]
    assert res["n_units"] == 2 and res["n_scored"] == 2 and res["median_e_par"] == pytest.approx(0.0, abs=1e-9)
    assert res["pooled_share_ci95"] == NOT_ESTIMABLE and "all 50 draws" in res["pooled_share_ci95_reason"]
    assert res["assignment"] == "observed rows only"
    with pytest.raises(ValueError, match="listed more than once"):
        pool_rows(rows, assigned=["u1", "u2", "u1"])
    with pytest.raises(ValueError, match="listed more than once"):
        pool_rows(rows, assigned={1: "all", "1": "all", "u1": "all", "u2": "all"})
    ok = next(iter(pool_rows(rows, assigned=["u1", "u2", "u3"]).values()))
    assert ok["assignment"] == "manifest" and ok["n_missing"] == 1



def _moment_row(unit: str, cluster: str, k: float, first: bool) -> dict:
    """A moments row for range rows a = k z; without the first moments z1, a1 its shift components do not exist."""
    z = np.linalg.norm(np.random.default_rng(7).normal(size=(12, 3)) + np.array([0.0, 0.0, 0.6]), axis=1)
    a = k * z + (0.01 if first else 0.0) * np.arange(12)
    m = {"pp": float(np.mean(a * a)), "pz": float(np.mean(a * z)), "zz": float(np.mean(z * z)), "n": 12}
    if first:
        m.update(z1=float(np.mean(z)), a1=float(np.mean(a)))
    return {"unit_id": unit, "cluster": cluster, "moments": m}


def test_a_component_without_first_moments_is_undefined_by_name_and_the_others_keep_their_intervals() -> None:
    """Gauge and off-orbit shares exist for every unit; the shift components need z1 and a1. A unit
    without them makes the full-cohort shift undefined. That is stated per component with its reason, no interval
    is printed for it, and the defined components keep theirs."""
    from alongray.pool import pool_rows

    rows = [_moment_row("u1", "A", 1.10, True), _moment_row("u2", "B", 1.05, True), _moment_row("u3", "C", 1.20, False)]
    res = next(iter(pool_rows(rows, cluster_key="cluster", n_boot=300).values()))
    share, ci = res["pooled_share"], res["pooled_share_ci95"]
    needs = {"shift", "shift_scale", "pure_shift_null"}
    assert all(share[k] is None for k in needs) and set(res["pooled_share_undefined"]) == needs
    assert "1 of 3 units" in res["pooled_share_undefined"]["shift"]
    assert all(isinstance(share[k], float) and math.isfinite(share[k]) for k in ("gauge", "off_orbit"))
    assert set(ci) == {"gauge", "off_orbit"} and all(math.isfinite(x) for k in ci for x in ci[k])
    assert set(res["pooled_share_ci95_component_reason"]) == needs
    assert "full-cohort value is undefined" in res["pooled_share_ci95_component_reason"]["shift"]
    full = next(iter(pool_rows(rows[:2], cluster_key="cluster", n_boot=300).values()))        # control: all defined
    assert set(full["pooled_share_ci95"]) == {"gauge", "off_orbit", *needs}
    assert "pooled_share_undefined" not in full and "pooled_share_ci95_component_reason" not in full


def test_the_bootstrap_counts_a_non_finite_value_per_key_and_never_feeds_it_to_a_percentile() -> None:
    info: dict = {}
    vals = iter([{"a": 1.0, "b": math.nan}, {"a": 2.0, "b": 5.0}] * 50)
    ci = cluster_bootstrap(lambda ix: next(vals), ["A", "B"], 100, 0, info=info)
    assert ci["a"] == (pytest.approx(1.0), pytest.approx(2.0)) and ci["b"] == (pytest.approx(5.0), pytest.approx(5.0))
    assert info["undefined_by_key"] == {"b": 50} and info["undefined_draws"] == 0
    info = {}
    ci = cluster_bootstrap(lambda ix: {"a": 1.0, "b": math.nan}, ["A", "B"], 20, 0, info=info)
    assert set(ci) == {"a"} and info["undefined_by_key"] == {"b": 20}                # no [nan, nan] interval


def test_an_all_zero_cohort_says_that_no_point_estimate_exists_and_gives_the_count() -> None:
    from alongray.pool import pool_rows

    z = np.linalg.norm(np.random.default_rng(5).normal(size=(12, 3)) + np.array([0.0, 0.0, 0.6]), axis=1)
    rows = [{"unit_id": u, "cluster": c, "z": z.tolist(), "a_r": z.tolist()} for u, c in (("u1", "A"), ("u2", "B"))]
    res = next(iter(pool_rows(rows, cluster_key="cluster", n_boot=50).values()))
    assert res["pooled_share"] is None and res["pooled_share_ci95_undefined_draws"] == 50
    assert "stands" not in res["pooled_share_ci95_reason"]
    assert "no point estimate exists either" in res["pooled_share_ci95_reason"]
