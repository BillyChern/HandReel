"""`fastdo.policy.eval_report`: the law floor (at least 20 clips AND at least 64 rollouts
per clip), per-clip outcomes whenever the law is not fitted, and pi_0 printed only with its bootstrap 95 % interval.
Pure functions over synthetic counts; no simulator."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import numpy as np
import pytest

from fastdo.policy.eval_report import (
    FITTED,
    MIN_LAW_CLIPS,
    MIN_LAW_ROLLOUTS,
    NOT_FITTED,
    UNIDENTIFIED,
    law_report,
    render_markdown,
)
from fastdo.policy.fallback_law import bootstrap_fallback_law
from fastdo.policy.report import wilson_interval

logger = logging.getLogger(__name__)

LAW_ONLY_WHEN_FITTED = ("alpha", "beta", "pi_0", "interval95", "smallest_k_excess_fitted", "smallest_k_excess_measured",
                        "smc_checks")
ALPHA, BETA, PI_0 = 0.6, 1.4, 0.02


def _ids(n: int) -> list[str]:
    return [f"clip{i:03d}" for i in range(n)]


def _row(method: str, k: int, clip_failures: list[int]) -> dict[str, Any]:
    """A rate row as `run/pipeline/eval_policy.measure_rates` writes it: one run per clip, per-clip outcomes kept."""
    n, failures = len(clip_failures), int(sum(clip_failures))
    low, high = wilson_interval(failures, n)
    return {"method": method, "k": k, "failures": failures, "n_clips": n, "rate": failures / n, "wilson95_low": low,
            "wilson95_high": high, "seconds_per_clip": 1.0, "clip_failures": clip_failures, "clip_trials": [1] * n}


def _probabilities(n_clips: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.where(rng.random(n_clips) < PI_0, 0.0, rng.beta(ALPHA, BETA, size=n_clips))


def _law(successes: np.ndarray, trials: np.ndarray, rows: list[dict[str, Any]] | None = None,
         **floors: int) -> dict[str, Any]:
    n = len(trials)
    rows = rows if rows is not None else [_row("best_of_k", 1, [0] * n), _row("smc", 4, [0] * n)]
    return law_report(np.asarray(successes), np.asarray(trials), rows, n_boot=5, seed=0, excess=0.05,
                      compute_induced=0.003, clip_ids=_ids(n), **floors)


def test_law_floor_is_20_clips_and_64_rollouts_per_clip() -> None:
    """150 clips with 2, 3 or 4 rollouts each (uniform, and mixed per clip), 150 clips x 8
    rollouts, 150 x 64 with one clip at 63 and 19 clips x 64 are not fitted, with the reason, per-clip outcomes and
    `meets_target` None; 20 clips x 64 rollouts is fitted, with no per-clip section. Configured minimums below the floors
    raise."""
    assert (MIN_LAW_CLIPS, MIN_LAW_ROLLOUTS) == (20, 64)
    rng = np.random.default_rng(1)
    p150 = _probabilities(150, seed=0)
    one_short = np.full(150, 64)
    one_short[17] = 63
    cases = {"2": np.full(150, 2), "3": np.full(150, 3), "4": np.full(150, 4), "mixed 2/3/4": rng.integers(2, 5, 150),
             "8": np.full(150, 8), "64 with one clip at 63": one_short}
    for name, trials in cases.items():
        law = _law(rng.binomial(trials, p150), trials)
        assert law["status"] == NOT_FITTED and law["reason"] == "not fitted (rollouts_per_clip < 64)", name
        assert law["meets_target"] is None and not set(LAW_ONLY_WHEN_FITTED) & set(law), name
        assert law["rollouts_per_clip"] == int(trials.min()) and len(law["per_clip"]) == law["n_clips"] == 150, name
    p20 = _probabilities(20, seed=2)
    successes20 = np.random.default_rng(3).binomial(64, p20)
    short = _law(successes20[:19], np.full(19, 64))
    assert short["reason"] == "not fitted (n_clips < 20)" and short["meets_target"] is None
    fitted = _law(successes20, np.full(20, 64))
    assert fitted["status"] == FITTED and (fitted["n_clips"], fitted["rollouts_per_clip"]) == (20, 64)
    assert set(fitted["meets_target"]) == {"4"} and "per_clip" not in fitted
    for bad in ({"min_clips": MIN_LAW_CLIPS - 1}, {"min_rollouts": MIN_LAW_ROLLOUTS - 1}):
        with pytest.raises(ValueError, match="the law's floors"):
            _law(successes20, np.full(20, 64), **bad)


def test_law_floor_counts_distinct_clip_ids() -> None:
    """The 20-clip floor counts distinct clip ids. 20 copies of one clip id x 64
    rollouts stay unfitted (reason names the distinct count, `meets_target` None, per-clip outcomes for all 20 rows),
    while the same counts under 20 distinct ids are fitted."""
    successes = np.random.default_rng(3).binomial(64, _probabilities(20, seed=2))
    rows = [_row("best_of_k", 1, [0] * 20), _row("smc", 4, [0] * 20)]
    copies = law_report(successes, np.full(20, 64), rows, n_boot=5, seed=0, excess=0.05, compute_induced=0.003,
                        clip_ids=["clip000"] * 20)
    assert copies["status"] == NOT_FITTED and copies["meets_target"] is None
    assert copies["reason"] == "not fitted (distinct clip ids 1 < 20)" and copies["n_distinct_clips"] == 1
    assert copies["n_clips"] == 20 and len(copies["per_clip"]) == 20 and not set(LAW_ONLY_WHEN_FITTED) & set(copies)
    distinct = _law(successes, np.full(20, 64))
    assert distinct["status"] == FITTED and distinct["n_distinct_clips"] == 20


def test_a6_pending_row_names_the_bc_target_and_velocity_rescaling() -> None:
    """In the report, the BC warm-start pending row says that it also measures the point-sampled BC target and the
    50 Hz velocity rescaling."""
    from fastdo.policy.eval_report import pending_rows

    a6 = next(r for r in pending_rows() if r["item"] == "bc_warm_start")
    assert "point-sampled" in a6["what"] and "50 Hz velocity rescaling" in a6["what"] and "also measures" in a6["what"]
    assert a6["what"] in render_markdown({"pending": pending_rows()})


def test_single_clip_and_unidentified_laws_report_no_target() -> None:
    """One clip with 200 of 256 successes (alpha at its bound and a 'met' target) is not fitted (n_clips < 20);
    120 clips that all succeed 200 of 256 times clear the floor but carry no overdispersion, so alpha lands on its bound:
    unidentified. Neither reports parameters, intervals, a smallest K or an f_SMC - pi_0 check; `meets_target` is None
    (null in JSON), both carry per-clip outcomes and the Markdown names the status."""
    one = _law(np.array([200]), np.array([256]))
    assert one["status"] == NOT_FITTED and one["reason"] == "not fitted (n_clips < 20)" and one["min_rollouts"] == 64
    flat = _law(np.full(120, 200), np.full(120, 256), min_clips=100)
    assert flat["status"] == UNIDENTIFIED and "parameter bound" in flat["reason"]
    for law in (one, flat):
        assert not set(LAW_ONLY_WHEN_FITTED) & set(law) and law["meets_target"] is None
        assert json.loads(json.dumps(law))["meets_target"] is None and law["rollouts_per_clip"] == 256
        assert len(law["per_clip"]) == law["n_clips"] and law["per_clip"][0]["successes"] == 200
        assert f"Fitted law: {law['status']}" in render_markdown({"law": law})


def test_below_the_floor_the_report_writes_per_clip_outcomes() -> None:
    """5 clips x 8 law rollouts; best-of-K at K 1 and 4, SMC and its no-floor ablation at K 4,
    each run once per clip. The unfitted law holds, per clip in clip order, its law successes / trials and, for every
    (method, K) row in row order, the clip's successes / trials with the Wilson 95 % interval of its fallback rate. The
    Markdown prints both tables and no pi_0 value. Rows without per-clip outcomes, or for another clip count, are
    refused."""
    successes, trials = np.array([0, 3, 8, 5, 1]), np.full(5, 8)
    rows = [_row("best_of_k", 1, [1, 1, 0, 0, 1]), _row("best_of_k", 4, [1, 0, 0, 0, 0]),
            _row("smc", 4, [1, 0, 0, 0, 1]), _row("smc_no_floor", 4, [1, 1, 0, 0, 0])]
    law = _law(successes, trials, rows)
    assert law["status"] == NOT_FITTED and law["reason"] == "not fitted (n_clips < 20; rollouts_per_clip < 64)"
    per_clip = law["per_clip"]
    assert [c["clip_id"] for c in per_clip] == _ids(5)
    assert [(c["successes"], c["trials"]) for c in per_clip] == [(0, 8), (3, 8), (8, 8), (5, 8), (1, 8)]
    for i, clip in enumerate(per_clip):
        assert [(r["method"], r["k"]) for r in clip["rates"]] == [(r["method"], r["k"]) for r in rows]
        for entry, row in zip(clip["rates"], rows, strict=True):
            failed = row["clip_failures"][i]
            assert (entry["failures"], entry["successes"], entry["trials"], entry["rate"]) == (failed, 1 - failed, 1,
                                                                                               failed)
            assert (entry["wilson95_low"], entry["wilson95_high"]) == wilson_interval(failed, 1)
    assert json.loads(json.dumps(law))["per_clip"] == per_clip
    markdown = render_markdown({"law": law})
    assert "| clip001 | 3 / 8 |" in markdown
    assert "| clip004 | smc | 4 | 0 / 1 | 1.0000 | [0.2065, 1.0000] |" in markdown
    assert "| clip002 | best_of_k | 1 | 1 / 1 | 0.0000 | [0.0000, 0.7935] |" in markdown
    assert not re.search(r"pi_0 \d", markdown)
    bare = {k: v for k, v in rows[0].items() if k not in ("clip_failures", "clip_trials")}
    with pytest.raises(ValueError, match="per-clip"):
        _law(successes, trials, [bare])
    with pytest.raises(ValueError, match="per-clip"):
        _law(successes, trials, [_row("smc", 4, [0, 0, 0])])


def test_pi_0_is_printed_only_with_its_bootstrap_interval(caplog: pytest.LogCaptureFixture) -> None:
    """In a fitted law's Markdown every printed pi_0 value carries its bootstrap 95 % interval (4 decimals),
    and the bootstrap's own log line prints pi_0 with its 95 % percentile interval too."""
    trials = np.full(40, 64)
    successes = np.random.default_rng(5).binomial(trials, _probabilities(40, seed=4))
    law = _law(successes, trials)
    assert law["status"] == FITTED
    markdown = render_markdown({"law": law})
    printed = re.findall(r"pi_0 \d", markdown)
    with_interval = re.findall(r"pi_0 (\d\.\d{4}) \[(\d\.\d{4}), (\d\.\d{4})\]", markdown)
    low, high = law["interval95"]["pi_0"]
    assert len(printed) == len(with_interval) == 1
    assert with_interval[0] == (f"{law['pi_0']:.4f}", f"{low:.4f}", f"{high:.4f}")
    with caplog.at_level(logging.INFO, logger="fastdo.policy.fallback_law"):
        fit = bootstrap_fallback_law(successes, trials, n_boot=10, seed=0)
    lines = [r.getMessage() for r in caplog.records if "pi_0" in r.getMessage()]
    fit_low, fit_high = fit.interval(0.95)[2]
    assert lines and all(re.search(r"pi_0 \d\.\d{4} \[\d\.\d{4}, \d\.\d{4}\]", line) for line in lines)
    assert f"pi_0 {fit.pi_0:.4f} [{fit_low:.4f}, {fit_high:.4f}]" in lines[-1]
    logger.info("fitted law line: %s", next(line for line in markdown.splitlines() if "pi_0" in line))
