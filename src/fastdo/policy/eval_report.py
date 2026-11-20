"""Report assembly of `run/pipeline/eval_policy.py`.
It lives in the package, not next to the script, so the script runs as documented.

Pure functions over measured numbers, tested without a simulator:

- `law_report`: the fitted zero-inflated law with 95 % clip-bootstrap intervals, the smallest K with f_BoK - pi_0 <=
  `excess` (fitted and measured) and, per SMC K, whether f_SMC - pi_0 <= `compute_induced` (`smc_checks`, and
  `meets_target` by K for the throughput method `smc`). The law is fitted only from at least `MIN_LAW_CLIPS` (20)
  distinct clips AND at least `MIN_LAW_ROLLOUTS` (64) rollouts per clip; configured minimums may only be stricter. The
  design analysis gives pi_0 an sd of 0.021 at 300 x 64 and 0.058 at 300 x 8, so pi_0 is never quoted from 8 rollouts.
  Below either floor the status is `not fitted` with the reason `not fitted (n_clips < 20)` and/or
  `not fitted (rollouts_per_clip < 64)`; a fit with alpha or beta at the optimizer's bound
  (`FallbackLawFit.at_parameter_bound`) is `unidentified`. Neither carries parameters, intervals, a smallest K or target
  checks, and `meets_target` is None, never True. Both carry `per_clip` (`per_clip_outcomes`): per clip its law
  successes / trials and, for every (method, K) rate row, the clip's successes / trials with the Wilson 95 % interval of
  its fallback rate (`clip_failures` / `clip_trials` of the rows). The pooled f_BoK(K) and f_SMC(K) rows with Wilson
  intervals stay in the report either way. A fitted law prints pi_0 only together with its bootstrap 95 % interval.
- `throughput_rows`: E[t] and accepted clips per GPU-hour for the throughput and quality-parity configurations.
- `pending_rows`, `render_markdown`, `write_report`: the pending items and the `report.json` / `report.md` pair. The
  Markdown opens with the evaluation set's provenance (held-out, or the training clip, labelled as such) and the timing
  protocol (environment construction and warm-up excluded from per-clip seconds, reported separately).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from fastdo.policy.fallback_law import bootstrap_fallback_law
from fastdo.policy.report import (
    RateEstimate,
    accepted_clips_per_gpu_hour,
    expected_clip_seconds,
    smallest_k_excess,
    smallest_measured_k,
    wilson_interval,
)

__all__ = ["FITTED", "MIN_LAW_CLIPS", "MIN_LAW_ROLLOUTS", "NOT_FITTED", "PENDING", "PENDING_ROWS", "UNIDENTIFIED",
           "law_report", "pending_rows", "per_clip_outcomes", "render_markdown", "throughput_rows", "write_report"]

PENDING = "pending"
FITTED = "fitted"
NOT_FITTED = "not fitted"
UNIDENTIFIED = "unidentified"
MIN_LAW_CLIPS = 20             # distinct clips below which the law is not fitted
MIN_LAW_ROLLOUTS = 64          # rollouts per clip below which the law is not fitted
PENDING_ROWS = (
    ("bc_warm_start", ("whether BC shortens PPO: steps to 50 % success, 3 seeds with and without BC; this row also "
                       "measures the point-sampled BC target (the MPC command at the start of the next control period) "
                       "and the 50 Hz velocity rescaling"), "GPU training"),
    ("held_out_policy", "f_BoK, f_SMC, fitted law and throughput on held-out clips of a trained policy",
     "GPU training + eval"),
    ("head_to_head", ("head-to-head through the physics isolation protocol; MPC arms on the baseline's 28.4 Hz "
                      "align_corners grid (35.2 ms per frame), the policy on the environment's 50 Hz clock"), "GPU"),
    ("clearance_sweep", "clearance band 2 / 3 / 5 mm through the physics isolation protocol on the whisking demo",
     "GPU"),
)


def _estimates(rows: Sequence[dict[str, Any]], method: str) -> list[RateEstimate]:
    return [RateEstimate(method=r["method"], k=r["k"], failures=r["failures"], n_clips=r["n_clips"]) for r in rows
            if r["method"] == method]


def per_clip_outcomes(clip_ids: Sequence[str], successes: np.ndarray, trials: np.ndarray,
                      rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per clip, in clip order: its law rollouts' `successes` / `trials` and, for every rate row in row order, the
    clip's `successes` / `trials` of that (method, K) with its fallback `rate` and Wilson 95 % interval.

    Raises:
        ValueError: If a row lacks per-clip outcomes (`clip_failures`, `clip_trials`) for exactly these clips.
    """
    n_clips = len(clip_ids)
    for r in rows:
        if len(r.get("clip_failures", ())) != n_clips or len(r.get("clip_trials", ())) != n_clips:
            raise ValueError(f"rate row {r['method']} K={r['k']} needs per-clip outcomes (clip_failures, clip_trials) "
                             f"for the law's {n_clips} clips")
    out = []
    for i, clip_id in enumerate(clip_ids):
        rates = []
        for r in rows:
            failed, n = int(r["clip_failures"][i]), int(r["clip_trials"][i])
            low, high = wilson_interval(failed, n)
            rates.append({"method": r["method"], "k": r["k"], "successes": n - failed, "failures": failed, "trials": n,
                          "rate": failed / n, "wilson95_low": low, "wilson95_high": high})
        out.append({"clip_id": str(clip_id), "successes": int(successes[i]), "trials": int(trials[i]), "rates": rates})
    return out


def law_report(successes: np.ndarray, trials: np.ndarray, rows: Sequence[dict[str, Any]], n_boot: int, seed: int,
               excess: float, compute_induced: float, min_clips: int = MIN_LAW_CLIPS,
               min_rollouts: int = MIN_LAW_ROLLOUTS, *, clip_ids: Sequence[str]) -> dict[str, Any]:
    """The law section of the report (module docstring) for the clips `clip_ids` (the order of `successes`): `status`
    FITTED with the parameters, their 95 % bootstrap intervals, the smallest K for `excess`, the f_SMC - pi_0 checks and
    `meets_target`; or `status` NOT_FITTED / UNIDENTIFIED with a `reason`, `meets_target` None, `per_clip`
    (`per_clip_outcomes`) and none of the rest. `rollouts_per_clip` is the fewest trials of any clip, the number the
    gate uses.

    Raises:
        ValueError: If `min_clips < MIN_LAW_CLIPS` or `min_rollouts < MIN_LAW_ROLLOUTS` (the floors), the
            counts disagree in shape or with `clip_ids`, or (unfitted) as `per_clip_outcomes`.
    """
    if min_clips < MIN_LAW_CLIPS or min_rollouts < MIN_LAW_ROLLOUTS:
        raise ValueError(f"min_clips >= {MIN_LAW_CLIPS} and min_rollouts >= {MIN_LAW_ROLLOUTS} (the law's floors), got "
                         f"{min_clips}, {min_rollouts}")
    successes, trials = np.asarray(successes), np.asarray(trials)
    if successes.shape != trials.shape or successes.ndim != 1 or len(clip_ids) != successes.shape[0]:
        raise ValueError(f"successes {successes.shape}, trials {trials.shape} and {len(clip_ids)} clip ids must "
                         "describe the same (n_clips,) clips")
    n_clips, n_distinct = int(successes.shape[0]), len({str(clip_id) for clip_id in clip_ids})
    head = {"n_clips": n_clips, "n_distinct_clips": n_distinct, "rollouts_per_clip": int(np.min(trials)) if n_clips else 0,
            "min_clips": min_clips, "min_rollouts": min_rollouts, "n_boot": n_boot, "excess_target": excess,
            "compute_induced_target": compute_induced}
    if n_clips < min_clips:
        short = [f"n_clips < {min_clips}"]
    else:            # the floor counts distinct clips: repeated ids of one clip are not independent clips
        short = [f"distinct clip ids {n_distinct} < {min_clips}"] if n_distinct < min_clips else []
    short += [f"rollouts_per_clip < {min_rollouts}"] if head["rollouts_per_clip"] < min_rollouts else []
    if short:
        return {**head, "status": NOT_FITTED, "reason": f"{NOT_FITTED} ({'; '.join(short)})", "meets_target": None,
                "per_clip": per_clip_outcomes(clip_ids, successes, trials, rows)}
    fit = bootstrap_fallback_law(successes, trials, n_boot=n_boot, seed=seed)
    if fit.at_parameter_bound:
        return {**head, "status": UNIDENTIFIED, "meets_target": None,
                "reason": f"alpha {fit.alpha:.4g} or beta {fit.beta:.4g} at the parameter bound e^(+-10): the clips "
                          "carry no overdispersion that identifies them",
                "per_clip": per_clip_outcomes(clip_ids, successes, trials, rows)}
    interval = fit.interval(0.95)
    smc_checks = [{"method": r["method"], "k": r["k"], "f_smc_minus_pi_0": r["rate"] - fit.pi_0,
                   "meets_target": r["rate"] - fit.pi_0 <= compute_induced}
                  for r in rows if str(r["method"]).startswith("smc")]
    return {**head, "status": FITTED, "alpha": fit.alpha, "beta": fit.beta, "pi_0": fit.pi_0,
            "interval95": {name: interval[i].tolist() for i, name in enumerate(("alpha", "beta", "pi_0"))},
            "smallest_k_excess_fitted": smallest_k_excess(fit.alpha, fit.beta, fit.pi_0, excess),
            "smallest_k_excess_measured": smallest_measured_k(_estimates(rows, "best_of_k"), fit.pi_0, excess),
            "smc_checks": smc_checks,
            "meets_target": {str(c["k"]): c["meets_target"] for c in smc_checks if c["method"] == "smc"}}


def throughput_rows(quality: dict[str, Any], rows: Sequence[dict[str, Any]], t_rest: float | None,
                    t_mpc: float) -> list[dict[str, Any]]:
    """E[t] and accepted clips per GPU-hour per budget Section 2 (amended); `t_rest` None reports the retargeting
    stage alone, marked pending for the end-to-end numbers."""
    rest = 0.0 if t_rest is None else t_rest
    scope = "retargeting stage only (t_rest pending)" if t_rest is None else "end to end"
    out = []
    for r in rows:
        if r["method"] != "smc":
            continue
        e_t = expected_clip_seconds(rest, r["seconds_per_clip"])
        out.append({"configuration": f"throughput: twisted SMC K={r['k']}", "scope": scope, "expected_seconds": e_t,
                    "yield": 1.0 - r["rate"], "accepted_clips_per_gpu_hour": accepted_clips_per_gpu_hour(e_t, r["rate"])})
    f_1 = quality["fallback_fraction"]
    e_t = expected_clip_seconds(rest, quality["seconds_per_clip"], f_1, t_mpc)
    out.append({"configuration": "quality parity: K=1 policy + MPC fallback", "scope": scope, "expected_seconds": e_t,
                "yield": 1.0, "accepted_clips_per_gpu_hour": accepted_clips_per_gpu_hour(e_t, 0.0),
                "note": "assumes the MPC fallback succeeds; its own failure rate comes from the physics isolation "
                        "protocol"})
    return out


def pending_rows() -> list[dict[str, str]]:
    return [{"item": item, "what": what, "status": PENDING, "needs": needs} for item, what, needs in PENDING_ROWS]


def _law_line(law: dict[str, Any] | None) -> str:
    if not law:
        return "Fitted law: " + PENDING
    if law.get("status") != FITTED:
        return (f"Fitted law: {law['status']} ({law['reason']}; {law['n_clips']} clips x {law['rollouts_per_clip']} "
                "rollouts); no parameters, smallest K or f_SMC - pi_0 check reported (meets_target None)")
    alpha, beta, pi_0 = (law["interval95"][name] for name in ("alpha", "beta", "pi_0"))
    return (f"Fitted law: alpha {law['alpha']:.3f} [{alpha[0]:.3f}, {alpha[1]:.3f}], beta {law['beta']:.3f} "
            f"[{beta[0]:.3f}, {beta[1]:.3f}], pi_0 {law['pi_0']:.4f} [{pi_0[0]:.4f}, {pi_0[1]:.4f}] ({law['n_clips']} "
            f"clips x {law['rollouts_per_clip']} rollouts; bootstrap 95 % intervals); smallest K with f_BoK - pi_0 <= "
            f"{law['excess_target']}: fitted {law['smallest_k_excess_fitted']}, measured "
            f"{law['smallest_k_excess_measured']}")


def _per_clip_lines(law: dict[str, Any] | None) -> list[str]:
    """The unfitted law's per-clip tables (module docstring); none for a fitted or absent law."""
    if not law or "per_clip" not in law:
        return []
    lines = ["", "Per-clip outcomes (no fitted law): the law rollouts of each clip, then each (method, K) run on it.", "",
             "| clip | law rollouts: successes / trials |", "|---|---|"]
    lines += [f"| {c['clip_id']} | {c['successes']} / {c['trials']} |" for c in law["per_clip"]]
    lines += ["", "| clip | method | K | successes / trials | f | Wilson 95 % |", "|---|---|---|---|---|---|"]
    lines += [f"| {c['clip_id']} | {r['method']} | {r['k']} | {r['successes']} / {r['trials']} | {r['rate']:.4f} | "
              f"[{r['wilson95_low']:.4f}, {r['wilson95_high']:.4f}] |" for c in law["per_clip"] for r in c["rates"]]
    return lines


def render_markdown(report: dict[str, Any]) -> str:
    """The report as Markdown tables (numbers as measured; `pending` where not measured)."""
    lines = ["# Tracking-policy evaluation", ""]
    eval_set = report.get("evaluation_set")
    if eval_set:
        lines += [(f"Evaluation set: {eval_set['n_clips']} clip(s) from `{eval_set['source']}` ({eval_set['split']} "
                   f"split; objects {', '.join(eval_set['objects'])}): **{eval_set['note']}**. Held-out evaluation: "
                   f"{eval_set.get('held_out_status', PENDING)}."), ""]
    sigma = report.get("sigma_v_calibration")
    if sigma:
        lines += [f"Tempering floor: sigma_V {sigma['sigma_v']:.4g} ({sigma['clips']}; {sigma['note']}).", ""]
    timing = report.get("timing")
    if timing:
        lines += [(f"Timing: {timing['note']}. Construction seconds by K: {timing['construction_seconds']}; warm-up "
                   f"seconds by K: {timing['warmup_seconds']}."), ""]
    quality = report.get("quality")
    if quality:
        summary = (f"Fallback fraction (K = 1): {quality['fallback_fraction']:.4f} over {quality['n_clips']} clips; "
                   f"{quality['seconds_per_clip']:.3f} s per clip; {quality['clips_per_gpu_hour']:.1f} clips per "
                   "GPU-hour (retargeting stage).")
        lines += [summary, "", "| object | clips | success rate |", "|---|---|---|"]
        lines += [f"| {name} | {int(r['clips'])} | {r['success_rate']:.3f} |" for name, r in quality["per_object"].items()]
    lines += ["", "| method | K | f | Wilson 95 % | clips | s per clip |", "|---|---|---|---|---|---|"]
    lines += [f"| {r['method']} | {r['k']} | {r['rate']:.4f} | [{r['wilson95_low']:.4f}, {r['wilson95_high']:.4f}] | "
              f"{r['n_clips']} | {r['seconds_per_clip']:.3f} |" for r in report.get("rates", [])]
    lines += ["", _law_line(report.get("law")), *_per_clip_lines(report.get("law"))]
    lines += ["", "| configuration | scope | E[t] s | yield | accepted clips / GPU-h |", "|---|---|---|---|---|"]
    lines += [f"| {r['configuration']} | {r['scope']} | {r['expected_seconds']:.3f} | {r['yield']:.4f} | "
              f"{r['accepted_clips_per_gpu_hour']:.1f} |" for r in report.get("throughput", [])]
    lines += ["", "| item | what | status | needs |", "|---|---|---|---|"]
    lines += [f"| {r['item']} | {r['what']} | {r['status']} | {r['needs']} |" for r in report.get("pending", [])]
    return "\n".join(lines) + "\n"


def write_report(report: dict[str, Any], out_dir: str | Path) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path, md_path = out / "report.json", out / "report.md"
    json_path.write_text(json.dumps(report, indent=2, default=lambda v: None))
    md_path.write_text(render_markdown(report))
    return json_path, md_path
