"""Evaluate a trained tracking policy.

  uv run python run/pipeline/eval_policy.py +policy=default policy.run=whisking_gpu

Evaluation set (`evaluation_set`). `policy.data.source=dexycb` evaluates the held-out side of the object split, objects
never trained on (references from `fastdo.policy.dexycb_refs`, each clip in its object's scene). `whisking` evaluates the demo's
single clip, which is the clip the policy trains on: the report labels it "training clip, not held-out", so no number
from it is a held-out measurement. Per-object success is keyed by each clip's object.

Writes `report.json` and `report.md` to `policy.eval.report_dir`:

1. Policy quality at K = 1 (mean action, no noise): per-object success (`clip_success`, the baseline criterion), the
   fallback fraction (clips whose trajectory fails and would go to a fallback), retargeting-stage seconds per clip and
   that stage's clips per GPU-hour.
2. Fallback rates measured directly on the clips: f_BoK(K) for K in {1, 4, 16, 64, 256} and f_SMC(K) for
   K in {4, 16, 64} with Wilson 95 % intervals, plus the required ablations of f_SMC (chunk-0 start potential on;
   tempering floor c = 0).
3. The fitted law (alpha, beta, pi_0) from `law_rollouts` independent rollouts per clip, with percentile intervals from
   `n_boot` clip bootstrap resamples (pi_0 is printed only with its 95 % interval), the smallest K with f_BoK - pi_0 <=
   5 % (fitted law, and the measured grid) and for each SMC K whether f_SMC - pi_0 <= 0.3 %, only when the law is
   identified (`fastdo.policy.eval_report.law_report`: at least `law_min_clips` >= 20 clips AND
   `law_min_rollouts` >= 64 rollouts per clip, no parameter at its bound). Otherwise: the reason, `meets_target` None,
   none of these, and per clip its law successes / trials plus, for every (method, K), the clip's successes / trials
   with a Wilson 95 % interval. The default 256 rollouts per clip clear the 64-rollout floor.
4. Throughput: E[t] and accepted clips per GPU-hour for the throughput configuration
   (twisted SMC at K, no fallback term, rejected clips count against yield) and the quality-parity configuration (the
   MPC fallback of `t_mpc_seconds` on the K = 1 failures), with `t_rest_seconds` pending until it is measured.
5. Pending rows for what needs GPU training or runs not made yet, never a number.

Timing (`EnvPool`). Every environment (one per particle count, in one pool per robot scene: the
DexYCB clips of each held-out object run in that object's scene, `scene_fn`) is built, then warmed up with one untimed
best-of-K rollout on its scene's first clip (plus one SMC pass where SMC runs on it), before any timed rollout.
Construction and warm-up seconds are reported separately under `timing` (summed over scenes by K, and per scene);
per-clip seconds, clips per GPU-hour and E[t] come only from the timed rollouts after warm-up.

sigma_V (`sigma_v_calibration`). With `calibrate_sigma_v` the tempering floor is calibrated on the evaluation clips
themselves (pooled over every scene's clips), before the SMC rates are measured on the same clips. Held-out rollouts
would be preferable; here the one scalar is tuned on the test set, and the report says so. A separate calibration split
needs more held-out clips than exist.

The measurement functions take the rollout as a callable (tested with injected policies); `main_from_config` wires
them to the simulator entry points of an `EvalBackend` (default `make_env`, `run_best_of_k`, `run_smc`,
`calibrate_sigma_v_scenes`; tests inject fakes) on the pools' environments.
"""

from __future__ import annotations

import dataclasses
import functools
import logging
import math
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from fastdo.bench.gpu_budget import apply_cpu_budget, apply_gpu_budget
from fastdo.policy.checkpoint import load_checkpoint
from fastdo.policy.data import ReferenceClip as EvalClip
from fastdo.policy.data import ReferenceFn
from fastdo.policy.env import BatchedTrackingEnv, make_env
from fastdo.policy.eval_report import law_report, pending_rows, throughput_rows, write_report
from fastdo.policy.networks import ActorCritic
from fastdo.policy.pipeline_config import (
    clip_scene_xml,
    env_config,
    load_reference_clips,
    ppo_config,
)
from fastdo.policy.report import rate_estimate
from fastdo.policy.smc import SmcConfig, run_best_of_k, run_smc
from fastdo.policy.smc_calibration import (
    calibrate_sigma_v_scenes,
    with_calibrated_floor,
    write_sigma_v,
)

logger = logging.getLogger(__name__)

TRAINING_CLIP_NOTE = "training clip, not held-out"
HELD_OUT_UNAVAILABLE = "unavailable (pending DexYCB references)"
SIGMA_V_ON_EVAL_NOTE = ("calibrated on the evaluation clips, which the SMC rates are then measured on; design 9b(d) "
                        "asks for held-out rollouts, so this one scalar is tuned on the test set")


@dataclass(frozen=True)
class EvalBackend:
    """The simulator entry points `main_from_config` calls, with the signatures of the defaults (`calibrate` takes one
    `(environment, references)` group per scene)."""

    make_env: Callable[..., Any] = make_env
    best_of_k: Callable[..., Any] = run_best_of_k
    smc: Callable[..., Any] = run_smc
    calibrate: Callable[..., tuple[float, int]] = calibrate_sigma_v_scenes


@dataclass(frozen=True)
class RolloutOutcome:
    success: bool
    seconds: float


RolloutFn = Callable[[EvalClip], RolloutOutcome]
RunFn = Callable[[EvalClip, str, int], RolloutOutcome]
CountFn = Callable[[EvalClip, int], int]
Timer = Callable[[Callable[[], Any]], tuple[Any, float]]


def evaluate_quality(clips: Sequence[EvalClip], rollout: RolloutFn) -> dict[str, Any]:
    """Per-object success, fallback fraction, seconds per clip and stage clips per GPU-hour of one rollout per clip.

    Raises:
        ValueError: Without clips.
    """
    if not clips:
        raise ValueError("evaluate_quality needs at least one clip")
    outcomes = [rollout(clip) for clip in clips]
    per_object: dict[str, dict[str, float]] = {}
    for clip, outcome in zip(clips, outcomes, strict=True):
        row = per_object.setdefault(clip.object_name, {"clips": 0.0, "successes": 0.0})
        row["clips"] += 1.0
        row["successes"] += float(outcome.success)
    for row in per_object.values():
        row["success_rate"] = row["successes"] / row["clips"]
    seconds = float(np.mean([o.seconds for o in outcomes]))
    failed = [clip.clip_id for clip, o in zip(clips, outcomes, strict=True) if not o.success]
    return {"n_clips": len(clips), "fallback_fraction": len(failed) / len(clips), "failed_clips": failed,
            "per_object": per_object, "seconds_per_clip": seconds,
            "clips_per_gpu_hour": 3600.0 / seconds if seconds > 0.0 else math.inf}


def measure_rates(clips: Sequence[EvalClip], run: RunFn, plan: Sequence[tuple[str, Sequence[int]]]
                  ) -> list[dict[str, Any]]:
    """`RateEstimate` rows (with `seconds_per_clip`) for every (method, K) of `plan`, one run per clip, with the
    per-clip outcomes in clip order (`clip_failures`, `clip_trials`) that the unfitted law's per-clip section reads."""
    rows = []
    for method, ks in plan:
        for k in ks:
            outcomes = [run(clip, method, int(k)) for clip in clips]
            estimate = rate_estimate(method, int(k), [not o.success for o in outcomes])
            rows.append({**estimate.as_dict(), "seconds_per_clip": float(np.mean([o.seconds for o in outcomes])),
                         "clip_failures": [int(not o.success) for o in outcomes], "clip_trials": [1] * len(outcomes)})
            logger.info("%s K=%d: f %.4f [%.4f, %.4f] over %d clips", method, k, estimate.rate, *estimate.interval,
                        estimate.n_clips)
    return rows


def law_counts(clips: Sequence[EvalClip], count: CountFn, n_rollouts: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-clip successes out of `n_rollouts` independent rollouts, and the trials."""
    successes = np.array([count(clip, n_rollouts) for clip in clips], dtype=np.int64)
    return successes, np.full(len(clips), n_rollouts, dtype=np.int64)


class EnvPool[E]:
    """Environments by world count for timed evaluation (module docstring, "Timing").

    `build(ks)` constructs each missing environment under `timer`; `warm_up(fn)` runs `fn(k, env)` once per environment
    under `timer`. `pool[k]` is a built environment for untimed work (calibration, law counts) and `pool.warmed(k)` one
    that is also warmed up, for timed rollouts. Neither constructs: a K not built (or not warmed up) beforehand raises.
    """

    def __init__(self, make: Callable[[int], E], timer: Timer) -> None:
        self._make, self._timer = make, timer
        self._envs: dict[int, E] = {}
        self.construction_seconds: dict[int, float] = {}
        self.warmup_seconds: dict[int, float] = {}

    def build(self, ks: Iterable[int]) -> None:
        for k in sorted({int(k) for k in ks}):
            if k not in self._envs:
                self._envs[k], self.construction_seconds[k] = self._timer(functools.partial(self._make, k))
                logger.info("built the %d-world environment in %.3f s", k, self.construction_seconds[k])

    def warm_up(self, fn: Callable[[int, E], Any]) -> None:
        for k, env in self._envs.items():
            _, self.warmup_seconds[k] = self._timer(functools.partial(fn, k, env))
            logger.info("warmed up the %d-world environment in %.3f s", k, self.warmup_seconds[k])

    def __getitem__(self, k: int) -> E:
        if k not in self._envs:
            raise RuntimeError(f"no {k}-world environment was built before evaluation (EnvPool.build)")
        return self._envs[k]

    def warmed(self, k: int) -> E:
        env = self[k]
        if k not in self.warmup_seconds:
            raise RuntimeError(f"the {k}-world environment was not warmed up before timing (EnvPool.warm_up)")
        return env

    def timing(self) -> dict[str, Any]:
        return {"construction_seconds": {str(k): s for k, s in sorted(self.construction_seconds.items())},
                "warmup_seconds": {str(k): s for k, s in sorted(self.warmup_seconds.items())},
                "note": "per-clip seconds exclude environment construction and one untimed warm-up per environment"}


def evaluation_set(cfg: DictConfig, reference_fn: ReferenceFn | None = None, reader: Any = None
                   ) -> tuple[list[EvalClip], dict[str, Any]]:
    """The clips to evaluate and their provenance for the report (module docstring, "Evaluation set").

    Raises:
        ValueError: If the set is empty, or as `fastdo.policy.pipeline_config.load_reference_clips` (a DexYCB set without
            `reference_fn`, an unknown source).
    """
    source = str(cfg.policy.data.source)
    if source == "whisking":
        clips = load_reference_clips(cfg, "train")
        split, held_out, held_out_status = "train", False, HELD_OUT_UNAVAILABLE
        note = f"{TRAINING_CLIP_NOTE}: the policy trains on this clip, so no number here measures generalization"
    else:
        clips = load_reference_clips(cfg, "held_out", reference_fn, reader)
        split, held_out, held_out_status = "held_out", True, "evaluated (object-level split)"
        note = "held-out objects, never trained on (object-level split)"
    if not clips:
        raise ValueError(f"no {split} clips for policy.data.source={source}")
    return clips, {"source": source, "split": split, "held_out": held_out, "held_out_status": held_out_status,
                   "n_clips": len(clips), "objects": sorted({clip.object_name for clip in clips}), "note": note}


def _latest_checkpoint(cfg: DictConfig) -> Path:
    if cfg.policy.eval.checkpoint is not None:
        return Path(str(cfg.policy.eval.checkpoint))
    found = sorted((Path(str(cfg.paths.outputs)) / "policy" / str(cfg.policy.run) / "checkpoints").glob("update_*.pt"))
    if not found:
        raise FileNotFoundError(f"no checkpoint for run {cfg.policy.run}")
    return found[-1]


def _timed(device: torch.device, fn: Callable[[], Any]) -> tuple[Any, float]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    out = fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return out, time.perf_counter() - start


def pooled_timing(pools: dict[str, EnvPool[Any]]) -> dict[str, Any]:
    """Construction and warm-up seconds summed over the scenes' pools by K, plus each scene's own `EnvPool.timing`."""
    per_scene = {scene: pool.timing() for scene, pool in pools.items()}
    out: dict[str, Any] = {}
    for key in ("construction_seconds", "warmup_seconds"):
        sums: dict[str, float] = {}
        for timing in per_scene.values():
            for k, seconds in timing[key].items():
                sums[k] = sums.get(k, 0.0) + seconds
        out[key] = dict(sorted(sums.items(), key=lambda item: int(item[0])))
    return {**out, "per_scene": per_scene,
            "note": "per-clip seconds exclude environment construction and one untimed warm-up per environment (one pool "
                    "per robot scene; seconds summed over scenes by K)"}


def main_from_config(cfg: DictConfig, reference_fn: ReferenceFn | None = None, *, reader: Any = None,
                     backend: EvalBackend | None = None, scene_fn: Callable[[EvalClip], str] | None = None
                     ) -> dict[str, Any]:
    """Measure and write the report of the module docstring for `policy.run`'s checkpoint; `reader` goes to
    `evaluation_set`, `backend` (default `EvalBackend()`) supplies the simulator and `scene_fn` (default
    `fastdo.policy.pipeline_config.clip_scene_xml`) names each clip's robot scene."""
    p, e, sim = cfg.policy, cfg.policy.eval, backend if backend is not None else EvalBackend()
    checkpoint = _latest_checkpoint(cfg)
    gamma = ppo_config(p.ppo).gamma
    env_cfg = dataclasses.replace(env_config(p.env, gamma), randomize=False)
    device = torch.device(str(p.device))
    model: ActorCritic = load_checkpoint(checkpoint, str(p.device)).model.eval()
    clips, eval_set = evaluation_set(cfg, reference_fn, reader)
    if not eval_set["held_out"]:
        logger.warning("evaluating %s: %s", eval_set["source"], eval_set["note"])
    scene_of = scene_fn if scene_fn is not None else functools.partial(clip_scene_xml, cfg)
    by_scene: dict[str, list[EvalClip]] = {}
    for clip in clips:
        by_scene.setdefault(str(scene_of(clip)), []).append(clip)
    clip_scene = {id(clip): scene for scene, group in by_scene.items() for clip in group}
    smc_ks, n_law = [int(k) for k in e.smc_k], int(e.law_rollouts)
    def scene_env(scene: str, k: int) -> BatchedTrackingEnv:
        env: BatchedTrackingEnv = sim.make_env(dataclasses.replace(env_cfg, n_worlds=k), scene, str(p.device))
        return env

    pools: dict[str, EnvPool[BatchedTrackingEnv]] = {}
    for scene in by_scene:
        pools[scene] = EnvPool(functools.partial(scene_env, scene), functools.partial(_timed, device))
        pools[scene].build([1, *(int(k) for k in e.best_of_k), *smc_ks, n_law])

    def pool_of(clip: EvalClip) -> EnvPool[BatchedTrackingEnv]:
        return pools[clip_scene[id(clip)]]

    base = SmcConfig(gamma=gamma, **{k: v for k, v in dict(p.smc).items() if k != "n_particles"})
    generator = torch.Generator(device=str(p.device)).manual_seed(int(p.seed))
    sigma_source = {"clips": "not calibrated here: read next to the checkpoint (0 = floor off)",
                    "note": "no calibration in this run"}
    if bool(e.calibrate_sigma_v):
        k_cal = max(smc_ks)
        groups = [(pools[scene][k_cal], [c.ref for c in group]) for scene, group in by_scene.items()]
        sigma_v, n_res = sim.calibrate(groups, model, dataclasses.replace(base, n_particles=k_cal), generator)
        write_sigma_v(checkpoint, sigma_v, n_res, dataclasses.replace(base, n_particles=k_cal))
        sigma_source = {"clips": f"evaluation set, {len(clips)} clip(s) in {len(by_scene)} scene(s), {n_res} "
                                 "chunk-horizon residuals (chunk 0 excluded)",
                        "note": SIGMA_V_ON_EVAL_NOTE}
    smc_cfg = with_calibrated_floor(base, checkpoint)
    variants = {"smc": smc_cfg, "smc_start_potential": dataclasses.replace(smc_cfg, subtract_start_potential=True),
                "smc_no_floor": dataclasses.replace(smc_cfg, lambda_floor_c=0.0)}
    quiet = dataclasses.replace(smc_cfg, action_noise_std=0.0, reference_noise=False)
    warm_generator = torch.Generator(device=str(p.device)).manual_seed(int(p.seed) + 1)

    def warm_up(clip: EvalClip, k: int, env: BatchedTrackingEnv) -> None:
        sim.best_of_k(env, model, clip.ref, k, warm_generator, cfg=smc_cfg)
        if k in smc_ks:
            sim.smc(env, model, clip.ref, dataclasses.replace(smc_cfg, n_particles=k), warm_generator)

    for scene, group in by_scene.items():
        pools[scene].warm_up(functools.partial(warm_up, group[0]))

    def rollout(clip: EvalClip) -> RolloutOutcome:
        env = pool_of(clip).warmed(1)
        res, sec = _timed(device, lambda: sim.best_of_k(env, model, clip.ref, 1, generator, cfg=quiet))
        return RolloutOutcome(res.success, sec)

    def run(clip: EvalClip, method: str, k: int) -> RolloutOutcome:
        env = pool_of(clip).warmed(k)
        if method == "best_of_k":
            res, sec = _timed(device, lambda: sim.best_of_k(env, model, clip.ref, k, generator, cfg=smc_cfg))
        else:
            cfg_k = dataclasses.replace(variants[method], n_particles=k)
            res, sec = _timed(device, lambda: sim.smc(env, model, clip.ref, cfg_k, generator))
        return RolloutOutcome(res.success, sec)

    quality = evaluate_quality(clips, rollout)
    plan = [("best_of_k", list(e.best_of_k)), *((name, smc_ks) for name in variants)]
    rows = measure_rates(clips, run, plan)
    successes, trials = law_counts(clips, lambda c, n: sim.best_of_k(pool_of(c)[n], model, c.ref, n, generator,
                                                                     cfg=smc_cfg).n_success, n_law)
    law = law_report(successes, trials, rows, int(e.n_boot), int(p.seed), float(e.excess_target),
                     float(e.compute_induced_target), int(e.law_min_clips), int(e.law_min_rollouts),
                     clip_ids=[clip.clip_id for clip in clips])
    t_rest = None if e.t_rest_seconds is None else float(e.t_rest_seconds)
    report = {"checkpoint": str(checkpoint), "evaluation_set": eval_set, "sigma_v": smc_cfg.sigma_v,
              "sigma_v_calibration": {"sigma_v": smc_cfg.sigma_v, **sigma_source},
              "timing": pooled_timing(dict(pools)), "quality": quality, "rates": rows, "law": law,
              "throughput": throughput_rows(quality, rows, t_rest, float(e.t_mpc_seconds)), "pending": pending_rows()}
    json_path, _ = write_report(report, str(e.report_dir))
    logger.info("wrote %s", json_path)
    return report


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    if "policy" not in cfg:
        raise ValueError("compose the policy group: eval_policy.py +policy=default")
    apply_cpu_budget()
    if str(cfg.policy.device).startswith("cuda"):
        apply_gpu_budget()
    main_from_config(cfg)


if __name__ == "__main__":
    main()
