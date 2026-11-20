"""`run/pipeline/eval_policy.py` and `fastdo.policy.eval_report`: measurement, law-identifiability, evaluation-set,
timing and reporting paths with injected policies (no simulator), plus one CPU run of the whole evaluation on the
whisking training clip. The law floor, per-clip outcomes and the pi_0 printing rule are in
`tests/policy/test_eval_report.py`."""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from fastdo.policy.eval_report import (
    FITTED,
    NOT_FITTED,
    PENDING,
    law_report,
    pending_rows,
    render_markdown,
    throughput_rows,
    write_report,
)
from fastdo.policy.reference import PolicyReference
from run.pipeline.eval_policy import (
    HELD_OUT_UNAVAILABLE,
    SIGMA_V_ON_EVAL_NOTE,
    TRAINING_CLIP_NOTE,
    EnvPool,
    EvalBackend,
    EvalClip,
    RolloutOutcome,
    evaluate_quality,
    evaluation_set,
    law_counts,
    measure_rates,
)
from tests.policy.env_fixtures import scene_missing

logger = logging.getLogger(__name__)

LAW_ONLY_WHEN_FITTED = ("alpha", "beta", "pi_0", "interval95", "smallest_k_excess_fitted", "smallest_k_excess_measured",
                        "smc_checks")


def _clip(clip_id: str, object_name: str) -> EvalClip:
    T = 3
    ref = PolicyReference(q_ref=torch.zeros(T, 28), obj_pose=torch.eye(4).repeat(T, 1, 1), log_scale=0.0,
                          contact_plan=torch.zeros(T, 6, dtype=torch.bool), covariance=None, hz=50.0)
    return EvalClip(clip_id, object_name, ref)


def test_eval_policy_reports_fallback_fraction(tmp_path: Path) -> None:
    """An injected policy that fails one of three synthetic clips: fallback fraction 1/3, per-object success rates,
    seconds per clip and the stage's clips per GPU-hour; the written report carries the number and the pending rows."""
    clips = [_clip("c0", "mug"), _clip("c1", "mug"), _clip("c2", "bowl")]
    seconds = {"c0": 1.0, "c1": 2.0, "c2": 3.0}

    def policy(clip: EvalClip) -> RolloutOutcome:
        return RolloutOutcome(success=clip.clip_id != "c1", seconds=seconds[clip.clip_id])

    quality = evaluate_quality(clips, policy)
    assert quality["fallback_fraction"] == pytest.approx(1.0 / 3.0) and quality["failed_clips"] == ["c1"]
    assert quality["per_object"]["mug"] == {"clips": 2.0, "successes": 1.0, "success_rate": 0.5}
    assert quality["per_object"]["bowl"]["success_rate"] == 1.0
    assert quality["seconds_per_clip"] == 2.0 and quality["clips_per_gpu_hour"] == 1800.0
    json_path, md_path = write_report({"quality": quality, "rates": [], "law": None, "throughput": [],
                                       "pending": pending_rows()}, tmp_path)
    assert json.loads(json_path.read_text())["quality"]["fallback_fraction"] == pytest.approx(1.0 / 3.0)
    markdown = md_path.read_text()
    assert "0.3333" in markdown and "Fitted law: pending" in markdown
    assert {r["item"] for r in pending_rows()} == {"bc_warm_start", "held_out_policy", "head_to_head",
                                                   "clearance_sweep"}
    assert all(r["status"] == PENDING for r in pending_rows())
    with pytest.raises(ValueError):
        evaluate_quality([], policy)


def test_measured_rates_law_and_throughput_rows() -> None:
    """Direct rates on 40 injected clips (clip i needs K >= 2^(i mod 6) rollouts; SMC fails only clips i mod 10 == 9),
    the fitted law from 300 clips x 64 rollouts drawn from a known zero-inflated Beta, and the Section 2 rows."""
    clips = [_clip(f"c{i}", "obj") for i in range(40)]

    def run(clip: EvalClip, method: str, k: int) -> RolloutOutcome:
        i = int(clip.clip_id[1:])
        ok = k >= 2 ** (i % 6) if method == "best_of_k" else i % 10 != 9
        return RolloutOutcome(ok, 0.5 * k if method == "best_of_k" else 1.5)

    rows = measure_rates(clips, run, [("best_of_k", [1, 4, 16, 64]), ("smc", [4, 16])])
    rates = {(r["method"], r["k"]): r for r in rows}
    assert rates[("best_of_k", 1)]["failures"] == 33 and rates[("best_of_k", 16)]["failures"] == 6
    assert rates[("best_of_k", 64)]["rate"] == 0.0 and rates[("smc", 4)]["rate"] == pytest.approx(0.1)
    assert all(r["wilson95_low"] <= r["rate"] <= r["wilson95_high"] for r in rows)
    assert rates[("best_of_k", 4)]["seconds_per_clip"] == 2.0
    assert rates[("best_of_k", 4)]["clip_failures"] == [int(4 < 2 ** (i % 6)) for i in range(40)]
    assert rates[("smc", 16)]["clip_failures"] == [int(i % 10 == 9) for i in range(40)]
    assert all(r["clip_trials"] == [1] * 40 for r in rows)                  # one run per clip per (method, K)

    rng = np.random.default_rng(0)
    p = np.where(rng.random(300) < 0.02, 0.0, rng.beta(0.6, 1.4, size=300))
    law_clips = [_clip(f"l{i}", "obj") for i in range(300)]
    successes, trials = law_counts(law_clips, lambda c, n: int(rng.binomial(n, p[int(c.clip_id[1:])])), 64)
    assert trials.tolist() == [64] * 300 and successes.shape == (300,)
    law = law_report(successes, trials, rows, n_boot=20, seed=0, excess=0.05, compute_induced=0.003, min_clips=100,
                     clip_ids=[c.clip_id for c in law_clips])
    assert law["status"] == FITTED
    for name in ("alpha", "beta", "pi_0"):
        low, high = law["interval95"][name]
        assert low <= high and math.isfinite(law[name])
    assert abs(law["alpha"] - 0.6) < 0.3 and law["rollouts_per_clip"] == 64 and law["n_clips"] == 300
    checks = {c["k"]: c for c in law["smc_checks"]}
    assert checks[4]["f_smc_minus_pi_0"] == pytest.approx(0.1 - law["pi_0"]) and not checks[4]["meets_target"]
    assert law["meets_target"] == {"4": False, "16": False}
    assert law["smallest_k_excess_measured"] in (16, 64)

    quality = {"fallback_fraction": 1.0 / 3.0, "seconds_per_clip": 2.0}
    table = throughput_rows(quality, rows, None, 900.0)
    smc16 = next(r for r in table if r["configuration"].endswith("K=16"))
    assert smc16["expected_seconds"] == 1.5 and smc16["yield"] == pytest.approx(0.9)
    assert smc16["accepted_clips_per_gpu_hour"] == pytest.approx(3600.0 * 0.9 / 1.5)
    parity = table[-1]
    assert parity["expected_seconds"] == pytest.approx(2.0 + 300.0) and "t_rest pending" in parity["scope"]
    assert throughput_rows(quality, rows, 18.0, 900.0)[0]["expected_seconds"] == 19.5
    report = {"quality": None, "rates": rows, "law": law, "throughput": table, "pending": pending_rows()}
    markdown = render_markdown(report)
    assert "| smc | 16 |" in markdown and "Fitted law: alpha" in markdown and "twisted SMC K=4" in markdown


def test_env_pool_builds_and_warms_up_before_any_timed_rollout() -> None:
    """The pool constructs each world count once and never on access; a timed accessor refuses an environment that was
    not warmed up; construction and warm-up seconds are recorded per K, separately."""
    made: list[int] = []
    warmed: list[int] = []
    durations = iter([2.0, 3.0, 4.0, 0.5, 0.25, 0.125])

    def make(k: int) -> str:
        made.append(k)
        return f"env{k}"

    def timer(fn: Any) -> tuple[Any, float]:
        return fn(), next(durations)

    pool: EnvPool[str] = EnvPool(make, timer)
    pool.build([4, 1, 4, 16])
    pool.build([1, 4])
    assert made == [1, 4, 16] and pool[4] == "env4"
    with pytest.raises(RuntimeError, match="not warmed up"):
        pool.warmed(4)
    with pytest.raises(RuntimeError, match="was built"):
        pool[64]
    pool.warm_up(lambda k, env: warmed.append(k))
    assert warmed == [1, 4, 16] and pool.warmed(16) == "env16" and made == [1, 4, 16]
    timing = pool.timing()
    assert timing["construction_seconds"] == {"1": 2.0, "4": 3.0, "16": 4.0}
    assert timing["warmup_seconds"] == {"1": 0.5, "4": 0.25, "16": 0.125} and "exclude" in timing["note"]


CONSTRUCT_SECONDS, COMPILE_SECONDS = 0.4, 0.3


@dataclass
class _FakeEnv:
    n_worlds: int
    compiled: bool = False
    scene: str = ""


class _SleepingSimulator:
    """A fake simulator for `EvalBackend`: building an environment sleeps CONSTRUCT_SECONDS, an environment's first
    rollout sleeps COMPILE_SECONDS (kernel compile or graph capture), later rollouts take ~1 ms and succeed."""

    def make_env(self, cfg: Any, scene_xml: str, device: str) -> _FakeEnv:
        time.sleep(CONSTRUCT_SECONDS)
        return _FakeEnv(cfg.n_worlds, scene=scene_xml)

    def _rollout(self, env: _FakeEnv, k: int) -> Any:
        if not env.compiled:
            time.sleep(COMPILE_SECONDS)
            env.compiled = True
        time.sleep(0.001)
        return SimpleNamespace(success=True, n_success=k)

    def best_of_k(self, env: _FakeEnv, model: Any, ref: Any, k: int, generator: Any, cfg: Any = None) -> Any:
        return self._rollout(env, k)

    def smc(self, env: _FakeEnv, model: Any, ref: Any, cfg: Any, generator: Any) -> Any:
        return self._rollout(env, cfg.n_particles)

    def calibrate(self, groups: Any, model: Any, cfg: Any, generator: Any) -> tuple[float, int]:
        return 0.5, 16


def test_eval_timing_excludes_construction_and_warm_up(tmp_path: Path) -> None:
    """Timing through `main_from_config` with the sleeping fake simulator on the DexYCB held-out set (K in {1, 2}
    best-of-K, SMC at 2, 8 law rollouts): construction and warm-up seconds hold the sleeps for every K, while the timed
    seconds of the K = 1 pass and of every rate row, summed over all clips, stay under half a compile, and so do E[t]
    per throughput configuration. The set is labelled held-out, the law is not fitted (fewer than 20 clips) with
    `meets_target` None and per-clip outcomes for all 5 rate rows, and sigma_V is labelled as calibrated on the
    evaluation clips."""
    from fastdo.policy.checkpoint import save_checkpoint
    from fastdo.policy.networks import ActorCritic
    from run.pipeline.eval_policy import main_from_config
    from tests.policy.test_data import _dexycb_root, _FakeReader, _tagging_fn
    from tests.run.test_train_policy_smoke import _compose

    root, _ = _dexycb_root(tmp_path / "dexycb")
    checkpoint = save_checkpoint(tmp_path / "update_000001.pt", ActorCritic(4, 2, 28, hidden=(4,)), None,
                                 {"ppo": {"gamma": 0.99}})
    cfg = _compose(tmp_path, ["policy.run=timing", "policy.device=cpu", "policy.data.source=dexycb",
                              f"policy.data.dexycb_root={root}", "policy.data.n_held_out=10",
                              "policy.eval.best_of_k=[1,2]", "policy.eval.smc_k=[2]", "policy.eval.law_rollouts=8",
                              f"policy.eval.checkpoint={checkpoint}"])
    fake = _SleepingSimulator()
    backend = EvalBackend(make_env=fake.make_env, best_of_k=fake.best_of_k, smc=fake.smc, calibrate=fake.calibrate)
    start = time.perf_counter()
    report = main_from_config(cfg, _tagging_fn([]), reader=_FakeReader(root), backend=backend,
                              scene_fn=lambda clip: "held_out_scene.xml")
    wall = time.perf_counter() - start

    timing, quality, budget = report["timing"], report["quality"], 0.5 * COMPILE_SECONDS
    assert set(timing["construction_seconds"]) == set(timing["warmup_seconds"]) == {"1", "2", "8"}
    assert min(timing["construction_seconds"].values()) >= CONSTRUCT_SECONDS
    assert min(timing["warmup_seconds"].values()) >= COMPILE_SECONDS
    assert quality["n_clips"] >= 1 and quality["seconds_per_clip"] * quality["n_clips"] < budget
    assert len(report["rates"]) == 5 and all(r["seconds_per_clip"] * r["n_clips"] < budget for r in report["rates"])
    assert all(r["expected_seconds"] < budget for r in report["throughput"])
    assert report["evaluation_set"]["held_out"] and report["evaluation_set"]["held_out_status"].startswith("evaluated")
    assert report["law"]["status"] == NOT_FITTED and report["law"]["meets_target"] is None
    per_clip = report["law"]["per_clip"]
    assert len(per_clip) == quality["n_clips"] and all(len(c["rates"]) == 5 and c["trials"] == 8 for c in per_clip)
    calibration = report["sigma_v_calibration"]
    assert report["sigma_v"] == calibration["sigma_v"] == 0.5 and calibration["note"] == SIGMA_V_ON_EVAL_NOTE
    assert "Tempering floor: sigma_V 0.5" in (tmp_path / "policy" / "timing" / "eval" / "report.md").read_text()
    logger.info("sleeping simulator: %.2f s wall, construction %s, warm-up %s, K=1 %.4f s per clip over %d clips",
                wall, timing["construction_seconds"], timing["warmup_seconds"], quality["seconds_per_clip"],
                quality["n_clips"])


class _SceneSimulator(_SleepingSimulator):
    """No sleeps: records every environment built (scene, worlds), the scene and reference tag of every rollout and
    the (scene, clip count) groups of every sigma_V calibration."""

    def __init__(self) -> None:
        self.built: list[tuple[str, int]] = []
        self.rollouts: list[tuple[str, float]] = []
        self.calibrations: list[list[tuple[str, int]]] = []

    def make_env(self, cfg: Any, scene_xml: str, device: str) -> _FakeEnv:
        self.built.append((scene_xml, cfg.n_worlds))
        return _FakeEnv(cfg.n_worlds, compiled=True, scene=scene_xml)

    def best_of_k(self, env: _FakeEnv, model: Any, ref: Any, k: int, generator: Any, cfg: Any = None) -> Any:
        self.rollouts.append((env.scene, ref.log_scale))
        return SimpleNamespace(success=True, n_success=k)

    def smc(self, env: _FakeEnv, model: Any, ref: Any, cfg: Any, generator: Any) -> Any:
        self.rollouts.append((env.scene, ref.log_scale))
        return SimpleNamespace(success=True, n_success=cfg.n_particles)

    def calibrate(self, groups: Any, model: Any, cfg: Any, generator: Any) -> tuple[float, int]:
        self.calibrations.append([(env.scene, len(refs)) for env, refs in groups])
        return 0.5, 16


def test_eval_runs_each_held_out_clip_in_its_object_scene(tmp_path: Path) -> None:
    """DexYCB held-out clips of three objects (n_held_out 10, seed 0: 004, 005, 006 in the fixture). One pool of
    environments is built per object scene, each K once per scene; every rollout runs in an environment of its clip's
    scene; sigma_V is calibrated once over all scene groups (every clip once); timing is reported per scene with the
    sums by K."""
    from fastdo.policy.checkpoint import save_checkpoint
    from fastdo.policy.networks import ActorCritic
    from run.pipeline.eval_policy import main_from_config
    from tests.policy.test_data import _dexycb_root, _FakeReader, _tagging_fn
    from tests.run.test_train_policy_smoke import _compose

    root, _ = _dexycb_root(tmp_path / "dexycb")
    checkpoint = save_checkpoint(tmp_path / "update_000001.pt", ActorCritic(4, 2, 28, hidden=(4,)), None,
                                 {"ppo": {"gamma": 0.99}})
    cfg = _compose(tmp_path, ["policy.run=scenes", "policy.device=cpu", "policy.data.source=dexycb",
                              f"policy.data.dexycb_root={root}", "policy.data.n_held_out=10",
                              "policy.eval.best_of_k=[1,2]", "policy.eval.smc_k=[2]", "policy.eval.law_rollouts=8",
                              f"policy.eval.checkpoint={checkpoint}"])
    fake = _SceneSimulator()
    names: list[str] = []
    backend = EvalBackend(make_env=fake.make_env, best_of_k=fake.best_of_k, smc=fake.smc, calibrate=fake.calibrate)
    report = main_from_config(cfg, _tagging_fn(names), reader=_FakeReader(root), backend=backend,
                              scene_fn=lambda clip: f"{clip.object_name}.xml")
    objects = report["evaluation_set"]["objects"]
    scenes = {f"{name}.xml" for name in objects}
    assert objects == ["004_sugar_box", "005_tomato_soup_can", "006_mustard_bottle"]
    assert sorted(fake.built) == sorted((scene, k) for scene in scenes for k in (1, 2, 8))
    tag_object = {float(i + 1): name for i, name in enumerate(names)}
    assert fake.rollouts and all(scene == f"{tag_object[tag]}.xml" for scene, tag in fake.rollouts)
    assert len(fake.calibrations) == 1 and {scene for scene, _ in fake.calibrations[0]} == scenes
    assert sum(n for _, n in fake.calibrations[0]) == report["evaluation_set"]["n_clips"] == len(names)
    timing = report["timing"]
    assert set(timing["per_scene"]) == scenes and set(timing["construction_seconds"]) == {"1", "2", "8"}


def test_evaluation_set_on_dexycb_takes_the_held_out_objects(tmp_path: Path) -> None:
    """`policy.data.source=dexycb`: the held-out side of object_split(YCB_OBJECTS, n_held_out, split_seed) over
    `policy.data.dexycb_root`, one view per sequence, each clip keyed by its grasped object and the set labelled
    held-out; the training side never holds those objects. Without an injected function the configured DexYCB reference
    function runs: with no validated scene under `policy.data.scenes_root` it skips every clip before loading,
    so the empty set raises."""
    from fastdo.io.datasets.dexycb import YCB_OBJECTS
    from fastdo.policy.data import object_split
    from fastdo.policy.pipeline_config import load_reference_clips
    from tests.policy.test_data import _dexycb_root, _FakeReader, _tagging_fn
    from tests.run.test_train_policy_smoke import _compose

    root, grasped = _dexycb_root(tmp_path / "dexycb")
    cfg = _compose(tmp_path, ["policy.data.source=dexycb", f"policy.data.dexycb_root={root}",
                              "policy.data.n_held_out=10", "policy.data.split_seed=0"])
    _, held_out = object_split(YCB_OBJECTS, 10, 0)
    expected = sorted(seq for seq, name in grasped.items() if name in held_out)
    assert expected and len(expected) < len(grasped)               # the fixture straddles the split at this seed
    clips, meta = evaluation_set(cfg, _tagging_fn([]), reader=_FakeReader(root))
    assert sorted(c.clip_id.rsplit("/", 1)[0] for c in clips) == expected
    assert all(c.clip_id.endswith("/cam0") and c.object_name == grasped[c.clip_id.rsplit("/", 1)[0]] for c in clips)
    assert meta["held_out"] and meta["split"] == "held_out" and meta["n_clips"] == len(expected)
    assert meta["objects"] == sorted({grasped[seq] for seq in expected})
    train = load_reference_clips(cfg, "train", _tagging_fn([]), reader=_FakeReader(root))
    assert len(train) == len(grasped) - len(expected) and not {c.object_name for c in train} & set(held_out)
    with pytest.raises(ValueError, match="no held_out clips"):
        evaluation_set(cfg, reader=_FakeReader(root))


@scene_missing
def test_eval_main_on_the_whisking_training_clip(tmp_path: Path) -> None:
    """The whole evaluation on the CPU twin (6-frame whisking clip, W = 5, best-of-K at 1 and 2, SMC at 2, 2 law
    rollouts): the report labels the set 'training clip, not held-out' with the held-out evaluation 'unavailable
    (pending DexYCB references)' and keys success by the scene's object, the law is not fitted (one clip, 2 rollouts)
    with `meets_target` None and the clip's outcomes written per (method, K), and each environment's construction and
    warm-up seconds are reported separately from the timed rollouts."""
    from fastdo.policy.checkpoint import save_checkpoint
    from fastdo.policy.env import make_env
    from fastdo.policy.networks import ActorCritic
    from fastdo.policy.smc_calibration import sigma_v_path
    from run.pipeline.eval_policy import main_from_config
    from tests.policy.env_fixtures import SCENE_XML, env_config
    from tests.run.test_train_policy_smoke import _compose

    probe = make_env(env_config(n_worlds=1, warmup_duration=0.1), SCENE_XML, "cpu")
    torch.manual_seed(0)
    model = ActorCritic(probe.obs_dim, probe.priv_dim, probe.act_dim, hidden=(16,))
    checkpoint = save_checkpoint(tmp_path / "update_000001.pt", model, None, {"ppo": {"gamma": 0.99}})
    cfg = _compose(tmp_path, ["policy.run=eval_smoke", "policy.device=cpu", "policy.env.warmup_duration=0.1",
                              "policy.data.n_frames=6", "policy.smc.chunk_seconds=0.04", "policy.eval.best_of_k=[1,2]",
                              "policy.eval.smc_k=[2]", "policy.eval.law_rollouts=2", "policy.eval.n_boot=5",
                              f"policy.eval.checkpoint={checkpoint}"])
    report = main_from_config(cfg)

    eval_set = report["evaluation_set"]
    assert not eval_set["held_out"] and eval_set["split"] == "train" and eval_set["note"].startswith(TRAINING_CLIP_NOTE)
    assert eval_set["objects"] == ["whisking"] and eval_set["n_clips"] == 1
    assert eval_set["held_out_status"] == HELD_OUT_UNAVAILABLE
    assert set(report["quality"]["per_object"]) == {"whisking"}
    assert report["law"]["reason"] == "not fitted (n_clips < 20; rollouts_per_clip < 64)"
    per_clip = report["law"]["per_clip"]
    assert [c["clip_id"] for c in per_clip] == ["whisking/0"] and per_clip[0]["trials"] == 2
    assert [(r["method"], r["k"]) for r in per_clip[0]["rates"]] == [(r["method"], r["k"]) for r in report["rates"]]
    assert report["law"]["status"] == NOT_FITTED and not set(LAW_ONLY_WHEN_FITTED) & set(report["law"])
    assert {(r["method"], r["k"]) for r in report["rates"]} == {
        ("best_of_k", 1), ("best_of_k", 2), ("smc", 2), ("smc_start_potential", 2), ("smc_no_floor", 2)}
    timing = report["timing"]
    assert set(timing["construction_seconds"]) == set(timing["warmup_seconds"]) == {"1", "2"}
    assert all(s > 0.0 for s in [*timing["construction_seconds"].values(), *timing["warmup_seconds"].values()])
    out = tmp_path / "policy" / "eval_smoke" / "eval"
    saved = (out / "report.json").read_text()
    assert json.loads(saved)["law"]["meets_target"] is None and json.loads(saved)["evaluation_set"]["held_out"] is False
    markdown = (out / "report.md").read_text()
    assert TRAINING_CLIP_NOTE in markdown and "Fitted law: not fitted" in markdown and "| whisking | 1 |" in markdown
    assert f"| whisking/0 | {per_clip[0]['successes']} / 2 |" in markdown
    assert HELD_OUT_UNAVAILABLE in markdown and report["sigma_v_calibration"]["note"] == SIGMA_V_ON_EVAL_NOTE
    assert sigma_v_path(checkpoint).exists()
    logger.info("eval smoke: construction s %s, warm-up s %s, K=1 rollout %.4f s per clip",
                timing["construction_seconds"], timing["warmup_seconds"], report["quality"]["seconds_per_clip"])
