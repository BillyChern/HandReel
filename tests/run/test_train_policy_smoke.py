"""Merge-gate smoke run of `run/pipeline/train_policy.py`: 3 PPO updates on the CPU twin with 8 worlds (< 5 min
on 2 threads), through the Hydra configuration `run/conf/policy/default.yaml`."""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path

import pytest
import torch

from fastdo.policy.checkpoint import load_checkpoint
from tests.policy.env_fixtures import REPO_ROOT, scene_missing

logger = logging.getLogger(__name__)


def _compose(outputs: Path, overrides: list[str]):  # -> DictConfig
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(version_base=None, config_dir=str(REPO_ROOT / "run" / "conf")):
        return compose("config", overrides=["+policy=default", f"paths.root={REPO_ROOT}", f"paths.outputs={outputs}",
                                            *overrides])


def test_policy_config_composes_with_documented_defaults(tmp_path: Path) -> None:
    """The default group: 4096 worlds on MuJoCo Warp, ~1e9 env-steps, checkpoints every 50 updates, randomization and
    shaping on, the SMC defaults (no chunk-0 start potential, floor c = 1), the B9 reporting grid, the law's
    floor (20 clips, 64 rollouts per clip) and the BC arm under the physics isolation protocol's output root."""
    from fastdo.policy.pipeline_config import env_config, ppo_config

    cfg = _compose(tmp_path, [])
    p = cfg.policy
    assert (p.env.n_worlds, p.device, p.checkpoint_every) == (4096, "cuda", 50)
    assert abs(p.env.n_worlds * p.ppo.n_steps * p.n_updates / 1e9 - 1.0) < 0.01          # ~1e9 env-steps (Section 8)
    assert (p.eval.law_min_clips, p.eval.law_min_rollouts) == (20, 64)
    assert p.data.n_held_out == 4 and p.data.max_seqs == 0 and p.bc.rescale_velocity and p.bc.clamp_ctrl
    arm = Path(str(cfg.isolation.output_root)) / "theirs" / "sharpa" / "right" / "whisking_c4" / "0"
    assert Path(str(p.bc.arm_dir)) == arm
    assert list(p.eval.best_of_k) == [1, 4, 16, 64, 256] and list(p.eval.smc_k) == [4, 16, 64]
    assert not p.smc.subtract_start_potential and p.smc.lambda_floor_c == 1.0
    env_cfg = env_config(p.env, ppo_config(p.ppo).gamma)
    assert env_cfg.randomize and env_cfg.reward.w_phi == 1.0 and env_cfg.reward.d_phi == 0.1
    assert env_cfg.reward.gamma == ppo_config(p.ppo).gamma == 0.99
    assert Path(str(p.scene_xml)).name == "scene.xml" and str(p.eval.report_dir).startswith(str(tmp_path))


@scene_missing
def test_train_policy_smoke(tmp_path: Path) -> None:
    """3 updates, 8 CPU worlds, curriculum stages of 2 updates, checkpoints every 2: checkpoints after updates 2 and 3,
    finite losses on every update, the stage-2 reference set at strength 1, and a checkpoint that reloads into the
    same actor."""
    from run.pipeline.train_policy import train_from_config

    start = time.perf_counter()
    cfg = _compose(tmp_path, ["policy.run=smoke", "policy.device=cpu", "policy.env.n_worlds=8", "policy.n_updates=3",
                              "policy.checkpoint_every=2", "policy.env.warmup_duration=0.1", "policy.data.n_frames=40",
                              "policy.curriculum.stage_updates=2", "policy.curriculum.strength_updates=2",
                              "policy.curriculum.length_updates=4", "policy.curriculum.start_seconds=0.3",
                              "policy.curriculum.end_seconds=0.6"])
    summary = train_from_config(cfg)
    elapsed = time.perf_counter() - start
    run_dir = tmp_path / "policy" / "smoke"
    assert [Path(c).name for c in summary["checkpoints"]] == ["update_000002.pt", "update_000003.pt"]
    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert [r["update"] for r in rows] == [1, 2, 3]
    for r in rows:
        assert r["policy_loss"] is not None and math.isfinite(r["policy_loss"]), r
        assert r["value_loss"] is not None and math.isfinite(r["value_loss"]), r
    assert [r["strength"] for r in rows] == [0.0, 0.0, 1.0]
    assert rows[0]["clip_seconds"] == pytest.approx(0.3) and rows[2]["clip_seconds"] == pytest.approx(0.45)
    assert sum(r["n_episodes"] for r in rows) > 0                    # 0.3-0.45 s clips end inside the rollouts

    loaded = load_checkpoint(summary["checkpoints"][-1])
    assert loaded.meta["update"] == 3 and loaded.meta["env_steps"] == 3 * 32 * 8
    assert loaded.optimizer_state is not None and (run_dir / "config.yaml").exists()
    obs = torch.randn(4, loaded.model.obs_dim)
    assert torch.isfinite(loaded.model.mean_action(obs)).all()
    logger.info("smoke run: 3 updates x 32 steps x 8 worlds in %.1f s; last policy loss %.4g, value loss %.4g",
                elapsed, rows[-1]["policy_loss"], rows[-1]["value_loss"])
    assert elapsed < 300.0
