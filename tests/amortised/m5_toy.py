"""M5 toy inputs: two toy clips through environment v2, a deterministic action-dependent scorer, and a CLI
that trains (fresh or resumed) and prints the run's digest, for the kill-and-resume known answer.

  python -m tests.amortised.m5_toy <out_dir> <max_steps> [--resume] [--checkpoint-every N]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from fastdo.amortised import tracker_obs as O
from fastdo.amortised.sac import SacConfig
from fastdo.amortised.schedule import AssistConfig
from fastdo.amortised.trainer import TrainerConfig, TrainInputs

CFG = O.TrackerConfig(k=5, horizon=4)
WRIST_TOL = 0.022          # mean |wrist x - reference| over the executed rows: about half the noisy episodes pass


def inputs() -> TrainInputs:
    import trimesh

    from fastdo.amortised.contact_ref import GridSDF
    from fastdo.amortised.tracker_env import clip_table_entry
    from tests.amortised.test_m3_env import _scorer_rows, _tips
    from tests.transductive.toy import toy_clip

    box = trimesh.creation.box((0.04, 0.04, 0.04))
    sdf = GridSDF.from_mesh(np.asarray(box.vertices), np.asarray(box.faces), resolution=24)
    pts = np.asarray(trimesh.sample.sample_surface_even(box, 64, seed=0)[0])[:8]
    specs = [toy_clip("a", T=80, W=20), toy_clip("b", T=60, W=15, ref_shift=0.03)]
    tips = [_tips(sp) for sp in specs]
    entries = [_scorer_rows(clip_table_entry(sp, t, sdf, pts)) for sp, t in zip(specs, tips, strict=True)]
    return TrainInputs(specs, tips, [sdf, sdf], entries, ["a", "b"], [scorer(e) for e in entries])


def scorer(entry: dict):
    """success iff the executed wrist x tracks the reference within `WRIST_TOL` on average (depends on the noise).
    Episodes end at the clip's last row, so n executed rows are compared with the reference's last n rows (an RSI
    episode can pass too, which the store must then refuse)."""
    q_ref, T = np.asarray(entry["q_ref"]), int(entry["T"])

    def f(q: np.ndarray) -> dict:
        err = float(np.abs(q[:, 0] - q_ref[T - q.shape[0]:T, 0]).mean())
        return {"success": bool(err <= WRIST_TOL), "wrist_err": err}

    return f


def config(max_steps: int = 60, checkpoint_every: int = 0, threads: int | None = None, assist: bool = True,
           rsi_prob: float = 0.0, seed: int = 0) -> TrainerConfig:
    sac = SacConfig(actor_hidden=32, critic_hidden=32, num_bins=21, batch_size=64, updates_per_transition=0.25)
    return TrainerConfig(seed=seed, n_worlds=4, max_steps=max_steps, learning_starts=8, replay_capacity=500,
                         act_sync_every=3, checkpoint_every=checkpoint_every, rsi_prob=rsi_prob, threads=threads,
                         assist=AssistConfig(enabled=assist, window=4, unassisted_every=2), sac=sac, tracker=CFG)


def main(argv: list[str] | None = None) -> int:
    from fastdo.amortised.trainer import Trainer

    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("max_steps", type=int)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--checkpoint-every", type=int, default=0)
    a = ap.parse_args(argv)
    t = Trainer(config(a.max_steps, a.checkpoint_every, threads=1, rsi_prob=0.25), inputs(), Path(a.out))
    resumed_from = t.step if (a.resume and t.resume()) else None
    t.run()
    print(json.dumps({"digest": t.digest(), "step": t.step, "resumed_from": resumed_from, **t.stats}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
