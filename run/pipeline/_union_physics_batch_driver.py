"""Run a QUEUE of same-hand clips' physics stage by continuous batching on one slot-union mujoco_warp model.

Runs inside the `retargeting` conda env, like `_fast_physics_batch_driver.py` (same spec format and per-clip config
construction, so each clip gets exactly the Config `_baseline_physics_driver.py` builds for it). OPT-IN research
path: nothing calls it by default.

  python run/pipeline/_union_physics_batch_driver.py --spec batch.json --slots 4 [--union-dir DIR]

Every clip of the spec must have the same `embodiment_type` (one union per hand side). Writes each clip's
tracking-error npz and arm_meta.json (`wall_seconds` = the queue's loop wall, plus `executor: "union"`, the queue,
slots and the run's tick/rebuild/busy statistics). Exit 0 iff every clip produced errors.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _baseline_physics_driver as drv
import _fast_physics_batch_driver as fbd

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--slots", type=int, default=4)
    ap.add_argument("--union-dir", default=None, help="where union scenes are written (default: <spec dir>/unions)")
    args = ap.parse_args(argv)
    spec = json.loads(Path(args.spec).read_text())
    clips = spec["clips"]
    sides = {c["embodiment_type"] for c in clips}
    if len(sides) != 1:
        raise ValueError(f"one hand side per union queue, got {sorted(sides)}")
    baseline_root = Path(spec["baseline_root"]).resolve()
    sys.path.insert(0, str(baseline_root))
    if str(drv.SRC) not in sys.path:
        sys.path.insert(0, str(drv.SRC))
    from dataclasses import fields

    from retargeting.config import Config
    from retargeting.pipeline import (
        optimize_physics,  # noqa: F401  -- imported before the timed region, as the drivers do
    )

    from fastdo.physics.union_run import run_union_queue

    allowed = {f.name for f in fields(Config)}
    overrides = [fbd.clip_overrides(c) for c in clips]
    for c in clips:
        drv.check_override_keys(dict(c.get("overrides") or {}), allowed)
    configs = [drv._load_config_dict(baseline_root, o) for o in overrides]
    k = int(configs[0].num_perturb_samples)
    worlds = int(configs[0].num_samples) * k
    union_dir = args.union_dir or str(Path(args.spec).resolve().parent / "unions")
    stats: dict = {}
    t0 = time.perf_counter()
    outs = run_union_queue(configs, args.slots, worlds, str(configs[0].device), union_dir,
                           names=[c["task"] for c in clips], stats_out=stats)
    wall_s = time.perf_counter() - t0
    ok = True
    queue = [c["task"] for c in clips]
    for clip, ov, cfg, out in zip(clips, overrides, configs, outs, strict=True):
        ok &= fbd.write_clip_outputs(clip, ov, cfg, out, wall_s, queue)
        meta_path = Path(clip["meta_out"])
        meta = json.loads(meta_path.read_text())
        meta.update({"executor": "union", "slots": args.slots, "union_stats": stats})
        meta_path.write_text(json.dumps(meta, indent=2))
    logger.info("=== union driver done: %d clips, %d slots, wall %.1f s, ok=%s ===", len(clips), args.slots, wall_s, ok)
    return 0 if ok else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s - %(message)s")
    raise SystemExit(main())
