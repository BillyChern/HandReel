"""Run SEVERAL clips' physics stage in ONE process, in lock-step on separate CUDA streams.

Runs inside the `retargeting` conda env, like `_baseline_physics_driver.py` (whose config construction it reuses, so
every clip gets exactly the Config that driver would build for it). Each clip keeps its own scene, reference, seed
and random stream; only the device is shared (`fastdo.physics.lockstep`).

  python run/pipeline/_fast_physics_batch_driver.py --spec batch.json

`batch.json`: {"baseline_root": ".../retargeting", "clips": [{"task", "output_root_dir", "embodiment_type",
"errors_out", "meta_out", optional "seed" (0), "num_perturb_samples", "num_samples", "data_id" (0),
"robot_type" ("sharpa"), "max_sim_steps" (0), "overrides" ({})}, ...]}.
Writes each clip's tracking-error npz and arm_meta.json (the driver's `_run_meta`, `wall_seconds` = the batch's
loop wall, plus `executor`, `batch` and the clip's own timings). Exit 0 iff every clip produced errors.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _baseline_physics_driver as drv

logger = logging.getLogger(__name__)


def clip_overrides(clip: dict[str, Any]) -> dict[str, Any]:
    """The overrides `_baseline_physics_driver.main` would build for this clip (same pinned run-mode values)."""
    cli = dict(clip.get("overrides") or {})
    bad = sorted(set(cli) & drv.DRIVER_MANAGED_KEYS)
    if bad:
        raise ValueError(f"overrides set driver-managed keys: {bad}")
    out: dict[str, Any] = {
        "task": clip["task"], "output_root_dir": clip["output_root_dir"], "data_id": int(clip.get("data_id", 0)),
        "robot_type": clip.get("robot_type", "sharpa"), "embodiment_type": clip["embodiment_type"],
        "seed": int(clip.get("seed", 0)), "max_sim_steps": int(clip.get("max_sim_steps", 0)), "force": True,
        "show_viewer": False, "wait_on_finish": False, "save_video": False}
    if clip.get("num_samples") is not None:
        out["num_samples"] = int(clip["num_samples"])
    if clip.get("num_perturb_samples") is not None:
        out["num_perturb_samples"] = int(clip["num_perturb_samples"])
    out.update(cli)
    return out


def write_clip_outputs(clip: dict[str, Any], ov: dict[str, Any], cfg: Any, lane: Any, wall_s: float,
                       batch: list[str]) -> bool:
    """One clip's tracking-error npz (if it produced errors) and arm_meta.json; a failed clip's meta carries the
    exception repr AND its full traceback (also logged with exc_info). Returns True iff the clip produced errors."""
    import numpy as np

    errors, timings = lane.result if lane.error is None and lane.result is not None else (None, {})
    tb = None
    if lane.error is not None:
        tb = "".join(traceback.format_exception(lane.error))
        logger.error("clip %s failed", clip["task"], exc_info=lane.error)
    if errors is not None:
        Path(clip["errors_out"]).parent.mkdir(parents=True, exist_ok=True)
        np.savez(clip["errors_out"], obj_pos_err=errors["obj_pos_err"], obj_quat_err=errors["obj_quat_err"],
                 pos_err_frames=errors["pos_err_frames"], quat_err_frames=errors["quat_err_frames"])
    meta = drv._run_meta(task=clip["task"], wall_s=wall_s, overrides=ov, config=cfg, errors=errors,
                         cli_overrides=dict(clip.get("overrides") or {}) or None)
    meta.update({"executor": "fast", "batch": batch, "batch_size": len(batch), "fast_timings": timings,
                 "error": None if lane.error is None else repr(lane.error), "traceback": tb})
    Path(clip["meta_out"]).parent.mkdir(parents=True, exist_ok=True)
    Path(clip["meta_out"]).write_text(json.dumps(meta, indent=2))
    return errors is not None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", required=True)
    args = ap.parse_args(argv)
    spec = json.loads(Path(args.spec).read_text())
    clips = spec["clips"]
    baseline_root = Path(spec["baseline_root"]).resolve()
    sys.path.insert(0, str(baseline_root))
    if str(drv.SRC) not in sys.path:
        sys.path.insert(0, str(drv.SRC))
    from dataclasses import fields

    from retargeting.config import Config
    from retargeting.pipeline import (
        optimize_physics,  # noqa: F401  -- imported before the timed region, as the driver
    )

    from fastdo.physics.clip import run_many

    allowed = {f.name for f in fields(Config)}
    overrides = [clip_overrides(c) for c in clips]
    for c in clips:
        drv.check_override_keys(dict(c.get("overrides") or {}), allowed)
    configs = [drv._load_config_dict(baseline_root, o) for o in overrides]
    t0 = time.perf_counter()
    lanes = run_many(configs)
    wall_s = time.perf_counter() - t0
    if len(lanes) != len(clips):
        raise RuntimeError(f"run_many returned {len(lanes)} lanes for {len(clips)} clips")
    ok = True
    for clip, ov, cfg, lane in zip(clips, overrides, configs, lanes, strict=True):
        ok &= write_clip_outputs(clip, ov, cfg, lane, wall_s, [c["task"] for c in clips])
    logger.info("=== batch driver done: %d clips, wall %.1f s, ok=%s ===", len(clips), wall_s, ok)
    return 0 if ok else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s - %(message)s")
    raise SystemExit(main())
