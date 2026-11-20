"""Profile the DO AS I DO baseline on one clip.

Example:
  uv run python run/pipeline/profile_baseline.py \
      video=third_party/do-as-i-do/reconstruction/whisking/whisking.mp4 frame_idx=125 object=whisk hand=right
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from fastdo.teacher.baseline_runner import BaselineClip, profile_baseline_clip

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    root = Path(hydra.utils.get_original_cwd())
    clip = BaselineClip(video=(root / cfg.video).resolve(), frame_idx=int(cfg.frame_idx),
                        object_name=str(cfg.object), anchor_hand=str(cfg.get("hand", "right")))
    out = root / "outputs" / "profiles" / "baseline" / f"{clip.name}.json"
    rep = profile_baseline_clip(clip, baseline_root=(root / cfg.paths.baseline).resolve(),
                                conda_base=Path(cfg.paths.conda_base), out_json=out,
                                skip_retarget=bool(cfg.get("skip_retarget", False)))
    for s in rep.stages:
        logger.info("%-40s %8.1f s", s.name, s.seconds)
    logger.info("TOTAL %.1f s  -> %.3f clips/GPU-hour", rep.total_seconds(), rep.clips_per_gpu_hour())


if __name__ == "__main__":
    main()
