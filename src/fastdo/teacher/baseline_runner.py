"""Run the DO AS I DO baseline headlessly and time its stages from stdout banners."""

from __future__ import annotations

import logging
import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from fastdo.bench.profiler import Profiler, ProfileReport

logger = logging.getLogger(__name__)
_BANNER = re.compile(r"^=== (.+?) ===\s*$")


@dataclass(frozen=True)
class BaselineClip:
    """One input clip for the baseline reconstruction pipeline."""

    video: Path
    frame_idx: int
    object_name: str
    anchor_hand: str = "right"

    @property
    def name(self) -> str:
        return self.video.stem


def make_headless_pipeline_script(src: Path, dst: Path) -> Path:
    """Copy run_pipeline.sh replacing the click GUI prompt with a text prompt."""
    text = src.read_text()
    if "--click" not in text:
        raise ValueError(f"expected a --click flag in {src}")
    text = text.replace("--click \\", '--text "$OBJ_NAME" \\')
    dst.write_text(text)
    dst.chmod(dst.stat().st_mode | 0o755)
    return dst


def run_with_banner_timing(cmd: list[str], cwd: Path, env: dict[str, str], prof: Profiler,
                           stage_prefix: str, log_path: Path) -> int:
    """Run cmd, tee stdout to log_path, and add a stage per '=== name ===' banner."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Force unbuffered stdout in python children so banners are flushed promptly.
    full_env = {**os.environ, **env, "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(cmd, cwd=str(cwd), env=full_env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    current: str | None = None
    t_stage = time.perf_counter()
    assert proc.stdout is not None
    with log_path.open("w") as log:
        for line in proc.stdout:
            log.write(line)
            m = _BANNER.match(line.strip())
            if m is None:
                continue
            now = time.perf_counter()
            if current is not None:
                prof.add(f"{stage_prefix}/{current}", now - t_stage)
            current, t_stage = m.group(1).strip(), now
    rc = proc.wait()
    if current is not None:
        prof.add(f"{stage_prefix}/{current}", time.perf_counter() - t_stage)
    if rc != 0:
        logger.error("command failed rc=%d, see %s", rc, log_path)
    return rc


def _conda_bash(conda_base: Path, script: str) -> list[str]:
    base = shlex.quote(str(conda_base))
    prelude = f'source {base}/etc/profile.d/conda.sh; export PATH={base}/bin:$PATH; '
    return ["bash", "-lc", prelude + script]


def profile_baseline_clip(clip: BaselineClip, baseline_root: Path, conda_base: Path,
                          out_json: Path, skip_retarget: bool = False) -> ProfileReport:
    """Profile reconstruction (+ retargeting) of one clip; returns the report.

    Raises:
        RuntimeError: if the reconstruction or retargeting subprocess exits non-zero. The
            report is still written to out_json first, since partial timings remain useful
            for debugging a failed run.
    """
    recon = baseline_root / "reconstruction"
    headless = make_headless_pipeline_script(recon / "run_pipeline.sh", recon / "run_pipeline_headless.sh")
    n_frames = _count_frames(clip.video)
    prof = Profiler(clip_id=clip.name, system="do-as-i-do", n_frames=n_frames, fps=_fps(clip.video),
                    sync_cuda=False)
    logs = out_json.parent / "logs"
    recon_log = logs / f"{clip.name}_recon.log"
    last_log = recon_log
    rc = run_with_banner_timing(
        _conda_bash(conda_base, f'cd {shlex.quote(str(recon))} && ./{headless.name} '
                                f'{shlex.quote(str(clip.video))} {clip.frame_idx} '
                                f'{shlex.quote(clip.object_name)} {shlex.quote(clip.anchor_hand)}'),
        cwd=recon, env={"SAM3_DISPLAY": ""}, prof=prof, stage_prefix="recon",
        log_path=recon_log)
    if rc == 0 and not skip_retarget:
        retarget = baseline_root / "retargeting"
        raw_dir = clip.video.parent
        retarget_log = logs / f"{clip.name}_retarget.log"
        last_log = retarget_log
        # Time the retarget call ourselves rather than wrapping it in prof.stage(): that
        # would double-count whenever the subprocess also emits '=== ... ===' banners, since
        # run_with_banner_timing(stage_prefix="retarget") already adds one stage per banner
        # and ProfileReport.total_seconds() sums every stage flat. Only fall back to a single
        # explicit "retarget/total" stage when no banner stages were recorded.
        n_before = len(prof.report().stages)
        t0 = time.perf_counter()
        rc = run_with_banner_timing(
            _conda_bash(conda_base, f'conda activate retargeting && cd {shlex.quote(str(retarget))} '
                                    f'&& python launch.py --task {shlex.quote(clip.name)} '
                                    f'--raw-dir {shlex.quote(str(raw_dir))} --no-show-viewer '
                                    f'--no-wait-on-finish'),
            cwd=retarget, env={}, prof=prof, stage_prefix="retarget",
            log_path=retarget_log)
        if len(prof.report().stages) == n_before:
            prof.add("retarget/total", time.perf_counter() - t0)
    rep = prof.report()
    rep.to_json(out_json)
    logger.info("clip %s rc=%d total %.1f s -> %s", clip.name, rc, rep.total_seconds(), out_json)
    if rc != 0:
        raise RuntimeError(f"baseline run for {clip.name} failed with rc={rc}; see {last_log}")
    return rep


def _ffprobe(video: Path, entry: str) -> str:
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                              f"stream={entry}", "-of", "csv=p=0", str(video)],
                             capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return ""
    return out.stdout.strip()


def _count_frames(video: Path) -> int:
    val = _ffprobe(video, "nb_frames")
    return int(val) if val.isdigit() else 0


def _fps(video: Path) -> float:
    val = _ffprobe(video, "r_frame_rate")
    if "/" in val:
        num, den = val.split("/")
        return float(num) / float(den) if float(den) else 0.0
    return float(val) if val else 0.0
