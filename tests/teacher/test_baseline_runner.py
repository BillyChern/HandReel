import sys
from pathlib import Path

import pytest

from fastdo.bench.profiler import Profiler
from fastdo.teacher.baseline_runner import (
    BaselineClip,
    make_headless_pipeline_script,
    profile_baseline_clip,
    run_with_banner_timing,
)


def test_headless_script_replaces_click(tmp_path: Path) -> None:
    src = tmp_path / "run_pipeline.sh"
    src.write_text('python run_sam3_video.py \\\n    --video "$VIDEO_PATH" \\\n    --click \\\n    --obj_id "$OBJ_ID"\n')
    dst = make_headless_pipeline_script(src, tmp_path / "headless.sh")
    text = dst.read_text()
    assert "--click" not in text
    assert '--text "$OBJ_NAME"' in text
    assert dst.stat().st_mode & 0o111


def test_banner_timing_records_stages(tmp_path: Path) -> None:
    script = (
        "import time,sys\n"
        "print('=== A ===', flush=True); time.sleep(0.05)\n"
        "print('=== B ===', flush=True); time.sleep(0.05)\n"
    )
    prof = Profiler("c", "test", 1, 1.0, sync_cuda=False)
    rc = run_with_banner_timing([sys.executable, "-c", script], cwd=tmp_path, env={},
                                prof=prof, stage_prefix="recon", log_path=tmp_path / "log.txt")
    rep = prof.report()
    assert rc == 0
    assert [s.name for s in rep.stages] == ["recon/A", "recon/B"]
    assert all(s.seconds >= 0.04 for s in rep.stages)
    assert "=== B ===" in (tmp_path / "log.txt").read_text()


def test_banner_timing_records_last_stage_on_nonzero_exit(tmp_path: Path) -> None:
    script = "import sys\nprint('=== A ===', flush=True)\nsys.exit(5)\n"
    prof = Profiler("c", "test", 1, 1.0, sync_cuda=False)
    rc = run_with_banner_timing([sys.executable, "-c", script], cwd=tmp_path, env={},
                                prof=prof, stage_prefix="recon", log_path=tmp_path / "log_fail.txt")
    rep = prof.report()
    assert rc == 5
    assert [s.name for s in rep.stages] == ["recon/A"]


def _write_fake_conda(conda_base: Path) -> None:
    """Minimal conda_base: empty conda.sh to source, plus a `conda` stub on PATH so
    `conda activate ...` succeeds without a real conda install."""
    profile_dir = conda_base / "etc" / "profile.d"
    profile_dir.mkdir(parents=True)
    (profile_dir / "conda.sh").write_text("")
    bin_dir = conda_base / "bin"
    bin_dir.mkdir(parents=True)
    conda_stub = bin_dir / "conda"
    conda_stub.write_text("#!/bin/bash\nexit 0\n")
    conda_stub.chmod(conda_stub.stat().st_mode | 0o755)


def _write_fake_recon(baseline_root: Path, body: str) -> None:
    recon = baseline_root / "reconstruction"
    recon.mkdir(parents=True)
    (recon / "run_pipeline.sh").write_text("#!/bin/bash\n# --click \\\n" + body)


def _write_fake_retarget(baseline_root: Path) -> None:
    retarget = baseline_root / "retargeting"
    retarget.mkdir(parents=True)
    (retarget / "launch.py").write_text("import time\n\ntime.sleep(0.02)\n")


def test_profile_baseline_clip_records_retarget_once(tmp_path: Path) -> None:
    baseline_root = tmp_path / "baseline"
    conda_base = tmp_path / "conda"
    _write_fake_conda(conda_base)
    _write_fake_recon(baseline_root, "echo '=== A ==='\nsleep 0.02\necho '=== B ==='\nsleep 0.02\n")
    _write_fake_retarget(baseline_root)

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")
    clip = BaselineClip(video=video, frame_idx=0, object_name="obj", anchor_hand="right")
    out_json = tmp_path / "out" / "clip.json"

    rep = profile_baseline_clip(clip, baseline_root=baseline_root, conda_base=conda_base,
                                out_json=out_json, skip_retarget=False)

    assert [s.name for s in rep.stages] == ["recon/A", "recon/B", "retarget/total"]
    assert rep.total_seconds() == pytest.approx(sum(s.seconds for s in rep.stages))


def test_profile_baseline_clip_raises_on_nonzero_rc(tmp_path: Path) -> None:
    baseline_root = tmp_path / "baseline"
    conda_base = tmp_path / "conda"
    _write_fake_conda(conda_base)
    _write_fake_recon(baseline_root, "echo '=== A ==='\nexit 3\n")

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")
    clip = BaselineClip(video=video, frame_idx=0, object_name="obj", anchor_hand="right")
    out_json = tmp_path / "out" / "clip.json"

    with pytest.raises(RuntimeError):
        profile_baseline_clip(clip, baseline_root=baseline_root, conda_base=conda_base,
                              out_json=out_json, skip_retarget=True)

    assert out_json.exists()
