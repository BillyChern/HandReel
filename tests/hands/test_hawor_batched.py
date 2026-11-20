"""Tests for `fastdo.hands.hawor_batched` (uv side) against `fake_hawor_driver.py`: no models, no GPU."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from fastdo.hands.hand_npz import validate_hand_npz
from fastdo.hands.hawor_batched import (
    BATCH_INVARIANT_KEYS,
    HandsClipInput,
    HandsConfig,
    WarmHandsDriver,
    call_timeout_s,
    driver_command,
    driver_env,
    hands_config_from_cfg,
    read_response,
    run_hands_batched,
    serve_command,
    write_request,
)
from tests.hands.fake_hawor_driver import PID_FILE, SECONDS_KEYS, clip_result, write_response

FAKE_DRIVER = Path(__file__).resolve().parent / "fake_hawor_driver.py"
CONFIG_YAML = Path(__file__).resolve().parents[2] / "run" / "conf" / "config.yaml"
OPTION_KEYS = ("det_conf", "det_frames_per_batch", "bf16", "temporal", "crops_per_backbone_batch",
               "windows_per_heads_batch", "clips_per_group", "max_crops_per_group", "upload_frames_per_chunk",
               "max_fill_gap")


@pytest.fixture
def cfg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> HandsConfig:
    monkeypatch.setattr("fastdo.hands.hawor_batched.DRIVER_SCRIPT", FAKE_DRIVER)
    return HandsConfig(hawor_python=Path(sys.executable), hawor_root=tmp_path)


def _clips(tmp_path: Path, n: int = 2) -> list[HandsClipInput]:
    clips = []
    for i in range(n):
        frames_path = tmp_path / f"c{i}" / "decode.npz"
        frames_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(frames_path, frames=np.zeros((3, 4, 6, 3), np.uint8), frame_idx=np.array([0, 3, 6], np.int64),
                 fps=np.float64(10.0))
        clips.append(HandsClipInput(f"c{i}", frames_path, 999.0, tmp_path / f"c{i}" / "hands_hawor.npz"))
    return clips


def test_run_hands_batched_ok(cfg: HandsConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_DRIVER_MODE", "ok")
    clips = _clips(tmp_path)

    report = run_hands_batched(clips, cfg, tmp_path / "job")

    assert report.returncode == 0
    assert [c.status for c in report.clips] == ["done", "done"] and [c.clip_id for c in report.clips] == ["c0", "c1"]
    assert all(validate_hand_npz(clip.out_npz) == 3 for clip in clips)
    assert all(set(c.seconds) == set(SECONDS_KEYS) and c.n_frames == 3 for c in report.clips)
    assert report.load_seconds == 1.0 and report.warmup_seconds == 0.5
    assert (tmp_path / "job" / "driver.log").is_file() and (tmp_path / "job" / "request.json").is_file()


def test_run_hands_batched_one_failed(cfg: HandsConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_DRIVER_MODE", "one_failed")

    report = run_hands_batched(_clips(tmp_path), cfg, tmp_path / "job")

    assert report.returncode == 1
    assert [c.status for c in report.clips] == ["done", "failed"]
    assert "fake detection failure on c1" in report.clips[1].error


def test_run_hands_batched_load_error_raises(cfg: HandsConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_DRIVER_MODE", "load_error")
    with pytest.raises(RuntimeError, match="weights missing"):
        run_hands_batched(_clips(tmp_path), cfg, tmp_path / "job")


def test_run_hands_batched_invalid_npz_marks_failed(cfg: HandsConfig, tmp_path: Path,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_DRIVER_MODE", "bad_npz")

    report = run_hands_batched(_clips(tmp_path), cfg, tmp_path / "job")

    assert [c.status for c in report.clips] == ["failed", "failed"]
    assert all("left_joints" in c.error for c in report.clips)


def _alive(pid: int) -> bool:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, ProcessLookupError):
        return False
    return stat.rsplit(")", 1)[1].split()[0] not in ("Z", "X")


def test_run_hands_batched_timeout_kills_group(cfg: HandsConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_DRIVER_MODE", "sleep")
    cfg = replace(cfg, timeout_s_base=3.0, timeout_s_per_clip=0.0)   # the fake records its pids at startup

    with pytest.raises(TimeoutError, match="driver.log"):
        run_hands_batched(_clips(tmp_path), cfg, tmp_path / "job")

    pids = [int(p) for p in (tmp_path / "job" / PID_FILE).read_text().split()]
    assert len(pids) == 2
    deadline = time.monotonic() + 5.0
    while any(_alive(pid) for pid in pids) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not any(_alive(pid) for pid in pids)


def test_call_timeout_scales_with_clip_count() -> None:
    cfg = HandsConfig(hawor_python=Path("/py"), hawor_root=Path("/root"))
    assert cfg.timeout_s_base == 120.0 and cfg.timeout_s_per_clip == 20.0
    assert call_timeout_s(cfg, 256) == 120.0 + 20.0 * 256


def test_driver_env_strips_overrides() -> None:
    env = driver_env({"TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1", "PYTHONPATH": "/x", "PATH": "/usr/bin"})

    assert "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD" not in env and "PYTHONPATH" not in env
    assert env["FASTDO_GPU_MEM_GB"] == "12" and env["FASTDO_CPU_THREADS"] == "4"
    assert env["OMP_NUM_THREADS"] == env["MKL_NUM_THREADS"] == env["OPENBLAS_NUM_THREADS"] == "4"
    assert env["PATH"] == "/usr/bin"
    assert driver_env({"FASTDO_CPU_THREADS": "2", "OMP_NUM_THREADS": "9"})["OMP_NUM_THREADS"] == "2"


def test_driver_command(cfg: HandsConfig, tmp_path: Path) -> None:
    command = driver_command(cfg, tmp_path / "request.json", tmp_path / "response.json")
    assert command == ["nice", "-n", "15", sys.executable, str(FAKE_DRIVER), "--request", str(tmp_path / "request.json"),
                       "--response", str(tmp_path / "response.json")]


def test_request_roundtrip(cfg: HandsConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    clips = [HandsClipInput("a", Path("a/decode.npz"), 812.5, Path("a/hands_hawor.npz"))]
    cfg = replace(cfg, det_conf=0.3, temporal="sampled", clips_per_group=3)

    request = json.loads(write_request(tmp_path / "job", clips, cfg).read_text())

    assert request["version"] == 1 and Path(request["hawor_root"]).is_absolute()
    assert request["options"] == {key: getattr(cfg, key) for key in OPTION_KEYS}
    assert request["clips"] == [{"clip_id": "a", "frames_path": str(tmp_path / "a/decode.npz"), "focal_px": 812.5,
                                 "out_npz": str(tmp_path / "a/hands_hawor.npz")}]


def test_batch_invariant_is_off_by_default_and_the_request_bytes_are_unchanged(cfg: HandsConfig,
                                                                              tmp_path: Path) -> None:
    clips = [HandsClipInput("a", tmp_path / "a/decode.npz", 812.5, tmp_path / "a/hands_hawor.npz")]
    before = {"version": 1, "hawor_root": str(Path(cfg.hawor_root).absolute()),         # body before batch_invariant
              "options": {key: getattr(cfg, key) for key in OPTION_KEYS},
              "clips": [{"clip_id": "a", "frames_path": str(tmp_path / "a/decode.npz"), "focal_px": 812.5,
                         "out_npz": str(tmp_path / "a/hands_hawor.npz")}]}

    written = write_request(tmp_path / "off", clips, cfg).read_text()

    assert cfg.batch_invariant is False and cfg.mano_rows_per_batch == 512
    assert written == json.dumps(before, indent=1)
    assert written == write_request(tmp_path / "off2", clips, replace(cfg, mano_rows_per_batch=64)).read_text()


def test_batch_invariant_on_adds_its_options_to_the_request(cfg: HandsConfig, tmp_path: Path) -> None:
    clips = [HandsClipInput("a", tmp_path / "a/decode.npz", 812.5, tmp_path / "a/hands_hawor.npz")]
    cfg = replace(cfg, batch_invariant=True, mano_rows_per_batch=256)

    options = json.loads(write_request(tmp_path / "job", clips, cfg).read_text())["options"]

    assert options == {key: getattr(cfg, key) for key in OPTION_KEYS} | {"batch_invariant": True,
                                                                         "mano_rows_per_batch": 256}
    assert BATCH_INVARIANT_KEYS == ("batch_invariant", "mano_rows_per_batch")
    with pytest.raises(ValueError, match="mano_rows_per_batch"):
        HandsConfig(hawor_python=Path("/py"), hawor_root=Path("/r"), mano_rows_per_batch=0)


def test_read_response_records_batch_invariant(tmp_path: Path) -> None:
    path = tmp_path / "response.json"
    write_response(path, "", [clip_result("a", "done", "", 46)])
    assert read_response(path, returncode=0).batch_invariant is False
    body = json.loads(path.read_text()) | {"batch_invariant": True, "mano_rows_per_batch": 512}
    path.write_text(json.dumps(body))
    assert read_response(path, returncode=0).batch_invariant is True


def test_read_response(tmp_path: Path) -> None:
    path = tmp_path / "response.json"
    write_response(path, "", [clip_result("a", "done", "", 46, total=1.25), clip_result("b", "failed", "x", 0)])

    report = read_response(path, returncode=1)

    assert report.returncode == 1 and report.load_seconds == 1.0 and report.compute_seconds == 0.25
    assert [(c.clip_id, c.status, c.error, c.n_frames) for c in report.clips] == [("a", "done", "", 46),
                                                                                  ("b", "failed", "x", 0)]
    assert report.clips[0].seconds["total"] == 1.25


def test_hands_config_rejects_bad_values() -> None:
    with pytest.raises(ValueError, match="temporal"):
        HandsConfig(hawor_python=Path("/py"), hawor_root=Path("/r"), temporal="interp30")
    with pytest.raises(ValueError, match="max_crops_per_group"):
        HandsConfig(hawor_python=Path("/py"), hawor_root=Path("/r"), max_crops_per_group=0)


def test_hands_config_from_repo_config_matches_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("FASTDO_HAWOR_ROOT", "FASTDO_EXT_ROOT", "FASTDO_CONDA_BASE"):   # the defaults, not a host override
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("FASTDO_ROOT", "/repo")
    cfg = OmegaConf.load(CONFIG_YAML)

    hands = hands_config_from_cfg(cfg)

    assert cfg.hands.backend == "hawor_batched"
    assert hands.hawor_python == Path("/opt/fastdo/miniforge3/envs/hawor/bin/python")
    assert hands.hawor_root == Path("/repo/third_party/do-as-i-do/reconstruction/modules/HaWoR")
    defaults = HandsConfig(hawor_python=hands.hawor_python, hawor_root=hands.hawor_root)
    assert hands.warm_driver is True and defaults.warm_driver is False    # promoted in the config
    assert hands.batch_invariant is True and defaults.batch_invariant is False    # promoted in the config
    assert hands.deterministic is True and defaults.deterministic is False    # promoted in the config
    promoted = ("warm_driver", "batch_invariant", "deterministic")
    assert all(getattr(hands, f.name) == getattr(defaults, f.name) for f in fields(HandsConfig) if f.name not in promoted)
    monkeypatch.setenv("FASTDO_HAWOR_ROOT", "/elsewhere/HaWoR")
    assert hands_config_from_cfg(OmegaConf.load(CONFIG_YAML)).hawor_root == Path("/elsewhere/HaWoR")


def test_the_cache_is_released_before_the_driver_starts(cfg: HandsConfig, tmp_path: Path,
                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    """Ordering is the whole point: the driver's memory is ADDED to ours while it runs.

    Measured on the shared card 2026-09-17: our engine process held 6758 MiB (overwhelmingly cached blocks from the
    mask and shape stages) while the driver held 7330 MiB, and the guard killed the job at a 14 GB total against
    an 8 GB cap. Releasing AFTER the launch would be useless, so this test pins the order.
    """
    import fastdo.hands.hawor_batched as hb

    order: list[str] = []
    real_popen = hb.subprocess.Popen

    def spy_release() -> float:
        order.append("release")
        return 1.25

    def spy_popen(*args: object, **kwargs: object):
        order.append("popen")
        return real_popen(*args, **kwargs)                              # type: ignore[arg-type]

    monkeypatch.setattr(hb, "release_cuda_cache", spy_release)
    monkeypatch.setattr(hb.subprocess, "Popen", spy_popen)
    monkeypatch.setenv("FAKE_DRIVER_MODE", "ok")

    report = run_hands_batched(_clips(tmp_path), cfg, tmp_path / "job")

    assert report.returncode == 0
    assert order == ["release", "popen"], f"the cache must be released BEFORE the driver starts, got {order}"


def test_releasing_the_cache_is_a_no_op_without_cuda() -> None:
    """A CPU-only caller (every unit test run, which masks CUDA) must not fail or pretend it freed anything."""
    from fastdo.hands.hawor_batched import release_cuda_cache

    assert release_cuda_cache() == 0.0


# -- the warm driver ------------------------------------------------------------------------------------------


def _pids(job: Path) -> list[int]:
    return [int(p) for p in (job / PID_FILE).read_text().split()]


def test_warm_driver_serves_three_calls_from_one_process(cfg: HandsConfig, tmp_path: Path,
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_DRIVER_MODE", "ok")
    clips = _clips(tmp_path)
    with WarmHandsDriver(cfg, tmp_path / "server") as warm:
        reports = [run_hands_batched(clips, cfg, tmp_path / f"job{i}", warm=warm) for i in range(3)]
        pid = warm.pid

        assert warm.spawns == 1 and pid is not None
        assert [_pids(tmp_path / f"job{i}") for i in range(3)] == [[pid]] * 3
    assert all(r.returncode == 0 and [c.status for c in r.clips] == ["done", "done"] for r in reports)
    assert all(validate_hand_npz(clip.out_npz) == 3 for clip in clips)
    assert all((tmp_path / f"job{i}" / "driver.log").is_file() for i in range(3))
    assert warm.pid is None and not _alive(pid)                       # closed on exit, by EOF
    warm.close()                                                      # idempotent


def test_warm_driver_timeout_kills_it_and_the_next_call_starts_fresh(cfg: HandsConfig, tmp_path: Path,
                                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_DRIVER_MODE", "sleep")
    cfg = replace(cfg, timeout_s_base=3.0, timeout_s_per_clip=0.0)
    warm = WarmHandsDriver(cfg, tmp_path / "server")

    with pytest.raises(TimeoutError, match="driver.log"):
        run_hands_batched(_clips(tmp_path), cfg, tmp_path / "job0", warm=warm)

    pids = _pids(tmp_path / "job0")
    assert len(pids) == 2 and warm.pid is None
    deadline = time.monotonic() + 5.0
    while any(_alive(pid) for pid in pids) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not any(_alive(pid) for pid in pids)                       # the server AND its grandchild

    monkeypatch.setenv("FAKE_DRIVER_MODE", "ok")
    report = run_hands_batched(_clips(tmp_path), cfg, tmp_path / "job1", warm=warm)

    assert report.returncode == 0 and warm.spawns == 2 and _pids(tmp_path / "job1") == [warm.pid]
    warm.close()


def test_warm_driver_death_mid_request_is_a_runtime_error(cfg: HandsConfig, tmp_path: Path,
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_DRIVER_MODE", "crash")
    warm = WarmHandsDriver(cfg, tmp_path / "server")

    with pytest.raises(RuntimeError, match="exited with code -9") as info:
        run_hands_batched(_clips(tmp_path), cfg, tmp_path / "job0", warm=warm)

    assert getattr(info.value, "fatal_to_run", False) is True        # today's SIGKILL rule, unchanged
    assert warm.pid is None
    monkeypatch.setenv("FAKE_DRIVER_MODE", "ok")
    assert run_hands_batched(_clips(tmp_path), cfg, tmp_path / "job1", warm=warm).returncode == 0
    assert warm.spawns == 2
    warm.close()


def test_warm_driver_load_error_raises_and_the_next_call_starts_fresh(cfg: HandsConfig, tmp_path: Path,
                                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_DRIVER_MODE", "load_error")
    warm = WarmHandsDriver(cfg, tmp_path / "server")

    with pytest.raises(RuntimeError, match="weights missing"):
        run_hands_batched(_clips(tmp_path), cfg, tmp_path / "job0", warm=warm)

    first = _pids(tmp_path / "job0")[0]
    assert warm.pid is None and not _alive(first)
    monkeypatch.setenv("FAKE_DRIVER_MODE", "ok")
    assert run_hands_batched(_clips(tmp_path), cfg, tmp_path / "job1", warm=warm).returncode == 0
    assert warm.spawns == 2 and _pids(tmp_path / "job1") != [first]
    warm.close()


def test_warm_driver_is_off_by_default_and_the_fresh_path_is_unchanged(cfg: HandsConfig, tmp_path: Path,
                                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    import fastdo.hands.hawor_batched as hb

    monkeypatch.setenv("FAKE_DRIVER_MODE", "ok")
    commands: list[list[str]] = []
    real_popen = hb.subprocess.Popen

    def spy_popen(command: list[str], **kwargs: object):
        commands.append(list(command))
        return real_popen(command, **kwargs)                           # type: ignore[call-overload]

    monkeypatch.setattr(hb.subprocess, "Popen", spy_popen)

    report = run_hands_batched(_clips(tmp_path), cfg, tmp_path / "job")

    assert cfg.warm_driver is False and report.returncode == 0
    assert commands == [driver_command(cfg, tmp_path / "job" / "request.json", tmp_path / "job" / "response.json")]
    assert serve_command(cfg)[-1] == "--serve" and "--serve" not in commands[0]
