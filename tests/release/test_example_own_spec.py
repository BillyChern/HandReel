"""release/example/run_example.py on your own video (`--spec`), and its one-line exits on input errors.

CPU only: the stub runs use placeholder workers; the non-stub runs stop at the input check, before any model loads.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TEMPLATE = REPO / "release" / "example" / "spec_template.json"


def _module():
    """The example script as a module (shared with test_example_smoke.py when that already loaded it)."""
    if "run_example" in sys.modules:
        return sys.modules["run_example"]
    spec = importlib.util.spec_from_file_location("run_example", REPO / "release" / "example" / "run_example.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


E = _module()


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv(E.GPU_MEM_ENV, E.STUB_GPU_MEM_GB)


def _no_fetch(url: str, dest: Path) -> None:
    raise AssertionError(f"the example tried to download {url}")


def _own_spec(tmp_path: Path, **changes) -> Path:
    spec = json.loads(TEMPLATE.read_text())
    spec.update(changes)
    path = tmp_path / "my_clip.json"
    path.write_text(json.dumps(spec))
    return path


def _exit_message(argv: list[str]) -> str:
    with pytest.raises(SystemExit) as exc:
        E.main(argv, fetch=_no_fetch)
    msg = str(exc.value.code)
    assert "\n" not in msg and "Traceback" not in msg
    return msg


def test_the_template_is_a_valid_own_spec() -> None:
    spec = E.load_spec(TEMPLATE, own=True)
    assert spec["source"] == "own_video" and "dataset" not in spec and "sha256" not in spec


def test_a_stub_run_on_your_own_spec_uses_its_clip_id_and_names_no_dataset(tmp_path: Path) -> None:
    video = tmp_path / "my_clip.mp4"
    video.write_bytes(b"placeholder: the stub never decodes it")
    out = tmp_path / "out"
    spec = _own_spec(tmp_path, clip_id="kitchen_cup_01")
    rc = E.main(["--stub", "--spec", str(spec), "--video", str(video), "--out", str(out)], fetch=_no_fetch)
    assert rc == 0
    report = json.loads((out / E.REPORT_FILE).read_text())
    assert report["clip_id"] == "kitchen_cup_01" and (out / "engine" / "kitchen_cup_01").is_dir()
    assert report["input"]["own_spec"] == "my_clip.json" and report["input"]["dataset"] is None
    assert report["input"]["sha256_matches_pin"] is False and len(report["input"]["sha256"]) == 64
    attribution = (out / E.ATTRIBUTION_FILE).read_text()
    assert "your own --spec file" in attribution and "NVIDIA" not in attribution
    row = json.loads((out / E.LEDGER_FILE).read_text().splitlines()[-1])
    assert row["clip_id"] == "kitchen_cup_01" and row["corpus"] == "own_video"


def test_spec_needs_video(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        E.parse_args(["--spec", str(_own_spec(tmp_path)), "--out", str(tmp_path / "o")])
    assert exc.value.code == 2 and "--spec needs --video" in capsys.readouterr().err


@pytest.mark.parametrize(("changes", "match"), [
    ({"clip_id": "../escape"}, "clip_id"),
    ({"box_xyxy": [0, 0, 5000, 10]}, "not inside"),
    ({"window": [0, 500]}, "window"),
    ({"video": {"fps": 30.0, "width": 1920, "height": 1080}}, "video.n_frames"),
    ({"keyframe": 100}, "keyframe 100 is not a decoded frame"),
    ({"keyframe": 3.5}, "keyframe must be whole numbers"),
    ({"keyframe": 3.0}, "keyframe must be whole numbers"),
    ({"keyframe": True}, "keyframe must be whole numbers"),
    ({"keyframe": "3"}, "keyframe must be whole numbers"),
    ({"window": [0.0, 300]}, "window[0] must be whole numbers"),
    ({"window": [0, 300, 600]}, "window must be"),
    ({"video": {"fps": 30.0, "n_frames": 300.0, "width": 1920, "height": 1080}}, "video.n_frames must be whole"),
])
def test_a_bad_own_spec_exits_with_one_line(tmp_path: Path, changes: dict, match: str) -> None:
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    msg = _exit_message(["--stub", "--spec", str(_own_spec(tmp_path, **changes)), "--video", str(video),
                         "--out", str(tmp_path / "o")])
    assert match in msg
    assert not (tmp_path / "o").exists()


def test_the_keyframe_must_be_a_decoded_frame_and_the_message_names_the_nearest() -> None:
    """A3 #10867: the mask is made on the source keyframe and attached to a decoded row; off the grid those are two
    frames. At 30 fps the stride is 3, so frame 100 of a window from 0 is refused and 99 and 102 are offered."""
    rows = list(range(0, 300, 3))
    for ok in (0, 3, 99, 297):
        E.require_decoded_keyframe(0, rows, 3, ok)
    with pytest.raises(ValueError, match="decoded every 3 source frames from frame 0.*use keyframe 99 or 102"):
        E.require_decoded_keyframe(0, rows, 3, 100)
    with pytest.raises(ValueError, match="use keyframe 297 and give"):
        E.require_decoded_keyframe(0, rows, 3, 299)                    # past the last decoded frame: one neighbour
    E.require_decoded_keyframe(40, rows, 3, 43)                        # the grid starts at the window, not at 0
    with pytest.raises(ValueError, match="from frame 40.*use keyframe 40 or 43"):
        E.require_decoded_keyframe(40, rows, 3, 42)
    E.require_decoded_keyframe(0, list(range(10)), 1, 7)               # stride 1: every frame is decoded


def test_a_missing_spec_key_exits_with_one_line(tmp_path: Path) -> None:
    spec = json.loads(TEMPLATE.read_text())
    del spec["box_xyxy"]
    path = tmp_path / "s.json"
    path.write_text(json.dumps(spec))
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    assert "missing box_xyxy" in _exit_message(["--stub", "--spec", str(path), "--video", str(video),
                                               "--out", str(tmp_path / "o")])


def test_a_wrong_pinned_clip_exits_with_one_line_not_a_traceback(tmp_path: Path) -> None:
    """Outside --stub a file whose sha256 is not the pin stops before any model loads (ValueError -> SystemExit)."""
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"not the pinned clip")
    msg = _exit_message(["--video", str(video), "--out", str(tmp_path / "o"), "--stages", "decode"])
    assert "differs from the pinned" in msg


@pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                    reason="ffmpeg and ffprobe are not on PATH (README Installation requires them)")
def test_a_spec_whose_size_differs_from_the_file_stops_before_any_model(tmp_path: Path) -> None:
    video = tmp_path / "tiny.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=10", "-frames:v", "5",
                    "-pix_fmt", "yuv420p", str(video)], check=True)
    assert E.probe_size(video) == (64, 48)
    msg = _exit_message(["--spec", str(_own_spec(tmp_path)), "--video", str(video), "--out", str(tmp_path / "o"),
                         "--stages", "decode"])
    assert "is 64x48, but" in msg and "1920x1080" in msg
    with pytest.raises(ValueError, match="no video stream"):
        E.probe_size(tmp_path / "my_clip.json")
