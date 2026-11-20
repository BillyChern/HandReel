"""Tests for `fastdo.bench.bench_clips`: the benchmark's clip sources, prompts and image decode."""

from __future__ import annotations

import itertools
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from fastdo.bench.bench_clips import (
    WINDOW_SECONDS,
    BenchClip,
    dexycb_clips,
    dexycb_intrinsics,
    dexycb_prompt,
    dexycb_template_mesh,
    doh_clip_name,
    doh_clip_path,
    doh_clips,
    doh_train_entries,
    keyframe_row,
    make_image_sequence_decode_worker,
    operating_stride,
    probe_windowed_clips,
    whisking_clips,
    whisking_prompt,
    window_offset,
    windowed_clips,
    windowed_clips_from_inventory,
    write_camera_json,
)
from fastdo.engine.worker_support import read_decode
from fastdo.engine.workers_qc import decode_timestamps
from fastdo.geometry.intrinsics import CAMERA_JSON, read_camera_json
from fastdo.io.datasets.dexycb import ycb_object_name
from tests.bench.dexycb_tree import SERIAL, write_fake_dexycb
from tests.bench.fake_doh import install_file_access_spy, write_fake_doh_bundle

_REPO = Path(__file__).resolve().parents[2]
_WHISKING = _REPO / "third_party/do-as-i-do/reconstruction/whisking"


def test_whisking_clips_are_distinct_copies(tmp_path: Path) -> None:
    video = tmp_path / "whisking.mp4"

    clips = whisking_clips(video, n_frames=138, copies=3)

    assert [c.clip_id for c in clips] == ["bench_whisking_000", "bench_whisking_001", "bench_whisking_002"]
    assert all((c.source, c.kind, c.path, c.n_frames) == ("whisking", "video", video, 138) for c in clips)
    assert whisking_clips(video, n_frames=138, copies=0) == []
    with pytest.raises(ValueError, match="copies"):
        whisking_clips(video, n_frames=138, copies=-1)


def test_doh_clips_select_only_train_entries_without_touching_test_clips(tmp_path: Path,
                                                                        monkeypatch: pytest.MonkeyPatch,
                                                                        caplog: pytest.LogCaptureFixture) -> None:
    bundle = write_fake_doh_bundle(tmp_path / "100DOH")
    counted: list[Path] = []

    def count_frames(path: Path) -> int:
        counted.append(path)
        return 250

    spy = install_file_access_spy(monkeypatch)
    with caplog.at_level(logging.INFO):
        clips = doh_clips(bundle.root, count_frames=count_frames)
    monkeypatch.undo()

    assert [c.clip_id for c in clips] == bundle.train_ids
    assert [c.path for c in clips] == bundle.train_paths
    assert all((c.source, c.kind, c.split, c.manifest_source) == ("100doh", "video", "train", "100doh:train")
               for c in clips)
    assert [c.n_frames for c in clips] == [300, 250]           # recorded decoded_frames, else counted
    assert counted == [bundle.train_paths[1]]
    for marker in bundle.held_out_markers:
        assert spy.touched(marker) == []                       # never opened, stat-ed, listed or probed
    assert not any(name.endswith(("scandir", "listdir")) for name, _ in spy.calls)
    assert any("left out, never opened" in r.getMessage() and "'test': 1" in r.getMessage() for r in caplog.records)


def test_doh_train_entries_and_clip_paths(tmp_path: Path) -> None:
    bundle = write_fake_doh_bundle(tmp_path / "100DOH")

    train, held_out = doh_train_entries(bundle.root)

    assert [doh_clip_name(e) for e in train] == [i.removeprefix("bench_100doh_") for i in bundle.train_ids]
    assert held_out == {"<missing>": 1, "test": 1}
    assert [doh_clip_path(bundle.root, e) for e in train] == bundle.train_paths
    for bad in ("/abs/clip.mp4", "clips/../../escape.mp4"):
        with pytest.raises(ValueError, match="doh_root"):
            doh_clip_path(bundle.root, {**train[0], "clip_path": bad})
    assert BenchClip("w", "whisking", "video", tmp_path, 1).manifest_source == "whisking"


def test_doh_clips_without_a_usable_bundle(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "stray.mp4").write_bytes(b"")                     # no manifest.json: never selected by listing

    assert doh_clips(None) == []
    with caplog.at_level(logging.WARNING):
        assert doh_clips(empty, count_frames=lambda path: 1) == []
    assert any("manifest.json" in r.getMessage() for r in caplog.records)
    with pytest.raises(FileNotFoundError, match="doh_root"):
        doh_clips(tmp_path / "absent")
    bundle = write_fake_doh_bundle(tmp_path / "100DOH")
    bundle.train_paths[0].unlink()
    with pytest.raises(FileNotFoundError, match="missing"):
        doh_clips(bundle.root, count_frames=lambda path: 1)
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / "manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="JSON list"):
        doh_clips(tmp_path / "bad")


def test_keyframe_row_is_the_nearest_decode_row() -> None:
    assert keyframe_row(138, 3, 125) == 42          # source frame 126, the head-to-head's 10 fps keyframe
    assert keyframe_row(138, 3, 0) == 0
    assert keyframe_row(10, 3, 500) == 3            # rows at source frames 0, 3, 6, 9
    with pytest.raises(ValueError, match="n_frames"):
        keyframe_row(0, 3, 0)


def test_dexycb_clip_prompt_and_camera(tmp_path: Path) -> None:
    seq = write_fake_dexycb(tmp_path)

    (clip,) = dexycb_clips(tmp_path, [seq])
    mask = dexycb_prompt(tmp_path, seq)
    K = dexycb_intrinsics(tmp_path, seq)
    write_camera_json(tmp_path / "artifacts", K)

    assert clip.clip_id == f"bench_dexycb_20200820-subject-03_20200820_135508_{SERIAL}"
    assert (clip.source, clip.kind, clip.n_frames, clip.path) == ("dexycb", "image_sequence", 5, tmp_path / seq)
    assert mask.dtype == bool and mask.shape == (12, 16) and int(mask.sum()) == 24 and mask[2:6, 3:9].all()
    np.testing.assert_allclose(K, [[600.0, 0.0, 320.5], [0.0, 610.0, 240.5], [0.0, 0.0, 1.0]])
    np.testing.assert_allclose(read_camera_json(tmp_path / "artifacts" / CAMERA_JSON), K)
    with pytest.raises(FileNotFoundError):
        dexycb_clips(tmp_path, ["20200820-subject-03/20200820_135508/000000000000"])


def test_dexycb_template_mesh_is_the_readers_object_mesh(tmp_path: Path) -> None:
    seq = write_fake_dexycb(tmp_path)
    expected = tmp_path / "models" / ycb_object_name(5) / "textured_simple.obj"     # ycb_ids[ycb_grasp_ind] = 5

    with pytest.raises(FileNotFoundError, match="textured_simple.obj"):
        dexycb_template_mesh(tmp_path, seq)
    expected.parent.mkdir(parents=True)
    expected.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")

    assert dexycb_template_mesh(tmp_path, seq) == expected


def test_image_sequence_decode_worker_writes_the_decode_contract(tmp_path: Path) -> None:
    seq = write_fake_dexycb(tmp_path, n=7)
    (clip,) = dexycb_clips(tmp_path, [seq])
    worker = make_image_sequence_decode_worker({clip.clip_id: clip.path}, stride=3, fps=30.0)
    artifact_dir = tmp_path / "engine" / clip.clip_id

    worker(clip.clip_id, artifact_dir)

    frames, frame_idx = read_decode(artifact_dir)
    expected = np.stack([np.asarray(Image.open(clip.path / f"color_{i:06d}.jpg").convert("RGB")) for i in (0, 3, 6)])
    assert frame_idx.tolist() == [0, 3, 6]
    np.testing.assert_array_equal(frames, expected)
    np.testing.assert_allclose(decode_timestamps(artifact_dir)[1], [0.0, 0.1, 0.2])
    with pytest.raises(KeyError, match="other"):
        worker("other", artifact_dir)
    with pytest.raises(ValueError, match="stride"):
        make_image_sequence_decode_worker({}, stride=0)


@pytest.mark.skipif(not (_WHISKING / "whisking.mp4").is_file(), reason="needs the shipped whisking demo")
def test_whisking_prompt_is_the_keyframe_layout_silhouette() -> None:
    prompt = whisking_prompt(
        _WHISKING / "obj_tracking_out/whisk/combined_visualization/layout_camera_frame_optimized.json",
        _WHISKING / "video_segmentation/masks/frame_000125_masks/whisk/whisk.obj", keyframe_frame_idx=125)

    assert prompt.dtype == bool and prompt.shape == (720, 1280)
    assert 100 < int(prompt.sum()) < prompt.size // 4


def test_a_windowed_clip_reaches_the_decode_workers_window_path() -> None:
    """The decode worker accepts `(path, start, stop)` and nothing could build one, so the window path was
    unreachable from the bench. `decode_source` is what
    `bench_stages` now passes, so this pins the handoff, not just the field."""
    from pathlib import Path

    from fastdo.bench.bench_clips import BenchClip

    whole = BenchClip(clip_id="c0", source="ego4d", kind="video", path=Path("/v.mp4"), n_frames=100)
    windowed = BenchClip(clip_id="c1", source="ego4d", kind="video", path=Path("/v.mp4"), n_frames=100,
                         window=(300, 600))
    assert whole.decode_source == Path("/v.mp4")
    assert windowed.decode_source == (Path("/v.mp4"), 300, 600)

    with pytest.raises(ValueError, match="0 <= start < stop"):
        BenchClip(clip_id="c2", source="ego4d", kind="video", path=Path("/v.mp4"), n_frames=100, window=(600, 300))
    with pytest.raises(ValueError, match="only a video or image_window clip"):
        BenchClip(clip_id="c3", source="dexycb", kind="image_sequence", path=Path("/d"), n_frames=100,
                  window=(0, 10))


def test_windowed_clips_implements_the_prereg_window_rule() -> None:
    """The window rule. The numbers checked here are the ones the rule states itself:
    stride 5 at 50 fps and 6 at 59.94 fps, and 100 decode rows per 10 s window at BOTH rates -- which is the
    point of a per-video stride, since a single run-wide stride cannot hold the operating rate across both."""
    assert operating_stride(50.0) == 5
    assert operating_stride(60000 / 1001) == 6                      # 59.94 fps
    assert operating_stride(10.0) == 1 and operating_stride(4.0) == 1   # never below 1

    for fps in (50.0, 60000 / 1001):
        clips = windowed_clips("P18_03", Path("/v.MP4"), duration_s=120.0, fps=fps,
                               source="epic_kitchens_100", split="TEST")
        stride = operating_stride(fps)
        assert all(c.decode_stride == stride for c in clips)
        # n_frames is SOURCE frames -- n_frames/fps is the seconds of video the yield law divides by, so a
        # row count here would under-report a 10 s window as 2 s at 50 fps.
        assert all(c.n_frames == c.window[1] - c.window[0] for c in clips)
        assert all(abs(c.n_frames / fps - WINDOW_SECONDS) < 1.0 / fps for c in clips), fps
        # and the pre-registration's own check: 100 decode rows per window at BOTH rates
        assert [len(range(*c.window, stride)) for c in clips] == [100] * len(clips), fps
        # windows tile the video with no overlap and no gap: each starts where the last stopped
        assert all(a.window[1] == b.window[0] for a, b in itertools.pairwise(clips))
        assert all(c.clip_id == f"P18_03:k{i:03d}" for i, c in enumerate(clips))

    # eligibility: a whole window must fit, and the offset eats into the head
    offset = window_offset("P18_03")
    assert 0.0 <= offset < WINDOW_SECONDS
    n = len(windowed_clips("P18_03", Path("/v.MP4"), duration_s=35.0, fps=50.0, source="ek"))
    assert n == int((35.0 - offset) // WINDOW_SECONDS)
    assert windowed_clips("P18_03", Path("/v.MP4"), duration_s=offset + 9.99, fps=50.0, source="ek") == []

    # the offset is a pure function of the id and the seed -- no pixel, annotation or pipeline output
    assert window_offset("P18_03") == window_offset("P18_03")
    assert window_offset("P18_03") != window_offset("P18_05")
    assert window_offset("P18_03", seed="other") != window_offset("P18_03")

    with pytest.raises(ValueError, match="may not contain a separator"):
        windowed_clips("a/b", Path("/v.MP4"), duration_s=100.0, fps=50.0, source="ek")


def test_a_windowed_clip_carries_its_own_stride_to_the_decode_worker() -> None:
    """S2 needs 5 and 6 in ONE run, so the stride has to live on the clip; `bench_stages` groups by it."""
    fifty = windowed_clips("P18_03", Path("/a.MP4"), 120.0, 50.0, "ek")[0]
    five994 = windowed_clips("P33_102", Path("/b.MP4"), 120.0, 60000 / 1001, "ek")[0]
    assert (fifty.decode_stride, five994.decode_stride) == (5, 6)
    assert fifty.decode_source == (Path("/a.MP4"), *fifty.window)

    with pytest.raises(ValueError, match="decode_stride must be >= 1"):
        BenchClip(clip_id="c", source="ek", kind="video", path=Path("/v"), n_frames=1, decode_stride=0)


def test_probe_windowed_clips_measures_the_duration_the_prereg_defines(tmp_path: Path) -> None:
    """D_v is DECODED frames / avg_frame_rate. Using the container's nb_frames instead would place the last
    window past the end of the video, so both probes are injected here and checked by their effect."""
    video = tmp_path / "v.MP4"
    video.write_bytes(b"")
    clips = probe_windowed_clips("P18_03", video, "epic_kitchens_100", split="TEST",
                                 fps_of=lambda _p: 50.0, count_frames=lambda _p: 50 * 120)
    assert [c.decode_stride for c in clips] == [5] * len(clips)
    assert len(clips) == len(windowed_clips("P18_03", video, 120.0, 50.0, "epic_kitchens_100"))
    assert clips[0].split == "TEST" and clips[0].source == "epic_kitchens_100"

    with pytest.raises(ValueError, match="avg_frame_rate probed as"):
        probe_windowed_clips("P18_03", video, "ego4d", fps_of=lambda _p: 0.0, count_frames=lambda _p: 100)


def test_an_inventory_is_the_runners_route_to_windowed_clips(tmp_path: Path) -> None:
    """`build_clips` reads this; without it the windowed builder is unreachable from the registered runner --
    the exact shape of the P3/P4 defects, where the capability shipped but nothing could invoke it."""
    for name in ("a.MP4", "b.MP4"):
        (tmp_path / name).write_bytes(b"")
    probes = {"fps_of": lambda p: 50.0 if p.name == "a.MP4" else 60000 / 1001,
              "count_frames": lambda p: round((50.0 if p.name == "a.MP4" else 60000 / 1001) * 120)}

    inventory = tmp_path / "windows.json"
    inventory.write_text(json.dumps([
        {"video_id": "P18_03", "path": "a.MP4", "source": "epic_kitchens_100", "split": "TEST"},
        {"video_id": "26924e0d", "path": str(tmp_path / "b.MP4"), "source": "ego4d", "split": "CALIB"},
    ]))
    clips = windowed_clips_from_inventory(inventory, root=tmp_path, **probes)

    # BOTH strides present in ONE clip list -- the thing a single run-wide stride cannot do
    assert {c.decode_stride for c in clips} == {5, 6}
    assert {c.source for c in clips} == {"epic_kitchens_100", "ego4d"}
    assert len({c.clip_id for c in clips}) == len(clips)
    assert all(c.window is not None and c.kind == "video" for c in clips)

    def write(entries: object) -> Path:
        inventory.write_text(json.dumps(entries))
        return inventory

    with pytest.raises(ValueError, match="missing"):
        windowed_clips_from_inventory(write([{"video_id": "x", "path": "a.MP4"}]), root=tmp_path, **probes)
    with pytest.raises(ValueError, match="known sources"):
        windowed_clips_from_inventory(write([{"video_id": "x", "path": "a.MP4", "source": "nope"}]),
                                      root=tmp_path, **probes)
    with pytest.raises(ValueError, match="appears twice"):
        windowed_clips_from_inventory(write([{"video_id": "x", "path": "a.MP4", "source": "ego4d"}] * 2),
                                      root=tmp_path, **probes)
    with pytest.raises(ValueError, match="list of objects"):
        windowed_clips_from_inventory(write({"video_id": "x"}), root=tmp_path, **probes)
    with pytest.raises(FileNotFoundError, match="not a file"):
        windowed_clips_from_inventory(write([{"video_id": "x", "path": "gone.MP4", "source": "ego4d"}]),
                                      root=tmp_path, **probes)


def test_a_truncated_inventory_says_so_loudly(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """`max_per_video` is for a bounded PILOT. A truncated run's window count is not the S2 corpus, so its
    yield denominator is not the corpus's -- and that has to be impossible to take for a real run by accident."""
    (tmp_path / "a.MP4").write_bytes(b"")
    inventory = tmp_path / "w.json"
    inventory.write_text(json.dumps([{"video_id": "v", "path": "a.MP4", "source": "ego4d", "split": "CALIB"}]))
    probes = {"fps_of": lambda _p: 30.0, "count_frames": lambda _p: 30 * 300}

    full = windowed_clips_from_inventory(inventory, root=tmp_path, **probes)
    assert len(full) > 6
    with caplog.at_level(logging.WARNING):
        few = windowed_clips_from_inventory(inventory, root=tmp_path, max_per_video=3, **probes)
    assert "TRUNCATED" in caplog.text and str(len(full)) in caplog.text

    # SPREAD, not the head: the first and last windows are both kept, and the middle one is near the middle.
    # Taking full[:3] would sample only the video's opening, which is where the first S2 pilot's refusals
    # came from -- egocentric openings are walking-in and setup, not manipulation.
    ids = [c.clip_id for c in few]
    assert ids != [c.clip_id for c in full[:3]]
    assert ids[0] == full[0].clip_id and ids[-1] == full[-1].clip_id
    assert full[len(full) // 2 - 1].clip_id <= ids[1] <= full[len(full) // 2 + 1].clip_id

    # a single window comes from the MIDDLE, not the start
    one = windowed_clips_from_inventory(inventory, root=tmp_path, max_per_video=1, **probes)
    assert [c.clip_id for c in one] == [full[len(full) // 2].clip_id]

    # asking for more than exist keeps them all, and stays in order
    allof = windowed_clips_from_inventory(inventory, root=tmp_path, max_per_video=10_000, **probes)
    assert [c.clip_id for c in allof] == [c.clip_id for c in full]

    # the default is the whole corpus, and it warns about nothing
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        windowed_clips_from_inventory(inventory, root=tmp_path, **probes)
    assert "TRUNCATED" not in caplog.text

    with pytest.raises(ValueError, match="max_per_video must be >= 1"):
        windowed_clips_from_inventory(inventory, root=tmp_path, max_per_video=0, **probes)


# --- the config keys that make the decode downscale and the local probe reachable ----------------------

def test_the_decode_config_keys_reach_the_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    """The handoff, pinned at the seam that was missing.

    `decode_window` has taken `scale_height` and `local_probe` for a while; `make_decode_worker` now
    forwards them; and NOTHING read them from config, so the decode downscale was a capability
    with passing unit tests and no way to switch on -- the same shape as the unreachable window path and
    the Gate A index writer that had no writer. This asserts the cfg values arrive at the worker, which is
    the one link the other tests cannot see.
    """
    from pathlib import Path

    from omegaconf import OmegaConf

    from fastdo.bench import bench_stages

    seen: list[dict] = []

    def spy(sources, *, stride, **kw):
        seen.append({"stride": stride, **kw})
        return lambda clip_id, artifact_dir: None

    monkeypatch.setattr(bench_stages, "make_decode_worker", spy)
    clip = BenchClip(clip_id="c1", source="ego4d", kind="video", path=Path("/v.mp4"), n_frames=100,
                     window=(300, 600), decode_stride=6)
    cfg = OmegaConf.create({"bench": {"decode_scale_height": 720, "decode_local_probe": True,
                                      "decode_packet_table": True}})
    bench_stages.build_video_decoders(cfg.bench, [clip], stride=3)

    assert seen, "make_decode_worker was never called"
    assert all(entry["scale_height"] == 720 for entry in seen), seen
    assert all(entry["local_probe"] is True for entry in seen), seen
    assert all(entry["packet_table"] is True for entry in seen), seen


def test_the_decode_config_defaults_leave_every_run_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """A null `decode_scale_height` must decode at source size and a missing key must not raise: the S2
    run opts in, rather than every other experiment silently changing resolution under it."""
    from pathlib import Path

    from omegaconf import OmegaConf

    from fastdo.bench import bench_stages

    seen: list[dict] = []
    monkeypatch.setattr(bench_stages, "make_decode_worker",
                        lambda sources, *, stride, **kw: (seen.append(kw), lambda c, a: None)[1])
    clip = BenchClip(clip_id="c1", source="ego4d", kind="video", path=Path("/v.mp4"), n_frames=100,
                     window=(300, 600))

    bench_stages.build_video_decoders(OmegaConf.create({"decode_scale_height": None}), [clip], stride=3)
    bench_stages.build_video_decoders(OmegaConf.create({}), [clip], stride=3)        # keys absent entirely
    assert [entry["scale_height"] for entry in seen] == [None, None]
    assert [entry["local_probe"] for entry in seen] == [False, False]
    assert [entry["packet_table"] for entry in seen] == [False, False]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available")
def test_a_scaled_worker_writes_scaled_pixels(tmp_path: Path) -> None:
    """And the other end of the chain, with real pixels: `scale_height=720` through the real decode worker
    puts 720-high frames in `decode.npz`, with the window's own indices intact."""
    import subprocess

    from fastdo.engine import make_decode_worker

    source = tmp_path / "v.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
                    "-i", "testsrc=size=1440x1080:rate=10:duration=3", "-pix_fmt", "yuv420p", str(source)],
                   check=True)
    for name, height, want in [("scaled", 720, (720, 960)), ("native", None, (1080, 1440))]:
        artifact_dir = tmp_path / name
        artifact_dir.mkdir()
        make_decode_worker({"c": (source, 4, 16)}, stride=3, scale_height=height)("c", artifact_dir)
        with np.load(artifact_dir / "decode.npz") as data:
            assert data["frames"].shape[1:3] == want, name
            assert data["frame_idx"].tolist() == [4, 7, 10, 13], name
