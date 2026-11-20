"""Tests for the pure frame-index selection helper in fastdo.io.video.

``decode_frames`` itself needs a real video file and a working decord/ffmpeg backend, so it is
exercised by the real-demo smoke run rather than here (per the adapter-contract testing note: GPU/
external-tool adapters are verified by the smoke run; only pure helpers get unit tests).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from fastdo.io.video import sample_frame_indices
from tests.baseline_paths import baseline_path


def test_stride_one_returns_every_index() -> None:
    assert sample_frame_indices(5, stride=1) == [0, 1, 2, 3, 4]


def test_stride_three() -> None:
    assert sample_frame_indices(10, stride=3) == [0, 3, 6, 9]


def test_max_frames_caps_after_striding() -> None:
    assert sample_frame_indices(10, stride=2, max_frames=2) == [0, 2]


def test_max_frames_larger_than_available_is_a_no_op() -> None:
    assert sample_frame_indices(3, stride=1, max_frames=100) == [0, 1, 2]


def test_invalid_stride_raises() -> None:
    with pytest.raises(ValueError, match="stride"):
        sample_frame_indices(10, stride=0)


def test_whisking_demo_stride_three_frame_count() -> None:
    # The real-demo smoke run: frames 0..137 (138 baseline frames) at stride 3 -> 46 frames.
    idx = sample_frame_indices(138, stride=3)
    assert len(idx) == 46
    assert idx[0] == 0 and idx[-1] == 135


# --- decode backends on the real whisking demo (2026-09-13) --------------------------------------
# whisking.mp4's container reports 172 frames (nb_frames, packets) but only 138 decode (ffprobe
# -count_frames, the baseline's own frame extraction, HaWoR). decord trusts the container count and,
# from requested index 87 on, returns the frame 34 earlier, so strided decord output is silently wrong.


WHISKING = baseline_path("reconstruction/whisking/whisking.mp4")
needs_video = pytest.mark.skipif(not WHISKING.exists() or shutil.which("ffmpeg") is None,
                                 reason="whisking demo video or ffmpeg not available")


def test_decode_frames_rejects_unknown_backend(tmp_path: Path) -> None:
    from fastdo.io.video import decode_frames

    with pytest.raises(ValueError, match="backend"):
        decode_frames(tmp_path / "missing.mp4", stride=3, backend="nvdec")


@needs_video
def test_count_decoded_frames_whisking() -> None:
    from fastdo.io.video import count_decoded_frames

    assert count_decoded_frames(WHISKING) == 138


@needs_video
def test_decode_frames_auto_matches_true_frame_sequence_whisking() -> None:
    from fastdo.io.video import decode_frames

    frames3, idx3 = decode_frames(WHISKING, stride=3)
    frames1, idx1 = decode_frames(WHISKING, stride=1)
    assert frames1.shape[0] == 138 and idx1 == list(range(138))
    assert frames3.shape[0] == 46 and idx3[-1] == 135
    assert int((frames3.to(dtype=frames1.dtype) - frames1[::3]).abs().max()) == 0


@needs_video
def test_decord_backend_refuses_container_count_mismatch() -> None:
    pytest.importorskip("decord")
    from fastdo.io.video import decode_frames

    with pytest.raises(ValueError, match="frame count"):
        decode_frames(WHISKING, stride=3, backend="decord")


@needs_video
def test_probe_fps_whisking() -> None:
    """D3 amendment: ffprobe `avg_frame_rate` of stream v:0 as an exact fraction (2580000/86599)."""
    from fastdo.io.video import probe_fps

    assert abs(probe_fps(WHISKING) - 29.7924918) < 1e-6
    assert abs(probe_fps(WHISKING) - 2580000 / 86599) < 1e-9


@needs_video
def test_count_decoded_frames_agrees_with_ffprobe_count_frames() -> None:
    """`count_decoded_frames` is served from the frame table, whose length IS this count by construction.

    The obvious implementation -- `ffprobe -count_frames` -- DECODES the whole stream: 50.4 s on a 468 s
    Ego4D video, a second whole-file pass per video on top of the frame table's own, for a number the
    table already holds. Sharing it makes that pass free once the table is cached.

    Equality is pinned here on the case that motivated the function (172 packets, 138 decodable) so that
    a divergence in ffmpeg's behaviour is caught, rather than silently shifting every window's frame
    indices. Verified independently on a real Ego4D video: both 14043.
    """
    from fastdo.io.video import _count_frames_ffprobe, count_decoded_frames, probe_frame_times

    independent = _count_frames_ffprobe(WHISKING)
    assert count_decoded_frames(WHISKING) == independent == 138
    assert len(probe_frame_times(WHISKING)[0]) == independent


@pytest.mark.parametrize("call,binary", [
    (lambda video, p: video.probe_fps(p), "ffprobe"),
    (lambda video, p: video.decode_frames(p, backend="ffmpeg"), "ffmpeg"),
])
def test_a_missing_ffmpeg_or_ffprobe_names_the_binary(call, binary: str, tmp_path: Path,
                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    """With the binary missing from PATH, the error names it and says how to install it."""
    from fastdo.io import video

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    if binary == "ffmpeg":                        # the ffmpeg backend probes the frame size with ffprobe first
        fake = bin_dir / "ffprobe"
        fake.write_text("#!/bin/sh\necho '{\"streams\": [{\"width\": 4, \"height\": 2}]}'\n")
        fake.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir))
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"")
    with pytest.raises(video.MediaToolNotFound) as info:
        call(video, clip)
    message = str(info.value)
    assert message.startswith(f"{binary} was not found on PATH")
    assert "install ffmpeg (which provides ffprobe) and put it on PATH" in message
    assert isinstance(info.value, FileNotFoundError) and info.value.filename == binary


def test_an_unrelated_file_not_found_error_is_not_rewritten(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastdo.io import video

    def missing_input(cmd, **_kwargs):
        raise FileNotFoundError(2, "No such file or directory", "/no/such/input.mp4")

    monkeypatch.setattr(video.subprocess, "run", missing_input)
    with pytest.raises(FileNotFoundError) as info:
        video.probe_fps(Path("/no/such/input.mp4"))
    assert not isinstance(info.value, video.MediaToolNotFound)
