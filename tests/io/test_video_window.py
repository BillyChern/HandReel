"""Window-aware decoding (S2 precondition P4).

EK-100 and Ego4D videos run 2-20 minutes and S2 scores 10 s windows, so a window must decode only its
own range. The contract these tests fix: a window decode returns exactly the frames a full decode
followed by row selection would return, with the SOURCE frame indices, on files whose container frame
count lies (whisking.mp4: 172 packets, 138 decodable) and whose windows start mid-GOP.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar
from unittest import mock

import numpy as np
import pytest
import torch

from fastdo.io import video
from fastdo.io.video import (
    WindowProbeUnusable,
    count_decoded_frames,
    decode_frames,
    decode_window,
    probe_frame_times,
    probe_window_times,
)
from tests.baseline_paths import baseline_path

T = TypeVar("T")
WHISKING = baseline_path("reconstruction/whisking/whisking.mp4")
needs_video = pytest.mark.skipif(not WHISKING.exists() or shutil.which("ffmpeg") is None,
                                 reason="whisking demo video or ffmpeg not available")
needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                                  reason="ffmpeg/ffprobe not available")


@needs_video
def test_probe_frame_times_matches_decoded_count() -> None:
    times, keys = probe_frame_times(WHISKING)
    assert len(times) == count_decoded_frames(WHISKING)          # not the container's 172
    assert times == sorted(times)                                # presentation order
    assert len(keys) == len(times) and any(keys)
    # whisking.mp4 is open-GOP with an edit list: its ONLY keyframe is at presentation index 53, so 53
    # frames precede it. A window starting before it must fall back to decoding from the file start.
    assert [i for i, k in enumerate(keys) if k] == [53]


@needs_video
@pytest.mark.parametrize(("start", "stop", "stride"), [
    (0, 12, 1),        # from the very start
    (37, 61, 1),       # mid-GOP start
    (37, 61, 3),       # mid-GOP start, strided
    (100, 138, 2),     # to the last decodable frame
    (60, 61, 1),       # single frame
])
def test_window_equals_full_decode_selection(start: int, stop: int, stride: int) -> None:
    full, full_idx = decode_frames(WHISKING, stride=1)
    want_idx = list(range(start, stop, stride))
    want = full[[full_idx.index(i) for i in want_idx]]

    got, got_idx = decode_window(WHISKING, start, stop, stride=stride)

    assert got_idx == want_idx
    assert got.shape == want.shape
    assert torch.equal(got, want)                                # byte-identical, not merely close


@needs_video
def test_window_rejects_an_empty_or_reversed_range() -> None:
    with pytest.raises(ValueError, match="range"):
        decode_window(WHISKING, 10, 10)
    with pytest.raises(ValueError, match="range"):
        decode_window(WHISKING, 10, 5)


@needs_video
def test_window_rejects_a_stop_past_the_last_decodable_frame() -> None:
    n = count_decoded_frames(WHISKING)
    with pytest.raises(ValueError, match="range"):
        decode_window(WHISKING, n - 2, n + 5)


def _testsrc(path: Path, frames: int) -> Path:
    """A 640x480 clip long enough that decoding it is not process-startup dominated."""
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=640x480:rate=30",
                    "-frames:v", str(frames), "-g", "30", "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available")
def test_a_window_does_not_pay_for_the_length_of_the_file(tmp_path: Path) -> None:
    """The point of P4, and the defect it guards.

    A `select` filter alone drops the unwanted frames but ffmpeg keeps DECODING to end of file, so an
    early window costs a full decode (measured: 38.6 s of a 40.3 s full decode on a 467 s Ego4D video).
    The window must therefore stop the decode at its own end.

    The contract is that a window's cost does not scale with the VIDEO's length, so that is what this
    measures: the SAME window on a 3x longer file, which cancels the constant a window pays whatever the
    file (process startup and probes, ~0.4 s here). An earlier version asserted `early < 0.5 * full` on
    a single 3,000-frame file. That is a proxy, and it broke on a fast host while the mechanism was
    intact: measured 2026-09-22, a window cost 0.402 s on 3,000 frames and 0.398 s on 6,000 -- flat, as
    designed -- but the short file's full decode was only 0.670 s, putting the ratio at 0.60.

    The full decodes are measured as a POSITIVE CONTROL: they must show that the two files really do
    differ in decode cost, otherwise a flat window cost would prove nothing. Without `-to` the windows
    track the full decodes instead (early_long/early_short ~= 3), which is what the 1.5x bound rejects.

    Cost is CPU seconds -- this thread's plus those of the ffmpeg processes it waited for -- not wall.
    What scales with the file is decode WORK; wall adds however long a loaded host kept the decoder
    waiting for a CPU, which scales with nothing here. Measured 2026-09-28 under cgroup CPU pressure
    (16 niced burners, the container's 17-core quota saturated): the wall positive control failed 2 of 20
    runs (1.714 s against 0.921 s, ratio 1.86), while over 15 rounds on the same host the CPU ratios were
    2.68-3.37 for the full decodes and 0.93-1.17 for single windows (wall: 2.35-3.15 and 0.77-1.48).
    """
    import resource
    import time

    short, long = _testsrc(tmp_path / "short.mp4", 3000), _testsrc(tmp_path / "long.mp4", 9000)
    assert count_decoded_frames(short) == 3000
    assert count_decoded_frames(long) == 9000

    table = probe_frame_times(short)           # the one ffprobe pass per FILE, amortised over its windows
    with mock.patch.object(video.subprocess, "run", side_effect=AssertionError("re-probed")) as run:
        again = probe_frame_times(short)
    assert run.call_count == 0 and again == table, "the frame table must be cached per file"
    probe_frame_times(long)                    # warm both, so the windows below measure the DECODE

    def cpu_seconds(call: Callable[[], T]) -> tuple[T, float]:
        """`call()` and the CPU seconds it burned here and in the child processes it waited for."""
        def now() -> float:
            children = resource.getrusage(resource.RUSAGE_CHILDREN)
            return time.thread_time() + children.ru_utime + children.ru_stime
        t0 = now()
        result = call()
        return result, now() - t0

    def full_decode_seconds(src: Path) -> float:
        return cpu_seconds(lambda: subprocess.run(["ffmpeg", "-v", "error", "-i", str(src), "-f", "null", "-"],
                                                  check=True))[1]

    def window_seconds(src: Path, start: int, stop: int) -> float:
        """The CHEAPEST of three identical windows: contention can only inflate a cost."""
        best = float("inf")
        for _ in range(3):
            (frames, idx), seconds = cpu_seconds(lambda: decode_window(src, start, stop, stride=1))
            best = min(best, seconds)
            assert idx == list(range(start, stop)) and frames.shape[0] == stop - start
        return best

    full_short, full_long = full_decode_seconds(short), full_decode_seconds(long)
    assert full_long > 2.0 * full_short, (
        f"positive control: decoding 9,000 frames ({full_long:.3f} CPU s) must cost materially more than "
        f"3,000 ({full_short:.3f} CPU s), or a flat window cost proves nothing")

    early_short, early_long = window_seconds(short, 60, 90), window_seconds(long, 60, 90)
    late_long = window_seconds(long, 9000 - 30, 9000)
    assert early_long < 1.5 * early_short, (
        f"the same window costs {early_long:.3f} CPU s on the 9,000-frame file against {early_short:.3f} on "
        f"the 3,000-frame one: the decode is running past the window's end")
    assert late_long < 1.5 * early_short, (
        f"a late window costs {late_long:.3f} CPU s against {early_short:.3f} for an early one: the seek is "
        f"not O(1) in the window's position")


def _tiny_video(path: Path, seconds: int = 1) -> Path:
    """A short constant-rate clip: these tests are about the frame TABLE, not about decoding."""
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=10",
                    "-frames:v", str(10 * seconds), "-g", "5", "-pix_fmt", "yuv420p", str(path)], check=True)
    return path

# --- the frame table is the cost of S2 ingest, not the decode -----------------------------------------
#
# Measured on a 45-window Ego4D refusal run: 2322.6 of 2439.6 decode seconds -- 95.2% --
# were the FIRST window of each video, i.e. the whole-file ffprobe. Every later window of the same video
# cost 1.3-9.1 s. One 468 s video took 64.8 s to probe; one took 866.5 s. `decode_window` itself is
# 1.5-1.7 s and, thanks to the seek, independent of where the window sits.
#
# The in-process lru_cache amortises this over one run's windows and loses it on every resume and every
# new process. Persisting the table makes it once per video EVER.


def test_the_frame_table_is_cached_on_disk_across_processes(tmp_path: Path, monkeypatch) -> None:
    """The probe is ~95% of S2's decode cost and an lru_cache does not survive a resume."""
    from fastdo.io import video

    src = _tiny_video(tmp_path / "v.mp4")
    cache = tmp_path / "frame_tables"
    monkeypatch.setenv("FASTDO_FRAME_TABLE_CACHE", str(cache))
    video.clear_frame_table_cache()

    times, keys = video.probe_frame_times(src)
    assert times and len(times) == len(keys)
    written = list(cache.glob("*.npz"))
    assert len(written) == 1, f"expected one cached table, got {written}"

    # a FRESH process would have an empty lru_cache: emulate that, then make ffprobe fail loudly.
    video.clear_frame_table_cache()
    calls: list[object] = []
    real_run = video.subprocess.run

    def no_probe(cmd, *a, **kw):                                     # type: ignore[no-untyped-def]
        if cmd and cmd[0] == "ffprobe":
            calls.append(cmd)
            raise AssertionError("re-probed a video whose frame table is already on disk")
        return real_run(cmd, *a, **kw)

    monkeypatch.setattr(video.subprocess, "run", no_probe)
    again_times, again_keys = video.probe_frame_times(src)
    assert not calls
    assert again_times == times and again_keys == keys


def test_a_changed_file_is_reprobed_not_served_stale(tmp_path: Path, monkeypatch) -> None:
    """The key carries size and mtime: serving a stale table would decode the wrong frames silently."""
    from fastdo.io import video

    src = _tiny_video(tmp_path / "v.mp4")
    monkeypatch.setenv("FASTDO_FRAME_TABLE_CACHE", str(tmp_path / "ft"))
    video.clear_frame_table_cache()
    first = video.probe_frame_times(src)

    longer = _tiny_video(tmp_path / "w.mp4", seconds=2)
    src.write_bytes(longer.read_bytes())                             # same path, different content
    video.clear_frame_table_cache()
    assert len(video.probe_frame_times(src)[0]) != len(first[0])
    assert len(list((tmp_path / "ft").glob("*.npz"))) == 2            # a second entry, not an overwrite


def test_a_corrupt_cache_entry_is_recomputed_rather_than_raised(tmp_path: Path, monkeypatch) -> None:
    """A truncated write must never take the pipeline down: the probe is expensive, not precious."""
    from fastdo.io import video

    src = _tiny_video(tmp_path / "v.mp4")
    cache = tmp_path / "ft"
    monkeypatch.setenv("FASTDO_FRAME_TABLE_CACHE", str(cache))
    video.clear_frame_table_cache()
    expected = video.probe_frame_times(src)

    entry = next(iter(cache.glob("*.npz")))
    entry.write_bytes(b"not an npz")
    video.clear_frame_table_cache()
    assert video.probe_frame_times(src) == expected


# whisking.mp4 is the ADVERSARIAL fixture for the local probe, not a friendly one: open-GOP, an edit list,
# and a single keyframe at presentation index 53. Ego4D's files (keyframes every ~4 s, no edit list) agreed
# exactly on all 20 windows measured; this file does not, which is why the probe audits itself.
@needs_video
@pytest.mark.parametrize(("start", "stop"), [(60, 80), (100, 138), (60, 61)])
def test_window_probe_agrees_with_the_whole_file_table(start: int, stop: int) -> None:
    """Where the local probe answers at all, it must report the window EXACTLY as the O(video) table does.

    This is the property the ingest fix rests on. The probe may return extra frames on either side (it is
    asked for a padded interval), but every frame the window wants must be present, at the same
    presentation time and with the same keyframe flag -- otherwise a cheaper probe would silently hand the
    solver a different window than the one the manifest names.
    """
    times, keys = probe_frame_times(WHISKING)
    want_t, want_k = times[start:stop], keys[start:stop]

    got_t, got_k = probe_window_times(WHISKING, want_t[0], want_t[-1])

    at = {round(t, 6): k for t, k in zip(got_t, got_k)}
    assert got_t == sorted(got_t)
    missing = [t for t in want_t if round(t, 6) not in at]
    assert missing == [], f"window frames absent from the local probe: {missing}"
    assert [at[round(t, 6)] for t in want_t] == want_k


@needs_video
def test_window_probe_keeps_the_frame_sitting_on_the_interval_end() -> None:
    """`-read_intervals a%b` drops the frame at exactly `b`; `pad_s` is the only reason it survives.

    Measured on real Ego4D windows before the fix: exactly one frame missing in 20 of 20 windows, always
    the last. The zero-pad half is the regression itself -- without it the padded half would pass just as
    happily against a probe that had never needed padding at all.
    """
    times, _ = probe_frame_times(WHISKING)
    t0, t1 = times[60], times[79]

    padded, _ = probe_window_times(WHISKING, t0, t1)
    assert round(t1, 6) in {round(t, 6) for t in padded}

    with pytest.raises(WindowProbeUnusable, match="trailing frames"):
        probe_window_times(WHISKING, t0, t1, pad_s=0.0)


@needs_video
@pytest.mark.parametrize(("start", "stop", "why"), [
    (0, 12, "keyframe|no frames"),     # nothing to seek to: the file's only keyframe is at index 53
    (37, 61, "keyframe|leading"),      # window opens mid-GOP, before that keyframe
])
def test_window_probe_refuses_rather_than_returning_a_short_window(start: int, stop: int, why: str) -> None:
    """A window the local probe cannot cover raises, so cost -- never correctness -- depends on it.

    A truncated frame list is indistinguishable from a correct one at the call site, so returning one
    would make every caller responsible for an audit it has no ground truth for. Raising also keeps the
    t=0 case honest: the interval start clamps to 0 rather than going negative, and what comes back is
    this exception, not an ffprobe error.
    """
    times, _ = probe_frame_times(WHISKING)
    with pytest.raises(WindowProbeUnusable, match=why):
        probe_window_times(WHISKING, times[start], times[stop - 1])


@pytest.fixture(scope="module")
def cfr_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A synthetic constant-rate clip with a 10-frame GOP -- the regime the cheap path is FOR.

    whisking.mp4 cannot exercise it: its single keyframe sends every early window down the fallback, so a
    suite built only on that file would report the fallback working and never touch the code that makes
    ingest independent of video length.
    """
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available")
    out = tmp_path_factory.mktemp("cfr") / "cfr.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=30:duration=4",
                    "-c:v", "libx264", "-g", "10", "-pix_fmt", "yuv420p", str(out)], check=True)
    return out


@pytest.mark.parametrize(("start", "stop", "stride"), [(0, 12, 1), (37, 61, 1), (37, 61, 3), (60, 61, 1)])
def test_local_probe_decodes_the_identical_window(cfr_video: Path, start: int, stop: int,
                                                  stride: int) -> None:
    """The cheap path is a COST change, not a result change: same frames, same indices, byte for byte.

    Asserting equality against the whole-file path is the only check that can fail for the right reason.
    A test that merely decoded the window and looked at its shape would pass against an off-by-one index
    mapping, which is exactly the failure the arithmetic `t ~ i/fps` mapping risks.
    """
    want, want_idx = decode_window(cfr_video, start, stop, stride=stride)
    got, got_idx = decode_window(cfr_video, start, stop, stride=stride, local_probe=True)

    assert got_idx == want_idx
    assert torch.equal(got, want)


def test_local_probe_actually_took_the_cheap_path(cfr_video: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The cheap path must be REACHED on this file, not silently falling back and passing anyway.

    "Implemented is not reachable" has cost this project five separate times, and a fallback makes it
    invisible: every equality test above would pass just as happily if `local_probe=True` never once
    avoided the whole-file table. So forbid the table and require the decode to still succeed.
    """
    from fastdo.io import video

    def forbidden(*_a: object, **_k: object) -> None:
        raise AssertionError("decode_window read the whole-file table despite local_probe=True")

    monkeypatch.setattr(video, "probe_frame_times", forbidden)
    got, got_idx = video.decode_window(cfr_video, 37, 61, stride=3, local_probe=True)
    assert got_idx == [37, 40, 43, 46, 49, 52, 55, 58]
    assert got.shape[0] == 8


@needs_video
def test_local_probe_falls_back_rather_than_guessing(caplog: pytest.LogCaptureFixture) -> None:
    """On whisking.mp4 the cheap path cannot work, and the answer must be right anyway.

    One keyframe at presentation index 53 and an edit list: the probe refuses, and `decode_window` owes
    the caller the same frames it would have returned without the flag. Correctness never depends on the
    assumption; only cost does.
    """
    want, want_idx = decode_window(WHISKING, 37, 61, stride=3)
    with caplog.at_level("INFO", logger="fastdo.io.video"):
        got, got_idx = decode_window(WHISKING, 37, 61, stride=3, local_probe=True)

    assert got_idx == want_idx
    assert torch.equal(got, want)
    assert any("falls back" in r.getMessage() for r in caplog.records), \
        "the fallback must say so, or a silent slow path is invisible"


def test_the_cheap_lookback_rung_gives_the_same_window_as_the_expensive_one(cfr_video: Path) -> None:
    """The ladder is safe because a shorter lookback does not change WHICH frames come back.

    Measured, and not what was assumed: ffprobe seeks to the keyframe at or before the interval start, so
    the nominal lookback stops mattering once it is below the GOP -- on Ego4D, 2 s and 5 s returned
    identical frame counts (7.05 s vs 6.61 s) while 10 s cost nearly double (9.80 s). The saving is real
    and it comes from seeking less far back, not from asking for fewer frames. If a future ffmpeg honours
    the interval start literally, the cheap rung would start returning a SHORTER list and this fails.
    """
    from fastdo.io.video import probe_frame_times

    times, _ = probe_frame_times(cfr_video)
    t0, t1 = times[37], times[60]

    cheap, cheap_k = probe_window_times(cfr_video, t0, t1, lookback_s=5.0)
    wide, wide_k = probe_window_times(cfr_video, t0, t1, lookback_s=10.0)

    want = {round(t, 6) for t in times[37:61]}
    assert want <= {round(t, 6) for t in cheap}, "the cheap rung dropped window frames"
    assert {round(t, 6): k for t, k in zip(cheap, cheap_k)} | {round(t, 6): k for t, k in zip(wide, wide_k)} \
        == {round(t, 6): k for t, k in zip(wide, wide_k)} | {round(t, 6): k for t, k in zip(cheap, cheap_k)}


@needs_video
def test_the_ladder_propagates_the_refusal_when_every_rung_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """When no rung can cover the window, the LAST refusal reaches the caller rather than being swallowed.

    whisking.mp4's early windows cannot be covered at any lookback (its only keyframe is at presentation
    index 53), so this is the real case, not a contrived one -- and `decode_window` depends on the raise
    to know it must pay for the whole-file table.
    """
    from fastdo.io import video

    monkeypatch.setattr(video, "WINDOW_LOOKBACK_LADDER", (2.0, 5.0, 10.0))
    with pytest.raises(video.WindowProbeUnusable):
        video._cheap_window_times(WHISKING, [0, 3, 6], video.probe_fps(WHISKING))


def test_probe_start_time_is_a_number_and_defaults_to_zero(cfr_video: Path) -> None:
    """A container with no reported start time starts at 0, rather than raising or returning 'N/A'."""
    from fastdo.io.video import probe_start_time

    assert probe_start_time(cfr_video) == pytest.approx(0.0, abs=1e-6)


def test_a_wrong_anchor_SHIFTS_the_window_silently(cfr_video: Path) -> None:
    """The half-frame tolerance does NOT guard the absolute index mapping. This pins that, deliberately.

    Measured: with the anchor wrong by 0.63 of a frame -- exactly the Ego4D case, where 4 of 8 videos
    start at 21.029 ms rather than 0 -- `_cheap_window_times` returns frames 38, 41, 44 when asked for
    37, 40, 43, and does NOT refuse. It cannot: frames are dense, so a target that is 0.63 frames wrong
    still sits 0.37 frames from its NEIGHBOUR, comfortably inside the tolerance. The tolerance guards
    local rate variation; nothing in the local probe can see a global clock offset.

    That is why `probe_start_time` exists, why `local_probe` is default off, and why enabling it for a
    source requires comparing the INDEX MAPPING against the whole-file table -- not merely the probe,
    which agrees 20/20 on Ego4D either way. Gate A does not catch it either: `frame_idx` records the
    window we INTENDED, so a shifted decode still writes the expected indices.

    If a future change makes a wrong anchor refuse instead of shifting, this test should fail and be
    replaced by the stronger guarantee.
    """
    from fastdo.io.video import _cheap_window_times, probe_fps, probe_frame_times

    times, _ = probe_frame_times(cfr_video)
    fps = probe_fps(cfr_video)
    want = [37, 40, 43]

    def source_indices(start_time: float) -> list[int]:
        picked, _seek, _eps = _cheap_window_times(cfr_video, want, fps, start_time=start_time)
        return [min(range(len(times)), key=lambda k: abs(times[k] - t)) for t in picked]

    assert source_indices(0.0) == want
    assert source_indices(0.63 / fps) == [38, 41, 44], "the silent shift this pins has changed"


def test_the_start_time_anchor_is_load_bearing(cfr_video: Path) -> None:
    """A GROSSLY wrong anchor refuses; see the test above for the sub-frame case, which does not.

    Measured on Ego4D: 4 of 8 videos start at 21.029 ms, not 0 -- 0.63 of a frame at 30 fps, wider than
    the half-frame match tolerance. Before the anchor those videos refused the cheap path on every window
    and silently paid for the whole-file table. An anchor that were merely decorative would leave this
    test passing with a deliberately wrong offset, so the assertion is on the REFUSAL.
    """
    from fastdo.io.video import WindowProbeUnusable, _cheap_window_times, probe_fps

    fps = probe_fps(cfr_video)
    want = [37, 40, 43]

    picked, _seek, _eps = _cheap_window_times(cfr_video, want, fps, start_time=0.0)
    assert len(picked) == len(want)

    # A whole-frame-plus anchor error DOES refuse, because then no probed frame is within tolerance of
    # the LAST wanted index once the interval itself has shifted past it.
    with pytest.raises(WindowProbeUnusable):
        _cheap_window_times(cfr_video, want, fps, start_time=60.0)


@needs_video
def test_packet_times_are_a_superset_of_frame_times_even_when_counts_differ() -> None:
    """The property the container-index shortcut rests on, pinned on the file that violates equality.

    `whisking.mp4` has 172 packets and 138 decodable frames, so packets are NOT frames. What holds -- and
    what makes time-addressing sound where index-addressing is not -- is that every real frame's time is
    still among the packet times. Measured across 13 files including three genuine negatives; this pins
    the worst of them.

    If this ever fails, a window addressed from packet times would silently LOSE a frame, which is why
    `decode_window` keeps its count assertion and why the timebase check exists.
    """
    from fastdo.io.video import probe_packet_times

    ptimes, pkeys = probe_packet_times(WHISKING)
    ftimes, _ = probe_frame_times(WHISKING)

    assert len(ptimes) == 172 and len(ftimes) == 138, "the fixture's counts have changed"
    at = {round(t, 6) for t in ptimes}
    assert [t for t in ftimes if round(t, 6) not in at] == [], "packet times must be a superset"
    assert len(pkeys) == len(ptimes) and any(pkeys)


def test_packet_times_equal_the_frame_table_on_a_well_formed_file(cfr_video: Path) -> None:
    """Where packets and frames agree in count they agree exactly -- times and keyframe flags.

    This is the case the 677x saving is for: five Ego4D videos and two EK-100 videos are bit-identical
    (`max|dt| = 0`, zero flag differences). The synthetic CFR fixture reproduces it without needing the
    corpus, so the property is checked on every run rather than only when someone remembers to.
    """
    from fastdo.io.video import probe_packet_times

    ptimes, pkeys = probe_packet_times(cfr_video)
    ftimes, fkeys = probe_frame_times(cfr_video)

    assert len(ptimes) == len(ftimes)
    assert [round(t, 6) for t in ptimes] == [round(t, 6) for t in ftimes]
    assert pkeys == fkeys


# --- the 720p decode downscale ----------------------------------------------------------------

# Measured against ffmpeg's own `scale=-2:720:flags=area`, one testsrc per row, 2026-09-21. The table is
# the specification here: a width wrong by two pixels makes every later reshape silently wrong, and the
# obvious formula (`round(w * 720 / h / 2) * 2`) gets the last row WRONG -- Python's round is banker's,
# so 202.5 -> 202 -> 404, where ffmpeg gives 406.
FFMPEG_SCALE_TABLE = [
    ((1440, 1080), (960, 720)),      # Ego4D, 7 of 21
    ((1920, 1080), (1280, 720)),     # Ego4D 5 of 21, and EK-100
    ((1920, 1440), (960, 720)),      # Ego4D, 5 of 21
    ((2560, 1920), (960, 720)),      # Ego4D, 4 of 21
    ((1280, 720), (1280, 720)),      # already at the target
    ((640, 480), (960, 720)),        # UPSCALES -- the literal filter does not clamp
    ((1080, 1920), (406, 720)),      # PORTRAIT: the filter scales the LONG side; 405 rounds UP to 406
]


@pytest.mark.parametrize(("src", "want"), FFMPEG_SCALE_TABLE)
def test_scaled_dims_reproduces_ffmpeg(src: tuple[int, int], want: tuple[int, int]) -> None:
    assert video.scaled_dims(src[0], src[1], 720) == want


@needs_ffmpeg
@pytest.mark.parametrize(("src", "want"), FFMPEG_SCALE_TABLE)
def test_scaled_dims_still_reproduces_ffmpeg_today(tmp_path: Path, src: tuple[int, int],
                                                   want: tuple[int, int]) -> None:
    """The table above is a measurement, and a measurement can go stale when ffmpeg changes. This re-runs
    it against the ffmpeg actually installed, so the pin is to the tool's behaviour, not to my notes."""
    out = tmp_path / "s.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
                    "-i", f"testsrc=size={src[0]}x{src[1]}:rate=10:duration=1",
                    "-vf", "scale=-2:720:flags=area", "-frames:v", "1", "-pix_fmt", "yuv420p", str(out)],
                   check=True)
    probed = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                             "stream=width,height", "-of", "csv=p=0", str(out)],
                            capture_output=True, text=True, check=True)
    assert tuple(int(x) for x in probed.stdout.strip().split(",")) == want
    assert video.scaled_dims(src[0], src[1], 720) == want


def test_scale_filter_refuses_to_upscale() -> None:
    """The one deliberate deviation from a literal `scale=-2:720`. 1.2 wants frames no LARGER than 720 --
    the pipeline is validated at 720p and below, and 7.4's budget assumes it. Enlarging a 480p source
    serves neither goal and costs 2.25x the bytes, so the caller decodes at source size instead."""
    assert video.scale_filter(640, 480, 720) is None
    assert video.scale_filter(1280, 720, 720) is None                  # already there: no resample at all
    assert video.scale_filter(1920, 1080, 720) == ("scale=-2:720:flags=area", (1280, 720))


@needs_ffmpeg
def test_the_scaled_window_is_byte_identical_to_a_sequential_decode(tmp_path: Path) -> None:
    """The downscale's equivalence test, in miniature: "the rows must be byte-identical to a
    sequential decode from frame 0 at the same scale". The reference is built by a DIFFERENT command --
    no seek, no select, scale only -- so this compares against an independent construction rather than
    asking the window decoder to confirm itself."""
    src = tmp_path / "src.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
                    "-i", "testsrc=size=1440x1080:rate=10:duration=2",
                    "-pix_fmt", "yuv420p", str(src)], check=True)

    frames, idx = video.decode_window(src, 2, 14, 3, scale_height=720)
    assert frames.shape[1:] == (720, 960, 3) and idx == [2, 5, 8, 11]

    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(src), "-vf", "scale=-2:720:flags=area",
                          "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    ref = np.frombuffer(raw, np.uint8).reshape(-1, 720, 960, 3)
    assert np.array_equal(frames.numpy(), ref[idx])


@needs_ffmpeg
def test_an_unscaled_window_is_unchanged(tmp_path: Path) -> None:
    """`scale_height=None` is the default, so every existing caller decodes exactly as before."""
    src = tmp_path / "src.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
                    "-i", "testsrc=size=1440x1080:rate=10:duration=2",
                    "-pix_fmt", "yuv420p", str(src)], check=True)
    frames, idx = video.decode_window(src, 2, 14, 3)
    assert frames.shape[1:] == (1080, 1440, 3) and idx == [2, 5, 8, 11]


# --- may a packet table stand in for a frame table? ---------------------------------------------------

@pytest.fixture           # the tests that use it carry `needs_ffmpeg`; a mark on a fixture is an error
def _packet_corpus(tmp_path: Path) -> dict[str, Path]:
    """Four files that between them exercise both ways the screen can be wrong.

    `trimmed` is the important one: a stream-copy `-ss` leaves an edit list, so the packets are all still
    there and 45 of them do not decode to presented frames. It reads `avg == r`, which is why the
    `avg != r` screen accepts it -- the one unsafe cell.
    """
    def src(name: str, *args: str) -> Path:
        out = tmp_path / f"{name}.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", *args, str(out)], check=True)
        return out

    clean = src("clean", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=30:duration=6",
                "-pix_fmt", "yuv420p")
    return {
        "clean": clean,
        "trimmed": src("trimmed", "-ss", "1.5", "-i", str(clean), "-c", "copy"),
        "vfr": src("vfr", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=30:duration=6",
                   "-vf", r"select='not(mod(n\,3))+gt(n\,90)'", "-vsync", "vfr", "-pix_fmt", "yuv420p"),
        "bframes": src("bframes", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=30:duration=6",
                       "-c:v", "libx264", "-bf", "3", "-g", "15", "-pix_fmt", "yuv420p"),
        # A container whose DURATION runs longer than its video packets -- here because a longer audio
        # track sets the format duration. 13 of 21 Ego4D files have this shape (by 0.02-0.47 s), and a
        # two-sided |span - duration| test refuses them all for nothing. Nothing is trimmed.
        "long_duration": src("long_duration", "-f", "lavfi", "-i",
                             "testsrc=size=64x48:rate=30:duration=6", "-f", "lavfi", "-i",
                             "anullsrc=r=44100:cl=mono", "-t", "6.5", "-c:v", "libx264",
                             "-pix_fmt", "yuv420p", "-c:a", "aac"),
    }


@needs_ffmpeg
def test_the_screen_refuses_a_trimmed_file_and_accepts_the_rest(_packet_corpus: dict[str, Path]) -> None:
    """The only unsafe cell is a FALSE NEGATIVE -- accepting a file whose packets are not its frames,
    which shifts every index after the first extra packet, silently (a one-frame-shift defect)."""
    for name in ("clean", "vfr", "bframes", "long_duration"):
        path = _packet_corpus[name]
        assert len(video.probe_packet_times(path)[0]) == len(video.probe_frame_times(path)[0]), name
        assert video.packet_table_is_frame_table(path) is True, name

    trimmed = _packet_corpus["trimmed"]
    extra = len(video.probe_packet_times(trimmed)[0]) - len(video.probe_frame_times(trimmed)[0])
    assert extra > 0, "the -ss stream copy did not leave extra packets; the fixture is not testing anything"
    assert video.packet_table_is_frame_table(trimmed) is False


@needs_ffmpeg
def test_the_screen_estimates_how_many_packets_are_trimmed(_packet_corpus: dict[str, Path]) -> None:
    """It does not only flag: `span - duration` in frames tracks the actual excess. On the constructed
    trim this is exact, and on whisking it is 33 against 34."""
    trimmed = _packet_corpus["trimmed"]
    times, _ = video.probe_packet_times(trimmed)
    extra = len(times) - len(video.probe_frame_times(trimmed)[0])
    predicted = video.packet_table_trim_frames(times, video._probe_container_duration(trimmed),
                                               video._probe_nominal_rate(trimmed))
    assert abs(predicted - extra) <= 1, (predicted, extra)


@needs_ffmpeg
def test_avg_vs_r_would_have_accepted_the_trimmed_file(_packet_corpus: dict[str, Path]) -> None:
    """Pinning the REFUTATION, not just the replacement. `avg == r` was floated as a possible gate; this
    file reads `avg == r` and carries extra packets, so that gate has a false negative and is withdrawn.
    A test that only asserted the new screen works would leave the old idea available to be re-proposed."""
    import subprocess as sp

    trimmed = _packet_corpus["trimmed"]
    rates = sp.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                    "stream=avg_frame_rate,r_frame_rate", "-of", "csv=p=0", str(trimmed)],
                   capture_output=True, text=True, check=True).stdout.strip().split(",")
    assert rates[0] == rates[1], f"the fixture no longer exhibits avg == r ({rates})"
    assert len(video.probe_packet_times(trimmed)[0]) > len(video.probe_frame_times(trimmed)[0])
    assert video.packet_table_is_frame_table(trimmed) is False        # the span screen still refuses it


@needs_ffmpeg
def test_a_duration_that_runs_long_is_still_accepted(_packet_corpus: dict[str, Path]) -> None:
    """One-sidedness at the DECISION, not just in the helper. 13 of 21 Ego4D files declare a duration
    longer than their video packets span, by 0.02-0.47 s, with nothing trimmed; a two-sided
    `|span - duration| >= 1/rate` refuses three of them and buys nothing. This file has that shape."""
    path = _packet_corpus["long_duration"]
    times, _ = video.probe_packet_times(path)
    excess = video.packet_table_trim_frames(times, video._probe_container_duration(path),
                                            video._probe_nominal_rate(path))
    assert excess <= -1, f"the fixture no longer runs long (excess {excess}); it tests nothing"
    assert len(times) == len(video.probe_frame_times(path)[0])       # and nothing is actually trimmed
    assert video.packet_table_is_frame_table(path) is True


def test_the_trim_estimate_is_one_sided() -> None:
    """A duration that runs LONG is not evidence of extra packets, and 13 of 21 Ego4D files run long by
    0.02-0.47 s. Only a span that EXCEEDS the duration indicates trimming, so the estimate must go
    negative rather than absolute there -- otherwise the screen refuses most of the corpus for nothing."""
    rate = 30.0
    exact = [i / rate for i in range(300)]
    assert video.packet_table_trim_frames(exact, 10.0, rate) == 0
    assert video.packet_table_trim_frames(exact, 10.5, rate) < 0       # duration runs long: NOT trimming
    assert video.packet_table_trim_frames(exact, 8.5, rate) == 45      # span exceeds: 1.5 s of packets
    assert video.packet_table_trim_frames([], 10.0, rate) == 0
    assert video.packet_table_trim_frames(exact, 10.0, 0.0) == 0       # no rate: nothing to compare


@needs_ffmpeg
def test_the_packet_route_decodes_identical_pixels(_packet_corpus: dict[str, Path],
                                                   monkeypatch: pytest.MonkeyPatch,
                                                   tmp_path: Path) -> None:
    """The claim that matters: on a file the screen accepts, addressing from the packet table returns the
    SAME frames as addressing from the decoded table -- byte for byte, with the same indices.

    Everything else about this route is an argument that the substitution is safe; this is the only test
    that checks the substitution itself. It is run with a cold cache directory, because the packet route
    is only ever taken on a cache miss.
    """
    monkeypatch.setenv(video.FRAME_TABLE_CACHE_ENV, str(tmp_path / "cold"))
    video.clear_frame_table_cache()

    for name in ("clean", "vfr", "bframes", "long_duration"):
        path = _packet_corpus[name]
        assert video.packet_table_is_frame_table(path), name
        by_frames, idx_f = video.decode_window(path, 6, 30, 3)
        by_packets, idx_p = video.decode_window(path, 6, 30, 3, packet_table=True)
        assert idx_p == idx_f, name
        assert torch.equal(by_packets, by_frames), name


@needs_ffmpeg
def test_a_cached_frame_table_beats_the_packet_table(_packet_corpus: dict[str, Path],
                                                     monkeypatch: pytest.MonkeyPatch,
                                                     tmp_path: Path) -> None:
    """Ground truth wins when it is free. The screen rests on two genuine negatives, so it is taken only
    on the expensive branch -- never in place of a table already on disk."""
    monkeypatch.setenv(video.FRAME_TABLE_CACHE_ENV, str(tmp_path / "warm"))
    video.clear_frame_table_cache()
    path = _packet_corpus["clean"]

    assert not video.frame_table_is_cached(path)
    video.probe_frame_times(path)                                   # pay for it once
    assert video.frame_table_is_cached(path)

    calls: list[Path] = []
    real_packets = video.probe_packet_times
    monkeypatch.setattr(video, "probe_packet_times",
                        lambda p, *a, **k: (calls.append(Path(p)), real_packets(p, *a, **k))[1])
    video.addressing_table(path, allow_packet_table=True)
    assert calls == [], "a cached frame table must not be replaced by a screened substitute"


@needs_ffmpeg
def test_a_trimmed_file_falls_back_to_the_frame_table(_packet_corpus: dict[str, Path],
                                                      monkeypatch: pytest.MonkeyPatch,
                                                      tmp_path: Path) -> None:
    """The refusal reaches the decode. `trimmed` has 45 extra packets, so addressing from them would
    shift every index past the first one -- the whole point of the screen."""
    monkeypatch.setenv(video.FRAME_TABLE_CACHE_ENV, str(tmp_path / "cold"))
    video.clear_frame_table_cache()
    path = _packet_corpus["trimmed"]

    table = video.addressing_table(path, allow_packet_table=True)
    assert table == video.probe_frame_times(path)                   # the decoded table, not the packets
    assert len(table[0]) < len(video.probe_packet_times(path)[0])

    by_frames, idx_f = video.decode_window(path, 6, 30, 3)
    by_packets, idx_p = video.decode_window(path, 6, 30, 3, packet_table=True)
    assert idx_p == idx_f and torch.equal(by_packets, by_frames)


@needs_ffmpeg
def test_the_screened_count_equals_the_decoded_count(_packet_corpus: dict[str, Path],
                                                     monkeypatch: pytest.MonkeyPatch,
                                                     tmp_path: Path) -> None:
    """`D_v`'s numerator by the screened route must equal the decoded count -- on files the screen accepts
    AND on the trimmed one it refuses, where the fallback has to supply the right answer.

    A wrong count here does not raise: it places a different set of windows, which is a silently
    different pre-registered sample (one case: a container duration admitted an 88th window whose last
    index is 26664 of 26660)."""
    monkeypatch.setenv(video.FRAME_TABLE_CACHE_ENV, str(tmp_path / "cold"))
    video.clear_frame_table_cache()

    for name, path in _packet_corpus.items():
        video.clear_frame_table_cache()
        assert video.screened_frame_count(path) == video.count_decoded_frames(path), name

    trimmed = _packet_corpus["trimmed"]
    assert video.count_decoded_frames(trimmed) < len(video.probe_packet_times(trimmed)[0])


@needs_ffmpeg
def test_the_frame_size_probe_is_cached_per_file(tmp_path: Path) -> None:
    """Every window of a video needs the frame size, and an ffprobe per window is 0.10 s wasted.

    A file cannot change its frame size without changing its size or its mtime, so those key the cache
    -- the same key the frame table uses. The mutation control is the second half: rewriting the file at
    a different size must re-probe, not serve the stale dimensions.
    """
    src = _testsrc(tmp_path / "dims.mp4", 30)
    calls: list[list[str]] = []
    real_run = subprocess.run

    def counting_run(cmd, *a, **k):           # type: ignore[no-untyped-def]
        calls.append(list(cmd))
        return real_run(cmd, *a, **k)

    with mock.patch.object(video.subprocess, "run", counting_run):
        assert video._probe_dims(src) == (640, 480)
        assert video._probe_dims(src) == (640, 480)
        assert len([c for c in calls if c[0] == "ffprobe"]) == 1, "the frame size must be probed once"

        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=30",
                        "-frames:v", "30", "-pix_fmt", "yuv420p", str(src)], check=True)
        assert video._probe_dims(src) == (320, 240), "a rewritten file must be re-probed"
