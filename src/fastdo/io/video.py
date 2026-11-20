"""Video frame decoding at a stride, with a decord/ffmpeg fallback chain.

Backend selection, as tested on 2026-09-12 on the development host (one H100, NVIDIA driver without CUDA 13 support):

- ``torchcodec`` (the plan's first choice) installs cleanly via ``uv add torchcodec``, but its
  default wheel pulls a CUDA 13 ``libnvrtc``; the project's torch is pinned to the CUDA 12.8
  wheel index (see ``pyproject.toml``'s ``[tool.uv.sources]`` comment -- the installed driver
  cannot initialize CUDA 13), so ``import torchcodec`` raises ``OSError: libnvrtc.so.13: cannot
  open shared object file``. Two remediation attempts (using the package as installed, and adding
  a matching ``nvidia-cuda-nvrtc-cu13`` wheel) did not resolve it, so ``torchcodec`` was dropped
  from ``pyproject.toml`` entirely rather than shipping an unusable pinned dependency.
- ``decord`` installs, but it does NOT decode the real ``whisking.mp4`` correctly (found 2026-09-13):
  the container reports 172 frames (``nb_frames`` and packet count) while only 138 decode
  (``ffprobe -count_frames``, the baseline's own frame extraction, HaWoR), and decord trusts the
  container count, so from requested index 87 on it returns the frame 34 earlier -- pixel-identical,
  no error. Edit lists and variable frame rate make this common in internet video. ``decord`` is
  therefore opt-in only (``backend="decord"``) and refuses to decode when its length disagrees with
  the decoded frame count.
- A raw ``ffmpeg`` subprocess pipe (``ffmpeg`` and ``ffprobe`` must be on ``PATH``; a missing binary
  raises :class:`MediaToolNotFound`) is the default backend (``backend="auto"`` or ``"ffmpeg"``); it uses ffmpeg's
  ``select`` filter on the decoded frame index so it only returns the requested strided frames, and
  it matches the true decoded frame sequence exactly (tests/io/test_video.py).

``decord`` lives in this project's optional ``frontend`` extra (not the base install), and is
imported lazily inside ``_decode_decord`` -- so this module, and ``decode_frames`` itself, both
import fine with only ``--extra dev``; without ``--extra frontend`` installed too, every call
simply falls through to the ffmpeg backend (no decord to try). Install both
(``uv sync --extra dev --extra frontend``) to exercise the decord path.
"""

from __future__ import annotations

import errno
import hashlib
import json
import logging
import math
import os
import subprocess
from functools import lru_cache
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)

__all__ = [
    "BACKENDS",
    "MediaToolNotFound",
    "WindowProbeUnusable",
    "addressing_table",
    "count_decoded_frames",
    "decode_frames",
    "decode_window",
    "frame_table_is_cached",
    "packet_table_is_frame_table",
    "packet_table_trim_frames",
    "probe_fps",
    "probe_frame_times",
    "probe_packet_times",
    "probe_start_time",
    "probe_window_times",
    "sample_frame_indices",
    "scale_filter",
    "scaled_dims",
    "screened_frame_count",
]


def sample_frame_indices(total_frames: int, stride: int = 1, max_frames: int | None = None) -> list[int]:
    """Pure helper: the 0-based frame indices ``decode_frames`` selects for a stride/max_frames.

    Args:
        total_frames: Number of frames in the source video.
        stride: Keep every ``stride``-th frame, starting at frame 0.
        max_frames: Cap on the number of returned indices (after striding), or ``None`` for no cap.

    Raises:
        ValueError: If ``stride < 1``.
    """
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    idx = list(range(0, total_frames, stride))
    return idx if max_frames is None else idx[:max_frames]


BACKENDS = ("auto", "ffmpeg", "decord")
MISSING_TOOL_HINT = "install ffmpeg (which provides ffprobe) and put it on PATH"


class MediaToolNotFound(FileNotFoundError):
    """``ffmpeg`` or ``ffprobe`` is not on ``PATH``. A ``FileNotFoundError``, so existing handlers still catch it."""

    def __str__(self) -> str:
        return str(self.strerror)


def _run_media_tool(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
    """``subprocess.run(cmd, **kwargs)``; a missing ``cmd[0]`` binary raises :class:`MediaToolNotFound` naming it."""
    try:
        return subprocess.run(cmd, **kwargs)  # noqa: PLW1510 (every caller passes check=True)
    except FileNotFoundError as exc:
        if exc.filename != cmd[0]:
            raise
        raise MediaToolNotFound(errno.ENOENT, f"{cmd[0]} was not found on PATH: {MISSING_TOOL_HINT}",
                                cmd[0]) from exc


def count_decoded_frames(path: Path) -> int:
    """Number of frames that actually decode from the first video stream.

    The container's ``nb_frames`` and the packet count can both be larger than the decodable frame count
    (edit lists, dropped packets), so this must not read either.

    Served from :func:`probe_frame_times`, whose length is this count BY CONSTRUCTION -- it is the same
    list of decodable frames. That matters because the obvious implementation, ``ffprobe -count_frames``,
    DECODES the whole stream: 50.4 s on a 468 s Ego4D video, a second whole-file pass per video on top of
    the frame table's own. Sharing the table makes the second pass free once the table is cached, and the
    table is cached on disk (see :func:`_frame_table`).

    ``test_count_decoded_frames_agrees_with_ffprobe_count_frames`` pins the equality on the case that
    motivated the function -- whisking.mp4, 172 packets and 138 decodable -- so a divergence in ffmpeg's
    behaviour is caught rather than silently changing every window's frame indices.
    """
    return len(probe_frame_times(path)[0])


def _count_frames_ffprobe(path: Path) -> int:
    """``ffprobe -count_frames``: the independent (and much slower) count, kept so a test can compare."""
    out = _run_media_tool(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
         "stream=nb_read_frames", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    return int(out.stdout.strip().split(",")[0])


def probe_fps(path: Path) -> float:
    """Average frame rate of the first video stream: ffprobe ``avg_frame_rate``, evaluated as a fraction.

    Raises:
        ValueError: If the stream reports no usable rate (e.g. ``0/0``).
    """
    out = _run_media_tool(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=avg_frame_rate",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    rate = out.stdout.strip().split(",")[0]
    num, _, den = rate.partition("/")
    numerator, denominator = int(num), int(den or "1")
    if numerator <= 0 or denominator <= 0:
        raise ValueError(f"{path}: ffprobe reports no usable avg_frame_rate ({rate!r})")
    return numerator / denominator


def decode_frames(path: Path, stride: int = 1, max_frames: int | None = None,
                  backend: str = "auto") -> tuple[Tensor, list[int]]:
    """Decodes frames from a video file at a stride.

    Args:
        path: Video file path.
        stride: Keep every ``stride``-th decoded frame, starting at frame 0.
        max_frames: Cap on the number of returned frames (after striding), or ``None`` for no cap.
        backend: ``"auto"`` or ``"ffmpeg"`` (the default, correct on edit-listed/variable-rate files)
            or ``"decord"`` (opt-in; raises ``ValueError`` when decord's frame count disagrees with
            ``count_decoded_frames``).

    Returns:
        ``(frames, frame_idx)``: ``frames`` is ``(N,H,W,3)`` uint8 (RGB); ``frame_idx`` is the
        list of decoded-frame indices each row of ``frames`` came from.

    Raises:
        ValueError: On an unknown backend, or a decord frame-count mismatch.
    """
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
    path = Path(path)
    if backend == "decord":
        return _decode_decord(path, stride, max_frames)
    return _decode_ffmpeg(path, stride, max_frames)


def _decode_decord(path: Path, stride: int, max_frames: int | None) -> tuple[Tensor, list[int]]:
    import decord

    decord.bridge.set_bridge("native")
    reader = decord.VideoReader(str(path))
    n_decodable = count_decoded_frames(path)
    if len(reader) != n_decodable:
        raise ValueError(
            f"{path}: decord frame count {len(reader)} != decoded frame count {n_decodable}; decord would "
            "return misaligned frames (container count includes non-decoding packets); use backend='ffmpeg'")
    idx = sample_frame_indices(len(reader), stride, max_frames)
    if not idx:
        raise ValueError(f"{path}: no frames selected (stride={stride}, max_frames={max_frames})")
    batch = reader.get_batch(idx).asnumpy()
    return torch.from_numpy(np.ascontiguousarray(batch)), idx


@lru_cache(maxsize=32)
def _dims(path_str: str, _size: int, _mtime_ns: int) -> tuple[int, int]:
    out = _run_media_tool(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
         "-of", "json", path_str],
        capture_output=True, text=True, check=True,
    )
    stream = json.loads(out.stdout)["streams"][0]
    return int(stream["width"]), int(stream["height"])


def _probe_dims(path: Path) -> tuple[int, int]:
    """Frame size of the video stream, cached IN PROCESS on ``(path, size, mtime)``.

    `decode_window` needs the dimensions to cut ffmpeg's rawvideo output into frames, so without this
    cache EVERY window of a video pays its own ffprobe: 0.10 s measured, about a quarter of a short
    window's 0.40 s decode. The key is the one `_frame_table` uses -- a file cannot change its frame
    size without changing its size or its mtime -- so a rewritten file re-probes.
    """
    stat = Path(path).stat()
    return _dims(str(path), stat.st_size, stat.st_mtime_ns)


def probe_frame_times(path: Path) -> tuple[list[float], list[bool]]:
    """Presentation times and keyframe flags of every DECODABLE frame, in presentation order.

    The list is the ground truth for frame indices: its length equals ``count_decoded_frames`` even when
    the container over-reports (whisking.mp4: 172 packets, 138 decodable), and index ``i`` of this list is
    the frame ``decode_frames(..., stride=1)`` returns in row ``i``.

    Args:
        path: Video file path.

    Returns:
        ``(times, key_frame)``: presentation times in seconds, and whether each frame is a keyframe.

    Raises:
        ValueError: If ffprobe reports no usable frame timestamps.
    """
    return _frame_table(str(Path(path)), *_stat_key(path))


def _stat_key(path: Path) -> tuple[int, int]:
    st = Path(path).stat()
    return st.st_size, st.st_mtime_ns


FRAME_TABLE_CACHE_ENV = "FASTDO_FRAME_TABLE_CACHE"


def frame_table_cache_dir() -> Path:
    """Where persisted frame tables live: `$FASTDO_FRAME_TABLE_CACHE`, else `<FASTDO_ROOT>/temp/cache/frame_tables`."""
    override = os.environ.get(FRAME_TABLE_CACHE_ENV)
    if override:
        return Path(override)
    root = Path(os.environ.get("FASTDO_ROOT", Path.cwd()))
    return root / "temp" / "cache" / "frame_tables"


def clear_frame_table_cache() -> None:
    """Drop the in-process table cache. For tests emulating a fresh process; does not touch the disk cache."""
    _frame_table.cache_clear()


def _cache_entry(path_str: str, size: int, mtime_ns: int) -> Path:
    """One file per (path, size, mtime): a CHANGED video gets a different entry rather than a stale hit."""
    key = hashlib.sha256(f"{Path(path_str).resolve()}|{size}|{mtime_ns}".encode()).hexdigest()[:32]
    return frame_table_cache_dir() / f"{key}.npz"


@lru_cache(maxsize=32)
def _frame_table(path_str: str, _size: int, _mtime_ns: int) -> tuple[list[float], list[bool]]:
    """One ffprobe pass per video file, cached IN PROCESS and ON DISK.

    This probe, not the decoding, is what S2 ingest costs. Measured on a 45-window Ego4D run:
    2322.6 of 2439.6 decode seconds -- **95.2%** -- were the first window of each
    video, i.e. this call; every later window of the same video cost 1.3-9.1 s, and `decode_window`
    itself is 1.5-1.7 s regardless of where the window sits. One 468 s video took 64.8 s to probe and
    one took 866.5 s, because the cost scales with the VIDEO's length, not the window's.

    The in-process `lru_cache` amortises that over a single run's windows and loses it on every resume
    and every new process, so the table is also written to `frame_table_cache_dir()`. It holds
    presentation timestamps and keyframe flags -- metadata, not pixels -- so persisting it is outside
    the Ego4D licence's decoded-pixel-cache deletion rule.

    A missing, unreadable or corrupt entry is recomputed rather than raised: the probe is expensive, not
    precious, and a truncated write must not take a run down.

    Fields are read BY NAME from JSON: ffprobe prints `-show_entries` fields in its own order, not the
    requested one, so positional parsing silently swaps `key_frame` into the timestamp column.
    """
    entry = _cache_entry(path_str, _size, _mtime_ns)
    try:
        with np.load(entry) as z:
            return [float(x) for x in z["times"]], [bool(x) for x in z["keys"]]
    except Exception:                     # missing, truncated, or written by an older numpy: re-probe
        pass
    # A MISS is the expensive branch (58.8-866.5 s), and it is otherwise invisible: the cache directory is
    # `FASTDO_ROOT`-relative and `FASTDO_ROOT` defaults to the CWD, so running the same code from a
    # worktree silently re-probes a corpus that was already warmed elsewhere. Measured: a warm 21-video
    # cache under the primary tree shared only 4 of 28 entries with a worktree's. Say so, once per video.
    logger.info("frame-table cache miss for %s (cache dir: %s); probing the whole file",
                Path(path_str).name, frame_table_cache_dir())
    times, keys = _probe_frame_table(path_str)
    try:
        entry.parent.mkdir(parents=True, exist_ok=True)
        # the temp name must END in .npz: np.savez silently APPENDS the extension otherwise, and the
        # os.replace below would then move a file that does not exist.
        tmp = entry.with_name(f"{entry.stem}.tmp{os.getpid()}.npz")
        np.savez(tmp, times=np.asarray(times, dtype=np.float64), keys=np.asarray(keys, dtype=bool))
        os.replace(tmp, entry)            # atomic: a reader never sees a half-written table
    except OSError:
        logger.warning("could not persist the frame table for %s; it will be re-probed", path_str)
    return times, keys


def _probe_frame_table(path_str: str) -> tuple[list[float], list[bool]]:
    """The ffprobe pass itself, without either cache."""
    out = _run_media_tool(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "frame=key_frame,pts_time,best_effort_timestamp_time", "-of", "json", path_str],
        capture_output=True, text=True, check=True,
    )
    rows: list[tuple[float, bool]] = []
    for fr in json.loads(out.stdout).get("frames", []):
        t = next((float(fr[k]) for k in ("best_effort_timestamp_time", "pts_time")
                  if fr.get(k) not in (None, "N/A", "")), None)
        if t is None:
            continue
        rows.append((t, int(fr.get("key_frame", 0)) == 1))
    if not rows:
        raise ValueError(f"{path_str}: ffprobe reported no frame timestamps")
    rows.sort(key=lambda r: r[0])                      # decode order -> presentation order
    return [t for t, _ in rows], [k for _, k in rows]


#: Lookbacks `_cheap_window_times` tries, cheapest first. See its body for the measurement; the last entry
#: is the value the design justified from Ego4D's GOP statistics (mean ~4.3 s, densest ~21 frames).
WINDOW_LOOKBACK_LADDER: tuple[float, ...] = (5.0, 10.0)


def _probe_nominal_rate(path: Path) -> float:
    """ffprobe `r_frame_rate` as a float, or 0.0 when the stream reports none.

    `r_frame_rate`, not `avg_frame_rate`: the span test compares against the rate the container CLAIMS,
    and on a trimmed file the average is computed over what survives the trim, which is the very thing
    being detected.
    """
    out = _run_media_tool(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate",
         "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True)
    num, _, den = out.stdout.strip().split(",")[0].partition("/")
    try:
        numerator, denominator = int(num), int(den or "1")
    except ValueError:
        return 0.0
    return numerator / denominator if numerator > 0 and denominator > 0 else 0.0


def _probe_container_duration(path: Path) -> float | None:
    """ffprobe `format=duration` in seconds, or `None` when the container declares none."""
    out = _run_media_tool(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                          str(path)], capture_output=True, text=True, check=True)
    text = out.stdout.strip()
    try:
        return float(text)
    except ValueError:
        return None


def packet_table_trim_frames(times: list[float], duration_s: float, rate: float) -> int:
    """How many packets the container appears to TRIM, from the packet table's own span.

    Extra packets -- the ones that make a packet table unequal to a frame table -- come from
    container-level trimming: an edit list presents less than the stream contains, so the packets are
    still there and do not decode to presented frames. That trimming is visible without decoding,
    because it makes the packet time SPAN exceed the declared duration. Measured:

    ==============  =======  =======  ==========  ===========
    file            frames   packets  span - dur  this returns
    ==============  =======  =======  ==========  ===========
    trimmed (-ss)      135       180     +1.5000           45
    whisking           138       172     +1.1103           33
    ==============  =======  =======  ==========  ===========

    45 extra against 45 predicted, and 34 against 33 -- so this estimates the DISCREPANCY, not merely
    its existence. Returns <= 0 for a file whose span does not exceed its duration.
    """
    if not times or rate <= 0:
        return 0
    span = max(times) - min(times) + 1.0 / rate
    return round((span - duration_s) * rate)


def packet_table_is_frame_table(path: Path) -> bool:
    """Whether this file's packet table may stand in for its frame table, decided WITHOUT decoding.

    The test is ONE-SIDED -- `span - duration >= 1/rate`, i.e. `packet_table_trim_frames(...) >= 1` --
    and the asymmetry is the whole point. Trimming ADDS packets and INFLATES the span; nothing removes
    packets while inflating it. So a positive excess is evidence of extra packets, while a negative one
    (a duration that merely runs long, which 13 of 21 Ego4D files do) is not.

    Measured over 26 files -- 21 Ego4D, whisking, and four constructed: clean CFR, `-ss` stream-copy
    trimmed, VFR with `avg != r`, and B-frames with an open GOP:

    ================  ==  ==  ==  =============
    screen            TP  FP  TN  FALSE NEGATIVE
    ================  ==  ==  ==  =============
    span (one-sided)   2   0  24              0
    span (two-sided)   2   3  21              0
    ``avg != r``       1   1  23              1
    ================  ==  ==  ==  =============

    **A false negative is the only unsafe cell** -- it accepts a file whose packets are not its frames, and every index
    after the first extra packet is then wrong, silently (the off-by-one-frame decode defect). `avg != r` has one: the
    `-ss` trimmed file reads `avg == r` and carries 45 extra packets. That is why this does NOT use it, and why
    `avg == r` must not gate the substitution.

    **This is not a proof.** The sample has TWO genuine negatives; a corrupt or truncated stream is a
    failure mode it does not contain. Treat a `True` as "no container-level evidence of trimming", and
    keep the decoded frame table as the authority wherever a run can afford it.
    """
    path = Path(path)
    times, _ = probe_packet_times(path)
    rate = _probe_nominal_rate(path)
    duration = _probe_container_duration(path)
    if rate <= 0 or duration is None:
        return False                      # no rate or no duration: nothing to compare the span against
    return packet_table_trim_frames(times, duration, rate) < 1


def probe_packet_times(path: Path) -> tuple[list[float], list[bool]]:
    """Presentation times and keyframe flags from the container's PACKETS, without decoding.

    `probe_frame_times` walks and decodes every frame; this demuxes only, reading what an MP4's sample
    table already stores. On five Ego4D videos the two agree BIT-IDENTICALLY -- equal counts,
    `max|dt| = 0`, zero keyframe-flag differences.

    Speed, measured SAME-PROCESS with a COLD frame-table cache over 203-1033 s videos: **0.10-0.38 s
    against 14.0-235.7 s**, i.e. **141x-963x, median 240x**. Quote the RANGE:
    the ratio is not length-proportional -- a 585 s video gives 963x and a 1033 s one 612x, because the
    frame probe's cost tracks encoding as well as duration -- and an earlier "median 677x" came from
    dividing by frame times measured hours earlier under load, which inflated them 1.13-1.86x.

    **The two are NOT interchangeable, and no cheap check tells you when they differ.** A packet need not
    decode to a frame: `whisking.mp4` has 172 packets and 138 decodable frames. Five candidate
    discriminators were measured and all failed -- `nb_frames` vs `duration x r_frame_rate` MISSES
    a real 270-vs-268 disagreement, `avg_frame_rate != r_frame_rate` falsely rejects a good VFR file,
    `has_b_frames` and `codec_name` do not separate the cases at all, and `-skip_frame nokey` is inert in
    this build.

    What IS measured, across 13 files including three genuine negatives and two built to attack it: the
    packet times are a **superset** of the frame times, with zero exceptions. So a packet table is sound
    for selecting frames **by time** -- every real frame's time is present, and a spurious time decodes to
    nothing -- and unsound for **indexing**, since the extra entries shift every index after them.

    Returns:
        `(times, key_frame)` in presentation order, the same shape as :func:`probe_frame_times`.

    Raises:
        ValueError: If ffprobe reports no packet timestamps.
    """
    out = _run_media_tool(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "packet=pts_time,flags", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    rows: list[tuple[float, bool]] = []
    for pkt in json.loads(out.stdout).get("packets", []):
        t = pkt.get("pts_time")
        if t in (None, "N/A", ""):
            continue
        rows.append((float(t), "K" in (pkt.get("flags") or "")))
    if not rows:
        raise ValueError(f"{path}: ffprobe reported no packet timestamps")
    rows.sort(key=lambda r: r[0])
    return [t for t, _ in rows], [k for _, k in rows]


class WindowProbeUnusable(RuntimeError):
    """The O(window) probe did not cover a requested window, so the caller must use the whole-file table.

    Raised rather than returned because a short frame list is indistinguishable from a correct one at the
    call site: whisking.mp4 (open-GOP, edit list, a single keyframe at presentation index 53) returns an
    EMPTY list for a window at t=0 and drops the leading frames of a window at index 37, while Ego4D's
    files -- keyframes every ~4 s, no edit list -- agree exactly on all 20 windows measured. Cost may
    depend on the local probe working; correctness may not.
    """


def probe_window_times(path: Path, t_start: float, t_stop: float,
                       lookback_s: float = 10.0, pad_s: float = 0.5) -> tuple[list[float], list[bool]]:
    """Presentation times and keyframe flags for ONE time interval, without reading the whole file.

    `ffprobe -read_intervals` restricts the pass to `[t_start − lookback_s, t_stop]`. The lookback exists so
    the interval contains the keyframe at or before `t_start`, which `decode_window`'s seek needs: Ego4D's
    mean GOP is ~128 frames (~4.3 s at 30 fps) and its densest video ~21, so 10 s covers that corpus with
    margin. A caller that finds no keyframe in the returned window must widen it or fall back.

    This is the `O(window)` half of the ingest fix. The whole-file table costs 58.8–866.5 s on a video's
    FIRST window and its price scales with the VIDEO's length, not the window's — the property that breaks
    at internet scale.

    Returns:
        `(times, key_frame)` in presentation order, for the frames ffprobe reports inside the interval.
        The times are absolute, on the same clock as :func:`probe_frame_times`. The list extends `pad_s`
        past `t_stop` and `lookback_s` before `t_start`; callers select the frames they want by time.

    Raises:
        WindowProbeUnusable: If the returned frames do not bracket `[t_start, t_stop]` or contain no
            keyframe at or before `t_start` -- an edit list or a sparse-keyframe file, where the caller
            must fall back to :func:`probe_frame_times`.
    """
    lo = max(0.0, float(t_start) - float(lookback_s))
    # `-read_intervals a%b` excludes the frame sitting exactly at `b`, which costs the window its LAST
    # frame -- measured as exactly 1 missing frame in ALL 20 of 20 real Ego4D windows before `pad_s`
    # existed, and 0 of 20 after. `pad_s` only widens the interval; callers select by time regardless.
    hi = float(t_stop) + float(pad_s)
    out = _run_media_tool(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-read_intervals", f"{lo:.6f}%{hi:.6f}",
         "-show_entries", "frame=key_frame,pts_time,best_effort_timestamp_time", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    rows: list[tuple[float, bool]] = []
    for fr in json.loads(out.stdout).get("frames", []):
        tt = next((float(fr[k]) for k in ("best_effort_timestamp_time", "pts_time")
                   if fr.get(k) not in (None, "N/A", "")), None)
        if tt is None:
            continue
        rows.append((tt, int(fr.get("key_frame", 0)) == 1))
    rows.sort(key=lambda r: r[0])
    times = [x for x, _ in rows]
    key_frame = [k for _, k in rows]

    # Bracketing is the audit: a probe whose frames start after `t_start` or end before `t_stop` has
    # truncated the window, which is what an edit list (ffprobe seeks on unshifted timestamps) and the
    # `%b` endpoint exclusion both do. It cannot prove the absence of an INTERIOR gap -- no local check
    # can, without the table this exists to avoid -- so the window audit in the caller stays the net.
    eps = 1e-6
    if not times:
        raise WindowProbeUnusable(f"{path.name}: no frames in [{lo:.3f}, {hi:.3f}]")
    if times[0] > float(t_start) + eps:
        raise WindowProbeUnusable(f"{path.name}: probe starts at {times[0]:.6f}, after the window's "
                                  f"{float(t_start):.6f}; leading frames would be missing")
    if times[-1] < float(t_stop) - eps:
        raise WindowProbeUnusable(f"{path.name}: probe ends at {times[-1]:.6f}, before the window's "
                                  f"{float(t_stop):.6f}; trailing frames would be missing")
    if not any(k for k, t in zip(key_frame, times) if t <= float(t_start) + eps):
        raise WindowProbeUnusable(f"{path.name}: no keyframe at or before {float(t_start):.6f} within "
                                  f"{lookback_s:.1f} s; a seek to the window start has nothing to land on")
    return times, key_frame


def probe_start_time(path: Path) -> float:
    """First presentation time of the video stream, from the container header (`stream=start_time`).

    Frame `i` sits at `start_time + i / fps`, NOT at `i / fps`: measured on the Ego4D S2 corpus, **4 of 8
    videos start at 21.029 ms rather than 0** — 0.63 of a frame interval at 30 fps, which is larger than
    the half-frame tolerance `_cheap_window_times` matches within. Without this anchor those videos refuse
    the cheap path on every window and silently pay for the whole-file table instead.

    `format=start_time` reads 0 on exactly the files whose `stream=start_time` is 21.029 ms, so the STREAM
    field is the one to ask for.

    Returns:
        The stream's start time in seconds, or 0.0 when the container does not report one.
    """
    out = _run_media_tool(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=start_time",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    text = out.stdout.strip()
    try:
        return float(text)
    except ValueError:                    # "N/A" or empty: a container with no start time starts at 0
        return 0.0


def _cheap_window_times(path: Path, want: list[int], fps: float,
                        start_time: float | None = None) -> tuple[list[float], float, float]:
    """`(times of `want`, seek time, eps)` from a window-local probe -- no whole-file table (design step 3).

    Under constant frame rate absolute index `i` sits at `t ~ start_time + i / fps`, so each wanted index
    is matched to
    the probed frame nearest that time. The match is GUARDED, not assumed: a frame further than half a
    frame interval from its target, or two indices landing on one frame, means the rate is not constant
    enough to address by arithmetic and the caller must pay for the real table. That is the case the table
    exists for -- VFR files where `nb_frames` and the decoded count genuinely disagree.

    Raises:
        WindowProbeUnusable: If the probe cannot cover the window, no frame matches within tolerance, two
            indices collide, or no keyframe precedes the window's first frame.
    """
    t0 = probe_start_time(path) if start_time is None else float(start_time)
    t_start, t_stop = t0 + want[0] / fps, t0 + want[-1] / fps
    # Try the CHEAP lookback first and widen only if the probe says it must. The lookback is most of the
    # interval -- 10 s of lookback around a 10 s window -- and measured on Ego4D, halving it to 5 s nearly
    # halves the probe (9.80 -> 6.61 s and 6.11 -> 3.43 s on two videos) while returning the same window
    # frames. Going below 5 s buys nothing (7.05 s and 3.31 s at 2 s, identical frame counts): ffprobe
    # seeks to a keyframe, so an interval start inside the same GOP does not move it. Widening is SAFE
    # because the audit refuses precisely when the lookback was too short to hold a preceding keyframe;
    # the refusal itself is cheap (0.10 s when the window is past EOF).
    times = keys = None
    for lookback in WINDOW_LOOKBACK_LADDER:
        try:
            times, keys = probe_window_times(path, t_start, t_stop, lookback_s=lookback)
            break
        except WindowProbeUnusable:
            if lookback == WINDOW_LOOKBACK_LADDER[-1]:
                raise

    tol = 0.5 / fps
    picked: list[float] = []
    for i in want:
        target = t0 + i / fps
        j = min(range(len(times)), key=lambda k: abs(times[k] - target))
        if abs(times[j] - target) > tol:
            raise WindowProbeUnusable(f"{path.name}: frame {i} expected near {target:.6f} s, nearest "
                                      f"probed frame is {times[j]:.6f} s (> {tol:.6f} s); not constant rate")
        picked.append(times[j])
    if len(set(picked)) != len(picked):
        raise WindowProbeUnusable(f"{path.name}: two window indices mapped to the same probed frame; "
                                  f"the arithmetic index mapping does not hold on this file")

    seek = next((t for t, k in zip(reversed(times), reversed(keys)) if k and t <= picked[0] + 1e-9), None)
    if seek is None:
        raise WindowProbeUnusable(f"{path.name}: no keyframe at or before {picked[0]:.6f} s in the probe")
    gaps = [b - a for a, b in pairwise(times) if b > a]
    return picked, seek, (0.4 * min(gaps) if gaps else 1e-3)


#: The window protocol's literal filter, with the target height substituted.
SCALE_FILTER = "scale=-2:{height}:flags=area"


def scaled_dims(width: int, height: int, target_height: int) -> tuple[int, int]:
    """Output size of ffmpeg's ``scale=-2:{target_height}`` for a ``width x height`` source.

    The ``-2`` means "derive the width from the aspect ratio, rounded to a multiple of 2". The rounding
    is HALF-UP, not Python's banker's rounding: a 1080x1920 source gives 405 -> **406**, where
    ``round(202.5) * 2`` would give 404. Measured against ffmpeg on seven sizes (see the test), because
    a width that is wrong by two pixels makes every later ``reshape`` silently wrong.

    Two properties of the literal filter that the surrounding prose does not share, both measured:

    * It sets the **height**, not the short side. A PORTRAIT source is scaled by its LONG side:
      1080x1920 -> 406x720. The protocol's prose says "720 px short side"; for portrait input
      the two disagree, and this function implements the filter, which is what the spec pins literally.
    * It does not clamp. A 640x480 source is **upscaled** to 960x720 -- 2.25x the bytes for no extra
      information. See ``scale_filter`` for the guard.

    Neither case occurs in the S2 corpus: all 21 Ego4D videos are landscape with height >= 1080
    (1440x1080 x7, 1920x1440 x5, 1920x1080 x5, 2560x1920 x4), and EK-100 is 1920x1080.
    """
    if width <= 0 or height <= 0 or target_height <= 0:
        raise ValueError(f"dimensions must be positive, got {width}x{height} -> {target_height}")
    half = math.floor(width * target_height / height / 2 + 0.5)      # half-up, as measured
    return int(max(1, half) * 2), int(target_height)


def scale_filter(width: int, height: int, target_height: int) -> tuple[str, tuple[int, int]] | None:
    """``(filter, (w, h))`` for downscaling a ``width x height`` source, or `None` if it would upscale.

    Returning `None` rather than an upscaling filter is the one place this deviates from a literal
    reading of ``scale=-2:720``, and it deviates in the direction the clause exists for: 1.2 wants
    frames no larger than 720, both because the pipeline is validated at 720p and below and because
    7.4's disk budget assumes it. Enlarging a 480p source serves neither -- it costs 2.25x the bytes and
    adds no information a downstream model can use. The caller decodes at source size instead.
    """
    if height <= target_height:
        return None
    return SCALE_FILTER.format(height=int(target_height)), scaled_dims(width, height, target_height)


def frame_table_is_cached(path: Path) -> bool:
    """Whether this file's decoded frame table is already on disk, so reading it costs nothing."""
    path = Path(path)
    try:
        stat = path.stat()
    except OSError:
        return False
    return _cache_entry(str(path), stat.st_size, stat.st_mtime_ns).is_file()


def addressing_table(path: Path, *, allow_packet_table: bool = False) -> tuple[list[float], list[bool]]:
    """`(times, key_frame)` for addressing a window: the decoded frame table, or the packet table.

    The decoded table is authoritative and is used whenever it is **already cached**, because then it is
    free and there is no reason to accept a screened substitute over ground truth. Only on a cache MISS
    -- the expensive branch, 58.8-866.5 s for one video, about 0.84x realtime -- is the packet table
    considered, and then only if `packet_table_is_frame_table` finds no container-level evidence of
    trimming.

    That ordering is the whole design. It means the substitution can only ever be taken where the
    alternative is to spend a second of CPU per second of video, and never where the truth is at hand.

    `allow_packet_table` is OPT-IN: the screen has two genuine negatives behind it, which is thin
    for a safety claim, so a run says explicitly that it accepts that evidence.
    """
    path = Path(path)
    if allow_packet_table and not frame_table_is_cached(path) and packet_table_is_frame_table(path):
        logger.info("%s: addressing from the PACKET table (no cached frame table, screen passed)",
                    path.name)
        return probe_packet_times(path)
    return probe_frame_times(path)


def screened_frame_count(path: Path) -> int:
    """`D_v`'s numerator -- the decoded frame count -- taken from the packet table where the screen allows.

    The window protocol defines `D_v` = decoded frames / `avg_frame_rate`, and counting decoded frames is
    what makes placing a video's windows an O(video) operation: 13.55 s for a 203 s file, up to 235.7 s
    for a 1033 s one. That cost is paid BEFORE anything is known about whether the video is worth using,
    which at internet scale is the wrong place to spend it (screening a video you then discard is
    the regime where this route is worth ~12x).

    Delegates to `addressing_table`, so it inherits that function's ordering exactly: a cached decoded
    table wins, the screened packet table is used only on the expensive branch, and a file with
    container-level evidence of trimming falls back to decoding. Sharing the implementation is
    deliberate -- an independent copy of the rule could drift from the one the decoder uses, and then
    eligibility and addressing would disagree about how many frames a video has.

    This is permitted for a source whose `nb_frames` has been validated against decoded
    counts; the trim screen is what makes it safe to apply per file rather than per corpus. It is
    NOT the default for `probe_windowed_clips`, which still counts frames: a run opts in.
    """
    return len(addressing_table(path, allow_packet_table=True)[0])


def decode_window(path: Path, start: int, stop: int, stride: int = 1, *,
                  local_probe: bool = False,
                  packet_table: bool = False,
                  scale_height: int | None = None,
                  return_times: bool = False) -> tuple[Tensor, list[int]] | tuple[Tensor, list[int], list[float]]:
    """Decodes frames ``start, start+stride, ... < stop`` WITHOUT decoding the whole file.

    ffmpeg seeks to the keyframe at or before the window, then a presentation-time ``select`` keeps
    exactly the wanted frames, so the result does not depend on where the seek landed. Indices are the
    source's own decodable-frame numbering, the same one ``decode_frames`` returns.

    Args:
        path: Video file path.
        start: First source frame index of the window (0-based, inclusive).
        stop: End of the window (exclusive).
        stride: Keep every ``stride``-th frame from ``start``.
        local_probe: Address the window with a window-local ffprobe instead of the whole-file table,
            making ingest independent of the VIDEO's length. OPT-IN PER SOURCE: verified on Ego4D
            (20 of 20 windows exact) and nowhere else, so EK-100 and any new source must be compared
            the same way first. It never changes the result -- a probe that cannot cover the window,
            or a frame rate too variable to address by arithmetic, falls back to the whole-file table.

        packet_table: On a frame-table cache MISS, address the window from the container's PACKET
            table when `packet_table_is_frame_table` passes, instead of decoding the whole video to
            build the frame table. 141x-963x cheaper (median 240x) and OPT-IN, because the screen rests
            on two genuine negatives. A cached frame table is always preferred -- see `addressing_table`.

        scale_height: Downscale the kept rows to this height with the window protocol's
            ``scale=-2:{h}:flags=area`` (the protocol fixes it at 720). `None` (the default) decodes at source
            size, so no existing caller changes. The filter runs AFTER the frame selection, so only the
            kept rows are resampled; and a source already at or below the target is left alone rather
            than enlarged (`scale_filter`).

        return_times: Also return each kept row's PRESENTATION TIME, which this function already knows
            (it selects rows by pts) and which the window protocol requires the decoder to record: "The
            window decoder records each row's best-effort pts." Off by default so the two-value unpacking
            every existing caller uses keeps working.

    Returns:
        ``(frames, frame_idx)``: ``(N,H,W,3)`` uint8 RGB, and the source indices of its rows. With
        ``return_times``, a third element: those rows' presentation times in seconds.

    Raises:
        ValueError: On ``stride < 1``, an empty or reversed range, a range outside the decodable
            frames, or a decode that returns a different number of frames than the window holds.
    """
    path = Path(path)
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    if not 0 <= start < stop:
        raise ValueError(f"{path}: frame range [{start}, {stop}) is empty or reversed")
    want = list(range(start, stop, stride))

    picked: list[float] | None = None
    if local_probe:
        # A window past the end of the file, or a rate too variable to address by arithmetic, refuses here
        # and is caught by the whole-file path below -- including its `stop <= len(times)` bound check.
        try:
            picked, seek, eps = _cheap_window_times(path, want, probe_fps(path))
        except (WindowProbeUnusable, ValueError, subprocess.CalledProcessError) as exc:
            logger.info("%s: window [%d, %d) falls back to the whole-file table (%s)", path.name, start,
                        stop, exc)
            picked = None
    if picked is None:
        times, keys = addressing_table(path, allow_packet_table=packet_table)
        if stop > len(times):
            raise ValueError(f"{path}: frame range [{start}, {stop}) is outside the {len(times)} "
                             f"decodable frames")
        gaps = [b - a for a, b in zip(times, times[1:]) if b > a]
        eps = 0.4 * min(gaps) if gaps else 1e-3
        seek = next((times[i] for i in range(start, -1, -1) if keys[i]), times[0])
        picked = [times[i] for i in want]
    select = "+".join(rf"between(t\,{t - eps:.6f}\,{t + eps:.6f})" for t in picked)
    width, height = _probe_dims(path)
    vf = f"select='{select}'"
    if scale_height is not None:
        # Scale AFTER select: resampling only the kept rows, not every decoded frame in the seek range.
        scaled = scale_filter(width, height, scale_height)
        if scaled is None:
            logger.info("%s: %dx%d is already <= %d high; decoding at source size",
                        path.name, width, height, scale_height)
        else:
            vf, (width, height) = f"{vf},{scaled[0]}", scaled[1]
    # `-to` (in the source timeline, because of `-copyts`) stops the DECODE at the window's end. Without it
    # ffmpeg decodes to end of file and merely drops the unwanted frames in the filter, so an early window
    # costs as much as the whole file: 38.6 s of a 40.3 s full decode, measured on a 467 s Ego4D video.
    cmd = ["ffmpeg", "-v", "error", "-copyts", "-ss", f"{max(0.0, seek):.6f}",
           "-to", f"{picked[-1] + eps:.6f}", "-i", str(path),
           "-vf", vf, "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    result = _run_media_tool(cmd, capture_output=True, check=True)
    frame_bytes = width * height * 3
    n = len(result.stdout) // frame_bytes
    if n != len(want):
        raise ValueError(f"{path}: window [{start}, {stop}) stride {stride} wanted {len(want)} frames, "
                         f"ffmpeg returned {n}")
    raw = np.frombuffer(result.stdout, dtype=np.uint8, count=n * frame_bytes)
    frames = torch.from_numpy(np.ascontiguousarray(raw.reshape(n, height, width, 3)))
    return (frames, want, list(picked)) if return_times else (frames, want)


def _decode_ffmpeg(path: Path, stride: int, max_frames: int | None) -> tuple[Tensor, list[int]]:
    width, height = _probe_dims(path)
    # `\,` escapes the comma from ffmpeg's own filtergraph parser (a filter-chain separator),
    # not from the shell -- subprocess is invoked with an argv list, no shell is involved.
    select = f"select=not(mod(n\\,{stride}))"
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-vf", select, "-vsync", "0",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    result = _run_media_tool(cmd, capture_output=True, check=True)
    frame_bytes = width * height * 3
    n_decoded = len(result.stdout) // frame_bytes
    raw = np.frombuffer(result.stdout, dtype=np.uint8, count=n_decoded * frame_bytes)
    frames = raw.reshape(n_decoded, height, width, 3)
    if max_frames is not None:
        frames = frames[:max_frames]
    idx = [i * stride for i in range(frames.shape[0])]
    if not idx:
        raise ValueError(f"{path}: ffmpeg decoded no frames (stride={stride}, max_frames={max_frames})")
    return torch.from_numpy(np.ascontiguousarray(frames)), idx
