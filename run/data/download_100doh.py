"""Download 10 s YouTube clips referenced by the 100DOH (100 Days of Hands) dataset.

100DOH (https://fouheylab.eecs.umich.edu/~dandans/projects/100DOH/) does not distribute raw
video; it distributes a metadata archive (``file.zip``, see ``run/data/sources.yaml`` ->
``datasets: [name: 100doh]``) listing every source YouTube video plus every annotated frame
sampled from it, and expects users to re-download the videos themselves.

**Terms.** The clips are YouTube videos. You are responsible for complying with YouTube's Terms of Service and the
videos' copyright when you download, keep or use them; this repository ships none of them.

Metadata format (unzipped from ``file.zip``, both are whitespace-column-aligned text tables
safe to parse with ``str.split()`` since no field contains embedded whitespace):

``meta_100DOH_video.txt`` -- one row per source video:
    video_id, genre, duration(s), resolution, frame_rate, frame_number, creation_time, link

``meta_100K_frame.txt`` -- one row per annotated frame (99899 rows over 27306 distinct videos):
    image_name, video_id, frame_index, genre, duration, resolution, frame_rate, frame_number,
    creation_time, link

Per ``README_100K.md`` in the archive, frames were extracted with
``ffmpeg -i video -vf fps=0.5 ...`` (one frame every 2 s), so the 1-based ``frame_index`` for a
frame maps to an approximate timestamp in the source video via
``t = (frame_index - 1) / 0.5 = (frame_index - 1) * 2.0`` seconds. This module uses that
timestamp as the "annotated frame time".

Clip selection: 100DOH videos can contribute many annotated frames each (spaced 2 s apart across
the whole video). To keep ``--max-clips`` a simple, predictable budget knob (rather than an
input-dependent multiplier), this downloader groups frames by video and takes exactly one clip
per video -- centered on that video's earliest annotated frame -- for the first ``--max-clips``
distinct videos encountered in ``meta_100K_frame.txt``. This yields at most ``--max-clips`` rows
in the manifest (fewer if some yt-dlp downloads fail).

Each clip is fetched with yt-dlp, restricted to a ``--download-sections`` window so only the
needed seconds are transferred (not the whole video), and capped at 720p to bound bandwidth:

    yt-dlp -f "bv*[height<=720]+ba/b[height<=720]" --download-sections "*START-END" \\
        -o out/clips/<video_id>_<start_s>.mp4 https://www.youtube.com/watch?v=<video_id>

Videos that are private, deleted, region-blocked, or otherwise fail to download are logged and
skipped; the run continues. Existing output files are treated as already-downloaded and skipped.

Note on TLS: ``fouheylab.eecs.umich.edu``'s certificate was found expired at research time
(``openssl s_client ... -> Verify return code: 10 (certificate has expired)``, 2026-09-12), a
server-side issue on the University of Michigan's end, not specific to any one client. By
default this downloader still performs normal, certificate-verified fetches and *fails loudly*
if that host's cert is still expired, rather than silently disabling verification (TLS
verification bypass is a MITM risk and must never be a silent default). An operator who has
independently confirmed the cert issue can pass ``--insecure-tls`` to explicitly opt in to an
unverified retry, restricted to the exact, hardcoded host in ``_KNOWN_EXPIRED_CERT_HOSTS`` --
every such retry is logged as a warning. No other host is ever affected by this flag.
"""

from __future__ import annotations

import argparse
import csv
import logging
import shutil
import ssl
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen, urlretrieve

import yaml

logger = logging.getLogger(__name__)

ANNOTATIONS_URL = "https://fouheylab.eecs.umich.edu/~dandans/projects/100DOH/downloads/file.zip"
FRAME_EXTRACTION_FPS = 0.5  # per README_100K.md: `ffmpeg -vf fps=0.5`
MANIFEST_COLUMNS = ["video_id", "start_s", "end_s", "path", "status"]

# Hosts with a verified, documented expired-certificate issue (see module docstring). Only a
# host in this set, AND an explicit --insecure-tls opt-in, can ever trigger an unverified retry.
_KNOWN_EXPIRED_CERT_HOSTS = frozenset({"fouheylab.eecs.umich.edu"})


def _fetch_url(url: str, dest: Path, *, allow_insecure_tls: bool) -> None:
    """Download ``url`` to ``dest``.

    Always tries a normal, certificate-verified fetch first. If that fails with a TLS
    certificate error on a host in ``_KNOWN_EXPIRED_CERT_HOSTS``, the behavior depends on
    ``allow_insecure_tls``: when False (the default), the error is re-raised with guidance
    instead of being silently worked around; when True (the operator explicitly passed
    ``--insecure-tls``), it retries once without verification and logs a warning. Any other
    failure, or any other host, always propagates unchanged -- this is never a blanket bypass.
    """
    try:
        urlretrieve(url, dest)
        return
    except URLError as exc:
        host = urlparse(url).hostname or ""
        if not isinstance(exc.reason, ssl.SSLCertVerificationError) or host not in (
            _KNOWN_EXPIRED_CERT_HOSTS
        ):
            raise
        if not allow_insecure_tls:
            logger.error(
                "TLS certificate verification failed for %s (known expired-cert host, see "
                "module docstring). Re-run with --insecure-tls to explicitly opt in to an "
                "unverified retry for this exact host, or fetch %s manually.",
                host,
                url,
            )
            raise
        logger.warning(
            "--insecure-tls set: retrying %s without certificate verification "
            "(known-expired-cert host %s)",
            url,
            host,
        )
        unverified_context = ssl.create_default_context()
        unverified_context.check_hostname = False
        unverified_context.verify_mode = ssl.CERT_NONE
        with urlopen(url, context=unverified_context) as response, dest.open("wb") as f:
            shutil.copyfileobj(response, f)


# Byte-string prefixes (checked case-insensitively, after stripping leading whitespace) that
# indicate a downloaded payload is actually an HTML page rather than the expected zip archive.
_HTML_MARKERS = (b"<!doctype", b"<html")


def _looks_like_html(head: bytes) -> bool:
    """Return True if ``head`` (a payload's leading bytes) looks like an HTML page."""
    stripped = head.lstrip()[:16].lower()
    return any(stripped.startswith(marker) for marker in _HTML_MARKERS)


def _validate_zip_payload(path: Path) -> None:
    """Guard against an HTML error page (or other non-zip payload) masquerading as ``file.zip``.

    Deletes ``path`` and raises ``ValueError`` if it is not a plausible zip archive.
    """
    with path.open("rb") as f:
        head = f.read(1024)
    if _looks_like_html(head):
        path.unlink(missing_ok=True)
        raise ValueError(f"{path} looks like an HTML page, not the expected zip archive")
    if not zipfile.is_zipfile(path):
        path.unlink(missing_ok=True)
        raise ValueError(f"{path} is not a valid zip archive")


@dataclass(frozen=True)
class FrameRecord:
    """One row of ``meta_100K_frame.txt``."""

    video_id: str
    frame_index: int

    @property
    def frame_time_s(self) -> float:
        """Approximate timestamp of this frame in its source video, in seconds."""
        return (self.frame_index - 1) / FRAME_EXTRACTION_FPS


def clip_window(frame_time_s: float, clip_seconds: float) -> tuple[float, float]:
    """Compute the ``[start, end]`` window (seconds) of a clip centered on a frame time.

    Args:
        frame_time_s: Timestamp of the annotated frame, in seconds.
        clip_seconds: Total clip duration, in seconds.

    Returns:
        ``(start_s, end_s)`` with ``start_s`` clamped to be non-negative.
    """
    half = clip_seconds / 2
    return max(0.0, frame_time_s - half), frame_time_s + half


def load_sources(sources_yaml: Path) -> dict[str, object]:
    """Load the ``100doh`` entry from ``sources.yaml``."""
    with sources_yaml.open() as f:
        manifest = yaml.safe_load(f)
    for entry in manifest["datasets"]:
        if entry["name"] == "100doh":
            return entry
    raise KeyError(f"no '100doh' entry found in {sources_yaml}")


def download_annotations(url: str, raw_dir: Path, *, allow_insecure_tls: bool = False) -> Path:
    """Download and extract ``file.zip`` into ``raw_dir`` (idempotent)."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "file.zip"
    frame_meta = raw_dir / "file" / "meta_100K_frame.txt"
    if frame_meta.exists():
        logger.info("annotations already extracted at %s", frame_meta)
        return frame_meta
    if not zip_path.exists():
        logger.info("downloading 100DOH annotations from %s", url)
        _fetch_url(url, zip_path, allow_insecure_tls=allow_insecure_tls)
        _validate_zip_payload(zip_path)
    logger.info("extracting %s", zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(raw_dir)
    if not frame_meta.exists():
        raise FileNotFoundError(f"expected {frame_meta} after extracting {zip_path}")
    return frame_meta


def iter_frames_grouped_by_video(frame_meta_path: Path, max_videos: int) -> list[FrameRecord]:
    """Parse ``meta_100K_frame.txt`` and return one representative frame per video.

    Returns the earliest-indexed frame for each of the first ``max_videos`` distinct video ids
    encountered (insertion order), so callers get at most ``max_videos`` records back.
    """
    best_by_video: dict[str, FrameRecord] = {}
    with frame_meta_path.open() as f:
        header = f.readline()  # skip column header
        if not header:
            raise ValueError(f"empty metadata file: {frame_meta_path}")
        for line in f:
            fields = line.split()
            if len(fields) < 3:
                continue
            video_id, frame_index_s = fields[1], fields[2]
            try:
                frame_index = int(frame_index_s)
            except ValueError:
                continue
            existing = best_by_video.get(video_id)
            if existing is None:
                if len(best_by_video) >= max_videos:
                    continue
                best_by_video[video_id] = FrameRecord(video_id, frame_index)
            elif frame_index < existing.frame_index:
                best_by_video[video_id] = FrameRecord(video_id, frame_index)
    return list(best_by_video.values())


def load_existing_manifest_keys(manifest_path: Path) -> set[tuple[str, str]]:
    """Return ``(video_id, path)`` pairs already recorded, to make appends idempotent-ish."""
    if not manifest_path.exists():
        return set()
    keys: set[tuple[str, str]] = set()
    with manifest_path.open(newline="") as f:
        for row in csv.DictReader(f):
            keys.add((row["video_id"], row["path"]))
    return keys


def append_manifest_row(
    manifest_path: Path,
    video_id: str,
    start_s: float,
    end_s: float,
    path: str,
    status: str,
) -> None:
    """Append one row to ``manifest.csv``, writing the header if the file is new."""
    is_new = not manifest_path.exists()
    with manifest_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow(
            {
                "video_id": video_id,
                "start_s": f"{start_s:.3f}",
                "end_s": f"{end_s:.3f}",
                "path": path,
                "status": status,
            }
        )


def download_clip(video_id: str, start_s: float, end_s: float, dest: Path) -> bool:
    """Fetch one clip window with yt-dlp. Returns True on success, False on failure."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    section = f"*{start_s:.3f}-{end_s:.3f}"
    cmd = [
        "yt-dlp",
        "-f",
        "bv*[height<=720]+ba/b[height<=720]",
        "--download-sections",
        section,
        "--force-keyframes-at-cuts",
        "--no-playlist",
        "-o",
        str(dest),
        f"https://www.youtube.com/watch?v={video_id}",
    ]
    logger.debug("running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not dest.exists():
        logger.warning(
            "yt-dlp failed for video_id=%s (exit=%d): %s",
            video_id,
            result.returncode,
            result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "no stderr",
        )
        return False
    return True


def run(
    out_dir: Path,
    max_clips: int,
    clip_seconds: float,
    sources_yaml: Path,
    allow_insecure_tls: bool = False,
) -> None:
    """Download annotations, then up to ``max_clips`` 10 s clips into ``out_dir``."""
    entry = load_sources(sources_yaml)
    raw_dir = out_dir / "raw"
    clips_dir = out_dir / "clips"
    manifest_path = out_dir / "manifest.csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_meta_path = download_annotations(
        str(entry["url"]), raw_dir, allow_insecure_tls=allow_insecure_tls
    )
    frames = iter_frames_grouped_by_video(frame_meta_path, max_videos=max_clips)
    logger.info("selected %d videos (max_clips=%d)", len(frames), max_clips)

    existing = load_existing_manifest_keys(manifest_path)
    n_ok, n_fail, n_skip = 0, 0, 0
    for frame in frames:
        start_s, end_s = clip_window(frame.frame_time_s, clip_seconds)
        clip_name = f"{frame.video_id}_{start_s:.3f}.mp4"
        clip_path = clips_dir / clip_name
        rel_path = str(Path("clips") / clip_name)

        if clip_path.exists() or (frame.video_id, rel_path) in existing:
            logger.info("skipping existing clip %s", clip_path)
            n_skip += 1
            continue

        ok = download_clip(frame.video_id, start_s, end_s, clip_path)
        status = "ok" if ok else "failed"
        append_manifest_row(manifest_path, frame.video_id, start_s, end_s, rel_path, status)
        if ok:
            n_ok += 1
        else:
            n_fail += 1

    logger.info(
        "100DOH download complete: %d ok, %d failed, %d skipped (manifest=%s)",
        n_ok,
        n_fail,
        n_skip,
        manifest_path,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument(
        "--max-clips", type=int, default=200, help="number of distinct videos to fetch clips for"
    )
    parser.add_argument(
        "--clip-seconds", type=float, default=10.0, help="total clip duration in seconds"
    )
    parser.add_argument(
        "--sources-yaml",
        type=Path,
        default=Path(__file__).parent / "sources.yaml",
        help="path to sources.yaml (defaults to the file next to this script)",
    )
    parser.add_argument(
        "--insecure-tls",
        action="store_true",
        help="explicitly allow an unverified-TLS retry, restricted to the exact host "
        "documented in this script's module docstring as having an expired certificate "
        "(never a blanket bypass; default off)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="enable DEBUG logging")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run(
        out_dir=args.out,
        max_clips=args.max_clips,
        clip_seconds=args.clip_seconds,
        sources_yaml=args.sources_yaml,
        allow_insecure_tls=args.insecure_tls,
    )


if __name__ == "__main__":
    main()
