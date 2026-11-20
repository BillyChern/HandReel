"""Download DexYCB's per-subject archives: by default those of the official toolkit's s0 test split (subjects 03-10).

DexYCB (https://dex-ycb.github.io/) hosts one CC-BY-NC-4.0 archive per subject (10 subjects) on
Google Drive, plus small shared archives (calibration, YCB object models, optional BOP-format
annotations). All URLs are read from ``run/data/sources.yaml`` -> ``datasets: [name: dexycb,
files: [...]]`` -- none are hardcoded here.

Split selection (``--subset eval``): the official dex-ycb-toolkit defines split ``s0`` /
``test`` as subject indices ``[2..9]`` (0-based, i.e. subjects 03 through 10) crossed with
sequence indices ``i % 5 == 4`` (20 of each subject's 100 sequences), for 8 * 20 = 160 total
sequences. The paper has no evaluation on those 160 sequences; its held-out DexYCB views all come from subject
08, one of these eight subjects. Google Drive only exposes per-subject archives, not
per-sequence ones, so ``--subset eval`` downloads the 8 full subject archives that *contain*
those 160 sequences (each ~12 GB, ~97.4 GB total with calibration + models, under the 100 GB
budget) plus every one of that subject's other 80 sequences, which is coarser than strictly
necessary; extra sequences can be pruned after extraction if disk space is tight. Subjects 01-02
(not part of s0 test) and the optional ``bop.tar.gz`` (BOP-format duplicate annotations, not
needed by ``DexYCBReader``) are skipped in this mode. ``--subset full`` additionally downloads
subjects 01-02 and bop.tar.gz, but still skips the redundant 119 GB single-archive alternative
(same content as calibration + all ten subject archives + models + bop, just bundled).

Each Google Drive file is streamed from
``https://drive.usercontent.google.com/download?id=<id>&export=download&confirm=t``, which was
verified (2026-09-12) to return raw archive bytes directly, without the interstitial
virus-scan-warning HTML page Drive shows for browser sessions on files over ~100 MB.

Every downloaded payload is still validated before it is trusted, though: after each file
finishes downloading (and before its ``.part`` staging file is renamed to the final name), this
module checks that the bytes are not an HTML page (Drive's virus-scan interstitial, or any other
error page a proxy might substitute) and that they form a real tar archive
(``tarfile.is_tarfile``); an orphaned ``.part`` file is deleted on any failure. Independently of
that, ``run()`` always finishes with an unconditional pass over every selected file checking
existence and that its size is within 25% of the ``size_gb`` recorded in ``sources.yaml`` (or,
if a file has no recorded size, that it is larger than 1 MB); any failures are logged and make
the process exit non-zero. ``--verify-only`` runs just that final check, without downloading
anything, so an already-downloaded ``--out`` directory can be re-validated on its own (e.g.
after a background run started from an older version of this script).
"""

from __future__ import annotations

import argparse
import logging
import tarfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import yaml

logger = logging.getLogger(__name__)

GDRIVE_DOWNLOAD_URL = (
    "https://drive.usercontent.google.com/download?id={id}&export=download&confirm=t"
)
SUBSET_CHOICES = ("eval", "full")

# Tolerance and floor used by size_within_tolerance() / verify_files().
_SIZE_TOLERANCE = 0.25
_MIN_EXPECTED_BYTES = 1_000_000  # 1 MB, used when a file has no recorded size_gb
_BYTES_PER_GB = 1_000_000_000  # decimal GB, matching sources.yaml's size_gb convention

# Byte-string prefixes (checked case-insensitively, after stripping leading whitespace) that
# indicate a downloaded payload is actually an HTML page rather than the expected archive.
_HTML_MARKERS = (b"<!doctype", b"<html")


class DownloadValidationError(Exception):
    """Raised when a downloaded payload fails validation (e.g. an HTML interstitial page)."""


@dataclass(frozen=True)
class DexYCBFile:
    """One downloadable component of the DexYCB dataset, as listed in sources.yaml."""

    name: str
    drive_id: str
    size_gb: float
    s0_test: bool


def load_dexycb_files(sources_yaml: Path) -> list[DexYCBFile]:
    """Load the ``dexycb`` entry's ``files`` list from ``sources.yaml``."""
    with sources_yaml.open() as f:
        manifest = yaml.safe_load(f)
    for entry in manifest["datasets"]:
        if entry["name"] == "dexycb":
            return [
                DexYCBFile(
                    name=f["name"],
                    drive_id=f["drive_id"],
                    size_gb=float(f["size_gb"]),
                    s0_test=bool(f["s0_test"]),
                )
                for f in entry["files"]
                if "drive_id" in f
            ]
    raise KeyError(f"no 'dexycb' entry found in {sources_yaml}")


def select_files(files: list[DexYCBFile], subset: str) -> list[DexYCBFile]:
    """Filter the full file list down to what ``--subset`` needs.

    ``eval``: calibration + models + the 8 subject archives needed for the s0 test split.
    ``full``: every file except the redundant single-archive bundle (``name == "full_archive"``).
    """
    if subset == "eval":
        return [f for f in files if f.name in ("calibration", "models") or f.s0_test]
    if subset == "full":
        return [f for f in files if f.name != "full_archive"]
    raise ValueError(f"unknown subset {subset!r}, expected one of {SUBSET_CHOICES}")


def is_html_interstitial(head: bytes) -> bool:
    """Return True if ``head`` (a payload's leading bytes) looks like an HTML page.

    Google Drive serves an HTML "can't scan this file for viruses" interstitial instead of the
    requested bytes when its large-file confirm-token bypass fails; this catches that (and any
    other HTML error page a proxy might substitute) before it gets treated as a valid archive.
    Real ``.tar.gz`` bytes start with the gzip magic number and will never match.
    """
    stripped = head.lstrip()[:16].lower()
    return any(stripped.startswith(marker) for marker in _HTML_MARKERS)


def validate_archive_payload(path: Path) -> None:
    """Raise ``DownloadValidationError`` if ``path`` is not a plausible downloaded archive.

    Checks, in order: (1) the first bytes are not an HTML interstitial; (2) the file is a tar
    archive that ``tarfile.is_tarfile`` accepts (every DexYCB file distributed here is
    ``.tar.gz``). Intended to run on a ``.part`` staging file before it is renamed to its final
    name, so a bad download never masquerades as a complete one.
    """
    with path.open("rb") as f:
        head = f.read(1024)
    if is_html_interstitial(head):
        raise DownloadValidationError(
            f"{path} looks like an HTML page, not an archive (Google Drive interstitial?)"
        )
    if not tarfile.is_tarfile(path):
        raise DownloadValidationError(f"{path} is not a valid tar archive")


def size_within_tolerance(
    actual_bytes: int,
    expected_gb: float | None,
    *,
    tolerance: float = _SIZE_TOLERANCE,
    min_bytes: int = _MIN_EXPECTED_BYTES,
) -> bool:
    """Return True if ``actual_bytes`` is a plausible size for a file expected to be ``expected_gb``.

    When ``expected_gb`` is known, ``actual_bytes`` must be within ``tolerance`` (relative,
    default 25%) of it. When ``expected_gb`` is ``None`` or non-positive (no usable size
    recorded in ``sources.yaml``), falls back to requiring ``actual_bytes > min_bytes`` (default
    1 MB) as a sanity floor.
    """
    if expected_gb is None or expected_gb <= 0:
        return actual_bytes > min_bytes
    expected_bytes = expected_gb * _BYTES_PER_GB
    return abs(actual_bytes - expected_bytes) <= tolerance * expected_bytes


def download_gdrive_file(drive_id: str, dest: Path) -> None:
    """Stream a Google Drive file to ``dest`` (idempotent: skips if ``dest`` already exists).

    Validates the downloaded bytes with ``validate_archive_payload`` before renaming the
    ``.part`` staging file to ``dest``. On any failure -- a network error or failed validation --
    the orphaned ``.part`` file is removed and the exception is re-raised.
    """
    if dest.exists():
        logger.info("already downloaded: %s", dest)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = GDRIVE_DOWNLOAD_URL.format(id=drive_id)
    logger.info("downloading %s -> %s", url, dest)
    tmp_dest = dest.with_suffix(dest.suffix + ".part")
    try:
        urlretrieve(url, tmp_dest)
        validate_archive_payload(tmp_dest)
    except DownloadValidationError:
        logger.error("downloaded payload for %s failed validation", dest.name)
        tmp_dest.unlink(missing_ok=True)
        raise
    except OSError:
        tmp_dest.unlink(missing_ok=True)
        raise
    tmp_dest.rename(dest)


def extract_archive(archive_path: Path, extract_dir: Path) -> None:
    """Extract a ``.tar.gz`` archive into ``extract_dir`` (idempotent best-effort)."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    logger.info("extracting %s -> %s", archive_path, extract_dir)
    with tarfile.open(archive_path) as tf:
        tf.extractall(extract_dir)


def verify_file_counts(root: Path, files: list[DexYCBFile]) -> dict[str, int]:
    """Count top-level extracted entries per downloaded component, for a sanity report."""
    counts: dict[str, int] = {}
    for f in files:
        target = root / f.name
        if target.exists():
            counts[f.name] = sum(1 for _ in target.rglob("*"))
        else:
            counts[f.name] = 0
    return counts


def verify_files(out_dir: Path, files: list[DexYCBFile]) -> list[str]:
    """Check that every file in ``files`` was downloaded into ``out_dir`` and looks complete.

    Returns a list of human-readable failure descriptions (empty means everything checked out):
    a file is a failure if its archive is missing, or if its size falls outside
    ``size_within_tolerance`` of the ``size_gb`` recorded for it in ``sources.yaml``.
    """
    failures: list[str] = []
    archives_dir = out_dir / "archives"
    for f in files:
        archive_path = archives_dir / f"{f.name}.tar.gz"
        if not archive_path.exists():
            failures.append(f"{f.name}: missing ({archive_path})")
            continue
        actual_bytes = archive_path.stat().st_size
        if not size_within_tolerance(actual_bytes, f.size_gb):
            expected_bytes = f.size_gb * _BYTES_PER_GB
            failures.append(
                f"{f.name}: {actual_bytes} bytes is outside {_SIZE_TOLERANCE:.0%} tolerance of "
                f"the expected ~{expected_bytes:,.0f} bytes ({f.size_gb} GB)"
            )
    return failures


def run(
    out_dir: Path,
    subset: str,
    sources_yaml: Path,
    extract: bool,
    verify_only: bool = False,
) -> bool:
    """Download (and optionally extract) the DexYCB files needed for ``subset``.

    When ``verify_only`` is True, no downloads are attempted; this only runs the final
    ``verify_files`` pass over whatever already exists in ``out_dir``.

    Always finishes with an unconditional ``verify_files`` pass over every selected file.
    Returns True if verification passed (no failures), False otherwise -- callers that want a
    process exit code should act on this return value.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    archives_dir = out_dir / "archives"
    extract_dir = out_dir / "extracted"

    all_files = load_dexycb_files(sources_yaml)
    selected = select_files(all_files, subset)
    total_gb = sum(f.size_gb for f in selected)
    logger.info(
        "subset=%s selects %d files, ~%.1f GB total: %s",
        subset,
        len(selected),
        total_gb,
        ", ".join(f.name for f in selected),
    )

    if verify_only:
        logger.info("--verify-only: skipping downloads, verifying existing files in %s", out_dir)
    else:
        for f in selected:
            archive_path = archives_dir / f"{f.name}.tar.gz"
            try:
                download_gdrive_file(f.drive_id, archive_path)
            except (OSError, DownloadValidationError):
                logger.exception("failed to download %s (drive_id=%s)", f.name, f.drive_id)
                continue
            if extract:
                try:
                    extract_archive(archive_path, extract_dir)
                except tarfile.TarError:
                    logger.exception("failed to extract %s", archive_path)

        if extract:
            counts = verify_file_counts(extract_dir, selected)
            logger.info("extracted entry counts per component: %s", counts)

    failures = verify_files(out_dir, selected)
    if failures:
        logger.error(
            "verification FAILED for %d/%d file(s):\n%s",
            len(failures),
            len(selected),
            "\n".join(f"  - {msg}" for msg in failures),
        )
    else:
        logger.info(
            "verification passed: all %d file(s) present and correctly sized", len(selected)
        )

    logger.info(
        "DexYCB run complete: subset=%s, out=%s, verify_only=%s", subset, out_dir, verify_only
    )
    return not failures


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument(
        "--subset",
        choices=SUBSET_CHOICES,
        default="eval",
        help="'eval' = the archives of the s0 test split's subjects 03-10 (holds subject 08); 'full' = everything "
        "except the redundant single-archive bundle",
    )
    parser.add_argument(
        "--sources-yaml",
        type=Path,
        default=Path(__file__).parent / "sources.yaml",
        help="path to sources.yaml (defaults to the file next to this script)",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="extract downloaded .tar.gz archives after downloading",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="skip downloading; just verify the files already present in --out (existence + "
        "size-tolerance check) and exit non-zero if any are missing or look incomplete",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="enable DEBUG logging")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ok = run(
        out_dir=args.out,
        subset=args.subset,
        sources_yaml=args.sources_yaml,
        extract=args.extract,
        verify_only=args.verify_only,
    )
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
