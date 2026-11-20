"""Unit tests for run/data/download_dexycb.py's payload validation and size-check helpers.

Pure/offline only: no network access, no real Google Drive downloads.
"""

from __future__ import annotations

import importlib.util
import sys
import tarfile
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOWNLOAD_DEXYCB_PY = REPO_ROOT / "run" / "data" / "download_dexycb.py"


def _load_download_dexycb_module() -> types.ModuleType:
    """Import run/data/download_dexycb.py by file path (run/data is a script dir, not a package)."""
    spec = importlib.util.spec_from_file_location("download_dexycb", DOWNLOAD_DEXYCB_PY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclasses inside the module can resolve their own
    # annotations via sys.modules[cls.__module__].
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def dexycb_module() -> types.ModuleType:
    return _load_download_dexycb_module()


class TestIsHtmlInterstitial:
    def test_rejects_doctype_html(self, dexycb_module: types.ModuleType) -> None:
        head = b"<!DOCTYPE html><html><body>Google Drive can't scan this file</body></html>"
        assert dexycb_module.is_html_interstitial(head) is True

    def test_rejects_bare_html_tag(self, dexycb_module: types.ModuleType) -> None:
        assert dexycb_module.is_html_interstitial(b"<html><head></head></html>") is True

    def test_tolerates_leading_whitespace(self, dexycb_module: types.ModuleType) -> None:
        assert dexycb_module.is_html_interstitial(b"\n\n  <!doctype html>") is True

    def test_accepts_gzip_magic_bytes(self, dexycb_module: types.ModuleType) -> None:
        gzip_header = b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\x03"
        assert dexycb_module.is_html_interstitial(gzip_header) is False

    def test_accepts_empty_bytes(self, dexycb_module: types.ModuleType) -> None:
        assert dexycb_module.is_html_interstitial(b"") is False


class TestValidateArchivePayload:
    def test_rejects_html_bytes(self, dexycb_module: types.ModuleType, tmp_path: Path) -> None:
        bogus = tmp_path / "bogus.tar.gz"
        bogus.write_bytes(b"<!DOCTYPE html><html><body>quota exceeded</body></html>")
        with pytest.raises(dexycb_module.DownloadValidationError):
            dexycb_module.validate_archive_payload(bogus)

    def test_rejects_non_tar_binary(self, dexycb_module: types.ModuleType, tmp_path: Path) -> None:
        junk = tmp_path / "junk.tar.gz"
        junk.write_bytes(b"\x00\x01\x02\x03not a tar file at all, just random bytes" * 5)
        with pytest.raises(dexycb_module.DownloadValidationError):
            dexycb_module.validate_archive_payload(junk)

    def test_accepts_real_tar_gz(self, dexycb_module: types.ModuleType, tmp_path: Path) -> None:
        member = tmp_path / "hello.txt"
        member.write_text("hello dexycb")
        archive = tmp_path / "real.tar.gz"
        with tarfile.open(archive, "w:gz") as tf:
            tf.add(member, arcname="hello.txt")
        dexycb_module.validate_archive_payload(archive)  # must not raise

    def test_accepts_uncompressed_tar(
        self, dexycb_module: types.ModuleType, tmp_path: Path
    ) -> None:
        member = tmp_path / "hello.txt"
        member.write_text("hello dexycb")
        archive = tmp_path / "real.tar"
        with tarfile.open(archive, "w") as tf:
            tf.add(member, arcname="hello.txt")
        dexycb_module.validate_archive_payload(archive)  # must not raise


class TestSizeWithinTolerance:
    def test_exact_match(self, dexycb_module: types.ModuleType) -> None:
        assert dexycb_module.size_within_tolerance(1_000_000_000, 1.0) is True

    def test_within_tolerance_above(self, dexycb_module: types.ModuleType) -> None:
        # expected 1.0 GB, actual 1.1 GB (10% over) -> within default 25%
        assert dexycb_module.size_within_tolerance(1_100_000_000, 1.0) is True

    def test_within_tolerance_below(self, dexycb_module: types.ModuleType) -> None:
        # expected 1.0 GB, actual 0.8 GB (20% under) -> within default 25%
        assert dexycb_module.size_within_tolerance(800_000_000, 1.0) is True

    def test_outside_tolerance_too_small(self, dexycb_module: types.ModuleType) -> None:
        # expected 1.0 GB, actual 0.5 GB (50% under) -> outside default 25%
        assert dexycb_module.size_within_tolerance(500_000_000, 1.0) is False

    def test_outside_tolerance_too_large(self, dexycb_module: types.ModuleType) -> None:
        # expected 1.0 GB, actual 2.0 GB (100% over) -> outside default 25%
        assert dexycb_module.size_within_tolerance(2_000_000_000, 1.0) is False

    def test_none_expected_uses_min_bytes_floor(self, dexycb_module: types.ModuleType) -> None:
        assert dexycb_module.size_within_tolerance(2_000_000, None) is True
        assert dexycb_module.size_within_tolerance(500, None) is False

    def test_non_positive_expected_uses_min_bytes_floor(
        self, dexycb_module: types.ModuleType
    ) -> None:
        assert dexycb_module.size_within_tolerance(2_000_000, 0.0) is True
        assert dexycb_module.size_within_tolerance(500, -1.0) is False

    def test_custom_tolerance(self, dexycb_module: types.ModuleType) -> None:
        # 5% over, with a strict 1% tolerance -> outside
        assert dexycb_module.size_within_tolerance(1_050_000_000, 1.0, tolerance=0.01) is False
        # 5% over, with a loose 10% tolerance -> within
        assert dexycb_module.size_within_tolerance(1_050_000_000, 1.0, tolerance=0.10) is True
