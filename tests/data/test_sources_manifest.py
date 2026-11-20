"""Tests for the dataset source manifest and the 100DOH clip-window helper."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCES_YAML = REPO_ROOT / "run" / "data" / "sources.yaml"
DOWNLOAD_100DOH_PY = REPO_ROOT / "run" / "data" / "download_100doh.py"

REQUIRED_KEYS = ("name", "url", "requires_registration", "size_gb", "notes")
EXPECTED_DATASETS = {"100doh", "dexycb", "ho3d", "hoi4d", "arctic", "oakink2", "ego4d"}


def _load_manifest() -> list[dict[str, object]]:
    with SOURCES_YAML.open() as f:
        manifest = yaml.safe_load(f)
    return manifest["datasets"]


def _load_download_100doh_module() -> types.ModuleType:
    """Import run/data/download_100doh.py by file path (run/data is a script dir, not a package)."""
    spec = importlib.util.spec_from_file_location("download_100doh", DOWNLOAD_100DOH_PY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclasses inside the module can resolve their own
    # annotations via sys.modules[cls.__module__].
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_sources_yaml_exists() -> None:
    assert SOURCES_YAML.is_file()


def test_every_entry_has_required_keys() -> None:
    entries = _load_manifest()
    assert entries, "sources.yaml has no datasets"
    for entry in entries:
        missing = [key for key in REQUIRED_KEYS if key not in entry]
        assert not missing, f"{entry.get('name', '<unnamed>')} missing keys: {missing}"


def test_requires_registration_is_boolean() -> None:
    for entry in _load_manifest():
        assert isinstance(entry["requires_registration"], bool), (
            f"{entry['name']}.requires_registration is "
            f"{type(entry['requires_registration']).__name__}, expected bool"
        )


def test_covers_expected_datasets() -> None:
    names = {entry["name"] for entry in _load_manifest()}
    assert EXPECTED_DATASETS <= names, f"missing datasets: {EXPECTED_DATASETS - names}"


def test_names_are_unique() -> None:
    names = [entry["name"] for entry in _load_manifest()]
    assert len(names) == len(set(names))


def test_gated_datasets_flagged_and_ungated_have_notes_on_status() -> None:
    by_name = {entry["name"]: entry for entry in _load_manifest()}
    for gated_name in ("hoi4d", "arctic", "oakink2", "ego4d"):
        assert by_name[gated_name]["requires_registration"] is True
    for open_name in ("100doh", "dexycb", "ho3d"):
        assert by_name[open_name]["requires_registration"] is False
        assert by_name[open_name]["notes"], f"{open_name} has empty notes"


@pytest.fixture(scope="module")
def clip_window():
    module = _load_download_100doh_module()
    return module.clip_window


class TestClipWindow:
    """Unit tests for download_100doh.clip_window."""

    def test_centered_window(self, clip_window) -> None:
        start, end = clip_window(20.0, 10.0)
        assert start == pytest.approx(15.0)
        assert end == pytest.approx(25.0)

    def test_clamped_at_zero(self, clip_window) -> None:
        start, end = clip_window(2.0, 10.0)
        assert start == pytest.approx(0.0)
        assert end == pytest.approx(7.0)

    def test_frame_time_exactly_zero(self, clip_window) -> None:
        start, end = clip_window(0.0, 10.0)
        assert start == pytest.approx(0.0)
        assert end == pytest.approx(5.0)

    def test_window_width_matches_clip_seconds_away_from_clamp(self, clip_window) -> None:
        start, end = clip_window(100.0, 10.0)
        assert end - start == pytest.approx(10.0)
