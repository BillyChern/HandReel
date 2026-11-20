"""Shape-cache hardening: the env probe keeps the imported distribution, worker_support is part
of the code identity, and a failing quarantine rename is a logged miss."""

from __future__ import annotations

import dataclasses
import errno
import json
import logging
import os
import sys
from pathlib import Path

import pytest

from fastdo.engine import shape_cache
from fastdo.engine.shape_cache import ENTRY_FILES, SHAPE_CODE_FILES, lookup, store
from fastdo.shape import ShapeConfig
from fastdo.shape.sam3d_backend import sam3d_env_identity

_DIST = "fastdo-shadow-probe"


def _dist_info(root: Path, version: str) -> None:
    info = root / f"fastdo_shadow_probe-{version}.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {_DIST}\nVersion: {version}\n")
    (info / "RECORD").write_text(f"fastdo_shadow_probe/__init__.py,sha256=x{version},1\n")


def test_env_probe_records_the_first_distribution_on_sys_path(tmp_path: Path) -> None:
    pkg = tmp_path / "pylib" / "sam3d_objects"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    _dist_info(tmp_path / "first", "1.0")                 # earlier on sys.path: the copy `import` would resolve
    _dist_info(tmp_path / "shadowed", "2.0")
    cfg = ShapeConfig(backend="sam3d", deterministic=False, sam3d_python=sys.executable,
                      sam3d_pythonpath=(str(pkg.parent), str(tmp_path / "first"), str(tmp_path / "shadowed")))
    probe = json.loads(sam3d_env_identity(cfg))["probe"]
    assert probe["distributions"][_DIST][0] == "1.0"
    swapped = dataclasses.replace(cfg, sam3d_pythonpath=(str(pkg.parent), str(tmp_path / "shadowed"),
                                                         str(tmp_path / "first")))
    assert json.loads(sam3d_env_identity(swapped))["probe"]["distributions"][_DIST][0] == "2.0"


def test_worker_support_is_part_of_the_code_identity() -> None:                       
    assert "engine/worker_support.py" in SHAPE_CODE_FILES          # save_npz and read_decode shape the cached bytes
    assert all((shape_cache.SHAPE_CODE_ROOT / rel).is_file() for rel in SHAPE_CODE_FILES)


def test_failing_quarantine_rename_is_a_logged_miss(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                    caplog: pytest.LogCaptureFixture) -> None:
    cache, key = tmp_path / "cache", "cd" + "1" * 62
    entry = store(cache, key, {"schema": "x"}, {name: name.encode() for name in ENTRY_FILES})
    (entry / ENTRY_FILES[0]).write_bytes(b"corrupted")

    def refuse(src: object, dst: object) -> None:
        raise OSError(errno.EACCES, "simulated permission error", str(src))

    monkeypatch.setattr(os, "rename", refuse)
    with caplog.at_level(logging.WARNING, logger=shape_cache.__name__):
        assert lookup(cache, key) is None
    assert entry.is_dir()                                   # nothing deleted in place; the next lookup retries
    assert any("could not quarantine" in record.getMessage() for record in caplog.records)
