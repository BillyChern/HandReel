"""Test helpers for the 100DOH selection: a synthetic bundle with train and test entries, and a file-access spy."""

from __future__ import annotations

import builtins
import io
import json
import os
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

__all__ = ["FakeDohBundle", "FileAccessSpy", "install_file_access_spy", "write_fake_doh_bundle"]

# (category, video_id, start, end, official_split, clip_path given, decoded_frames recorded)
_ENTRIES: tuple[tuple[str, str, int, int, str | None, bool, bool], ...] = (
    ("boardgame", "3Bsm34JRL7w", 64, 74, "train", True, True),
    ("cooking", "TESTaa-b_c", 5, 15, "test", True, False),
    ("drink", "x_Yz-9", 1, 11, "train", False, False),     # path from the naming rule; frames counted from the file
    ("misc", "TESTnosplit", 0, 10, None, True, False),
)


@dataclass(frozen=True)
class FakeDohBundle:
    root: Path
    train_ids: list[str]        # expected benchmark clip ids, in manifest order
    train_paths: list[Path]
    held_out_paths: list[Path]  # test-split and split-less clip files, which must never be touched
    held_out_markers: tuple[str, ...] = ("TESTaa-b_c", "TESTnosplit")


def write_fake_doh_bundle(root: Path) -> FakeDohBundle:
    """`<root>/manifest.json` (the bundle's field names) plus a junk-byte clip file for every entry under `clips/`."""
    (root / "clips").mkdir(parents=True)
    entries: list[dict[str, Any]] = []
    train_ids: list[str] = []
    train_paths: list[Path] = []
    held_out: list[Path] = []
    for category, video_id, start, end, split, give_path, give_frames in _ENTRIES:
        name = f"{category}_{video_id}_{start}_{end}"
        path = root / "clips" / f"{name}.mp4"
        path.write_bytes(b"not a real video")
        entry: dict[str, Any] = {"category": category, "video_id": video_id, "start_seconds": start,
                                 "end_seconds": end, "fps": 29.97}
        if split is not None:
            entry["official_split"] = split
        if give_path:
            entry["clip_path"] = f"clips/{name}.mp4"
        if give_frames:
            entry["decoded_frames"] = 300
        entries.append(entry)
        if split == "train":
            train_ids.append(f"bench_100doh_{name}")
            train_paths.append(path)
        else:
            held_out.append(path)
    (root / "manifest.json").write_text(json.dumps(entries))
    return FakeDohBundle(root, train_ids, train_paths, held_out)


@dataclass
class FileAccessSpy:
    """Every path handed to the patched open/stat/listing/subprocess entry points, as `(function, str(argument))`."""

    calls: list[tuple[str, str]] = field(default_factory=list)

    def wrap(self, name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
        def spy(*args: Any, **kwargs: Any) -> Any:
            target = args[0] if args else kwargs.get("args", kwargs.get("path", kwargs.get("file", "")))
            self.calls.append((name, str(target)))
            return fn(*args, **kwargs)

        return spy

    def touched(self, marker: str) -> list[tuple[str, str]]:
        return [call for call in self.calls if marker in call[1]]


def install_file_access_spy(monkeypatch: pytest.MonkeyPatch) -> FileAccessSpy:
    """Patches `open` (builtins, io, os), `os.stat`/`os.lstat`, `os.scandir`/`os.listdir` and `subprocess.run`/
    `Popen` (ffprobe/ffmpeg) for the rest of the test."""
    spy = FileAccessSpy()
    for module, name in ((builtins, "open"), (io, "open"), (os, "open"), (os, "stat"), (os, "lstat"),
                         (os, "scandir"), (os, "listdir"), (subprocess, "run"), (subprocess, "Popen")):
        monkeypatch.setattr(module, name, spy.wrap(f"{module.__name__}.{name}", getattr(module, name)))
    return spy
