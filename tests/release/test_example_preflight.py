"""The GPU example stops before any GPU work when the gated inputs are missing: the SAM 3D Objects files in
the Hugging Face cache at the pinned revision (shape stage) and MANO in the HaWoR checkout (hands stage)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
_SPEC = importlib.util.spec_from_file_location("run_example_preflight", REPO / "release" / "example" / "run_example.py")
E = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(E)
SAM3D = [f["file"] for m in json.loads((REPO / "release" / "models.json").read_text())["models"]
         if m["key"] == "sam3d" for f in m["files"]]


def _cache(present: set[str], seen: list[tuple[str, str, str]]):
    def cached(repo: str, filename: str, revision: str) -> str | None:
        seen.append((repo, filename, revision))
        return f"/cache/{filename}" if filename in present else None
    return cached


def _hawor(tmp_path: Path, files: tuple[str, ...]) -> Path:
    for rel in files:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes(b"x")
    return tmp_path


def test_everything_in_place_passes(tmp_path: Path) -> None:
    from fastdo.model_pins import HF_REVISIONS

    seen: list[tuple[str, str, str]] = []
    assert E.missing_gated_inputs(["shape", "hands"], _hawor(tmp_path, E.MANO_FILES), _cache(set(SAM3D), seen)) == []
    assert {r for _, _, r in seen} == {HF_REVISIONS["facebook/sam-3d-objects"]} and len(seen) == len(SAM3D) == 19


def test_missing_sam3d_and_mano_are_named_with_how_to_get_them(tmp_path: Path) -> None:
    lines = E.missing_gated_inputs(["decode", "shape", "hands"], _hawor(tmp_path, E.MANO_FILES[:1]),
                                   _cache(set(SAM3D[1:]), []))
    assert len(lines) == 2
    assert "1 of 19 files" in lines[0] and "hf auth login" in lines[0] and "--only sam3d" in lines[0]
    assert "MANO_LEFT.pkl is missing" in lines[1] and "mano.is.tue.mpg.de" in lines[1]


def test_only_the_stages_that_run_are_checked(tmp_path: Path) -> None:
    assert E.missing_gated_inputs(["decode", "mask"], tmp_path, _cache(set(), [])) == []
