"""Opt-in shape REUSE across windows of one object instance: a window listed in a donor map takes a
donor window's SAM 3D template instead of running the shape model.

Why: shape (SAM 3D) is the largest stage per attempted Ego4D window (~8.5 s), and an object recurs 5-8x per video. The
"best view of two generators" donor (larger keyframe mask area) is accuracy-neutral on GT -- lab scout (180 DexYCB
clips, post-contact) median paired delta -0.04 mm [-1.74, +1.29], and a fresh HOI4D confirmation (46 egocentric windows)
+0.15 mm [-0.80, +0.61], within a reseed's noise -- but on Ego4D it disagrees with the own template ~2x as often as a
SAM 3D reseed does (>10% scale moves 18-20 vs 9 of 64). Hence OPT-IN.

`bench.shape_reuse.donors` (null = OFF: the shape stage is the unwrapped model worker, byte for byte) names a JSON
file `{clip_id: donor_clip_id}`. For a listed clip the worker copies the donor's `mesh.obj`, `sdf.npz` and
`samples.npz` byte for byte from the donor's artifact directory (a sibling under the same artifact root) and writes
`shape.json` = the donor's metadata with this clip's `clip_id` and a `reuse` block (donor id and its keyframe). If the
donor's shape is not complete yet, the window's own shape is computed and `reuse.fallback` records why, so a window is
never left without a template. An unlisted clip runs the model worker unchanged.

`plan_donors` is the deployment rule the scout measured: per instance (e.g. the instance-size fusion's (video, noun)
key), the first `k_generators` windows in processing order run the model; every later window takes the generator with
the largest keyframe mask area (label-free).
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

__all__ = ["REUSED_FILES", "load_donor_map", "make_reuse_shape_worker", "plan_donors"]

REUSED_FILES = ("mesh.obj", "sdf.npz", "samples.npz")
_META = "shape.json"

StageFn = Callable[[str, Path], None]


def plan_donors(instance_of: Mapping[str, str], order: Sequence[str], mask_area: Mapping[str, float],
                k_generators: int = 2) -> dict[str, str]:
    """`{clip_id: donor_clip_id}` for every window after the first `k_generators` of its instance (in `order`); the
    donor is the generator with the largest `mask_area` (ties: the earlier one). Windows without an instance key
    (absent or None in `instance_of`) are never listed.

    Raises:
        ValueError: If `k_generators < 1`, `order` repeats a clip, or a generator has no `mask_area`.
    """
    if k_generators < 1:
        raise ValueError(f"k_generators must be >= 1, got {k_generators}")
    if len(set(order)) != len(order):
        raise ValueError("order repeats a clip id")
    generators: dict[str, list[str]] = {}
    donors: dict[str, str] = {}
    for clip in order:
        key = instance_of.get(clip)
        if key is None:
            continue
        gens = generators.setdefault(key, [])
        if len(gens) < k_generators:
            if clip not in mask_area:
                raise ValueError(f"generator {clip!r} has no mask_area")
            gens.append(clip)
            continue
        donors[clip] = max(gens, key=lambda g: (mask_area[g], -gens.index(g)))
    return donors


def load_donor_map(path: Path) -> dict[str, str]:
    """The `{clip_id: donor_clip_id}` JSON; refuses a self-donor or a donor that is itself a reusing window."""
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in data.items()):
        raise ValueError(f"{path}: expected a JSON object {{clip_id: donor_clip_id}}")
    bad = sorted(k for k, v in data.items() if k == v or v in data)
    if bad:
        raise ValueError(f"{path}: self-donors or chained donors (a donor must compute its own shape): {bad[:5]}")
    return dict(data)


def make_reuse_shape_worker(inner: StageFn, donors: Mapping[str, str]) -> StageFn:
    """Wrap the shape stage's worker (module docstring). `inner` runs for unlisted clips and on fallback."""
    table = dict(donors)

    def reuse_shape_worker(clip_id: str, artifact_dir: Path) -> None:
        artifact_dir = Path(artifact_dir)
        donor = table.get(clip_id)
        if donor is None:
            inner(clip_id, artifact_dir)
            return
        src = artifact_dir.parent / donor
        missing = [f for f in (*REUSED_FILES, _META) if not (src / f).is_file()]
        if missing:
            inner(clip_id, artifact_dir)
            meta = json.loads((artifact_dir / _META).read_text())
            meta["reuse"] = {"donor": donor, "fallback": f"donor shape incomplete (missing {missing})"}
            (artifact_dir / _META).write_text(json.dumps(meta, indent=2))
            return
        artifact_dir.mkdir(parents=True, exist_ok=True)
        for f in REUSED_FILES:
            shutil.copyfile(src / f, artifact_dir / f)
        meta: dict[str, Any] = json.loads((src / _META).read_text())
        meta["reuse"] = {"donor": donor, "donor_keyframe_index": meta.get("keyframe_index"),
                         "donor_frame_idx": meta.get("frame_idx")}
        meta["clip_id"] = clip_id
        (artifact_dir / _META).write_text(json.dumps(meta, indent=2))

    return reuse_shape_worker
