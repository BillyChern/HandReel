# Copyright 2026 The HandReel authors. SPDX-License-Identifier: Apache-2.0
"""CPU entry point: the kinematic null audit of DO AS I DO's object test on stored references.

    python -m nullaudit clip_a.npz clip_b.npz ... [--json report.json]

Each .npz holds one clip:
- `reference`: (n, 7) or (n, 4, 4) reference on the executor grid, warm-up rows included; or
- `frames` (h, 7) raw reference poses plus scalars `ref_dt`, `sim_dt`, `warmup_rows`, `n_rows`
  (placed on the grid by `reference_on_executor_grid`);
- optional scalars `required_rows` (T; a shorter run fails) and `release_row` (W).
The release row sets the default lag of the time-shifted null (a quarter of the post-release rows). With `frames`
it is `warmup_rows`, and a stored `release_row` that differs is refused. With `reference` and no `release_row` the
release is ASSUMED at row 0; the report marks such clips (`release_assumed`).
The simulated nulls need a simulator; use the Python API (`audit_clip(..., simulator=...)`) for those.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .audit import KINEMATIC_NULLS, audit_clip, summarize
from .doasido import DoAsIDoObjectTest, reference_on_executor_grid


def _load(path: Path) -> tuple[np.ndarray, int | None, int | None, bool]:
    """One clip: the reference on the grid, T, W, and whether W was assumed (no release row stored or derivable)."""
    with np.load(path, allow_pickle=False) as z:
        rel = _count(z, "release_row", path) if "release_row" in z else None
        if "reference" in z:
            ref = np.asarray(z["reference"], np.float64)
        else:
            warm, rows = _count(z, "warmup_rows", path), _count(z, "n_rows", path)
            ref = reference_on_executor_grid(z["frames"], _step(z, "ref_dt", path), _step(z, "sim_dt", path),
                                             warm, rows).astype(np.float64)
            if rel is not None and rel != warm:
                raise ValueError(f"{path}: `release_row` {rel} differs from `warmup_rows` {warm}; on a grid built "
                                 "from frames the release is the first row after the warm-up")
            rel = warm
        req = _count(z, "required_rows", path) if "required_rows" in z else None
    return ref, req, rel, rel is None


def _step(z: Any, key: str, path: Path) -> float:
    v = np.asarray(z[key])
    if v.ndim != 0 or v.dtype.kind not in "iuf" or not np.isfinite(v) or v <= 0:
        raise ValueError(f"{path}: `{key}` must be a positive finite number of seconds, got {v.tolist()!r}")
    return float(v)


def _count(z: Any, key: str, path: Path) -> int:
    """A whole, non-negative row count as stored. `int()` would turn 4.5 into 4 and True into 1; neither is a
    count, so both are refused before any coercion."""
    v = np.asarray(z[key])
    if v.ndim != 0 or v.dtype.kind not in "iuf" or not np.isfinite(v) or v < 0 or v != np.floor(v):
        raise ValueError(f"{path}: `{key}` must be a whole non-negative number, got {v.tolist()!r}")
    return int(v)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="nullaudit", description=__doc__.split("\n\n")[0])
    ap.add_argument("clips", nargs="+", type=Path)
    ap.add_argument("--json", type=Path, default=None, help="write the full report here")
    a = ap.parse_args(argv)
    audits, assumed = [], []
    for p in a.clips:
        ref, req, rel, no_release = _load(p)
        assumed.append(no_release)
        audits.append(audit_clip(ref, DoAsIDoObjectTest(ref, required_rows=req), clip_id=p.stem,
                                 release_row=rel, nulls=KINEMATIC_NULLS))
    rep = summarize(audits)
    print(rep.headline())
    if any(assumed):
        print(f"release row not stored for {sum(assumed)} of {len(audits)} clips: assumed at row 0 "
              "(it sets the time-shifted null's default lag)")
    if a.json:
        doc = {**rep.to_dict(), "per_clip": [{"clip": c.clip_id, "z0": c.z0, "passing": c.passing(),
                                                 "release_assumed": no_release}
                                                for c, no_release in zip(audits, assumed, strict=True)]}
        a.json.write_text(json.dumps(doc, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
