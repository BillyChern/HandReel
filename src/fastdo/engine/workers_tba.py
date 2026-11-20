"""The opt-in `tba` stage (TBA-V: temporal object BA + hand-aware carved template, BASE's gauge held).

OFF by default: `tba_stage: null` in run/conf/config.yaml. Then `fastdo.bench.bench_stages.build_stage_fns` registers
no `tba` stage, the stage list and every artefact are BASE's byte for byte (KA-ID), and nothing here runs.

With a block (`fastdo.tba.config.TbaConfig`; `{}` = the default TBA-V) the stage runs on CPU after `solve` and, when
registered, `contact`, and before `qc`; it reads the stored artefacts only (`fastdo.tba.window.load_window`) and
writes, beside them (nothing existing is modified, so the BASE trajectory stays on disk as provenance):

- `tba.npz`: `poses` (T,4,4) float64, `scale` 0-d float64, `frame_idx` (T,) int64, `carved` 0-d bool,
  `source_sha256` 0-d str (the `contact.npz`, else `solve.npz`, it was computed from: a stale result is detectable);
- `tba_mesh.obj`: the template (canonical frame) the poses place; the stored `mesh.obj` when nothing was carved;
- `tba.json`: the run record (BA rows, anchors, gauge, carving fractions, seconds).

`qc` and `clip_record.npz` do NOT read `tba.npz` yet: its variants are scored by a separate harness on BASE artefacts.
Wiring the corrected trajectory into qc is a separate change, made only if TBA-V becomes a default.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from fastdo.engine.orchestrator import StageFn
from fastdo.engine.worker_support import save_npz
from fastdo.tba.config import TbaConfig
from fastdo.tba.window import load_window, run_tba

__all__ = ["TBA_FILE", "TBA_MESH_FILE", "TBA_META_FILE", "make_tba_worker", "with_tba"]

TBA_FILE, TBA_MESH_FILE, TBA_META_FILE = "tba.npz", "tba_mesh.obj", "tba.json"


def with_tba(stages: tuple[str, ...]) -> tuple[str, ...]:
    """`stages` with "tba" right after "contact" (or after "solve" without contact) and before "qc" (idempotent).

    Raises:
        ValueError: If "solve" is absent or "tba" already sits elsewhere.
    """
    stages = tuple(stages)
    if "solve" not in stages:
        raise ValueError(f"with_tba needs a 'solve' stage, got {stages}")
    at = stages.index("contact") if "contact" in stages else stages.index("solve")
    if "tba" in stages:
        if stages.count("tba") == 1 and stages.index("tba") == at + 1:
            return stages
        raise ValueError(f"'tba' must follow {stages[at]!r} immediately (once), got {stages}")
    return stages[:at + 1] + ("tba",) + stages[at + 1:]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_tba_worker(cfg: TbaConfig) -> StageFn:
    """The `tba` stage worker for an ENABLED config (module docstring). Building it reads nothing."""

    def worker(clip_id: str, artifact_dir: Path) -> None:
        import trimesh

        artifact_dir = Path(artifact_dir)
        for name in (TBA_FILE, TBA_MESH_FILE, TBA_META_FILE):
            (artifact_dir / name).unlink(missing_ok=True)
        win = load_window(artifact_dir)
        out = run_tba(win, cfg)
        src = artifact_dir / ("contact.npz" if win.has_contact else "solve.npz")
        tmp = artifact_dir / (TBA_MESH_FILE + ".tmp")
        trimesh.Trimesh(vertices=out.mesh_v, faces=out.mesh_f, process=False).export(str(tmp), file_type="obj")
        os.replace(tmp, artifact_dir / TBA_MESH_FILE)
        save_npz(artifact_dir / TBA_FILE, poses=np.asarray(out.poses, np.float64),
                 scale=np.array(out.scale, np.float64), frame_idx=win.frame_idx, carved=np.array(out.carved),
                 source_sha256=np.array(_sha256(src)))
        meta = {"clip_id": clip_id, "source": src.name, **out.record}
        tmpj = artifact_dir / (TBA_META_FILE + ".tmp")
        tmpj.write_text(json.dumps(meta, indent=2, default=float))
        os.replace(tmpj, artifact_dir / TBA_META_FILE)

    return worker
