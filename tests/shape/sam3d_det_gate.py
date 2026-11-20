"""The SAM 3D determinism gate runner (checks H2, M1, M2 and M3); real weights, GPU.

Process 1: per clip one discarded warm-up request, then A D A D A D (A: `deterministic` False, D: True). Process 2, a
fresh worker (H2): per clip a warm-up, then D D. Every request asks for the M2 digests.
- G1 bit identity: per clip, the three process-1 D meshes and the two process-2 D meshes all equal process 1's first D
  mesh byte for byte (vertices float64, faces int64). It is "pass" only if some clip's A meshes differ in the same run
  (the M3 positive control); with equal D meshes and no A difference it is "inconclusive".
- G2 memory: no failed request, and the largest `peak_gpu_gib_reserved` of any D request is <= 21.0 GiB (M1).
- G3 time: per clip, median(D) - median(A) of process 1's `structured_latent_flow + decode` seconds is <= 0.5 s.
"""

from __future__ import annotations

import dataclasses
import os
import statistics
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from fastdo.shape import ShapeConfig
from fastdo.shape.sam3d_backend import Sam3dBackend, start_sam3d_worker

__all__ = ["G2_LIMIT_GIB", "G3_LIMIT_S", "GATE_PHASES", "evaluate_gates", "run_f2_gates", "run_worker_sequence"]

GATE_PHASES = ("structured_latent_flow", "decode")
G2_LIMIT_GIB = 21.0
G3_LIMIT_S = 0.5
_DIGEST_PARTS = ("slat_feats", "flexicubes_input", "mesh_vertices", "mesh_faces")

Rep = dict[str, Any]


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_worker_sequence(cfg: ShapeConfig, clips: dict[str, tuple[np.ndarray, np.ndarray]],
                        sequence: str) -> tuple[dict[str, list[Rep]], dict[str, Any]]:
    """One fresh SAM 3D worker: per clip a warm-up (kept only if it fails), then `sequence` of A/D requests."""
    reps: dict[str, list[Rep]] = {clip: [] for clip in clips}
    info: dict[str, Any] = {"started_utc": _utc(), "sequence": "W" + sequence}
    client = start_sam3d_worker(cfg)
    info.update(pid=client.pid, load_seconds=client.load_seconds)
    params = Sam3dBackend(cfg).sampling_params
    try:
        for clip, (image, mask) in clips.items():
            for mode in "W" + sequence:
                start = time.perf_counter()
                try:
                    result = client.reconstruct(image, mask, params | {"deterministic": mode == "D", "digest": True})
                except RuntimeError as exc:
                    reps[clip].append({"mode": mode, "error": str(exc)})
                    continue
                if mode == "W":
                    continue
                reps[clip].append({
                    "mode": mode, "vertices": result.vertices, "faces": result.faces,
                    "phase_seconds": dict(result.phase_seconds), "wall_seconds": time.perf_counter() - start,
                    "peak_gpu_gib_allocated": client.last_peak_gpu_gib,
                    "peak_gpu_gib_reserved": client.last_peak_gpu_gib_reserved, "digests": client.last_digests})
    finally:
        info.update(exit_code=client.close(), ended_utc=_utc())
    return reps, info


def _same(a: Rep, b: Rep) -> bool:
    return (a["vertices"].shape == b["vertices"].shape and a["vertices"].dtype == b["vertices"].dtype
            and a["vertices"].tobytes() == b["vertices"].tobytes() and a["faces"].shape == b["faces"].shape
            and a["faces"].tobytes() == b["faces"].tobytes())


def _all_same(reps: Sequence[Rep], expected: int) -> bool:
    return len(reps) == expected and all(_same(reps[0], rep) for rep in reps[1:])


def _surface_mean(a: Rep, b: Rep, n: int = 20000, seed: int = 3) -> float:
    """Mean two-sided point-to-surface distance over `a`'s bounding-box diagonal."""
    mesh_a = trimesh.Trimesh(a["vertices"], a["faces"], process=False)
    mesh_b = trimesh.Trimesh(b["vertices"], b["faces"], process=False)
    diag = float(np.linalg.norm(mesh_a.bounds[1] - mesh_a.bounds[0]))
    pa = trimesh.sample.sample_surface(mesh_a, n, seed=seed)[0]
    pb = trimesh.sample.sample_surface(mesh_b, n, seed=seed + 1)[0]
    d = np.r_[trimesh.proximity.closest_point(mesh_b, pa)[1], trimesh.proximity.closest_point(mesh_a, pb)[1]]
    return float(d.mean() / diag)


def _median(values: Sequence[float]) -> float | None:
    return statistics.median(values) if values else None


def _clip_record(p1: list[Rep], p2: list[Rep]) -> dict[str, Any]:
    ok1, ok2 = [r for r in p1 if "error" not in r], [r for r in p2 if "error" not in r]
    a, d = [r for r in ok1 if r["mode"] == "A"], [r for r in ok1 if r["mode"] == "D"]
    d2 = [r for r in ok2 if r["mode"] == "D"]
    gate_s = {name: [sum(r["phase_seconds"][p] for p in GATE_PHASES) for r in reps] for name, reps in (("A", a), ("D", d))}
    total_s = {name: [sum(r["phase_seconds"].values()) for r in reps] for name, reps in (("A", a), ("D", d))}
    med_a, med_d = _median(gate_s["A"]), _median(gate_s["D"])
    digests = {part: {"A_distinct": len({(r["digests"] or {}).get(part) for r in a}),
                      "D_in_process_distinct": len({(r["digests"] or {}).get(part) for r in d}),
                      "D_both_processes_distinct": len({(r["digests"] or {}).get(part) for r in d + d2})}
               for part in _DIGEST_PARTS}
    return {
        "errors": [r["error"] for r in p1 + p2 if "error" in r],
        "faces_A": [len(r["faces"]) for r in a], "faces_D": [len(r["faces"]) for r in d],
        "faces_D_process2": [len(r["faces"]) for r in d2],
        "A_all_equal": _all_same(a, 3), "D_in_process_equal": _all_same(d, 3),
        "D_cross_process_equal": len(d) == 3 and len(d2) == 2 and all(_same(d[0], r) for r in d2),
        "digests": digests,
        "gate_seconds": gate_s, "median_gate_seconds": {"A": med_a, "D": med_d},
        "delta_gate_seconds": None if med_a is None or med_d is None else med_d - med_a,
        "total_seconds": total_s, "median_total_seconds": {k: _median(v) for k, v in total_s.items()},
        "median_phase_seconds": {name: {p: _median([r["phase_seconds"][p] for r in reps]) for p in reps[0]["phase_seconds"]}
                                 for name, reps in (("A", a), ("D", d), ("D_process2", d2)) if reps},
        "wall_seconds": {name: [r["wall_seconds"] for r in reps] for name, reps in (("A", a), ("D", d), ("D_process2", d2))},
        "peak_gpu_gib_reserved": {name: [r["peak_gpu_gib_reserved"] for r in reps]
                                  for name, reps in (("A", a), ("D", d), ("D_process2", d2))},
        "peak_gpu_gib_allocated": {name: [r["peak_gpu_gib_allocated"] for r in reps]
                                   for name, reps in (("A", a), ("D", d), ("D_process2", d2))},
        "surface_mean_D0_vs_A0_over_diag": _surface_mean(d[0], a[0]) if a and d else None,
        "surface_mean_A0_vs_A1_over_diag": _surface_mean(a[0], a[1]) if len(a) > 1 else None,
    }


def evaluate_gates(p1: dict[str, list[Rep]], p2: dict[str, list[Rep]]) -> dict[str, Any]:
    """The per-clip records and the G1-G3 verdicts (module docstring)."""
    clips = {clip: _clip_record(p1[clip], p2.get(clip, [])) for clip in p1}
    errors = any(c["errors"] for c in clips.values())
    d_equal = all(c["D_in_process_equal"] and c["D_cross_process_equal"] for c in clips.values())
    positive = any(not c["A_all_equal"] for c in clips.values())
    g1 = "pass" if d_equal and positive else "inconclusive" if d_equal else "fail"
    peaks = [p for c in clips.values() for key in ("D", "D_process2") for p in c["peak_gpu_gib_reserved"][key]]
    max_peak = max((p for p in peaks if p is not None), default=None)
    g2 = "pass" if not errors and peaks and None not in peaks and max_peak is not None and max_peak <= G2_LIMIT_GIB \
        else "fail"
    deltas = [c["delta_gate_seconds"] for c in clips.values()]
    g3 = "pass" if deltas and all(x is not None and x <= G3_LIMIT_S for x in deltas) else "fail"
    return {"clips": clips, "positive_control": positive,
            "G1_in_process_only": all(c["D_in_process_equal"] for c in clips.values()),
            "max_peak_gpu_gib_reserved_D": max_peak, "gates": {"G1": g1, "G2": g2, "G3": g3}}


def run_f2_gates(inputs: Path, clip_names: Sequence[str], cfg: ShapeConfig | None = None) -> dict[str, Any]:
    """Runs both worker processes on `<inputs>/kf_<clip>.npz` and returns the JSON-ready gate record."""
    cfg = cfg if cfg is not None else ShapeConfig(backend="sam3d")
    clips: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in clip_names:
        with np.load(Path(inputs) / f"kf_{name}.npz", allow_pickle=False) as data:
            clips[name] = (np.asarray(data["image"], dtype=np.uint8), np.asarray(data["mask"], dtype=bool))
    started = _utc()
    p1, info1 = run_worker_sequence(cfg, clips, "ADADAD")
    p2, info2 = run_worker_sequence(cfg, clips, "DD")
    record = evaluate_gates(p1, p2)
    record.update(started_utc=started, ended_utc=_utc(), shape_config=dataclasses.asdict(cfg),
                  processes=[info1, info2], spconv_algo=os.environ.get("SPCONV_ALGO"),
                  limits={"G2_reserved_gib": G2_LIMIT_GIB, "G3_seconds": G3_LIMIT_S, "G3_phases": list(GATE_PHASES)})
    return record
