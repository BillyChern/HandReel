"""The GPU gate G-pin: spconv kernel pin identity; real weights, GPU.

Twelve fresh workers through `start_sam3d_worker` in six same-order pairs D1, W1, D2, W2, D1, W1, one pinned
(`spconv_kernel_pin=True`) and one unpinned (the stock timing tuner, the positive control) per pair. The lead mode
alternates (ABBA), so both modes share the request orders, first requests and mean position in time, and host drift
cannot masquerade as a pin effect (an 8-process plan was rejected because its late processes were all pinned). Every
process records the host load average at its start and end. Every reply's mesh goes through
the shape stage's preparation (normalize_mesh, check_watertight, mesh_sdf_grid 64/0.15, canonical_samples 2048/0,
`_write_mesh`, `save_npz`) and its mesh.obj, sdf.npz and samples.npz are hashed; a request's state is the three digests.
- G-id: per clip, every pinned request gives one state, and every pinned reply's `kernels` map is identical.
- G-ctrl: some clip has >= 2 unpinned states; otherwise "inconclusive" (never "pass").
- G-mem: no failed request, and the largest pinned `peak_gpu_gib_reserved` is <= 21.0 GiB.
- G-time: per clip, median pinned worker seconds at positions >= 1 minus the unpinned median there is <= 0.5 s, and the
  pinned first-request median is <= the unpinned one + 0.5 s; a clip without both medians fails G-time.
The pinned states are compared with the reference states; a mismatch is reported (environment drift), not failed.

`artifact_state` and `request_measurements` are also imported by the shape fast-path evaluation: they
are the per-request row its T block is built from. `request_measurements` keeps a wall-QC fallback's seconds
and peak beside the gated pass's, never inside them.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import statistics
import tempfile
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from fastdo.engine.worker_support import save_npz
from fastdo.engine.workers_shape import _write_mesh
from fastdo.shape import ShapeConfig
from fastdo.shape.backend import ShapeResult
from fastdo.shape.mesh_utils import canonical_samples, check_watertight, normalize_mesh
from fastdo.shape.sam3d_backend import Sam3dBackend, Sam3dWorkerClient, start_sam3d_worker
from fastdo.solver.sdf import mesh_sdf_grid

__all__ = ["CLIP_FILES", "EXPECTED_B_STATES", "ORDERS", "PLAN", "artifact_state", "evaluate",
           "request_measurements", "run_kernel_pin_gate"]

ORDERS = {"D1": ("135545", "135250", "whisk"), "W1": ("whisk", "135545", "135250"),
          "D2": ("135250", "135545", "whisk"), "W2": ("whisk", "135250", "135545")}
PLAN = (("pinned", "D1"), ("tuner", "D1"), ("tuner", "W1"), ("pinned", "W1"), ("pinned", "D2"), ("tuner", "D2"),
        ("tuner", "W2"), ("pinned", "W2"), ("pinned", "D1"), ("tuner", "D1"), ("tuner", "W1"), ("pinned", "W1"))
CLIP_FILES = {"135250": "kf_135250.npz", "135545": "kf_135545.npz", "whisk": "whisk.npz"}
# SPEC 2.2, candidate (b) on H100 sm90, spconv 2.3.6: sha256 prefixes of mesh.obj / sdf.npz / samples.npz.
EXPECTED_B_STATES = {"135250": "5012528455d4/e49a5f49402a/12e6bed40634",
                     "135545": "8f4093543b47/b88ec3e81648/6530174db97b",
                     "whisk": "3061573b60a6/f00606f00cf6/5b9a2aff86d3"}
MEM_LIMIT_GIB = 21.0
TIME_LIMIT_S = 0.5

Row = dict[str, Any]


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact_state(vertices: np.ndarray, faces: np.ndarray, workdir: Path) -> dict[str, Any]:
    """The shape stage's mesh.obj, sdf.npz and samples.npz for one raw mesh (defaults 64/0.15/2048/0), hashed."""
    with tempfile.TemporaryDirectory(dir=workdir) as tmp:
        d = Path(tmp)
        v, f, _ = normalize_mesh(vertices, faces)
        v, f, watertight = check_watertight(v, f)
        sdf = mesh_sdf_grid(v, f, resolution=64, padding=0.15)
        points, normals = canonical_samples(v, f, 2048, 0)
        _write_mesh(d / "mesh.obj", v, f)
        save_npz(d / "sdf.npz", values=sdf.values.numpy(), origin=sdf.origin.numpy(),
                 step=np.array(sdf.step, dtype=np.float64))
        save_npz(d / "samples.npz", points=points.numpy(), normals=normals.numpy())
        files = {name: _sha(d / name) for name in ("mesh.obj", "sdf.npz", "samples.npz")}
    return {"files": files, "state": "/".join(files[n] for n in ("mesh.obj", "sdf.npz", "samples.npz")),
            "faces": len(f), "watertight": bool(watertight)}


def request_measurements(result: ShapeResult, client: Sam3dWorkerClient) -> Row:
    """One request's timing, QC and memory fields, with the gated pass and a wall-QC fallback kept apart.

    `worker_seconds` and `peak_gpu_gib_reserved` are the **gated** pass alone - what G-time and G-mem read (G-mem
    is on the reserved peak) - and the fallback's sit beside them under `fallback_*`. Both separations exist
    because the fallback is the slower and higher-peaking arm, so a merged quantity would report a fallback-path
    number under the gated arm's name (for the seconds and for the peaks).

    `fallback_worker_seconds` and `qc` are what a T-block harness computes **t_eff** from: t_eff
    (median gated worker seconds + flag rate x the fallback's median seconds) is the ONE place a fallback's seconds are
    charged, so without them a row has the gated numerator and nothing to charge t_eff with.
    """
    fallback = result.fallback_phase_seconds
    stats = client.last_postprocess_stats
    return {"worker_seconds": float(sum(result.phase_seconds.values())),
            "phase_seconds": dict(result.phase_seconds),
            "fallback_phase_seconds": dict(fallback) if fallback else None,
            "fallback_worker_seconds": float(sum(fallback.values())) if fallback else None,
            "qc": None if stats is None else stats.get("qc"),
            "postprocess_stats": dict(stats) if stats is not None else None,
            "peak_gpu_gib_reserved": client.last_peak_gpu_gib_reserved,
            "fallback_peak_gpu_gib_reserved": client.last_fallback_peak_gpu_gib_reserved}


def _run_worker(cfg: ShapeConfig, clips: dict[str, tuple[np.ndarray, np.ndarray]], proc: int, mode: str, order: str,
                workdir: Path) -> tuple[list[Row], dict[str, Any]]:
    rows: list[Row] = []
    info: dict[str, Any] = {"proc": proc, "mode": mode, "order": order, "started_utc": _utc(),
                            "loadavg_start": list(os.getloadavg())}
    client = start_sam3d_worker(cfg)
    info.update(pid=client.pid, load_seconds=client.load_seconds, spconv_kernels=client.spconv_kernels)
    params = Sam3dBackend(cfg).sampling_params | {"digest": True}
    try:
        for pos, clip in enumerate(ORDERS[order]):
            row: Row = {"proc": proc, "mode": mode, "order": order, "pos": pos, "clip": clip, "utc": _utc()}
            try:
                result = client.reconstruct(*clips[clip], params)
            except RuntimeError as exc:
                row.update(error=str(exc), peak_gpu_gib_reserved=client.last_peak_gpu_gib_reserved)
                rows.append(row)
                continue
            row.update(error=None, **request_measurements(result, client),
                       kernels=client.last_reply.get("kernels"), digests=client.last_digests,
                       **artifact_state(result.vertices, result.faces, workdir))
            rows.append(row)
    finally:
        info.update(exit_code=client.close(), ended_utc=_utc(), loadavg_end=list(os.getloadavg()))
    return rows, info


def _median(values: Sequence[float]) -> float | None:
    return statistics.median(values) if values else None


def _delta(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else a - b


def evaluate(rows: Sequence[Row]) -> dict[str, Any]:
    """Per-clip states and seconds, and the G-id, G-ctrl, G-mem and G-time verdicts (module docstring)."""
    ok = [r for r in rows if r.get("error") is None]
    pinned, tuner = [r for r in ok if r["mode"] == "pinned"], [r for r in ok if r["mode"] == "tuner"]
    clips: dict[str, Any] = {}
    for clip in sorted({r["clip"] for r in rows}):
        p, t = [r for r in pinned if r["clip"] == clip], [r for r in tuner if r["clip"] == clip]
        secs = {f"{mode}_{when}": [r["worker_seconds"] for r in reps if (r["pos"] == 0) == (when == "first")]
                for mode, reps in (("pinned", p), ("tuner", t)) for when in ("first", "later")}
        med = {name: _median(values) for name, values in secs.items()}
        states = sorted({r["state"] for r in p})
        short = ["/".join(h[:12] for h in s.split("/")) for s in states]
        clips[clip] = {
            "pinned_requests": len(p), "pinned_first_requests": sum(r["pos"] == 0 for r in p),
            "pinned_states": states, "pinned_faces": sorted({r["faces"] for r in p}),
            "tuner_requests": len(t), "tuner_states": sorted({r["state"] for r in t}),
            "tuner_faces": sorted({r["faces"] for r in t}),
            "matches_spec_b_state": short == [EXPECTED_B_STATES.get(clip)],
            "worker_seconds": secs, "median_worker_seconds": med,
            "delta_later_s": _delta(med["pinned_later"], med["tuner_later"]),
            "delta_first_s": _delta(med["pinned_first"], med["tuner_first"])}
    errors = [r for r in rows if r.get("error") is not None]
    kernel_maps = {repr(sorted((r.get("kernels") or {}).items())) for r in pinned}
    identical = bool(pinned) and all(len(c["pinned_states"]) == 1 for c in clips.values())
    g_id = "pass" if identical and len(kernel_maps) == 1 and "[]" not in kernel_maps and not any(
        r["mode"] == "pinned" for r in errors) else "fail"
    g_ctrl = "pass" if any(len(c["tuner_states"]) >= 2 for c in clips.values()) else "inconclusive"
    peaks = [r.get("peak_gpu_gib_reserved") for r in pinned]
    max_peak = max((x for x in peaks if x is not None), default=None)
    g_mem = "pass" if not errors and peaks and None not in peaks and max_peak is not None and max_peak <= MEM_LIMIT_GIB \
        else "fail"
    time_ok = bool(clips) and all(
        c["delta_later_s"] is not None and c["delta_later_s"] <= TIME_LIMIT_S
        and c["delta_first_s"] is not None and c["delta_first_s"] <= TIME_LIMIT_S for c in clips.values())
    return {"clips": clips, "errors": [{k: r.get(k) for k in ("proc", "mode", "clip", "pos", "error")} for r in errors],
            "kernels": pinned[-1].get("kernels") if pinned else None,
            "distinct_kernel_maps": len(kernel_maps), "max_peak_gpu_gib_reserved_pinned": max_peak,
            "gates": {"G-id": g_id, "G-ctrl": g_ctrl, "G-mem": g_mem, "G-time": "pass" if time_ok else "fail"},
            "verdict": "pass" if g_id == g_mem == "pass" and time_ok and g_ctrl == "pass"
            else "inconclusive" if g_id == g_mem == "pass" and time_ok else "fail"}


def _load_clips(inputs: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    clips = {}
    for clip, name in CLIP_FILES.items():
        with np.load(Path(inputs) / name, allow_pickle=False) as data:
            clips[clip] = (np.asarray(data["image"], dtype=np.uint8), np.asarray(data["mask"], dtype=bool))
    return clips


def run_kernel_pin_gate(inputs: Path, workdir: Path, cfg: ShapeConfig | None = None) -> dict[str, Any]:
    """Runs `PLAN` on `<inputs>/{kf_135250,kf_135545,whisk}.npz` and returns the JSON-ready gate record."""
    base = cfg if cfg is not None else ShapeConfig(backend="sam3d")
    clips, rows, processes = _load_clips(inputs), [], []
    started, t0 = _utc(), time.perf_counter()
    for proc, (mode, order) in enumerate(PLAN):
        proc_cfg = dataclasses.replace(base, spconv_kernel_pin=mode == "pinned")
        proc_rows, info = _run_worker(proc_cfg, clips, proc, mode, order, Path(workdir))
        rows.extend(proc_rows)
        processes.append(info)
    record = evaluate(rows)
    record.update(started_utc=started, ended_utc=_utc(), wall_seconds=time.perf_counter() - t0, processes=processes,
                  rows=rows, shape_config=dataclasses.asdict(base), plan=[list(p) for p in PLAN],
                  limits={"reserved_gib": MEM_LIMIT_GIB, "seconds": TIME_LIMIT_S},
                  expected_b_states=EXPECTED_B_STATES)
    return record
