"""Instance-size fusion: a POST-PASS over already-solved windows of one object instance.

The engine solves windows independently, so nothing here is a stage. After the windows of a video (or, on the lab, of a
labelled object) have their production `solve.npz`, the caller groups them by an instance key it owns (the narration
noun on Ego4D, the YCB id on DexYCB) and calls `plan_instance_scales`. For every window of an instance with K >= `min_k`
solved members it returns the scale that gives the window the instance's fused metric size; `write_instance_scale`
writes it where `solve_stage.fixed_scale` reads it (`read_fixed_scale`), and the fixed-scale re-solve then holds it.

Definitions (a registration-free size):
- sigma(T) = exp(mean_i log sqrt(lambda_i)), lambda_i the eigenvalues of the population covariance of `SIZE_SAMPLES`
  area-uniform surface samples of the canonical `mesh.obj` (`trimesh.sample.sample_surface`, seed `SIZE_SEED`);
- a window's metric size m_j = s_j * sigma(T_j), s_j its solved scale;
- the instance's fused size m* = exp(median_k log m_k) over its K solved members (own window included);
- the window's fused scale s^M_j = m* / sigma(T_j). K < `min_k`: no record, the window keeps its production solve.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from fastdo.engine.worker_support import load_npz
from fastdo.engine.workers_qc import write_json

__all__ = ["MIN_K", "RULE", "SIZE_SAMPLES", "SIZE_SEED", "plan_instance_scales", "read_fixed_scale",
           "template_size", "window_size", "write_instance_scale"]

MIN_K = 3
SIZE_SAMPLES = 10_000
SIZE_SEED = 0
RULE = "log-median of s*sigma(T) over the instance's solved windows (own included), K >= 3"
_MESH_FILE, _SOLVE_FILE = "mesh.obj", "solve.npz"


def template_size(vertices: np.ndarray, faces: np.ndarray, n: int = SIZE_SAMPLES, seed: int = SIZE_SEED) -> float:
    """sigma(T): the geometric mean of the principal standard deviations of `n` seeded area-uniform surface samples.

    Raises:
        ValueError: If the mesh has no area or the samples are degenerate (a zero principal deviation).
    """
    mesh = trimesh.Trimesh(vertices=np.asarray(vertices, dtype=np.float64), faces=np.asarray(faces, dtype=np.int64),
                           process=False)
    if not mesh.area > 0.0:
        raise ValueError("template_size needs a mesh with positive area")
    points, _ = trimesh.sample.sample_surface(mesh, n, seed=seed)
    x = np.asarray(points, dtype=np.float64)
    x = x - x.mean(0)
    std = np.sqrt(np.clip(np.linalg.eigvalsh(x.T @ x / len(x)), 0.0, None))
    if not np.all(std > 0.0):
        raise ValueError(f"template_size: degenerate samples (principal deviations {std.tolist()})")
    return float(np.exp(np.mean(np.log(std))))


def window_size(artifact_dir: Path) -> dict[str, float]:
    """`{scale: s_j, template_size: sigma(T_j), size_m: m_j}` of a solved window, from its `solve.npz` and `mesh.obj`.

    Raises:
        FileNotFoundError: If either file is missing.
        ValueError: If the scale is not finite and > 0.
    """
    artifact_dir = Path(artifact_dir)
    for name in (_SOLVE_FILE, _MESH_FILE):
        if not (artifact_dir / name).is_file():
            raise FileNotFoundError(f"{artifact_dir / name} is missing: the fusion reads solved windows only")
    scale = float(np.asarray(load_npz(artifact_dir / _SOLVE_FILE)["scale"]))
    if not (math.isfinite(scale) and scale > 0.0):
        raise ValueError(f"{artifact_dir}: solve.npz scale {scale!r} is not finite and > 0")
    mesh = trimesh.load(str(artifact_dir / _MESH_FILE), force="mesh", process=False)
    sigma = template_size(np.asarray(mesh.vertices), np.asarray(mesh.faces))
    return {"scale": scale, "template_size": sigma, "size_m": scale * sigma}


def plan_instance_scales(sizes: Mapping[str, Mapping[str, float]], instance_of: Mapping[str, str],
                         min_k: int = MIN_K) -> dict[str, dict[str, Any]]:
    """Per window, the fixed-scale record of its instance's fused size; windows of instances with K < `min_k` are absent.

    Args:
        sizes: window id -> `window_size` record, for the SOLVED windows only (an unsolved window is not a member).
        instance_of: window id -> instance key, for every window the caller grouped (solved or not).
        min_k: The smallest K that fuses (default 3).

    Raises:
        ValueError: If `min_k` < 1, or a solved window has no instance key.
    """
    if min_k < 1:
        raise ValueError(f"min_k must be >= 1, got {min_k}")
    missing = sorted(set(sizes) - set(instance_of))
    if missing:
        raise ValueError(f"solved windows without an instance key: {missing[:5]}")
    members: dict[str, list[str]] = {}
    for window in sorted(sizes):
        members.setdefault(instance_of[window], []).append(window)
    out: dict[str, dict[str, Any]] = {}
    for instance, windows in sorted(members.items()):
        if len(windows) < min_k:
            continue
        fused = float(np.exp(np.median([math.log(float(sizes[w]["size_m"])) for w in windows])))
        for w in windows:
            sigma = float(sizes[w]["template_size"])
            out[w] = {"scale": fused / sigma, "instance": instance, "k": len(windows), "members": list(windows),
                      "fused_size_m": fused, "template_size": sigma, "window_size_m": float(sizes[w]["size_m"]),
                      "production_scale": float(sizes[w]["scale"]), "rule": RULE}
    return out


def write_instance_scale(artifact_dir: Path, record: Mapping[str, Any], file: str = "instance_scale.json") -> Path:
    """Writes `record` (a `plan_instance_scales` value) to `<artifact_dir>/<file>` (tmp file then `os.replace`)."""
    path = Path(artifact_dir) / file
    write_json(path, dict(record))
    return path


def read_fixed_scale(path: Path) -> tuple[float, dict[str, Any]]:
    """`(scale, record)` from a fixed-scale file.

    Raises:
        FileNotFoundError: If the file is missing (`solve_stage.fixed_scale` never falls back silently).
        ValueError: If it is not a JSON object with a finite `scale` > 0.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing: solve_stage.fixed_scale needs the fusion post-pass's per-window "
                                "file (fastdo.engine.instance_scale); this window has no fused scale")
    record = json.loads(path.read_text())
    if not isinstance(record, dict) or "scale" not in record:
        raise ValueError(f"{path}: a fixed-scale file is a JSON object with a 'scale'")
    scale = float(record["scale"])
    if not (math.isfinite(scale) and scale > 0.0):
        raise ValueError(f"{path}: scale {record['scale']!r} is not finite and > 0")
    return scale, record
