"""Profile `retarget_kinematic` on the whisking demo: per-clip walls, component breakdown, cProfile.

Clips (T = the demo's 138 frames, s = `profile_kinematic.engine_stride`):
  engine_whisking  keys `arange(0, T, s)` plus `T - 1` (47 frames) at `frequency / s`;
  engine_10s       the keys forward, backward without the repeated endpoint, forward again: first 100 frames;
  full_whisking    `arange(T)` at the demo frequency (50 Hz), the gate clip;
  full_10s         the same forward-backward-forward construction on `arange(T)`: first 300 frames.
The frame lists are computed inline, so the script also runs on a commit without `fastdo.retarget.rate`.

Per clip: a warm-up call on the clip's first 10 frames; one run with the component functions wrapped
wherever they are bound (inclusive time `breakdown_s`, `calls`, and `attributed_s`: a component call counts
only when it starts inside an LM evaluation with no other component active, so nested timers such as `fk`
inside `wrist_prior` count once); a cProfile run for the clips in `cprofile_clips`; then `repeats` plain runs,
each with the cgroup `throttled_usec` delta, the cgroup "some avg10" before the call and the 1-min load
average. Every timed call is `retarget_kinematic(...)` plus `KinematicTrajectory.save`.

`profile_kinematic.legacy=true` selects the earlier merged path (`RetargetConfig(fast_path=False)`) when that field
exists. `write_reference` (repo-relative npz path) stores the legacy qpos/qvel of the engine and full-rate
whisking clips, the parity fixture. Output: `<paths.outputs>/retarget/profile_<label>.json`.
"""

from __future__ import annotations

import contextlib
import cProfile
import dataclasses
import functools
import importlib
import importlib.util
import json
import logging
import os
import pstats
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
import trimesh
from omegaconf import DictConfig

import fastdo.retarget.kinematic as kinematic_module
from fastdo.bench.gpu_budget import apply_cpu_budget, apply_gpu_budget
from fastdo.retarget.baseline_io import baseline_world_inputs
from fastdo.retarget.kinematic import KinematicTrajectory, RetargetConfig
from fastdo.retarget.labels import mano_finger_labels
from fastdo.retarget.mj_model import RobotScene
from fastdo.solver.sdf import mesh_sdf_grid

logger = logging.getLogger(__name__)

_EVALUATE = ("evaluate_refresh", "evaluate_trial")
_COMPONENTS = ("kinematics", "fk", "canonical", "transport", "sinkhorn", "capsule_search", "primitive_term",
               "band_term", "wrist_prior")
_TARGETS: dict[str, tuple[str, ...]] = {
    "kinematics": ("RobotScene.point_kinematics", "BatchedKinematics.forward", "BatchedKinematics.world_points"),
    "fk": ("RobotScene.fk",),
    "canonical": ("robot_canonical_points", "BatchedKinematics.canonical_jacobian"),
    "transport": ("contact_transport_term", "contact_transport_term_inband"),
    "sinkhorn": ("sinkhorn_unbalanced", "sinkhorn_unbalanced_uniform_b"),
    "capsule_search": ("capsule_min_distance",),
    "primitive_term": ("primitive_penetration_term",),
    "band_term": ("contact_band_term",),
    "wrist_prior": ("wrist_prior_term", "wrist_prior_term_closed_form"),
    "lm_assemble": ("assemble_block_system",),
    "lm_solve": ("solve_block_tridiagonal",),
    "human_measure": ("human_contact_measure",),
}
_FUNC_MODULES = ("fastdo.retarget.kinematic", "fastdo.retarget.kinematic_fast", "fastdo.retarget.kinematic_terms",
                 "fastdo.retarget.transport", "fastdo.retarget.primitives", "fastdo.solver.lm_blocks")
_CLASS_MODULES = {"RobotScene": "fastdo.retarget.mj_model", "BatchedKinematics": "fastdo.retarget.batched_kinematics"}
_LM_MODULES = ("fastdo.retarget.kinematic", "fastdo.retarget.kinematic_fast")
_REFERENCE_CLIPS = ("engine_whisking", "engine_10s", "full_whisking")


class _Recorder:
    """Inclusive and outermost-component times of wrapped calls, kept with one stack of active calls."""

    def __init__(self) -> None:
        self.breakdown: dict[str, float] = defaultdict(float)
        self.attributed: dict[str, float] = defaultdict(float)
        self.calls: dict[str, int] = defaultdict(int)
        self.stage_iterations: list[int] = []
        self._stack: list[str] = []

    def wrap(self, name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def timed(*args: Any, **kwargs: Any) -> Any:
            outermost = (name in _COMPONENTS and any(s in _EVALUATE for s in self._stack)
                         and not any(s in _COMPONENTS for s in self._stack))
            self._stack.append(name)
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                dt = time.perf_counter() - t0
                self._stack.pop()
                self.breakdown[name] += dt
                self.calls[name] += 1
                if outermost:
                    self.attributed[name] += dt
        return timed

    def wrap_lm(self, lm: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(lm)
        def optimize(x0: Any, evaluate: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
            refresh_fn, trial_fn = self.wrap("evaluate_refresh", evaluate), self.wrap("evaluate_trial", evaluate)

            def split(x: Any, refresh: bool) -> Any:
                return (refresh_fn if refresh else trial_fn)(x, refresh)

            result = lm(x0, split, *args, **kwargs)
            self.stage_iterations.append(int(result.iterations))
            return result
        return optimize

    def unexplained_eval_frac(self) -> float:
        total = sum(self.breakdown.get(name, 0.0) for name in _EVALUATE)
        if total <= 0.0:
            return float("nan")
        frac = 1.0 - sum(self.attributed.get(name, 0.0) for name in _COMPONENTS) / total
        if frac < -1e-9:
            raise RuntimeError(f"attributed component time exceeds evaluate time (frac {frac:.3g}): a double count")
        return frac


@contextlib.contextmanager
def _patched(rec: _Recorder) -> Iterator[None]:
    """Wrap every `_TARGETS` name where it is bound (one wrapper per name) and each module's LM call."""
    undo: list[tuple[object, str, object]] = []

    def _set(owner: object, attr: str, value: object) -> None:
        undo.append((owner, attr, getattr(owner, attr)))
        setattr(owner, attr, value)

    modules = [importlib.import_module(m) for m in _FUNC_MODULES if importlib.util.find_spec(m) is not None]
    for name, targets in _TARGETS.items():
        for target in targets:
            if "." in target:
                cls_name, method = target.split(".")
                if importlib.util.find_spec(_CLASS_MODULES[cls_name]) is None:
                    continue
                cls = getattr(importlib.import_module(_CLASS_MODULES[cls_name]), cls_name, None)
                if cls is not None and hasattr(cls, method):
                    _set(cls, method, rec.wrap(name, getattr(cls, method)))
                continue
            bound = [m for m in modules if hasattr(m, target)]
            if bound:
                original = getattr(bound[0], target)
                wrapper = rec.wrap(name, original)
                for module in bound:
                    if getattr(module, target) is original:
                        _set(module, target, wrapper)
    for module in modules:
        if module.__name__ in _LM_MODULES and hasattr(module, "lm_optimize_blocks"):
            _set(module, "lm_optimize_blocks", rec.wrap_lm(module.lm_optimize_blocks))
    try:
        yield
    finally:
        for owner, attr, value in reversed(undo):
            setattr(owner, attr, value)


def _clip_frames(name: str, n_frames: int, stride: int) -> tuple[np.ndarray, int]:
    """`(frames int64, rate divisor)` of a named clip (module docstring)."""
    keys = np.arange(0, n_frames, stride, dtype=np.int64)
    if keys[-1] != n_frames - 1:
        keys = np.append(keys, n_frames - 1)
    full = np.arange(n_frames, dtype=np.int64)
    if name == "engine_whisking":
        return keys, stride
    if name == "engine_10s":
        return np.concatenate([keys, keys[::-1][1:], keys[1:]])[:100], stride
    if name == "full_whisking":
        return full, 1
    if name == "full_10s":
        return np.concatenate([full, full[::-1][1:], full[1:]])[:300], 1
    raise ValueError(f"unknown clip {name!r}")


def _cgroup_sample() -> tuple[int | None, float | None]:
    """cgroup `cpu.stat` throttled_usec and `cpu.pressure` "some avg10" (None when unreadable)."""
    throttled: int | None = None
    pressure: float | None = None
    with contextlib.suppress(OSError, ValueError, IndexError):
        for line in Path("/sys/fs/cgroup/cpu.stat").read_text().splitlines():
            key, value = line.split()
            if key == "throttled_usec":
                throttled = int(value)
    with contextlib.suppress(OSError, ValueError, IndexError):
        some = Path("/sys/fs/cgroup/cpu.pressure").read_text().splitlines()[0].split()
        pressure = float(some[1].split("=")[1])
    return throttled, pressure


def _git_sha(root: Path) -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _top_cumulative(profile: cProfile.Profile, n: int = 25) -> list[list[Any]]:
    stats = pstats.Stats(profile).sort_stats("cumulative")
    rows: list[list[Any]] = []
    for func in stats.fcn_list[:n]:  # type: ignore[attr-defined]
        _cc, ncalls, _tt, cumtime, _callers = stats.stats[func]  # type: ignore[attr-defined]
        rows.append([f"{func[0]}:{func[1]}({func[2]})", int(ncalls), round(float(cumtime), 4)])
    return rows


def _resolve(root: Path, path: str) -> Path:
    return Path(path) if Path(path).is_absolute() else root / path


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    from fastdo import hostpaths

    hostpaths.check_config_paths(cfg, [("isolation", k, v, False) for k, v, create in hostpaths.ISOLATION_HOST_KEYS
                                       if not create] + [("isolation", "isolation.mano_right_pkl", "FASTDO_MANO_DIR",
                                                          False)])
    apply_cpu_budget()
    apply_gpu_budget()
    root = Path(hydra.utils.get_original_cwd())
    pk = cfg.profile_kinematic
    clips = [str(c) for c in pk.clips]
    fields = {f.name for f in dataclasses.fields(RetargetConfig)}
    rcfg = (dataclasses.replace(RetargetConfig(), fast_path=False)  # type: ignore[call-arg]
            if bool(pk.legacy) and "fast_path" in fields else RetargetConfig())
    legacy_path = not getattr(rcfg, "fast_path", False)
    write_reference = str(pk.get("write_reference", ""))
    if write_reference and not (legacy_path and all(c in clips for c in _REFERENCE_CLIPS)):
        sys.stderr.write(f"write_reference needs the legacy path and clips {list(_REFERENCE_CLIPS)}; "
                         f"got legacy_path={legacy_path}, clips={clips}\n")
        sys.exit(2)

    setup: dict[str, float] = {}
    t0 = time.perf_counter()
    scene = RobotScene.from_xml(cfg.isolation.demo_scene_xml)
    setup["scene_from_xml"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    mesh = trimesh.load(cfg.isolation.whisk_obj, process=False)
    assert isinstance(mesh, trimesh.Trimesh), f"expected a single mesh, got {type(mesh)}"
    setup["mesh_load"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    sdf = mesh_sdf_grid(np.asarray(mesh.vertices), np.asarray(mesh.faces), resolution=64)
    setup["sdf_grid"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    surf_pts, face_idx = trimesh.sample.sample_surface(mesh, 1024, seed=np.random.default_rng(0))
    surface = torch.as_tensor(np.asarray(surf_pts), dtype=torch.float64)
    normals = torch.as_tensor(np.asarray(mesh.face_normals[face_idx]), dtype=torch.float64)
    setup["surface_samples"] = time.perf_counter() - t0

    obj_poses, wrist_poses, hand_verts, frequency = baseline_world_inputs(
        cfg.isolation.demo_keypoints_npz, cfg.isolation.demo_trajectory_kinematic_npz)
    labels = mano_finger_labels(cfg.isolation.mano_right_pkl)
    scale = float(cfg.isolation.mesh_scale)
    tmp_dir = Path(tempfile.mkdtemp(prefix="profile_kinematic_"))

    def solve(frames: np.ndarray, freq: float) -> tuple[KinematicTrajectory, float]:
        idx = torch.as_tensor(frames)
        obj, wrist, verts = obj_poses[idx], wrist_poses[idx], hand_verts[idx]
        start = time.perf_counter()
        traj = kinematic_module.retarget_kinematic(obj, wrist, scale, verts, labels, scene, sdf, surface, normals,
                                                   freq, rcfg)
        traj.save(tmp_dir / "trajectory_kinematic.npz")
        return traj, time.perf_counter() - start

    n_total = int(obj_poses.shape[0])
    result: dict[str, Any] = {"git_sha": _git_sha(root), "legacy_path": legacy_path, "label": str(pk.label),
                              "threads": torch.get_num_threads(), "setup_s": setup, "clips": {}}
    last: dict[str, tuple[np.ndarray, KinematicTrajectory, float]] = {}
    cprofile_clips = {str(c) for c in pk.get("cprofile_clips", ["engine_whisking", "engine_10s"])}
    for name in clips:
        frames, divisor = _clip_frames(name, n_total, int(pk.engine_stride))
        freq = frequency / divisor
        logger.info("clip %s: %d frames at %.4g Hz (legacy_path=%s)", name, frames.shape[0], freq, legacy_path)
        solve(frames[:10], freq)
        rec = _Recorder()
        with _patched(rec):
            solve(frames, freq)
        entry: dict[str, Any] = {
            "n_frames": int(frames.shape[0]), "frequency": freq,
            "breakdown_s": dict(rec.breakdown), "attributed_s": dict(rec.attributed), "calls": dict(rec.calls),
            "n_refresh": rec.calls.get("evaluate_refresh", 0), "n_trial": rec.calls.get("evaluate_trial", 0),
            "stage_iterations": rec.stage_iterations, "unexplained_eval_frac": rec.unexplained_eval_frac(),
        }
        if name in cprofile_clips:
            profile = cProfile.Profile()
            profile.enable()
            solve(frames, freq)
            profile.disable()
            entry["cprofile_top25_cumulative"] = _top_cumulative(profile)
        runs: list[dict[str, Any]] = []
        for _ in range(int(pk.repeats)):
            throttled0, pressure = _cgroup_sample()
            load = os.getloadavg()[0]
            traj, wall = solve(frames, freq)
            throttled1, _ = _cgroup_sample()
            delta = None if throttled0 is None or throttled1 is None else throttled1 - throttled0
            runs.append({"wall_s": wall, "throttled_usec_delta": delta, "some_avg10_before": pressure,
                         "loadavg_1min": load})
            last[name] = (frames, traj, freq)
            logger.info("  %s wall %.3f s (some avg10 %s, throttled delta %s usec)", name, wall, pressure, delta)
        walls = [r["wall_s"] for r in runs]
        entry.update(runs=runs, wall_min_s=min(walls) if walls else None,
                     wall_median_s=float(np.median(walls)) if walls else None)
        result["clips"][name] = entry
        logger.info("clip %s: min %s s, stage iterations %s, refresh/trial %d/%d, unexplained %.3f", name,
                    entry["wall_min_s"], rec.stage_iterations, entry["n_refresh"], entry["n_trial"],
                    entry["unexplained_eval_frac"])

    out = _resolve(root, str(cfg.paths.outputs)) / "retarget" / f"profile_{pk.label}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    logger.info("profile -> %s", out)

    if write_reference:
        (f_full, t_full, freq_full), (f_eng, t_eng, freq_eng), (f_10s, t_10s, _) = (
            last["full_whisking"], last["engine_whisking"], last["engine_10s"])
        ref = _resolve(root, write_reference)
        ref.parent.mkdir(parents=True, exist_ok=True)
        np.savez(ref, qpos_full=t_full.qpos, qvel_full=t_full.qvel, frequency_full=np.asarray(freq_full),
                 qpos_engine=t_eng.qpos, qvel_engine=t_eng.qvel, frequency_engine=np.asarray(freq_eng),
                 key_frames=f_eng.astype(np.int64), qpos_engine10s=t_10s.qpos, qvel_engine10s=t_10s.qvel,
                 frames_engine10s=f_10s.astype(np.int64), git_sha=np.asarray(result["git_sha"]),
                 legacy_path=np.asarray(legacy_path))
        logger.info("reference fixture -> %s (full %d, engine %d, engine_10s %d frames)", ref, f_full.shape[0],
                    f_eng.shape[0], f_10s.shape[0])


if __name__ == "__main__":
    main()
