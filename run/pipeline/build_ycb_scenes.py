"""One Sharpa MuJoCo scene per DexYCB YCB object, through the baseline's own scene stages.

  FASTDO_ROOT=/path/to/checkout uv run python run/pipeline/build_ycb_scenes.py 'policy_scenes.objects=[061_foam_brick]'

For every object (`policy_scenes.objects`, default all of `YCB_OBJECTS`) and hand side (`policy_scenes.sides`):
1. `run/pipeline/_ycb_scene_driver.py` runs in the retargeting conda env (`policy_scenes.python`; CPU, CUDA hidden) on
   `<dexycb_root>/models/<object>/textured_simple.obj`, into `<out_root>/<object>/`. It runs the baseline's
   decompose_mesh (CoACD, parameters recorded with the part count), generate_scene, solve_ik and resolve_pedestal, all
   imported from third_party at runtime.
2. `validate_scene` loads `scene.xml` here with `RobotScene.from_xml` and checks nq = 35, nv = 34, one free joint (the
   object) and every object collision geom in a contact pair with a hand-link geom, at least one.
3. The record (driver output, validation, `pass`) goes to `<out_root>/<object>/scene_<side>.json`
   (`fastdo.policy.dexycb_inputs`), and `<out_root>/scenes.json` summarizes the run.

A stored passing record is reused unless `policy_scenes.force`. A failed driver run is recorded with `pass` False and
its log path.

The driver runs under a one-CPU affinity (`taskset -c <cpu>`). CoACD's statically linked oneTBB sizes its worker
pool from the affinity mask, not from OMP_NUM_THREADS, TBB environment variables or the cgroup quota (160 CPUs, 161
threads on a 17-core quota); the idle pool added a third to the pooled scene-build core-seconds and nearly doubled the
median, and pinned the hulls are byte-identical. `policy_scenes.cpu` (config.yaml, default null) picks the
CPU; null takes `sorted(mask)[pid % len(mask)]` of this process's mask, so concurrent builder processes spread over the
mask rather than all sharing its lowest CPU (hulls do not depend on the CPU). `policy_scenes.pin: false` restores the
inherited mask. The chosen CPU is logged and stored in the record (`driver_cpu`), and the driver records the mask it
ran under (`affinity`). Builds within one process run one after another, so one CPU is enough; parallel callers that
must not share a core should give each build its own `cpu`. With `pin` true, a missing `taskset` is refused when the
config is built, before any log is opened.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import shutil
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import hydra
import mujoco
from omegaconf import DictConfig

from fastdo.io.datasets.dexycb import YCB_OBJECTS
from fastdo.policy.dexycb_inputs import SIDES, read_scene_record, write_scene_record
from fastdo.retarget.mj_model import RobotScene

logger = logging.getLogger(__name__)

__all__ = ["CoacdParams", "SceneBuildConfig", "build_object_scene", "build_scenes", "validate_scene"]

DRIVER = Path(__file__).resolve().parent / "_ycb_scene_driver.py"
LAYOUT = (35, 34)


@dataclass(frozen=True)
class CoacdParams:
    """launch.py's stage-2 call (thicken, dilate) on decompose_mesh's defaults; `preprocess_resolution` is fixed in the
    baseline and only recorded."""

    threshold: float = 0.05
    max_convex_hull: int = 32
    max_ch_vertex: int = 256
    preprocess_resolution: int = 100
    thicken: float = 0.002
    dilate: float = 0.002


@dataclass(frozen=True)
class SceneBuildConfig:
    """The Hydra block `policy_scenes` (module docstring)."""

    out_root: str
    baseline_root: str
    python: str
    coacd: CoacdParams = field(default_factory=CoacdParams)
    friction_scale: float = 1.5
    add_ur3_arm: bool = True
    force: bool = False
    timeout_s: float = 7200.0
    pin: bool = True
    cpu: int | None = None

    def __post_init__(self) -> None:
        if self.pin and shutil.which("taskset") is None:
            raise FileNotFoundError("taskset (util-linux) is needed to pin the scene driver to one CPU; install it or "
                                    "set policy_scenes.pin=false")

    @classmethod
    def from_mapping(cls, node: Mapping[str, Any]) -> SceneBuildConfig:
        return cls(out_root=str(node["out_root"]), baseline_root=str(node["baseline_root"]), python=str(node["python"]),
                   coacd=CoacdParams(**dict(node.get("coacd") or {})), friction_scale=float(node["friction_scale"]),
                   add_ur3_arm=bool(node["add_ur3_arm"]), force=bool(node["force"]), timeout_s=float(node["timeout_s"]),
                   pin=bool(node.get("pin", True)), cpu=None if node.get("cpu") is None else int(node["cpu"]))


def validate_scene(scene_xml: str | Path) -> dict[str, Any]:
    """The layout and contact checks of the module docstring: `pass`, `reasons`, `nq`, `nv`, `n_free_joints`,
    `n_object_collision_geoms` and `n_object_contact_geoms`. A model without exactly one free joint fails with that
    reason (no object body to check)."""
    try:
        scene = RobotScene.from_xml(scene_xml)
    except ValueError as exc:
        return {"pass": False, "reasons": [f"RobotScene.from_xml: {exc}"], "nq": None, "nv": None, "n_free_joints": None,
                "n_object_collision_geoms": 0, "n_object_contact_geoms": 0}
    model = scene.model
    free = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    counts = {"nq": int(model.nq), "nv": int(model.nv), "n_free_joints": len(free)}
    reasons = []
    if (model.nq, model.nv) != LAYOUT:
        reasons.append(f"nq, nv = {model.nq}, {model.nv}, expected {LAYOUT}")
    if len(free) != 1:
        reasons.append(f"{len(free)} free joints: a scene needs exactly one free joint (the object)")
        return {"pass": False, "reasons": reasons, **counts, "n_object_collision_geoms": 0, "n_object_contact_geoms": 0}
    object_body = int(model.jnt_bodyid[free[0]])
    object_geoms = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == object_body and model.geom_group[g] == 3]
    hand = set(scene.hand_collision_geoms())
    paired = {int(a) if int(b) in hand else int(b) for a, b in zip(model.pair_geom1, model.pair_geom2, strict=True)
              if (int(a) in hand) != (int(b) in hand)}
    contact = [g for g in object_geoms if g in paired]
    if not contact:
        reasons.append("no object contact geoms: no object collision geom is in a contact pair with the hand")
    elif len(contact) != len(object_geoms):
        reasons.append(f"{len(object_geoms) - len(contact)} object collision geoms have no contact pair with the hand")
    return {"pass": not reasons, "reasons": reasons, **counts, "n_object_collision_geoms": len(object_geoms),
            "n_object_contact_geoms": len(contact)}


def _driver_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "VIRTUAL_ENV")}
    threads = os.environ.get("OMP_NUM_THREADS", "2")
    env.update(CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads,
               OPENBLAS_NUM_THREADS=threads)
    return env


def _driver_cpu(cfg: SceneBuildConfig) -> int | None:
    """The CPU the driver is pinned to (module docstring), or None when `cfg.pin` is False.

    Raises:
        ValueError: If `cfg.cpu` is not in this process's affinity mask.
    """
    if not cfg.pin:
        return None
    allowed = sorted(os.sched_getaffinity(0))
    cpu = allowed[os.getpid() % len(allowed)] if cfg.cpu is None else cfg.cpu
    if cpu not in allowed:
        raise ValueError(f"cpu {cpu} is not in this process's affinity mask")
    return cpu


def _pinned(cmd: list[str], cpu: int | None) -> list[str]:
    """`cmd` under `taskset -c <cpu>`, or unchanged when `cpu` is None.

    Raises:
        FileNotFoundError: If `taskset` is not on PATH.
    """
    if cpu is None:
        return cmd
    taskset = shutil.which("taskset")
    if taskset is None:
        raise FileNotFoundError("taskset (util-linux) is needed to pin the scene driver to one CPU")
    return [taskset, "-c", str(cpu), *cmd]


def build_object_scene(name: str, mesh_path: str | Path, side: str, cfg: SceneBuildConfig) -> dict[str, Any]:
    """Build (or reuse) and validate the scene of `name` held by the `side` hand; returns the stored record.

    Raises:
        ValueError: On an unknown side, or a `cfg.cpu` outside this process's affinity mask.
        subprocess.TimeoutExpired: If the driver exceeds `cfg.timeout_s`.
    """
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}, got {side!r}")
    stored = read_scene_record(cfg.out_root, name, side)
    if stored is not None and stored.get("pass") and not cfg.force:
        logger.info("%s (%s hand): reusing the stored scene %s", name, side, stored["scene_xml"])
        return stored
    object_root = Path(cfg.out_root).resolve() / name
    object_root.mkdir(parents=True, exist_ok=True)
    out_json, log_path = object_root / f"build_{side}.json", object_root / f"build_{side}.log"
    out_json.unlink(missing_ok=True)
    c = cfg.coacd
    cmd = [cfg.python, str(DRIVER), "--baseline-root", cfg.baseline_root, "--output-root", str(object_root),
           "--object", name, "--mesh", str(Path(mesh_path).resolve()), "--side", side, "--threshold", str(c.threshold),
           "--max-convex-hull", str(c.max_convex_hull), "--max-ch-vertex", str(c.max_ch_vertex),
           "--preprocess-resolution", str(c.preprocess_resolution), "--thicken", str(c.thicken), "--dilate",
           str(c.dilate), "--friction-scale", str(cfg.friction_scale), "--add-ur3-arm", str(int(cfg.add_ur3_arm)),
           "--out-json", str(out_json)]
    cpu = _driver_cpu(cfg)
    logger.info("%s (%s hand): scene driver on CPU %s", name, side, "inherited mask" if cpu is None else cpu)
    start = time.perf_counter()
    with log_path.open("w") as log:
        done = subprocess.run(_pinned(cmd, cpu), stdout=log, stderr=subprocess.STDOUT, env=_driver_env(),
                              cwd=str(object_root), timeout=cfg.timeout_s, check=False)
    wall = time.perf_counter() - start
    if done.returncode != 0 or not out_json.is_file():
        record: dict[str, Any] = {"object": name, "side": side, "scene_xml": "", "pass": False, "log": str(log_path),
                                  "driver_wall_seconds": wall, "driver_cpu": cpu,
                                  "error": f"scene driver exited {done.returncode} after {wall:.1f} s; see the log"}
        write_scene_record(cfg.out_root, record)
        logger.error("%s (%s hand): %s (%s)", name, side, record["error"], log_path)
        return record
    record = json.loads(out_json.read_text())
    validation = validate_scene(record["scene_xml"])
    record.update({"validation": validation, "pass": validation["pass"], "driver_wall_seconds": wall,
                   "driver_cpu": cpu, "log": str(log_path),
                   "validated_utc": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")})
    write_scene_record(cfg.out_root, record)
    logger.info("%s (%s hand): %d CoACD parts, pass %s %s, %.1f s", name, side, record["n_parts"], record["pass"],
                validation["reasons"], wall)
    return record


def build_scenes(objects: Sequence[str], sides: Sequence[str], dexycb_root: str | Path,
                 cfg: SceneBuildConfig) -> dict[str, Any]:
    """`build_object_scene` for every object and side; writes and returns the `<out_root>/scenes.json` summary."""
    rows: dict[str, dict[str, Any]] = {}
    for name in objects:
        for side in sides:
            record = build_object_scene(name, Path(dexycb_root) / "models" / name / "textured_simple.obj", side, cfg)
            rows[f"{name}/{side}"] = {"pass": bool(record["pass"]), "n_parts": record.get("n_parts"),
                                      "stage_seconds": record.get("stage_seconds"), "scene_xml": record["scene_xml"],
                                      "reasons": (record.get("validation") or {}).get("reasons") or record.get("error")}
    summary = {"scenes": rows, "n": len(rows), "n_pass": sum(r["pass"] for r in rows.values()),
               "failed": sorted(k for k, r in rows.items() if not r["pass"]), "coacd": vars(cfg.coacd),
               "updated_utc": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")}
    path = Path(cfg.out_root) / "scenes.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=1))
    return summary


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    from fastdo import hostpaths

    hostpaths.check_config_paths(cfg, [
        ("policy scenes", "policy_scenes.baseline_root", "FASTDO_PRIMARY_ROOT (or isolation.baseline_root=<dir>)",
         False),
        ("policy scenes", "policy_scenes.python", "FASTDO_CONDA_BASE (or policy_scenes.python=<interpreter>)", False)])
    node = cfg.policy_scenes
    objects = [str(o) for o in node.objects] or list(YCB_OBJECTS)
    summary = build_scenes(objects, [str(s) for s in node.sides], str(node.dexycb_root),
                           SceneBuildConfig.from_mapping(node))
    logger.info("scenes: %d of %d pass; failed %s", summary["n_pass"], summary["n"], summary["failed"])


if __name__ == "__main__":
    main()
