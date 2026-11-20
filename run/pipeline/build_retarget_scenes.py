"""Object-agnostic Sharpa scenes for the `retarget` stage, one per hand side, through DO AS I DO's own scene stages.

  FASTDO_ROOT=/path/to/checkout python run/pipeline/build_retarget_scenes.py [--out-root DIR] [--sides right left]

The retarget stage solves a KINEMATIC reference: it reads the robot's kinematics, collision primitives and surface
samples from the scene, and the object only through the scene's free joint (qpos 28:35), whose pose the stage writes
itself; the object's geometry comes from the clip's own `sdf.npz`/`samples.npz`. So one scene per side serves every
clip. The configured right scene (`isolation.demo_scene_xml`) is the baseline's whisking scene, whose object is the
whisk; the left side previously borrowed a scene built for a YCB can
(`outputs/policy_scenes/005_tomato_soup_can/.../left`).

This builds both sides with a neutral placeholder object (`PLACEHOLDER`, a 2 cm cube written under `--out-root`) via
`run/pipeline/build_ycb_scenes.build_object_scene` (the baseline's decompose_mesh, generate_scene, solve_ik and
resolve_pedestal in the retargeting conda env, third_party read-only, the `policy_scenes` parameters), then checks
that each built scene's HAND is kinematically identical to a reference scene of the same side (`--reference-right`,
`--reference-left`): every hand body pose, every `<side>_*` site and every capsule cover of the retarget stage at
`N_CHECK` random joint configurations, and the joint ranges, to `ATOL`. The record goes to
`<out-root>/retarget_scenes.json`.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import trimesh
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_ycb_scenes import SceneBuildConfig, build_object_scene  # noqa: E402

from fastdo.retarget.mj_model import RobotScene  # noqa: E402
from fastdo.retarget.primitives import hand_capsule_covers  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
PLACEHOLDER = "retarget_placeholder"
CUBE_EDGE_M = 0.02
N_CHECK = 64
ATOL = 1e-9


def placeholder_mesh(path: Path) -> Path:
    """A closed `CUBE_EDGE_M` cube centred on the origin (the placeholder object's mesh)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    trimesh.creation.box(extents=(CUBE_EDGE_M,) * 3).export(path)
    return path


def hand_kinematics(scene: RobotScene, q: np.ndarray) -> dict[str, np.ndarray]:
    """Hand-side body poses, `<side>_*` site positions and capsule-cover endpoints at `q (N,35)`, keyed by name."""
    model, data = scene.model, mujoco.MjData(scene.model)
    names = {b: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(model.nbody)}
    bodies = [b for b, n in names.items() if n and n.split("_")[0] in ("right", "left") and "object" not in n]
    sites = [s for s in range(model.nsite)
             if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, s) or "").split("_")[0] in ("right", "left")]
    covers = hand_capsule_covers(scene, 5e-4)
    out: dict[str, list[np.ndarray]] = {}
    for row in q:
        data.qpos[:] = row
        mujoco.mj_kinematics(model, data)
        for b in bodies:
            out.setdefault(f"body:{names[b]}", []).append(np.concatenate([data.xpos[b], data.xmat[b]]))
        for s in sites:
            out.setdefault(f"site:{mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, s)}", []).append(
                data.site_xpos[s].copy())
        ends = [data.xpos[b] + data.xmat[b].reshape(3, 3) @ p
                for b, p in zip(np.concatenate([covers.body_ids, covers.body_ids]),
                                np.concatenate([covers.local_a, covers.local_b]), strict=True)]
        out.setdefault("covers", []).append(np.concatenate([np.ravel(ends), covers.radius]))
    return {k: np.asarray(v) for k, v in out.items()}


def compare_hands(built: str, reference: str, seed: int = 0) -> dict[str, Any]:
    """Max abs difference of `hand_kinematics` between two scenes of one side at `N_CHECK` random in-range `q`."""
    a, b = RobotScene.from_xml(built), RobotScene.from_xml(reference)
    rng = np.random.default_rng(seed)
    dof = a.robot_dof
    lo, hi = a.model.jnt_range[:dof.stop - dof.start].T.copy()
    lo[:3], hi[:3] = -0.5, 0.5                                          # wrist slides: a working volume
    q = np.zeros((N_CHECK, a.model.nq))
    q[:, :dof.stop - dof.start] = rng.uniform(lo, hi, size=(N_CHECK, dof.stop - dof.start))
    q[:, 28:31] = rng.uniform(-0.3, 0.3, size=(N_CHECK, 3))
    q[:, 31] = 1.0
    ka, kb = hand_kinematics(a, q), hand_kinematics(b, q)
    missing = sorted(set(ka) ^ set(kb))
    diffs = {k: float(np.abs(ka[k] - kb[k]).max()) for k in sorted(set(ka) & set(kb)) if ka[k].shape == kb[k].shape}
    shape_mismatch = sorted(k for k in set(ka) & set(kb) if ka[k].shape != kb[k].shape)
    ranges = float(np.abs(a.model.jnt_range[:28] - b.model.jnt_range[:28]).max())
    worst = max(diffs.values()) if diffs else float("inf")
    return {"built": built, "reference": reference, "n_q": N_CHECK, "max_abs_diff": worst, "joint_range_diff": ranges,
            "missing_keys": missing, "shape_mismatch": shape_mismatch, "n_keys": len(diffs),
            "identical": not missing and not shape_mismatch and worst <= ATOL and ranges <= ATOL}


def main() -> None:
    root = Path(os.environ.get("FASTDO_ROOT") or REPO)
    cfg = OmegaConf.load(REPO / "run/conf/config.yaml")
    cfg.paths.root = str(root)
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=str(root / "outputs/retarget_scenes"))
    ap.add_argument("--sides", nargs="+", default=["right", "left"])
    ap.add_argument("--reference-right", default=str(OmegaConf.select(cfg, "isolation.demo_scene_xml")))
    ap.add_argument("--reference-left", default=str(root / "outputs/policy_scenes/005_tomato_soup_can/sharpa/left/"
                                                    "005_tomato_soup_can/0/scene.xml"))
    args = ap.parse_args()
    from fastdo import hostpaths

    try:
        hostpaths.check_config_paths(cfg, [
            ("policy scenes", "policy_scenes.baseline_root", "FASTDO_PRIMARY_ROOT (or FASTDO_EXT_ROOT)", False),
            ("policy scenes", "policy_scenes.python", "FASTDO_CONDA_BASE", False)])
        if args.reference_right:
            hostpaths.require(args.reference_right, "FASTDO_PRIMARY_ROOT (or --reference-right)")
    except hostpaths.HostPathNotSet as e:
        ap.error(str(e))
    node = OmegaConf.to_container(cfg.policy_scenes, resolve=True)
    assert isinstance(node, dict)
    build_cfg = SceneBuildConfig.from_mapping({**node, "out_root": args.out_root})
    mesh = placeholder_mesh(Path(args.out_root) / "placeholder_mesh" / "cube_2cm.obj")
    result: dict[str, Any] = {"placeholder": {"object": PLACEHOLDER, "mesh": str(mesh), "cube_edge_m": CUBE_EDGE_M},
                              "built_utc": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
                              "sides": {}}
    for side in args.sides:
        record = build_object_scene(PLACEHOLDER, mesh, side, build_cfg)
        entry: dict[str, Any] = {"scene_xml": record.get("scene_xml"), "pass": bool(record.get("pass")),
                                 "validation": record.get("validation")}
        reference = args.reference_right if side == "right" else args.reference_left
        if record.get("pass") and reference and Path(reference).is_file():
            entry["hand_equivalence"] = compare_hands(record["scene_xml"], reference)
        result["sides"][side] = entry
        print(side, json.dumps(entry, default=str)[:600], flush=True)
    out = Path(args.out_root) / "retarget_scenes.json"
    out.write_text(json.dumps(result, indent=1, default=str))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
