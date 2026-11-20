"""Export a G1 + Dex3-1 replay package from one reconstructed clip (CPU only).

    python run/robot/export_g1_package.py <engine clip dir> --out <package dir> [--ego-motion ego_motion.npz]
        [--first-row 0] [--no-trim] [--workers 6] [--table-z-range 0.70 1.00]
        [--object-xy 0.35 -0.20 --table-z 0.80 --yaw-offset-deg 0 --sym-axis 2 --sym-deg 90 --up-face 0]
        [--camera-pitch-deg 45]
        [--object-mass 0.2] [--arm-vmax 1.0] [--hand-vmax 2.0] [--home-q <17 joints>]
        [--start-pose start_pose_a1.json] [--support-plane support_plane.json] [--speed-scale 0.5]
        [--approach palm] [--grasp-closure [--closure-target-m 0.002]]

`--approach palm` / `--grasp-closure`: the synthesised approach
before the first contact (`fastdo.robot.approach`; no approach that passes its collision check exits 2) and the
flexion-bias grasp closure over the contact phase (`fastdo.robot.grasp_closure`, at the frozen target depth unless
`--closure-target-m`, used only to tune it on development clips). The simulated object mass is `--object-mass`
(0.2 kg: ASSUMED until the object is weighed; the printed JSON and package.json say so).

The clip directory is an engine output directory with `mesh.obj` and either `clip_record.npz` or the solve and hand
artifacts the record is rebuilt from (`fastdo.robot.world_frame`); `--ego-motion` replaces its static camera by that
ego-motion. The script runs the kinematic path (`fastdo.robot.kinematic_path`: DexPilot-style hand retargeting,
collision-aware wrist IK, 250 / 100 Hz export, MuJoCo replay on the official Unitree model) and writes the package
(`fastdo.robot.package`). Without `--object-xy` the placement (object start, task yaw, table height in
`--table-z-range`, the assumed head pitch or resting face when gravity was never estimated, a grasp rotation about a
symmetry axis) is
searched for the largest reach margin over the whole trajectory (`fastdo.robot.reach`, `--workers` processes); with it
the placement is explicit (`--table-z`, `--yaw-offset-deg`, `--sym-axis/--sym-deg`). Exit code 0 when the
package was written (whatever the simulated success; the printed JSON and the README list the export's warnings, e.g.
a reconstruction whose gravity was never estimated, and the run script's execution gate: `execute_allowed` false =
`--execute` will refuse the package), 2 on bad input. `--camera-pitch-deg` replaces the record's up
direction by a level camera pitched down by that angle: an operator assumption, stated as such in the package.
`--home-q` adds a checked home (right arm 7, left arm 7, waist 3, rad: the values the run script prints when it refuses
a robot that is not at a checked home) to the official zero posture and the arms-hanging home
(`fastdo.robot.transfer`); the moves from it are planned and simulated like theirs. Its waist must be within
`g1_sdk_replay.WAIST_TOL` (0.01 rad) of 0 (the simulation welds the waist at 0 and the run script
refuses a measured waist farther from it), else exit 2.
`--speed-scale` (default 1): the run script's playback speed scale the run is SIMULATED at and
recorded in `sim_stats.json` (`transfers`); its execution gate accepts only that value. A motion's first lab package is
exported at 0.5; export a 1.0 package only after a clean 0.5 run on the robot.
`--start-pose` / `--support-plane` (`fastdo.robot.start_pose`): the window's start pose (`start_pose_a1.json`;
consolidated, or snapped onto the accepted support plane) replaces the contact-onset row's object pose, and an
ACCEPTED `support_plane.json` (its own acceptance AND >= 2 rest rows with the normals within 3 deg) gives
up (`gravity_cam`): the head pitch / resting face is then not searched. A sidecar of another window, world frame or
plane, a refused start-pose record, a record whose ruled rest verdict is not True or whose start pose is more than
8.6 cm / 12.2 deg from the onset pose, or an accepted plane without a start pose exits 2
(`start_pose.SidecarError`, the only error turned into exit 2). Without them nothing changes.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""          # CPU only, before torch is imported

import argparse
import json
import logging
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from fastdo.bench.gpu_budget import apply_cpu_budget
from fastdo.robot import g1_constants as C
from fastdo.robot import g1_sdk_replay as S
from fastdo.robot.approach import ApproachError
from fastdo.robot.grasp_closure import ClosureConfig
from fastdo.robot.kinematic_path import (
    KinematicPathConfig,
    load_reconstruction,
    run_kinematic_path,
)
from fastdo.robot.package import write_package
from fastdo.robot.reach import ReachSearchConfig
from fastdo.robot.reach_eval import make_pool
from fastdo.robot.replay import SceneConfig
from fastdo.robot.start_pose import SidecarError, load_start_pose, load_support_plane
from fastdo.robot.trajectory import TrajectoryLimits
from fastdo.robot.transfer import TransferConfig

logger = logging.getLogger("export_g1_package")


def build_config(args: argparse.Namespace) -> KinematicPathConfig:
    """The kinematic-path config of the CLI arguments (velocity caps are clipped to the URDF limits)."""
    limits = TrajectoryLimits(arm_vmax=tuple(min(args.arm_vmax, v) for v in C.ARM_URDF_VELOCITY),
                              hand_vmax=tuple(min(args.hand_vmax, v) for v in C.HAND_URDF_VELOCITY))
    xy = None if args.object_xy is None else (float(args.object_xy[0]), float(args.object_xy[1]))
    search = replace(ReachSearchConfig(), table_z_range=(float(args.table_z_range[0]), float(args.table_z_range[1])))
    transfer = replace(TransferConfig(), speed_scale=float(args.speed_scale))
    if args.home_q is not None:
        transfer = replace(transfer, homes=transfer.homes + (("measured", tuple(float(v) for v in args.home_q)),))
    start = None if args.start_pose is None else load_start_pose(args.start_pose)
    plane = None if args.support_plane is None else load_support_plane(args.support_plane)
    return KinematicPathConfig(table_z=args.table_z, table_xy=xy, yaw_offset=float(np.radians(args.yaw_offset_deg)),
                               sym_axis=args.sym_axis, sym_deg=float(args.sym_deg), up_face=args.up_face,
                               first_row=args.first_row,
                               trim_to_contact=not args.no_trim, camera_pitch_deg=args.camera_pitch_deg,
                               limits=limits, search=search, transfer=transfer,
                               scene=SceneConfig(table_z=args.table_z, object_mass=args.object_mass),
                               start_pose=start, support_plane=plane, approach_mode=args.approach,
                               grasp_closure=bool(args.grasp_closure),
                               closure=ClosureConfig() if args.closure_target_m is None
                               else ClosureConfig(target_m=float(args.closure_target_m)))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("clip_dir", type=Path)
    ap.add_argument("--out", type=Path, required=True, help="package folder (must not exist or be empty)")
    ap.add_argument("--ego-motion", type=Path, default=None, help="ego_motion.npz of the clip's rows")
    ap.add_argument("--ego-max-gap-rows", type=int, default=0,
                    help="fill interior ego-motion gaps of <= this many rows (0 = off, refuse)")
    ap.add_argument("--first-row", type=int, default=0, help="rows before it are never used (misregistered)")
    ap.add_argument("--no-trim", action="store_true", help="keep every row, not just the contact span +- 1 s")
    ap.add_argument("--workers", type=int, default=6, help="placement-search IK processes")
    ap.add_argument("--table-z-range", type=float, nargs=2, default=(0.70, 1.00), help="searched table heights (m)")
    ap.add_argument("--sym-axis", type=int, default=None, help="explicit grasp rotation: principal axis 0/1/2")
    ap.add_argument("--sym-deg", type=float, default=0.0, help="explicit grasp rotation angle (deg)")
    ap.add_argument("--up-face", type=int, default=None, help="explicit up: index of a resting-face candidate")
    ap.add_argument("--table-z", type=float, default=0.80, help="table top height with --object-xy (m)")
    ap.add_argument("--object-xy", type=float, nargs=2, default=None, help="object start x y in the robot frame (m)")
    ap.add_argument("--yaw-offset-deg", type=float, default=0.0)
    ap.add_argument("--camera-pitch-deg", type=float, default=None,
                    help="assumed downward head pitch (deg) replacing the record's up direction")
    ap.add_argument("--object-mass", type=float, default=0.2, help="object mass in the sim (kg)")
    ap.add_argument("--arm-vmax", type=float, default=1.0, help="arm joint speed cap (rad/s)")
    ap.add_argument("--hand-vmax", type=float, default=2.0, help="hand joint speed cap (rad/s)")
    ap.add_argument("--home-q", type=float, nargs=17, default=None,
                    help="an extra checked home: right arm 7, left arm 7, waist 3 (rad), e.g. the measured pose")
    ap.add_argument("--start-pose", type=Path, default=None, help="the window's start_pose_a1.json sidecar")
    ap.add_argument("--support-plane", type=Path, default=None, help="the window's support_plane.json (gravity)")
    ap.add_argument("--approach", choices=("vertical", "palm"), default="vertical",
                    help="palm: the synthesised palm approach (fastdo.robot.approach)")
    ap.add_argument("--grasp-closure", action="store_true", help="close the grasp (fastdo.robot."
                                                                 "grasp_closure)")
    ap.add_argument("--closure-target-m", type=float, default=None,
                    help="fingertip penetration target (default: the frozen CLOSURE_TARGET_M)")
    ap.add_argument("--speed-scale", type=float, default=1.0,
                    help="playback speed scale the run is simulated at and the gate accepts (a first package: 0.5)")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    apply_cpu_budget()
    need = ["mesh.obj"] + ([] if (args.clip_dir / "clip_record.npz").is_file() else
                           ["clip_record.npz or solve.npz", "hands.npz", "intrinsics.npz"])
    for name in need:
        if not (args.clip_dir / name.split(" or ")[-1]).is_file():
            logger.error("%s has no %s", args.clip_dir, name)
            return 2
    if args.arm_vmax <= 0 or args.hand_vmax <= 0 or args.object_mass <= 0:
        logger.error("speed caps and mass must be positive")
        return 2
    if args.home_q is not None and not np.isfinite(args.home_q).all():
        logger.error("--home-q must be 17 finite joint angles")
        return 2
    if args.home_q is not None and np.abs(np.asarray(args.home_q[14:17])).max() > S.WAIST_TOL:
        logger.error("--home-q waist %s is more than %g rad from 0: the simulation welds the waist at 0 and the run "
                     "script refuses such a measured waist; zero the waist and measure again",
                     list(args.home_q[14:17]), S.WAIST_TOL)
        return 2
    if not 0.0 < args.speed_scale <= 1.0:
        logger.error("--speed-scale must be in (0, 1]")
        return 2
    if args.camera_pitch_deg is not None and not -90.0 < args.camera_pitch_deg < 90.0:
        logger.error("--camera-pitch-deg must be in (-90, 90)")
        return 2
    t0 = time.perf_counter()
    lo, hi = args.table_z_range
    if not 0.5 <= lo <= hi <= 1.2 or args.first_row < 0:
        logger.error("--table-z-range must satisfy 0.5 <= lo <= hi <= 1.2 and --first-row >= 0")
        return 2
    recon = load_reconstruction(args.clip_dir, args.ego_motion, args.ego_max_gap_rows)
    try:
        cfg = build_config(args)
    except SidecarError as exc:
        logger.error("%s", exc)
        return 2
    pool = make_pool(args.workers) if args.object_xy is None else None
    try:
        result = run_kinematic_path(recon, cfg, pool=pool)
    except SidecarError as exc:                       # a sidecar check (`start_pose`, `kinematic_task`) refused
        logger.error("a sidecar check refused: %s", exc)
        return 2
    except ApproachError as exc:                      # no approach passes the collision check
        logger.error("the synthesised approach refused: %s", exc)
        return 2
    finally:
        if pool is not None:
            pool.shutdown()
    out = write_package(args.out, result, recon, limits=cfg.limits, object_mass=args.object_mass)
    summary = result.summary()
    logger.info("package %s written in %.1f s", out, time.perf_counter() - t0)
    meta = json.loads((out / "package.json").read_text())
    print(json.dumps({"package": str(out), "success": summary["replay"]["success"],
                      "failed": summary["replay"]["failed_criteria"], "warnings": summary["warnings"],
                      "adaptations": summary["adaptations"], "reach_margin": summary["reach_margin"],
                      "execute_allowed": meta["sim_gate"]["execute_allowed"],
                      "object_mass_kg": args.object_mass, "object_mass_note": meta["object_mass_note"],
                      "approach": summary["approach"], "grasp_closure": summary["grasp_closure"],
                      "gate_failures": meta["sim_gate"]["failures"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
