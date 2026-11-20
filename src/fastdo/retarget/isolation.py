# Copyright 2026 The HandReel authors. SPDX-License-Identifier: Apache-2.0
# Portions adapted from DO AS I DO (https://github.com/malik-group/do-as-i-do, commit 824591b),
# Copyright (c) 2026 malik-group, MIT License; see licenses/DO-AS-I-DO-MIT.txt and THIRD_PARTY_LICENSES.md.
"""Task C4: isolation study on the baseline's physics stage (its `optimize_physics.py`).

Measures what the baseline's MuJoCo-Warp physics/MPC stage adds on top of a kinematic reference,
using OUR contact-transport reference (Task C3, `retarget_kinematic`) versus THEIR mink-IK
reference (the shipped `trajectory_kinematic.npz`), on the whisking demo. This module holds the
reusable pieces: arm-directory setup (mirroring the baseline's own `get_processed_data_dir` layout
so `optimize_physics.py` can be called directly, per the addendum, without going through
`launch.py`'s stages 1-4.5 again) and the kinematic-only clearance/penetration statistic (the same
computation `tests/retarget/test_kinematic.py` uses for our own reference, applied uniformly to
both references and to the physics stage's own output).

Neither this module nor `run/pipeline/retarget_whisking.py` ever writes into `third_party/`: arm
directories live under an `output_root` the caller supplies (`isolation.output_root`: the primary checkout's
`outputs/isolation/<arm>`, never a worktree). The baseline's own
`assets/` tree (convex hulls, robot meshes -- ~19 MB, shared identically by every task under the
same `output_root_dir`) is symlinked rather than copied: `scene.xml`'s `meshdir` is a path relative
to the compiled scene file, and the two arms otherwise share the exact same object/robot scene
(only the kinematic *trajectory* differs between arms), so duplicating that tree would waste disk
and time for zero benefit, and a symlink does not modify anything under `third_party/`.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from fastdo.retarget.contact import robot_contact_measure
from fastdo.retarget.labels import INDEX, MIDDLE, THUMB, robot_link_labels
from fastdo.retarget.mj_model import RobotScene
from fastdo.solver.sdf import SdfGrid, sdf_lookup

logger = logging.getLogger(__name__)

__all__ = [
    "ArmLayout",
    "GpuProbeResult",
    "KinematicStats",
    "gpu_dry_run_probe",
    "kinematic_reference_stats",
    "processed_data_dir",
    "setup_arm_directory",
]


def processed_data_dir(output_root: Path, robot_type: str, embodiment_type: str, task: str,
                      data_id: int) -> Path:
    """Mirrors the baseline's `retargeting.utils.io.get_processed_data_dir` layout exactly (same
    four path segments), so a directory built by this function is what `optimize_physics.py`'s own
    `process_config` will look for when given the matching `output_root_dir`/`task`/etc."""
    return output_root / robot_type / embodiment_type / task / str(data_id)


@dataclass(frozen=True)
class ArmLayout:
    """Resolved paths for one arm's copied scene directory (`processed_data_dir` plus its parent
    `task_info.json` and the shared `assets/` symlink)."""

    output_root: Path
    data_dir: Path          # processed_data_dir(...): holds scene.xml + trajectory_kinematic.npz
    task_info_json: Path    # data_dir.parent / "task_info.json"
    assets_link: Path       # output_root / "assets" -> the baseline's own outputs/assets


def setup_arm_directory(output_root: Path, demo_scene_xml: Path, demo_assets_dir: Path,
                        trajectory_kinematic_src: Path, task: str, robot_type: str,
                        embodiment_type: str, data_id: int,
                        demo_keypoints_npz: Path | None = None) -> ArmLayout:
    """Build one arm's copied scene directory under `output_root` (never under `third_party/`).

    Copies `scene.xml` and the reference `trajectory_kinematic.npz` (either the shipped one, for
    the "theirs" arm, or ours, for the "ours" arm) into `processed_data_dir(...)`; symlinks
    `output_root/assets` to the baseline's own `outputs/assets` (read-only reference, never
    modified); writes a minimal `task_info.json` one level up (the only two fields
    `optimize_physics.py` actually reads from it -- `ref_dt`, `contact_site_ids` -- are both
    absent here on purpose, since this study uses the baseline's own default `ref_dt` and
    `contact_rew_scale=0`, so it falls back to `{}` with a logged warning, which is expected).

    When `demo_keypoints_npz` is given, it is also copied to `{output_root}/mano/{embodiment_type}/
    {task}/{data_id}/trajectory_keypoints.npz` -- `retargeting.utils.mjwp._load_mano_keypoints_for_sim`
    hard-requires this file (under the baseline's own `robot_type="mano"` layout, regardless of
    which robot this arm retargets to) for its perturbation/in-hand gating, since the do_as_i_do
    override enables `pedestal_penalty_scale`/`perturb_*_scale`. It reflects the *human*
    demonstration's own in-hand timing (from HaWoR/the dataset processor), not either arm's
    kinematic solve, so the identical shipped file is correct for both arms.
    """
    output_root.mkdir(parents=True, exist_ok=True)
    data_dir = processed_data_dir(output_root, robot_type, embodiment_type, task, data_id)
    data_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(demo_scene_xml, data_dir / "scene.xml")
    shutil.copy2(trajectory_kinematic_src, data_dir / "trajectory_kinematic.npz")

    if demo_keypoints_npz is not None:
        mano_dir = processed_data_dir(output_root, "mano", embodiment_type, task, data_id)
        mano_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(demo_keypoints_npz, mano_dir / "trajectory_keypoints.npz")

    assets_link = output_root / "assets"
    if not assets_link.exists():
        assets_link.symlink_to(demo_assets_dir.resolve(), target_is_directory=True)

    task_info_json = data_dir.parent / "task_info.json"
    task_info_json.write_text(json.dumps({
        "task": task, "dataset_name": "do_as_i_do", "robot_type": robot_type,
        "embodiment_type": embodiment_type, "data_id": data_id,
    }, indent=2))
    return ArmLayout(output_root=output_root, data_dir=data_dir, task_info_json=task_info_json,
                     assets_link=assets_link)


@dataclass(frozen=True)
class KinematicStats:
    """Clearance/penetration summary for one `qpos (T,35)` trajectory against one object mesh,
    computed identically for the kinematic-only references and the physics-stage output (Task C3's
    own whisking-demo test uses exactly this computation for our reference; this dataclass just
    makes it reusable and comparable across all four trajectories the isolation study needs)."""

    n_frames: int
    n_contact_frames: int
    frac_under_2mm: float
    frac_any_penetration: float
    pen_mean_mm: float
    pen_p95_mm: float
    pen_max_mm: float
    thumb_median_mm: float
    index_median_mm: float
    middle_median_mm: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "n_frames": self.n_frames, "n_contact_frames": self.n_contact_frames,
            "frac_under_2mm": self.frac_under_2mm, "frac_any_penetration": self.frac_any_penetration,
            "pen_mean_mm": self.pen_mean_mm, "pen_p95_mm": self.pen_p95_mm, "pen_max_mm": self.pen_max_mm,
            "thumb_median_mm": self.thumb_median_mm, "index_median_mm": self.index_median_mm,
            "middle_median_mm": self.middle_median_mm,
        }


def kinematic_reference_stats(qpos: np.ndarray, scene: RobotScene, sdf: SdfGrid, surface: Tensor,
                              mesh_scale: float, n_per_link_pen: int, seed: int) -> KinematicStats:
    """Clearance (thumb/index/middle median min|phi| on contact frames) and penetration (<2mm frame
    fraction per Task C3's own convention, any-penetration frame fraction per the C4 addendum's
    literal wording, mean/p95/max of per-frame max penetration) for `qpos (T,35)`.

    Identical computation to `tests/retarget/test_kinematic.py::
    test_retarget_kinematic_on_whisking_demo_penetration_and_clearance`'s own measurement of our
    reference, applied here to any qpos trajectory sharing that scene's layout (the shipped
    reference, our reference, or a physics-stage output) so the four numbers the isolation study
    needs are directly comparable.
    """
    q = torch.as_tensor(qpos, dtype=torch.float64)
    local_pts, body_ids, _ = scene.surface_points(n_per_link_pen, seed=seed)
    robot_labels = robot_link_labels(scene.model, body_ids)
    measure = robot_contact_measure(q, scene, body_ids, local_pts, robot_labels, sdf, surface,
                                    mesh_scale, jacobian=False)
    assert not isinstance(measure, tuple)
    phi, _ = sdf_lookup(sdf, measure.raw_points)
    penetration_mm = torch.clamp(-phi, min=0.0) * mesh_scale * 1000.0
    pen_per_frame_max = penetration_mm.max(dim=1).values
    frac_under_2mm = (pen_per_frame_max < 2.0).float().mean().item()
    frac_any_penetration = (pen_per_frame_max > 0.0).float().mean().item()

    contact_frames = (measure.weights.sum(dim=1) > 0).nonzero().flatten()

    def _median_mm(label: int) -> float:
        sel = robot_labels == label
        if not sel.any() or contact_frames.numel() == 0:
            return float("nan")
        dist_mm = phi[:, sel].abs() * mesh_scale * 1000.0
        min_per_frame = dist_mm.min(dim=1).values
        return float(torch.median(min_per_frame[contact_frames]))

    return KinematicStats(
        n_frames=int(q.shape[0]), n_contact_frames=int(contact_frames.numel()),
        frac_under_2mm=frac_under_2mm, frac_any_penetration=frac_any_penetration,
        pen_mean_mm=float(pen_per_frame_max.mean()),
        pen_p95_mm=float(torch.quantile(pen_per_frame_max, 0.95)),
        pen_max_mm=float(pen_per_frame_max.max()),
        thumb_median_mm=_median_mm(THUMB), index_median_mm=_median_mm(INDEX),
        middle_median_mm=_median_mm(MIDDLE),
    )


@dataclass(frozen=True)
class GpuProbeResult:
    """Peak GPU memory (MiB) attributable to the dry-run process during the probe, plus the raw
    samples (timestamp, incremental MiB over the pre-launch baseline) for the report.

    `peak_mib` is `max(sample) - baseline`, not the raw card-wide total: this host is shared with
    another project's long training runs (tens of GB, independent of anything this study does), so
    a raw total would conflate their usage with ours and make the 11 GB budget meaningless.
    """

    peak_mib: float
    baseline_mib: float
    samples: tuple[tuple[float, float], ...]
    terminated: bool
    confirmed_dead: bool


def _nvidia_smi_used_mib() -> float:
    """Sum of `used_memory` (MiB) across every process `nvidia-smi --query-compute-apps` reports
    (not filtered to one pid: MuJoCo Warp/CUDA driver internals can spawn helper processes, and the
    addendum's own budget is stated as a peak over the whole dry run, not per-pid)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=10.0,
        )
    except (subprocess.SubprocessError, FileNotFoundError, OSError) as e:
        logger.warning("nvidia-smi query failed during GPU dry-run probe: %s", e)
        return 0.0
    total = 0.0
    for line in out.stdout.splitlines():
        line = line.strip()
        if line:
            with contextlib.suppress(ValueError):
                total += float(line)
    return total


def _pgid_alive(pgid: int) -> bool:
    """True if any process in process group `pgid` still exists."""
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, just not signalable (shouldn't happen for our own child)
    return True


def _wait_for_group_death(proc: subprocess.Popen[bytes], pgid: int, timeout: float) -> bool:
    """Poll until every process in `pgid` is gone, reaping our own direct child (`proc`) along the
    way via `proc.poll()` -- a killed-but-unreaped direct child is a zombie, and a zombie still
    answers `kill(pid, 0)`/`killpg` as "alive", so without this reap the group would never be
    observed as dead even once every process in it has actually exited. Any *other* member of the
    group (e.g. a grandchild the wrapper forked and never waited on) is reparented on the wrapper's
    death and reaped by the kernel's subreaper, which does not depend on us at all.
    """
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        proc.poll()
        if not _pgid_alive(pgid):
            return True
        time.sleep(0.5)
    return False


def _kill_process_group(proc: subprocess.Popen[bytes], pgid: int, confirm_timeout: float = 20.0) -> bool:
    """SIGTERM the whole process group, escalate to SIGKILL, and poll until every process in it is
    gone. Returns True once confirmed dead, False if it is somehow still alive after the timeout
    (logged, never silently assumed). `pgid` must be captured by the caller immediately after
    `Popen(..., start_new_session=True)` -- once `proc` may have exited, `os.getpgid(proc.pid)`
    can no longer be trusted to resolve the same group (the pid could already be reaped/recycled).

    Required because `cmd`'s outer wrapper (`nice ... conda run -n ... python driver.py`) does not
    reliably forward a signal sent only to its own PID down to the grandchild `optimize_physics.py`
    process `conda run` forks (observed directly: `proc.terminate()` on the wrapper alone left the
    real MuJoCo-Warp MPC process running a full, un-timeboxed optimization after the dry run
    "finished"). Every process this module launches must
    therefore start its own session (`start_new_session=True`) so the whole tree shares one process
    group, killable as a unit.
    """
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pgid, signal.SIGTERM)
    if _wait_for_group_death(proc, pgid, confirm_timeout):
        return True
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pgid, signal.SIGKILL)
    return _wait_for_group_death(proc, pgid, confirm_timeout)


def gpu_dry_run_probe(cmd: list[str], cwd: Path, env: dict[str, str], seconds: float,
                      poll_seconds: float, log_path: Path) -> GpuProbeResult:
    """Run `cmd` as a subprocess *in its own process group*, sample total GPU memory every
    `poll_seconds` for up to `seconds` wall-clock, then kill the whole group (SIGTERM, escalating
    to SIGKILL) and block until every process in it is confirmed gone before returning.

    Per the addendum: MuJoCo Warp's allocations are outside torch's per-process CUDA cap, so this
    is the only way to observe the baseline physics stage's real peak footprint before committing
    to a ~30-minute run. The subprocess is always killed at the end of this function -- the dry run
    is purely diagnostic, never counted as one of the two timed arms. `start_new_session=True` is
    required, not cosmetic: `cmd`'s wrapper chain (`nice` -> `conda run` -> the real Python
    process) forks rather than exec-replacing at the `conda run` step, so signalling only the
    wrapper's own PID leaves the real MPC process running untouched -- confirmed the hard way (see
    the module docstring reference above); killing the whole process group is the only reliable fix.
    """
    baseline = _nvidia_smi_used_mib()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        proc = subprocess.Popen(cmd, cwd=str(cwd), env=env, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        pgid = os.getpgid(proc.pid)
        samples: list[tuple[float, float]] = []
        t0 = time.monotonic()
        try:
            while time.monotonic() - t0 < seconds:
                if proc.poll() is not None:
                    logger.warning("dry-run process exited early (rc=%d) after %.1fs",
                                   proc.returncode, time.monotonic() - t0)
                    break
                used = _nvidia_smi_used_mib() - baseline
                samples.append((time.monotonic() - t0, used))
                time.sleep(poll_seconds)
        finally:
            still_running_before_kill = proc.poll() is None
            confirmed_dead = _kill_process_group(proc, pgid)
            if not confirmed_dead:
                logger.error("dry-run process group pgid=%d NOT confirmed dead after kill -- "
                            "manual cleanup may be required (ps -g %d)", pgid, pgid)
    peak = max((u for _, u in samples), default=0.0)
    logger.info("GPU dry-run peak (over a %.0f MiB pre-launch baseline): %.0f MiB across %d "
               "samples over %.1fs; process group %s", baseline, peak, len(samples),
               time.monotonic() - t0, "confirmed killed" if confirmed_dead else "NOT CONFIRMED DEAD")
    return GpuProbeResult(peak_mib=peak, baseline_mib=baseline, samples=tuple(samples),
                          terminated=still_running_before_kill, confirmed_dead=confirmed_dead)
