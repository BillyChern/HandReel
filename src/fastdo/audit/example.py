# Copyright 2026 The HandReel authors. SPDX-License-Identifier: Apache-2.0
"""Worked example on bundled synthetic data (CPU, MuJoCo, a few seconds).

    python -m fastdo.audit.example [OUTDIR]

writes two toy scenes and three certified runs to OUTDIR (default: a fresh temporary directory), runs the audit CLI
on them under DO AS I DO's certificate, and prints the report's headline. The scene is `fastdo.audit.toy`: a cube,
a start pedestal, and a hand plate on a vertical servo.

- `pressed_still`: the reference object never moves; the plate presses it into the pedestal (servo 5 cm below the
  cube's top). The certificate passes the motionless object (the clip is Z0), passes the run's controls with the
  hand-object contact removed, and the run grips with about 7.5 times the cube's weight. The certificate has
  learnt nothing about the hand on this clip.
- `carried`: no pedestal under the cube; the plate receives a 0.3 m lift command. The motionless object fails, and without
  hand-object contact the cube falls: both nulls fail, the clip is informative, and the grip is about the weight
  (0.95x in this replay).
- `kinematic_only`: the `carried` reference with no scene: only the motionless-object null can run.

Expected report: Z0 base rate 1/3; the contact-removed null passes on 1 of 2 runs, the Z0 one.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from . import toy

__all__ = ["N_ROWS", "WARMUP_ROWS", "main", "run_example", "write_example"]

N_ROWS, WARMUP_ROWS = 160, 20
PRESS_DEPTH, LIFT_M = 0.05, 0.3


def _carried(n: int, w: int) -> dict[str, Any]:
    """Execute a 0.3 m lift command on the carry scene; the executed object rows become the reference (a perfect run)."""
    from nullaudit.mujoco_adapter import MujocoExecutor

    m = toy.carry_model()
    st = toy.carry(m)
    u = np.full((n, 1), st.ctrl[0])
    u[w:, 0] += np.linspace(0.0, LIFT_M, n - w)
    ex = MujocoExecutor(model=m, q0=st.qpos, hand_geom=toy.masks(m)[0], ctrl0=st.ctrl, warmup_rows=w)
    roll = ex(u)
    return {"reference": np.asarray(roll.object_traj, np.float64), "controls": u, "q0": st.qpos, "ctrl0": st.ctrl,
            "qpos": ex.last_qpos, "qvel": ex.last_qvel, "ctrl": ex.last_ctrl}      # executed rows: replay must match


def write_example(outdir: str | Path) -> list[Path]:
    """Write `scene_pedestal.xml`, `scene_carry.xml` and the three run files; return the run paths."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "scene_pedestal.xml").write_text(toy.XML)
    (out / "scene_carry.xml").write_text(toy.CARRY_XML)
    n, w = N_ROWS, WARMUP_ROWS
    base = {"warmup_rows": np.int64(w), "required_rows": np.int64(n)}
    m = toy.model()
    executor = {"weld_eq_ids": np.zeros(0, np.int64), "gravity": np.array(m.opt.gravity), "ctrl_steps": np.int64(1),
                "terminate_z": np.float64(-np.inf), "sim_dt": np.float64(m.opt.timestep),
                "hand_geom": toy.masks(m)[0]}

    st = toy.press(toy.model(), depth=PRESS_DEPTH)
    u_still = np.full((n, 1), st.ctrl[0])
    from nullaudit.mujoco_adapter import MujocoExecutor

    ex = MujocoExecutor(model=m, q0=st.qpos, hand_geom=toy.masks(m)[0], ctrl0=st.ctrl, warmup_rows=w)
    ex(u_still)                                    # the executed rows the positive control compares the replay to
    still = {**base, "reference": np.repeat(toy.object_rows(st.qpos[None]), n, 0),
             "scene": np.str_("scene_pedestal.xml"), "controls": u_still, "q0": st.qpos, "ctrl0": st.ctrl,
             "qpos": ex.last_qpos, "qvel": ex.last_qvel, "ctrl": ex.last_ctrl, **executor}
    carried = {**base, **_carried(n, w), "scene": np.str_("scene_carry.xml"), **executor}
    kin = {**base, "reference": carried["reference"]}
    paths = []
    for name, arrays in (("pressed_still", still), ("carried", carried), ("kinematic_only", kin)):
        p = out / f"{name}.npz"
        np.savez(p, **arrays)
        paths.append(p)
    return paths


def run_example(outdir: str | Path, certificate: str = "CD") -> dict[str, Any]:
    """Write the example, audit it with the CLI and return the JSON report."""
    from .__main__ import main as cli

    runs = write_example(outdir)
    report = Path(outdir) / "report.json"
    rc = cli([str(p) for p in runs] + ["--certificate", certificate, "--json", str(report)])
    if rc != 0:
        raise RuntimeError(f"audit CLI returned {rc}: the example should have no problems")
    return json.loads(report.read_text())


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if any(a in ("-h", "--help") for a in args):
        print("usage: python -m fastdo.audit.example [OUTDIR]\n\n"
              "Write the audit example into OUTDIR (a new temporary directory if omitted), "
              "audit it and print the report.")
        return 0
    if len(args) > 1 or (args and args[0].startswith("-")):
        print("usage: python -m fastdo.audit.example [OUTDIR]", file=sys.stderr)
        return 2
    outdir = Path(args[0]) if args else Path(tempfile.mkdtemp(prefix="fastdo_audit_example_"))
    doc = run_example(outdir)
    for r in doc["runs"]:
        g = r["grip_census"].get("grip_over_weight_p50")
        print(f"  {r['clip']:<15} Z0 {r['z0']!s:<5} contact-removed {r['contact_removed_null']['status']:<8} "
              f"grip/weight {'-' if g is None else f'{g:.2f}'}")
    print(f"Report: {outdir / 'report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
