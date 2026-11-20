"""M9 (P1) toy inputs WITHOUT pytest or libigl, so DO AS I DO's `retargeting` env (mujoco_warp 0.0.1) can run
them: the two toy clips as in `m5_toy.inputs`, the box's SDF filled analytically on the same grid layout as
`GridSDF.from_mesh`, and a CLI that runs the WARP backend on warp's CPU device against the CPU env.

  CUDA_VISIBLE_DEVICES= PYTHONPATH=<tree>/src:<tree> <retargeting python> -m tests.amortised.m9_toy compare OUT.json
  ...                                                                       -m tests.amortised.m9_toy trainer OUT.json

`compare`: the same seeded actions through `DeviceTrackerEnv(backend="warp", device="cpu")` and `TrackerEnv`, open
loop: the applied controls must be bit-identical (host-side control computation), and the trajectories, rewards and
records are reported with their largest differences (mujoco_warp float32 vs MuJoCo float64: not bit-equal; the
episode-end comparison is reported, the KA is the controls plus finiteness plus small early-row deviations).
`press`: a pressing toy clip, zero actions: the per-finger contact force on the warp backend beside the CPU env's.
`trainer`: `DeviceTrainer` on the warp backend (warp CPU device) for a few vectorised steps with learner updates;
reports the stats and whether the after-warm-up path ran.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from fastdo.amortised import tracker_obs as O
from fastdo.amortised.contact_ref import GridSDF
from fastdo.amortised.tracker_env import TipSpec, clip_table_entry
from fastdo.amortised.trainer import TrainInputs
from tests.amortised import m5_toy

CFG = m5_toy.CFG
REF_ROWS = ("obj_p", "obj_q", "obj_v", "q_ref", "tips_ref", "dref", "kappa")
HALF, PAD, RES = 0.02, 0.05, 24


def box_sdf() -> GridSDF:
    half = HALF + PAD
    step = 2.0 * half / (RES - 1)
    origin = np.full(3, -half)
    ax = origin[0] + step * np.arange(RES)
    g = np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), -1)
    q = np.abs(g) - HALF
    sd = np.linalg.norm(np.maximum(q, 0.0), axis=-1) + np.minimum(q.max(-1), 0.0)
    return GridSDF(sd, origin, float(step))


def _tips(spec) -> TipSpec:
    import mujoco

    m = spec.ex.model
    bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
    return TipSpec(wrist_body=bid("right_palm"), tip_body=(bid("right_finger"), bid("right_palm")),
                   tip_offset=np.array([[0.0, 0.0, -0.05], [0.04, 0.0, -0.01]]))


def _scorer_rows(entry: dict) -> dict:
    return {k: (v[1:] if k in REF_ROWS else v[:-1] if k == "ctrl_ref" else v) for k, v in entry.items()}


def inputs() -> TrainInputs:
    from tests.transductive.toy import toy_clip

    sdf = box_sdf()
    rng = np.random.default_rng(0)
    pts = rng.uniform(-HALF, HALF, size=(8, 3))
    pts[np.arange(8), rng.integers(0, 3, 8)] = HALF * np.sign(rng.normal(size=8))   # on the box's faces
    specs = [toy_clip("a", T=80, W=20), toy_clip("b", T=60, W=15, ref_shift=0.03)]
    tips = [_tips(sp) for sp in specs]
    entries = [_scorer_rows(clip_table_entry(sp, t, sdf, pts)) for sp, t in zip(specs, tips, strict=True)]
    return TrainInputs(specs, tips, [sdf, sdf], entries, ["a", "b"], [m5_toy.scorer(e) for e in entries])


def compare(steps: int = 12, n_worlds: int = 4, seed: int = 0) -> dict:
    from fastdo.amortised.tracker_env import TrackerEnv
    from fastdo.amortised.tracker_gpu import DeviceTrackerEnv, grouped_world_clip

    inp = inputs()
    table = O.RefTable.stack(list(inp.entries))
    wc = grouped_world_clip(n_worlds, 2)
    t0 = time.perf_counter()
    dev = DeviceTrackerEnv(CFG, inp.specs, table, inp.tips, inp.sdfs, wc, backend="warp", device="cpu")
    build_s = time.perf_counter() - t0
    cpu = TrackerEnv(CFG, inp.specs, table, inp.tips, inp.sdfs, wc)
    rd, rc = dev.reset_all(), cpu.reset_all()
    out = {"build_s": build_s, "reset_record_maxdiff": {k: float(np.abs(rd[k].astype(float) - rc[k].astype(float)).max())
                                                       for k in rd}, "steps": []}
    rng = np.random.default_rng(seed)
    ctrl_equal, finite = True, True
    for s in range(steps):
        a = rng.normal(size=(n_worlds, cpu.R)).astype(np.float32)
        t0 = time.perf_counter()
        od = dev.step(a)
        dt = time.perf_counter() - t0
        oc = cpu.step(a)
        row = {"s": s, "dev_s": dt, "end_dev": od.end, "end_cpu": oc.end}
        dq, dc = [], []
        for i in range(n_worlds):
            n = min(len(od.qpos_rows[i]), len(oc.qpos_rows[i]))
            dq.append(float(np.abs(od.qpos_rows[i][:n] - oc.qpos_rows[i][:n]).max()) if n else 0.0)
            ctrl_equal &= bool(np.array_equal(od.ctrl_rows[i][:n], oc.ctrl_rows[i][:n]))
            finite &= bool(np.isfinite(od.qpos_rows[i]).all())
        finite &= bool(np.isfinite(od.reward).all() and all(np.isfinite(np.asarray(v, float)).all()
                                                           for v in od.record.values()))
        row.update(max_dq=max(dq), max_dreward=float(np.abs(od.reward - oc.reward).max()))
        out["steps"].append(row)
        if any(e is not None for e in od.end) or any(e is not None for e in oc.end):
            break
    out.update(ctrl_equal=ctrl_equal, finite=finite, first_step_max_dq=out["steps"][0]["max_dq"],
               first_step_max_dreward=out["steps"][0]["max_dreward"])
    return out


def press(steps: int = 15) -> dict:
    """A pressing toy clip (`toy_clip(press=True)`: the palm presses the box), zero actions, 2 worlds: the per-finger
    normal force of the records on the warp backend (`mujoco_warp.contact_force`) beside the CPU env's
    (`mj_contactForce`), and the contact scalars, per step."""
    from fastdo.amortised.tracker_env import TrackerEnv
    from fastdo.amortised.tracker_gpu import DeviceTrackerEnv
    from tests.transductive.toy import toy_clip

    sdf = box_sdf()
    pts = np.random.default_rng(0).uniform(-HALF, HALF, size=(8, 3))
    sp = toy_clip("p", T=80, W=20, press=True)
    tip = _tips(sp)
    table = O.RefTable.stack([_scorer_rows(clip_table_entry(sp, tip, sdf, pts))])
    dev = DeviceTrackerEnv(CFG, [sp], table, [tip], [sdf], [0, 0], backend="warp", device="cpu")
    cpu = TrackerEnv(CFG, [sp], table, [tip], [sdf], [0, 0])
    dev.reset_all()
    cpu.reset_all()
    rows = []
    for _ in range(steps):
        a = np.zeros((2, cpu.R), np.float32)
        od, oc = dev.step(a), cpu.step(a)
        rows.append({"tf_warp": od.record["tip_force"].astype(float).tolist(),
                     "tf_cpu": oc.record["tip_force"].astype(float).tolist(),
                     "contact_warp": od.record["contact"].astype(float).tolist(),
                     "contact_cpu": oc.record["contact"].astype(float).tolist()})
    return {"steps": rows}


def trainer(out_dir: Path, steps: int = 30) -> dict:
    from fastdo.amortised.trainer_dev import DeviceTrainer

    inp = inputs()
    cfg = m5_toy.config(max_steps=steps, threads=1, assist=True)
    t = DeviceTrainer(cfg, inp, out_dir, backend="warp", device="cpu")
    s = t.run()
    return {k: v for k, v in s.items() if k in ("step", "rows", "updates", "episodes", "ends", "offers",
                                                  "certificates", "certified", "alpha0_full", "assisted_full")}


def main(argv: list[str] | None = None) -> int:
    a = sys.argv[1:] if argv is None else argv
    if len(a) != 2 or a[0] not in ("compare", "trainer", "press"):
        print("usage: -m tests.amortised.m9_toy {compare,trainer,press} OUT.json", file=sys.stderr)
        return 2
    out = Path(a[1])
    res = {"compare": compare, "press": press}[a[0]]() if a[0] != "trainer" else trainer(out.with_suffix(".run"))
    out.write_text(json.dumps(res, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
