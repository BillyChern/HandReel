"""M9 P1: the GPU code path on warp's CPU device,
in DO AS I DO's `retargeting` env (mujoco_warp 0.0.1), subprocess `tests/amortised/m9_toy.py`.

- `compare`: `DeviceTrackerEnv(backend="warp")` and the CPU `TrackerEnv` driven open loop by the same actions apply
  bit-identical controls (host-side control computation); the warp states, rewards and records are finite; the
  first step's states agree to 1e-6 m and its rewards to 1e-4 (mujoco_warp float32 vs MuJoCo float64; the reset
  records are identical); both engines end the same episodes (clip end) at the same step.
- `press` (the palm pressing the box, zero actions): the warp backend's per-finger normal force
  (`mujoco_warp.contact_force`) lands on the same finger as the CPU env's (`mj_contactForce`), zero on the others, the
  touch flags agree, and after settling (steps 5-14) its magnitude is within 25% per step and 10% on average
  (float32 solver vs float64; measured about 1% on average when this test was written).
- `trainer`: `DeviceTrainer` on the warp backend runs acting, the environment, the replay and learner updates, the
  assist schedule, and inline CPU certification of stored candidates (hash noise) end to end."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

TREE = Path(__file__).resolve().parents[2]
PY = Path("/opt/fastdo/miniforge3/envs/retargeting/bin/python")
pytestmark = [pytest.mark.slow, pytest.mark.skipif(not PY.is_file(), reason="needs the retargeting env")]


def _run(cmd: str, out: Path) -> dict:
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": f"{TREE / 'src'}:{TREE}", "OMP_NUM_THREADS": "2"}
    r = subprocess.run([str(PY), "-m", "tests.amortised.m9_toy", cmd, str(out)], cwd=str(TREE), env=env,
                       capture_output=True, text=True, timeout=2400, check=False)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return json.loads(out.read_text())


def test_warp_backend_on_the_cpu_device_matches_the_cpu_env_open_loop(tmp_path: Path) -> None:
    res = _run("compare", tmp_path / "compare.json")
    assert res["ctrl_equal"] and res["finite"]
    assert all(v == 0.0 for v in res["reset_record_maxdiff"].values()), res["reset_record_maxdiff"]
    assert res["first_step_max_dq"] < 1e-6 and res["first_step_max_dreward"] < 1e-4, res["steps"][0]
    last = res["steps"][-1]
    assert last["end_dev"] == last["end_cpu"] and any(e is not None for e in last["end_dev"]), last
    assert all(s["max_dq"] < 5e-3 for s in res["steps"]), [s["max_dq"] for s in res["steps"]]


def test_device_trainer_on_the_warp_backend_end_to_end(tmp_path: Path) -> None:
    s = _run("trainer", tmp_path / "trainer.json")
    assert s["step"] == 30 and s["updates"] > 0 and s["rows"] == 30 * 4 * 5
    assert s["assisted_full"] + s["alpha0_full"] >= 1
    assert s["certificates"] == s["offers"].get("stored", 0) and s["certified"] == s["certificates"]


def test_warp_contact_force_matches_the_cpu_envs_on_a_pressing_clip(tmp_path: Path) -> None:
    import numpy as np

    rows = _run("press", tmp_path / "press.json")["steps"]
    w = np.array([r["tf_warp"] for r in rows])
    c = np.array([r["tf_cpu"] for r in rows])
    assert ((w > 0) == (c > 0)).all() and (c > 0).any()
    assert np.array_equal(np.array([r["contact_warp"] for r in rows])[..., 1],
                          np.array([r["contact_cpu"] for r in rows])[..., 1])
    ws, cs = w[5:].sum(-1), c[5:].sum(-1)
    assert (cs > 1.0).all()
    assert (np.abs(ws / cs - 1.0) <= 0.25).all(), (ws / cs).tolist()
    assert abs(ws.mean() / cs.mean() - 1.0) <= 0.10, (ws.mean(), cs.mean())
