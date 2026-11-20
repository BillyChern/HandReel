"""The baseline's warm-up start (open hand backed off along the palm normal), fastdo.policy.warmup."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
import torch

from fastdo.policy.scene import PolicyScene, load_policy_scene
from fastdo.policy.warmup import (
    MAX_POINTS,
    HandCloud,
    backoff_distance,
    hand_cloud,
    object_points,
    open_hand_start,
    palm_normal_local,
)
from tests.policy.env_fixtures import SCENE_XML, env_config, scene_missing, whisking_reference

pytestmark = scene_missing
CLEARANCE = 0.1                       # the baseline's warmup_min_clearance (do_as_i_do config)


def _scene() -> PolicyScene:
    cfg = env_config()
    return load_policy_scene(SCENE_XML, cfg.hz, cfg.n_substeps, cfg.kp, cfg.kd, cfg.wrist_kp, cfg.wrist_kd)


# _baseline_t is transcribed from DO AS I DO (https://github.com/malik-group/do-as-i-do, commit
# 824591b, retargeting/retargeting/utils/mjwp.py `warmup_analytical_init`), Copyright (c) 2026
# malik-group, MIT License; see licenses/DO-AS-I-DO-MIT.txt.
def _baseline_t(hand: np.ndarray, obj: np.ndarray, n: np.ndarray, c: float) -> float:
    """Transcription of the baseline's `warmup_analytical_init` solve (explicit pair differences, in chunks)."""
    best = 0.0
    for start in range(0, hand.shape[0], 500):
        d = hand[start:start + 500, None, :] - obj[None, :, :]
        n_dot_d = d @ n
        d_norm_sq = (d ** 2).sum(-1)
        delta = c * c - (d_norm_sq - n_dot_d ** 2)
        r_hi = n_dot_d + np.sqrt(np.maximum(delta, 0.0))
        active = (delta > 0) & (d_norm_sq < c ** 2)
        best = max(best, float(np.where(active, np.maximum(r_hi, 0.0), 0.0).max()))
    return best


def test_backoff_matches_the_baseline_solve_and_clears_the_object() -> None:
    scene = _scene()
    m = scene.model
    ref = whisking_reference()
    cloud = hand_cloud(scene, ref.q_ref[0, :28], torch.device("cpu"))
    obj_local = torch.as_tensor(object_points(scene))
    assert 1000 < cloud.points.shape[0] <= MAX_POINTS and 1000 < obj_local.shape[0] <= MAX_POINTS

    # Independent forward kinematics: MuJoCo at the full reference frame 0 places the same object vertices.
    data = mujoco.MjData(m)
    data.qpos[:] = ref.q_ref[0].numpy()
    mujoco.mj_kinematics(m, data)
    obj_world = obj_local.numpy() @ data.xmat[scene.obj_body].reshape(3, 3).T + data.xpos[scene.obj_body]
    n_local = palm_normal_local(scene)
    assert abs(np.linalg.norm(n_local) - 1.0) < 1e-9
    wrist_R = data.xmat[scene.wrist_body].reshape(3, 3)            # the hinge chain has no offsets (scene tests)
    np.testing.assert_allclose(cloud.normal.numpy(), wrist_R @ n_local, atol=1e-9)

    t = float(backoff_distance(cloud, obj_local, ref.obj_pose[:1], CLEARANCE)[0])
    hand, n = cloud.points.numpy(), cloud.normal.numpy()
    assert t == pytest.approx(_baseline_t(hand, obj_world, n, CLEARANCE), abs=1e-9)
    assert t > 0.01                                                 # frame 0 has hand vertices within 0.1 m

    # Meaning of t: every pair that starts inside the clearance ends on or outside it, and one ends exactly on it.
    inside = np.linalg.norm(hand[:, None] - obj_world[None], axis=-1) < CLEARANCE
    after = np.linalg.norm((hand - t * n)[:, None] - obj_world[None], axis=-1)
    assert float(after[inside].min()) == pytest.approx(CLEARANCE, abs=1e-9)

    # Batched over object poses (randomized draws) equals one pose at a time.
    shifted = ref.obj_pose[0].clone()
    shifted[:3, 3] += torch.tensor([0.01, -0.02, 0.005], dtype=torch.float64)
    poses = torch.stack([ref.obj_pose[0], shifted, ref.obj_pose[40]])
    batched = backoff_distance(cloud, obj_local, poses, CLEARANCE)
    single = torch.cat([backoff_distance(cloud, obj_local, p[None], CLEARANCE) for p in poses])
    torch.testing.assert_close(batched, single, rtol=0.0, atol=1e-12)
    assert float((batched[1:] - batched[0]).abs().max()) > 1e-4


def test_backoff_counts_only_pairs_that_start_inside_the_clearance() -> None:
    """The baseline's rule, on a synthetic cloud: a hand point already outside the clearance never constrains t, even
    when backing off would carry it through the object (point B), so t comes from point A alone."""
    a, b = [0.0, 0.0, 0.05], [0.0, 0.0, 0.3]                 # A inside the 0.1 m ball of the vertex, B outside it
    cloud = HandCloud(points=torch.tensor([a, b], dtype=torch.float64),
                      normal=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64))   # back-off direction -z
    t = backoff_distance(cloud, torch.zeros(1, 3, dtype=torch.float64), torch.eye(4, dtype=torch.float64)[None],
                         CLEARANCE)
    torch.testing.assert_close(t, torch.tensor([0.15], dtype=torch.float64))   # A: 0.05 + 0.1; B alone would say 0.4
    far = HandCloud(points=cloud.points[1:], normal=cloud.normal)
    assert float(backoff_distance(far, torch.zeros(1, 3, dtype=torch.float64), torch.eye(4, dtype=torch.float64)[None],
                                  CLEARANCE)[0]) == 0.0


def test_open_hand_start_zeroes_fingers_and_backs_the_wrist_off() -> None:
    q = torch.randn(3, 28, dtype=torch.float64)
    t = torch.tensor([0.0, 0.05, 0.2], dtype=torch.float64)
    n = torch.nn.functional.normalize(torch.randn(3, 3, dtype=torch.float64), dim=1)
    start = open_hand_start(q, t, n)
    assert float(start[:, 6:].abs().max()) == 0.0
    torch.testing.assert_close(start[:, 3:6], q[:, 3:6], rtol=0.0, atol=0.0)
    torch.testing.assert_close(start[:, :3], q[:, :3] - t[:, None] * n, rtol=0.0, atol=1e-15)
