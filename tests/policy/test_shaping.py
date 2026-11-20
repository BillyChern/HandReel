"""Potential-based reward shaping (design Section 5; Ng, Harada and Russell, ICML 1999).

F(s, s') = gamma Phi(s') - Phi(s) with Phi(s') = 0 exactly at absorbing terminations and kept at time-outs leaves the
optimal policies unchanged: checked by value iteration on a random tabular MDP, and on the CPU twin, where a time-out
step keeps gamma Phi(s') and a drop step uses Phi(s') = 0.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import torch

from fastdo.policy.rewards import RewardConfig, pose_errors, shaping_potential, shaping_term
from tests.policy.env_fixtures import SCENE_XML, env_config, scene_missing, whisking_reference

logger = logging.getLogger(__name__)

N_STATES, N_ACTIONS, GAMMA = 40, 4, 0.99


def _value_iteration(P: np.ndarray, R: np.ndarray, absorbing: np.ndarray, gamma: float) -> tuple[np.ndarray, np.ndarray]:
    """Optimal V (S,) and Q (S, A) of transitions P (S, A, S) and rewards R (S, A, S); absorbing states have V = 0."""
    V = np.zeros(P.shape[0])
    for _ in range(20000):
        Q = (P * (R + gamma * V[None, None, :])).sum(-1)
        Q[absorbing] = 0.0
        V_new = Q.max(1)
        if np.abs(V_new - V).max() < 1e-14:
            return V_new, Q
        V = V_new
    raise AssertionError("value iteration did not converge")


def _random_mdp(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    absorbing = np.zeros(N_STATES, dtype=bool)
    absorbing[:6] = True
    P = rng.dirichlet(np.full(N_STATES, 0.3), size=(N_STATES, N_ACTIONS))
    P[absorbing] = 0.0
    for s in np.nonzero(absorbing)[0]:
        P[s, :, s] = 1.0
    R = rng.normal(size=(N_STATES, N_ACTIONS, N_STATES))
    R[absorbing] = 0.0
    return P, R, absorbing


def _shaping(phi: np.ndarray, absorbing: np.ndarray, gamma: float) -> np.ndarray:
    """F (S, A, S) through `shaping_term`, Phi(s') replaced by 0 where s' is absorbing."""
    shape = (N_STATES, N_ACTIONS, N_STATES)
    phi_t = torch.as_tensor(phi, dtype=torch.float64)
    return shaping_term(RewardConfig(gamma=gamma), phi_t[:, None, None].expand(shape),
                        phi_t[None, None, :].expand(shape), torch.as_tensor(absorbing)[None, None, :].expand(shape)
                        ).numpy()


def test_shaping_preserves_optimal_policy_tabular() -> None:
    """Random 40-state MDP (6 absorbing states, gamma 0.99) and a potential of scale 3 (far larger than the rewards):
    value iteration with and without the shaping term gives the same greedy policy, and the shaped values are the
    originals minus Phi within 1e-10. A potential that stays non-zero at the absorbing states, i.e. Phi(s') kept
    where an episode ends, changes the greedy policy (the bonus for ending early the rule prevents)."""
    P, R, absorbing = _random_mdp(0)
    rng = np.random.default_rng(1)
    phi = 3.0 * rng.normal(size=N_STATES)
    phi[absorbing] = 5.0                                  # shaping_term must ignore Phi on absorbing successors
    V, Q = _value_iteration(P, R, absorbing, GAMMA)
    V_s, Q_s = _value_iteration(P, R + _shaping(phi, absorbing, GAMMA), absorbing, GAMMA)
    live = ~absorbing
    np.testing.assert_array_equal(Q[live].argmax(1), Q_s[live].argmax(1))
    gaps = np.sort(Q[live], axis=1)
    assert float((gaps[:, -1] - gaps[:, -2]).min()) > 1e-6          # no near-ties: the policy check is decisive
    np.testing.assert_allclose((V_s - V)[live], -phi[live], rtol=0.0, atol=1e-10)
    np.testing.assert_allclose((V_s - V)[absorbing], 0.0, rtol=0.0, atol=0.0)

    lure = np.where(absorbing, 60.0, phi)                  # Phi(s') paid on entering an absorbing state
    shape = (N_STATES, N_ACTIONS, N_STATES)
    phi_t = torch.as_tensor(lure, dtype=torch.float64)
    no_zero = shaping_term(RewardConfig(gamma=GAMMA), phi_t[:, None, None].expand(shape),
                           phi_t[None, None, :].expand(shape), torch.zeros(shape, dtype=torch.bool)).numpy()
    _, Q_bad = _value_iteration(P, R + no_zero, absorbing, GAMMA)
    changed = int((Q[live].argmax(1) != Q_bad[live].argmax(1)).sum())
    logger.info("shaped values = V - Phi to %.2e; a non-zero terminal potential changes %d of %d greedy actions",
                float(np.abs((V_s - V)[live] + phi[live]).max()), changed, int(live.sum()))
    assert changed > 0


def _potential(cfg: RewardConfig, obj_pose: torch.Tensor, ref_pose: torch.Tensor) -> torch.Tensor:
    """Phi = -w_phi sqrt(w_rot e_rot^2 + w_tr e_pos^2) / d_phi from `pose_errors`, independently of the env's path."""
    e_pos, e_rot = pose_errors(obj_pose.double(), ref_pose.double())
    return -cfg.w_phi * torch.sqrt(cfg.w_rot * e_rot**2 + cfg.w_tr * e_pos**2) / cfg.d_phi


@scene_missing
def test_shaping_zero_only_on_absorbing() -> None:
    """CPU twin without a warm-up, zero residuals: the unheld object falls. On a 4-frame clip every step's shaping is
    reward_scale (gamma Phi(s') - Phi(s)) with Phi against the frame of each state, including the time-out onto the
    last frame, whose Phi(s') is far from 0; on a 60-frame clip the drop step (a failure) pays -reward_scale Phi(s)."""
    from fastdo.policy.env import make_env

    cfg = env_config(n_worlds=1, warmup_duration=0.0)
    rc, scale = cfg.reward, cfg.reward_scale
    assert rc.gamma == GAMMA and rc.w_phi > 0.0
    env = make_env(cfg, SCENE_XML, "cpu")

    def run(ref_frames: int) -> tuple[list[dict[str, float]], bool]:
        ref = whisking_reference(n_frames=ref_frames)
        env.reset(ref, None)
        o = env.scene.obj_qpos
        from fastdo.policy.scene import object_pose

        phi = float(_potential(rc, object_pose(env._read_state()[0][:, o:o + 7]), ref.obj_pose[:1])[0])
        rows = []
        for _ in range(ref_frames + 20):
            _, reward, done, info = env.step(torch.zeros(1, 28))
            k = int(info["k"][0])
            phi_next = float(_potential(rc, info["obj_pose"], ref.obj_pose[k:k + 1])[0])
            raw = sum(float(info["reward_terms"][name][0]) for name in ("pose", "joint", "contact", "drop",
                                                                        "action_rate"))
            rows.append({"phi": phi, "phi_next": phi_next, "shaping": float(info["shaping"][0]), "raw": raw,
                         "reward": float(reward[0]), "failed": bool(info["terminated"][0]),
                         "time_out": bool(info["time_outs"][0])})
            phi = phi_next
            if bool(done[0]):
                return rows, bool(info["time_outs"][0])
        raise AssertionError("the episode never ended")

    rows, timed_out = run(4)
    assert timed_out and len(rows) == 3 and not any(r["failed"] for r in rows)
    for r in rows:
        assert math.isclose(r["shaping"], scale * (GAMMA * r["phi_next"] - r["phi"]), rel_tol=0.0, abs_tol=2e-6), r
        assert math.isclose(r["reward"], scale * r["raw"] + r["shaping"], rel_tol=0.0, abs_tol=2e-6), r
    assert abs(rows[-1]["phi_next"]) > 0.1                   # the kept potential is not negligible
    kept_phi = rows[-1]["phi_next"]

    rows, timed_out = run(60)
    drop = rows[-1]
    assert not timed_out and drop["failed"] and abs(drop["phi_next"]) > 1.0
    assert math.isclose(drop["shaping"], -scale * drop["phi"], rel_tol=0.0, abs_tol=2e-6), drop
    assert abs(drop["shaping"] - scale * (GAMMA * drop["phi_next"] - drop["phi"])) > 0.01
    for r in rows[:-1]:
        assert math.isclose(r["shaping"], scale * (GAMMA * r["phi_next"] - r["phi"]), rel_tol=0.0, abs_tol=2e-6), r
    logger.info("time-out step: Phi(s') %.4f kept; drop step after %d steps: Phi(s') %.4f replaced by 0", kept_phi,
                len(rows), drop["phi_next"])


def test_shaping_potential_is_the_scaled_product_metric_distance() -> None:
    """Phi = -w_phi d / d_phi with d^2 = w_rot |Log(R_ref^T R)|^2 + w_tr |p - p_ref|^2, zero on the reference (to the
    angle accuracy of `so3_log`, about 1e-8 relative)."""
    cfg = RewardConfig(w_phi=2.0, d_phi=0.1, w_rot=1.0, w_tr=10.0)
    ref = torch.eye(4, dtype=torch.float64).repeat(2, 1, 1)
    pose = ref.clone()
    pose[1, :3, 3] = torch.tensor([0.03, 0.0, 0.04])                         # 5 cm
    c, s = math.cos(0.2), math.sin(0.2)
    pose[1, :2, :2] = torch.tensor([[c, -s], [s, c]], dtype=torch.float64)  # 0.2 rad about z
    phi = shaping_potential(cfg, pose, ref)
    assert float(phi[0]) == 0.0
    assert math.isclose(float(phi[1]), -2.0 * math.sqrt(0.2**2 + 10.0 * 0.05**2) / 0.1, rel_tol=1e-7)
