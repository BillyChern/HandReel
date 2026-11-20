# Copyright 2026 The HandReel authors. SPDX-License-Identifier: Apache-2.0
# Portions adapted from DO AS I DO (https://github.com/malik-group/do-as-i-do, commit 824591b),
# Copyright (c) 2026 malik-group, MIT License; see licenses/DO-AS-I-DO-MIT.txt and THIRD_PARTY_LICENSES.md.
"""C2 whole-clip rollouts of the tier-0 controller on DO AS I DO's batched MuJoCo-Warp env (retargeting env only).

The loop mirrors DO AS I DO's `rollout` (sampling.py) for a rollout started at sim step 0: perturbation state reset,
weld active and the warmup gravity as the executor has it (zero) until the release, perturbation from the release
on, the step reward with the warmup multiplier, and the same cleanup (weld re-activated, default gravity, then state
and env params restored: `load_env_params` puts back the pre-rollout gravity and weld state). The weld/gravity
switch happens before step index W, as in DO AS I DO's EXECUTION loop (their rollout switches one step earlier), so
world 0 predicts the execution. Differences otherwise: the horizon is the whole clip; per world, the controls come
from the tier-0 controller with a per-world theta; the worlds in `deterministic` get their perturbation zeroed after
DO AS I DO's `apply_perturbation` (they predict the undisturbed execution). Recorded per world on the device: whole / post-warmup object error sums, drop flags,
summed reward; for world 0 also the qpos trajectory (one host copy at the end).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

__all__ = ["RolloutResult", "TierZeroController", "quat_angle", "whole_clip_rollout"]


def quat_angle(q: torch.Tensor, q_ref: torch.Tensor) -> torch.Tensor:
    """Rotation angle between unit quaternions (wxyz), (W, 4) vs (4,) or (W, 4)."""
    dot = (q * q_ref).sum(-1).abs().clamp(max=1.0)
    return 2.0 * torch.acos(dot)


@dataclass
class TierZeroController:
    """u(t, w) = u_ref(t) + base(t) + sum_f exp(theta_wf) * finger_f(t) [clamped] + theta_w6 * e_z; servo separate."""

    ctrl_ref: torch.Tensor      # (T, nu) DO AS I DO's control reference (warmup interpolation included)
    base: torch.Tensor          # (T, nu) gravity feedforward (z column) + gap term (finger columns)
    finger: torch.Tensor        # (5, T, nu) per-finger penetration terms
    finger_cols: torch.Tensor   # (n_f,) long
    z_col: int
    joint_max: float
    ref_obj_pos: torch.Tensor   # (T, 3) reference object position
    obj_pos_slice: slice
    release_step: int
    servo_sat: float

    def offsets(self, t: int, theta: torch.Tensor) -> torch.Tensor:
        """(W, nu) offsets at step t for per-world theta (W, 7)."""
        t = min(t, self.base.shape[0] - 1)
        scale = torch.exp(theta[:, :5])                                     # (W, 5)
        fin = torch.einsum("wf,fu->wu", scale, self.finger[:, t])
        off = self.base[t].unsqueeze(0) + fin
        cols = self.finger_cols
        off[:, cols] = off[:, cols].clamp(-self.joint_max, self.joint_max)
        off[:, self.z_col] += theta[:, 6]
        return off

    def plan(self, s: int, h: int, theta: torch.Tensor) -> torch.Tensor:
        """(h, nu) open-loop plan from step s for one theta (7,): u_ref + offsets (DO AS I DO's padding holds)."""
        idx = [min(s + i, self.ctrl_ref.shape[0] - 1) for i in range(h)]
        th = theta.reshape(1, -1)
        return torch.stack([self.ctrl_ref[i] + self.offsets(i, th)[0] for i in idx])

    def servo(self, qpos: torch.Tensor, t_idx: torch.Tensor, gains: torch.Tensor) -> torch.Tensor:
        """(W, 3) wrist-position correction sat(K_o * (p_ref - p)), zero before the release."""
        t_idx = t_idx.clamp(0, self.ref_obj_pos.shape[0] - 1)
        e = self.ref_obj_pos[t_idx] - qpos[:, self.obj_pos_slice]
        corr = gains.reshape(-1, 1) * e
        n = corr.norm(dim=1, keepdim=True).clamp(min=1e-12)
        corr = corr * torch.clamp(self.servo_sat / n, max=1.0)
        return corr * (t_idx >= self.release_step).reshape(-1, 1).to(corr.dtype)


@dataclass
class RolloutResult:
    reward: np.ndarray          # (W,) summed reward (DO AS I DO's warmup multiplier)
    whole_pos: np.ndarray       # (W,) mean object position error over rows [0, T)
    whole_rot: np.ndarray
    post_pos: np.ndarray        # (W,) mean over rows [W_warm, end)
    post_rot: np.ndarray
    dropped: np.ndarray         # (W,) bool: object below ref - 0.5 m, below the termination z, or non-finite
    qpos0: np.ndarray           # (T, nq) world 0, one row per executed step (state after the step)
    world_steps: int
    wall_s: float


def whole_clip_rollout(config: Any, env: Any, m: Any, ctl: TierZeroController, theta: torch.Tensor,
                       deterministic: torch.Tensor, n_steps: int, qpos_ref: torch.Tensor, ref_data: tuple,
                       with_reward: bool, end_step: int, drop_margin: float = 0.5) -> RolloutResult:
    """Run the controller for `n_steps` from the current state (sim step 0) on all worlds; restore the state."""
    import time

    import warp as wp

    t0 = time.perf_counter()
    nw = env.num_worlds
    dev = config.device
    w_steps = int(config.warmup_steps)
    init_state = m.save_state(env)
    init_param = m.save_env_params(config, env)
    env = m.load_env_params(config, env, config.env_params_list[0][0])
    for st in (env._perturb_active, env._perturb_force, env._perturb_torque):
        for tensor in st.values():
            tensor.zero_()
    m.set_weld_active(config, env, True)
    perturbing = any(float(getattr(config, k)) != 0 for k in
                     ("perturb_force", "perturb_torque", "perturb_force_scale", "perturb_torque_scale"))
    obj_body = int(env.model_cpu.body(f"{'left' if config.embodiment_type == 'left' else 'right'}_object").id)
    gains = theta[:, 5].to(dev)
    qpos_t = wp.to_torch(env.data_wp.qpos)
    xfrc = wp.to_torch(env.data_wp.xfrc_applied)
    cum = torch.zeros(nw, device=dev)
    pos_all = torch.zeros(nw, device=dev)
    rot_all = torch.zeros(nw, device=dev)
    pos_post = torch.zeros(nw, device=dev)
    rot_post = torch.zeros(nw, device=dev)
    dropped = torch.zeros(nw, dtype=torch.bool, device=dev)
    q0 = torch.zeros((n_steps, qpos_t.shape[1]), device=dev)
    ps = ctl.obj_pos_slice
    qs = slice(ps.stop, ps.stop + 4)
    n_post = 0
    for t in range(n_steps):
        abs_step = t + 1
        if t == w_steps:
            m.set_weld_active(config, env, False)
            m.set_gravity(env, list(config.default_gravity))
        ctrl = ctl.ctrl_ref[min(t, ctl.ctrl_ref.shape[0] - 1)].unsqueeze(0) + ctl.offsets(t, theta)
        ctrl[:, :3] += ctl.servo(qpos_t, torch.full((nw,), t, device=dev, dtype=torch.long), gains)
        with wp.ScopedDevice(env.device):
            if perturbing and abs_step >= w_steps:
                m.apply_perturbation(config, env, sim_step=abs_step)
                xfrc[deterministic, obj_body, :] = 0.0
            wp.copy(env.data_wp.ctrl, wp.from_torch(ctrl.to(torch.float32).contiguous()))
            wp.capture_launch(env.graph)
        if with_reward:
            ref = [r[min(abs_step, r.shape[0] - 1)] for r in ref_data]
            rew, _ = (m.get_reward(config, env, ref, sim_step=abs_step) if t < n_steps - 1
                      else m.get_terminal_reward(config, env, ref, sim_step=abs_step))
            cum += (config.warmup_rew_multiplier if t < w_steps - 1 else 1.0) * rew
        # executed row t (state after step t+1) is compared with reference row t, as DO AS I DO's error does
        rq = qpos_ref[min(t, qpos_ref.shape[0] - 1)]
        pe = (qpos_t[:, ps] - rq[ps]).norm(dim=1)
        re = quat_angle(qpos_t[:, qs], rq[qs])
        pe = torch.nan_to_num(pe, nan=1e3)
        re = torch.nan_to_num(re, nan=math.pi)
        pos_all += pe
        rot_all += re
        if w_steps <= t < end_step:
            pos_post += pe
            rot_post += re
            n_post += 1
        z = qpos_t[:, ps.start + 2]
        dropped |= (z < rq[ps.start + 2] - drop_margin) | (z < config.terminate_z_threshold) | ~torch.isfinite(z)
        q0[t] = qpos_t[0]
    m.set_weld_active(config, env, True)
    m.set_gravity(env, list(config.default_gravity))
    env = m.load_state(env, init_state)
    env = m.load_env_params(config, env, init_param)
    n_post = max(n_post, 1)
    res = RolloutResult(
        reward=cum.cpu().numpy(), whole_pos=(pos_all / n_steps).cpu().numpy(), whole_rot=(rot_all / n_steps).cpu().numpy(),
        post_pos=(pos_post / n_post).cpu().numpy(), post_rot=(rot_post / n_post).cpu().numpy(),
        dropped=dropped.cpu().numpy(), qpos0=q0.cpu().numpy().astype(np.float64), world_steps=int(nw * n_steps),
        wall_s=0.0)
    res.wall_s = time.perf_counter() - t0
    return res
