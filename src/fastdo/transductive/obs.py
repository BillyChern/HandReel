"""Observation of one world of the transductive residual environment.

At a policy step the world is at state x_s (s executor rows done); everything is in the clip's world frame (the
scenes are gravity aligned), and the reference rows are indexed like DO AS I DO's reward (state x_s against
reference row s). With R robot dofs (6 wrist + fingers), in this order:

- object position error (3, /pos) and rotation error quat_sub(q_obj, q_ref[s]) (3, /rot); object velocity (6);
- robot joint error q - q_ref[s] (R, per-group scales) and joint velocity (R);
- plan: ctrl_ref[s + K] - q (R): where the warm start drives the joints next (the residual adds to it);
- look-ahead, for each offset d in `future_rows`: reference object position and rotation relative to the current
  object (6) and reference joints relative to the current joints (R);
- object position relative to the wrist slide coordinates (3, /2 pos);
- the previous residual target over delta_max (R);
- contact: max hand-object penetration (/5 mm), hand-object touch, object-pedestal touch (3);
- phase: remaining warmup fraction, s / T (2);
- running PRIMARY budget: partial sums of position / rotation errors over 0.1 T / 0.5 T (2; 1 = no longer passable);
- the episode's training assist level alpha (1; 0 for every executor episode, `EnvConfig` assist curriculum);
- optionally the clip's one-hot id.
Every entry is clamped to +-obs_clip.
"""

from __future__ import annotations

import numpy as np

from fastdo.physics.fd_problem import quat_sub
from fastdo.transductive.clip import ClipSpec
from fastdo.transductive.config import SUCCESS_POS_M, SUCCESS_ROT_RAD, EnvConfig

__all__ = ["delta_scale", "joint_scale", "obs_dim", "observe"]


def joint_scale(cfg: EnvConfig, n_robot: int) -> np.ndarray:
    s = np.full(n_robot, cfg.obs_scale_joint)
    s[:3], s[3:6] = cfg.obs_scale_pos, cfg.obs_scale_rot
    return s


def delta_scale(cfg: EnvConfig, n_robot: int) -> np.ndarray:
    """Per-dimension delta_max (wrist position, wrist rotation, fingers)."""
    s = np.full(n_robot, cfg.delta_max[2])
    s[:3], s[3:6] = cfg.delta_max[0], cfg.delta_max[1]
    return s


def _vel_scale(n_robot: int) -> np.ndarray:
    s = np.full(n_robot, 10.0)
    s[:3], s[3:6] = 1.0, 5.0
    return s


def obs_dim(cfg: EnvConfig, n_robot: int, n_clips: int) -> int:
    R = n_robot
    return 12 + 3 * R + len(cfg.future_rows) * (6 + R) + 3 + R + 3 + 2 + 2 + 1 + (n_clips if cfg.clip_onehot else 0)


def observe(cfg: EnvConfig, sp: ClipSpec, s: int, qpos: np.ndarray, qvel: np.ndarray, delta_prev: np.ndarray,
            sums: tuple[float, float], contact: tuple[float, bool, bool], clip_index: int, n_clips: int,
            alpha: float = 0.0) -> np.ndarray:
    """The observation vector (module docstring) of a world of clip `sp` at row s."""
    R, a0 = sp.n_robot, sp.ex.obj_qadr
    qref, cref = sp.rw.qpos_ref, sp.ctrl_ref
    last = qref.shape[0] - 1
    js = joint_scale(cfg, R)
    ps, rs = cfg.obs_scale_pos, cfg.obs_scale_rot
    q, ob_p, ob_q = qpos[:R], qpos[a0:a0 + 3], qpos[a0 + 3:a0 + 7]
    r = qref[min(s, last)]
    parts = [(ob_p - r[a0:a0 + 3]) / ps, quat_sub(ob_q, r[a0 + 3:a0 + 7]) / rs,
             qvel[a0:a0 + 3], qvel[a0 + 3:a0 + 6] / 5.0,
             (q - r[:R]) / js, qvel[:R] / _vel_scale(R),
             (cref[min(s + cfg.rows_per_step, cref.shape[0] - 1)] - q) / js]
    for d in cfg.future_rows:
        f = qref[min(s + d, last)]
        parts += [(f[a0:a0 + 3] - ob_p) / ps, quat_sub(f[a0 + 3:a0 + 7], ob_q) / rs, (f[:R] - q) / js]
    T = sp.T
    pen, ho, ped = contact
    parts += [(ob_p - q[:3]) / (2 * ps), delta_prev / delta_scale(cfg, R),
              np.array([pen / 0.005, float(ho), float(ped)]),
              np.array([max(sp.W - s, 0) / max(sp.W, 1), s / T]),
              np.array([sums[0] / (SUCCESS_POS_M * T), sums[1] / (SUCCESS_ROT_RAD * T)]), np.array([alpha if cfg.assist_observed else 0.0])]
    if cfg.clip_onehot:
        oh = np.zeros(n_clips)
        oh[clip_index] = 1.0
        parts.append(oh)
    out = np.concatenate([np.asarray(p, np.float64).ravel() for p in parts])
    return np.clip(out, -cfg.obs_clip, cfg.obs_clip).astype(np.float32)
