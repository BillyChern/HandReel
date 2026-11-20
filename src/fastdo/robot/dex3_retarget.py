"""Right-hand MANO -> Dex3-1 kinematic retargeting with the semantics of Unitree's own DexPilot config.

Unitree's teleop (`xr_teleoperate/assets/unitree_hand/unitree_dex3.yml`, solved by Unitree's dex-retargeting fork
`optimizer.py::DexPilotOptimizer`) retargets FOUR keypoints: wrist, thumb tip, index tip and middle tip, expressed in
the Dex3 URDF `base_link` frame (the "Unitree hand convention": x palm -> back of hand, y middle finger -> wrist, z
index -> pinky for the right hand). This module reproduces that objective on the official G1 model
(`fastdo.robot.g1_model`, site `right_dex3_base` = URDF `base_link`, sites `right_*_tip` = URDF `*_tip`):

- Six vectors (yml `target_link_human_indices_dexpilot` [[9,14,14,0,0,0],[4,4,9,4,9,14]], origin row / task row):
  thumb - index, thumb - middle, index - middle (tip-to-tip), then wrist -> thumb, index, middle.
- Per-vector cost `SmoothL1(||r_i(q) - s v_i||; beta = huber_delta) * w_i / 6`, plus `norm_delta ||q - q_last||^2`.
- Weights: tip-to-tip 1, wrist-to-tip `len_proj + num_fingers` = 3 + 3 = 6 (the source's expression; its comment says
  10, which is the 4-finger value).
- Pinch projection with hysteresis: a thumb-to-finger target closer than `project_dist` (3 cm) is replaced by its
  direction times eta1 (1e-4 m) with weight 200, and released above `escape_dist` (5 cm); the index-middle vector is
  projected (eta2 = 3 cm, weight 400) when both thumb vectors are projected and it is <= 3 cm.
- Joint limits: `g1_constants.RIGHT_HAND_COMMAND_LIMITS` (the intersection of the official files) instead of the URDF's
  limits +- 1e-3 (a command must be valid on hardware).
- Sequence: warm start from the previous frame's UNFILTERED solution (first frame: mid-range), output through the
  first-order low-pass `y += alpha (x - y)`. Unitree runs it at 100 Hz with alpha 0.2; at another rate `fs` the same
  time constant needs `alpha_fs = 1 - (1 - 0.2)^(100 / fs)` (`rate_matched_alpha`).

Differences from the source, all deliberate: the optimiser is scipy L-BFGS-B (nlopt SLSQP is not in our env); the
`norm_delta` term is in the objective VALUE as well as the gradient (the source adds it to the gradient only); the
filter alpha is rate-matched. MANO input: wrist = joint 0, tips = MANO vertices 745 / 317 / 444 (`hawor_npz`); the
hand frame is built from the MCP joints with the same anatomical axes as Unitree's convention (`mano_hand_frame`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np
from scipy.optimize import minimize

from fastdo.robot import g1_constants as C
from fastdo.robot.g1_model import G1Upper

__all__ = [
    "MANO_TIP_VERTICES",
    "Dex3DexPilot",
    "DexPilotConfig",
    "dex3_tip_positions",
    "dexpilot_vectors",
    "mano_hand_frame",
    "mano_keypoints",
    "rate_matched_alpha",
]

MANO_TIP_VERTICES = (745, 317, 444)          # thumb, index, middle (fastdo.hands.hawor_npz._FINGERTIP_VERTEX_IDS)
_ORIGIN = np.array([2, 3, 3, 0, 0, 0])        # keypoint rows: 0 wrist, 1 thumb, 2 index, 3 middle
_TASK = np.array([1, 1, 2, 1, 2, 3])
_N_VEC = 6
_LEN_PROJ = 3


@dataclass(frozen=True)
class DexPilotConfig:
    """`unitree_dex3.yml` + `DexPilotOptimizer` defaults."""

    scaling: float = 1.0
    huber_delta: float = 0.03
    norm_delta: float = 4e-3
    project_dist: float = 0.03
    escape_dist: float = 0.05
    eta1: float = 1e-4
    eta2: float = 3e-2
    low_pass_alpha: float = 0.2
    low_pass_rate_hz: float = 100.0
    max_iter: int = 200
    limits: tuple[tuple[float, float], ...] = field(default=C.RIGHT_HAND_COMMAND_LIMITS)


def rate_matched_alpha(alpha: float, alpha_rate_hz: float, rate_hz: float) -> float:
    """The low-pass coefficient at `rate_hz` with the time constant of `alpha` at `alpha_rate_hz`."""
    return float(1.0 - (1.0 - alpha) ** (alpha_rate_hz / rate_hz))


def mano_hand_frame(joints: np.ndarray) -> np.ndarray:
    """`(T,3,3)` rotation (columns = axes in the input frame) of the right hand in Unitree's hand convention.

    y = middle MCP (OpenPose 9) -> wrist (0); z = index MCP (5) -> pinky MCP (17), made orthogonal to y; x = y x z, which
    points from the palm to the back of the hand for a right hand. Right hand only.
    """
    j = np.asarray(joints, dtype=np.float64)
    y = j[:, 0] - j[:, 9]
    y /= np.linalg.norm(y, axis=-1, keepdims=True)
    z = j[:, 17] - j[:, 5]
    z = z - np.sum(z * y, axis=-1, keepdims=True) * y
    z /= np.linalg.norm(z, axis=-1, keepdims=True)
    x = np.cross(y, z)
    return np.stack([x, y, z], axis=-1)


def mano_keypoints(joints: np.ndarray, vertices: np.ndarray | None) -> np.ndarray:
    """`(T,4,3)` wrist, thumb, index, middle tips in the input frame (tips from MANO vertices when given, else from
    the OpenPose tip joints 4/8/12, which HaWoR regresses a few mm from the vertices)."""
    j = np.asarray(joints, dtype=np.float64)
    tips = j[:, [4, 8, 12]] if vertices is None else np.asarray(vertices, dtype=np.float64)[:, list(MANO_TIP_VERTICES)]
    return np.concatenate([j[:, :1], tips], axis=1)


def dexpilot_vectors(keypoints_hand: np.ndarray) -> np.ndarray:
    """`(...,6,3)` DexPilot target vectors (task - origin) from `(...,4,3)` keypoints."""
    return keypoints_hand[..., _TASK, :] - keypoints_hand[..., _ORIGIN, :]


class _HandFK:
    """Tip positions and Jacobians in the `right_dex3_base` frame (independent of the arm pose)."""

    def __init__(self, g1: G1Upper) -> None:
        self.g1 = g1
        self.data = mujoco.MjData(g1.model)
        self.data.qpos[:] = g1.default_qpos()
        self._jac = np.zeros((3, g1.model.nv))

    def __call__(self, q_hand: np.ndarray, need_jac: bool = True) -> tuple[np.ndarray, np.ndarray | None]:
        m, d, g = self.g1.model, self.data, self.g1
        d.qpos[g.right_hand_qpos] = q_hand
        mujoco.mj_kinematics(m, d)
        rot = d.site_xmat[g.site_base].reshape(3, 3)
        base = d.site_xpos[g.site_base]
        pts = np.zeros((4, 3))
        pts[1:] = (d.site_xpos[g.site_tips] - base) @ rot
        if not need_jac:
            return pts, None
        mujoco.mj_comPos(m, d)
        jac = np.zeros((4, 3, 7))
        for k, site in enumerate(g.site_tips):
            mujoco.mj_jacSite(m, d, self._jac, None, int(site))
            jac[k + 1] = rot.T @ self._jac[:, g.right_hand_dof]
        return pts, jac


def dex3_tip_positions(g1: G1Upper, q_hand: np.ndarray) -> np.ndarray:
    """`(T,4,3)` Dex3 keypoints (wrist = base origin, thumb, index, middle tips) in `right_dex3_base`, per `(T,7)`."""
    fk = _HandFK(g1)
    return np.stack([fk(q, need_jac=False)[0] for q in np.atleast_2d(q_hand)])


def _smooth_l1(x: np.ndarray, beta: float) -> tuple[np.ndarray, np.ndarray]:
    small = x < beta
    return np.where(small, 0.5 * x * x / beta, x - 0.5 * beta), np.where(small, x / beta, 1.0)


class Dex3DexPilot:
    """Stateful DexPilot retargeter for the right Dex3-1 (module docstring). Joint vectors are in package enum order."""

    def __init__(self, g1: G1Upper, cfg: DexPilotConfig | None = None, rate_hz: float | None = None) -> None:
        cfg = DexPilotConfig() if cfg is None else cfg
        self.cfg = cfg
        self.fk = _HandFK(g1)
        self.limits = np.asarray(cfg.limits, dtype=np.float64)
        self.alpha = cfg.low_pass_alpha if rate_hz is None else rate_matched_alpha(
            cfg.low_pass_alpha, cfg.low_pass_rate_hz, rate_hz)
        self.reset()

    def reset(self) -> None:
        self.projected = np.zeros(_LEN_PROJ, dtype=bool)
        self.last_q = self.limits.mean(axis=1)
        self._filtered: np.ndarray | None = None

    def _reference(self, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Projection update (stateful) -> `(reference vectors (6,3), weights (6,))`."""
        cfg = self.cfg
        dist = np.linalg.norm(target[:_LEN_PROJ], axis=1)
        s1 = self.projected[:2]
        s1[dist[:2] < cfg.project_dist] = True
        s1[dist[:2] > cfg.escape_dist] = False
        self.projected[2] = bool(s1[0] and s1[1] and dist[2] <= 0.03)
        ref = target * cfg.scaling
        direction = target[:_LEN_PROJ] / (dist[:, None] + 1e-6)
        proj = direction * np.array([cfg.eta1, cfg.eta1, cfg.eta2])[:, None]
        ref[:_LEN_PROJ] = np.where(self.projected[:, None], proj, ref[:_LEN_PROJ])
        weight = np.ones(_N_VEC)
        weight[:_LEN_PROJ] = np.where(self.projected, [200.0, 200.0, 400.0], 1.0)
        weight[_LEN_PROJ:] = _LEN_PROJ + 3
        return ref, weight

    def objective(self, q: np.ndarray, ref: np.ndarray, weight: np.ndarray, last_q: np.ndarray
                  ) -> tuple[float, np.ndarray]:
        pts, jac = self.fk(q)
        assert jac is not None
        robot = pts[_TASK] - pts[_ORIGIN]
        diff = robot - ref
        dist = np.linalg.norm(diff, axis=1)
        loss, dloss = _smooth_l1(dist, self.cfg.huber_delta)
        w = weight / _N_VEC
        unit = diff / np.maximum(dist, 1e-12)[:, None]
        g_vec = (w * dloss)[:, None] * unit                                  # dL/d robot vector, (6,3)
        jac_vec = jac[_TASK] - jac[_ORIGIN]                                  # (6,3,7)
        grad = np.einsum("vk,vkj->j", g_vec, jac_vec) + 2.0 * self.cfg.norm_delta * (q - last_q)
        value = float(np.sum(w * loss) + self.cfg.norm_delta * np.sum((q - last_q) ** 2))
        return value, grad

    def solve(self, target: np.ndarray, init: np.ndarray | None = None) -> np.ndarray:
        """One frame: `(7,)` unfiltered joints for `(6,3)` target vectors; updates the projection and warm start."""
        ref, weight = self._reference(np.asarray(target, dtype=np.float64))
        last = np.clip(self.last_q, self.limits[:, 0], self.limits[:, 1])
        x0 = last if init is None else np.clip(init, self.limits[:, 0], self.limits[:, 1])
        res = minimize(self.objective, x0, args=(ref, weight, last), jac=True, method="L-BFGS-B",
                       bounds=self.limits, options={"maxiter": self.cfg.max_iter, "ftol": 1e-15, "gtol": 1e-10})
        q = np.clip(res.x, self.limits[:, 0], self.limits[:, 1])
        self.last_q = q
        return q

    def step(self, target: np.ndarray) -> np.ndarray:
        """One frame of the teleop loop: solve, then low-pass (the returned command)."""
        q = self.solve(target)
        self._filtered = q.copy() if self._filtered is None else self._filtered + self.alpha * (q - self._filtered)
        return self._filtered.copy()

    def retarget_sequence(self, vectors: np.ndarray, valid: np.ndarray | None = None, low_pass: bool = True
                          ) -> np.ndarray:
        """`(T,7)` joints for `(T,6,3)` target vectors; invalid rows repeat the previous output (the first valid one
        before it)."""
        t = vectors.shape[0]
        mask = np.ones(t, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
        out = np.full((t, 7), np.nan)
        for k in np.flatnonzero(mask):
            out[k] = self.step(vectors[k]) if low_pass else self.solve(vectors[k])
        return _fill_rows(out)


def _fill_rows(q: np.ndarray) -> np.ndarray:
    """Forward-fill NaN rows, then back-fill the leading ones. Raises ValueError when every row is NaN."""
    ok = np.isfinite(q).all(axis=1)
    if not ok.any():
        raise ValueError("no valid row to retarget")
    idx = np.where(ok, np.arange(len(q)), -1)
    idx = np.maximum.accumulate(idx)
    idx[idx < 0] = int(np.flatnonzero(ok)[0])
    return q[idx]
