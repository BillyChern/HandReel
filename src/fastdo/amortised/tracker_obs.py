"""Environment v2, pure part: reference table, compressed record, observation, reward and training termination.

`fastdo.amortised.tracker_env` runs them on the CPU backend.

Row convention (the registered scorer's): the state after executing control row r,
x_{r+1}, is "executed row r" and is compared with reference row r (DO AS I DO's own per-step reward compares
x_{r+1} with reference row r + 1, one row later; the scorer is what success is judged by, so it governs here). A world
at row s has executed s rows, so its state is executed row t = s - 1; its observation reads reference row t (clamped
to 0 at the start state), the look-ahead rows t + j k, j = 1..K, and the warm start's control ctrl_ref[t + k] (the
last control row of the coming step); every reference index is clamped to the clip's rows.

Reference table (`RefTable`): per clip, the rows of the object reference T^ref (position, wxyz), its velocity
(free-joint qvel: world linear, body angular), the robot reference q^ref, the warm start `ctrl_ref`, the
(a)-corrected kinematic robot fingertips and their SDF distance d^ref_f, the reliability kappa; per clip W, T, 64
object surface points (body frame), mass and friction. Clips are stacked row-wise with offsets (the layout of a
device-resident gather table).

Record (`record_fields`): the compressed per-transition state the host replay stores: qpos fp32,
qvel fp16, the wrist pose fp32 (7; FK is not repeated at rebuild), fingertip positions fp16 (F, 3), fingertip SDF
distance and unit gradient fp16 (F, 4; world frame), contact scalars fp16 (max hand-object penetration, hand-object
touch, object-pedestal touch), the previous residual target fp16 (R), the assist level fp16, clip id and row int32,
and for the asymmetric critic per-fingertip normal force fp16 (F) and the exact object velocity fp32 (6). The
environment computes its observation FROM the compressed record (`observe(table, record)`), so the replay's rebuild
from stored records is bitwise the observation the policy saw; every operation is element-wise (no reduction whose
order depends on the batch), so a row's observation does not depend on the batch it is computed in. Size for Sharpa
(nq 35, nv 34, R 28, F 5): 412 B per record plus 124 B of action (fp32), reward and flags in `HostReplay` = 536 B per
transition, not the earlier ≈ 350 B estimate (the wrist pose, the critic's exact object velocity and an fp32 action
were not in it); 50M transitions = 26.8 GB of host memory.

The row convention is checked on a real clip's PRODUCTION table (no test-only row shift): a state equal to the
kinematic reference row s shows exactly one row of reference motion as error. Record size, Lambda and the scales are
implementation choices.

Reward per executed row r (kappa, tips and d^ref at reference row r):
    r = kappa [w_o exp(-|rho|/s_p - theta/s_th) + w_h exp(-e_tip/s_h) + w_q exp(-|q - q^ref|^2_Lambda)]
        + w_c (1/F) sum_f exp(-max(0, d_f - d^ref_f)/s_c) - w_pen max(0, pen - m_pen)/m_pen,
plus -w_a |delta_k - delta_{k-1}|^2 per policy step (= the per-row term summed over the step's k rows).
Lambda = diag(1/s^2) with s = `q_scale` per group (wrist position, wrist rotation, fingers): an implementation choice,
the per-group scales the transductive observation (`fastdo.transductive.obs`) uses.

Training termination (`TermState`): TRUE terminations are the executor's own (drop, invalid state), the
clip end, the ENVELOPE (|rho| > 0.06 m or theta > 45 deg on `env_rows` consecutive rows) and FUTILITY (the running sum
of the scorer's position error |p - p^ref| over the scoring window [W + g, T) exceeds 0.03 |window|: the window mean
can then no longer pass SR_MT's 3 cm, so a passable episode is never ended). Evaluation and certification use only
the executor's termination and the clip end (`evaluation=True`).

Envelope scope (the rule was fixed before the data were seen): the envelope's object term on rows < W (the welded
warmup) is `env_pos` / `env_rot` (unchanged); on rows >= W it is the POST-RELEASE term `env_post_pos` /
`env_post_rot`, set from the rule's measurement on the 8 informative MPC successes, or None = DROPPED. The measurement
gave max(0.06 m, 1.25 x 0.4745 m) = 0.593 m and max(45 deg, 1.25 x 111.47 deg) = 139.3 deg, both above the fallback's
0.25 m / 90 deg, so the post-release object term is DROPPED (both None): after the release, training ends only by
the clip end, futility or the executor.
The envelope counter runs over consecutive rows whatever term judged each row. The certificate (CPU, clip end) is
unchanged: the envelope affects only training truncation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from fastdo.amortised.se3 import object_error, quat_conj, quat_rotate

__all__ = ["RefTable", "TermState", "TrackerConfig", "obs_dims", "observe", "record_fields", "row_rewards",
           "step_penalty"]


@dataclass(frozen=True)
class TrackerConfig:
    """Environment v2 settings (initial values; TUNE only)."""

    k: int = 5                                        # rows per policy step
    delta_max: tuple[float, float, float] = (0.05, 0.10, 0.40)
    horizon: int = 10                                 # K look-ahead offsets of k rows
    w_o: float = 1.0
    w_h: float = 0.3
    w_q: float = 0.2
    w_c: float = 0.5
    w_pen: float = 0.5
    w_a: float = 0.01
    s_p: float = 0.05
    s_th: float = 0.5
    s_h: float = 0.03
    s_c: float = 0.01
    m_pen: float = 0.002
    q_scale: tuple[float, float, float] = (0.05, 0.3, 0.3)
    env_pos: float = 0.06
    env_rot: float = math.radians(45.0)
    env_rows: int = 10
    env_post_pos: float | None = None                 # rows >= W; None = the object term is DROPPED
    env_post_rot: float | None = None                 # (radians) the measured bound exceeded 0.25 m / 90 deg
    fut_pos: float = 0.03                             # SR_MT's object position threshold
    grace: int = 50                                   # the scorer's post-warmup grace g
    obs_clip: float = 10.0
    pts_scale: float = 0.1
    sdf_scale: float = 0.02
    force_scale: float = 10.0

    def __post_init__(self) -> None:
        if (self.env_post_pos is None) != (self.env_post_rot is None):
            raise ValueError("the post-release envelope term needs both bounds or neither (None = dropped)")


def _group(n: int, vals: tuple[float, float, float]) -> np.ndarray:
    s = np.full(n, vals[2], np.float64)
    s[:3], s[3:6] = vals[0], vals[1]
    return s


@dataclass(frozen=True, eq=False)
class RefTable:
    """Per-clip references stacked row-wise (module docstring). Row arrays have N = sum of the clips' rows."""

    offset: np.ndarray        # (C,) int64 first row of each clip
    n_rows: np.ndarray        # (C,) rows of each clip
    W: np.ndarray             # (C,) release row
    T: np.ndarray             # (C,) executed rows of a full run
    obj_p: np.ndarray         # (N, 3)
    obj_q: np.ndarray         # (N, 4) wxyz
    obj_v: np.ndarray         # (N, 6) free-joint qvel (world linear, body angular)
    q_ref: np.ndarray         # (N, R)
    ctrl_ref: np.ndarray      # (N, R)
    tips_ref: np.ndarray      # (N, F, 3) world
    dref: np.ndarray          # (N, F)
    kappa: np.ndarray         # (N,)
    points: np.ndarray        # (C, P, 3) object surface points, body frame
    mass: np.ndarray          # (C,)
    friction: np.ndarray      # (C,)

    @property
    def R(self) -> int:
        return int(self.q_ref.shape[1])

    @property
    def F(self) -> int:
        return int(self.tips_ref.shape[1])

    @property
    def P(self) -> int:
        return int(self.points.shape[1])

    @staticmethod
    def stack(clips: list[dict[str, np.ndarray | float | int]]) -> RefTable:
        """From per-clip dicts with the row arrays (obj_p ... kappa) and W, T, points, mass, friction."""
        rows = ("obj_p", "obj_q", "obj_v", "q_ref", "ctrl_ref", "tips_ref", "dref", "kappa")
        n = np.array([np.asarray(c["obj_p"]).shape[0] for c in clips], np.int64)
        for c, m in zip(clips, n, strict=True):
            for k in rows:
                if np.asarray(c[k]).shape[0] != m:
                    raise ValueError(f"{k}: {np.asarray(c[k]).shape[0]} rows, expected {m}")
            if m < int(c["T"]):
                raise ValueError(f"reference has {m} rows < T = {int(c['T'])}")
        cat = {k: np.concatenate([np.asarray(c[k], np.float64) for c in clips]) for k in rows}
        return RefTable(offset=np.concatenate([[0], np.cumsum(n)[:-1]]).astype(np.int64), n_rows=n,
                        W=np.array([int(c["W"]) for c in clips], np.int64),
                        T=np.array([int(c["T"]) for c in clips], np.int64), **cat,
                        points=np.stack([np.asarray(c["points"], np.float64) for c in clips]),
                        mass=np.array([float(c["mass"]) for c in clips]),
                        friction=np.array([float(c["friction"]) for c in clips]))

    def idx(self, clip: np.ndarray, row: np.ndarray) -> np.ndarray:
        """Table row of (clip, reference row), the row clamped to the clip's last."""
        return self.offset[clip] + np.minimum(np.maximum(row, 0), self.n_rows[clip] - 1)


def record_fields(nq: int, nv: int, R: int, F: int) -> dict[str, tuple[tuple[int, ...], type]]:
    """The compressed record (module docstring): name -> (shape per world, dtype)."""
    return {"qpos": ((nq,), np.float32), "qvel": ((nv,), np.float16), "wrist": ((7,), np.float32),
            "tips": ((F, 3), np.float16), "tip_sdf": ((F, 4), np.float16), "contact": ((3,), np.float16),
            "prev_delta": ((R,), np.float16), "alpha": ((), np.float16), "clip": ((), np.int32),
            "row": ((), np.int32), "tip_force": ((F,), np.float16), "obj_vel": ((6,), np.float32)}


def obs_dims(cfg: TrackerConfig, R: int, F: int, P: int) -> tuple[int, int]:
    """(actor, critic) observation widths."""
    K = cfg.horizon
    actor = 12 + 3 * R + K * (6 + R) + 6 * P + 5 * F + 3 + R + 2 + 1 + K
    return actor, actor + 2 + F + 6


def observe(table: RefTable, rec: dict[str, np.ndarray], cfg: TrackerConfig) -> tuple[np.ndarray, np.ndarray]:
    """(actor, critic) float32 observations of a batch of records (B, ...) (module docstring). Element-wise only."""
    R, K, k = table.R, cfg.horizon, cfg.k
    c = np.asarray(rec["clip"], np.int64)
    s = np.asarray(rec["row"], np.int64)
    t = s - 1                                                             # executed row of the state (convention)
    qp = np.asarray(rec["qpos"], np.float64)
    qv = np.asarray(rec["qvel"], np.float64)
    q, p, qo = qp[:, :R], qp[:, R:R + 3], qp[:, R + 3:R + 7]
    wr = np.asarray(rec["wrist"], np.float64)
    pw, qwi = wr[:, :3], quat_conj(wr[:, 3:7])
    to_wrist = lambda x: quat_rotate(qwi[:, None], x - pw[:, None])
    js, ds = _group(R, cfg.q_scale), _group(R, cfg.delta_max)
    vs = _group(R, (1.0, 5.0, 10.0))
    i0 = table.idx(c, t)
    rho, om = object_error(p, qo, table.obj_p[i0], table.obj_q[i0])
    ov = qv[:, R:R + 6]
    vr = table.obj_v[i0]
    v_err = quat_rotate(quat_conj(table.obj_q[i0]), ov[:, :3] - vr[:, :3])
    w_err = ov[:, 3:6] - vr[:, 3:6]
    parts = [rho / cfg.s_p, om / cfg.s_th, v_err, w_err / 5.0, (q - table.q_ref[i0]) / js, qv[:, :R] / vs,
             (table.ctrl_ref[table.idx(c, t + k)] - q) / js]
    for j in range(1, K + 1):
        ij = table.idx(c, t + j * k)
        fr, fo = object_error(table.obj_p[ij], table.obj_q[ij], p, qo)    # xi(T_t^-1 T^ref_{t+jk})
        parts += [fr / cfg.s_p, fo / cfg.s_th, (table.q_ref[ij] - q) / js]
    pts = table.points[c]                                                 # (B, P, 3) body frame
    now = p[:, None] + quat_rotate(qo[:, None], pts)
    i1 = table.idx(c, t + k)
    nxt = table.obj_p[i1][:, None] + quat_rotate(table.obj_q[i1][:, None], pts)
    parts += [to_wrist(now).reshape(len(c), -1) / cfg.pts_scale, to_wrist(nxt).reshape(len(c), -1) / cfg.pts_scale]
    ts = np.asarray(rec["tip_sdf"], np.float64)
    grad_w = quat_rotate(qwi[:, None], ts[..., 1:4])
    parts += [ts[..., 0] / cfg.sdf_scale, grad_w.reshape(len(c), -1), table.dref[i0] / cfg.sdf_scale]
    ct = np.asarray(rec["contact"], np.float64)
    W, T = table.W[c].astype(np.float64), table.T[c].astype(np.float64)
    kap = np.stack([table.kappa[table.idx(c, t + j * k)] for j in range(1, K + 1)], 1)
    parts += [np.stack([ct[:, 0] / 0.005, ct[:, 1], ct[:, 2]], 1), np.asarray(rec["prev_delta"], np.float64) / ds,
              np.stack([np.maximum(W - s, 0) / np.maximum(W, 1.0), s / T], 1),
              np.asarray(rec["alpha"], np.float64)[:, None], kap]
    actor = np.clip(np.concatenate([np.asarray(x, np.float64).reshape(len(c), -1) for x in parts], 1),
                    -cfg.obs_clip, cfg.obs_clip)
    priv = [table.mass[c][:, None], table.friction[c][:, None],
            np.asarray(rec["tip_force"], np.float64) / cfg.force_scale, np.asarray(rec["obj_vel"], np.float64)]
    critic = np.concatenate([actor, np.clip(np.concatenate(priv, 1), -cfg.obs_clip, cfg.obs_clip)], 1)
    return actor.astype(np.float32), critic.astype(np.float32)


def row_rewards(table: RefTable, cfg: TrackerConfig, clip: int, rows: np.ndarray, qpos: np.ndarray,
                tips: np.ndarray, d_tip: np.ndarray, pen: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(reward, |rho|, theta) of executed rows `rows` (n,) with post-row states qpos (n, nq), fingertips (n, F, 3),
    their SDF distances (n, F) and the rows' max penetration (n,) (module docstring)."""
    R = table.R
    i = table.idx(np.full(rows.shape, clip), rows)
    rho, om = object_error(qpos[:, R:R + 3], qpos[:, R + 3:R + 7], table.obj_p[i], table.obj_q[i])
    rn, th = np.sqrt((rho * rho).sum(-1)), np.sqrt((om * om).sum(-1))
    e_tip = np.sqrt(((tips - table.tips_ref[i]) ** 2).sum(-1)).mean(-1)
    lam = 1.0 / _group(R, cfg.q_scale) ** 2
    dq2 = ((qpos[:, :R] - table.q_ref[i]) ** 2 * lam).sum(-1)
    track = (cfg.w_o * np.exp(-rn / cfg.s_p - th / cfg.s_th) + cfg.w_h * np.exp(-e_tip / cfg.s_h)
             + cfg.w_q * np.exp(-dq2))
    contact = cfg.w_c * np.exp(-np.maximum(0.0, d_tip - table.dref[i]) / cfg.s_c).mean(-1)
    pen_cost = cfg.w_pen * np.maximum(0.0, pen - cfg.m_pen) / cfg.m_pen
    return table.kappa[i] * track + contact - pen_cost, rn, th


def step_penalty(cfg: TrackerConfig, delta: np.ndarray, delta_prev: np.ndarray) -> np.ndarray:
    """-w_a |delta_k - delta_{k-1}|^2 per policy step, the residual targets normalised by delta_max."""
    ds = _group(delta.shape[-1], cfg.delta_max)
    return -cfg.w_a * (((delta - delta_prev) / ds) ** 2).sum(-1)


@dataclass
class TermState:
    """Per-world envelope counter and futility sum (module docstring); `reset(i)` at every episode start."""

    n: int
    env_count: np.ndarray = field(init=False)
    pos_sum: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.env_count = np.zeros(self.n, np.int64)
        self.pos_sum = np.zeros(self.n)

    def reset(self, i: int) -> None:
        self.env_count[i], self.pos_sum[i] = 0, 0.0

    def rows(self, i: int, cfg: TrackerConfig, rows: np.ndarray, rho_n: np.ndarray, theta: np.ndarray,
             p_err: np.ndarray, W: int, T: int, evaluation: bool) -> tuple[int, str | None]:
        """Feed executed rows in order; (rows consumed, reason) -- the episode ends after the returned count when
        a reason is given. `p_err` is the scorer's |p - p^ref| of each row."""
        lo, n_win = W + cfg.grace, max(T - W - cfg.grace, 1)
        bound = cfg.fut_pos * n_win * (1.0 + 1e-9)        # float slack: never below the scorer's own mean
        for j, r in enumerate(rows):
            if r + 1 >= T:
                return j + 1, "clip_end"
            if evaluation:
                continue
            if r < W:                                     # the welded warmup, as registered
                out = rho_n[j] > cfg.env_pos or theta[j] > cfg.env_rot
            elif cfg.env_post_pos is None:                # the post-release object term is dropped
                out = False
            else:
                out = rho_n[j] > cfg.env_post_pos or theta[j] > cfg.env_post_rot
            self.env_count[i] = self.env_count[i] + 1 if out else 0
            if self.env_count[i] >= cfg.env_rows:
                return j + 1, "envelope"
            if r >= lo:
                self.pos_sum[i] += p_err[j]
                if self.pos_sum[i] > bound:
                    return j + 1, "futile"
        return len(rows), None
