"""Per-row reach and orientation margins of a G1 right-arm wrist path (`fastdo.robot.reach` searches with
them). Pure functions plus a process-pool evaluator, CPU only.

For a wrist target path, `solve_arm_ik` (`fastdo.robot.arm_ik`) gives per row the joint solution and its position and
rotation residual; this module adds two margins of the solution (`ik_margins`):
- joint-limit margin: the smallest distance (rad) of any of the 7 joints to its COMMAND limit;
- sigma_min: the smallest singular value of the body Jacobian of `right_dex3_base` with the rotation rows weighted by the
  IK's `rot_weight` (1/sqrt(50) m/rad), i.e. the distance to a singularity in the IK's own metric (m per rad of joint
  motion). Over 3,000 uniform samples of the command box its quantiles are 0.003 / 0.017 / 0.034 / 0.056 / 0.074
  (5 / 25 / 50 / 75 / 95 %; `MarginConfig.sigma_ref` is the median).
and the per-row MARGIN m_t (`row_margins`), which the search maximises in its minimum over the rows:
- a row is FEASIBLE when its position residual is at most `pos_tol` (1 cm), its rotation residual at most `rot_tol`
  (0.1 rad) and its collision clearance (when computed) is not negative; then
  m_t = min(1, jl_margin / jl_ref, sigma_min / sigma_ref, clearance / clearance_ref) in [0, 1];
- otherwise m_t = -max(pos_err / pos_tol, rot_err / rot_tol, -clearance / clearance_ref) <= -1 (a row 3 cm out of
  reach scores -3), so ANY infeasible row ranks a path below every feasible one, and among infeasible paths the one
  with the smaller worst violation wins.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
from collections.abc import Callable
from concurrent.futures import Executor, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from typing import Self, TypeVar

import numpy as np

from fastdo.robot import g1_constants as C
from fastdo.robot.arm_ik import ArmIkConfig, ArmIkResult, IkCollider, _ArmFK, solve_arm_ik
from fastdo.robot.g1_model import G1Upper

logger = logging.getLogger(__name__)
T = TypeVar("T")
R = TypeVar("R")

__all__ = ["IkJob", "IkOutcome", "MarginConfig", "RetryPool", "evaluate_jobs", "ik_margins", "make_pool", "row_margins", "run_job"]


@dataclass(frozen=True)
class MarginConfig:
    pos_tol: float = 0.01                 # m
    rot_tol: float = 0.10                 # rad
    jl_ref: float = 0.20                  # rad: a joint 0.2 rad inside its limit has full margin
    sigma_ref: float = 0.034              # module docstring
    clearance_ref: float = 0.01           # m: the IK collider's d_safe


@dataclass(frozen=True)
class IkJob:
    key: int
    targets: np.ndarray                   # (R,4,4) robot frame
    times: np.ndarray                     # (R,)
    hand_q: np.ndarray                    # (R,7)
    table_z: float | None                 # None: no collider; else the IK model's table top (m)
    ik: ArmIkConfig = field(default_factory=ArmIkConfig)
    table_x: tuple[float, float] = (0.18, 0.78)       # the table rectangle (`replay.SceneConfig`)
    table_y: tuple[float, float] = (-0.60, 0.40)


@dataclass(frozen=True)
class IkOutcome:
    key: int
    q: np.ndarray                         # (R,7)
    pos_err: np.ndarray                   # (R,)
    rot_err: np.ndarray                   # (R,)
    clearance: np.ndarray                 # (R,) NaN without the collider
    jl_margin: np.ndarray                 # (R,)
    sigma_min: np.ndarray                 # (R,)


def ik_margins(g1: G1Upper, q: np.ndarray, rot_weight: float) -> tuple[np.ndarray, np.ndarray]:
    """`(joint-limit margin (R,), sigma_min (R,))` of `(R,7)` right-arm solutions (module docstring)."""
    lim = np.asarray(C.RIGHT_ARM_COMMAND_LIMITS, dtype=np.float64)
    qq = np.atleast_2d(np.asarray(q, dtype=np.float64))
    jl = np.minimum(qq - lim[:, 0], lim[:, 1] - qq).min(axis=1)
    fk = _ArmFK(g1)
    w = np.diag([1.0, 1.0, 1.0] + [rot_weight] * 3)
    sig = np.empty(len(qq))
    for k, row in enumerate(qq):
        fk.set(row)
        sig[k] = np.linalg.svd(w @ fk.body_jac(), compute_uv=False)[-1]
    return jl, sig


def row_margins(pos_err: np.ndarray, rot_err: np.ndarray, jl_margin: np.ndarray, sigma_min: np.ndarray,
                clearance: np.ndarray | None, cfg: MarginConfig) -> np.ndarray:
    """Per-row margin m_t (module docstring); NaN clearance = not computed."""
    pos, rot = np.asarray(pos_err, dtype=np.float64), np.asarray(rot_err, dtype=np.float64)
    clr = np.full(pos.shape, np.nan) if clearance is None else np.asarray(clearance, dtype=np.float64)
    clr_term = np.where(np.isfinite(clr), clr / cfg.clearance_ref, np.inf)
    violation = np.maximum.reduce([pos / cfg.pos_tol, rot / cfg.rot_tol, np.where(np.isfinite(clr), -clr_term, -np.inf)])
    feasible = (pos <= cfg.pos_tol) & (rot <= cfg.rot_tol) & ~(np.isfinite(clr) & (clr < 0.0))
    good = np.minimum.reduce([np.ones_like(pos), np.asarray(jl_margin) / cfg.jl_ref,
                              np.asarray(sigma_min) / cfg.sigma_ref, clr_term])
    return np.where(feasible, np.clip(good, 0.0, 1.0), -np.maximum(violation, 1.0))


_WORKER: dict = {}


def _model(job: IkJob) -> tuple[G1Upper, IkCollider | None]:
    """The worker's models for `job`: the plain model without a table, or ONE collider model per table rectangle
    whose table body is moved to `job.table_z` (a model per height cost ~300 MB each, and the local search visits
    dozens of heights: the first C3 export lost a worker to the container's OOM killer)."""
    import mujoco

    from fastdo.robot.g1_model import load_g1_upper
    from fastdo.robot.replay import SceneConfig, build_ik_model

    if job.table_z is None:
        if None not in _WORKER:
            _WORKER[None] = (load_g1_upper(), None)
        return _WORKER[None]
    key = (tuple(job.table_x), tuple(job.table_y))
    if key not in _WORKER:
        g1 = build_ik_model(SceneConfig(table_x=job.table_x, table_y=job.table_y))
        _WORKER[key] = (g1, IkCollider(g1))
    g1, collider = _WORKER[key]
    assert collider is not None
    thickness = SceneConfig().table_thickness
    for model in (g1.model, collider.model):
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table")
        model.body_pos[body, 2] = float(job.table_z) - thickness / 2
    return g1, collider


def run_job(job: IkJob) -> IkOutcome:
    """Solve one job's IK (collision-aware when `table_z` is set) and its margins."""
    g1, collider = _model(job)
    res: ArmIkResult = solve_arm_ik(g1, job.targets, cfg=job.ik, collider=collider, times=job.times,
                                    hand_q=job.hand_q if collider is not None else None)
    jl, sig = ik_margins(g1, res.q, job.ik.rot_weight)
    return IkOutcome(key=job.key, q=res.q, pos_err=res.pos_err, rot_err=res.rot_err, clearance=res.clearance,
                     jl_margin=jl, sigma_min=sig)


def _init_worker() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch

    torch.set_num_threads(1)


class RetryPool:
    """A spawn process pool of single-threaded workers that survives a killed worker: on `BrokenProcessPool` (a worker
    terminated abruptly, e.g. by the container's OOM killer) the pool is rebuilt and the whole batch re-run, at most
    `retries` times. Jobs are pure functions of their inputs, so a re-run gives the same results."""

    def __init__(self, workers: int, retries: int = 2) -> None:
        self.workers, self.retries, self.rebuilds = workers, retries, 0
        self._ex = self._new()

    def _new(self) -> ProcessPoolExecutor:
        return ProcessPoolExecutor(max_workers=self.workers, mp_context=mp.get_context("spawn"),
                                   initializer=_init_worker)

    def map(self, fn: Callable[[T], R], items: list[T]) -> list[R]:
        for attempt in range(self.retries + 1):
            try:
                return list(self._ex.map(fn, items, chunksize=1))
            except BrokenProcessPool:
                if attempt == self.retries:
                    raise
                logger.warning("IK process pool broke (a worker was killed); rebuilding and re-running %d jobs",
                               len(items))
                self._ex.shutdown(wait=False, cancel_futures=True)
                self._ex = self._new()
                self.rebuilds += 1
        raise AssertionError("unreachable")

    def shutdown(self) -> None:
        self._ex.shutdown()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.shutdown()


def make_pool(workers: int) -> RetryPool | None:
    """A `RetryPool` of `workers` single-threaded workers (None for `workers` <= 1); the caller shuts it down (`with`
    block). Each worker keeps its models across jobs."""
    if workers <= 1:
        return None
    return RetryPool(workers)


def evaluate_jobs(jobs: list[IkJob], pool: RetryPool | Executor | None = None) -> list[IkOutcome]:
    """`run_job` on every job, in order, in `pool` when given (`make_pool`), else in this process."""
    if pool is None or len(jobs) <= 1:
        return [run_job(j) for j in jobs]
    return list(pool.map(run_job, jobs))
