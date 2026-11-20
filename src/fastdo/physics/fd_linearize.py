"""D4' linearisation: exact-engine finite-difference transition Jacobians along a nominal rollout.

For every executed row s, MuJoCo's `mjd_transitionFD` at the recorded full integration state x_s (with the row's
weld / gravity phase) gives A_s = d x_{s+1} / d x_s and B_s = d x_{s+1} / d u_s, x = (q-tangent, qvel) (na = 0).
The rows are independent once the nominal states are known, so they are split into contiguous chunks over a
process pool (fork; each worker holds its own model copy). Chaining the (A_s, B_s) gives the trajectory's
sensitivity to the controls; that chain was measured against whole-rollout finite differences at 1.8e-5 relative
error (centered) and 1.3e-4 (forward) on a 50-step contact-rich segment.

Cost per row: forward differences 2 nv + nu + 1 one-step simulations, centered 2 (2 nv + nu) + 1.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import mujoco
import numpy as np

from fastdo.physics.fd_problem import STATE_SPEC, ExecutorSpec, Rollout, set_phase

__all__ = ["FDPool", "Linearization", "fd_steps_per_row", "linearize"]

_W: dict = {}


@dataclass(eq=False)
class Linearization:
    A: np.ndarray            # (T, nx, nx)
    B: np.ndarray            # (T, nx, nu)
    wall_s: float
    cpu_s: float             # summed worker process CPU time
    fd_steps: int            # one-step simulations spent


def fd_steps_per_row(m: mujoco.MjModel, centered: bool) -> int:
    k = 2 * m.nv + m.na + m.nu
    return (2 * k if centered else k) + 1


def _init(ex: ExecutorSpec) -> None:
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    _W["ex"] = ex
    _W["m"] = ex.model
    _W["d"] = mujoco.MjData(ex.model)


def _chunk(steps: np.ndarray, states: np.ndarray, eps: float, centered: bool) -> tuple:
    t0 = time.process_time()
    ex, m, d = _W["ex"], _W["m"], _W["d"]
    nx = 2 * m.nv + m.na
    A = np.zeros((len(steps), nx, nx))
    B = np.zeros((len(steps), nx, m.nu))
    for k, s in enumerate(steps):
        set_phase(ex, m, d, int(s))
        mujoco.mj_setState(m, d, states[k], STATE_SPEC)
        mujoco.mjd_transitionFD(m, d, eps, int(centered), A[k], B[k], None, None)
    return A, B, time.process_time() - t0


class FDPool:
    """A process pool bound to one executor problem (workers = 0 runs in-process, for tests)."""

    def __init__(self, ex: ExecutorSpec, workers: int, nice: int = 10) -> None:
        self.ex = ex
        self.workers = int(workers)
        self._pool = None
        if self.workers > 0:
            ctx = mp.get_context("fork")
            self._pool = ProcessPoolExecutor(max_workers=self.workers, mp_context=ctx, initializer=_niced_init,
                                             initargs=(ex, nice))
        else:
            _init(ex)

    def map_chunks(self, steps: np.ndarray, states: np.ndarray, eps: float, centered: bool) -> list[tuple]:
        if self._pool is None:
            return [_chunk(steps, states, eps, centered)]
        n = max(1, self.workers * 2)
        parts = np.array_split(np.arange(len(steps)), n)
        futs = [self._pool.submit(_chunk, steps[p], states[p], eps, centered) for p in parts if len(p)]
        return [f.result() for f in futs]

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

    def __enter__(self) -> "FDPool":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _niced_init(ex: ExecutorSpec, nice: int) -> None:
    try:
        os.nice(int(nice))
    except OSError:
        pass
    _init(ex)


def linearize(pool: FDPool, ro: Rollout, eps: float = 1e-5, centered: bool = False) -> Linearization:
    """(A_s, B_s) for every executed row of the nominal rollout `ro`."""
    t0 = time.perf_counter()
    T = ro.states.shape[0]
    steps = ro.s0 + np.arange(T)
    res = pool.map_chunks(steps, ro.states, eps, centered)
    A = np.concatenate([r[0] for r in res], axis=0)
    B = np.concatenate([r[1] for r in res], axis=0)
    cpu = float(sum(r[2] for r in res))
    return Linearization(A=A, B=B, wall_s=time.perf_counter() - t0, cpu_s=cpu,
                         fd_steps=T * fd_steps_per_row(pool.ex.model, centered))
