"""Open-loop rollout of stored controls through the quasi-dynamic step (a fidelity probe).

One QD step per control row (h = the executor's sim dt by default). The executor's termination rule is applied:
after every block of `terminate_every` rows, an object below `terminate_z` ends the rollout.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fastdo.physics.qd.model import QDModel, step

__all__ = ["QDRollout", "replay"]


@dataclass(eq=False)
class QDRollout:
    qpos: np.ndarray               # (T, nq) state after each row
    ncon: np.ndarray               # (T,) contacts in the step's query
    nrows: np.ndarray              # (T,) QP inequality rows
    iters: np.ndarray              # (T,) IPM iterations
    status: list[str]
    t_query: np.ndarray
    t_assemble: np.ndarray
    t_solve: np.ndarray
    terminated_at: int | None = None


def replay(qm: QDModel, q0: np.ndarray, ctrls: np.ndarray, obj_z_adr: int, s0: int = 0,
           terminate_every: int = 100, terminate_z: float = -1.0) -> QDRollout:
    """Apply `ctrls` (T, nu) as rows s0 .. s0+T-1 from q0 (the state before row s0)."""
    T = int(ctrls.shape[0])
    nq = qm.m.nq
    out = QDRollout(qpos=np.zeros((T, nq)), ncon=np.zeros(T, int), nrows=np.zeros(T, int), iters=np.zeros(T, int),
                    status=[], t_query=np.zeros(T), t_assemble=np.zeros(T), t_solve=np.zeros(T))
    q = np.asarray(q0, np.float64).copy()
    u32 = np.asarray(ctrls, np.float64).astype(np.float32).astype(np.float64)
    for s in range(T):
        r = step(qm, q, u32[s])
        q = r.q_next
        out.qpos[s] = q
        out.ncon[s] = r.contacts.dist.size
        out.nrows[s] = r.qp.G.shape[0]
        out.iters[s] = r.sol.iterations
        out.status.append(r.sol.status)
        out.t_query[s], out.t_assemble[s], out.t_solve[s] = r.t_query, r.t_assemble, r.t_solve
        if not np.isfinite(q).all():
            out.terminated_at = s + 1
            break
        if (s0 + s + 1) % terminate_every == 0 and q[obj_z_adr] < terminate_z:
            out.terminated_at = s + 1
            break
    if out.terminated_at is not None:
        k = out.terminated_at
        for name in ("qpos", "ncon", "nrows", "iters", "t_query", "t_assemble", "t_solve"):
            setattr(out, name, getattr(out, name)[:k])
    return out
