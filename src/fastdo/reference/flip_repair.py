"""Flip repair (design 6, R1.8): exact Viterbi over the discrete symmetry representatives.

Nodes are the rows with a finite pose in time order; a consolidated rest segment is ONE node (its rows share a label).
Labels are group elements g; a node's candidate rotation is O_t S_g. Cost:

    E(g) = sum_nodes mu * size * 1[g != e] + sum_consecutive angle(O_t S_g, O_t' S_g')^2 / gap(t, t'),

mu = (10 deg)^2 in rad^2, gap = the row distance between the nodes. Activation: G non-trivial and some step between
consecutive finite rows > 90 deg. The pair term uses the object's own rotation only (no hand, no rigidity), and a
relabelling moves the rendered surface by at most the element's mismatch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fastdo.reference.so3 import angle_deg

__all__ = ["FlipResult", "max_step_deg", "viterbi_labels"]


@dataclass(frozen=True)
class FlipResult:
    active: bool
    labels: np.ndarray        # (n_nodes,) chosen element per node
    cost_before: float
    cost_after: float


def max_step_deg(O: np.ndarray, rows: np.ndarray) -> float:
    """Largest rotation change between consecutive listed rows (0 for fewer than two)."""
    r = np.asarray(rows, dtype=int)
    if len(r) < 2:
        return 0.0
    return float(angle_deg(O[r[:-1]], O[r[1:]]).max())


def viterbi_labels(O_nodes: np.ndarray, sizes: np.ndarray, gaps: np.ndarray, elements: np.ndarray, mu: float
                   ) -> FlipResult:
    """Exact minimiser of the module-docstring energy. `O_nodes (N,3,3)`, `sizes (N,)` rows per node, `gaps (N-1,)`
    row distance between consecutive nodes, `elements (K,3,3)` with elements[0] = identity."""
    N, K = len(O_nodes), len(elements)
    cand = np.einsum("nij,kjl->nkil", O_nodes, elements)               # (N,K,3,3)
    unary = mu * np.asarray(sizes, dtype=np.float64)[:, None] * (np.arange(K)[None, :] != 0)
    cost = unary[0].copy()
    back = np.zeros((N, K), dtype=np.int64)
    for t in range(1, N):
        ang = np.radians(angle_deg(cand[t - 1][:, None], cand[t][None, :]))   # (K_prev, K_cur)
        tot = cost[:, None] + ang ** 2 / float(gaps[t - 1])
        back[t] = np.argmin(tot + 1e-15 * np.arange(K)[:, None], axis=0)
        cost = tot[back[t], np.arange(K)] + unary[t]
    lab = np.zeros(N, dtype=np.int64)
    lab[-1] = int(np.argmin(cost + 1e-15 * np.arange(K)))
    for t in range(N - 1, 0, -1):
        lab[t - 1] = back[t, lab[t]]
    before = float(sum(np.radians(angle_deg(O_nodes[t - 1], O_nodes[t])) ** 2 / gaps[t - 1] for t in range(1, N)))
    return FlipResult(True, lab, before, float(cost.min()))
