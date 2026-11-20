"""Contact-set stage of the transfer: where a certified trajectory carried its
object, transported onto the few sites a Dex3 can touch, and whether those sites can carry the load.

- Surface distance is the GEODESIC on the object mesh, approximated by shortest paths on a 2-RING graph: every edge,
  plus the chord between the two vertices opposite each interior edge (`surface_graph`, `geodesic_matrix`; scipy
  Dijkstra).
  - The heat method needs a library not installed here, so this is the stated fallback.
  - MEASURED 2026-10-17 on icospheres (r = 4 cm, 40 deg arcs): an EDGE-only graph overestimates by 16-18% at 642,
    2,562 and 10,242 vertices alike. That metrication bias does NOT shrink with resolution.
  - The 2-ring graph errs by <= 2.2%, 1.4% and 0.7% respectively.
- `w1_uniform`: the exact W1 distance between two equal-size uniform contact sets, which is a linear assignment.
- `quantise`: W1 QUANTISATION of a weighted contact measure (weights = the certified contact forces) onto k sites
  chosen among the contact vertices. It minimises sum_i w_i min_j d(x_i, s_j), the transport cost to the best k-atom
  measure with free atom weights. Solved exactly by enumerating candidate k-subsets; it refuses more than `max_subsets`.
- `load_share_weights` (certified grasps squeeze ~1,000x the object's weight, so raw normal
  forces measure squeeze, not load): each contact's share of the load is the sum of its pyramid-edge forces in the
  MINIMUM-NORM nonnegative force set that supplies the required wrench, min ||W^T a - w_req||^2 + ridge * ||a||^2 over
  a >= 0 (an NNLS on the ridge-augmented system). Internal squeeze is null-space force and gets no share.
- `site_contacts`: the k quantisation sites as REAL contacts. Each site is represented by the source contact
  geodesically nearest its vertex, at that contact's ORIGINAL coordinates and normal. Vertex ids are only for the
  transport cost. MEASURED 2026-10-17 (oak5e99, 6,674-vertex mesh): snapping contacts to vertices moved them by
  2.7 mm median and cut a barely-closed grasp (Q 0.007) from 39/40 to 22/40 accepted rows. `pad_contacts` keeps every
  source contact within `pad_radius_m` (geodesic) of a site: a fingertip pad realises a patch, not a point.
- `accept_sites`: the friction-pyramid wrenches at the sites (`force_closure`), the signed quality Q and the
  static-hold residual against the required wrench. `accepted` iff the residual is <= `SiteConfig.max_residual`:
  T1's rule is that the sites must supply the certified load. Q is reported beside it.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from math import comb

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import dijkstra

from fastdo.robot.force_closure import contact_wrenches, signed_quality, static_hold_residual

__all__ = ["SiteConfig", "accept_sites", "geodesic_matrix", "load_share_weights", "pad_contacts", "quantise",
           "site_contacts", "surface_graph", "w1_uniform"]


@dataclass(frozen=True)
class SiteConfig:
    mu: float = 0.5                  # mu' = MuJoCo friction 1.0 / s_mu 2.0, as in `grasp_refine`
    char_len_m: float = 0.03         # torque unit (about the half-size of a YCB object)
    max_residual: float = 1e-3       # the sites must supply the required wrench to this fraction
    ridge: float = 1e-6              # load-share regulariser (selects the minimum-norm balancing forces)


def surface_graph(vertices: np.ndarray, faces: np.ndarray) -> csr_matrix:
    """`(V, V)` symmetric sparse 2-ring graph (edges + the chords across each interior edge), weighted by length."""
    import trimesh

    f = np.asarray(faces, dtype=np.int64)
    mesh = trimesh.Trimesh(np.asarray(vertices, float), f, process=False)
    e = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]], mesh.face_adjacency_unshared])
    e = np.unique(np.sort(e, axis=1), axis=0)
    v = np.asarray(vertices, float)
    w = np.linalg.norm(v[e[:, 0]] - v[e[:, 1]], axis=1)
    n = len(v)
    g = coo_matrix((np.concatenate([w, w]), (np.concatenate([e[:, 0], e[:, 1]]), np.concatenate([e[:, 1], e[:, 0]]))),
                   shape=(n, n))
    return g.tocsr()


def geodesic_matrix(graph: csr_matrix, src: list[int] | np.ndarray, tgt: list[int] | np.ndarray) -> np.ndarray:
    """`(len(src), len(tgt))` graph-geodesic distances."""
    d = dijkstra(graph, directed=False, indices=np.asarray(src, dtype=np.int64))
    return np.asarray(d[:, np.asarray(tgt, dtype=np.int64)], float)


def w1_uniform(cost: np.ndarray) -> float:
    """Exact W1 between two equal-size uniform sets (the mean cost of the optimal assignment)."""
    c = np.asarray(cost, float)
    if c.shape[0] != c.shape[1]:
        raise ValueError("w1_uniform needs equal-size sets")
    r, k = linear_sum_assignment(c)
    return float(c[r, k].mean())


def quantise(graph: csr_matrix, src: list[int] | np.ndarray, weights: np.ndarray, k: int,
             max_subsets: int = 200_000, unreachable_m: float = 1.0) -> tuple[list[int], float]:
    """W1 quantisation of the weighted contact measure on `src` onto `k` sites among `src` (module docstring).
    Pairs in different connected parts of the mesh (several geoms concatenated) cost `unreachable_m`.
    Returns `(site vertex ids, transport cost / total weight)`."""
    src = np.asarray(src, dtype=np.int64)
    uniq = np.unique(src)
    w = np.asarray(weights, float)
    if len(w) != len(src) or (w < 0).any() or w.sum() <= 0:
        raise ValueError("weights must be non-negative, one per source contact, not all zero")
    if k >= len(uniq):
        return [int(u) for u in uniq], 0.0
    if comb(len(uniq), k) > max_subsets:
        raise ValueError(f"C({len(uniq)}, {k}) candidate site sets exceed max_subsets={max_subsets}")
    d = geodesic_matrix(graph, src, uniq)                            # (n_src, n_candidates)
    d[~np.isfinite(d)] = unreachable_m
    best, best_cost = None, np.inf
    for sub in itertools.combinations(range(len(uniq)), k):
        c = float(w @ d[:, list(sub)].min(axis=1))
        if c < best_cost:
            best, best_cost = sub, c
    return [int(uniq[j]) for j in best], best_cost / float(w.sum())


def load_share_weights(points: np.ndarray, normals: np.ndarray, com: np.ndarray, w_req: np.ndarray,
                       cfg: SiteConfig | None = None) -> np.ndarray:
    """`(n,)` load share (N) of each contact: its pyramid-edge forces summed, in the minimum-norm nonnegative force
    set that balances `w_req` (module docstring)."""
    from scipy.optimize import nnls

    cfg = SiteConfig() if cfg is None else cfg
    w = contact_wrenches(points, normals, com, cfg.mu, cfg.char_len_m)          # (4n, 6) unit-force wrenches
    if len(w) == 0:
        return np.zeros(0)
    a_mat = np.vstack([w.T, np.sqrt(cfg.ridge) * np.eye(len(w))])
    b = np.concatenate([np.asarray(w_req, float), np.zeros(len(w))])
    a, _ = nnls(a_mat, b)
    return a.reshape(-1, 4).sum(axis=1)


def site_contacts(graph: csr_matrix, src_vid: np.ndarray, sites: list[int], unreachable_m: float = 1.0) -> np.ndarray:
    """`(k,)` index into the source contacts representing each site: the contact geodesically nearest the site."""
    d = geodesic_matrix(graph, src_vid, sites)
    d[~np.isfinite(d)] = unreachable_m
    return d.argmin(axis=0)


def pad_contacts(graph: csr_matrix, src_vid: np.ndarray, sites: list[int], pad_radius_m: float,
                 unreachable_m: float = 1.0) -> np.ndarray:
    """`(n,)` bool: the source contacts within `pad_radius_m` (geodesic) of some site."""
    d = geodesic_matrix(graph, src_vid, sites)
    d[~np.isfinite(d)] = unreachable_m
    return d.min(axis=1) <= pad_radius_m


def accept_sites(points: np.ndarray, normals: np.ndarray, com: np.ndarray, w_req: np.ndarray,
                 cfg: SiteConfig | None = None) -> dict:
    """{accepted, quality, static_hold_residual} of contacts at `points` pushing along `normals` (into the object)."""
    cfg = SiteConfig() if cfg is None else cfg
    w = contact_wrenches(points, normals, com, cfg.mu, cfg.char_len_m)
    res = static_hold_residual(w, w_req)
    return {"accepted": bool(res <= cfg.max_residual), "quality": signed_quality(w), "static_hold_residual": res}
