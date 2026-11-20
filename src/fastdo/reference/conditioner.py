"""The reference conditioner: segmentation -> rest consolidation -> flip repair, per window.

OPT-IN, post-reconstruction. Input: the PRE-ego record poses (`solve.npz` / `contact.npz` convention), the raw
camera-frame covariance, confidence, the template, both hands (pre-ego camera frame) and a world source. Output: a
`ConditionedReference` whose `poses` are pre-ego record poses; every row with `changed = False` is byte-identical to the
input. Nothing here writes a reconstruction artifact or changes a production default.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from fastdo.reference import frames as FR
from fastdo.reference import frechet as FM
from fastdo.reference import motion as MT
from fastdo.reference import segmentation as SG
from fastdo.reference.config import ReferenceConfig, constants_hash
from fastdo.reference.flip_repair import max_step_deg, viterbi_labels
from fastdo.reference.so3 import angle_deg
from fastdo.reference.symmetry_group import SymmetryGroup, compute_group

__all__ = ["ConditionedReference", "SegmentReport", "WindowInputs", "condition_window", "consolidate_rows",
           "effective_covariance"]


@dataclass(frozen=True)
class WindowInputs:
    poses: np.ndarray                      # (T,4,4) pre-ego record poses (metric template -> camera)
    times: np.ndarray                      # (T,) seconds
    scale: float                           # template metric scale
    mesh_vertices: np.ndarray              # (V,3) canonical
    mesh_faces: np.ndarray                 # (F,3)
    hands: dict[str, SG.HandInput]         # PRE-ego camera-frame vertices; precomputed distances must be ego-scaled
    world: FR.WorldSource
    confidence: np.ndarray | None = None   # (T,); None -> 1
    covariance: np.ndarray | None = None   # (T,6,6) camera-frame left perturbation; None -> floor only
    alpha: float = 1.0
    group: SymmetryGroup | None = None     # precomputed G (applied tier); None -> compute_group
    window_id: str = ""


@dataclass
class SegmentReport:
    rows: np.ndarray
    n_vote: int
    status: str                            # "consolidated" | "rest_short" | "refused"
    reason: str = ""                       # refusal reason
    p: np.ndarray | None = None
    O: np.ndarray | None = None
    cov: np.ndarray | None = None
    replaced: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    outlier_kept: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    labels: np.ndarray | None = None       # representative per voting row
    vote_rows: np.ndarray | None = None
    inlier_frac: float | None = None
    iterations: int | None = None
    drift: MT.MotionTest | None = None   # rev 2: the noise-calibrated motion test (R2.1)
    spread: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {"rows": [int(self.rows[0]), int(self.rows[-1])], "n_rows": len(self.rows), "n_vote": self.n_vote,
             "status": self.status, "reason": self.reason, "n_replaced": len(self.replaced),
             "n_outlier_kept": len(self.outlier_kept), "inlier_frac": self.inlier_frac, "iterations": self.iterations,
             "spread": self.spread}
        if self.drift is not None:
            d["drift"] = self.drift.to_dict()
        if self.p is not None:
            d["mean_centre_m"] = np.asarray(self.p).tolist()
        return d


@dataclass
class ConditionedReference:
    poses: np.ndarray                      # (T,4,4) conditioned pre-ego record poses (input dtype)
    poses_world: np.ndarray                # (T,4,4) world (O, p - O c*), NaN without a pose
    centres_world: np.ndarray              # (T,3) conditioned world centres
    row_state: np.ndarray                  # (T,) str
    changed: np.ndarray                    # (T,) bool
    symmetry_label: np.ndarray             # (T,) int
    segments: list[SegmentReport]
    group: SymmetryGroup
    world_source: str
    world_reason: str
    flip: dict
    distances: dict[str, np.ndarray]
    c_star: np.ndarray
    sigma_eff: np.ndarray                  # (T,6,6)
    world: FR.WorldSource
    cfg: ReferenceConfig
    flags: tuple[str, ...] = ()

    def summary(self) -> dict:
        states, counts = np.unique(self.row_state, return_counts=True)
        return {"world_source": self.world_source, "world_reason": self.world_reason,
                "row_states": {str(s): int(c) for s, c in zip(states, counts, strict=True)},
                "changed": int(self.changed.sum()), "group": self.group.to_dict(), "flip": self.flip,
                "segments": [s.to_dict() for s in self.segments], "flags": list(self.flags),
                "constants": constants_hash(self.cfg)}

    def save(self, path: str | Path) -> None:
        """`reference_conditioning.npz` + `.json` (never over a reconstruction artifact)."""
        path = Path(path)
        if path.name in ("solve.npz", "contact.npz", "clip_record.npz", "qc.json"):
            raise ValueError(f"refusing to overwrite a reconstruction artifact: {path}")
        np.savez_compressed(path, poses=self.poses, poses_world=self.poses_world, row_state=self.row_state,
                            changed=self.changed, symmetry_label=self.symmetry_label, c_star=self.c_star)
        path.with_suffix(".json").write_text(json.dumps(self.summary(), indent=1, default=float))


def effective_covariance(cov_cam, poses, c_star, ws, alpha, cfg) -> np.ndarray:
    """Sigma_eff = alpha Sigma^w + Sigma_0 (design 4.2) with the R1.1 transport."""
    n = len(poses)
    cov = FR.sanitize_covariance(cov_cam, n, cfg.cov_eig_min, cfg.cov_eig_max)
    P = np.where(np.isfinite(poses), poses, 0.0)
    return alpha * FR.transport_covariance(cov, P, c_star, ws) + FM.floor_covariance(cfg)[None]


def _spread(p, O, pbar, Obar, w, labels, elements) -> dict:
    d = np.linalg.norm(p - pbar, axis=1) * 1e3
    a = angle_deg(np.einsum("nij,njk->nik", O, elements[labels]), Obar[None])
    ww = np.asarray(w) / max(float(np.sum(w)), 1e-300)
    return {"centre_wrms_mm": float(np.sqrt(ww @ d ** 2)), "centre_p95_mm": float(np.percentile(d, 95)),
            "centre_max_mm": float(d.max()), "angle_wrms_deg": float(np.sqrt(ww @ a ** 2)),
            "angle_p95_deg": float(np.percentile(a, 95)), "angle_max_deg": float(a.max())}


def _cap_mask(p_rows, O_rows, pbar, Obar, elements, cfg) -> np.ndarray:
    """Action cap (R1.6): a row is replaced only if its centre is within `cap_pos_m` AND its QUOTIENT angle is within
    `cap_rot_deg` of the mean."""
    dist = np.linalg.norm(p_rows - pbar, axis=1)
    ang = FM.quotient_angle_deg(O_rows, Obar, elements)
    return (dist <= cfg.cap_pos_m) & (ang <= cfg.cap_rot_deg)


def consolidate_rows(rows, p, O, sigma_eff, conf, times, elements, cfg, estimator="cond") -> SegmentReport:
    """Rest consolidation of one segment `rows` (with the drift refusal). `estimator` "cond" (robust weighted mean)
    or "ctrl" (same refusals and representatives as cond, median / SVD mean)."""
    rows = np.asarray(rows, dtype=int)
    vote = rows[np.asarray(conf)[rows] >= cfg.min_confidence]
    rep = SegmentReport(rows=rows, n_vote=len(vote), status="rest_short", vote_rows=vote)
    if len(vote) < cfg.min_vote:
        return rep
    m = FM.robust_mean(p[vote], O[vote], sigma_eff[vote], elements, cfg)
    rep.iterations, rep.labels = m.iterations, m.labels
    rep.inlier_frac = float(np.mean(m.w >= 1.0))
    rep.spread = _spread(p[vote], O[vote], m.p, m.O, m.w, m.labels, elements)
    if not m.converged:
        rep.status, rep.reason = "refused", m.reason
        return rep
    if rep.inlier_frac < cfg.min_inlier_frac:
        rep.status, rep.reason = "refused", "few_inliers"
        return rep
    rep.drift = MT.motion_test_residuals(np.asarray(times)[vote], m.r, cfg, cfg.drift_seed_base + int(rows[0]))
    if rep.drift.moved:
        rep.status, rep.reason = "refused", "moved"
        return rep
    pbar, Obar = (m.p, m.O) if estimator == "cond" else FM.ctrl_mean(p[vote], O[vote], m.labels, elements)
    inside = _cap_mask(p[rows], O[rows], pbar, Obar, elements, cfg)
    rep.status, rep.p, rep.O, rep.cov = "consolidated", pbar, Obar, m.cov
    rep.replaced, rep.outlier_kept = rows[inside], rows[~inside]
    return rep


def _flip(p_rows, O, ok, states, seg_of_row, group, cfg) -> tuple[np.ndarray, dict]:
    """Per-row labels from the Viterbi (design 6); zeros when inactive."""
    n = len(O)
    labels = np.zeros(n, dtype=np.int64)
    rows = np.flatnonzero(ok)
    before = max_step_deg(O, rows)
    info = {"active": False, "max_step_before_deg": before, "max_step_after_deg": before, "relabelled_rows": 0,
            "continuous": bool(group.continuous)}
    if group.trivial or before <= cfg.flip_step_deg:
        return labels, info
    nodes: list[list[int]] = []
    for r in rows:                     # consecutive replaced rows of one segment form one node
        key = seg_of_row[r]
        if nodes and key >= 0 and seg_of_row[nodes[-1][-1]] == key and nodes[-1][-1] == r - 1:
            nodes[-1].append(int(r))
        else:
            nodes.append([int(r)])
    O_nodes = np.stack([O[nd[0]] for nd in nodes])
    sizes = np.array([len(nd) for nd in nodes])
    gaps = np.array([nodes[i + 1][0] - nodes[i][-1] for i in range(len(nodes) - 1)], dtype=np.float64)
    res = viterbi_labels(O_nodes, sizes, gaps, group.elements, np.radians(cfg.flip_unary_deg) ** 2)
    for nd, lab in zip(nodes, res.labels, strict=True):
        labels[nd] = lab
    O_after = np.einsum("nij,njk->nik", O, group.elements[labels])
    info.update(active=True, max_step_after_deg=max_step_deg(O_after, rows), relabelled_rows=int((labels != 0).sum()),
                cost_before=res.cost_before, cost_after=res.cost_after)
    return labels, info


def condition_window(inp: WindowInputs, cfg: ReferenceConfig | None = None, estimator: str = "cond"
                     ) -> ConditionedReference:
    """Run the conditioner on one window (module docstring)."""
    cfg = cfg or ReferenceConfig()
    P_in = np.asarray(inp.poses)
    P = P_in.astype(np.float64)
    n = len(P)
    ok = np.isfinite(P).all(axis=(1, 2))
    conf = np.ones(n) if inp.confidence is None else np.asarray(inp.confidence, dtype=np.float64)
    ws = inp.world
    flags: list[str] = []
    group = inp.group if inp.group is not None else compute_group(inp.mesh_vertices, inp.mesh_faces, inp.scale, cfg)
    c_star = group.c_star
    p, O = np.full((n, 3), np.nan), np.full((n, 3, 3), np.nan)
    if ok.any():
        sub = FR.WorldSource(ws.kind, ws.Q[ok], ws.u[ok], ws.sigma[ok], ws.reason)
        p[ok], O[ok] = FR.to_world(P[ok], c_star, sub)
    sigma_eff = effective_covariance(inp.covariance, P, c_star, ws, inp.alpha, cfg)
    states = np.where(ok, "no_world_frame", "no_pose").astype(object)
    segments: list[SegmentReport] = []
    dists: dict[str, np.ndarray] = {}
    if ws.can_consolidate:
        Ps = P.copy()
        Ps[:, :3, 3] *= ws.sigma[:, None]
        hands = {s: SG.HandInput(None if h.vertices is None else FR.scaled_points(h.vertices, ws), h.valid,
                                 h.declared_absent, h.distance) for s, h in inp.hands.items()}
        st, segs, dists = SG.segment_window(hands, Ps, inp.times, inp.mesh_vertices, inp.mesh_faces, inp.scale, cfg)
        states = st.astype(object)
        segments = [consolidate_rows(s, p, O, sigma_eff, conf, inp.times, group.elements, cfg, estimator)
                    for s in segs]
    else:
        flags.append("no_world_frame")
    p_new, O_new = p.copy(), O.copy()
    changed = np.zeros(n, dtype=bool)
    seg_of_row = np.full(n, -1)
    for k, s in enumerate(segments):
        states[s.rows] = "rest_short" if s.status == "rest_short" else ("rest_refused" if s.status == "refused"
                                                                         else "rest_consolidated")
        if s.status != "consolidated":
            continue
        states[s.outlier_kept] = "outlier_kept"
        p_new[s.replaced], O_new[s.replaced] = s.p, s.O
        changed[s.replaced] = True
        seg_of_row[s.replaced] = k
    labels, flip = _flip(p_new, O_new, ok, states, seg_of_row, group, cfg)
    relabel = labels != 0
    O_new[relabel] = np.einsum("nij,njk->nik", O_new[relabel], group.elements[labels[relabel]])
    changed |= relabel
    out = np.array(P_in, copy=True)
    if changed.any():
        sub = FR.WorldSource(ws.kind, ws.Q[changed], ws.u[changed], ws.sigma[changed], ws.reason)
        out[changed] = FR.to_record(p_new[changed], O_new[changed], c_star, sub).astype(P_in.dtype)
    pw = np.full((n, 4, 4), np.nan)
    pw[ok] = np.eye(4)
    pw[ok, :3, :3] = O_new[ok]
    pw[ok, :3, 3] = p_new[ok] - O_new[ok] @ c_star
    return ConditionedReference(out, pw, p_new, states.astype(str), changed, labels, segments, group, ws.kind,
                                ws.reason, flip, dists, c_star, sigma_eff, ws, cfg, tuple(flags))
