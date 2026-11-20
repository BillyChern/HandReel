"""Truncated-Sinkhorn gradient-check report.
`python -m fastdo.tracker.gradient_check_report --out outputs/tracker/gradient_check.json [--quick]`

Problem: frame `frame` of a synthetic box clip at `SolverConfig`'s fine level (eps, rho in `scale2` units) with the
visibility-masked template marginal. Probes: `E_k = 2 (x_i - y_j) . m_k(x_i)` for 7 motions `m_k` of the
observed cloud (translations `T3`, rotations about the valid-row centroid `R3`, scale `S1`), each scaled to
`||E_k||_inf = eps`. Inits: g-driven `cluster_oracle` (a lower bound on the bias of a 64-cluster `dual_g`
head), `dual_noise`, `lm_warm` (the solver's warm start after a small step), `cold` (an untrained tracker's zero
heads); f-driven `f_oracle` and `f_noise`. Every estimator treats the init as a constant (`dual_init_detached`).
Decision (valid only for an `ot_objective` energy): per estimator, the maximum over `decision_inits` of the maximum
probe-group bias at `budget_n_iter`; the first of `detached, implicit, unrolled` at or below `threshold`, else "none".
The barycentric protocol numbers compare the plan-fixed and exact FD gradients of `total_cost([pointmap_term])` along
the same motions at converged plans: a detached plan is not an envelope gradient of that energy.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import itertools
import json
import logging
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from fastdo.geometry.lie import se3_exp
from fastdo.solver.api import SolverConfig
from fastdo.solver.batch import sample_points, valid_pixels
from fastdo.solver.ot import sinkhorn_unbalanced_potentials
from fastdo.solver.synthetic import SyntheticConfig, make_synthetic_clip
from fastdo.solver.terms import pointmap_term, transform_points
from fastdo.tracker.clusters import mesh_cluster_ids
from fastdo.tracker.sinkhorn_check import (
    Drive,
    converged_reference,
    dual_gradients,
    sinkhorn_iterate,
    truncated_sinkhorn_gradient_check,
)

__all__ = ["DEFAULT_CONFIG", "INIT_DRIVE", "QUICK_CONFIG", "GradientCheckConfig", "GradientCheckProblem",
           "build_gradient_check_problem", "main", "run_gradient_check_report", "warm_start"]

logger = logging.getLogger(__name__)

F64 = torch.float64
SCHEMA = "fastdo.tracker.gradient_check/v1"
ESTIMATORS = ("detached", "unrolled", "implicit")
DECISION_ORDER = ("detached", "implicit", "unrolled")
PROBE_GROUPS: dict[str, tuple[int, int]] = {"T3": (0, 3), "R3": (3, 6), "S1": (6, 7)}
INIT_DRIVE: dict[str, Drive] = {"cluster_oracle": "g", "f_oracle": "f", "dual_noise": "g", "f_noise": "f",
                                "lm_warm": "g", "cold": "g"}
N_CLUSTERS = 64
CONVERGED_DELTA = 1e-8   # iterates within this many eps of the fixed point are at round-off: no ratio is formed
SYNTHETIC_DEFAULT_DTYPE = torch.float32   # pinned: make_synthetic_clip draws its noise in the default dtype (I-1)


@contextlib.contextmanager
def _pinned_default_dtype() -> Iterator[None]:
    previous = torch.get_default_dtype()
    torch.set_default_dtype(SYNTHETIC_DEFAULT_DTYPE)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


@dataclass(frozen=True)
class GradientCheckConfig:
    seed: int = 0
    n_frames_clip: int = 6
    frame: int = 0
    n_obs: int = 96
    n_surface: int = 256
    n_tracks: int = 16
    eps_scale2: float | None = None       # None -> SolverConfig().eps_fine
    rho_a_scale2: float | None = None     # None -> SolverConfig().rho_x
    rho_b_scale2: float | None = None     # None -> SolverConfig().rho_y
    n_iters: tuple[int, ...] = (2, 3, 4, 5)
    inits: tuple[str, ...] = ("cluster_oracle", "f_oracle", "dual_noise", "f_noise", "lm_warm", "cold")
    primary_init: str = "cluster_oracle"
    decision_inits: tuple[str, ...] = ("cluster_oracle", "f_oracle", "lm_warm", "cold")
    noise_sigma: float = 0.5
    lm_warm_step: float = 0.002
    h: float = 1e-4
    n_converged: int = 5000
    n_converged_fd: int = 3000
    threshold: float = 0.1
    budget_n_iter: int = 5


DEFAULT_CONFIG = GradientCheckConfig()
QUICK_CONFIG = GradientCheckConfig(n_frames_clip=2, n_obs=32, n_surface=64, eps_scale2=1e-3, inits=("dual_noise",),
                                   primary_init="dual_noise", decision_inits=("dual_noise",), n_converged=2000,
                                   n_converged_fd=1000)


@dataclass(frozen=True)
class GradientCheckProblem:
    C: Tensor                            # (1,N,M) squared distances at the frame's GT pose
    a: Tensor                            # (1,N) observed marginal (valid rows)
    b: Tensor                            # (1,M) visibility-masked template marginal
    eps: float
    rho_a: float
    rho_b: float
    directions: Tensor                   # (7,1,N,M) probes T3 | R3 | S1
    cluster_id: Tensor | None            # (M,) 64-cluster ids, None when n_surface < 64
    cost_at: Callable[[Tensor], Tensor]  # pose (4,4) -> (1,N,M) cost at that pose
    length_scale: float                  # L = sqrt(scale2)
    pose: Tensor                         # (4,4) GT pose of the frame
    X: Tensor                            # (1,N,3) observed points, zeros on invalid rows
    X_valid: Tensor                      # (1,N) bool
    Y: Tensor                            # (M,3) template samples
    log_scale: Tensor                    # 0-dim
    motions: Tensor                      # (7,N,3) scaled motion fields: d C / d z_k = directions[k]


def build_gradient_check_problem(cfg: GradientCheckConfig) -> GradientCheckProblem:
    """The report problem, with its 7 probes."""
    sc = SolverConfig()
    with _pinned_default_dtype():   # the clip's noise stream depends on the default dtype: pin it
        clip = make_synthetic_clip(SyntheticConfig(n_frames=cfg.n_frames_clip, image_hw=(120, 160), focal=150.0,
                                                   shape="box", n_surface=cfg.n_surface, n_tracks=cfg.n_tracks),
                                   seed=cfg.seed)
    points, counts = valid_pixels(clip.masks, clip.pointmaps)
    X_all, V_all = sample_points(points, torch.cat([counts.new_zeros(1), counts.cumsum(0)]), cfg.n_obs,
                                 torch.Generator().manual_seed(cfg.seed))
    k, s = cfg.frame, float(clip.scale_gt)
    if not bool((valid := V_all[k:k + 1]).any()):
        raise ValueError(f"frame {k} has no valid observed row (n_obs {cfg.n_obs}): the marginal a would be all zero")
    X = torch.where(valid[..., None], X_all[k:k + 1], torch.zeros_like(X_all[k:k + 1]))
    Y, normals, pose = clip.surface_points.double(), clip.surface_normals.double(), clip.poses_gt.double()[k]
    scale2 = (s * float((Y.max(0).values - Y.min(0).values).norm())) ** 2

    def cost_at(p: Tensor) -> Tensor:
        return torch.cdist(X, transform_points(Y, p[None], s)) ** 2

    Yc = transform_points(Y, pose[None], s)[0]
    visible = ((normals @ pose[:3, :3].T) * (-Yc)).sum(-1) > 0
    eps = (sc.eps_fine if cfg.eps_scale2 is None else cfg.eps_scale2) * scale2
    x, eye = X[0], torch.eye(3, dtype=F64)
    centroid = x[valid[0]].mean(0)
    fields = [eye[d].expand_as(x) for d in range(3)]
    fields += [torch.cross(eye[d].expand_as(x), x - centroid, dim=-1) for d in range(3)] + [x - centroid]
    raw = torch.stack([2 * ((x[:, None, :] - Yc[None, :, :]) * m[:, None, :]).sum(-1) for m in fields])
    scale = eps / raw.abs().flatten(1).max(1).values                      # ||E_k||_inf = eps
    return GradientCheckProblem(
        C=cost_at(pose), a=valid.to(F64) / valid.sum(), b=(visible.to(F64) / visible.sum())[None],
        eps=eps, rho_a=(sc.rho_x if cfg.rho_a_scale2 is None else cfg.rho_a_scale2) * scale2,
        rho_b=(sc.rho_y if cfg.rho_b_scale2 is None else cfg.rho_b_scale2) * scale2,
        directions=(raw * scale[:, None, None])[:, None],
        cluster_id=mesh_cluster_ids(Y, N_CLUSTERS, 0) if cfg.n_surface >= N_CLUSTERS else None, cost_at=cost_at,
        length_scale=math.sqrt(scale2), pose=pose, X=X, X_valid=valid, Y=Y,
        log_scale=torch.tensor(math.log(s), dtype=F64), motions=torch.stack(fields) * scale[:, None, None])


def warm_start(problem: GradientCheckProblem, kind: str, cfg: GradientCheckConfig,
               converged: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
    """The `(f0, g0)` init of `kind`; f-driven inits return `g0 = 0`, which the f drive replaces."""
    (fs, gs), p = converged, problem
    if kind == "cluster_oracle":
        if p.cluster_id is None:
            raise ValueError("cluster_oracle needs n_surface >= 64 (mesh_cluster_ids)")
        ids, vis = p.cluster_id, p.b[0] > 0
        sums = torch.zeros(N_CLUSTERS, dtype=F64).index_add(0, ids[vis], gs[0, vis])
        count = torch.bincount(ids[vis], minlength=N_CLUSTERS).to(F64)
        return fs.clone(), torch.where(count > 0, sums / count.clamp_min(1), torch.zeros_like(sums))[ids][None]
    if kind in ("dual_noise", "f_noise"):                     # one generator, float64 draws (1,N) then (1,M)
        gen = torch.Generator().manual_seed(cfg.seed + 1)
        f0 = fs + cfg.noise_sigma * p.eps * torch.randn(fs.shape, generator=gen, dtype=F64)
        g0 = gs + cfg.noise_sigma * p.eps * torch.randn(gs.shape, generator=gen, dtype=F64)
        return (f0, g0) if kind == "dual_noise" else (f0, torch.zeros_like(gs))
    if kind in ("f_oracle", "cold"):
        return (fs.clone() if kind == "f_oracle" else torch.zeros_like(fs)), torch.zeros_like(gs)
    if kind == "lm_warm":
        L = p.length_scale
        xi = cfg.lm_warm_step * torch.tensor([L, -0.5 * L, 0.7 * L, 1.0, -0.6, 0.4], dtype=F64)
        return sinkhorn_unbalanced_potentials(p.cost_at(se3_exp(xi) @ p.pose), p.a, p.b, p.eps, p.rho_a, p.rho_b,
                                              cfg.n_converged)[1:]
    raise ValueError(f"unknown init {kind!r}; expected one of {sorted(INIT_DRIVE)}")


def _rel(G: Tensor, ref: Tensor) -> float:
    return float((G - ref).norm() / ref.norm())


def _group_biases(G: Tensor, ref: Tensor) -> dict[str, float]:
    out = {name: _rel(G[lo:hi], ref[lo:hi]) for name, (lo, hi) in PROBE_GROUPS.items()}
    return {**out, "all7": _rel(G, ref), "max_group": max(out.values())}


def _decide(biases: Mapping[str, float], threshold: float) -> str:
    """The cheapest estimator (`detached, implicit, unrolled`) whose bias is `<= threshold`, else "none"."""
    return next((name for name in DECISION_ORDER if biases[name] <= threshold), "none")


def _barycentric(p: GradientCheckProblem, cfg: GradientCheckConfig) -> dict[str, dict[str, float]]:
    """Plan-fixed vs exact FD gradient of `total_cost([pointmap_term])` along the scaled motions (converged plans)."""
    args = (p.X_valid, p.Y, p.pose[None], p.log_scale, p.eps, p.rho_a, p.rho_b)

    def energy(z: float, k: int, plan: Tensor | None, init: tuple[Tensor, Tensor]) -> Tensor:
        Xz = p.X + z * p.motions[k][None]
        if plan is not None:
            term, _ = pointmap_term(Xz, *args, 0, plan=plan, mesh_mask=p.b > 0)
        else:
            term, _, _ = pointmap_term(Xz, *args, cfg.n_converged_fd, mesh_mask=p.b > 0, potentials=init)
        return (term.weight.clamp_min(0.0) * (term.residual ** 2).sum(-1)).sum()   # lm.total_cost (lm imports scipy)

    _, plan0, (f0, g0) = pointmap_term(p.X, *args, cfg.n_converged, mesh_mask=p.b > 0, potentials=None)
    assert isinstance(plan0, Tensor) and f0 is not None and g0 is not None
    h, K = cfg.h, p.motions.shape[0]
    fd = {name: torch.stack([(energy(h, k, plan, (f0, g0)) - energy(-h, k, plan, (f0, g0))) / (2 * h)
                             for k in range(K)]) for name, plan in (("fixed", plan0), ("exact", None))}
    fixed, exact = fd["fixed"], fd["exact"]
    return {name: {"rel": _rel(fixed[lo:hi], exact[lo:hi]),
                   "cos": float(fixed[lo:hi] @ exact[lo:hi] / (fixed[lo:hi].norm() * exact[lo:hi].norm()))}
            for name, (lo, hi) in {**PROBE_GROUPS, "all7": (0, K)}.items()}


@_pinned_default_dtype()   # the numbers must not depend on the caller's default dtype
def run_gradient_check_report(out_path: Path, cfg: GradientCheckConfig = DEFAULT_CONFIG) -> dict[str, Any]:
    """Build the problem, run every `(init, n)`, write the schema-v1 JSON to `out_path` and return it."""
    if (any(k not in INIT_DRIVE for k in cfg.inits) or cfg.primary_init not in cfg.inits
            or not set(cfg.decision_inits) <= set(cfg.inits) or cfg.budget_n_iter not in cfg.n_iters):
        raise ValueError(f"inconsistent config: inits {cfg.inits}, primary {cfg.primary_init}, decision "
                         f"{cfg.decision_inits}, budget {cfg.budget_n_iter} in n_iters {cfg.n_iters}")
    p = build_gradient_check_problem(cfg)
    eps, R, S, scale2 = p.eps, p.a[0] > 0, p.b[0] > 0, p.length_scale ** 2
    ref = converged_reference(p.C, p.a, p.b, eps, p.rho_a, p.rho_b, p.directions, cfg.h, cfg.n_converged,
                              cfg.n_converged_fd)
    tau_a, tau_b = p.rho_a / (p.rho_a + eps), p.rho_b / (p.rho_b + eps)
    unrolled_fd, grad_ratios, contractions = 0.0, list[float](), list[float]()   # formed only off the fixed point
    cases: dict[str, dict[str, Any]] = {}
    for kind in cfg.inits:
        drive = INIT_DRIVE[kind]
        init = warm_start(p, kind, cfg, (ref.f, ref.g))
        err_key, err = (("z0_sup_err_g_support_over_eps", float((init[1] - ref.g)[0, S].abs().max()) / eps)
                        if drive == "g" else
                        ("z0_sup_err_f_over_eps", float((init[0] - ref.f)[0, R].abs().max()) / eps))
        per_n: dict[str, dict[int, dict[str, float]]] = {name: {} for name in ESTIMATORS}
        for n in cfg.n_iters:
            est = truncated_sinkhorn_gradient_check(p.C, p.a, p.b, eps, p.rho_a, p.rho_b, n, init, p.directions,
                                                    h=cfg.h, drive=drive, reference=ref)
            for name in ESTIMATORS:
                per_n[name][n] = _group_biases(getattr(est, name), ref.fd_converged)
            unrolled_fd = max(unrolled_fd, _rel(est.unrolled, est.fd_truncated))
        cases[kind] = {"drive": drive, err_key: err,
                       **{name: {group: {f"n_iter_{n}": per_n[name][n][group] for n in cfg.n_iters}
                                 for group in (*PROBE_GROUPS, "all7", "max_group")} for name in ESTIMATORS}}
        deltas = []                                  # sup error of (f_R, g_S) after n = 1, 2, ... iterations
        for n in range(1, max(cfg.n_iters) + 2):
            _, f, g = sinkhorn_iterate(p.C, p.a, p.b, eps, p.rho_a, p.rho_b, n, init, drive)
            deltas.append(max(float((f - ref.f)[0, R].abs().max()), float((g - ref.g)[0, S].abs().max())))
            if deltas[-1] > CONVERGED_DELTA * eps:
                grad_f, grad_g = dual_gradients(p.C, p.a, p.b, eps, p.rho_a, p.rho_b, f, g)
                grad_ratios.append(float(grad_g.abs().max() / grad_f.abs().max()))
        contractions += [d1 / d0 for d0, d1 in itertools.pairwise(deltas) if d0 > CONVERGED_DELTA * eps]
    if not grad_ratios or not contractions:   # no validator may read as passing when it measured nothing
        raise ValueError(f"inits {cfg.inits}: no iterate beyond {CONVERGED_DELTA} eps of the fixed point to validate")
    grad_ratio, contraction = max(grad_ratios), max(contractions)
    budget = f"n_iter_{cfg.budget_n_iter}"
    bias_at_budget = {name: max(cases[k][name]["max_group"][budget] for k in cfg.decision_inits)
                      for name in ESTIMATORS}
    ot_gradient = _decide(bias_at_budget, cfg.threshold)
    bary = _barycentric(p, cfg)
    config = {**dataclasses.asdict(cfg), "eps_scale2": eps / scale2, "rho_a_scale2": p.rho_a / scale2,
              "rho_b_scale2": p.rho_b / scale2}
    report: dict[str, Any] = {
        "schema": SCHEMA, "created_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "torch_version": torch.__version__, "scope": "dense_float64",
        "config": {k: list(v) if isinstance(v, tuple) else v for k, v in config.items()},
        "problem": {"N": p.C.shape[1], "M": p.C.shape[2], "n_visible": int(S.sum()), "n_valid_obs": int(R.sum()),
                    "scale2": scale2, "length_scale": p.length_scale, "eps": eps, "rho_a": p.rho_a,
                    "rho_b": p.rho_b, "probe_groups": {k: list(range(*v)) for k, v in PROBE_GROUPS.items()},
                    "synthetic_default_dtype": str(SYNTHETIC_DEFAULT_DTYPE).removeprefix("torch.")},
        "tau_a": tau_a, "tau_b": tau_b, "kappa": tau_a * tau_b,
        "protocol": {"envelope_vs_fd": _rel(ref.envelope, ref.fd_converged),
                     "unrolled_vs_fd_truncated_max": unrolled_fd, "grad_g_over_grad_f_max": grad_ratio,
                     "contraction_ratio_max": contraction,
                     "contraction_ratio_over_kappa_max": contraction / (tau_a * tau_b),
                     "barycentric_planfixed_vs_exact_rel": bary["all7"]["rel"],
                     "barycentric_planfixed_vs_exact_cos": bary["all7"]["cos"], "barycentric_groups": bary},
        **{f"{prefix}n_iter_{n}": cases[cfg.primary_init][name]["max_group"][f"n_iter_{n}"]
           for prefix, name in (("", "detached"), ("unrolled_", "unrolled"), ("implicit_", "implicit"))
           for n in cfg.n_iters},
        "cases": cases,
        "decision": {"threshold": cfg.threshold, "budget_n_iter": cfg.budget_n_iter,
                     "primary_init": cfg.primary_init, "decision_inits": list(cfg.decision_inits),
                     "metric": "max over decision_inits of the max probe-group bias at budget_n_iter",
                     "bias_at_budget": bias_at_budget,
                     "material": {k: v > cfg.threshold for k, v in bias_at_budget.items()},
                     "ot_gradient": ot_gradient, "dual_init_detached": True, "applies_to": "ot_objective"},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    logger.info("gradient check: kappa=%.6f, bias at n=%d over %s: %s -> ot_gradient=%s; wrote %s", tau_a * tau_b,
                cfg.budget_n_iter, list(cfg.decision_inits), bias_at_budget, ot_gradient, out_path)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Truncated-Sinkhorn OT-value gradient bias report.")
    parser.add_argument("--out", type=Path, required=True, help="output JSON path")
    parser.add_argument("--quick", action="store_true", help="use QUICK_CONFIG (small linear-regime problem)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    run_gradient_check_report(args.out, QUICK_CONFIG if args.quick else DEFAULT_CONFIG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
