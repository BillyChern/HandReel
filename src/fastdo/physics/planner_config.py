"""Frozen, opt-in configuration of our planners. Every planner is OFF unless its config says `enabled=True`."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = ["C2Config", "CertificateConfig", "D1Config", "D4PrimeConfig", "PlannerConfig", "SqueezeConfig", "from_dict"]


@dataclass(frozen=True)
class D1Config:
    """Variance-reduced sampling MPC (D1). Fields not named here stay DO AS I DO's."""

    enabled: bool = False
    antithetic: bool = True          # mirror knot noise of rows [0, N/2) into [N/2, N)
    rank_weights: bool = True        # Blom rank weights normalised to the registered expected step
    accept_never_worse: bool = True  # keep the previous iterate if it scores higher under the same draws
    registered_top_q: float = 0.1    # the registered selection fraction the expected step is matched to


@dataclass(frozen=True)
class SqueezeConfig:
    """Tier-0 squeeze: commanded penetration of contacting finger links along the contact normal."""

    contact_dist_m: float = 0.005    # a hand link within this of the object at a reference frame is a contact
    delta_base_m: float = 0.0015     # internal-squeeze penetration per contact (scaled per finger by exp(theta_f))
    k_contact_n_per_m: float = 2000.0  # nominal soft-contact stiffness: extra penetration = f_n / k
    pen_min_m: float = 0.0005
    pen_max_m: float = 0.004
    move_max_m: float = 0.006        # |gap + penetration| cap per contact
    mu_scale: float = 0.7            # friction safety factor on the pair friction
    dls_lambda: float = 0.005        # damped least squares on the finger joints
    smooth_frames: int = 5           # moving average over reference frames
    ramp_steps: int = 100            # squeeze ramps in over the last 0.5 s of the warmup
    joint_offset_max_rad: float = 0.3


@dataclass(frozen=True)
class CertificateConfig:
    """Tighter-than-success acceptance of a tier-0 trajectory (criteria C1-C7 + robustness)."""

    success_pos_m: float = 0.1       # DO AS I DO's success thresholds (whole-trace and post-warmup means)
    success_rot_rad: float = 0.5
    post_pos_max_m: float = 0.06     # C4
    penetration_max_m: float = 0.008
    contact_kept_min: float = 0.5    # C6 (post-hoc only)
    displacement_ratio_min: float = 0.7  # C7
    robust_frac_min: float = 0.875   # perturbed worlds that must hold (predicted certificate only)
    drop_below_ref_m: float = 0.5    # the scorer's object_drop margin
    fail_err_m: float = 0.05         # repair: first post-release step with object error above this
    repair_lead_steps: int = 200     # repair checkpoint >= 1 s before the failure, on the 0.5 s control grid


@dataclass(frozen=True)
class C2Config:
    """Certified structured controller + D1 repair (C2)."""

    enabled: bool = False
    squeeze: SqueezeConfig = field(default_factory=SqueezeConfig)
    certificate: CertificateConfig = field(default_factory=CertificateConfig)
    gravity_ff: bool = True
    servo_gain0: float = 0.5         # theta0's K_o
    servo_sat_m: float = 0.02
    search_candidates: int = 64      # tier 0b: candidates x worlds_per_candidate = env worlds
    search_iterations: int = 4
    search_elite: int = 8
    search_sigma: tuple[float, ...] = (0.5, 0.5, 0.5, 0.5, 0.5, 0.3, 0.004)
    search_lo: tuple[float, ...] = (-1.4, -1.4, -1.4, -1.4, -1.4, 0.0, -0.01)
    search_hi: tuple[float, ...] = (1.4, 1.4, 1.4, 1.4, 1.4, 2.0, 0.01)
    terms_off: bool = False          # KA-T: every tier-0 term zero (gravity, squeeze, servo)
    force_repair_from: int | None = None  # second-process repair: tier-0 prefix up to this step, then D1 ...
    force_theta: tuple[float, ...] | None = None  # ... with this tier-0 theta (no tier-0 rollouts are run)


@dataclass(frozen=True)
class D4PrimeConfig:
    """Exact-engine finite-difference Gauss-Newton / Levenberg-Marquardt on the knot vector (D4'; default OFF).

    Knots: DO AS I DO's knot spacing (knot_dt / sim_dt = 40 steps) and linear (align_corners) interpolation, over the
    whole clip. Units: knot values are scaled by DO AS I DO's knot noise (pos / rot / joint) before the trust
    region and the damping act on them."""

    enabled: bool = False
    knot_steps: int = 40
    sigma_pos: float = 0.01          # registered pos_noise_scale (m)
    sigma_rot: float = 0.01          # registered rot_noise_scale (rad)
    sigma_joint: float = 0.1         # registered joint_noise_scale (rad)
    fd_eps: float = 1e-5
    fd_centered: bool = False
    max_iters: int = 60              # linearisations (GN iterations)
    max_trials: int = 6              # damping increases within one iteration before it counts as stalled
    cg_max_iters: int = 100
    cg_rtol: float = 1e-3
    rel_tol: float = 1e-4            # stop after `stall_checks` accepted iterations each below this relative decrease
    stall_checks: int = 2
    a_floor: float = 1e-3            # IRLS floor of the weighted pose-difference norm
    b_floor: float = 1e-2            # IRLS floor of the velocity-difference norm
    lambda0_scale: float = 0.1       # lambda_0 = scale * Rayleigh quotient of J^T J along the gradient
    workers: int = 4


@dataclass(frozen=True)
class PlannerConfig:
    planner: str = "ref"             # "ref" | "d1" | "c2" | "d4p"
    d1: D1Config = field(default_factory=D1Config)
    c2: C2Config = field(default_factory=C2Config)
    d4p: D4PrimeConfig = field(default_factory=D4PrimeConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def from_dict(d: dict[str, Any]) -> PlannerConfig:
    """Inverse of `PlannerConfig.to_dict` (JSON round trip; tuples come back from lists)."""
    c2 = dict(d.get("c2", {}))
    if "squeeze" in c2:
        c2["squeeze"] = SqueezeConfig(**c2["squeeze"])
    if "certificate" in c2:
        c2["certificate"] = CertificateConfig(**c2["certificate"])
    for k in ("search_sigma", "search_lo", "search_hi", "force_theta"):
        if c2.get(k) is not None:
            c2[k] = tuple(c2[k])
    return PlannerConfig(planner=d.get("planner", "ref"), d1=D1Config(**d.get("d1", {})), c2=C2Config(**c2),
                         d4p=D4PrimeConfig(**d.get("d4p", {})))
