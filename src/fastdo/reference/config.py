"""Frozen constants of the reference conditioner.

Every number the module acts on lives here; `constants_hash` names the set in records.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

__all__ = ["ReferenceConfig", "constants_hash"]


@dataclass(frozen=True)
class ReferenceConfig:
    # segmentation (design 3, R1.3)
    contact_m: float = 0.015            # d_on: a hand within this of the posed surface touches the object
    release_m: float = 0.100            # d_off: a hand farther than this is FAR; in between is NEAR
    vertex_stride: int = 4              # every 4th MANO vertex (the export's routine)
    guard_rows: int = 3                 # rows on each side of a touched row that are never rest
    bridge_s: float = 1.0               # unknown-hand certification window on each side
    # voting and weights (design 4.2, R1.4)
    min_confidence: float = 0.1
    min_vote: int = 5
    floor_pos_m: float = 0.005
    floor_rot_deg: float = 1.0
    huber_k: float = 3.55
    chi2_6_median: float = 5.348
    # solver (R1.4)
    gn_tol: float = 1e-7
    gn_max_iter: int = 100
    gn_max_halvings: int = 30
    gn_max_cond: float = 1e12
    # refusals (R1.5, R1.6)
    drift_permutations: int = 1999
    drift_alpha: float = 0.05           # rev 2: the JOINT (Bonferroni) level of trend + split-half
    drift_seed_base: int = 299
    split_half_min_vote: int = 6
    # rev 2 (R2.1): magnitude floor of the motion test = P7b's tolerance = the action cap; robust-scale floors
    motion_floor_m: float = 0.020
    motion_floor_deg: float = 15.0
    motion_min_rows: int = 5
    motion_min_scale_m: float = 1e-4
    motion_min_scale_deg: float = 0.01
    min_inlier_frac: float = 0.5
    cap_pos_m: float = 0.020
    cap_rot_deg: float = 15.0
    # symmetry (R1.7)
    sym_tol_abs_m: float = 0.003
    sym_tol_rel: float = 0.05
    sym_near_tol_m: float = 0.010
    sym_step_deg: int = 15
    sym_samples: int = 16000
    sym_seed: int = 0
    sym_max_elements: int = 48
    # flip repair (design 6, R1.8)
    flip_step_deg: float = 90.0
    flip_unary_deg: float = 10.0
    # covariance sanitising (design 1)
    cov_eig_min: float = 1e-12
    cov_eig_max: float = 1e2

    def to_dict(self) -> dict:
        return asdict(self)


def constants_hash(cfg: ReferenceConfig) -> str:
    """sha256 (first 16 hex) of the sorted JSON of the constants."""
    return hashlib.sha256(json.dumps(cfg.to_dict(), sort_keys=True).encode()).hexdigest()[:16]
