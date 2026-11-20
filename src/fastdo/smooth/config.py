"""`smooth_stage` configuration.

`smooth_stage: null` (the default) means OFF: no stage, no artefact, BASE byte-identical (KA-ID). A
mapping turns it on; `smooth_stage: {}` is the primary candidate C1 SE3-RTS-HO (`arm: se3_contact`) with the frozen
constants below, and `{arm: alongray_contact}` is candidate C2 AR1D-HO.

DOMAIN GATE: with `contact_only` True (the default) the stage acts ONLY
where the production contact stage ran and produced a contact side, on a clip that is not declared egocentric (the
source-domain rule that also switches the contact stage). Everywhere else it returns BASE unchanged and
writes `skipped` into its record: egocentric video (HOI4D, Ego4D, V2D, the freeze headline) stays at BASE. The
object-only arms set `contact_only: False`; they are development and ablation arms only.

FROZEN constants (pooled medians over 96 SEEN screening units; GT-referenced on seen data only, fixed before any
fresh draw exists):
- `a_par` 0.0533: within-window sd of the signed along-ray centroid error / |c_gt| (DexYCB 0.0687, HOI4D 0.0247).
  ONE pooled value for every domain: the Gaussian MAP depends only on noise RATIOS, so the absolute value acts only
  through the Huber threshold (a verifier's sweep, 5-100 mm, moved C-SMOOTH's DexYCB M by <= 0.011).
- `kappa` 6.04: median per-unit ratio along-ray / perpendicular (per dof) within-window sd (D 7.78, H 3.83);
  a_perp = a_par / kappa.
- `s_rot` 0.182 rad: per-dof within-window rotation dispersion about the window's chordal mean (D 0.316, H 0.066).
- `band` 0.25: velocity random-walk sd per row = band x the measurement sd (C-SMOOTH's ratio; the
  verifier's sweep: CI excludes 0 on DexYCB for 0.1-2).
- hand (contact arm only; DexYCB contact side, HaWoR wrist vs GT MANO joint 0): `a_par_h` 0.0248, `a_perp_h` 0.00694;
  grasp-rigidity sd `q_c_rel` 0.0288 x zbar per row: the per-unit RMS (median over the 60 seen DexYCB units) of the
  factor's OWN residual g_k evaluated on GT (GT pose, twist, wrist and wrist velocity) over the rows the factor acts
  on (contact.npz shift_mm != 0). An earlier value, 0.0264, was the sd of the rigid-translation relative velocity,
  which is not the factor's residual.
  The residual is heavy-tailed: 73% of the 1,066 flagged rows are within 0.0264; per unit a median 79% of flagged
  rows have a GT hand joint within 20 mm of the GT object surface; the factor's Huber weight absorbs the rest;
  wrist process noise `q_h_rel` 0.0149 x zbar per row = the GT wrist's acceleration sd per row (per dof, relative),
  i.e. the CV model's own definition of its process noise.
- `q_v_rel` 0 (= derived): the object's velocity random-walk sd per row is band x a_par x zbar (0.0133 zbar). A
  positive value overrides it with q_v_rel x zbar (the S-F arm `q_v = GT acc`, DexYCB's GT acceleration sd 0.0108:
  the two bodies' process noise follows two principles, and this arm reports the difference). (The object's band
  0.25 x a_par = 0.0133 sits between
  the object's GT acceleration sd, DexYCB 0.0108 / pooled 0.0075; the first seen screen used 0.25 x a_par_h = 0.0062
  for the wrist, 2.4x too stiff, and the wrist's along-ray error rose 2.5% on the seen set.)
- `smooth_rotation` False: the SE(3) smoother estimates the rotation jointly (it enters the process and the
  grasp factor's lever term) but WRITES BASE's rotation. On seen DexYCB, rotation smoothing bought nothing
  (C1 rotation error M -0.004, 97.5% CI [-0.027, +0.035], margin log 1.05 = 0.049), and 12.6% of DexYCB rows sit
  > 60 deg from the window's chordal rotation in contiguous runs of up to 11 rows (symmetric branches), which per-row
  Huber cannot see. True is an ablation arm.
- `couple` (contact arm): the grasp-rigidity factor on contact rows; False smooths both bodies without it (ablation).
- Huber k 1.345 on the per-dof Mahalanobis RMS of each block, 3 reweightings (C-SMOOTH's).
Robustness multipliers (`noise_mult`, `band_mult`, `kappa_mult`) are 1 in the candidate; the S-F arms set 0.5 / 2.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from fastdo.smooth.rts import Noise

__all__ = ["ARMS", "SMOOTH_KEY", "SmootherConfig", "smoother_from_mapping"]

SMOOTH_KEY = "smooth_stage"
ARMS = ("se3", "se3_contact", "se3_cotransport", "alongray", "alongray_contact")
_PROCESSES = ("se3", "so3xr3")


@dataclass(frozen=True)
class SmootherConfig:
    """One window's smoother settings (module docstring). Frozen; validated on construction."""

    arm: str = "se3_contact"
    contact_only: bool = True
    process: str = "se3"
    a_par: float = 0.0533
    kappa: float = 6.04
    s_rot: float = 0.182
    band: float = 0.25
    a_par_h: float = 0.0248
    a_perp_h: float = 0.00694
    q_c_rel: float = 0.0288
    q_h_rel: float = 0.0149
    q_v_rel: float = 0.0
    huber_k: float = 1.345
    reweights: int = 3
    gn_iters: int = 8
    smooth_rotation: bool = False
    couple: bool = True
    gauge_hold: bool = True
    noise_ref: str = "estimate"
    noise_mult: float = 1.0
    band_mult: float = 1.0
    kappa_mult: float = 1.0

    def __post_init__(self) -> None:
        if self.arm not in ARMS:
            raise ValueError(f"smooth_stage.arm must be one of {ARMS}, got {self.arm!r}")
        if self.process not in _PROCESSES:
            raise ValueError(f"smooth_stage.process must be one of {_PROCESSES}, got {self.process!r}")
        if self.noise_ref not in ("estimate", "measured"):
            raise ValueError(f"smooth_stage.noise_ref must be 'estimate' or 'measured', got {self.noise_ref!r}")
        for name in ("a_par", "kappa", "s_rot", "band", "a_par_h", "a_perp_h", "q_c_rel", "q_h_rel", "huber_k", "noise_mult",
                     "band_mult", "kappa_mult"):
            v = getattr(self, name)
            if not (isinstance(v, (int, float)) and v > 0):
                raise ValueError(f"smooth_stage.{name} must be a positive number, got {v!r}")
        if not (isinstance(self.q_v_rel, (int, float)) and self.q_v_rel >= 0):
            raise ValueError(f"smooth_stage.q_v_rel must be >= 0 (0 = derived from band x a_par), got {self.q_v_rel!r}")
        if self.reweights < 0 or self.gn_iters < 1:
            raise ValueError("smooth_stage.reweights must be >= 0 and gn_iters >= 1")

    def noise(self, zbar: float) -> Noise:
        """Absolute noise for a window whose median measured centroid range is `zbar` (metres)."""
        a_par = self.a_par * self.noise_mult
        a_perp = a_par / (self.kappa * self.kappa_mult)
        s_rot = self.s_rot * self.noise_mult
        b = self.band * self.band_mult
        q_v = (self.q_v_rel * self.band_mult * self.noise_mult if self.q_v_rel > 0 else b * a_par) * zbar
        return Noise(s_rot=s_rot, a_par=a_par, a_perp=a_perp, q_w=b * s_rot, q_v=q_v,
                     a_par_h=self.a_par_h * self.noise_mult, a_perp_h=self.a_perp_h * self.noise_mult,
                     q_h=self.q_h_rel * self.band_mult * self.noise_mult * zbar, q_c=self.q_c_rel * self.noise_mult * zbar,
                     huber_k=self.huber_k, reweights=self.reweights, gn_iters=self.gn_iters,
                     noise_ref=self.noise_ref, process=self.process)


def smoother_from_mapping(block: Mapping[str, Any] | None) -> SmootherConfig | None:
    """`smooth_stage` block -> config; None (the default) means the stage is off."""
    if block is None:
        return None
    known = {f.name for f in dataclasses.fields(SmootherConfig)}
    unknown = set(block) - known
    if unknown:
        raise ValueError(f"unknown smooth_stage keys: {sorted(unknown)}")
    return SmootherConfig(**dict(block))
