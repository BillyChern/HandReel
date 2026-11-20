"""`tba_stage` configuration (TBA / TBA-V).

`tba_stage: null` (the default in run/conf/config.yaml) means OFF: the engine registers no stage, writes nothing and
every artefact is the BASE pipeline's (KA-ID). A mapping turns it on; every field below has the registered default,
so `tba_stage: {}` is TBA-V as registered.

Registered values:
- BA unknowns are frames with >= `min_live` visible anchored keyframe tracks; other frames keep BASE's pose.
- Anchor depth prior centred at BASE's anchor (the keyframe ray's hit on BASE's posed template), sd `prior_sd_m`
  (25 mm) **at the reference depth `prior_ref_depth_m` (0.6 m), proportional to the anchor's BASE depth**. An
  absolute-length prior is not gauge-equivariant (KA-G: TBA-V(g_beta BASE) must equal g_beta TBA-V(BASE) to 1e-6),
  so the 25 mm is expressed as a 4.17% depth prior; this is the only reading of "sd 25 mm" that KA-G admits.
- Weak motion priors (second differences of the anchor-cloud centroid, relative sd `acc_sd_m` at the same reference
  depth, and of the frame-to-frame rotation, sd `rot_acc_sd_deg`), same equivariance argument.
- Cauchy track loss, scale `cauchy_c_px` = the production solver's `cauchy_c` (4 px), by IRLS: one weighted
  least-squares solve per entry of `gnc_c_px` (Cauchy scale of that round's weights), default four rounds at 4 px.
  The WIP's graduated non-convexity from 64 px was dropped on SYNTHETIC evidence only (tests/tba, before any
  screening output): BASE is already a Cauchy-4 px fit of the same tracks, and starting at 64 px let tracks swapped
  ~55 px (M-TRACK) steer the early rounds (median |dE1| 4.6 mm over 6 synthetic windows, one window's off-orbit RMS
  1.1 -> 42 mm); four rounds at 4 px: 0.8 mm, off-orbit recovery unchanged (0.7-1.1 mm).
- `gauge`: "K" (BASE's window gauge held: l_K = median over (k, live i) of log(z_BASE / z_BA)) or "consensus" (the
  draft's pointmap consensus l = median log(z_obs / z_BA); arm TBA-V-C, never promoted).
- R2 carving (`carve: true` = TBA-V, false = TBA): vote quantile `vote_q` 0.6 over frames that see a voxel, at least
  `min_views` such frames; UNKNOWN (no vote) inside HaWoR hand silhouettes dilated `hand_dilate_px` (5 px) and outside
  the image; object region = closing (`close_px` 3 px) of the pixels hit by the frame's stored object points; abstain
  if carving would remove more than `cap` (25%) of the template volume.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

__all__ = ["TBA_KEY", "TbaConfig", "tba_from_mapping"]

TBA_KEY = "tba_stage"
_GAUGES = ("K", "consensus")


@dataclass(frozen=True)
class TbaConfig:
    """One window's TBA / TBA-V settings (module docstring). Frozen; validated on construction."""

    carve: bool = True
    gauge: str = "K"
    min_live: int = 8
    prior_sd_m: float = 0.025
    prior_ref_depth_m: float = 0.6
    acc_sd_m: float = 0.02
    rot_acc_sd_deg: float = 5.0
    cauchy_c_px: float = 4.0
    gnc_c_px: tuple[float, ...] = (4.0, 4.0, 4.0, 4.0)
    robust: bool = True
    max_nfev: int = 60
    vote_q: float = 0.6
    min_views: int = 3
    hand_dilate_px: int = 5
    close_px: int = 3
    cap: float = 0.25
    hand_unknown: bool = True
    voxel_upsample: int = 2
    focal_mult: float = 1.0

    def __post_init__(self) -> None:
        if self.gauge not in _GAUGES:
            raise ValueError(f"{TBA_KEY}.gauge must be one of {_GAUGES}, got {self.gauge!r}")
        for name in ("prior_sd_m", "prior_ref_depth_m", "acc_sd_m", "rot_acc_sd_deg", "cauchy_c_px", "focal_mult"):
            v = getattr(self, name)
            if isinstance(v, bool) or not (math.isfinite(float(v)) and float(v) > 0.0):
                raise ValueError(f"{TBA_KEY}.{name} must be finite and > 0, got {v!r}")
            object.__setattr__(self, name, float(v))
        for name in ("min_live", "max_nfev", "min_views", "voxel_upsample"):
            n = getattr(self, name)
            if isinstance(n, bool) or int(n) != n or int(n) < 1:
                raise ValueError(f"{TBA_KEY}.{name} must be an integer >= 1, got {n!r}")
            object.__setattr__(self, name, int(n))
        for name in ("hand_dilate_px", "close_px"):
            n = getattr(self, name)
            if isinstance(n, bool) or int(n) != n or int(n) < 0:
                raise ValueError(f"{TBA_KEY}.{name} must be an integer >= 0, got {n!r}")
            object.__setattr__(self, name, int(n))
        if not (0.0 < float(self.vote_q) <= 1.0 and 0.0 < float(self.cap) <= 1.0):
            raise ValueError(f"{TBA_KEY}.vote_q and cap must lie in (0, 1], got {self.vote_q!r}, {self.cap!r}")
        c = tuple(float(x) for x in self.gnc_c_px)
        if not c or any(not (math.isfinite(x) and x > 0) for x in c) or c[-1] != self.cauchy_c_px:
            raise ValueError(f"{TBA_KEY}.gnc_c_px must be positive and end at cauchy_c_px, got {self.gnc_c_px!r}")
        object.__setattr__(self, "gnc_c_px", c)

    @property
    def prior_rel_sd(self) -> float:
        """The anchor prior as a fraction of depth (0.025 / 0.6)."""
        return self.prior_sd_m / self.prior_ref_depth_m

    @property
    def acc_rel_sd(self) -> float:
        """The centroid-acceleration prior as a fraction of the window's median anchor depth."""
        return self.acc_sd_m / self.prior_ref_depth_m


def tba_from_mapping(block: Any) -> TbaConfig | None:
    """`tba_stage`: null -> None (OFF, nothing registered); a mapping -> `TbaConfig` (unknown keys refused).

    Raises:
        ValueError: On a non-mapping, an unknown key, or anything `TbaConfig` rejects.
    """
    if block is None:
        return None
    if not isinstance(block, Mapping):
        raise ValueError(f"{TBA_KEY} must be null or a mapping, got {block!r}")  # noqa: TRY004
    data = {str(k): block[k] for k in block}
    known = {f.name for f in dataclasses.fields(TbaConfig)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(f"{TBA_KEY} has unknown keys {unknown}; known keys: {sorted(known)}")
    if "gnc_c_px" in data:
        data["gnc_c_px"] = tuple(data["gnc_c_px"])
    return TbaConfig(**data)
