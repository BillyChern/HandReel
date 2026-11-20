"""The hand-contact along-ray correction: the object's depth from where the hand touches it.

**Why it exists.** A monocular SE(3)^T x R+ solve cannot see along-ray placement well: a global rescaling moves no
pixel, so the tracks carry exactly zero information about it, and in the shipped configuration along-ray
placement is the binding quality constraint. A hand that is *holding* the object supplies that missing
coordinate directly: touching needs no relative-depth comparison, so the hand's own depth pins the object's
(34/38 clips within 25 mm of a 10 mm ruler).

**The estimator, and why this form.** For a frame, slide the solved object along the ray to its predicted centroid
and record the mean joint-to-surface distance, `C(delta)`. Taking `argmin C` and calibrating it linearly is the
earlier form; it loses on two counts. The grid truncates (28 % of frames sit on a +-120 mm bound, and those carry the
LARGEST errors), and an argmin throws away the curvature that says how much to trust the reading. So the
promoted form (held out on 49 clips) treats the profile as a likelihood and returns the POSTERIOR MEAN of the
correction `c`:

    p(c) ~ N(mu, sig^2)                                     the along-ray error's own distribution, calibrated
    L(c) ~ exp(-(C(delta) - min C)^2 / (2 tau^2)),           c = a * delta + b on the grid
    c* = E[c | C]

`a, b` map a slide to a correction (Theil-Sen, so a few wild frames cannot steer it), `sig` is the MAD of the
along-ray error, and `tau` is the hand-noise scale -- 20 mm, fixed a priori from the measured HaWoR ruler noise
(MAD 14.8 mm right, 22.2 mm left), never fitted. The two limits are the reason to prefer it: a sharp
profile reduces to the calibrated argmin, a flat one returns the prior mean, and everything between is shrunk by how
informative the contact actually is -- which is the shrinkage the earlier form had to add by hand, now implied.

**Measured (49 held-out clips, production depth, HaWoR hands, constants frozen beforehand):** 43/49 clips
improved, sign p = 5.7e-8; the along-ray median fell 73.7 -> 37.6 mm; the along-ray quality pass rate went 8/49 ->
17-20/49.
Measured cost of this implementation: 211 ms per 25-frame clip at the ±300 mm grid (8.4 ms per frame,
2048 mesh points, 21 joints), 85 ms at ±120 mm -- 1.2 % of an 18.27 s batched clip. A KD-tree query per grid point
dominates, so a coarse-then-refine search would cut it about 5x.

Constants are per depth level and per hand source; a constants JSON (`m88c_production_constants.json`) holds the
production fit (all 87 GT-labelled clips). This module never fits anything.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

__all__ = ["ContactConstants", "ContactCorrection", "contact_correction", "contact_identifiability",
           "contact_profile", "correct_poses", "load_constants", "posterior_mean_shift", "posterior_shift_moments"]

DEFAULT_GRID_MM = np.arange(-300, 301, 2, dtype=float)


@dataclass(frozen=True)
class ContactConstants:
    """The calibrated map from a contact reading to an along-ray correction.

    Attributes:
        a: Slope of correction against slide (dimensionless).
        b_mm: Intercept, in mm.
        mu_mm: Prior mean of the correction, in mm.
        sig_mm: Prior standard deviation (a MAD in practice), in mm; must be positive.
        tau_mm: Likelihood width in cost units (mm of mean joint-to-surface distance); must be positive.
    """

    a: float
    b_mm: float
    mu_mm: float
    sig_mm: float
    tau_mm: float = 20.0

    def __post_init__(self) -> None:
        for name in ("a", "b_mm", "mu_mm", "sig_mm", "tau_mm"):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite, got {getattr(self, name)!r}")
        if self.sig_mm <= 0:
            raise ValueError(f"sig_mm must be > 0, got {self.sig_mm}")
        if self.tau_mm <= 0:
            raise ValueError(f"tau_mm must be > 0, got {self.tau_mm}")

    def bias_free(self) -> ContactConstants:
        """The same ruler with its two BIAS terms zeroed (`b_mm = mu_mm = 0`); slope, spread and width unchanged.

        Why: `b` and `mu` encode the mean along-ray error of the corpus the constants were fitted on
        (DexYCB production solves sat too FAR, so both are about -25 / -40 mm). On a corpus whose pre-contact
        solve is unbiased or sits too NEAR (egocentric HOI4D: -25 mm), they add a ~-44 mm nearward pull on every
        corrected frame. Zeroing them keeps what transfers (the contact reading's slope and its shrinkage) and
        drops what does not (one corpus's solve bias). Fixed a priori, never fitted.
        """
        return dataclasses.replace(self, b_mm=0.0, mu_mm=0.0)

    def geometric(self) -> ContactConstants:
        """The GEOMETRIC ruler: slope `a = 1` and both bias terms zeroed; spread and width unchanged.

        Why: the shipped slope (a = 0.68) and biases were fitted at calibrated K on DexYCB. Applied offline at every
        focal operating point (estimated, debiased, GT), this un-fitted ruler beat the shipped one on DexYCB by -0.12
        to -0.15 median log centre error, with no sign flip. Nothing here is fitted.
        """
        return dataclasses.replace(self, a=1.0, b_mm=0.0, mu_mm=0.0)


def load_constants(path: Path, cell: str, hands: str) -> ContactConstants:
    """Reads `<cell>/<hands>` from a constants JSON (the `m88c_production_constants.json` schema).

    Raises:
        KeyError: If the file has no entry for that cell and hand source.
    """
    spec = json.loads(Path(path).read_text())
    key = f"{cell}/{hands}"
    if key not in spec.get("constants", {}):
        raise KeyError(f"{path}: no constants for {key!r}; have {sorted(spec.get('constants', {}))}")
    c = spec["constants"][key]
    return ContactConstants(a=float(c["a"]), b_mm=float(c["b_mm"]), mu_mm=float(c["mu_mm"]),
                            sig_mm=float(c["sig_mm"]), tau_mm=float(spec.get("tau_mm", 20.0)))


def contact_profile(surface_cam: np.ndarray, joints_cam: np.ndarray, ray_unit: np.ndarray,
                    grid_m: np.ndarray) -> np.ndarray:
    """The contact cost `C(delta)` in MILLIMETRES, for sliding the object by `delta` along `ray_unit`.

    Sliding the object by `+delta` is the same as sliding the joints by `-delta`, which is what this evaluates, so
    the surface's KD-tree is built once.

    Args:
        surface_cam: `(M,3)` object surface points in the camera frame, at the solved pose and scale.
        joints_cam: `(J,3)` hand joints in the camera frame. Rows with a non-finite entry are dropped.
        ray_unit: `(3,)` unit vector to the object's predicted centroid.
        grid_m: `(K,)` slides in METRES.

    Returns:
        `(K,)` mean joint-to-surface distance in mm, one per grid point.

    Raises:
        ValueError: If no joint is finite, or the shapes are wrong.
    """
    surface = np.asarray(surface_cam, dtype=float)
    joints = np.asarray(joints_cam, dtype=float)
    ray = np.asarray(ray_unit, dtype=float).reshape(3)
    if surface.ndim != 2 or surface.shape[1] != 3:
        raise ValueError(f"surface_cam must be (M,3), got {surface.shape}")
    if joints.ndim != 2 or joints.shape[1] != 3:
        raise ValueError(f"joints_cam must be (J,3), got {joints.shape}")
    joints = joints[np.isfinite(joints).all(axis=1)]
    if joints.size == 0:
        raise ValueError("no finite hand joint")
    tree = cKDTree(surface)
    return np.array([tree.query(joints - d * ray, k=1)[0].mean() for d in np.asarray(grid_m, dtype=float)]) * 1e3


def _posterior_weights(profile_mm: np.ndarray, grid_mm: np.ndarray,
                       constants: ContactConstants) -> tuple[np.ndarray, np.ndarray]:
    """`(c_grid, w)`: the correction each grid point implies, and its unnormalised posterior weight.

    In log space so a huge cost range cannot overflow. Shared by the mean and the moments, so the promoted
    estimator and anything derived from it can never drift apart.
    """
    profile = np.asarray(profile_mm, dtype=float)
    grid = np.asarray(grid_mm, dtype=float)
    if profile.shape != grid.shape:
        raise ValueError(f"profile {profile.shape} and grid {grid.shape} must match")
    c_grid = constants.a * grid + constants.b_mm
    log_w = (-((profile - profile.min()) ** 2) / (2.0 * constants.tau_mm ** 2)
             - ((c_grid - constants.mu_mm) ** 2) / (2.0 * constants.sig_mm ** 2))
    return c_grid, np.exp(log_w - log_w.max())


def posterior_mean_shift(profile_mm: np.ndarray, grid_mm: np.ndarray, constants: ContactConstants) -> float:
    """`E[c | C]` in mm: the module docstring's estimator, in log space so a huge cost range cannot overflow."""
    c_grid, w = _posterior_weights(profile_mm, grid_mm, constants)
    return float((w * c_grid).sum() / w.sum())


def posterior_shift_moments(profile_mm: np.ndarray, grid_mm: np.ndarray,
                            constants: ContactConstants) -> tuple[float, float]:
    """`(E[c | C], sd[c | C])` in mm. The mean is `posterior_mean_shift` exactly; the sd is what it discards.

    The estimator already shrinks toward the prior by how informative the contact is (module docstring), but
    only the shrunk point estimate survives. The spread is the other half of that same posterior: it says how
    tightly the contact evidence pinned the object along the ray.
    """
    c_grid, w = _posterior_weights(profile_mm, grid_mm, constants)
    total = w.sum()
    mean = float((w * c_grid).sum() / total)
    var = float((w * (c_grid - mean) ** 2).sum() / total)
    return mean, float(np.sqrt(max(var, 0.0)))


def contact_identifiability(profile_mm: np.ndarray, grid_mm: np.ndarray, constants: ContactConstants) -> float:
    """`1 - Var[c | C] / Var[c]`: the fraction of the prior's variance the contact evidence removed.

    Dimensionless, so it compares across clips, depth levels and hand sources. 0 means the profile said
    nothing and the posterior is the prior; 1 means the contact pinned the correction exactly. It can go
    NEGATIVE, and that is meaningful rather than a defect: a profile with minima at both ends of the grid
    spreads the posterior wider than the prior, which is the signature of an actively ambiguous contact.

    **Both variances are taken on the SAME grid.** Normalising by `sig_mm ** 2` would be wrong: the grid
    truncates the prior, so on a narrow grid the posterior variance is bounded below the prior's untruncated
    variance no matter what the profile says, and an uninformative profile would score as well identified.
    `test_a_narrow_grid_cannot_fake_identifiability` holds that trap shut.

    This is the quantity the shipped 5 acceptance features cannot express: they measure residuals and
    confidences, none of which sees ALONG-RAY placement -- the direction measured to be the binding error
    (79.8 mm along-ray against 0.9 mm perpendicular, 38/38 clips).

    Raises:
        ValueError: If the prior has no spread on this grid (one grid point, or a degenerate `a`).
    """
    c_grid, _ = _posterior_weights(profile_mm, grid_mm, constants)
    _, post_sd = posterior_shift_moments(profile_mm, grid_mm, constants)
    prior = np.exp(-((c_grid - constants.mu_mm) ** 2) / (2.0 * constants.sig_mm ** 2))
    prior_mean = float((prior * c_grid).sum() / prior.sum())
    prior_var = float((prior * (c_grid - prior_mean) ** 2).sum() / prior.sum())
    if not prior_var > 0:
        raise ValueError("the prior has no variance on this grid, so identifiability is undefined")
    return 1.0 - post_sd ** 2 / prior_var


@dataclass(frozen=True)
class ContactCorrection:
    """Everything `contact_correction` computed per frame.

    Attributes:
        poses: `(T,4,4)` corrected poses (a copy; unchanged rows equal the input).
        shift_mm: `(T,)` applied along-ray shift, 0 where the frame was not corrected.
        min_cost_mm: `(T,)` `min_delta C(delta)` -- the smallest mean joint-to-surface distance the hand reaches
            anywhere along the ray; NaN where no profile was computed (no valid hand, or no ray).
        gated: `(T,)` bool, frames that HAD a profile but were left uncorrected because `min_cost_mm` exceeded
            the touch gate.
    """

    poses: np.ndarray
    shift_mm: np.ndarray
    min_cost_mm: np.ndarray
    gated: np.ndarray


def contact_correction(poses: np.ndarray, scale: float, mesh_points: np.ndarray, joints: np.ndarray,
                       hand_valid: np.ndarray, constants: ContactConstants, grid_mm: np.ndarray | None = None,
                       touch_gate_mm: float | None = None) -> ContactCorrection:
    """`correct_poses`, plus the per-frame minimum contact cost and an optional touch gate.

    **The touch gate (opt-in).** The posterior's likelihood uses `C(delta) - min C`, i.e. only the SHAPE of
    the profile: a frame whose hand never comes within 60 mm of the surface at any depth still gets a full-strength
    correction around its argmin. The absolute level `min C` is the touching evidence the estimator discards. It is
    label-free and depth-free (the slide makes it invariant to the hand's along-ray error, which is what the
    correction exists to fix). With `touch_gate_mm` set, a frame whose `min C` exceeds it is left exactly as solved.
    `None` (the default) reproduces the promoted behaviour bit for bit.

    Args:
        poses, scale, mesh_points, joints, hand_valid, constants, grid_mm: as `correct_poses`.
        touch_gate_mm: Maximum `min C` (mm) at which a frame counts as touching; None disables the gate.

    Raises:
        ValueError: On shape mismatches or a non-positive / non-finite gate.
    """
    poses = np.asarray(poses, dtype=float)
    mesh = np.asarray(mesh_points, dtype=float)
    joints = np.asarray(joints, dtype=float)
    valid = np.asarray(hand_valid, dtype=bool).reshape(-1)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"poses must be (T,4,4), got {poses.shape}")
    if joints.ndim != 3 or joints.shape[0] != poses.shape[0] or valid.shape[0] != poses.shape[0]:
        raise ValueError(f"joints {joints.shape} and hand_valid {valid.shape} must have {poses.shape[0]} rows")
    if touch_gate_mm is not None and not (np.isfinite(touch_gate_mm) and touch_gate_mm > 0):
        raise ValueError(f"touch_gate_mm must be a positive finite number or None, got {touch_gate_mm!r}")
    grid = DEFAULT_GRID_MM if grid_mm is None else np.asarray(grid_mm, dtype=float)
    centroid = mesh.mean(axis=0)

    out = poses.copy()
    shift_mm = np.zeros(poses.shape[0], dtype=float)
    min_cost = np.full(poses.shape[0], np.nan, dtype=float)
    gated = np.zeros(poses.shape[0], dtype=bool)
    for t in range(poses.shape[0]):
        if not valid[t] or not np.isfinite(joints[t]).all(axis=1).any():
            continue
        rot, trans = poses[t, :3, :3], poses[t, :3, 3]
        surface = scale * (mesh @ rot.T) + trans
        cp = scale * (rot @ centroid) + trans
        norm = float(np.linalg.norm(cp))
        if not np.isfinite(norm) or norm <= 0:            # a centroid at or behind the camera has no ray
            continue
        ray = cp / norm
        profile = contact_profile(surface, joints[t], ray, grid / 1e3)
        min_cost[t] = float(profile.min())
        if touch_gate_mm is not None and not min_cost[t] <= touch_gate_mm:
            gated[t] = True
            continue
        shift = posterior_mean_shift(profile, grid, constants)
        shift_mm[t] = shift
        out[t, :3, 3] = trans + (shift / 1e3) * ray
    return ContactCorrection(poses=out, shift_mm=shift_mm, min_cost_mm=min_cost, gated=gated)


def correct_poses(poses: np.ndarray, scale: float, mesh_points: np.ndarray, joints: np.ndarray,
                  hand_valid: np.ndarray, constants: ContactConstants,
                  grid_mm: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Applies the correction to every frame that has a valid hand; leaves the others exactly as they were.

    Only the translation moves, and only along the ray to the object's predicted centroid: the correction claims
    nothing about the two coordinates the image already constrains (the branch is a pure along-ray ambiguity).

    Args:
        poses: `(T,4,4)` solved object poses (camera frame).
        scale: The solve's scale, applied to `mesh_points` as the solver applies it.
        mesh_points: `(M,3)` template points in the object frame.
        joints: `(T,J,3)` hand joints in the camera frame; non-finite rows are ignored.
        hand_valid: `(T,)` bool, whether that frame has a usable hand.
        constants: The calibrated constants for this depth level and hand source.
        grid_mm: Slide grid in mm; the default is -300..300 in 2 mm steps.

    Returns:
        `(poses_corrected, shift_mm)`: new poses, and the applied shift per frame (0 where no hand).

    Raises:
        ValueError: On shape mismatches.
    """
    res = contact_correction(poses, scale, mesh_points, joints, hand_valid, constants, grid_mm)
    return res.poses, res.shift_mm
