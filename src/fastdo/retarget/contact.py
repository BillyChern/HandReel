"""Contact measures on the object surface: human (MANO) and robot (Sharpa).

Both the human hand's contact and the robot hand's contact are represented as `ContactMeasure`s --
weighted point sets living on/near the SAME canonical (unit-scale) object surface -- so the kinematic retargeting's
contact-transport objective can compare them with optimal transport instead of per-hand keypoint
matching.

Band weight (design Sec. 2, both human and robot): for an atom at canonical-frame position `y` with
signed distance `phi = sdf(y)` (canonical units),

    w(y) = exp(-phi^2 / (2 * sigma_c^2))   if |phi| < tau_c   else 0

`sigma`/`tau` are supplied in metres (`sigma=0.004`, `tau=0.008` by default, i.e. 4 mm / 8 mm) and
divided by the metres-per-canonical-unit scale (`scale`/`mesh_scale`) once to get `sigma_c`/`tau_c`,
per the design doc. The cutoff at `|phi| = tau_c` is a hard indicator, not a smooth taper: weights
are C0 there (documented, not a bug -- these weights are only ever used inside a majorize-
minimize loop, where a boundary discontinuity is acceptable).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
from torch import Tensor

from fastdo.geometry.lie import quat_wxyz_to_matrix
from fastdo.solver.sdf import SdfGrid, sdf_lookup

__all__ = ["ContactMeasure", "human_contact_measure", "robot_contact_measure"]


class _SceneLike(Protocol):
    """Structural interface `robot_contact_measure` needs from a `fastdo.retarget.mj_model.RobotScene`.

    Kept as a local `Protocol` (rather than importing `RobotScene` itself) so `contact.py` states
    exactly what it uses and `mypy` can check the call site without an import-time coupling edge
    back onto `mj_model`; `RobotScene` satisfies this structurally, no inheritance needed.
    """

    def fk(self, q: np.ndarray) -> dict[str, np.ndarray]: ...

    def point_kinematics(self, q: np.ndarray, body_ids: np.ndarray, local_pts: np.ndarray,
                         jacobian: bool = ...) -> tuple[np.ndarray, np.ndarray | None]: ...

    @property
    def robot_dof(self) -> slice: ...

    @property
    def nv(self) -> int: ...


@dataclass(frozen=True)
class ContactMeasure:
    """A weighted point measure on the canonical object surface, for one hand over `T` frames.

    `points[k]` are the *projected* (nearest-surface-sample) positions of each atom in frame `k`;
    `raw_points[k]` are each atom's own canonical-frame position before that projection (`y_v` for
    the human measure, `r_u(q)` for the robot measure) -- design Sec. 4's v2 geodesic surface-distance
    cost needs both `pi_S(r)` (`points`) and `r` itself (`raw_points`). `sample_idx` is the index into
    the `surface (M,3)` array each atom was projected to, and `sdf_values` is each atom's raw
    (pre-projection) signed distance, kept for diagnostics. `weights` are the raw (unnormalized) band
    weights: `mass` is NOT normalized away here -- the kinematic retargeting's unbalanced OT
    term needs the true per-frame mass. `labels` is one finger label per atom (shared across all `T`
    frames: atom `v`/`u` always belongs to the same MANO vertex / robot surface sample, hence the same
    finger).
    """

    points: Tensor        # (T, N, 3) float, canonical frame, nearest surface sample per atom
    raw_points: Tensor    # (T, N, 3) float, canonical frame, each atom's own (unprojected) position
    weights: Tensor       # (T, N) float, raw band weights (not normalized)
    labels: Tensor        # (N,) int8, finger label per atom (fastdo.retarget.labels enum)
    sample_idx: Tensor    # (T, N) int64, index into `surface` each atom was projected to
    sdf_values: Tensor    # (T, N) float, phi(y) at each atom's own (unprojected) position

    @property
    def mass(self) -> Tensor:
        """Per-frame total mass `(T,)`: `mass[k] = weights[k].sum()`."""
        return self.weights.sum(dim=-1)


def _band_weights(phi: Tensor, sigma_c: float, tau_c: float) -> Tensor:
    """`exp(-phi^2 / 2 sigma_c^2)` where `|phi| < tau_c`, else 0. `phi`: any shape, canonical units."""
    raw = torch.exp(-(phi ** 2) / (2.0 * sigma_c ** 2))
    return torch.where(phi.abs() < tau_c, raw, torch.zeros_like(raw))


def _in_grid(sdf: SdfGrid, y: Tensor) -> Tensor:
    """True where `y` (…,3) falls *inside* `sdf`'s lattice before `sdf_lookup`'s border clamp.

    `sdf_lookup` clamps out-of-grid queries to the border cell rather than raising, so a raw
    `phi`/`grad` from it is defined (and can spuriously land inside the contact band) even for
    points well outside the grid. Vertices outside the SDF grid get
    weight 0, not an exception, so those atoms must be masked out explicitly -- this replicates
    `sdf_lookup`'s own normalization to decide which atoms that applies to.
    """
    G = sdf.resolution
    norm = (y - sdf.origin.to(y.dtype)) / (sdf.step * (G - 1)) * 2.0 - 1.0
    return (norm.abs() <= 1.0).all(dim=-1)


def _nearest_surface(y: Tensor, surface: Tensor) -> tuple[Tensor, Tensor]:
    """Nearest-sample projection `pi_S`: `y (T,N,3)`, `surface (M,3)` -> `(idx (T,N) int64, pts (T,N,3))`.

    Looped over `T` (not one `(T*N, M)` `cdist`) to keep peak memory to one frame's `(N, M)`
    distance matrix -- `T` can be in the hundreds and `M` in the low thousands, and both grow with
    the object/hand resolution independently of this function.
    """
    surf = surface.to(y.dtype)
    T = y.shape[0]
    idx = torch.empty(y.shape[:2], dtype=torch.int64, device=y.device)
    for t in range(T):
        d = torch.cdist(y[t], surf)               # (N, M)
        idx[t] = d.argmin(dim=1)
    return idx, surf[idx]

def human_contact_measure(hand_verts_cam: Tensor, poses: Tensor, scale: float, labels: Tensor,
                          sdf: SdfGrid, surface: Tensor, sigma: float = 0.004, tau: float = 0.008,
                          ) -> ContactMeasure:
    """Human (MANO) contact measure, design Sec. 2: camera-frame hand vertices -> canonical band weights.

    Args:
        hand_verts_cam: `(T, V, 3)` MANO vertices in the camera frame (metres), e.g. HaWoR's
            `{side}_vertices`.
        poses: `(T, 4, 4)` object-to-camera transforms (rotation + translation only), e.g. from
            `read_baseline_layout`.
        scale: Metres-per-canonical-unit object scale (`mesh_scale` from `read_baseline_layout`).
        labels: `(V,)` finger label per MANO vertex, e.g. `labels.mano_finger_labels`.
        sdf: Canonical-unit-mesh signed-distance grid (`mesh_sdf_grid` on the *unit-scale* mesh).
        surface: `(M, 3)` canonical-frame object surface samples atoms project onto.
        sigma: Band Gaussian width, metres.
        tau: Band half-width, metres.

    Returns:
        `ContactMeasure` with `points`/`raw_points`/`weights`/`sample_idx`/`sdf_values` of shape
        `(T, V, ...)`.
    """
    dtype = torch.float64
    h = hand_verts_cam.to(dtype)
    R, t = poses[:, :3, :3].to(dtype), poses[:, :3, 3].to(dtype)
    y = torch.einsum("tji,tvj->tvi", R, h - t[:, None, :]) / scale       # R^T (h - t) / scale
    phi, _ = sdf_lookup(sdf, y)
    w = _band_weights(phi, sigma / scale, tau / scale) * _in_grid(sdf, y).to(dtype)
    idx, pts = _nearest_surface(y, surface.to(dtype))
    return ContactMeasure(points=pts, raw_points=y, weights=w, labels=labels.to(torch.int8),
                          sample_idx=idx, sdf_values=phi)


def robot_contact_measure(q: Tensor, scene: _SceneLike, body_ids: np.ndarray, local_pts: np.ndarray,
                          labels: Tensor, sdf: SdfGrid, surface: Tensor, mesh_scale: float,
                          sigma: float = 0.004, tau: float = 0.008, jacobian: bool = False,
                          ) -> ContactMeasure | tuple[ContactMeasure, Tensor]:
    """Robot (Sharpa) contact measure, design Sec. 2: FK'd surface points -> canonical band weights.

    The object pose is read out of `q`'s own free-joint block (`q[:, 28:35]`, pos + wxyz quat --
    `RobotScene`'s qpos layout, see `mj_model.py`), not supplied separately: M2's per-frame solve
    holds the M1 object trajectory fixed and only varies the robot's own `robot_dof` (wrist+hand)
    columns, which is exactly the span `jacobian=True` differentiates over.

    Args:
        q: `(T, nq)` qpos, nq=35 (`RobotScene.model.nq`).
        scene: A `fastdo.retarget.mj_model.RobotScene` (typed here as `_SceneLike`, a small
            structural `Protocol`, to avoid an import-time dependency edge back onto `mj_model`).
        body_ids: `(U,)` MuJoCo body ids for `local_pts`, e.g. from `scene.surface_points`.
        local_pts: `(U, 3)` points in each `body_ids[u]`'s local frame.
        labels: `(U,)` finger label per point, e.g. `labels.robot_link_labels(scene.model, body_ids)`.
        sdf: Canonical-unit-mesh signed-distance grid (the SAME canonical mesh/grid as the human side).
        surface: `(M, 3)` canonical-frame object surface samples.
        mesh_scale: Metres-per-canonical-unit object scale (same convention/value as `scale` in
            `human_contact_measure`; the JSON-derived value, not a `scene.xml` mesh attribute, is
            authoritative here).
        sigma: Band Gaussian width, metres.
        tau: Band half-width, metres.
        jacobian: If True, additionally return `d weights / d q_robot`, `(T, U, 28)`.

    Returns:
        `ContactMeasure` alone, or `(ContactMeasure, d_weights_dq)` when `jacobian=True`.
    """
    dtype = torch.float64
    q_t = q.to(dtype) if isinstance(q, Tensor) else torch.as_tensor(q, dtype=dtype)
    q_np = q_t.detach().cpu().numpy()
    world_np, jac_np = scene.point_kinematics(q_np, body_ids, local_pts, jacobian=jacobian)
    world = torch.from_numpy(world_np).to(dtype)                          # (T, U, 3)

    obj_pos = q_t[:, 28:31]                                               # (T, 3)
    R_obj = quat_wxyz_to_matrix(q_t[:, 31:35])                            # (T, 3, 3): canonical -> world

    y = torch.einsum("tji,tuj->tui", R_obj, world - obj_pos[:, None, :]) / mesh_scale   # R_obj^T (.) / s
    phi, grad = sdf_lookup(sdf, y)                                        # (T,U), (T,U,3) d phi/d y
    sigma_c, tau_c = sigma / mesh_scale, tau / mesh_scale
    inside = _in_grid(sdf, y).to(dtype)
    w = _band_weights(phi, sigma_c, tau_c) * inside
    idx, pts = _nearest_surface(y, surface.to(dtype))
    measure = ContactMeasure(points=pts, raw_points=y, weights=w, labels=labels.to(torch.int8),
                             sample_idx=idx, sdf_values=phi)
    if not jacobian:
        return measure

    assert jac_np is not None
    jac = torch.from_numpy(jac_np).to(dtype)                              # (T, U, 3, 28)
    # Chain rule:
    #   d phi/d q = (R_obj @ grad) . (d world / d q) / mesh_scale
    #   d w/d q   = (d w/d phi) * (d phi/d q),  d w/d phi = -phi / sigma_c^2 * w  (0 outside the band)
    a = torch.einsum("tji,tui->tuj", R_obj, grad)                         # (T, U, 3) = R_obj @ grad
    dphi_dq = torch.einsum("tuj,tujk->tuk", a, jac) / mesh_scale          # (T, U, 28)
    dw_dphi = -phi / (sigma_c ** 2) * w                                    # (T, U), already 0 outside band
    dw_dq = dw_dphi[..., None] * dphi_dq                                   # (T, U, 28)
    return measure, dw_dq
