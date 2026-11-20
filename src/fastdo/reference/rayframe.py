"""Per-row camera RAY frames (identifiable components of an object's pose noise).

Monocular reconstruction cannot identify an object's translation ALONG the camera-to-object ray; the two
perpendicular directions and the rotation are identifiable. The decomposition therefore splits every row's world
residual in that row's own ray frame:

- u_i  = unit(target_i - C_i): the along-ray axis, C_i the camera centre of row i (world), target_i the object centre
  the ray points at (the run's / lead's ROBUST centre is used, not the row's own noisy centre, so the axis does not
  move with the noise it decomposes);
- e1_i = unit(x_i - (x_i . u_i) u_i): the camera's x axis (column 0 of the camera-to-world rotation) projected off the
  ray (column 1, the camera's y axis, when x is within 1e-6 of the ray);
- e2_i = u_i x e1_i.

`A_i = [u_i, e1_i, e2_i]` (columns, world coordinates) is a rotation; `to_ray(A, v)` gives a world vector's
components (along-ray, perp-1, perp-2).
"""

from __future__ import annotations

import numpy as np

__all__ = ["ray_frames", "to_ray"]


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def ray_frames(cam_centres: np.ndarray, cam_rots: np.ndarray, target: np.ndarray) -> np.ndarray:
    """(n,3,3) ray frames [u, e1, e2] (columns, world) of n rows (module docstring).

    Args:
        cam_centres: (n,3) camera centres in the world (m).
        cam_rots: (n,3,3) camera-to-world rotations (rigid).
        target: (3,) or (n,3) world point(s) the rays point at (m).
    """
    C = np.asarray(cam_centres, dtype=np.float64)
    Rc = np.asarray(cam_rots, dtype=np.float64)
    T = np.broadcast_to(np.asarray(target, dtype=np.float64), C.shape)
    u = _unit(T - C)
    x = Rc[:, :, 0]
    xp = x - np.sum(x * u, axis=1, keepdims=True) * u
    bad = np.linalg.norm(xp, axis=1) < 1e-6
    if bad.any():
        y = Rc[bad, :, 1]
        xp[bad] = y - np.sum(y * u[bad], axis=1, keepdims=True) * u[bad]
    e1 = _unit(xp)
    e2 = np.cross(u, e1)
    return np.stack([u, e1, e2], axis=2)


def to_ray(A: np.ndarray, v: np.ndarray) -> np.ndarray:
    """(n,3) components (along-ray, perp-1, perp-2) of world vectors `v (n,3)` in the ray frames `A (n,3,3)`."""
    return np.einsum("nji,nj->ni", np.asarray(A, dtype=np.float64), np.asarray(v, dtype=np.float64))
