"""World(video) -> robot-base placement from the object's start pose on the table.

Robot frame: the MuJoCo world of `fastdo.robot.g1_model` -- origin on the floor directly below the pelvis origin, x
forward, y left, z up; the pelvis origin is at z = 0.793 m (the official MJCF's standing height).

Video world: the frame of the `ClipRecord` (`object_pose`, `cam_to_world`, hand joints moved through `cam_to_world`),
metric, with `up_world` the up direction (`cam_to_world[0] @ gravity_cam`).

`place_on_table` returns ONE rigid transform `T_rv` (no scaling: the reconstruction is metric and the hand-object
geometry must be preserved), built in three documented steps:
1. Gravity: the minimal rotation taking `up_world` to robot +z.
2. Heading: a yaw that takes the horizontal part of the egocentric camera's forward axis at frame 0 (the human's view
   direction) to robot +x (the robot looks along +x), plus `yaw_offset`. When the camera looks almost straight down
   (horizontal part < 0.2), the camera's image-up axis (-y_cam) gives the heading instead.
3. Translation: the object's start pose is moved so its centre lands on `table_xy` and its LOWEST mesh point (mesh
   vertices x scale, at the start pose) sits `clearance` above the table top `table_z`.
The lab places the real look-alike object at `object_start` (the package README prints it); `export_g1_package
--object-xy/--table-z/--yaw-offset` re-exports for a different measured start pose.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Placement", "camera_heading", "footprint_inside", "gravity_rotation", "pitched_up", "place_on_table",
           "transform_points", "transform_poses"]


@dataclass(frozen=True)
class Placement:
    T_rv: np.ndarray            # (4,4) video world -> robot frame
    object_start: np.ndarray    # (4,4) object pose (canonical mesh -> robot frame, before scale) at frame 0
    table_z: float
    table_xy: tuple[float, float]
    yaw: float                  # heading yaw applied in step 2 (rad), including yaw_offset


def gravity_rotation(up: np.ndarray) -> np.ndarray:
    """`(3,3)` minimal rotation taking `up` to +z (Rodrigues; a pi turn about x for up = -z)."""
    u = np.asarray(up, dtype=np.float64)
    norm = float(np.linalg.norm(u))
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError(f"up must be a finite non-zero 3-vector, got {u.tolist()}")
    u = u / norm
    c = float(u[2])
    if c <= -1.0 + 1e-12:
        return np.diag([1.0, -1.0, -1.0])
    v = np.cross(u, [0.0, 0.0, 1.0])
    k = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + k + k @ k / (1.0 + c)


def camera_heading(cam_to_world0: np.ndarray, r_up: np.ndarray, min_horizontal: float = 0.2) -> np.ndarray:
    """Unit horizontal heading `(2,)` in the gravity-aligned frame `r_up @ world` (step 2 of the module docstring)."""
    rot = r_up @ np.asarray(cam_to_world0, dtype=np.float64)[:3, :3]
    fwd = rot[:, 2]
    if np.linalg.norm(fwd[:2]) < min_horizontal:
        fwd = -rot[:, 1]
    h = fwd[:2]
    norm = float(np.linalg.norm(h))
    if norm < 1e-9:
        raise ValueError("camera has no horizontal heading")
    return h / norm


def transform_poses(T: np.ndarray, poses: np.ndarray) -> np.ndarray:
    """`T @ poses` for `(…,4,4)` poses."""
    return np.einsum("ij,...jk->...ik", T, poses)


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Rigid transform of `(…,3)` points."""
    return points @ T[:3, :3].T + T[:3, 3]


def pitched_up(cam_to_world0: np.ndarray, pitch_deg: float) -> np.ndarray:
    """`(3,)` world up of a camera pitched DOWN by `pitch_deg` from level: `(0, -cos, -sin)` in the OpenCV camera frame
    (x right, y down, z optical axis) of frame 0, moved to the world by `cam_to_world0`. An operator assumption for
    reconstructions whose gravity was not estimated (QC `gravity_source` "default": up = the camera's -y)."""
    th = np.radians(float(pitch_deg))
    return np.asarray(cam_to_world0, dtype=np.float64)[:3, :3] @ np.array([0.0, -np.cos(th), -np.sin(th)])


def footprint_inside(object_start: np.ndarray, mesh_vertices: np.ndarray, scale: float,
                     table_x: tuple[float, float], table_y: tuple[float, float], margin: float = 0.02) -> bool:
    """Whether the placed object's footprint (robot-frame xy of its mesh vertices x `scale` at `object_start`) lies
    inside the table rectangle shrunk by `margin` (m). An overhanging object tips off the edge in the replay
    (smoke test, clip C1: the placement search put a 5-6 cm object's centre 2 cm from the near edge; it slid 0.65 m
    and fell)."""
    pts = (scale * np.asarray(mesh_vertices, dtype=np.float64)) @ object_start[:3, :3].T + object_start[:3, 3]
    lo = np.array([table_x[0] + margin, table_y[0] + margin])
    hi = np.array([table_x[1] - margin, table_y[1] - margin])
    return bool((pts[:, :2] >= lo).all() and (pts[:, :2] <= hi).all())


def place_on_table(object_pose0: np.ndarray, mesh_vertices: np.ndarray, scale: float, up_world: np.ndarray,
                   cam_to_world0: np.ndarray, table_xy: tuple[float, float], table_z: float, *,
                   yaw_offset: float = 0.0, clearance: float = 0.001) -> Placement:
    """The placement of the module docstring.

    Args:
        object_pose0: `(4,4)` object pose at frame 0 in the video world (canonical mesh point y -> scale R y + t).
        mesh_vertices: `(V,3)` canonical mesh vertices.
        scale: Metres per canonical unit.
        up_world: `(3,)` up direction in the video world.
        cam_to_world0: `(4,4)` camera-to-world of frame 0 (its +z is the optical axis).
        table_xy: Robot-frame `(x, y)` of the object's centre at the start.
        table_z: Robot-frame height of the table top (m).
        yaw_offset: Extra yaw about robot +z (rad) after the heading alignment.
        clearance: Gap between the lowest object point and the table top (m).
    """
    r_up = gravity_rotation(up_world)
    h = camera_heading(cam_to_world0, r_up)
    yaw = -float(np.arctan2(h[1], h[0])) + yaw_offset
    cz, sz = np.cos(yaw), np.sin(yaw)
    rot = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]]) @ r_up
    pose = np.asarray(object_pose0, dtype=np.float64)
    pts = (scale * np.asarray(mesh_vertices, dtype=np.float64)) @ pose[:3, :3].T + pose[:3, 3]
    pts_r = pts @ rot.T
    centre = rot @ pose[:3, 3]
    t = np.array([table_xy[0] - centre[0], table_xy[1] - centre[1], table_z + clearance - pts_r[:, 2].min()])
    T_rv = np.eye(4)
    T_rv[:3, :3] = rot
    T_rv[:3, 3] = t
    return Placement(T_rv=T_rv, object_start=T_rv @ pose, table_z=float(table_z),
                     table_xy=(float(table_xy[0]), float(table_xy[1])), yaw=yaw)
