"""Reader for the NVIDIA V2D Challenge Track 3 PUBLIC split (CC BY 4.0), used as a metric-frame benchmark.

Source: huggingface.co/datasets/nvidia/video_to_data_challenge, folder ``track_3/public`` (20 episodes with Vicon
object poses, scanned meshes, OAK calibration and three ego streams). The evaluation videos (``track_3/evaluation``)
are never read here. Conventions, from the split's README and parquet schema metadata:

- ``observation.objects`` is ``list<struct{name, pose[x, y, z, qw, qx, qy, qz], visible}>``; membership is constant
  within an episode and sorted by name; the order is also in the schema metadata key ``objects``.
- ``pose`` is ``world_T_object`` in the SCANNED MESH frame (metres, quaternion w-first). x and y are centred per
  episode, z is not. When ``visible`` is false the pose is all zeros: such rows become NaN here.
- Video row i is frame i of every ego clip (frame-exact per the README).
- The calibration has NO camera-to-mocap-world extrinsic, so a camera-frame ground truth needs a separate construction
  (:mod:`fastdo.bench.v2d_camera_gt`). Everything in this module is frame-free.

Meshes are loaded the way the challenge kit's ``v2dlb/vertices.py`` loads them: ``trimesh.load(path, force="mesh",
process=False)``, preferring ``<name>_visual.glb`` over ``<name>.glb``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from fastdo import hostpaths

logger = logging.getLogger(__name__)

#: Revision of nvidia/video_to_data_challenge the local copy was downloaded at (2026-10-11, lastModified 2026-09-22).
HF_REVISION = "5f68335f3acc802033d1e80728c1633197521de8"
DEFAULT_ROOT = hostpaths.v2d_track3_root()   # $FASTDO_V2D_TRACK3_ROOT, default <ext root>/data/v2d_track3_public/...
EGO_STREAMS = ("ego_cam_a", "ego_cam_b", "ego_cam_c")

__all__ = [
    "DEFAULT_ROOT",
    "EGO_STREAMS",
    "HF_REVISION",
    "CameraCalib",
    "Episode",
    "StereoPair",
    "episode_indices",
    "load_calibration",
    "load_episode",
    "load_mesh",
    "mesh_path_for",
    "pose7_to_matrix",
    "video_path",
]


def pose7_to_matrix(pose7: np.ndarray) -> np.ndarray:
    """(..., 7) ``[x, y, z, qw, qx, qy, qz]`` -> (..., 4, 4) homogeneous transforms.

    The quaternion is normalised first. An all-zero quaternion (the split's encoding of an invisible row) gives NaN.
    """
    p = np.asarray(pose7, dtype=np.float64)
    if p.shape[-1] != 7:
        raise ValueError(f"pose must end in 7 values, got shape {p.shape}")
    t = p[..., :3]
    q = p[..., 3:]
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        q = np.where(norm > 0.0, q / norm, np.nan)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    rot = np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], axis=-1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], axis=-1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=-1),
    ], axis=-2)
    out = np.zeros(p.shape[:-1] + (4, 4), dtype=np.float64)
    out[..., :3, :3] = rot
    out[..., :3, 3] = t
    out[..., 3, 3] = 1.0
    bad = ~np.isfinite(rot).all(axis=(-1, -2))
    out[bad] = np.nan
    return out


def _root(root: str | Path) -> Path:
    """`root` as a Path; refused with the variable to set while it is still the /opt/fastdo placeholder."""
    return hostpaths.require(root, "FASTDO_V2D_TRACK3_ROOT")


@dataclass(frozen=True)
class CameraCalib:
    """One OAK camera's factory calibration (pixels at ``resolution``; metres for extrinsics)."""

    name: str
    resolution: tuple[int, int]          # (width, height) as stored
    K: np.ndarray                         # (3, 3)
    distortion_model: str
    distortion: np.ndarray                # (14,) OpenCV rational + thin-prism + tilt order, as stored
    hfov_deg: float
    rotation_deg: int                     # 180 on the ego rig: the device is mounted upside down
    extrinsics_to_stereo_left: np.ndarray | None  # (4, 4) or None when the file has none


@dataclass(frozen=True)
class StereoPair:
    """A calibrated stereo pair (left/right camera names, baseline in metres, left->right transform)."""

    rig: str
    left: str
    right: str
    baseline_m: float
    left_to_right: np.ndarray             # (4, 4)
    rotation_deg: int


def _as_matrix4(value: object) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape == (4, 4):
        return arr
    if arr.shape == (3, 4):
        out = np.eye(4)
        out[:3, :] = arr
        return out
    raise ValueError(f"extrinsic must be 3x4 or 4x4, got {arr.shape}")


def load_calibration(root: Path = DEFAULT_ROOT) -> tuple[dict[str, CameraCalib], list[StereoPair], dict]:
    """Parse ``meta/camera_calibration.json`` into cameras, stereo pairs and the raw dict (for provenance)."""
    raw = json.loads((_root(root) / "meta" / "camera_calibration.json").read_text())
    cams: dict[str, CameraCalib] = {}
    for name, c in raw["cameras"].items():
        ext = c.get("extrinsics_to_stereo_left")
        cams[name] = CameraCalib(
            name=name,
            resolution=(int(c["resolution"][0]), int(c["resolution"][1])),
            K=np.asarray(c["intrinsic_matrix"], dtype=np.float64),
            distortion_model=str(c["distortion_model"]),
            distortion=np.asarray(c["distortion_coefficients"], dtype=np.float64),
            hfov_deg=float(c["hfov_deg"]),
            rotation_deg=int(c["rotation_deg"]),
            extrinsics_to_stereo_left=None if ext is None else _as_matrix4(ext),
        )
    pairs = [
        StereoPair(rig=str(p["camera"]), left=str(p["left"]), right=str(p["right"]),
                   baseline_m=float(p["baseline_m"]), left_to_right=_as_matrix4(p["left_to_right"]),
                   rotation_deg=int(p["rotation_deg"]))
        for p in raw.get("stereo_pairs", [])
    ]
    return cams, pairs, raw


@dataclass(frozen=True)
class Episode:
    """One public episode's object trajectories (frame-free; world = the episode's mocap frame, xy-centred)."""

    index: int
    task: str
    objects: tuple[str, ...]
    frame_index: np.ndarray               # (T,)
    timestamp: np.ndarray                 # (T,) seconds
    world_T_object: np.ndarray            # (T, N_obj, 4, 4); NaN where not visible
    visible: np.ndarray                   # (T, N_obj) bool
    meta: dict = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return int(self.frame_index.shape[0])


def episode_indices(root: Path = DEFAULT_ROOT) -> list[int]:
    """Episode indices declared by ``meta/info.json``."""
    info = json.loads((_root(root) / "meta" / "info.json").read_text())
    return [int(i) for i in info["episode_indices"]]


def _task_names(root: Path) -> dict[int, str]:
    out: dict[int, str] = {}
    for line in (_root(root) / "meta" / "episodes_metadata.jsonl").read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            out[int(rec["episode_index"])] = str(rec["task"])
    return out


def load_episode(index: int, root: Path = DEFAULT_ROOT) -> Episode:
    """Read one episode parquet; invisible rows become NaN poses (never zero transforms)."""
    import pyarrow.parquet as pq

    path = _root(root) / "data" / "chunk-000" / f"episode_{index:06d}.parquet"
    table = pq.read_table(path)
    md = {k.decode(): v.decode() for k, v in (table.schema.metadata or {}).items()}
    if md.get("pose_convention", "world_T_object") != "world_T_object":
        raise ValueError(f"{path}: unexpected pose_convention {md.get('pose_convention')!r}")
    names = tuple(json.loads(md["objects"])) if "objects" in md else None
    rows = table.column("observation.objects").to_pylist()
    if names is None:
        names = tuple(o["name"] for o in rows[0])
    n_t, n_o = len(rows), len(names)
    pose7 = np.zeros((n_t, n_o, 7), dtype=np.float64)
    vis = np.zeros((n_t, n_o), dtype=bool)
    slot = {name: j for j, name in enumerate(names)}
    row_order_differs = 0
    for t, objs in enumerate(rows):
        row_names = [o["name"] for o in objs]
        if sorted(row_names) != sorted(names) or len(set(row_names)) != len(row_names):
            raise ValueError(f"{path}: row {t} objects {row_names} != {list(names)}")
        # The README says rows are sorted by name; episode 11 is not (measured 2026-10-11). Match by NAME, never by
        # position, and record how many rows differ from the metadata order.
        row_order_differs += int(tuple(row_names) != names)
        for o in objs:
            j = slot[o["name"]]
            pose7[t, j] = o["pose"]
            vis[t, j] = bool(o["visible"])
    md["row_order_differs_from_metadata"] = str(row_order_differs)
    mats = pose7_to_matrix(pose7)
    mats[~vis] = np.nan
    frame_index = np.asarray(table.column("frame_index").to_pylist(), dtype=np.int64)
    if not np.array_equal(frame_index, np.arange(n_t)):
        raise ValueError(f"{path}: frame_index is not 0..T-1")
    return Episode(
        index=int(index),
        task=_task_names(root).get(int(index), ""),
        objects=names,
        frame_index=frame_index,
        timestamp=np.asarray(table.column("timestamp").to_pylist(), dtype=np.float64),
        world_T_object=mats,
        visible=vis,
        meta=md,
    )


def mesh_path_for(name: str, root: Path = DEFAULT_ROOT) -> Path:
    """The scan to use for an object, preferring the visual variant (the kit's rule)."""
    directory = _root(root) / "mesh" / name
    for candidate in (f"{name}_visual.glb", f"{name}.glb"):
        path = directory / candidate
        if path.exists():
            return path
    raise FileNotFoundError(f"no mesh for object {name!r} under {directory}")


def load_mesh(name: str, root: Path = DEFAULT_ROOT) -> tuple[np.ndarray, np.ndarray]:
    """(vertices (V, 3) float64 metres in the scan frame, faces (F, 3) int64), loaded as the kit loads them."""
    import trimesh

    mesh = trimesh.load(str(mesh_path_for(name, root)), force="mesh", process=False)
    return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64)


def video_path(index: int, stream: str, root: Path = DEFAULT_ROOT) -> Path:
    """Path of one ego stream's mp4 for an episode (streams: ego_cam_a | ego_cam_b | ego_cam_c)."""
    if stream not in EGO_STREAMS:
        raise ValueError(f"stream must be one of {EGO_STREAMS}, got {stream!r}")
    return _root(root) / "videos" / "chunk-000" / f"observation.images.{stream}" / f"episode_{index:06d}.mp4"
