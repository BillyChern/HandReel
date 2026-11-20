"""Finger-label enum, MANO vertex segmentation, and Sharpa link-name classification.

Shared by `human_contact_measure`/`robot_contact_measure` (`contact.py`): every atom of a
`ContactMeasure` carries one of these six labels, so the contact-transport cost can add
a soft anatomical term (thumb-to-thumb, etc.) on top of surface distance.
"""

from __future__ import annotations

import logging
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)

__all__ = [
    "FINGER_LABEL_NAMES",
    "INDEX",
    "MIDDLE",
    "PALM",
    "PINKY",
    "RING",
    "THUMB",
    "mano_finger_labels",
    "robot_link_labels",
]

# Finger-label enum, shared by both hands (MANO) and the robot (Sharpa Wave).
PALM, THUMB, INDEX, MIDDLE, RING, PINKY = 0, 1, 2, 3, 4, 5
FINGER_LABEL_NAMES: tuple[str, ...] = ("palm", "thumb", "index", "middle", "ring", "pinky")

# MANO's 16-joint kinematic tree (joint 0 = wrist/root) grouped into fingers in the model's own
# joint order: index(1-3), middle(4-6), pinky(7-9), ring(10-12), thumb(13-15) -- confirmed against
# the shipped MANO_RIGHT.pkl's `kintree_table`. Joint 0 (root/wrist) maps
# to palm.
_MANO_JOINT_TO_LABEL: tuple[int, ...] = (
    PALM,                                  # 0: root/wrist
    INDEX, INDEX, INDEX,                   # 1-3
    MIDDLE, MIDDLE, MIDDLE,                # 4-6
    PINKY, PINKY, PINKY,                   # 7-9
    RING, RING, RING,                      # 10-12
    THUMB, THUMB, THUMB,                   # 13-15
)

# Sharpa Wave body-name -> finger regex table (checked in order; first match wins). Bodies that
# match none of these (wrist chain `right_base_*`, palm `right_hand_C_MC`, and any reference/site
# body) fall back to palm. Table is intentionally simple substring regexes on the compiled model's
# own body names (case-insensitive) rather than a hardcoded id list, so it survives left/right or
# naming-suffix variants of the same MJCF family.
_LINK_NAME_TO_LABEL: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"thumb", re.IGNORECASE), THUMB),
    (re.compile(r"index", re.IGNORECASE), INDEX),
    (re.compile(r"middle", re.IGNORECASE), MIDDLE),
    (re.compile(r"ring", re.IGNORECASE), RING),
    (re.compile(r"pinky", re.IGNORECASE), PINKY),
)


class _ChumpyStub:
    """Stand-in for any `chumpy.*` class referenced by a pickled MANO model.

    The official MANO pickle stores its LBS blend weights (`weights`) and kinematic tree
    (`kintree_table`) as plain numpy arrays, but a handful of *other* fields (`shapedirs`,
    `posedirs`'s defining expression, ...) are `chumpy.Ch` autodiff nodes. Standard `pickle.load`
    must be able to resolve every class referenced anywhere in the file, even ones whose value is
    never read, so loading the file at all normally requires the (old, numpy<2-only) `chumpy`
    package installed. Since this module only ever reads `weights`, every chumpy-backed object is
    substituted with this inert stub instead of installing chumpy as a real dependency.
    """

    def __setstate__(self, state: Any) -> None:
        if isinstance(state, dict):
            self.__dict__.update(state)

    def __reduce__(self) -> tuple[type[_ChumpyStub], tuple[()]]:
        return (_ChumpyStub, ())


class _MANOUnpickler(pickle.Unpickler):
    """`pickle.Unpickler` that redirects any `chumpy.*` class to `_ChumpyStub`."""

    def find_class(self, module: str, name: str) -> Any:
        if module.startswith("chumpy"):
            return _ChumpyStub
        return super().find_class(module, name)


def _default_cache_dir() -> Path:
    """`<repo root>/data/mano` -- `data/` is git-ignored repo-wide (see `.gitignore`)."""
    return Path(__file__).resolve().parents[3] / "data" / "mano"


def _infer_side(pkl_path: Path) -> str:
    name = str(pkl_path).lower()
    has_right, has_left = "right" in name, "left" in name
    if has_right == has_left:
        raise ValueError(f"cannot infer hand side (left/right) from path {pkl_path}; "
                         "expected exactly one of 'left'/'right' in the path")
    return "right" if has_right else "left"


def mano_finger_labels(pkl_path: str | Path, cache_dir: str | Path | None = None) -> Tensor:
    """MANO per-vertex finger labels `(778,)` int8, derived from the model's own LBS skinning weights.

    For each of the 778 template vertices, takes the kinematic-tree joint with the largest blend
    weight (`weights` is `(778,16)`) and maps that joint to a finger via `_MANO_JOINT_TO_LABEL`.
    Never requires `chumpy` to be installed (see `_ChumpyStub`) and never commits MANO-derived data:
    the result is cached at `<cache_dir or repo root>/data/mano/finger_labels_<side>.npy`, which is
    covered by the repo-wide `/data/` gitignore entry.

    Args:
        pkl_path: Path to `MANO_RIGHT.pkl` or `MANO_LEFT.pkl`. Side (for the cache filename) is
            inferred from whether "right" or "left" (case-insensitive) appears in the path.
        cache_dir: Directory for the cached `.npy`; defaults to `<repo root>/data/mano`.

    Raises:
        FileNotFoundError: If `pkl_path` does not exist.
        ValueError: If the side cannot be inferred from `pkl_path`, or `weights` has an unexpected shape.
    """
    pkl_path = Path(pkl_path)
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"MANO pickle not found at {pkl_path}. Typical locations for MANO_RIGHT.pkl/"
            "MANO_LEFT.pkl: this repo's own data/mano/, or a HaWoR checkout's "
            "_DATA/data/mano/ (right) / _DATA/data_left/mano_left/ (left). The config finds them through "
            "FASTDO_MANO_DIR and FASTDO_MANO_LEFT_DIR (fastdo.hostpaths.mano_dir / mano_left_dir)."
        )
    side = _infer_side(pkl_path)
    cache_path = (Path(cache_dir) if cache_dir is not None else _default_cache_dir()) / f"finger_labels_{side}.npy"
    if cache_path.exists():
        logger.info("loading cached MANO finger labels from %s", cache_path)
        return torch.from_numpy(np.load(cache_path)).to(torch.int8)

    # Security note: this unpickles a local MANO model file the caller supplies explicitly (found
    # under this machine's own data directories, never fetched over the network or accepted from an
    # untrusted upload) -- the standard, unavoidable MANO/SMPL distribution format -- and it is
    # trusted local data, not adversarial input. `_MANOUnpickler` substitutes chumpy classes so the
    # pickle loads without the (unmaintained, numpy<2-incompatible) `chumpy` package installed; it
    # does NOT sandbox or restrict what the stream can execute -- every non-`chumpy` class still
    # resolves through the real, unmodified `pickle.Unpickler.find_class`, exactly as plain
    # `pickle.load` would, so this offers no protection against a malicious pickle.
    with pkl_path.open("rb") as f:
        model = _MANOUnpickler(f, encoding="latin1").load()
    weights = np.asarray(model["weights"])
    if weights.ndim != 2 or weights.shape != (778, 16):
        raise ValueError(f"{pkl_path}: expected 'weights' shape (778, 16), got {weights.shape}")

    joint = weights.argmax(axis=1)                                     # (778,) in [0, 16)
    table = np.asarray(_MANO_JOINT_TO_LABEL, dtype=np.int8)
    labels = table[joint]                                              # (778,) int8

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, labels)
    logger.info("cached MANO finger labels (%s) to %s", side, cache_path)
    return torch.from_numpy(labels).to(torch.int8)


def robot_link_labels(model: Any, body_ids: np.ndarray | Tensor) -> Tensor:
    """Per-point finger label `(U,)` int8 for Sharpa `body_ids`, from body names via `_LINK_NAME_TO_LABEL`.

    Args:
        model: A `mujoco.MjModel` (the compiled scene `RobotScene.model`).
        body_ids: `(U,)` MuJoCo body ids, e.g. from `RobotScene.surface_points`.

    Raises:
        ValueError: If a body id has no name in `model`.
    """
    import mujoco

    ids = np.asarray(body_ids.detach().cpu().numpy() if isinstance(body_ids, Tensor) else body_ids)
    out = np.full(ids.shape, PALM, dtype=np.int8)
    name_cache: dict[int, str] = {}
    for i, b in enumerate(ids.tolist()):
        name = name_cache.get(b)
        if name is None:
            resolved = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)
            if resolved is None:
                raise ValueError(f"body id {b} has no name in the compiled model")
            name = name_cache[b] = resolved
        for pattern, label in _LINK_NAME_TO_LABEL:
            if pattern.search(name):
                out[i] = label
                break
    return torch.from_numpy(out)
