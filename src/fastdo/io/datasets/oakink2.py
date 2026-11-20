"""OakInk2 (Zhan et al., CVPR 2024) as a READ-ONLY source: per-sequence views, frames, task programs and GT poses.

LICENCE: OakInk2 is CC BY-SA 4.0 (data/oakink2/LICENSE_NOTE.md). SHARE-ALIKE: anything derived from OakInk2 data
(windows, masks, meshes, poses, figures) that is redistributed must carry the same licence and attribution. Nothing
here copies, moves or writes under the data root. Other readers may use the same files concurrently, and every access
here is a read.

Layout under `root` (data/oakink2; a SUBSET of 89 of 627 sequences, 2 of 4 views, see LICENSE_NOTE.md):
- `data/<seq>/<serial>/<mocap_id:06d>.png`: RGB frames. The file name is the MOCAP frame id (120 Hz); images run at
  ~30 fps (ids step 2-6, mean 4).
- `data/<seq>/.subset_info`: one line `views=<v1>,<v2> serials=<s1>,<s2> png_counts=... frame_id_list=<n> tar_sha256=...`,
  written by the verified download. It holds the view -> camera serial map, in order.
- `program/program_info/<seq>.json`: the task program. Keys are `"(<lh interval>, <rh interval>)"`, each interval
  `(start, stop)` in MOCAP frame ids, or None. Values hold `obj_list_lh`, `obj_list_rh`, `primitive_*` and
  `interaction_mode`.
- `anno_preview/<seq>.pkl` (~60 MB, torch tensors inside): `cam_def` (serial -> view), `frame_id_list`,
  `cam_intr[view][mocap_id]` (3,3), `cam_extr[view][mocap_id]` (4,4), `obj_list`, `obj_transf[obj][mocap_id]` (4,4).
  It is read through `_RestrictedUnpickler`: a six-global allow-list, plus numpy 2's name for the same
  reconstructor.
- `object_raw/align_ds/<obj>/<mesh>.{ply,obj}`, `object_repair/align_ds/<obj>/model.obj`, `object_raw/obj_desc.json`.

Views. `egocentric` is HEAD-MOUNTED: its extrinsics move within a sequence (translation sd 3-5 cm on
scene_01__A001++seq__0b3d...). The `allocentric_*` views are static rig cameras (sd 0). This is measured on the stored
extrinsics, not assumed.

Pose convention (a NUMERIC check, NOT a reprojection against pixels). `cam_extr` is WORLD -> CAMERA:
- with `x_cam = E @ x_world`, a static scene object sits at +1.14 m depth and projects inside the 848x480 image;
- the inverse reading puts it behind the camera (-0.65 m).
So the object-to-camera pose is `E[view][id] @ obj_transf[obj][id]`. It must be verified by reprojection (mesh
silhouette vs an independent mask) before any GT scale or pose endpoint is scored on it. `object_pose_in_camera`
records this in its docstring and returns the composition only.

Frame indexing: an IMAGE INDEX i is the position in the sorted frame list (0..n-1); `frame_ids(seq)[i]` is its mocap id.
Windows, decode rows and `decode.npz` `frame_idx` use image indices.
"""

from __future__ import annotations

import ast
import json
import pickle
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["FPS", "MOCAP_FPS", "RESERVED", "VIEWS", "OakInk2Reader", "Primitive", "parse_program",
           "refuse_reserved"]

FPS = 30.0            # image rate (mean id step 4 at 120 Hz mocap)
MOCAP_FPS = 120.0
VIEWS = ("egocentric", "allocentric_top", "allocentric_left", "allocentric_right")
RESERVED = ("subject-07", "subject-08", "FINAL_TEST")
"""Names this project never opens (DexYCB 07/08, HOI4D FINAL_TEST). OakInk2 contains none; the guard keeps it so."""

# Six globals, plus numpy 2's spelling of the same reconstructor (numpy._core), which a pickle written under
# numpy 2 names; both resolve to one function.
_PICKLE_ALLOWED = frozenset({("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
                             ("numpy", "ndarray"), ("numpy", "dtype"),
                             ("torch._utils", "_rebuild_tensor_v2"), ("torch.storage", "_load_from_bytes"),
                             ("collections", "OrderedDict")})


def refuse_reserved(name: str | Path) -> None:
    """Raise if `name` (a path or id) names reserved data (`RESERVED`)."""
    if any(tag in str(name) for tag in RESERVED):
        raise PermissionError(f"reserved data refused: {name}")


class _RestrictedUnpickler(pickle.Unpickler):
    """OakInk2's annotation pickles need exactly `_PICKLE_ALLOWED`; any other global is refused.

    Pickle is the dataset's only release format. The files are a local download whose sha256 was checked against the
    HF LFS oids (LICENSE_NOTE.md), and this allow-list limits what a load can construct."""

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) not in _PICKLE_ALLOWED:
            raise pickle.UnpicklingError(f"refused global {module}.{name}")
        return super().find_class(module, name)


@dataclass(frozen=True)
class Primitive:
    """One task-program step: per hand, a MOCAP-id interval `[start, stop]` (or None) and its object ids."""

    lh: tuple[int, int] | None
    rh: tuple[int, int] | None
    obj_lh: tuple[str, ...]
    obj_rh: tuple[str, ...]
    name: str
    mode: str


def _interval(v: Any) -> tuple[int, int] | None:
    if v is None:
        return None
    a, b = (int(x) for x in v)
    if b < a:
        raise ValueError(f"interval ({a}, {b}) ends before it starts")
    return a, b


def parse_program(data: dict[str, Any]) -> list[Primitive]:
    """`program_info/<seq>.json` as `Primitive`s, sorted by their first frame.

    Raises:
        ValueError: On a key that is not a `(lh, rh)` pair, or an interval that ends before it starts.
    """
    out: list[Primitive] = []
    for key, v in data.items():
        pair = ast.literal_eval(key)
        if not (isinstance(pair, tuple) and len(pair) == 2):
            raise ValueError(f"program key {key!r} is not a (lh, rh) pair")
        lh, rh = _interval(pair[0]), _interval(pair[1])
        obj_lh = tuple(v.get("obj_list_lh") or ())
        obj_rh = tuple(v.get("obj_list_rh") or ())
        out.append(Primitive(lh, rh, obj_lh, obj_rh, str(v.get("primitive")), str(v.get("interaction_mode"))))
    return sorted(out, key=lambda p: min(i[0] for i in (p.lh, p.rh) if i is not None) if (p.lh or p.rh) else 0)


class OakInk2Reader:
    """Read-only access to one OakInk2 root (module docstring). Nothing is written under `root`."""

    def __init__(self, root: Path) -> None:
        refuse_reserved(root)
        self.root = Path(root)

    # --- the subset on disk ---------------------------------------------------------------------------------------
    def sequences(self) -> list[str]:
        """Sequence names with a verified `.subset_info` (the complete downloads), sorted."""
        base = self.root / "data"
        names = sorted(p.parent.name for p in base.glob("*/.subset_info"))
        for n in names:
            refuse_reserved(n)
        return names

    def view_serials(self, seq: str) -> dict[str, str]:
        """`view -> camera serial` from `.subset_info` (the order of `views=` matches `serials=`)."""
        refuse_reserved(seq)
        fields = dict(tok.split("=", 1) for tok in (self.root / "data" / seq / ".subset_info").read_text().split()
                      if "=" in tok)
        views, serials = fields["views"].split(","), fields["serials"].split(",")
        if len(views) != len(serials):
            raise ValueError(f"{seq}: .subset_info lists {len(views)} views and {len(serials)} serials")
        return dict(zip(views, serials, strict=True))

    def view_dir(self, seq: str, view: str) -> Path:
        """The directory of `view`'s PNG frames. Raises KeyError for a view not on disk."""
        serials = self.view_serials(seq)
        if view not in serials:
            raise KeyError(f"{seq}: view {view!r} is not on disk (have {sorted(serials)})")
        return self.root / "data" / seq / serials[view]

    def frame_files(self, seq: str, view: str) -> list[Path]:
        """The view's PNG frames in image-index order (sorted by mocap id)."""
        files = sorted(self.view_dir(seq, view).glob("*.png"), key=lambda p: int(p.stem))
        if not files:
            raise FileNotFoundError(f"{seq}/{view}: no PNG frames")
        return files

    def frame_ids(self, seq: str, view: str = "egocentric") -> np.ndarray:
        """(n,) int64 mocap ids of the view's images, in image-index order (the file names; no pickle read)."""
        return np.array([int(p.stem) for p in self.frame_files(seq, view)], dtype=np.int64)

    # --- ground truth -----------------------------------------------------------------------------------------------
    def program(self, seq: str) -> list[Primitive]:
        refuse_reserved(seq)
        return parse_program(json.loads((self.root / "program" / "program_info" / f"{seq}.json").read_text()))

    def object_name(self, obj: str) -> str:
        return str(_obj_desc(self.root).get(obj, {}).get("obj_name", obj))

    def object_mesh_path(self, obj: str) -> Path:
        """`object_repair/align_ds/<obj>/model.obj` when present, else the single mesh under `object_raw/align_ds/<obj>`.

        Raises:
            FileNotFoundError: When neither exists, or `object_raw` holds no single mesh.
        """
        repaired = self.root / "object_repair" / "align_ds" / obj / "model.obj"
        if repaired.is_file():
            return repaired
        raw = sorted(p for p in (self.root / "object_raw" / "align_ds" / obj).glob("*") if p.suffix in (".ply", ".obj"))
        if len(raw) != 1:
            raise FileNotFoundError(f"object {obj}: expected one mesh under object_raw/align_ds, found {len(raw)}")
        return raw[0]

    def annotation(self, seq: str) -> dict[str, Any]:
        """`anno_preview/<seq>.pkl` through the restricted unpickler (~6 s, ~GB of RAM; not cached)."""
        refuse_reserved(seq)
        with (self.root / "anno_preview" / f"{seq}.pkl").open("rb") as fh:
            return dict(_RestrictedUnpickler(fh).load())

    @staticmethod
    def object_pose_in_camera(anno: dict[str, Any], view: str, obj: str, ids: Sequence[int]
                              ) -> tuple[np.ndarray, np.ndarray]:
        """`(pose (T,4,4) float64, valid (T,) bool)`: `cam_extr[view][id] @ obj_transf[obj][id]` for each mocap id.

        The world->camera reading of `cam_extr` is a NUMERIC check only (module docstring). Verify it by reprojection
        before scoring on it. A missing id or a non-finite matrix gives identity with valid False.
        """
        extr, transf = anno["cam_extr"][view], anno["obj_transf"][obj]
        out = np.tile(np.eye(4), (len(ids), 1, 1))
        valid = np.zeros(len(ids), dtype=bool)
        for i, fid in enumerate(ids):
            if fid in extr and fid in transf:
                m = np.asarray(extr[fid], np.float64) @ np.asarray(transf[fid], np.float64)
                if np.isfinite(m).all():
                    out[i], valid[i] = m, True
        return out, valid

    @staticmethod
    def intrinsics(anno: dict[str, Any], view: str, ids: Sequence[int]) -> np.ndarray:
        """(T,3,3) float64 pixel intrinsics of `view` at each mocap id."""
        k = anno["cam_intr"][view]
        return np.stack([np.asarray(k[int(fid)], np.float64) for fid in ids])


@lru_cache(maxsize=4)
def _obj_desc(root: Path) -> dict[str, Any]:
    path = Path(root) / "object_raw" / "obj_desc.json"
    return json.loads(path.read_text()) if path.is_file() else {}
