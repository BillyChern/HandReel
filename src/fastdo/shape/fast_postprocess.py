# Copyright 2026 The HandReel authors. SPDX-License-Identifier: Apache-2.0
# This file adapts `postprocess_mesh` and `_fill_holes` of SAM 3D Objects
# (https://github.com/facebookresearch/sam-3d-objects @ 875b0100, as pinned inside DO AS I DO;
# sam3d_objects/model/backbone/tdfy_dit/utils/postprocessing_utils.py, "Copyright (c) Meta
# Platforms, Inc. and affiliates."), a SAM Material of Meta Platforms, Inc. The adapted parts are
# the outer-face visibility rule (quantile 0.75, clamped to [0.25, 0.5]), the dual-graph mincut
# (edge weights x 1000) and the meshfix hole fill (`fill_small_boundaries(nbe=..., refine=True)`).
# Those parts are distributed under the SAM License (licenses/SAM_LICENSE), the rest of the file
# under Apache-2.0. See THIRD_PARTY_LICENSES.md.
"""The instrumented fast replacement for SAM 3D's `postprocess_mesh`.

SAM 3D's own postprocess costs 1.3-8.2 s per request, 33-87% of it in the decimator, and its only material effect on
the geometry is deleting the decoder's inner wall (the decimation error is 0.003 mm on DexYCB). This
module keeps the algorithm and makes it fast:

1. a selectable decimator: pyvista/VTK as today, or `fast_simplification`;
2. the 1000 camera matrices built in one batched `view_look_at` instead of a Python loop (0.34-0.37 s);
3. rasterisation in chunks of views, per-view visibility counted by `scatter` instead of `torch.unique`;
4. the igraph source/sink edge lists built from host ints instead of iterating CUDA scalars (up to 0.90 s).

The mincut validity checks, the meshfix pass and the caller's vertex merge are unchanged, and the function returns the
mesh **before** the z-up to y-up rotation, exactly as `postprocess_mesh` does, so `geometry_postprocess` finishes the
job identically for both implementations.

**Decimator rule (fixed).** VTK below `min_raw_faces` = 250,000 raw FlexiCubes faces,
`fast_simplification` at or above it. Below the threshold the output equals SAM 3D's postprocess bit for bit (P2b and
P4, 4/4 clips), which is what makes arm A0v a byte-exact replica of the control; above it, `fast_simplification` costs
0.11 s per 100k faces against VTK's 0.79. Every whisk mesh measured sits below the threshold and every DexYCB mesh
above it, so the threshold is a size proxy for thin geometry. A threshold above any mesh size is the instrumented
VTK-only mode that serves A0v.

**Detached fragments.** `min_component_area_frac` > 0 adds one last mesh step, after hole filling and
before the wall-QC statistics: every connected component below that fraction of the total surface area is dropped
(`drop_small_components`), with the connectivity measured on the worker's output - faces
joined when they share an undirected edge, after `geometry_postprocess`'s y-up rotation and trimesh's 8-digit vertex
merge, which this step reproduces. The largest component is always kept. At 0.0 (the default) the step does not run
and the output is byte-identical to the path without this step.

The GPU body needs `torch`, `utils3d`, `igraph`, `pyvista` or `fast_simplification`, `pymeshfix` and
`sam3d_objects`; every one is imported lazily, inside the call. This module runs under the SAM 3D environment's
interpreter beside `sam3d_worker.py`, so it imports nothing from `fastdo`.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

if __package__:
    from . import wall_qc
else:                                   # run as a script under the SAM 3D env: the sibling file, never `fastdo`
    import wall_qc  # type: ignore[no-redef]

__all__ = [
    "DECIMATORS",
    "DEFAULT_MIN_COMPONENT_AREA_FRAC",
    "DEFAULT_MIN_RAW_FACES",
    "POSTPROCESS_IMPLS",
    "Y_UP_KEY_ROTATION",
    "check_min_component_area_frac",
    "choose_decimator",
    "decimate",
    "drop_small_components",
    "fast_postprocess",
]

DEFAULT_MIN_RAW_FACES = 250_000
DECIMATORS: tuple[str, ...] = ("vtk", "fast_simplification")
POSTPROCESS_IMPLS: tuple[str, ...] = ("official", "fast")
DEFAULT_MIN_COMPONENT_AREA_FRAC = 0.0
# `sam3d_worker.Z_UP_TO_Y_UP` (asserted equal by a test): the merge that follows this module groups vertices in the
# rotated frame, so the fragment filter keys them in that frame too. Kept here because this module imports nothing.
Y_UP_KEY_ROTATION = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
_MERGE_DIGITS = 8          # trimesh's tol.merge = 1e-8, which `geometry_postprocess`'s Trimesh(...) merge uses


def check_min_component_area_frac(value: Any) -> float:
    """`value` as the fragment filter's fraction; the message is shared by `ShapeConfig` and the worker boundary.

    Raises:
        ValueError: Unless `value` is a finite int or float (not a bool) in [0, 1).
    """
    ok = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))
    if not ok or not 0.0 <= float(value) < 1.0:
        raise ValueError(f"min_component_area_frac must be a finite number in [0, 1), got {value!r}")
    return float(value)


def _merged_face_components(vertices: np.ndarray, faces: np.ndarray) -> tuple[int, np.ndarray]:
    """`(count, per-face label)`: G-topo (a)'s components of the mesh `geometry_postprocess` would write.

    Vertices are grouped exactly as trimesh's merge groups the referenced ones (after the y-up rotation, rounded at
    8 digits with its 1e-6 bias), then faces are joined when they share an undirected edge of merged vertices.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    f = np.asarray(faces, dtype=np.int64)
    used, local = np.unique(f.reshape(-1), return_inverse=True)
    rotated = np.asarray(vertices, dtype=np.float64)[used] @ Y_UP_KEY_ROTATION
    keys = np.round(rotated * 10**_MERGE_DIGITS - 1e-6).astype(np.int64)
    _, merged = np.unique(keys, axis=0, return_inverse=True)
    mf = merged.reshape(-1)[local].reshape(f.shape)
    n_vertices = int(mf.max()) + 1 if len(mf) else 0
    e = np.sort(np.concatenate([mf[:, [0, 1]], mf[:, [1, 2]], mf[:, [2, 0]]]), axis=1)
    fid = np.concatenate([np.arange(len(mf))] * 3)
    key = e[:, 0] * (n_vertices + 1) + e[:, 1]
    order = np.argsort(key, kind="stable")
    ks, fs = key[order], fid[order]
    same = ks[1:] == ks[:-1]
    graph = coo_matrix((np.ones(int(same.sum())), (fs[:-1][same], fs[1:][same])), shape=(len(mf), len(mf)))
    n, labels = connected_components(graph, directed=False)
    return int(n), labels


def drop_small_components(vertices: np.ndarray, faces: np.ndarray, min_area_frac: float
                          ) -> tuple[np.ndarray, np.ndarray, dict[str, Any] | None]:
    """Drops every component below `min_area_frac` of the total area; the largest is always kept.

    Returns:
        `(vertices, faces, info)`. At `min_area_frac == 0`, or when nothing is dropped, `vertices` and `faces` are
        the input objects themselves (at 0, `info` is None). Otherwise the kept faces, re-indexed onto the vertices
        they reference (in their original order and dtype), and `info` = `{components, dropped_components,
        dropped_faces, dropped_area_frac}`.

    Raises:
        ValueError: On a fraction outside [0, 1) (`check_min_component_area_frac`).
    """
    frac = check_min_component_area_frac(min_area_frac)
    if frac == 0.0:
        return vertices, faces, None
    f = np.asarray(faces, dtype=np.int64)
    v64 = np.asarray(vertices, dtype=np.float64)
    if len(f) == 0:
        return vertices, faces, {"components": 0, "dropped_components": 0, "dropped_faces": 0,
                                 "dropped_area_frac": 0.0}
    n, labels = _merged_face_components(v64, f)
    p0, p1, p2 = v64[f[:, 0]], v64[f[:, 1]], v64[f[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
    per = np.bincount(labels, weights=area, minlength=n)
    total = float(area.sum())
    drop = per < frac * total
    drop[int(np.argmax(per))] = False
    info: dict[str, Any] = {"components": n, "dropped_components": int(drop.sum()), "dropped_faces": 0,
                            "dropped_area_frac": float(per[drop].sum() / total) if total > 0 else 0.0}
    if not drop.any():
        return vertices, faces, info
    keep = ~drop[labels]
    info["dropped_faces"] = int((~keep).sum())
    kept = f[keep]
    used, remapped = np.unique(kept.reshape(-1), return_inverse=True)
    return np.asarray(vertices)[used], remapped.reshape(kept.shape).astype(np.int64), info


def choose_decimator(n_raw_faces: int, min_raw_faces: int) -> str:
    """`"vtk"` below `min_raw_faces` raw faces, `"fast_simplification"` at or above it.

    Raises:
        ValueError: On a negative raw face count or a negative threshold.
    """
    if int(n_raw_faces) < 0:
        raise ValueError(f"the raw face count must be >= 0, got {n_raw_faces!r}")
    if int(min_raw_faces) < 0:
        raise ValueError(f"fast_decimator_min_raw_faces must be >= 0, got {min_raw_faces!r}")
    return "vtk" if int(n_raw_faces) < int(min_raw_faces) else "fast_simplification"


def decimate(vertices: np.ndarray, faces: np.ndarray, decimator: str,
             simplify_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    """One decimation pass, `simplify_ratio` being the fraction of faces removed (0.95, as `to_glb` asks).

    Raises:
        ValueError: On an unknown decimator.
    """
    if decimator == "vtk":
        import pyvista as pv

        v, f = np.asarray(vertices, np.float32), np.asarray(faces, np.int64)
        mesh = pv.PolyData(v, np.concatenate([np.full((f.shape[0], 1), 3), f], axis=1)).decimate(simplify_ratio)
        return np.asarray(mesh.points), mesh.faces.reshape(-1, 4)[:, 1:]
    if decimator == "fast_simplification":
        import fast_simplification

        v, f = fast_simplification.simplify(np.asarray(vertices, np.float32),
                                            np.asarray(faces, np.int32), target_reduction=simplify_ratio)
        return np.asarray(v, np.float32), np.asarray(f, np.int64)
    raise ValueError(f"unknown decimator {decimator!r}; expected one of {DECIMATORS}")


def _cameras(num_views: int, device: Any) -> tuple[Any, Any]:
    """`(view matrices, projection)` for `num_views` Hammersley directions, built in one batched call."""
    import torch
    import utils3d
    from sam3d_objects.model.backbone.tdfy_dit.utils.random_utils import sphere_hammersley_sequence

    yaw_pitch = [sphere_hammersley_sequence(i, num_views) for i in range(num_views)]
    yaws = torch.tensor([a for a, _ in yaw_pitch]).to(device)
    pitchs = torch.tensor([b for _, b in yaw_pitch]).to(device)
    eye = torch.stack([torch.sin(yaws) * torch.cos(pitchs), torch.cos(yaws) * torch.cos(pitchs),
                       torch.sin(pitchs)], -1).float() * 2.0
    up = torch.tensor([0, 0, 1]).float().to(device).expand_as(eye)
    views = utils3d.torch.view_look_at(eye, torch.zeros_like(eye), up)
    fov = torch.deg2rad(torch.tensor(40)).to(device)
    return views, utils3d.torch.perspective_from_fov_xy(fov, fov, 1, 3)


def _visibility(verts: Any, fcs: Any, views: Any, projection: Any, num_views: int, resolution: int,
                chunk: int) -> Any:
    """Per-face visible-view fraction, rasterised in chunks of `chunk` views and counted by scatter."""
    import torch
    import utils3d

    n_faces = fcs.shape[0]
    vis_count = torch.zeros(n_faces, dtype=torch.int32, device=verts.device)
    rastctx = utils3d.torch.RastContext(backend="cuda")
    for start in range(0, num_views, chunk):
        batch = min(chunk, num_views - start)
        buf = utils3d.torch.rasterize_triangle_faces(
            rastctx, verts[None].expand(batch, -1, -1).contiguous(), fcs, resolution, resolution,
            view=views[start:start + batch], projection=projection)
        face_id = buf["face_id"].long() - 1
        ok = (buf["mask"] > 0.95) & (face_id >= 0)
        seen = torch.zeros((batch, n_faces), dtype=torch.bool, device=verts.device)
        rows = torch.arange(batch, device=verts.device)[:, None, None].expand_as(face_id)
        seen[rows[ok], face_id[ok]] = True
        vis_count += seen.sum(0, dtype=torch.int32)
    return vis_count.float() / num_views


def _cut_is_valid(component: Any, removable: Any, visibility: Any, verts: Any, edges: Any, face2edge: Any,
                  boundary_edge_indices: Any, max_hole_size: float, stats: dict[str, Any]) -> bool:
    """SAM 3D's two checks on one removable component: it must be hidden, and it must not open a large hole."""
    import torch
    import utils3d

    faces = removable[component]
    if visibility[faces].median() > 0.25:
        return False
    edge_ids, degree = torch.unique(face2edge[faces], return_counts=True)
    cut_boundary = edge_ids[degree == 1]
    new_boundary = cut_boundary[~torch.isin(cut_boundary, boundary_edge_indices)]
    if len(new_boundary) == 0:
        return True
    loops = utils3d.torch.compute_edge_connected_components(edges[new_boundary])
    areas = []
    for loop in loops:
        loop_edges = edges[new_boundary[loop]]
        centre = verts[loop_edges].mean(dim=1).mean(dim=0)
        e1, e2 = verts[loop_edges[:, 0]] - centre, verts[loop_edges[:, 1]] - centre
        areas.append(float(torch.norm(torch.cross(e1, e2, dim=-1), dim=1).sum() * 0.5))
    if any(area > max_hole_size for area in areas):
        stats.setdefault("refused_cut_areas", []).append(areas)
        return False
    return True


def fast_postprocess(vertices: Any, faces: Any, *, simplify: bool = True, simplify_ratio: float = 0.95,
                     fill_holes: bool = True, fill_holes_max_hole_size: float = 0.04,
                     fill_holes_max_hole_nbe: int = 55, fill_holes_resolution: int = 1024,
                     fill_holes_num_views: int = 1000, verbose: bool = False,
                     min_raw_faces: int = DEFAULT_MIN_RAW_FACES, stats: dict[str, Any] | None = None,
                     chunk: int = 25,
                     min_component_area_frac: float = DEFAULT_MIN_COMPONENT_AREA_FRAC) -> tuple[np.ndarray, np.ndarray]:
    """`postprocess_mesh(**GEOMETRY_POSTPROCESS)`, same semantics, fast; the mesh before the y-up rotation.

    Args:
        vertices: `(V,3)` raw FlexiCubes vertices.
        faces: `(F,3)` raw FlexiCubes faces.
        simplify, simplify_ratio, fill_holes, fill_holes_max_hole_size, fill_holes_max_hole_nbe,
            fill_holes_resolution, fill_holes_num_views, verbose: `postprocess_mesh`'s own arguments, so this is a
            drop-in `postprocess_fn` for `geometry_postprocess`.
        min_raw_faces: The decimator threshold (`choose_decimator`).
        stats: When given, filled with the decimator, the face counts, the refused cut loops, the per-step seconds and
            the wall-QC statistics (`removed` and `area_ratio`, computed here, before the rotation).
        chunk: Views per rasterisation batch.
        min_component_area_frac: When > 0, the last mesh step drops every connected component below this fraction
            of the total area (`drop_small_components`) and `stats["fragments"]` records what it dropped. 0.0 (the
            default) skips the step entirely.

    Returns:
        `(vertices (V,3) float32, faces (F,3) int64)`, ready for `geometry_postprocess`'s rotation and merge.
    """
    import torch

    record: dict[str, Any] = {} if stats is None else stats
    seconds: dict[str, float] = {}
    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    v = vertices.detach().cpu().numpy() if hasattr(vertices, "detach") else np.asarray(vertices)
    f = faces.detach().cpu().numpy() if hasattr(faces, "detach") else np.asarray(faces)
    v, f = np.asarray(v, np.float32), np.asarray(f, np.int64)
    record["faces_raw"] = len(f)

    decimator = choose_decimator(len(f), min_raw_faces)
    record["decimator"] = decimator if simplify else None
    sync()
    start = time.perf_counter()
    if simplify:
        v, f = decimate(v, f, decimator, simplify_ratio)
    seconds["decimate"] = time.perf_counter() - start
    record["faces_decimated"] = len(f)
    record["faces_removed_mincut"] = 0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    verts = torch.tensor(np.asarray(v, np.float32)).to(device)
    fcs = torch.tensor(np.asarray(f, np.int32)).to(device)
    if fill_holes:
        verts, fcs = _fill_holes(verts, fcs, fill_holes_num_views, fill_holes_resolution, chunk,
                                 fill_holes_max_hole_size, fill_holes_max_hole_nbe, record, seconds, sync)
    out_v = np.asarray(verts.cpu().numpy(), np.float32)
    out_f = np.asarray(fcs.cpu().numpy(), np.int64)
    if check_min_component_area_frac(min_component_area_frac) > 0.0:
        start = time.perf_counter()
        out_v, out_f, info = drop_small_components(out_v, out_f, min_component_area_frac)
        seconds["fragments"] = time.perf_counter() - start
        record["fragments"] = {"min_component_area_frac": float(min_component_area_frac), **(info or {})}
    record["faces_final"] = len(out_f)
    record["seconds"] = seconds
    record.update(wall_qc.statistics(out_v, out_f, faces_decimated=record["faces_decimated"],
                                     faces_removed_mincut=record["faces_removed_mincut"]))
    return out_v, out_f


def _fill_holes(verts: Any, fcs: Any, num_views: int, resolution: int, chunk: int, max_hole_size: float,
                max_hole_nbe: int, record: dict[str, Any], seconds: dict[str, float], sync: Any) -> tuple[Any, Any]:
    """The visibility, mincut and meshfix half of `postprocess_mesh`, on the decimated mesh."""
    import torch
    import utils3d
    from pymeshfix import _meshfix

    start = time.perf_counter()
    views, projection = _cameras(num_views, verts.device)
    sync()
    seconds["cameras"] = time.perf_counter() - start
    start = time.perf_counter()
    visibility = _visibility(verts, fcs, views, projection, num_views, resolution, chunk)
    sync()
    seconds["visibility_raster"] = time.perf_counter() - start
    start = time.perf_counter()

    n_faces = fcs.shape[0]
    edges, face2edge, edge_degrees = utils3d.torch.compute_edges(fcs)
    boundary_edge_indices = torch.nonzero(edge_degrees == 1).reshape(-1)
    components = utils3d.torch.compute_connected_components(fcs, edges, face2edge)
    outer = torch.zeros(n_faces, dtype=torch.bool, device=fcs.device)
    for component in components:
        outer[component] = visibility[component] > min(max(visibility[component].quantile(0.75).item(), 0.25), 0.5)
    outer_idx = outer.nonzero().reshape(-1)
    inner_idx = torch.nonzero(visibility == 0).reshape(-1)
    record["n_inner_faces"] = int(inner_idx.numel())
    if inner_idx.shape[0] == 0:
        seconds["mincut"] = time.perf_counter() - start
        return verts, fcs

    removable = _mincut(verts, fcs, edges, face2edge, inner_idx, outer_idx, n_faces)
    keep = []
    for component in utils3d.torch.compute_connected_components(fcs[removable]):
        if _cut_is_valid(component, removable, visibility, verts, edges, face2edge, boundary_edge_indices,
                         max_hole_size, record):
            keep.append(component)
    if keep:
        removable = removable[torch.cat(keep)]
        mask = torch.ones(n_faces, dtype=torch.bool, device=fcs.device)
        mask[removable] = 0
        record["faces_removed_mincut"] = int((~mask).sum())
        fcs = fcs[mask]
        fcs, verts = utils3d.torch.remove_unreferenced_vertices(fcs, verts)
    sync()
    seconds["mincut"] = time.perf_counter() - start
    start = time.perf_counter()
    mesh = _meshfix.PyTMesh()
    mesh.load_array(verts.cpu().numpy(), fcs.cpu().numpy())
    mesh.fill_small_boundaries(nbe=max_hole_nbe, refine=True)
    filled_v, filled_f = mesh.return_arrays()
    verts = torch.tensor(filled_v, device=verts.device, dtype=torch.float32)
    fcs = torch.tensor(filled_f, device=fcs.device, dtype=torch.int32)
    sync()
    seconds["meshfix"] = time.perf_counter() - start
    return verts, fcs


def _mincut(verts: Any, fcs: Any, edges: Any, face2edge: Any, inner_idx: Any, outer_idx: Any, n_faces: int) -> Any:
    """SAM 3D's hidden/visible mincut on the dual graph; the source-side faces, as a face index tensor."""
    import igraph
    import torch
    import utils3d

    dual_edges, dual_edge2edge = utils3d.torch.compute_dual_graph(face2edge)
    dual_edge2edge = edges[dual_edge2edge]
    weights = torch.norm(verts[dual_edge2edge[:, 0]] - verts[dual_edge2edge[:, 1]], dim=1)
    graph = igraph.Graph()
    graph.add_vertices(n_faces)
    graph.add_edges(dual_edges.cpu().numpy())
    graph.es["weight"] = weights.cpu().numpy()
    graph.add_vertex("s")
    graph.add_vertex("t")
    inner, outer = inner_idx.cpu().tolist(), outer_idx.cpu().tolist()
    graph.add_edges([(x, "s") for x in inner], attributes={"weight": np.ones(len(inner), dtype=np.float32)})
    graph.add_edges([(x, "t") for x in outer], attributes={"weight": np.ones(len(outer), dtype=np.float32)})
    cut = graph.mincut("s", "t", (np.array(graph.es["weight"]) * 1000).tolist())
    return torch.tensor([x for x in cut.partition[0] if x < n_faces], dtype=torch.long, device=fcs.device)
