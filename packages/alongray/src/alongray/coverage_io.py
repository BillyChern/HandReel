# Copyright 2026 The alongray authors
# SPDX-License-Identifier: Apache-2.0
"""Reading and validating the inputs of :mod:`alongray.coverage`: one protocol file and one folder per method.

Formats (all JSON; arrays in ``.npz`` files without pickles):

``protocol.json`` (schema ``alongray-coverage-protocol/1``)
    ``provenance`` {dataset, split, gt_release}; ``modality`` {input, compare: [output names]}; ``conventions``
    (the keys of :data:`CONVENTION_KEYS`); ``cluster`` ("sequence" or "object"; the resampling cluster);
    optional ``frame_rate``; ``overlap_policy`` ("refuse", the default, or "keep_with_multiplicity");
    optional ``objects`` {name: {symmetry: [3x3 rotation, ...]}}; ``units`` [{unit_id, sequence, camera, object,
    frames, scorable}], where ``scorable`` is the subset of ``frames`` with ground truth; optional
    ``published_counts`` (counts-only comparisons).

``<folder>/manifest.json`` (schema ``alongray-coverage-method/1``)
    ``method``; ``outputs`` ("per_frame", "reported_only" or "counts_only"); ``provenance``; ``modality`` {input,
    outputs}; optional ``inputs`` (disclosed, compared, never a refusal); ``conventions``; ``units`` {unit_id:
    {status, reason, file, sequence, camera, object}}; optional ``scores`` {unit_id: {n_frames, metric: value}};
    ``counts`` (counts_only); ``reported`` (reported_only; optional ``evaluated_frames`` {sequence: [ids]}).

A unit ``file`` holds ``frame_ids`` and optionally ``object_position`` (N, 3), ``object_rotation`` (N, 3, 3),
``camera_from_world`` (N, 4, 4) and ``timestamps`` (N,) seconds. File names must stay inside the folder.

Every block is type-checked on load; a wrong type, a protocol with units but no annotated frame, or an unreadable
unit file is a ValueError (exit status 2), never an uncaught exception.
"""

from __future__ import annotations

import re
import zipfile
import zlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from alongray import strict_json

__all__ = ["CONVENTION_KEYS", "STATUSES", "Method", "Protocol", "UnitSpec", "load_method", "load_protocol",
           "load_unit_arrays", "timing_fields"]

CONVENTION_KEYS = ("length_unit", "frame", "camera_axes", "frame_index_base", "clock", "object_origin", "scale",
                   "symmetry", "registration", "alignment_fit_support", "aggregation")
CONVERTED_KEYS = ("length_unit", "frame", "camera_axes", "frame_index_base")
STATUSES = ("delivered", "failed", "refused", "not_run", "excluded")
OUTPUT_KINDS = ("per_frame", "reported_only", "counts_only")
_TIMING = re.compile(r"(^|_)(sec|secs|second|seconds|wall|time|fps|speed|runtime|latency|throughput|gpu_s|"
                     r"gpu_h|core_s|core_h|elapsed)($|_)", re.IGNORECASE)


@dataclass(frozen=True)
class UnitSpec:
    unit_id: str
    sequence: str
    camera: str
    object: str
    frames: tuple[int, ...]
    scorable: frozenset[int]


@dataclass(frozen=True)
class Protocol:
    name: str
    provenance: Mapping[str, Any]
    modality: Mapping[str, Any]
    conventions: Mapping[str, Any]
    cluster: str
    frame_rate: float | None
    overlap_policy: str
    symmetry: Mapping[str, tuple[np.ndarray, ...]]
    units: tuple[UnitSpec, ...]
    published_counts: Mapping[str, int] | None


@dataclass(frozen=True)
class Method:
    name: str
    folder: Path
    outputs: str
    provenance: Mapping[str, Any]
    modality: Mapping[str, Any]
    inputs: Mapping[str, Any]
    conventions: Mapping[str, Any]
    units: Mapping[str, Mapping[str, Any]]
    scores: Mapping[str, Mapping[str, Any]]
    counts: Mapping[str, int]
    reported: Mapping[str, Any]
    timing_fields: tuple[str, ...]


def _read(path: Path) -> dict[str, Any]:
    try:
        d = strict_json.loads(path.read_text(), str(path))
    except (OSError, ValueError) as e:                 # ValueError: malformed JSON or a repeated key
        raise ValueError(f"cannot read {path}: {e}") from e
    if not isinstance(d, dict):
        raise ValueError(f"{path} must hold a JSON object")  # noqa: TRY004 (input error, exit 2)
    return d


def _obj(x: Any, what: str) -> dict[str, Any]:
    if not isinstance(x, dict):
        raise ValueError(f"{what} must be a JSON object, got {type(x).__name__}")  # noqa: TRY004 (input error)
    return x


def _opt_obj(d: dict[str, Any], key: str, where: str) -> dict[str, Any]:
    return dict(_obj(d[key], f"{where}'{key}'")) if key in d and d[key] is not None else {}


def _str_list(x: Any, what: str) -> list[str]:
    if not isinstance(x, list) or not all(isinstance(v, str) for v in x):
        raise ValueError(f"{what} must be a list of strings")
    return x


def _int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _int_list(x: Any, what: str) -> list[int]:
    if not isinstance(x, list) or not all(isinstance(v, int) and not isinstance(v, bool) for v in x):
        raise ValueError(f"{what} must be a list of integers")
    return x


def timing_fields(d: Any, prefix: str = "") -> list[str]:
    """Paths of keys that look like timing or speed fields (they are listed and ignored, never used)."""
    out: list[str] = []
    if isinstance(d, dict):
        for k, v in d.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            if _TIMING.search(str(k)):
                out.append(p)
            out += timing_fields(v, p)
    elif isinstance(d, list):
        for v in d[:1]:  # one element shows the shape; lists of records repeat it
            out += timing_fields(v, prefix + "[]")
    return out


def load_protocol(path: str | Path) -> Protocol:
    """Read and validate a protocol file.

    Raises:
        ValueError: missing keys, non-integer frames, scorable frames outside ``frames``, duplicate unit ids,
            duplicate frames within a unit, or an unknown cluster / overlap policy.
    """
    d = _read(Path(path))
    for k in ("provenance", "modality", "conventions"):
        if not isinstance(d.get(k), dict):
            raise ValueError(f"protocol needs a '{k}' object")  # noqa: TRY004 (input error, exit 2)
    if "compare" in d["modality"]:
        _str_list(d["modality"]["compare"], "protocol modality.compare")
    raw_units = d.get("units", [])
    if not isinstance(raw_units, list):
        raise ValueError("protocol 'units' must be a list")  # noqa: TRY004 (input error, exit 2)
    cluster = d.get("cluster", "sequence")
    if cluster not in ("sequence", "object"):
        raise ValueError(f"cluster must be 'sequence' or 'object', got {cluster!r}")
    policy = d.get("overlap_policy", "refuse")
    if policy not in ("refuse", "keep_with_multiplicity"):
        raise ValueError(f"overlap_policy must be 'refuse' or 'keep_with_multiplicity', got {policy!r}")
    units, seen = [], set()
    for i, u in enumerate(raw_units):
        _obj(u, f"protocol unit {i}")
        uid = str(u.get("unit_id", ""))
        if not uid or uid in seen:
            raise ValueError(f"unit ids must be non-empty and unique, got {uid!r}")
        seen.add(uid)
        frames = _int_list(u.get("frames"), f"{uid}.frames")
        if len(set(frames)) != len(frames):
            raise ValueError(f"{uid}: duplicate frames inside one unit")
        scor = set(_int_list(u.get("scorable", []), f"{uid}.scorable"))
        if not scor <= set(frames):
            raise ValueError(f"{uid}: scorable frames must be a subset of frames")
        for k in ("sequence", "camera", "object"):
            if not isinstance(u.get(k), str) or not u[k]:
                raise ValueError(f"{uid}: '{k}' must be a non-empty string (declared source identity)")
        units.append(UnitSpec(uid, u["sequence"], u["camera"], u["object"], tuple(frames), frozenset(scor)))
    pub = d.get("published_counts")
    if not units and pub is None:
        raise ValueError("protocol needs units, or published_counts for a counts-only comparison")
    if units and not any(u.scorable for u in units):
        raise ValueError("protocol has units but no annotated (scorable) frame: there is nothing to compare")
    if pub is not None:
        for k, v in _obj(pub, "protocol published_counts").items():
            if not (_int(v) and v >= 0):
                raise ValueError(f"published_counts.{k} must be a non-negative integer, got {v!r}")
    sym = {}
    for name, o in _opt_obj(d, "objects", "protocol ").items():
        g = tuple(np.asarray(m, float) for m in _obj(o, f"object {name}").get("symmetry", []))
        if any(m.shape != (3, 3) for m in g):
            raise ValueError(f"object {name}: symmetry elements must be 3x3 rotations")
        sym[name] = g
    fr = d.get("frame_rate")
    if fr is not None and not (isinstance(fr, (int, float)) and not isinstance(fr, bool) and fr > 0):
        raise ValueError(f"frame_rate must be a positive number, got {fr!r}")
    return Protocol(str(d.get("name", Path(path).stem)), d["provenance"], d["modality"], d["conventions"], cluster,
                    None if fr is None else float(fr), policy, sym, tuple(units), pub)


def load_method(folder: str | Path) -> Method:
    """Read and validate ``folder/manifest.json``.

    Raises:
        ValueError: unknown outputs kind or unit status, a unit file outside the folder, or a malformed block.
    """
    folder = Path(folder)
    d = _read(folder / "manifest.json")
    outputs = d.get("outputs")
    if outputs not in OUTPUT_KINDS:
        raise ValueError(f"{folder}: outputs must be one of {OUTPUT_KINDS}, got {outputs!r}")
    where = f"{folder}: "
    blocks = {k: _opt_obj(d, k, where) for k in ("provenance", "modality", "inputs", "conventions", "units", "scores",
                                                 "counts", "reported")}
    if "outputs" in blocks["modality"]:
        _str_list(blocks["modality"]["outputs"], f"{folder}: modality.outputs")
    units = blocks["units"]
    for uid, u in units.items():
        _obj(u, f"{folder}: unit {uid}")
        if u.get("status") not in STATUSES:
            raise ValueError(f"{folder}: unit {uid} status {u.get('status')!r} is not one of {STATUSES}")
        f = u.get("file")
        if f is not None and not isinstance(f, str):
            raise ValueError(f"{folder}: unit {uid} file must be a string")
        if f is not None and (Path(f).is_absolute() or ".." in Path(f).parts):
            raise ValueError(f"{folder}: unit {uid} file {f!r} must be a relative path inside the folder")
    for uid, sc in blocks["scores"].items():
        _obj(sc, f"{folder}: scores.{uid}")
        if not (_int(sc.get("n_frames")) and sc["n_frames"] >= 0):
            raise ValueError(f"{folder}: scores.{uid}.n_frames must be a non-negative integer")
    counts = blocks["counts"]
    if outputs == "counts_only" and not _int(counts.get("delivered")):
        raise ValueError(f"{folder}: a counts_only manifest needs counts.delivered")
    return Method(str(d.get("method", folder.name)), folder, outputs, blocks["provenance"], blocks["modality"],
                  blocks["inputs"], blocks["conventions"], units, blocks["scores"], counts, blocks["reported"],
                  tuple(timing_fields(d)))


def load_unit_arrays(method: Method, unit_id: str) -> dict[str, np.ndarray] | None:
    """The unit's arrays, or None when the manifest names no file. Pickled arrays are refused."""
    f = method.units[unit_id].get("file")
    if f is None:
        return None
    path = (method.folder / f).resolve()
    if method.folder.resolve() not in path.parents:
        raise ValueError(f"{method.name}: unit {unit_id} file resolves outside its folder")
    try:
        with path.open("rb") as fh:  # opened here, so the file is closed on every failure path
            z = np.load(fh, allow_pickle=False)
            if not isinstance(z, np.lib.npyio.NpzFile):
                raise ValueError("a single .npy array, not an .npz archive")  # noqa: TRY004 (re-raised below)
            with z:
                return {k: np.asarray(z[k]) for k in z.files}
    except (ValueError, zipfile.BadZipFile, zlib.error, EOFError, KeyError) as e:
        raise ValueError(f"{method.name}: unit {unit_id} file {f!r} is not a readable .npz "
                         f"({type(e).__name__}: {e})") from e
