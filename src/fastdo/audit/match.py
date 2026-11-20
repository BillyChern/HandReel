# Copyright 2026 The HandReel authors. SPDX-License-Identifier: Apache-2.0
"""Is a pre-removed scene the executor's model minus the hand|object pairs?

A pipeline that loads a scene XML often edits the compiled model. DO AS I DO's executor sets timestep 0.005,
implicitfast and 200 line-search iterations (the stored XML says 0.002, Euler, 50) and writes the warm-up weld's
relative pose into `eq_data`; a loader may also change a pair's friction or a body's mass. A model compiled from the
XML alone is then not the executor's model. `executor_diff` lists every difference outside names and the removed
pairs: `MjOption` fields, numeric model arrays, `MjStatistic` fields and the rows of every surviving pair.
`match_executor` copies them. A layout mismatch (different counts or array shapes) raises in both.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .errors import InputError

__all__ = ["LAYOUT", "executor_diff", "match_executor", "missing_pairs"]

LAYOUT = ("nq", "nv", "nu", "na", "nbody", "njnt", "ngeom", "nsite", "neq", "nmesh", "nmocap")
_SKIP = ("name", "pair_", "text_", "paths")         # names shift with the deleted pairs; pairs are matched by geoms


def _fields(obj: Any) -> list[str]:
    return sorted(k for k in dir(obj) if not k.startswith("_") and not callable(getattr(obj, k)))


def _arrays(m: Any, prefix: tuple[str, ...] = (), skip: tuple[str, ...] = _SKIP) -> list[str]:
    out = []
    for k in dir(m):
        if k.startswith("_") or (prefix and not k.startswith(prefix)) or (skip and k.startswith(skip)):
            continue
        if k.endswith("pathadr"):                     # offsets into the asset-path text, which may be rewritten
            continue
        try:
            v = getattr(m, k)
        except (AttributeError, RuntimeError, TypeError):
            continue
        if isinstance(v, np.ndarray) and v.dtype.kind in "fiub":
            out.append(k)
    return sorted(out)


def _pair_fields(m: Any) -> list[str]:
    return [k for k in _arrays(m, prefix=("pair_",), skip=()) if getattr(m, k).shape[:1] == (m.npair,)]


def _pair_map(executor: Any, other: Any) -> dict[int, int]:
    """Surviving pair `p` of `other` -> its executor pair, matched by (geom1, geom2) (geom ids are unchanged)."""
    ex = {(int(executor.pair_geom1[q]), int(executor.pair_geom2[q])): q for q in range(executor.npair)}
    if len(ex) != executor.npair:
        raise InputError("the executor's model has two pairs on the same geoms; pairs cannot be matched")
    out = {}
    for p in range(other.npair):
        key = (int(other.pair_geom1[p]), int(other.pair_geom2[p]))
        if key not in ex:
            raise InputError(f"pair {p} on geoms {key} is not a pair of the executor's model")
        out[p] = ex[key]
    return out


def _check_layout(executor: Any, other: Any) -> None:
    bad = [k for k in LAYOUT if getattr(executor, k) != getattr(other, k)]
    bad += [k for k in _arrays(executor) if getattr(other, k, None) is None
            or getattr(other, k).shape != getattr(executor, k).shape]
    if other.npair > executor.npair:
        bad.append("npair")
    if bad:
        raise InputError(f"layout differs from the executor's model ({', '.join(bad[:8])}): not the same scene")


def missing_pairs(executor: Any, other: Any) -> list[tuple[int, int]]:
    """(geom1, geom2) of the executor's pairs that `other` does not have (the removed ones)."""
    kept = set(_pair_map(executor, other).values())
    return [(int(executor.pair_geom1[q]), int(executor.pair_geom2[q])) for q in range(executor.npair)
            if q not in kept]


def executor_diff(executor: Any, other: Any) -> list[str]:
    """Every field outside names and removed pairs on which `other` differs from the executor's model (module
    docstring): "opt.<f>", "<array>", "stat.<f>", "pair[<geom1>,<geom2>].<f>". Raises on a layout mismatch."""
    _check_layout(executor, other)
    out = [f"opt.{k}" for k in _fields(executor.opt)
           if not np.array_equal(np.asarray(getattr(executor.opt, k)), np.asarray(getattr(other.opt, k)))]
    out += [k for k in _arrays(executor) if not np.array_equal(getattr(executor, k), getattr(other, k),
                                                               equal_nan=True)]
    out += [f"stat.{k}" for k in _fields(executor.stat)
            if not np.array_equal(np.asarray(getattr(executor.stat, k)), np.asarray(getattr(other.stat, k)))]
    for p, q in _pair_map(executor, other).items():
        tag = f"pair[{int(other.pair_geom1[p])},{int(other.pair_geom2[p])}]"
        out += [f"{tag}.{k[5:]}" for k in _pair_fields(executor)
                if not np.array_equal(getattr(executor, k)[q], getattr(other, k)[p], equal_nan=True)]
    return out


def match_executor(dst: Any, executor: Any) -> list[str]:
    """Copy the executor's options, numeric arrays, statistics and surviving pair rows onto `dst`; return what
    differed (`executor_diff`). Raises on a layout mismatch."""
    changed = executor_diff(executor, dst)
    for k in _fields(executor.opt):
        v = getattr(executor.opt, k)
        if isinstance(v, np.ndarray):
            getattr(dst.opt, k)[...] = v
        else:
            setattr(dst.opt, k, v)
    for k in _arrays(executor):
        getattr(dst, k)[...] = getattr(executor, k)
    for k in _fields(executor.stat):
        v = getattr(executor.stat, k)
        if isinstance(v, np.ndarray):
            getattr(dst.stat, k)[...] = v
        else:
            setattr(dst.stat, k, v)
    for p, q in _pair_map(executor, dst).items():
        for k in _pair_fields(executor):
            getattr(dst, k)[p] = getattr(executor, k)[q]
    left = executor_diff(executor, dst)
    if left:
        raise InputError(f"could not copy {left[:5]} from the executor's model")
    return changed
