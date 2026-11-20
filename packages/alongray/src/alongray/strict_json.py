# Copyright 2026 The alongray authors. SPDX-License-Identifier: Apache-2.0
"""Strict JSON reading for every JSON input alongray reads (rows, protocols, reports).

`json.loads` keeps the LAST value of a repeated key, so a row such as {"status": "ok", ..., "status": "failed"} is read
silently as one of the two. :func:`loads` and :func:`load` refuse a repeated key in any JSON object, nested objects
included, and name the key and the place (``where``, e.g. "rows.jsonl line 3").
"""

from __future__ import annotations

import json
from typing import IO, Any

__all__ = ["DuplicateKeyError", "load", "loads"]


class DuplicateKeyError(ValueError):
    """A JSON object repeats a key."""


def loads(text: str, where: str = "JSON input", **kw: Any) -> Any:
    """``json.loads(text, **kw)`` with repeated keys refused (the object hooks are reserved)."""
    if "object_pairs_hook" in kw or "object_hook" in kw:
        raise TypeError("strict_json.loads sets the object hook itself")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in items:
            if k in out:
                raise DuplicateKeyError(f"{where}: repeated key {k!r}")
            out[k] = v
        return out

    return json.loads(text, object_pairs_hook=pairs, **kw)


def load(fh: IO[str], where: str = "JSON input", **kw: Any) -> Any:
    """:func:`loads` on the text of an open file."""
    return loads(fh.read(), where, **kw)
