# Copyright 2026 The alongray authors. SPDX-License-Identifier: Apache-2.0
"""Strict JSON parsing for every JSON input that reproduce/ reads: the one shared helper, `alongray.strict_json`
(alongray 0.4.1), re-exported here so the scripts import it from one place.

`json.loads` keeps the LAST value of a repeated key, so a manifest row such as {"ours": "a.npz", ..., "ours": null}
silently means non-delivery. `loads` refuses any repeated key in any object (nested objects included) and names the
key and the place (`where`, e.g. "manifest line 3").
"""

from __future__ import annotations

from alongray.strict_json import DuplicateKeyError, load, loads

__all__ = ["DuplicateKeyError", "load", "loads"]
