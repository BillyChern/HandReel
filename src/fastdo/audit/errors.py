# Copyright 2026 The HandReel authors. SPDX-License-Identifier: Apache-2.0
"""The one exception the audit raises for an input it refuses."""

from __future__ import annotations

__all__ = ["InputError"]


class InputError(ValueError):
    """An input the audit refuses: missing, malformed or inconsistent with the scene. The CLI exits 2 on it with
    the message; every other exception is an internal error and propagates with its traceback."""
