"""Locate files of the DO AS I DO submodule (`third_party/do-as-i-do`) for tests.

`baseline_path` returns the path inside this checkout when it exists, and otherwise the same path under the
placeholder layout `/opt/fastdo/handreel`. A test that needs the file still skips with its own reason when neither
exists (for example, a clone in which `scripts/fetch_baseline.sh` has not run).
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_REL = Path("third_party/do-as-i-do")
PLACEHOLDER_ROOT = Path("/opt/fastdo/handreel")

__all__ = ["REPO_ROOT", "baseline_path"]


def baseline_path(rel: str | Path) -> Path:
    """Return `third_party/do-as-i-do/<rel>` in this checkout if it exists, else under the placeholder root."""
    local = REPO_ROOT / BASELINE_REL / rel
    if local.exists():
        return local
    return PLACEHOLDER_ROOT / BASELINE_REL / rel
