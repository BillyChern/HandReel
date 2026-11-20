#!/usr/bin/env python3
"""Check that every weight file of release/models.json is where the pipeline reads it, with the listed size (and,
with --sha256, the listed digest), plus the two torch.hub code directories. Reads files only; downloads nothing.

    python release/env/check_weights.py [--sha256] [--with-hands23] [--only mano,...]
    python release/env/check_weights.py --sha256sum-list [--only mano] | sha256sum -c -

Paths expand `$HF_HOME` (default `~/.cache/huggingface`), `$TORCH_HOME` (default `~/.cache/torch`) and the `FASTDO_*`
variables of docs/installation.md, "Host paths"; an unset `FASTDO_*` variable is a failure, never a guess. Hands23 is opt-in (the
internet-video prompt stage only). A torch.hub code directory is OK only when its `fastdo.model_pins.CODE_PIN_FILE`
(written by `scripts/fetch_weights.py`) names the pinned commit and its files are unchanged; without that file it is
UNVERIFIED (a cached `facebookresearch_dinov2_main` holds whatever `main` was), which does not fail the run; another
commit or changed files fail it. Exit status 0 when every check passes. Stdlib only; `release/env/verify.sh` runs it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections.abc import Mapping
from pathlib import Path

REL = Path(__file__).resolve().parents[1]
REPO = REL.parent
DEFAULTS = {"HF_HOME": "~/.cache/huggingface", "TORCH_HOME": "~/.cache/torch"}
_VAR = re.compile(r"\$([A-Z_][A-Z0-9_]*)")


def expand(template: str, env: Mapping[str, str]) -> Path | str:
    """`template` with its variables filled in, or the name of the first unset `FASTDO_*` variable (a str)."""
    names = _VAR.findall(template)
    for name in names:
        if name not in env and name not in DEFAULTS:
            return name
    out = _VAR.sub(lambda m: env.get(m.group(1)) or DEFAULTS[m.group(1)], template)
    return Path(os.path.expanduser(out))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256sum_list(manifest: Mapping, env: Mapping[str, str], only: set[str] | None = None) -> list[str]:
    """`sha256sum -c` lines (`<sha256>  <path>`) for every expected file of the models in `only` (default: all).

    Raises:
        SystemExit: If a path names an unset `FASTDO_*` variable.
    """
    lines = []
    for m in manifest["models"]:
        if only is not None and m["key"] not in only:
            continue
        for f in m["files"]:
            for template in f["expected_at"].split(" and "):
                where = expand(template.strip(), env)
                if isinstance(where, str):
                    raise SystemExit(f"{m['key']}/{f['file']}: set {where} first")
                lines.append(f"{f['sha256']}  {where}")
    return lines


def checks(manifest: Mapping, env: Mapping[str, str], with_hands23: bool = False, sha: bool = False,
           only: set[str] | None = None) -> list[tuple[bool, str]]:
    """`(ok, line)` for every expected file (of the models in `only`, default all) and the torch.hub code dirs."""
    sys.path.insert(0, str(REPO / "src"))
    from fastdo.model_pins import COTRACKER_HUB_COMMIT, DINOV2_HUB_COMMIT, code_pin_status

    out: list[tuple[bool, str]] = []
    for m in manifest["models"]:
        if (m["key"] == "hands23" and not with_hands23 and only is None) or (only is not None and m["key"] not in only):
            continue
        for f in m["files"]:
            for template in f["expected_at"].split(" and "):
                where = expand(template.strip(), env)
                if isinstance(where, str):
                    out.append((False, f"{m['key']}/{f['file']}: {where} is not set"))
                    continue
                if not where.is_file():
                    out.append((False, f"{m['key']}/{f['file']}: missing at {where}"))
                elif where.stat().st_size != f["size"]:
                    size = where.stat().st_size
                    out.append((False, f"{m['key']}/{f['file']}: {where} has {size} bytes, expected {f['size']}"))
                elif sha and _sha256(where) != f["sha256"]:
                    out.append((False, f"{m['key']}/{f['file']}: {where} sha256 differs from {f['sha256']}"))
                else:
                    out.append((True, f"{m['key']}/{f['file']}: {where}"))
    hub = expand("$TORCH_HOME/hub", env)
    assert isinstance(hub, Path)
    codes = ((f"facebookresearch_co-tracker_{COTRACKER_HUB_COMMIT}", "facebookresearch/co-tracker", COTRACKER_HUB_COMMIT),
             ("facebookresearch_dinov2_main", "facebookresearch/dinov2", DINOV2_HUB_COMMIT))
    for code, repo, commit in () if only is not None else codes:
        status, detail = code_pin_status(hub / code, repo, commit)
        line = f"torch.hub code: {hub / code}"
        if status == "verified":
            out.append((True, f"{line} ({detail})"))
        elif status == "unverified":         # passes (nothing wrong is visible) but is labelled, never plain OK
            out.append((True, f"{line} {UNVERIFIED}: {detail}"))
        elif status == "missing":
            out.append((False, f"{line}/hubconf.py missing"))
        else:
            out.append((False, f"{line} {status}: {detail}"))
    return out


UNVERIFIED = "UNVERIFIED"


def label(ok: bool, line: str) -> str:
    """The printed status of one check: FAIL, UNVERIFIED (passes, but its content is not known to be the pin), OK."""
    if not ok:
        return "FAIL "
    return f"{UNVERIFIED} " if f" {UNVERIFIED}: " in line else "OK   "


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sha256", action="store_true", help="also hash every file (about 22 GB; minutes)")
    ap.add_argument("--with-hands23", action="store_true", help="check the opt-in Hands23 weights too")
    ap.add_argument("--only", default="", help="comma-separated model keys (default: all)")
    ap.add_argument("--sha256sum-list", action="store_true", help="print `sha256sum -c` input instead of checking")
    a = ap.parse_args(argv)
    manifest = json.loads((REL / "models.json").read_text())
    only = {k for k in a.only.split(",") if k} or None
    unknown = sorted((only or set()) - {m["key"] for m in manifest["models"]})
    if unknown:
        ap.error(f"unknown keys {unknown}")
    if a.sha256sum_list:
        print("\n".join(sha256sum_list(manifest, os.environ, only)))
        return 0
    results = checks(manifest, os.environ, a.with_hands23, a.sha256, only)
    for ok, line in results:
        print(label(ok, line) + line)
    return 0 if all(ok for ok, _ in results) else 1


if __name__ == "__main__":
    sys.exit(main())
