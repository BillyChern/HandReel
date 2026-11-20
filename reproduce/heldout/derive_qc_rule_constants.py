"""Derive `qc_rule_constants.json` from the QC rule file that the held-out protocol registered by sha256.

    python3 reproduce/heldout/derive_qc_rule_constants.py <registered rule file> --out <file>
    python3 reproduce/heldout/derive_qc_rule_constants.py <registered rule file> --check reproduce/heldout/qc_rule_constants.json

The registered file (sha256 `REGISTERED_SHA256`) is not released: its provenance names the development clip and
sequence IDs it was fitted on, host paths and an internal commit. This script keeps every constant of it unchanged
(the keys in `CONSTANT_KEYS`) and, of its provenance, only the fit's counts and settings in `PROVENANCE_KEYS`, and
adds a note and the registered hash prefix. The output is written byte for byte as the shipped copy was, so anyone
holding the registered file can confirm the copy with `--check` (exit 0 only if the bytes match). By default the
script refuses an input whose sha256 is not the registered one. With `--allow-unregistered` (tests use that) it
accepts another input, and then nothing it writes or prints is a statement about the registered rule: `--out` writes
a file whose provenance says UNREGISTERED and names the input's sha256, and `--check` says that a match shows only
that this input carries the same constants.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REGISTERED_SHA256 = "e564b52e92d3a5ad6392734e36d23efda2d305aed5f2d6ce119271a2816150f0"
CONSTANT_KEYS = ("format", "version", "feature_names", "logistic_weights", "feature_mean", "feature_std", "tau",
                 "calibration")
PROVENANCE_KEYS = ("n_clips", "n_skipped", "n_good", "n_bad", "alpha", "tau", "fit_fraction", "seed",
                   "laplace_std_threshold", "coverage_guaranteed", "shape_source", "label")
NOTE = ("Constants-only copy of the QC acceptance rule that the held-out test's protocol registered by sha256 "
        "(original prefix e564b52e92d3). Every key above is the original's value, unrounded. The original's "
        "provenance listed the development clip and sequence IDs it was fitted on, host paths and an internal "
        "commit; those are left out, so this file's sha256 differs from the registered one.")


UNREGISTERED_NOTE = ("UNREGISTERED: constants-only copy of an input whose sha256 is not the one that the held-out "
                     "test's protocol registered. This file is not the registered QC rule and not a copy of it.")


def derive(src: dict[str, Any], unregistered_sha256: str | None = None) -> bytes:
    """The constants-only copy of the rule `src`. For the registered file (`unregistered_sha256` None) these are the
    bytes of the shipped file. For any other input the provenance says so and carries that input's sha256 instead
    of the registered prefix, so the output cannot pass for the registered rule."""
    prov = src["provenance"]
    out = {k: src[k] for k in CONSTANT_KEYS}
    head = ({"copy": NOTE, "registered_sha256_prefix_of_original": REGISTERED_SHA256[:12]}
            if unregistered_sha256 is None else
            {"copy": UNREGISTERED_NOTE, "registered": False, "input_sha256": unregistered_sha256})
    out["provenance"] = {**head, **{k: prov[k] for k in PROVENANCE_KEYS}}
    return (json.dumps(out, indent=1, allow_nan=False) + "\n").encode()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("rule", type=Path, help="the registered QC rule file (JSON)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--out", type=Path, help="write the constants-only copy here")
    g.add_argument("--check", type=Path, help="compare the derived bytes with this file")
    ap.add_argument("--allow-unregistered", action="store_true", help="accept an input with another sha256")
    a = ap.parse_args(argv)
    raw = a.rule.read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    if got != REGISTERED_SHA256 and not a.allow_unregistered:
        raise SystemExit(f"derive_qc_rule_constants: {a.rule} has sha256 {got}, not the registered {REGISTERED_SHA256}")
    unregistered = got != REGISTERED_SHA256
    if a.out is not None:
        data = derive(json.loads(raw), got if unregistered else None)
        a.out.write_bytes(data)
        print(f"wrote {a.out} (sha256 {hashlib.sha256(data).hexdigest()})"
              + (f"; UNREGISTERED input {got}: not the registered rule" if unregistered else ""))
        return 0
    # --check compares the constants and counts: the bytes the registered file would give for this input's values
    data = derive(json.loads(raw))
    same = a.check.read_bytes() == data
    print(f"{a.check}: {'matches' if same else 'DIFFERS FROM'} the copy derived from {a.rule} "
          f"(sha256 {hashlib.sha256(data).hexdigest()})")
    if unregistered:
        print(f"UNREGISTERED input {got}: a match shows only that this input carries the same constants and counts, "
              "not that it is the registered file")
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
