#!/usr/bin/env python3
"""Write release/MODELS.md from release/models.json, so the two never drift (tests/release checks the output).

    python release/render_models_md.py            # rewrite release/MODELS.md
    python release/render_models_md.py --check    # exit 1 if MODELS.md differs from what models.json gives
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REL = Path(__file__).resolve().parent
HEAD = """# External models of the reconstruction pipeline

The external files pinned for the released configurations of the reconstruction (stages decode to qc, plus the
opt-in prompt stage), with where each comes from, its licence, and the sha256 to verify it against: the digest of
the file the authors' runs had in place. The list includes files that only an opt-in stage or flag reads, files fetched by hand,
and one zero-byte placeholder. A digest pins a file; it does not show that a given run opened it. **No weights are shipped.**
`models.json` holds the same data in machine-readable form, and this page is generated from it
(`release/render_models_md.py`); `tests/release/test_release_manifest.py` keeps the two in step and checks them
against `scripts/fetch_weights.py`.

How to fetch: `uv run --frozen python scripts/fetch_weights.py` prints the download table; `--yes --only <keys>`
downloads the listed files and compares each with its SHA-256 (QUICKSTART step 3). Two things stay outside that
automatic check. MANO is a manual download that the script neither fetches nor hashes. A code directory that was
already present without the script's pin file (for example a tree that torch.hub cached) is left as it is and
reported UNVERIFIED, which is a warning and not a failure. Four model loaders (SAM 2.1, SAM 3D Objects, MoGe v1 and MoGe-2) fetch and load at the revisions of
`src/fastdo/model_pins.py`. For HaWoR and the WiLoR detector, the hand worker opens local files: verify them against
the source revisions and SHA-256 values listed below. MANO is never downloaded for you.

Path variables: `$HF_HOME` (default `~/.cache/huggingface`), `$TORCH_HOME` (default `~/.cache/torch`), and the
`FASTDO_*` variables of README.md, "Host paths".

**Outputs are bound too.** What the pipeline writes (meshes, poses, hand parameters, robot trajectories) is produced
by these models, so their licences (non-commercial, NoDerivatives, MANO's) also govern what you may do with the
outputs. Whether an output is a derivative work is an open question for several of them (THIRD_PARTY_LICENSES.md).

This states what the licence texts say as we read them. It is not legal advice.
"""


def _cell(text: str) -> str:
    return str(text).replace("|", "\\|")


def render(manifest: dict[str, Any]) -> str:
    models = manifest["models"]
    out = [HEAD, "## Summary", "", "| key | model | stage | env | licence | commercial use | flags |",
           "|---|---|---|---|---|---|---|"]
    for m in models:
        out.append(f"| `{m['key']}` | {_cell(m['name'])} | {', '.join(m['stages'])} | {m['env']} | "
                   f"{_cell(m['license'])} | {'yes' if m['commercial_use'] else '**no**'} | "
                   f"{_cell('; '.join(m['flags'])) or '-'} |")
    out += ["", manifest["summary"], "", "## Code licence flags in the worker environments", "",
            manifest["code_licence_note"], "", "| package | licence | env | how the pipeline uses it | notes |",
            "|---|---|---|---|---|"]
    for c in manifest["code_licence_flags"]:
        out.append(f"| {_cell(c['package'])} | {_cell(c['licence'])} | {c['env']} | {_cell(c['use'])} | "
                   f"{_cell(c['decision'])} |")
    out += ["", "## Per model", ""]
    for m in models:
        out += [f"### `{m['key']}`: {m['name']}", "", f"- **Version:** {m['version']}", f"- **Code:** {m['code']}",
                f"- **Source:** <{m['source_url']}>",
                f"- **Licence:** {m['license']} (<{m['license_url']}>){' **GATED**' if m['gated'] else ''}",
                f"- **Fetch:** {m['fetch']}", "", "| file | bytes | sha256 | expected at |", "|---|---|---|---|"]
        for f in m["files"]:
            out.append(f"| `{f['file']}` | {f['size']:,} | `{f['sha256']}` | `{f['expected_at']}` |")
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    text = render(json.loads((REL / "models.json").read_text()))
    target = REL / "MODELS.md"
    if a.check:
        same = target.read_text() == text
        print("MODELS.md is up to date" if same else "MODELS.md differs from models.json: run release/render_models_md.py")
        return 0 if same else 1
    target.write_text(text)
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
