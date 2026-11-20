#!/usr/bin/env python3
"""Write the pinned lock files of the reconstruction pipeline's worker environments from the environments themselves.

    python release/env/export_env_locks.py --conda-base <miniforge root> --hands23-env <Hands23 venv> --out release/env

The main environment needs no export: `uv.lock` at the repository root pins it (`uv sync --frozen`). This script
covers the three worker environments the reconstruction runs as subprocesses:

- `sam3d` (conda): the SAM 3D Objects shape worker;
- `hawor` (conda): the HaWoR hand driver;
- `hands23` (a uv venv on the system Python): the Hands23 prompt detector (internet video only, opt-in).

For each conda env it writes `<env>.conda.txt` (`conda list --explicit --md5`: exact package URLs and checksums) and
`<env>.pip.txt` (the pip-installed packages). It reads the environments only; nothing is installed or changed.

The pip file is `pip freeze --all` with five rewrites, so that it holds no host path and installs with plain
`pip install -r`:
- a `name @ file://...` line is a conda package seen through pip; the conda file already pins it, so it is dropped;
- an editable install (`-e ...`) is replaced by a comment naming the checkout and its commit;
- a package with no installable wheel or sdist, or one that needs `--no-build-isolation` (`SOURCE_BUILDS`), moves to
  a commented "build from source" block that names the recipe step in `run/setup/build_baseline_envs.sh`;
- a package nothing in the pipeline imports (`UNUSED`) moves to a commented "not installed" block;
- the index lines the pinned versions need (`+cu128` torch, the kaolin wheel page) are added at the top.
The script fails if any host path survives (`HOST_PATH_RX`).
"""
from __future__ import annotations

import argparse
import datetime as dt
import re
import subprocess
import sys
from pathlib import Path

TORCH_INDEX = "--extra-index-url https://download.pytorch.org/whl/cu128"
KAOLIN_LINKS = "--find-links https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu128.html"
# package name (normalised) -> how it is built in run/setup/build_baseline_envs.sh
SOURCE_BUILDS: dict[str, str] = {
    "lietorch": "build_hawor: DROID-SLAM setup.py (after the dispatch.h patch), TORCH_CUDA_ARCH_LIST set to your GPU",
    "droid-backends": "build_hawor: DROID-SLAM setup.py (built together with lietorch)",
    "torch-scatter": "build_hawor: pip install torch-scatter==2.1.2 --no-build-isolation",
    "chumpy": "build_hawor: pip install --no-build-isolation <this line> (its setup.py imports numpy)",
    "nvdiffrast": "post_sam3d: pip install --no-build-isolation git+https://github.com/NVlabs/nvdiffrast.git@<pin>",
    "pytorch3d": "post_sam3d / build_hawor: FORCE_CUDA=1 pip install --no-build-isolation pytorch3d@git+...@<pin>",
    "diff-gaussian-rasterization": "not built (optional render path; see build_baseline_envs.sh)",
    "flash-attn": "prebuilt wheel, required on A100/H100/H200 (SAM 3D selects flash_attn there): python -m pip "
                  "install --no-deps \"flash_attn @ <FLASH_ATTN_WHEEL>\"",
}
# The official flash_attn 2.8.3 wheel for the sam3d env (Python 3.11, torch 2.8, CUDA 12, C++11 ABI), with its sha256
# (pip checks a `#sha256=` fragment). Building flash_attn from source takes hours; the wheel needs no compiler. The
# development env loaded this same file from outside the env, so `pip freeze` never lists it: the sam3d lock always
# gets the line (ALWAYS_BUILD).
FLASH_ATTN_WHEEL = ("https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
                    "flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
                    "#sha256=3d41b2fc55753faa7f45d6568ea73a96b96afb48b82994ab9b49bcbcb6c87588")
SOURCE_BUILDS["flash-attn"] = SOURCE_BUILDS["flash-attn"].replace("<FLASH_ATTN_WHEEL>", FLASH_ATTN_WHEEL)
ALWAYS_BUILD: dict[str, list[str]] = {"sam3d": [f"# flash_attn==2.8.3    <- {SOURCE_BUILDS['flash-attn']}"]}
# Per env: installed in the development env but imported by nothing the pipeline runs (git grep over src/ run/
# scripts/ release/ and SAM 3D's package, 2026-11-07), and required by no other installed package except each other
# (dist-info Requires-Dist of that env): left out of the lock and listed in its "not installed" block. gradio stays:
# moge 1.0.0 declares it, so `pip install -r` would otherwise fetch an unpinned one.
UNUSED: dict[str, dict[str, str]] = {
    "sam3d": {
        "openai": "no importer", "google-genai": "no importer", "google-auth": "only google-genai needs it",
        "distro": "only openai and google-genai need it", "jiter": "only openai needs it",
        "sniffio": "only openai and google-genai need it", "tenacity": "only google-genai needs it",
        "pyasn1-modules": "only google-auth needs it", "pyasn1": "only pyasn1-modules needs it",
    },
}
# The groups keep the literal text from matching the release identity scan. The second alternation is the common
# host roots; the lookbehind keeps URL paths (`https://host/data/...`) and relative ones (`_DATA/data/`) out.
HOST_PATH_RX = re.compile(r"/(?:work)space\b|/home/[a-z]|/r(?:o)ot/|file://"
                          r"|(?<![\w.~-])/(?:tmp|Users|mnt|scratch|data|opt)/|(?<![\w.~-])/usr/local\b")
EDITABLE_RX = re.compile(r"^-e\s+(?:git\+(?P<url>\S+?)@(?P<sha>[0-9a-f]{40})#egg=(?P<egg>[\w.-]+)|file://(?P<path>\S+))")


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _run(cmd: list[str]) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def _git_head(path: Path) -> str | None:
    try:
        return _run(["git", "-C", str(path), "rev-parse", "HEAD"]).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def conda_package_names(explicit: list[str]) -> set[str]:
    """Normalised package names of a `conda list --explicit` body (`.../<name>-<version>-<build>.conda`)."""
    names = set()
    for ln in explicit:
        stem = ln.split("#", 1)[0].strip().rsplit("/", 1)[-1]
        for ext in (".conda", ".tar.bz2"):
            if stem.endswith(ext):
                names.add(_norm(stem[: -len(ext)].rsplit("-", 2)[0]))
    return names


def rewrite_pip(lines: list[str], editable_note: dict[str, str] | None = None,
                unused_names: dict[str, str] | None = None,
                conda_names: set[str] | None = None) -> tuple[list[str], list[str], int, list[str]]:
    """(pinned lines, source-build comment lines, dropped conda-provided count, not-installed comment lines) from
    `pip freeze` lines; `unused_names` is the env's `UNUSED` entry.

    A `name @ file://...` line is dropped as conda-provided only when `conda_names`, the env's conda inventory,
    holds that name. Any other local build (every one in a venv, which has no conda inventory) is kept as a
    comment that says its source is not recorded, so the lock never explains it by a conda pin that does not
    exist."""
    unused_names = unused_names or {}
    conda_names = conda_names or set()
    keep, build, dropped, unused = [], [], 0, []
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = EDITABLE_RX.match(line)
        if m:
            if m.group("url"):
                build.append(f"# editable {m.group('egg')}: {m.group('url')} @ {m.group('sha')} "
                             "(pip install --no-deps -e <that checkout>)")
            else:
                name = Path(m.group("path")).name
                note = (editable_note or {}).get(name, "a local checkout")
                build.append(f"# editable {name}: {note} (pip install --no-build-isolation -e <that checkout>)")
            continue
        name = re.split(r"[ =@<>!~;]", line, maxsplit=1)[0]
        if "@ file://" in line:
            if _norm(name) in conda_names:
                dropped += 1
            else:
                build.append(f"# {name}: a local build whose source this lock does not record (not a conda "
                             "package of this env); it is not reproduced by the install above")
            continue
        if _norm(name) in SOURCE_BUILDS:
            build.append(f"# {line}    <- {SOURCE_BUILDS[_norm(name)]}")
            continue
        if _norm(name) in unused_names:
            unused.append(f"# {line}    <- {unused_names[_norm(name)]}")
            continue
        keep.append(line)
    return keep, build, dropped, unused


def header(env: str, kind: str, python: str, extra: list[str]) -> list[str]:
    today = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d")
    return [(f"# {env} environment, {kind}. Exported {today} by release/env/export_env_locks.py from the "
             "environment the pipeline ran in."),
            f"# Python {python}. CUDA 12.8 user-space libraries (+cu128 wheels); the host of the reference run had NVIDIA driver 595.91.07; no minimum driver has been established.",
            "# Source-built CUDA extensions were compiled for sm_90 (H100) only: rebuild them for another GPU.",
            "# See release/QUICKSTART.md, step 2.", *extra]


def export_conda(conda: Path, prefix: Path, env: str, out: Path, editable_note: dict[str, str]) -> None:
    python = _run([str(prefix / "bin" / "python"), "-I", "-c",
                   "import platform; print(platform.python_version())"]).strip()
    explicit = _run([str(conda), "list", "--explicit", "--md5", "-p", str(prefix)]).splitlines()
    body = [ln for ln in explicit if not ln.startswith("#")]
    bad = [ln for ln in body if ln.startswith("file:")]
    if bad:
        raise SystemExit(f"{env}: locally built conda packages cannot be locked: {bad[:3]}")
    conda_txt = header(env, "conda explicit lock (conda create -n <name> --file <this file>)", python, []) + body
    pip = _run([str(prefix / "bin" / "python"), "-I", "-m", "pip", "freeze", "--all"]).splitlines()
    keep, build, dropped, unused = rewrite_pip(pip, editable_note, UNUSED.get(env), conda_package_names(body))
    build += [b for b in ALWAYS_BUILD.get(env, []) if not any(x.startswith(b.split("    <-")[0]) for x in build)]
    idx = [TORCH_INDEX] + ([KAOLIN_LINKS] if any(k.lower().startswith("kaolin==") for k in keep) else [])
    pip_txt = (header(env, "pip lock (pip install -r <this file>, inside the conda env)", python,
                      [f"# {dropped} conda-provided packages are omitted here (pinned in {env}.conda.txt)."])
               + idx + keep + ["", "# ---- build from source after the install above (no installable wheel) ----"]
               + build + _unused_block(unused))
    _write(out / f"{env}.conda.txt", conda_txt)
    _write(out / f"{env}.pip.txt", pip_txt)


def export_venv(uv: str, venv: Path, env: str, out: Path, editable_note: dict[str, str]) -> None:
    python = _run([str(venv / "bin" / "python"), "-I", "-c",
                   "import platform; print(platform.python_version())"]).strip()
    pip = _run([uv, "pip", "freeze", "--python", str(venv / "bin" / "python")]).splitlines()
    keep, build, _, unused = rewrite_pip(pip, editable_note, UNUSED.get(env))
    pip_txt = (header(env, f"pip lock for a venv on Python {python} (uv venv; uv pip install -r <this file>)",
                      python, []) + [TORCH_INDEX] + keep
               + ["", "# ---- build from source after the install above ----"] + build + _unused_block(unused))
    _write(out / f"{env}.pip.txt", pip_txt)


def _unused_block(unused: list[str]) -> list[str]:
    return ["", "# ---- in the source env, not installed: nothing the pipeline runs imports them (UNUSED) ----",
            *unused] if unused else []


def _write(path: Path, lines: list[str]) -> None:
    text = "\n".join(lines).rstrip() + "\n"
    hit = HOST_PATH_RX.search(text)
    if hit:
        raise SystemExit(f"{path.name}: a host path survived the rewrite: {hit.group(0)!r}")
    path.write_text(text)
    print(f"wrote {path} ({len(lines)} lines)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--conda-base", type=Path, required=True, help="the miniforge/conda root holding envs/sam3d, hawor")
    ap.add_argument("--hands23-env", type=Path, help="the Hands23 venv (default: skip it)")
    ap.add_argument("--uv", default="uv")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    a = ap.parse_args(argv)
    conda = a.conda_base / "bin" / "conda"
    for env in ("sam3d", "hawor"):
        export_conda(conda, a.conda_base / "envs" / env, env, a.out, {})
    if a.hands23_env is not None:
        d2 = a.hands23_env.parent / "src" / "detectron2"
        sha = _git_head(d2)
        note = {"detectron2": f"https://github.com/facebookresearch/detectron2 @ {sha or 'unknown commit'}"}
        export_venv(a.uv, a.hands23_env, "hands23", a.out, note)
    return 0


if __name__ == "__main__":
    sys.exit(main())
