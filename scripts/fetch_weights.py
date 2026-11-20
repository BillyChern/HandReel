#!/usr/bin/env python3
"""List, and on --yes fetch, the model weights the fastdo pipeline needs. MANO is never fetched.

    python scripts/fetch_weights.py                     # print the table: source, revision, licence, sha256
    python scripts/fetch_weights.py --json              # the same, machine-readable
    python scripts/fetch_weights.py --yes [--only sam2,moge2,...] [--hawor-root DIR] [--hands23-root DIR]

Nothing is downloaded without --yes. By passing --yes you confirm that you have read and accepted each fetched model's
licence (several are NON-COMMERCIAL; see the table and THIRD_PARTY_LICENSES.md). SAM 3D Objects is gated on Hugging
Face: accept its terms on the model page and log in (`uv run --frozen hf auth login`) first.

Where files go (each is where the pipeline already looks):
- Hugging Face models (SAM 2.1, SAM 3D Objects, MoGe-2, and SAM 3D's MoGe v1): the Hugging Face cache (`$HF_HOME`),
  at the revision `fastdo.model_pins.HF_REVISIONS` pins. The loaders read at the same revisions (a download at a
  commit writes no `refs/main`, so an unpinned loader would not find it offline).
- CoTracker3: the torch.hub dir (`$TORCH_HOME/hub`). The code goes to `facebookresearch_co-tracker_<commit>`, which
  `torch.hub.load("facebookresearch/co-tracker:<commit>")` reads, so set FASTDO_COTRACKER_HUB_REF=<commit> to use it
  (release/example/run_example.py sets it); the checkpoint goes to `hub/checkpoints/`.
- DINOv2 (SAM 3D's image conditioner): the checkpoint goes to `hub/checkpoints/`, and the code at
  `fastdo.model_pins.DINOV2_HUB_COMMIT` to `hub/facebookresearch_dinov2_main`, the directory SAM 3D's unpinned
  `torch.hub.load("facebookresearch/dinov2", ...)` reads when it exists. An existing directory of that name is left as
  it is (its commit is whatever `main` was when it was cached).
- Every code directory this script unpacks gets `fastdo.model_pins.CODE_PIN_FILE` (repository, commit, tree digest).
  A present code directory without one is reported UNVERIFIED (not a failure: nothing is replaced); one whose pin file
  names another commit, or whose files changed since, is a MISMATCH and fails the run.
- HaWoR and the WiLoR detector: `<hawor root>/weights/...`; `--hawor-root` or `$FASTDO_HAWOR_ROOT` is REQUIRED with
  --yes (no default).
- Nothing is ever written under this repository's `third_party/` (the DO AS I DO submodule stays unmodified): a file
  that is missing there is refused, not fetched.
- Hands23: the repo at its pinned commit to `<hands23 root>/src/hands23_detector` and the weights to
  `<hands23 root>/weights/` (default `$FASTDO_HANDS23_ROOT`; see `fastdo.hostpaths`).

Every file with a known sha256 is verified, including files that are never downloaded but must be present (HaWoR's
`mano_mean_params.npz`, tracked in the HaWoR repository). A present file is never overwritten or deleted (only the
script's own download temporaries are removed): a mismatch is reported and fails the run (exit 1). A fresh download
that mismatches is kept as `<name>.sha256-mismatch`, which nothing reads. A mismatch in the HF cache or at a
destination stays where it is, and the pipeline reads that path: do not run the pipeline until it is resolved.
Exit status: 0 when every requested model is present and verified, 1 otherwise.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from fastdo.model_pins import (  # stdlib only
    COTRACKER_HUB_COMMIT,
    DINOV2_HUB_COMMIT,
    HF_REVISIONS,
    code_pin_status,
    write_code_pin,
)

URL_TIMEOUT_S = 120.0          # per blocking socket operation, not for the whole download
MANO_LICENSE_URL = "https://mano.is.tue.mpg.de/license.html"
MANO_MESSAGE = (f"MANO is NEVER fetched by this script. Register at https://mano.is.tue.mpg.de, accept the MANO "
                f"licence ({MANO_LICENSE_URL}; non-commercial, no redistribution), download it yourself and point "
                "FASTDO_MANO_DIR / FASTDO_MANO_LEFT_DIR at MANO_RIGHT.pkl / MANO_LEFT.pkl.")
# sam-2 is a uv git dependency; this is the rev uv.lock resolves it to (pyproject.toml names no rev).
SAM2_GIT_REV = "2b90b9f5ceec907a1c18123530e92e794ad901a4"


@dataclass(frozen=True)
class File:
    remote: str                 # path in the repo, or a full URL for kind "url"
    dest: str                   # where it lands, relative to the model's root ("" for the HF cache)
    sha256: str | None          # None: not published; the file is fetched but cannot be verified
    size: int | None = None
    fetch: bool = True          # False: never downloaded, only verified where it must already be (a tracked file)


@dataclass(frozen=True)
class Model:
    key: str
    name: str
    kind: str                   # hf | url | torch_hub | manual
    source: str                 # HF repo id, URL base or git repo
    revision: str | None
    licence: str
    licence_url: str
    commercial: bool
    gated: bool = False
    root: str = ""              # hf_cache | torch_hub | hawor | hands23
    files: tuple[File, ...] = field(default_factory=tuple)
    git_rev: str | None = None  # a code checkout fetched with the weights (torch.hub code, Hands23 repo)
    code_repo: str | None = None  # GitHub owner/repo of that checkout
    code_dest: str | None = None  # where it lands, relative to the model's root
    code_entry: str = "hubconf.py"  # a root file of that checkout; torch.hub's entry unless the model names its own
    note: str = ""


MODELS: tuple[Model, ...] = (
    Model("sam2", "SAM 2.1 Hiera-L (mask, prompt)", "hf", "facebook/sam2.1-hiera-large",
          HF_REVISIONS["facebook/sam2.1-hiera-large"], "Apache-2.0",
          "https://github.com/facebookresearch/sam2/blob/main/LICENSE", True, root="hf_cache",
          files=(File("sam2.1_hiera_large.pt", "", "2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318",
                      898083611),),
          note=f"code: sam-2 git rev {SAM2_GIT_REV} (uv.lock); its model config ships in the sam-2 package"),
    Model("sam3d", "SAM 3D Objects (shape)", "hf", "facebook/sam-3d-objects",
          HF_REVISIONS["facebook/sam-3d-objects"], "SAM License (custom; read it)",
          "https://huggingface.co/facebook/sam-3d-objects/blob/main/LICENSE", False, gated=True, root="hf_cache",
          files=(
               File("checkpoints/pipeline.yaml", "", "53c3d226b21df85c0bb3d16e6e4fa63abde0d6167525765eb929d02bfa9d358c", 3548),
               File("checkpoints/slat_decoder_gs.ckpt", "", "f8077c36a06eaf890dd93cda1937411f793dea1eb80b3dd9329f2038ba84a111", 171476155),
               File("checkpoints/slat_decoder_gs.yaml", "", "53f054e02a0c185f0a6885d30bb6ca0ec92efe0a98148688e7f05f7c26afe670", 576),
               File("checkpoints/slat_decoder_gs_4.ckpt", "", "731a0eceaa47945b52aa27f650d695b2aea9cc70945751e5609e5cb5b49f0186", 170269801),
               File("checkpoints/slat_decoder_gs_4.yaml", "", "3d1dfd4c56cdac56f30e0cf5eb310d4df973fe25c60ac0a462c86ca4e3bf8b48", 575),
               File("checkpoints/slat_decoder_mesh.ckpt", "", "85907b37b67d8ce5b099a96629bdcfbd873eb407dee6b3aa9a75deb15038db33", 363726862),
               File("checkpoints/slat_decoder_mesh.pt", "", "93333fcd57a3e36ded0b3bca6969e05ce2b35142029dadab514f41df46d2f985", 363728714),
               File("checkpoints/slat_decoder_mesh.yaml", "", "8f46952764aa985c50109a56f0b4f07625cdb06457aaded74957d26f3520c69e", 300),
               File("checkpoints/slat_encoder.ckpt", "", "6485623145535f42c8afa4cbb68ab9953e54e2f0c1cb1eaf95dcb41051e10181", 173263986),
               File("checkpoints/slat_encoder.yaml", "", "ca80da7f1395a2acef9c6cea30ec3ccc89321ef890c644860f316a4ef1e8ebdb", 268),
               File("checkpoints/slat_generator.ckpt", "", "91529bde8e7daa12d09618a66c319e3a5a6398db6b23b958cedcb1c3f28faabb", 4906537684),
               File("checkpoints/slat_generator.yaml", "", "53029fadff6fe34a0344381a16d64d65d0567d603484952712e8969319559c4e", 1986),
               File("checkpoints/ss_decoder.ckpt", "", "6dac1cd7b7fda5a38e0614fadae441f1794f80e39ea2981f1ac8aff0a7e99340", 147609242),
               File("checkpoints/ss_decoder.yaml", "", "baacff269b664f84f7aa1896ebd66128065f6568cdb88d419d5c3d2ccb4193ae", 244),
               File("checkpoints/ss_encoder.ckpt", "", "dcc47810ac568b11fe6e4821ea1c8d6b960dfbda3e5f68e94c19f44b3bf9e83b", 119085402),
               File("checkpoints/ss_encoder.safetensors", "", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", 0),
               File("checkpoints/ss_encoder.yaml", "", "d3df5e18b2cbe87697d319adce0fe7e4fb6266809ad9d2f8c502761739bc4a06", 231),
               File("checkpoints/ss_generator.ckpt", "", "225f40479e4cff4f39d6fa14c55be3abad1475bf55b61af3bec1e19ed2f6c146", 6690136964),
               File("checkpoints/ss_generator.yaml", "", "3c265448bca7c057f94e3ef56adea3a895a10bcd9f15f992a41dd03fa35412cd", 5076),
          ),
          note="GATED: accept the terms on the HF page and log in. The checkpoint sha256s equal HF's LFS ids; the "
               "yaml digests are of our copy. ss_encoder.safetensors is empty upstream too (size 0 in the HF tree). "
               "Shape stage non-commercial as run: NVIDIA's nvdiffrast (and kaolin's non_commercial package) run in its "
               "process"),
    Model("moge1_sam3d", "MoGe ViT-L v1 (SAM 3D's pointmap model)", "hf", "Ruicheng/moge-vitl",
          HF_REVISIONS["Ruicheng/moge-vitl"], "MIT", "https://github.com/microsoft/MoGe/blob/main/LICENSE", True,
          root="hf_cache",
          files=(File("model.pt", "", "da96b09a0485a3c45a5aa455e67743c8b4efc4dd8437c1f2aa93c2b4303d957f", 1256823446),),
          note="read by the SAM 3D worker at this revision (sam3d_worker.py --hf-revision)"),
    Model("dinov2_sam3d", "DINOv2 ViT-L/14 + registers (SAM 3D's image conditioner)", "url",
          "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/", None, "Apache-2.0 (backbone code and weights)",
          "https://github.com/facebookresearch/dinov2/blob/main/LICENSE", True, root="torch_hub",
          files=(File("dinov2_vitl14_reg4_pretrain.pth", "checkpoints/dinov2_vitl14_reg4_pretrain.pth",
                      "36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51", 1217607321),),
          git_rev=DINOV2_HUB_COMMIT, code_repo="facebookresearch/dinov2", code_dest="facebookresearch_dinov2_main",
          note="unversioned weights URL (sha256 is of our copy). The hub code at that commit also imports "
               "Cell-DINO modules (CC BY 4.0 / CC BY-NC statements); SAM 3D uses none (THIRD_PARTY_LICENSES.md M14)"),
    Model("moge2", "MoGe-2 ViT-L (intrinsics, geometry)", "hf", "Ruicheng/moge-2-vitl",
          HF_REVISIONS["Ruicheng/moge-2-vitl"], "MIT", "https://github.com/microsoft/MoGe/blob/main/LICENSE",
          True, root="hf_cache",
          files=(File("model.pt", "", "3eefd4abb2102f38f12b2d1992e5ff15e4923e5431c67dd494afe157e0111cd5",
                      1305030700),),
          note="code: moge git rev b942f00bdc2a2a23ebb474fbe034d487e6dcceec (pyproject.toml)"),
    Model("cotracker3", "CoTracker3 online (tracks)", "torch_hub", "facebookresearch/co-tracker",
          "bf55ea50d4390e1820a267f131cd6587240fb2c5", "CC BY-NC 4.0",
          "https://github.com/facebookresearch/co-tracker/blob/main/LICENSE.md", False, root="torch_hub",
          files=(File("https://huggingface.co/facebook/cotracker3/resolve/bf55ea50d4390e1820a267f131cd6587240fb2c5/"
                      "scaled_online.pth", "checkpoints/scaled_online.pth",
                      "205d34789f19699d64b22cf93f9b697f15f28d4025240e31532e504109837218", 101695610),),
          git_rev=COTRACKER_HUB_COMMIT, code_repo="facebookresearch/co-tracker",
          code_dest=f"facebookresearch_co-tracker_{COTRACKER_HUB_COMMIT}",
          note="weights revision is HF facebook/cotracker3; code is the torch.hub repo at git_rev"),
    Model("hawor", "HaWoR (hands)", "url", "https://huggingface.co/ThunderVVV/HaWoR/resolve/"
          "da6335f47f9806308992d5ae1002a4cc5f7252c2/", "da6335f47f9806308992d5ae1002a4cc5f7252c2",
          "CC BY-NC-ND 4.0", "https://github.com/ThunderVVV/HaWoR/blob/main/license.txt", False, root="hawor",
          files=(File("hawor/checkpoints/hawor.ckpt", "weights/hawor/checkpoints/hawor.ckpt",
                      "4d1cc43853c190d6f2c10d9b6295c73109f0faf9ef41ac817a2b31d94b4823f2", 3267481572),
                 File("hawor/model_config.yaml", "weights/hawor/model_config.yaml",
                      "edfe12dc14ce371d698da722b59acfed5b4a38a7f8f5116cbc1fce459a07dd2d", 2743),
                 File("", "_DATA/data/mano_mean_params.npz",
                      "efc0ec58e4a5cef78f3abfb4e8f91623b8950be9eff8b8e0dbb0d036ebc63988", 1178, fetch=False)),
          note="weights from HF ThunderVVV/HaWoR; --hawor-root is a clone of the malik-group/HaWoR fork at "
               "2c3fa0c06e54a1b192c1ff692eaa89d526cd6317 (the commit DO AS I DO pins), outside third_party/"),
    Model("wilor_detector", "WiLoR hand detector (via HaWoR)", "url",
          "https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/99fe3d7acff8104ecca1055df7467709506c2fa6/",
          "99fe3d7acff8104ecca1055df7467709506c2fa6", "CC BY-NC-ND 4.0 (stricter of two conflicting statements)",
          "https://github.com/rolpotamias/WiLoR/blob/main/license.txt", False, root="hawor",
          files=(File("pretrained_models/detector.pt", "weights/external/detector.pt",
                      "5ef3df44e42d2db52d4ffe91f83a22ce9925e2acc9abebf453f2c5d22e380033", 53582271),),
          note="loaded by ultralytics (AGPL-3.0)"),
    Model("hands23", "Hands23 detector (internet prompt stage)", "url",
          "https://fouheylab.eecs.umich.edu/~dandans/projects/hands23/model_weights/", None,
          "code MIT; weights no licence text; training data CC BY-NC 4.0",
          "https://github.com/EvaCheng-cty/hands23_detector", False, root="hands23",
          files=(File("model_hands23.pth", "weights/model_hands23.pth",
                      "b3d2ff966d8a19b991a3ce170e2d8f86040f614ef2edf4a55423288b278f723c", 467798701),),
          git_rev="1bc0f919ffa7a9f375e7e8042e1d26f3743d5819", code_repo="EvaCheng-cty/hands23_detector",
          code_dest="src/hands23_detector", code_entry="demo.py",      # not a torch.hub repository: no hubconf.py
          note="unversioned URL; sha256 is the copy the pipeline was measured with. Needs a detectron2 env"),
    Model("mano", "MANO hand model (hands, retarget)", "manual", "https://mano.is.tue.mpg.de", None,
          "MANO licence (non-commercial; no redistribution)", MANO_LICENSE_URL, False, note=MANO_MESSAGE),
)


def by_key() -> dict[str, Model]:
    return {m.key: m for m in MODELS}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def table() -> str:
    rows = ["| key | model | source | revision | licence | commercial (as run, note 1) | sha256 |",
            "|---|---|---|---|---|---|---|"]
    for m in MODELS:
        sha = (", ".join(f"`{f.sha256[:12]}`" if f.sha256 else "unpublished" for f in m.files) if len(m.files) <= 3
               else f"{len(m.files)} files, {sum(f.sha256 is not None for f in m.files)} with a pinned sha256") or "n/a"
        rows.append(f"| {m.key} | {m.name}{' (GATED)' if m.gated else ''} | {m.source} | {m.revision or 'unversioned'}"
                    f"{' + code ' + m.git_rev[:8] if m.git_rev else ''} | {m.licence} | "
                    f"{'yes' if m.commercial else '**no**'} | {sha} |")
    rows += ["", "Note 1, licence as run: whether commercial use is allowed for the model as this pipeline runs it, not "
             "for its model licence alone. For example, the SAM 3D Objects licence allows commercial use, but its "
             "shape stage loads non-commercial code (nvdiffrast, kaolin's non_commercial package) in the same "
             "process, so it reads **no** (release/MODELS.md)."]
    return "\n".join(rows)


# ---- transports (injected, so tests run on fakes with no network) ----

def _url_fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=URL_TIMEOUT_S) as r, open(dest, "wb") as fh:
        shutil.copyfileobj(r, fh)


def _hf_fetch(repo: str, pattern: str, revision: str) -> list[Path]:
    from huggingface_hub import snapshot_download

    snap = Path(snapshot_download(repo_id=repo, revision=revision, allow_patterns=[pattern]))
    return sorted(p for p in snap.glob(pattern) if p.is_file())


def _git_zip_fetch(repo: str, rev: str, dest: Path, url_fetch: Callable[[str, Path], None]) -> None:
    """GitHub archive of `repo` at `rev`, extracted so `dest` holds the tree (no `.git`)."""
    z = dest.with_name(dest.name + ".zip")
    tmp = dest.with_name(dest.name + ".extract")
    _refuse_existing(z, tmp, dest)
    url_fetch(f"https://github.com/{repo}/archive/{rev}.zip", z)
    with zipfile.ZipFile(z) as zf:
        for n in zf.namelist():
            if n.startswith("/") or ".." in Path(n).parts:
                raise RuntimeError(f"unsafe path in archive: {n}")
        zf.extractall(tmp)
    (top,) = list(tmp.iterdir())
    if dest.exists():
        raise RuntimeError(f"{dest} already exists; not replacing it")
    top.replace(dest)
    shutil.rmtree(tmp)
    z.unlink()


@dataclass
class Transports:
    url: Callable[[str, Path], None] = _url_fetch
    hf: Callable[[str, str, str], list[Path]] = _hf_fetch
    git_zip: Callable[[str, str, Path, Callable[[str, Path], None]], None] = _git_zip_fetch


def roots(args: argparse.Namespace) -> dict[str, Path]:
    sys.path.insert(0, str(REPO / "src"))
    from fastdo import hostpaths

    torch_home = Path(os.environ.get("TORCH_HOME") or Path.home() / ".cache" / "torch")
    out = {"torch_hub": Path(args.hub_dir or torch_home / "hub"),
           "hands23": Path(args.hands23_root or hostpaths.hands23_root())}
    hawor = args.hawor_root or os.environ.get("FASTDO_HAWOR_ROOT")
    if hawor:                                # no default: HaWoR's usual home is the baseline submodule
        out["hawor"] = Path(hawor)
    return out


def third_party_dirs(repo: Path) -> list[Path]:
    """`third_party/` of this checkout and of EVERY worktree of its repository (the primary checkout included), found
    through `git rev-parse --git-common-dir` and `git worktree list`. Without git, this checkout's only."""
    out = {(repo / "third_party").resolve()}
    try:
        common = subprocess.run(["git", "-C", str(repo), "rev-parse", "--path-format=absolute", "--git-common-dir"],
                                capture_output=True, text=True, check=True).stdout.strip()
        out.add((Path(common).parent / "third_party").resolve())     # the primary checkout owns the common dir
        lst = subprocess.run(["git", "--git-dir", common, "worktree", "list", "--porcelain"], capture_output=True,
                             text=True, check=True).stdout
        out |= {(Path(line[len("worktree "):]) / "third_party").resolve() for line in lst.splitlines()
                if line.startswith("worktree ")}
    except (OSError, subprocess.CalledProcessError):
        pass
    return sorted(out)


def _refuse_third_party(path: Path) -> None:
    """Nothing is ever written under any worktree's third_party/ (the DO AS I DO submodule stays unmodified)."""
    p = path.resolve()
    for tp in third_party_dirs(REPO):
        if p == tp or tp in p.parents:
            raise RuntimeError(f"refusing to write {path}: it is under {tp}, which this script never modifies. Pass "
                               "--hawor-root / --hands23-root / --hub-dir pointing outside every third_party/")


def _refuse_existing(*paths: Path) -> None:
    """The script's temporary names must be free: a leftover is never reused, replaced or deleted."""
    for p in paths:
        if p.exists() or p.is_symlink():
            raise RuntimeError(f"{p} already exists (a leftover of an earlier run?); inspect and remove it, then re-run")


def _verify(path: Path, f: File, read_by_pipeline: bool = True) -> bool:
    """Report whether `path` matches `f.sha256`. It never deletes or replaces anything (an HF-cache entry is a
    shared link; a mismatch is for the user to inspect). `read_by_pipeline` False for a fresh download, which the
    caller then renames to a name nothing reads."""
    if f.sha256 is None:
        print(f"  {path}: present, sha256 unpublished (unverified)")
        return True
    got = sha256_file(path)
    if got != f.sha256:
        where = ("left in place, and the pipeline reads this path: do not run it until this is resolved"
                 if read_by_pipeline else "not installed")
        print(f"  {path}: sha256 MISMATCH {got} != {f.sha256}; {where}", file=sys.stderr)
        return False
    print(f"  {path}: sha256 ok")
    return True


def _refuse_mano(*names: str) -> None:
    """A last guard: no transport is ever called with anything that names MANO."""
    for n in names:
        if "mano" in n.lower():
            raise RuntimeError(f"refusing to fetch {n!r}: {MANO_MESSAGE}")


def fetch(m: Model, where: dict[str, Path], t: Transports) -> bool:
    """Fetch one model; True when every file is present and (where a sha256 is known) verified."""
    if m.kind == "manual" or m.key == "mano":
        print(MANO_MESSAGE)
        return True
    fetched = [f for f in m.files if f.fetch]         # a verify-only file is never handed to a transport
    _refuse_mano(m.source, *(f.remote for f in fetched), *(f.dest for f in fetched))
    print(f"== {m.key}: {m.name} [{m.licence}]")
    ok = True
    if m.kind == "hf":
        for f in m.files:
            paths = t.hf(m.source, f.remote, str(m.revision))
            if not paths:
                print(f"  {m.source}@{m.revision}: nothing matched {f.remote}", file=sys.stderr)
                ok = False
            for p in paths:                      # verify every file, not just up to the first failure
                ok = _verify(p, f) and ok
        return ok
    root = where[m.root]
    if m.git_rev and m.code_repo and m.code_dest:
        code = root / m.code_dest
        if code.exists():                    # never replaced; a `<repo>_main` directory may hold any commit
            status, detail = code_pin_status(code, m.code_repo, m.git_rev, m.code_entry)
            if status == "verified":
                print(f"  code: {code} present, verified ({detail})")
            elif status == "unverified":
                print(f"  code: {code} present, left as it is, UNVERIFIED: {detail}", file=sys.stderr)
            else:
                print(f"  code: {code} present, left as it is, MISMATCH: {detail}", file=sys.stderr)
                ok = False
        else:
            _refuse_third_party(code)
            code.parent.mkdir(parents=True, exist_ok=True)
            t.git_zip(m.code_repo, m.git_rev, code, t.url)
            write_code_pin(code, m.code_repo, m.git_rev)
            print(f"  code: {code} @ {m.git_rev}")
    for f in m.files:
        dest = root / f.dest
        if dest.exists():                    # present: verify, never overwrite
            ok = _verify(dest, f) and ok
            continue
        if not f.fetch:                      # tracked in the checkout: must be there, never downloaded
            print(f"  {dest}: MISSING (it is tracked in the {m.key} repository checkout; not downloaded)",
                  file=sys.stderr)
            ok = False
            continue
        _refuse_third_party(dest)
        part = dest.with_name(dest.name + ".part")
        bad = dest.with_name(dest.name + ".sha256-mismatch")
        _refuse_existing(part, bad)
        url = f.remote if f.remote.startswith("https://") else m.source + f.remote
        t.url(url, part)
        if _verify(part, f, read_by_pipeline=False):
            part.replace(dest)
            print(f"  {dest}: installed")
        else:                                # keep the bad download for inspection, under a name nothing reads
            part.replace(bad)
            print(f"  kept the mismatched download as {bad}", file=sys.stderr)
            ok = False
    return ok


def main(argv: list[str] | None = None, transports: Transports | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--yes", action="store_true", help="download; you accept each fetched model's licence")
    ap.add_argument("--only", default="", help="comma-separated keys (default: all)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--hawor-root", type=Path)
    ap.add_argument("--hands23-root", type=Path)
    ap.add_argument("--hub-dir", type=Path)
    a = ap.parse_args(argv)
    models = by_key()
    keys = [k for k in a.only.split(",") if k] or list(models)
    unknown = sorted(set(keys) - set(models))
    if unknown:
        ap.error(f"unknown keys {unknown}; known: {sorted(models)}")
    if a.json:
        print(json.dumps([asdict(models[k]) for k in keys], indent=1))
    else:
        print(table())
        print()
        print(MANO_MESSAGE)
    if a.json and a.yes:
        ap.error("--json prints the manifest and downloads nothing; run --yes without --json to fetch")
    if not a.yes:
        # with --json, stdout is one JSON document: the note goes to stderr
        print("\nNothing downloaded: pass --yes to fetch (you accept each model's licence).",
              file=sys.stderr if a.json else sys.stdout)
        return 0
    where = roots(a)
    need_hawor = sorted(k for k in keys if models[k].root == "hawor")
    if need_hawor and "hawor" not in where:
        ap.error(f"{need_hawor} need --hawor-root or FASTDO_HAWOR_ROOT with --yes (there is no default: HaWoR's usual "
                 "home is the DO AS I DO submodule, which this script never writes to)")
    need_hands23 = sorted(k for k in keys if models[k].root == "hands23")
    if need_hands23 and not a.hands23_root:
        from fastdo import hostpaths  # importable: roots() put src/ on the path

        try:
            hostpaths.require(where["hands23"], "--hands23-root or FASTDO_HANDS23_ROOT", create=True)
        except hostpaths.HostPathNotSet as e:
            ap.error(f"{need_hands23} with --yes: {e}")
    t = transports or Transports()
    results: dict[str, bool] = {}
    errors: dict[str, str] = {}
    for k in keys:                  # one key's failure (a gated repo, a network error) never stops the others
        try:
            results[k] = fetch(models[k], where, t)
        except OSError as e:        # urllib, huggingface_hub (GatedRepoError, HfHubHTTPError) and disk errors
            results[k] = False
            errors[k] = f"{type(e).__name__}: {e}".splitlines()[0]
            print(f"== {k}: NOT FETCHED ({errors[k]})", file=sys.stderr)
    bad = sorted(k for k, ok in results.items() if not ok)
    print(summary(models, keys, bad, errors))
    return 1 if bad else 0


def how_to_obtain(m: Model) -> str:
    """One line: what a user must do before `m` can be fetched again."""
    if m.gated:
        return (f"{m.source} is gated: accept its terms at https://huggingface.co/{m.source} with your Hugging Face "
                f"account, run `uv run --frozen hf auth login`, then rerun with --only {m.key} (revision {m.revision})")
    return f"check the network and the message above, then rerun with --only {m.key}"


def summary(models: dict[str, Model], keys: list[str], bad: list[str], errors: dict[str, str]) -> str:
    """The closing report: every requested key's state, how to get the missing ones, and MANO."""
    lines = ["", "Summary:"]
    for k in keys:
        state = "missing" if k in bad else ("manual (see MANO below)" if k == "mano" else "ok")
        lines.append(f"  {k:16s} {state}{': ' + errors[k] if k in errors else ''}")
    for k in bad:
        lines.append(f"  to get {k}: {how_to_obtain(models[k])}")
    lines.append(f"  MANO (never fetched here): {MANO_MESSAGE}")
    lines.append(f"FAILED: {bad}" if bad else "every requested key is present and verified")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
