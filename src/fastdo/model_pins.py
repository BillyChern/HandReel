"""Pinned revisions of the external models the reconstruction loads: the one table that `scripts/fetch_weights.py`
downloads at and the loaders read at.

A Hugging Face loader called without a revision reads `refs/main` of the local cache, or asks the Hub what `main` is
now. `fetch_weights.py` downloads at a commit hash, which writes no `refs/main`, so an unpinned loader then finds
nothing offline and may fetch a different file online. Every loader of a model below therefore passes
`hf_revision(repo_id)`:

- `fastdo.shape.sam3d_backend.resolve_sam3d_pipeline_config` (SAM 3D Objects: `hf_revision(cfg.sam3d_repo)`, the repo id
  of `ShapeConfig.sam3d_repo`);
- `fastdo.shape.masks_sam2` (SAM 2.1 video and image predictors);
- `fastdo.geometry.moge_model._load_model` (MoGe-2);
- the SAM 3D worker's own MoGe v1 pointmap model, through `sam3d_worker.py --hf-revision` (`sam3d_hf_revisions`).

The torch.hub code pins are here too. CoTracker3's is the tracks stage's default (`FASTDO_COTRACKER_HUB_REF`
overrides it). DINOv2's is the commit `scripts/fetch_weights.py` unpacks into
`$TORCH_HOME/hub/facebookresearch_dinov2_main`, the directory SAM 3D's
`torch.hub.load("facebookresearch/dinov2", ...)` reads (release/QUICKSTART.md, step 3). A torch.hub directory's name
does not prove its content (`facebookresearch_dinov2_main` holds whatever `main` was when it was cached), so
`fetch_weights.py` writes `CODE_PIN_FILE` into every code directory it unpacks: the repository, the commit and
`code_tree_sha256` of the tree. `code_pin_status` checks a directory against it; `release/env/check_weights.py` reports
a directory without one as UNVERIFIED.

Stdlib only: `scripts/fetch_weights.py` and `release/env/check_weights.py` import it without torch.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

__all__ = [
    "CODE_PIN_FILE",
    "COTRACKER_HUB_COMMIT",
    "DINOV2_HUB_COMMIT",
    "HF_REVISIONS",
    "MOGE1_REPO",
    "SAM3D_REPO",
    "code_pin_status",
    "code_tree_sha256",
    "hf_revision",
    "sam3d_hf_revisions",
    "write_code_pin",
]

SAM3D_REPO = "facebook/sam-3d-objects"
MOGE1_REPO = "Ruicheng/moge-vitl"          # SAM 3D's checkpoints/pipeline.yaml `depth_model`
# repo id -> the commit our runs read (each equals the development host's cached refs/main, checked 2026-11-07).
HF_REVISIONS: dict[str, str] = {
    "facebook/sam2.1-hiera-large": "665f8e2ad61cf5f53d65644ff27c8ee525124610",
    SAM3D_REPO: "2e73555018d2741ccd486e56c24fac41155a1dc6",
    "Ruicheng/moge-2-vitl": "39c4d5e957afe587e04eec59dc2bcc3be5ecd968",
    MOGE1_REPO: "ad326bfb61facd6c52b5a825bc1e34d7c97d9672",
}
# torch.hub code. CoTracker3: the default of fastdo.tracks.cotracker.cotracker_hub_ref (the variable overrides it).
COTRACKER_HUB_COMMIT = "82e02e8029753ad4ef13cf06be7f4fc5facdda4d"
# DINOv2 (inside SAM 3D): the development host's cached `main`, equal file for file to this commit (2026-11-07). It
# carries the Cell-DINO code; whether to pin a pre-Cell-DINO commit instead is open (THIRD_PARTY_LICENSES.md).
DINOV2_HUB_COMMIT = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"


def hf_revision(repo_id: str) -> str | None:
    """The pinned commit of `repo_id`, or None for a model this table does not pin (its loader reads `main`)."""
    return HF_REVISIONS.get(repo_id)


def sam3d_hf_revisions() -> dict[str, str]:
    """The pins the SAM 3D worker applies to its pipeline YAML's Hugging Face loaders (MoGe v1)."""
    return {MOGE1_REPO: HF_REVISIONS[MOGE1_REPO]}


# ---- torch.hub code directories: what fetch_weights.py unpacked, and whether it is still that ----

CODE_PIN_FILE = ".fastdo_pin.json"


def code_tree_sha256(root: Path) -> str:
    """sha256 over the sorted `(relative path, file sha256)` pairs of the files under `root`, without `CODE_PIN_FILE`
    and Python bytecode (importing `hubconf.py` writes `__pycache__/` into the directory)."""
    entries: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        for name in filenames:
            path = Path(dirpath) / name
            rel = path.relative_to(root).as_posix()
            if rel == CODE_PIN_FILE or name.endswith(".pyc"):
                continue
            entries.append(f"{rel}\0{hashlib.sha256(path.read_bytes()).hexdigest()}")
    return hashlib.sha256("\n".join(sorted(entries)).encode()).hexdigest()


def write_code_pin(root: Path, repo: str, commit: str) -> None:
    """Records that `root` holds `repo` at `commit` as just unpacked (`code_tree_sha256` of it now)."""
    body = {"repo": repo, "commit": commit, "tree_sha256": code_tree_sha256(root)}
    (Path(root) / CODE_PIN_FILE).write_text(json.dumps(body, indent=1) + "\n")


def code_pin_status(root: Path, repo: str, commit: str, entry: str = "hubconf.py") -> tuple[str, str]:
    """`(status, detail)` of a code directory against `repo` at `commit`: `verified` (its `CODE_PIN_FILE` names that
    commit and the tree is unchanged), `unverified` (no `CODE_PIN_FILE`: not unpacked by fetch_weights.py, so its
    commit is unknown), `mismatch` (another repository or commit, or the tree changed since), or `missing`.

    `entry` is the file whose presence shows that the directory holds the checkout. It is `hubconf.py` for a
    torch.hub directory. A repository that torch.hub does not load has none (the pinned Hands23 detector has
    `demo.py` at its root and no `hubconf.py`), so its caller names its own entry."""
    root = Path(root)
    if not (root / entry).is_file():
        return "missing", f"{root / entry} missing"
    marker = root / CODE_PIN_FILE
    if not marker.is_file():
        return "unverified", (f"no {CODE_PIN_FILE}: not unpacked by scripts/fetch_weights.py, so its commit is "
                              f"unknown (pinned: {commit}); move it aside and re-run the fetch to pin it")
    try:
        body = json.loads(marker.read_text())
    except (OSError, ValueError) as exc:
        return "mismatch", f"{marker} unreadable ({exc})"
    if (body.get("repo"), body.get("commit")) != (repo, commit):
        return "mismatch", f"unpacked as {body.get('repo')}@{body.get('commit')}, pinned {repo}@{commit}"
    if body.get("tree_sha256") != code_tree_sha256(root):
        return "mismatch", f"files changed since they were unpacked at {commit}"
    return "verified", f"{repo}@{commit}"
