"""Host path defaults for code that cannot read the Hydra config, each overridable by an environment variable.

`run/conf/config.yaml` declares the same variables under `paths.*` (with `${oc.env:VAR,default}`), so the config and
this module agree. With no variable set, every function returns a path under the placeholder layout `/opt/fastdo`
(the defaults are pinned by `tests/test_hostpaths.py`); set the variables on your machine. A variable that is set must
be non-empty.

| variable | default | used for |
|---|---|---|
| `FASTDO_EXT_ROOT` | `/opt/fastdo` | the host directory holding this checkout and its sibling checkouts |
| `FASTDO_PRIMARY_ROOT` | `$FASTDO_EXT_ROOT/handreel` | the primary checkout (scratch and outputs never live in a worktree) |
| `FASTDO_SCRATCH_ROOT` | `$FASTDO_PRIMARY_ROOT/temp` | host scratch: unit registries, the Hands23 env |
| `FASTDO_GPU_TOOLS_DIR` | unset: no shared GPU lock | opt-in shared lock directory: `gpu.lock`, `gpu_priority.json`, `peer_hold.py` |
| `FASTDO_UNIT_REGISTRY` | `$FASTDO_SCRATCH_ROOT/units` | the amortised-unit registry |
| `FASTDO_MANO_DIR` | HaWoR's `_DATA/data/mano` in a sibling checkout | the user's own `MANO_RIGHT.pkl` (never shipped) |
| `FASTDO_MANO_LEFT_DIR` | HaWoR's `_DATA/data_left/mano_left` | the user's own `MANO_LEFT.pkl` (never shipped) |
| `FASTDO_HANDS23_ROOT` | `$FASTDO_SCRATCH_ROOT/hands23` | the Hands23 env (`env_hands23/`), weights (`weights/`) and repo (`src/`) |
| `FASTDO_EGO4D_LISTS` | `$FASTDO_PRIMARY_ROOT/data/ego4d/lists` | the Ego4D video-uid lists (licensed metadata, never shipped) |
| `FASTDO_V2D_TRACK3_ROOT` | `$FASTDO_EXT_ROOT/data/v2d_track3_public/track_3/public` | the video_to_data track-3 public set |

The shared GPU lock is opt-in. With `FASTDO_GPU_TOOLS_DIR` unset, `gpu_tools_dir()` is None and the amortised units
(`fastdo.amortised.unit`) run with no host-wide lock and no admission check. Set it, to the same directory for every
GPU job on a host, to make those jobs take turns on `gpu.lock` there. It is never derived from the root variables, so
moving a root (e.g. a worktree's own scratch) never moves it.

`fastdo.prompt.hands23_detect` runs in the Hands23 env without `fastdo` on its path, so it repeats the
`FASTDO_HANDS23_ROOT` lookup inline; `tests/test_hostpaths.py` checks that the two agree.

The functions above only compute paths. Code that is about to read or create something under a host path calls
`require` first (`check_stage_paths` and `check_config_paths` do so for config keys): a path that is still
under the placeholder layout raises `HostPathNotSet`, naming the variable to set, instead of failing later on a missing
file or creating directories under `/opt/fastdo`. Nothing is checked once `/opt/fastdo` exists, so a host that really
uses that layout runs unchanged. `tests/test_hostpaths_unset.py` runs the entry points with no variable set.
"""
from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

__all__ = [
    "HOST_ROOT",
    "ISOLATION_HOST_KEYS",
    "STAGE_HOST_KEYS",
    "HostPathNotSet",
    "check_config_paths",
    "check_stage_paths",
    "ego4d_lists_dir",
    "env_path",
    "ext_root",
    "gpu_tools_dir",
    "hands23_root",
    "mano_dir",
    "mano_left_dir",
    "primary_root",
    "require",
    "scratch_root",
    "unit_registry_dir",
    "v2d_track3_root",
]

HOST_ROOT = "/opt/fastdo"
logger = logging.getLogger(__name__)
_HAWOR_DATA = "HaWoR/_DATA"


def env_path(var: str, default: Path) -> Path:
    """`$var` as a path when it is set, else `default`.

    Raises:
        ValueError: When `var` is set to the empty string (an unset value would silently mean the current directory).
    """
    v = os.environ.get(var)
    if v is None:
        return default
    if not v:
        raise ValueError(f"{var} is set but empty; unset it to use the default {default}")
    return Path(v)


def ext_root() -> Path:
    return env_path("FASTDO_EXT_ROOT", Path(HOST_ROOT))


def primary_root() -> Path:
    return env_path("FASTDO_PRIMARY_ROOT", ext_root() / "handreel")


def scratch_root() -> Path:
    return env_path("FASTDO_SCRATCH_ROOT", primary_root() / "temp")


def gpu_tools_dir() -> Path | None:
    """The opt-in shared GPU lock directory: `$FASTDO_GPU_TOOLS_DIR`, or None (no shared lock) when it is unset.

    Raises:
        ValueError: When the variable is set to the empty string.
    """
    v = os.environ.get("FASTDO_GPU_TOOLS_DIR")
    if v is None:
        return None
    if not v:
        raise ValueError("FASTDO_GPU_TOOLS_DIR is set but empty; unset it to run without a shared GPU lock")
    return Path(v)


def unit_registry_dir() -> Path:
    return env_path("FASTDO_UNIT_REGISTRY", scratch_root() / "units")


def mano_dir() -> Path:
    return env_path("FASTDO_MANO_DIR", ext_root() / _HAWOR_DATA / "data" / "mano")


def mano_left_dir() -> Path:
    return env_path("FASTDO_MANO_LEFT_DIR", ext_root() / _HAWOR_DATA / "data_left" / "mano_left")


def hands23_root() -> Path:
    return env_path("FASTDO_HANDS23_ROOT", scratch_root() / "hands23")


def ego4d_lists_dir() -> Path:
    return env_path("FASTDO_EGO4D_LISTS", primary_root() / "data" / "ego4d" / "lists")


def v2d_track3_root() -> Path:
    return env_path("FASTDO_V2D_TRACK3_ROOT", ext_root() / "data" / "v2d_track3_public" / "track_3" / "public")


class HostPathNotSet(RuntimeError):
    """A host path is still the placeholder default under `/opt/fastdo`, which does not exist on this machine."""


def _under_placeholder(path: Path) -> bool:
    root = Path(HOST_ROOT)
    return path == root or root in path.parents


def require(path: str | os.PathLike[str], variables: str, *, create: bool = False) -> Path:
    """`path` as a `Path`, unless it lies under the placeholder layout `/opt/fastdo` while that layout is not there.

    Call it before reading (`create=False`) or creating (`create=True`) anything at a host path. A path outside
    `/opt/fastdo` is returned unchecked. Under it, an input path must exist, and a path to create needs `/opt/fastdo`
    itself to exist, so nothing is ever created under the placeholder.

    Args:
        path: The resolved host path.
        variables: The environment variables (or config keys) that move this path, named in the error.
        create: True when the caller is about to create `path` (or files in it) rather than read it.

    Raises:
        HostPathNotSet: When `path` is under `/opt/fastdo` and the check above fails.
    """
    p = Path(path)
    if not _under_placeholder(p):
        return p
    if create and Path(HOST_ROOT).is_dir():
        return p
    if not create and p.exists():
        return p
    raise HostPathNotSet(
        f"{p} is a placeholder default under {HOST_ROOT}, which does not exist on this machine: set {variables} "
        f"(README.md, \"Host paths\"). Nothing was created under {HOST_ROOT}.")


# Config keys that the real engine stages read as host paths, with the variables that move them. `check_stage_paths`
# checks the ones whose stage runs; the stub workers and injected StageFns read none of them.
STAGE_HOST_KEYS: dict[str, tuple[tuple[str, str], ...]] = {
    "hands": (("hands.hawor_python", "FASTDO_CONDA_BASE (or hands.hawor_python=<interpreter>)"),),
    "prompt": (("prompt_stage.python", "FASTDO_HANDS23_PYTHON or FASTDO_HANDS23_ROOT"),
               ("prompt_stage.weights", "FASTDO_HANDS23_WEIGHTS or FASTDO_HANDS23_ROOT")),
    "retarget": (("retarget_stage.mano_pkl_path.right", "FASTDO_MANO_DIR"),
                 ("retarget_stage.mano_pkl_path.left", "FASTDO_MANO_LEFT_DIR"),
                 ("retarget_stage.scene_xml", ("FASTDO_PRIMARY_ROOT (or isolation.baseline_root=<baseline "
                                               "retargeting dir>, or retarget_stage.scene_xml=<scene.xml>)"))),
}


def check_stage_paths(cfg: Any, stages: Iterable[str]) -> None:
    """Checks, with `require`, every `STAGE_HOST_KEYS` path of the stages that will run, before any work starts.

    The `prompt` stage runs when `prompt_stage.enabled` is true, `retarget` when it is in `stages` and
    `retarget_stage.enabled` is true, `hands` when it is in `stages`. Keys set to null are skipped, and so are the
    other hand's keys when `retarget_stage.side` fixes the side (the right hand's scene is `retarget_stage.scene_xml`).

    Raises:
        HostPathNotSet: Listing every unset path at once.
    """
    from omegaconf import OmegaConf

    def on(key: str) -> bool:
        return bool(OmegaConf.select(cfg, key, default=False))

    names = set(stages)
    run = [s for s in ("hands", "retarget") if s in names and (s != "retarget" or on("retarget_stage.enabled"))]
    if on("prompt_stage.enabled"):
        run.append("prompt")
    side = OmegaConf.select(cfg, "retarget_stage.side", default=None)
    skip = {"right": {"retarget_stage.mano_pkl_path.left"},
            "left": {"retarget_stage.mano_pkl_path.right", "retarget_stage.scene_xml"}}.get(str(side), set())
    check_config_paths(cfg, [(f"{stage} stage", key, variables, False) for stage in run
                             for key, variables in STAGE_HOST_KEYS[stage] if key not in skip])


# Config keys of the baseline-physics pipelines (`run/pipeline/retarget_whisking.py`, `profile_retarget_kinematic.py`,
# `build_retarget_scenes.py`, `build_ycb_scenes.py`): (key, variables, create). They default under the primary
# checkout, so `FASTDO_PRIMARY_ROOT` (or `FASTDO_EXT_ROOT`) moves them all.
ISOLATION_HOST_KEYS: tuple[tuple[str, str, bool], ...] = (
    ("isolation.baseline_root", "FASTDO_PRIMARY_ROOT (or isolation.baseline_root=<baseline retargeting dir>)", False),
    ("isolation.whisk_obj", "FASTDO_PRIMARY_ROOT (or isolation.whisk_obj=<whisk.obj>)", False),
    ("isolation.output_root", "FASTDO_PRIMARY_ROOT (or isolation.output_root=<dir>)", True),
    ("isolation.temp_dir", "FASTDO_SCRATCH_ROOT (or isolation.temp_dir=<dir>)", True),
)


def check_config_paths(cfg: Any, items: Iterable[tuple[str, str, str, bool]]) -> None:
    """Checks, with `require`, each `(label, config key, variables, create)` of `items` before any work starts.

    A key that is absent or null is skipped.

    Raises:
        HostPathNotSet: Listing every unset path at once.
    """
    from omegaconf import OmegaConf

    problems = []
    for label, key, variables, create in items:
        value = OmegaConf.select(cfg, key, default=None)
        if value is None:
            continue
        try:
            require(str(value), variables, create=create)
        except HostPathNotSet as e:
            problems.append(f"{label}, {key}: {e}")
    if problems:
        raise HostPathNotSet("host paths are not set for this run:\n  " + "\n  ".join(problems))
