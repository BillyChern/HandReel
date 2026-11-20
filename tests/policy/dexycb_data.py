"""Where the DexYCB policy tests find real DexYCB data, the baseline conda envs and the reference hand files (no tests).

Only subjects 03-06 (and, for the ported left-hand validation, subject 09) are opened; subjects 07 and 08 are reserved
for calibration and confirmation and no test reads any of their sequences (`RESERVED_SUBJECTS`).

Candidates resolve in order: an environment variable, this checkout, then the primary checkout (a worktree has no
`data/` or `temp/`). Tests that need a missing input skip with the reason.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PRIMARY_ROOT = Path("/opt/fastdo/handreel")


def _first(*candidates: Path | None) -> Path | None:
    return next((p for p in candidates if p is not None and p.exists()), None)


def _env(name: str) -> Path | None:
    value = os.environ.get(name, "")
    return Path(value) if value else None


DEXYCB_ROOT = _first(_env("FASTDO_DEXYCB_ROOT"), REPO_ROOT / "data" / "dexycb", PRIMARY_ROOT / "data" / "dexycb")
MANO_PCA = _first(REPO_ROOT / "data" / "mano" / "mano_pca.npz", PRIMARY_ROOT / "data" / "mano" / "mano_pca.npz")
def _dir_with(marker: str, *candidates: Path | None) -> Path | None:
    """The first candidate directory holding `marker` (a worktree's third_party/do-as-i-do exists but is empty)."""
    return next((p for p in candidates if p is not None and (p / marker).exists()), None)


_BASELINE_REL = Path("third_party") / "do-as-i-do" / "retargeting"
_HAWOR_REL = Path("third_party") / "do-as-i-do" / "reconstruction" / "modules" / "HaWoR"
BASELINE_ROOT = _dir_with("launch.py", REPO_ROOT / _BASELINE_REL, PRIMARY_ROOT / _BASELINE_REL)   # retargeting/
HAWOR_ROOT = _dir_with("lib/models/mano_wrapper.py", _env("FASTDO_HAWOR_ROOT"), REPO_ROOT / _HAWOR_REL,
                       PRIMARY_ROOT / _HAWOR_REL)
CONDA_ENVS = Path("/opt/fastdo/miniforge3/envs")
RETARGETING_PYTHON = _first(CONDA_ENVS / "retargeting" / "bin" / "python")
HAWOR_PYTHON = _first(_env("FASTDO_HAWOR_PYTHON"), CONDA_ENVS / "hawor" / "bin" / "python")
MANO_RIGHT_PKL = _first(Path("/opt/fastdo/HaWoR/_DATA/data/mano/MANO_RIGHT.pkl"))
MANO_LEFT_PKL = _first(Path("/opt/fastdo/HaWoR/_DATA/data_left/mano_left/"
                            "MANO_LEFT.pkl"))
WHISKING_SCENE_XML = _first(PRIMARY_ROOT / "third_party" / "do-as-i-do" / "retargeting" / "outputs" / "sharpa" / "right"
                            / "whisking" / "0" / "scene.xml")
H2H_CLIPS = _first(PRIMARY_ROOT / "temp" / "h2h" / "clips")
REGTRACKS_HANDS = _first(PRIMARY_ROOT / "temp" / "regtracks" / "out" / "hands")

RESERVED_SUBJECTS = ("subject-07", "subject-08")
SERIAL = "836212060125"
SEQ_RIGHT = "20200820-subject-03/20200820_135250"          # 002_master_chef_can, right hand, 72 frames
SEQ_RIGHT_FILLED = "20200903-subject-04/20200903_104655"   # 006_mustard_bottle, right hand, 74 frames, rows 0-10 without GT
SEQ_LEFT = "20201015-subject-09/20201015_143556"           # left hand, 74 frames (regtracks validation, pass)
SEQ_LEFT_OCCLUDED = "20201015-subject-09/20201015_142815"  # left hand, label fraction 0.571 (regtracks, fail)
FOUR_OBJECT_SEQS = (                                       # subject 03, right hand, four distinct grasped objects
    ("20200820-subject-03/20200820_135250", "002_master_chef_can"),
    ("20200820-subject-03/20200820_135545", "003_cracker_box"),
    ("20200820-subject-03/20200820_135841", "004_sugar_box"),
    ("20200820-subject-03/20200820_140158", "005_tomato_soup_can"),
)

dexycb_missing = pytest.mark.skipif(DEXYCB_ROOT is None or MANO_PCA is None,
                                    reason="DexYCB data or data/mano/mano_pca.npz not on this host")
hawor_missing = pytest.mark.skipif(
    HAWOR_PYTHON is None or HAWOR_ROOT is None or DEXYCB_ROOT is None or MANO_PCA is None or MANO_RIGHT_PKL is None
    or MANO_LEFT_PKL is None, reason="hawor env, HaWoR checkout, DexYCB data or MANO models not on this host")
retargeting_env_missing = pytest.mark.skipif(RETARGETING_PYTHON is None or BASELINE_ROOT is None,
                                             reason="retargeting conda env or the baseline checkout not on this host")
whisking_scene_missing = pytest.mark.skipif(WHISKING_SCENE_XML is None, reason="baseline whisking scene not on this host")


def assert_not_reserved(sequence: str) -> None:
    """Guard used by the tests before they open a sequence."""
    assert not any(sequence.split("/")[0].endswith(s) for s in RESERVED_SUBJECTS), sequence
