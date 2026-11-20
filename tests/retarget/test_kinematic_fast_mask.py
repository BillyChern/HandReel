"""Masked frames in the fast kinematic reference (D6 amendment: hand rows with `valid` False must not create contacts or
enter the retargeting cost), on the shipped whisking scene with CPU MuJoCo.

- `frame_valid=None` and an all-True mask give bitwise the same solve.
- On masked frames the human measure has no in-band atom (so no transport and no band residual) and the wrist prior's
  rows carry weight 0.
- The solve does not depend on masked frames' hand vertices or wrist targets at all, initialization included: NaN there
  gives bitwise the same `qpos`.
- The legacy path refuses a mask, and a malformed or all-False mask raises before any solve.

The demo data resolve against this checkout's `third_party/do-as-i-do`, else `$FASTDO_BASELINE_ROOT` (a worktree
without the submodule checked out).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import trimesh
from torch import Tensor

from fastdo.retarget import kinematic_fast
from fastdo.retarget.baseline_io import baseline_world_inputs
from fastdo.retarget.contact import human_contact_measure
from fastdo.retarget.kinematic import RetargetConfig, _human_atom_subset, retarget_kinematic
from fastdo.retarget.labels import mano_finger_labels
from fastdo.retarget.mj_model import RobotScene
from fastdo.retarget.transport import InbandMeasure
from fastdo.solver.sdf import mesh_sdf_grid
from tests.retarget.whisking_data import MANO_RIGHT_PKL, MESH_SCALE

REPO_ROOT = Path(__file__).resolve().parents[2]
_SCENE = Path("retargeting/outputs/sharpa/right/whisking/0/scene.xml")
_KEYPOINTS = Path("retargeting/outputs/mano/right/whisking/0/trajectory_keypoints.npz")
_TRAJECTORY = _SCENE.parent / "trajectory_kinematic.npz"
_WHISK = Path("reconstruction/whisking/video_segmentation/masks/frame_000125_masks/whisk/whisk.obj")


def _baseline_root() -> Path | None:
    candidates = [REPO_ROOT / "third_party" / "do-as-i-do"]
    if os.environ.get("FASTDO_BASELINE_ROOT"):
        candidates.append(Path(os.environ["FASTDO_BASELINE_ROOT"]))
    for root in candidates:
        if all((root / rel).is_file() for rel in (_SCENE, _KEYPOINTS, _TRAJECTORY, _WHISK)):
            return root
    return None


BASELINE_ROOT = _baseline_root()
pytestmark = pytest.mark.skipif(BASELINE_ROOT is None or MANO_RIGHT_PKL is None,
                                reason="whisking demo data (third_party or $FASTDO_BASELINE_ROOT) or MANO pkl missing")

WINDOW = np.arange(58, 66)            # 8 demo frames, every one with in-band human contact atoms (64-atom subset)
MASKED = [3, 4]                       # demo frames 61 and 62


@pytest.fixture(scope="module")
def setup() -> dict[str, Any]:
    assert BASELINE_ROOT is not None and MANO_RIGHT_PKL is not None
    obj, wrist, verts, frequency = baseline_world_inputs(BASELINE_ROOT / _KEYPOINTS, BASELINE_ROOT / _TRAJECTORY)
    mesh = trimesh.load(BASELINE_ROOT / _WHISK, process=False)
    assert isinstance(mesh, trimesh.Trimesh)
    sdf = mesh_sdf_grid(np.asarray(mesh.vertices), np.asarray(mesh.faces), resolution=64)
    surf_pts, face_idx = trimesh.sample.sample_surface(mesh, 1024, seed=np.random.default_rng(0))
    idx = torch.as_tensor(WINDOW)
    return {"scene": RobotScene.from_xml(BASELINE_ROOT / _SCENE), "sdf": sdf,
            "surface": torch.as_tensor(np.asarray(surf_pts), dtype=torch.float64),
            "normals": torch.as_tensor(np.asarray(mesh.face_normals[face_idx]), dtype=torch.float64),
            "obj": obj[idx], "wrist": wrist[idx], "verts": verts[idx], "frequency": frequency,
            "labels": mano_finger_labels(MANO_RIGHT_PKL)}


def _mask() -> Tensor:
    valid = torch.ones(WINDOW.size, dtype=torch.bool)
    valid[MASKED] = False
    return valid


def _solve(s: dict[str, Any], wrist: Tensor | None = None, verts: Tensor | None = None,
           frame_valid: Tensor | None = None, cfg: RetargetConfig | None = None) -> np.ndarray:
    traj = retarget_kinematic(s["obj"], s["wrist"] if wrist is None else wrist, MESH_SCALE,
                              s["verts"] if verts is None else verts, s["labels"], s["scene"], s["sdf"], s["surface"],
                              s["normals"], s["frequency"], cfg if cfg is not None else RetargetConfig(),
                              frame_valid=frame_valid)
    return traj.qpos


@pytest.fixture(scope="module")
def unmasked(setup: dict[str, Any]) -> np.ndarray:
    return _solve(setup)


@pytest.fixture(scope="module")
def masked_run(setup: dict[str, Any]) -> tuple[np.ndarray, list[InbandMeasure], list[Tensor]]:
    """The masked solve, recording every in-band measure and every wrist-prior weight the fast path builds."""
    measures: list[InbandMeasure] = []
    wrist_weights: list[Tensor] = []
    real_measure = kinematic_fast.inband_measure
    real_prior = kinematic_fast.FastTermBuilder._wrist_prior

    def measure_spy(*args: Any, **kwargs: Any) -> InbandMeasure:
        measures.append(real_measure(*args, **kwargs))
        return measures[-1]

    def prior_spy(self: kinematic_fast.FastTermBuilder, *args: Any, **kwargs: Any) -> Any:
        term = real_prior(self, *args, **kwargs)
        wrist_weights.append(term.weight.detach().clone())
        return term

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(kinematic_fast, "inband_measure", measure_spy)
        mp.setattr(kinematic_fast.FastTermBuilder, "_wrist_prior", prior_spy)
        qpos = _solve(setup, frame_valid=_mask())
    return qpos, measures, wrist_weights


def test_all_true_mask_is_bitwise_the_unmasked_solve(setup: dict[str, Any], unmasked: np.ndarray) -> None:
    assert np.array_equal(_solve(setup, frame_valid=torch.ones(WINDOW.size, dtype=torch.bool)), unmasked)


def test_masked_frames_create_no_contact_and_leave_the_wrist_prior(
        setup: dict[str, Any], masked_run: tuple[np.ndarray, list[InbandMeasure], list[Tensor]]) -> None:
    cfg = RetargetConfig()
    subset = _human_atom_subset(778, cfg.n_human_atoms, cfg.seed)
    human = human_contact_measure(setup["verts"][:, subset], setup["obj"], MESH_SCALE, setup["labels"][subset],
                                  setup["sdf"], setup["surface"], sigma=cfg.sigma, tau=cfg.tau)
    assert bool((human.weights > 0).any(dim=1).all())                    # unmasked, every window frame is in contact
    qpos, measures, wrist_weights = masked_run
    (measure,) = measures
    expected = _mask()
    assert torch.equal(measure.nonempty, expected)
    assert wrist_weights and all(w.shape == (WINDOW.size, 6) for w in wrist_weights)
    for weight in wrist_weights:
        assert torch.equal(weight, expected.to(weight.dtype)[:, None].expand(-1, 6))
    assert np.isfinite(qpos).all()


def test_masked_solve_ignores_masked_frames_inputs(setup: dict[str, Any], unmasked: np.ndarray,
                                                  masked_run: tuple[np.ndarray, list[InbandMeasure], list[Tensor]]
                                                  ) -> None:
    masked_qpos = masked_run[0]
    wrist, verts = setup["wrist"].clone(), setup["verts"].clone()
    wrist[MASKED] = float("nan")
    verts[MASKED] = float("nan")
    assert np.array_equal(_solve(setup, wrist=wrist, verts=verts, frame_valid=_mask()), masked_qpos)
    assert float(np.abs(masked_qpos - unmasked).max()) > 1e-6          # the mask does change the reference


def test_mask_validation(setup: dict[str, Any]) -> None:
    with pytest.raises(NotImplementedError):
        _solve(setup, frame_valid=_mask(), cfg=RetargetConfig(fast_path=False))
    with pytest.raises(ValueError):
        _solve(setup, frame_valid=torch.ones(WINDOW.size - 1, dtype=torch.bool))
    with pytest.raises(ValueError):
        _solve(setup, frame_valid=torch.zeros(WINDOW.size, dtype=torch.bool))
    with pytest.raises(ValueError):
        _solve(setup, frame_valid=torch.ones(WINDOW.size))                # not bool
