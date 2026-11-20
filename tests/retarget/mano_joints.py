"""MANO rest-joint helpers for the wrist-target tests (no fixtures, no tests).

HaWoR's MANO layer poses joints through the kinematic tree. The MCP joints are children of the root, so they sit at
`R_global (J_k(beta) - J_0(beta)) + wrist`, where `J(beta) = J_regressor (v_template + shapedirs beta)` are the rest
joints and the wrist is `J_0(beta) + transl`. `tests/retarget/test_wrist_target.py` checks this against HaWoR's own
joints in the whisking `all_hand_meshes.npz`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from fastdo.retarget.labels import _MANOUnpickler

MCP_OPENPOSE = [5, 9, 13]      # index, middle, ring MCP in the 21-joint OpenPose order
MCP_MANO = [1, 4, 10]          # the same joints in MANO's 16-joint order
RestJointModel = tuple[np.ndarray, np.ndarray, np.ndarray]


def mano_rest_joint_model(pkl: Path) -> RestJointModel:
    """`(v_template (778,3), shapedirs (778,3,10), J_regressor (16,778))` float64 from the MANO pickle.

    The pickle is the project's own licensed MANO model, read through `fastdo.retarget.labels`' chumpy stub, where
    `shapedirs` is a chumpy selection `a.x.ravel()[idxs]`.
    """
    with Path(pkl).open("rb") as f:
        mano = _MANOUnpickler(f, encoding="latin1").load()
    stub = mano["shapedirs"]
    shapedirs = np.asarray(stub.a.x).ravel()[np.asarray(stub.idxs)].reshape(stub.preferred_shape)
    return (np.asarray(mano["v_template"], dtype=np.float64), shapedirs.astype(np.float64),
            np.asarray(mano["J_regressor"].toarray(), dtype=np.float64))


def rest_joints(model: RestJointModel, betas: np.ndarray) -> np.ndarray:
    """`(T,16,3)` rest joints for `(T,10)` shape coefficients."""
    v_template, shapedirs, regressor = model
    return np.einsum("jv,tvc->tjc", regressor, v_template[None] + np.einsum("vck,tk->tvc", shapedirs, betas))


def posed_mcp_joints(model: RestJointModel, rotvec: np.ndarray, betas: np.ndarray, wrist: np.ndarray) -> np.ndarray:
    """`(T,21,3)` with the wrist (joint 0) and the three MCP joints posed by global orient `rotvec (T,3)`; the other
    joints are zero."""
    rest = rest_joints(model, betas)
    joints = np.zeros((rotvec.shape[0], 21, 3))
    joints[:, 0] = wrist
    joints[:, MCP_OPENPOSE] = wrist[:, None] + np.einsum("tij,tkj->tki", Rotation.from_rotvec(rotvec).as_matrix(),
                                                         rest[:, MCP_MANO] - rest[:, [0]])
    return joints
