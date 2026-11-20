"""Toy clips for the transductive env tests: the single-hand scene of `tests/physics/fd_toy.py` as `ClipSpec`s."""

from __future__ import annotations

import numpy as np

from fastdo.transductive.clip import ClipSpec
from tests.physics.fd_toy import toy_problem


def toy_clip(task: str = "toy_a", T: int = 80, W: int = 20, ref_shift: float = 0.0, ctrl_steps: int = 20,
             press: bool = False) -> ClipSpec:
    """A clip whose warm start reproduces its reference exactly when `ref_shift` = 0 (zero residual = success)."""
    ex, rw, u0 = toy_problem(T=T, W=W, ref_shift=ref_shift, press=press)
    ctrl_ref = np.concatenate([u0, u0[-1:]], axis=0)
    return ClipSpec(task=task, side="right", corpus="toy", ex=ex, rw=rw, ctrl_ref=ctrl_ref, max_sim_steps=T,
                    ctrl_steps=ctrl_steps, ref_exec_ctrl=u0.copy())
