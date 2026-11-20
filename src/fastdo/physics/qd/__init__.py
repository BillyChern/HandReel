"""Quasi-dynamic convex contact model: one step as a convex QP (Anitescu relaxation), implicit gradients,
and open-loop rollouts. Proposals only: every plan is certified by the unchanged MuJoCo executor."""

from fastdo.physics.qd.grad import active_set, dv_db, dv_dcmd_vel, dv_dctrl
from fastdo.physics.qd.model import (
                                     ContactSet,
                                     QDConfig,
                                     QDModel,
                                     StepQP,
                                     StepResult,
                                     assemble,
                                     build_model,
                                     ctrl_from_command_velocity,
                                     edge_angles,
                                     query_contacts,
                                     step,
)
from fastdo.physics.qd.rollout import QDRollout, replay

__all__ = ["ContactSet", "QDConfig", "QDModel", "QDRollout", "StepQP", "StepResult", "active_set", "assemble",
           "build_model", "ctrl_from_command_velocity", "dv_db", "dv_dcmd_vel", "dv_dctrl", "edge_angles",
           "query_contacts", "replay", "step"]
