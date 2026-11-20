"""Policy checkpoints: the actor-critic, its optimizer state, the run configuration and progress in one file.

`save_checkpoint(path, model, optimizer, meta)` writes `{"format", "dims", "model", "optimizer", "meta"}` with
`torch.save` through a temporary file and an atomic rename, so an interrupted run never leaves a truncated checkpoint;
`load_checkpoint(path, device)` rebuilds the `ActorCritic` from its dimensions and weights. `meta` must be plain data
(numbers, strings, bools, None, lists, tuples, dicts): the file loads with `weights_only=True`, so a checkpoint cannot
execute code. Training writes `update_XXXXXX.pt` (`checkpoint_name`) every `checkpoint_every` updates.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from fastdo.policy.networks import ActorCritic

logger = logging.getLogger(__name__)

__all__ = ["CHECKPOINT_FORMAT", "PolicyCheckpoint", "checkpoint_name", "load_checkpoint", "save_checkpoint"]

CHECKPOINT_FORMAT = "fastdo.policy.checkpoint/1"


@dataclass(frozen=True)
class PolicyCheckpoint:
    """A loaded checkpoint: the model (on the requested device), the optimizer state (None if not saved) and `meta`."""

    model: ActorCritic
    optimizer_state: dict[str, Any] | None
    meta: dict[str, Any]


def checkpoint_name(update: int) -> str:
    """File name of the checkpoint written after `update` PPO updates."""
    return f"update_{update:06d}.pt"


def _hidden(model: ActorCritic) -> tuple[int, ...]:
    widths = [layer.out_features for layer in model.actor if isinstance(layer, nn.Linear)]
    return tuple(widths[:-1])


def save_checkpoint(path: str | Path, model: ActorCritic, optimizer: torch.optim.Optimizer | None,
                    meta: dict[str, Any]) -> Path:
    """Write the checkpoint (module docstring) and return its path; parent directories are created."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format": CHECKPOINT_FORMAT,
               "dims": {"obs_dim": model.obs_dim, "priv_dim": model.priv_dim, "act_dim": model.act_dim,
                        "hidden": list(_hidden(model))},
               "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
               "optimizer": optimizer.state_dict() if optimizer is not None else None,
               "meta": meta}
    tmp = path.with_name(f".{path.name}.tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    logger.info("checkpoint %s (%s)", path, ", ".join(f"{k}={v}" for k, v in meta.items() if not isinstance(v, dict)))
    return path


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> PolicyCheckpoint:
    """Load a checkpoint written by `save_checkpoint`.

    Raises:
        ValueError: If the file is not a policy checkpoint of this format.
    """
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path} is not a {CHECKPOINT_FORMAT} checkpoint")
    dims = payload["dims"]
    model = ActorCritic(int(dims["obs_dim"]), int(dims["priv_dim"]), int(dims["act_dim"]),
                        hidden=tuple(int(h) for h in dims["hidden"]))
    model.load_state_dict(payload["model"])
    return PolicyCheckpoint(model=model.to(device), optimizer_state=payload["optimizer"], meta=dict(payload["meta"]))
