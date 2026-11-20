"""Frozen DINOv2-large image encoder: lazy Hugging Face load and batched bf16 crop encoding.

`load_dinov2` loads the open `facebook/dinov2-large` checkpoint (ungated) through `transformers`, imported
lazily so `fastdo.encoder` imports with only the base install. `transformers` is not a dependency of this
project yet: the encoder stage needs it installed (`uv add transformers`) before its first real run, and
tests inject a fake model instead (no weight downloads).

`encode_crops` returns one `(2 * hidden,)` descriptor per crop: the CLS token concatenated with the mean of
the patch tokens of the last hidden layer (1024 + 1024 = 2048 for DINOv2-large), computed under bf16 autocast
and returned as float32 on the CPU. Crops are RGB in [0,1] (uint8 crops are divided by 255) and are
normalized with the ImageNet mean and std DINOv2 was trained with.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import Tensor

__all__ = ["IMAGENET_MEAN", "IMAGENET_STD", "MODEL_ID", "encode_crops", "load_dinov2"]

logger = logging.getLogger(__name__)

MODEL_ID = "facebook/dinov2-large"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_dinov2(device: str = "cuda") -> Any:
    """Loads DINOv2-large in eval mode on `device` -- untyped, `transformers` models carry no stubs here.

    Raises:
        RuntimeError: If `transformers` is not importable, with an actionable install hint.
    """
    try:
        from transformers import AutoModel
    except ImportError as exc:
        raise RuntimeError(
            "transformers is not installed, and the DINOv2 encoder loads through it: run `uv add "
            "transformers`. Weights download from (or are read from the local Hugging Face cache for) "
            f"https://huggingface.co/{MODEL_ID}."
        ) from exc
    logger.info("loading DINOv2 checkpoint %s on %s", MODEL_ID, device)
    return AutoModel.from_pretrained(MODEL_ID).to(device).eval()


def encode_crops(model: Any, crops: Tensor, batch_size: int = 32) -> Tensor:
    """Pooled DINOv2 descriptors for a stack of square crops.

    Args:
        model: A `transformers` `Dinov2Model`-compatible callable: `model(pixel_values=(B,3,S,S))` returns an
            object with `last_hidden_state` `(B, 1+P, hidden)`, CLS first. Inputs go to `model.device` when
            the model has one, else the CPU.
        crops: `(N,3,S,S)` RGB crops, float in [0,1] or uint8.
        batch_size: Crops per forward pass.

    Returns:
        `(N, 2*hidden)` float32 on the CPU: `concat(CLS, mean(patch tokens))` from a bf16 autocast forward.

    Raises:
        ValueError: If `crops` is not `(N,3,S,S)` or `batch_size < 1`.
    """
    if crops.ndim != 4 or crops.shape[1] != 3 or crops.shape[2] != crops.shape[3]:
        raise ValueError(f"crops must be (N,3,S,S), got {tuple(crops.shape)}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    device = torch.device(getattr(model, "device", "cpu"))
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
    scale = 255.0 if crops.dtype == torch.uint8 else 1.0
    pooled: list[Tensor] = []
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        for start in range(0, int(crops.shape[0]), batch_size):
            chunk = (crops[start:start + batch_size].float() / scale - mean) / std
            hidden = model(pixel_values=chunk.to(device)).last_hidden_state
            tokens = torch.cat([hidden[:, 0], hidden[:, 1:].mean(dim=1)], dim=-1)
            pooled.append(tokens.float().cpu())
    return torch.cat(pooled) if pooled else torch.empty(0, 0, dtype=torch.float32)
