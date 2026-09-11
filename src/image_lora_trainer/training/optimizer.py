"""Optimizer factory."""

from __future__ import annotations

from collections.abc import Iterable

from image_lora_trainer.config import OptimizerConfig, OptimizerName
from image_lora_trainer.logging_utils import get_logger

logger = get_logger(__name__)


def iter_trainable_parameters(*modules) -> list:
    params = []
    for module in modules:
        if module is None:
            continue
        for param in module.parameters():
            if param.requires_grad:
                params.append(param)
    return params


def build_optimizer(config: OptimizerConfig, parameters: Iterable):
    params = list(parameters)
    if not params:
        raise ValueError("No trainable parameters found for optimizer.")

    if config.name == OptimizerName.ADAMW:
        import torch.optim as optim

        logger.info("Using AdamW optimizer (lr=%s)", config.learning_rate)
        return optim.AdamW(
            params,
            lr=config.learning_rate,
            betas=config.betas,
            weight_decay=config.weight_decay,
            eps=config.epsilon,
        )

    if config.name == OptimizerName.ADAMW_8BIT:
        try:
            import bitsandbytes as bnb
        except ImportError as exc:
            raise ImportError(
                "optimizer.name=adamw_8bit requires bitsandbytes. "
                "Install with: pip install 'image-lora-trainer[qlora]'"
            ) from exc
        logger.info("Using AdamW8bit optimizer (lr=%s)", config.learning_rate)
        return bnb.optim.AdamW8bit(
            params,
            lr=config.learning_rate,
            betas=config.betas,
            weight_decay=config.weight_decay,
            eps=config.epsilon,
        )

    raise ValueError(f"Unsupported optimizer: {config.name}")
