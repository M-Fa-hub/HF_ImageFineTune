"""Learning-rate scheduler factory."""

from __future__ import annotations

from image_lora_trainer.config import SchedulerConfig, SchedulerName
from image_lora_trainer.logging_utils import get_logger

logger = get_logger(__name__)


def build_lr_scheduler(optimizer, config: SchedulerConfig, num_training_steps: int):
    from diffusers.optimization import get_scheduler

    name = config.name.value
    logger.info(
        "Using LR scheduler '%s' (warmup=%d, steps=%d)",
        name,
        config.warmup_steps,
        num_training_steps,
    )
    return get_scheduler(
        name,
        optimizer=optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=config.num_cycles,
        power=config.power,
    )


SUPPORTED_SCHEDULERS = {s.value for s in SchedulerName}
