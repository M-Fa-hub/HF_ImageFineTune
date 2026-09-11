"""Diffusion training losses for epsilon / v-prediction / flow matching."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from image_lora_trainer.config import ArchitectureFamily, PredictionType
from image_lora_trainer.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class LossOutput:
    loss: torch.Tensor
    prediction_type: str


def resolve_prediction_type(
    scheduler,
    architecture: ArchitectureFamily,
    configured: PredictionType,
) -> str:
    if configured != PredictionType.AUTO:
        return configured.value

    if architecture == ArchitectureFamily.FLUX:
        return PredictionType.FLOW_MATCHING.value

    pred = getattr(getattr(scheduler, "config", scheduler), "prediction_type", "epsilon")
    return str(pred)


def compute_diffusion_loss(
    *,
    model_pred: torch.Tensor,
    noise: torch.Tensor,
    latents: torch.Tensor,
    timesteps: torch.Tensor,
    scheduler,
    prediction_type: str,
) -> LossOutput:
    if prediction_type == PredictionType.EPSILON.value:
        target = noise
    elif prediction_type == PredictionType.V_PREDICTION.value:
        target = scheduler.get_velocity(latents, noise, timesteps)
    elif prediction_type == PredictionType.SAMPLE.value:
        target = latents
    elif prediction_type == PredictionType.FLOW_MATCHING.value:
        # Flow matching target is typically (noise - latents) for rectified flow.
        target = noise - latents
    else:
        raise ValueError(f"Unsupported prediction type: {prediction_type}")

    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
    return LossOutput(loss=loss, prediction_type=prediction_type)


def add_noise(
    scheduler,
    latents: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
    *,
    prediction_type: str,
) -> torch.Tensor:
    if prediction_type == PredictionType.FLOW_MATCHING.value:
        # Rectified-flow style interpolation used by FLUX-family schedulers.
        # sigmas may be provided by FlowMatchEulerDiscreteScheduler.
        if hasattr(scheduler, "sigmas"):
            # Map timesteps to sigma indices carefully.
            schedule_timesteps = scheduler.timesteps.to(timesteps.device)
            step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
            sigmas = scheduler.sigmas[step_indices].flatten()
            while len(sigmas.shape) < len(latents.shape):
                sigmas = sigmas.unsqueeze(-1)
            mixed: torch.Tensor = (1.0 - sigmas) * latents + sigmas * noise
            return mixed
        # Fallback: linear mix using normalized timestep.
        t = timesteps.float() / float(getattr(scheduler.config, "num_train_timesteps", 1000))
        while len(t.shape) < len(latents.shape):
            t = t.unsqueeze(-1)
        mixed_fb: torch.Tensor = (1.0 - t) * latents + t * noise
        return mixed_fb

    noised: torch.Tensor = scheduler.add_noise(latents, noise, timesteps)
    return noised


def encode_prompt_sd(
    *,
    captions: list[str],
    tokenizer,
    text_encoder,
    tokenizer_2=None,
    text_encoder_2=None,
    device,
    dtype,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Encode captions for SD / SDXL."""
    text_inputs = tokenizer(
        captions,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = text_inputs.input_ids.to(device)
    prompt_embeds = text_encoder(input_ids, return_dict=False)[0].to(dtype=dtype)

    pooled = None
    if tokenizer_2 is not None and text_encoder_2 is not None:
        text_inputs_2 = tokenizer_2(
            captions,
            padding="max_length",
            max_length=tokenizer_2.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids_2 = text_inputs_2.input_ids.to(device)
        outputs_2 = text_encoder_2(input_ids_2, return_dict=False)
        prompt_embeds_2 = outputs_2[0].to(dtype=dtype)
        pooled = outputs_2[1].to(dtype=dtype) if len(outputs_2) > 1 else None
        prompt_embeds = torch.cat([prompt_embeds, prompt_embeds_2], dim=-1)

    return prompt_embeds, pooled


def compute_time_ids(
    original_sizes: torch.Tensor,
    crop_top_lefts: torch.Tensor,
    target_sizes: torch.Tensor,
    dtype,
    device,
) -> torch.Tensor:
    """SDXL micro-conditioning add_time_ids."""
    add_time_ids = torch.cat(
        [
            original_sizes.to(device=device),
            crop_top_lefts.to(device=device),
            target_sizes.to(device=device),
        ],
        dim=-1,
    )
    return add_time_ids.to(dtype=dtype)


def prepare_latents(vae, pixel_values: torch.Tensor, weight_dtype) -> torch.Tensor:
    pixel_values = pixel_values.to(dtype=vae.dtype if hasattr(vae, "dtype") else weight_dtype)
    latents = vae.encode(pixel_values).latent_dist.sample()
    scaling = getattr(vae.config, "scaling_factor", 0.18215)
    scaled: torch.Tensor = latents * scaling
    return scaled
