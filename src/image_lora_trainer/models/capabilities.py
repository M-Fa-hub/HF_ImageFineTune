"""Architecture capability detection for Diffusers text-to-image models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from image_lora_trainer.config import (
    ArchitectureFamily,
    QuantizationConfig,
    TrainConfig,
)
from image_lora_trainer.logging_utils import get_logger

logger = get_logger(__name__)


# Sensible LoRA target defaults by family. Explicit config overrides win.
DEFAULT_DENOISER_TARGETS: dict[ArchitectureFamily, list[str]] = {
    ArchitectureFamily.SD15: ["to_q", "to_k", "to_v", "to_out.0"],
    ArchitectureFamily.SD21: ["to_q", "to_k", "to_v", "to_out.0"],
    ArchitectureFamily.SDXL: ["to_q", "to_k", "to_v", "to_out.0"],
    ArchitectureFamily.FLUX: [
        "to_q",
        "to_k",
        "to_v",
        "to_out.0",
        "add_q_proj",
        "add_k_proj",
        "add_v_proj",
    ],
    ArchitectureFamily.UNKNOWN: ["to_q", "to_k", "to_v", "to_out.0"],
}

DEFAULT_TEXT_ENCODER_TARGETS = ["q_proj", "k_proj", "v_proj", "out_proj"]


@dataclass
class ModelCapabilities:
    architecture: ArchitectureFamily
    supports_lora: bool = True
    supports_text_encoder_lora: bool = True
    supports_text_encoder_2_lora: bool = False
    supports_4bit: bool = False
    supports_8bit: bool = False
    supports_gradient_checkpointing: bool = True
    supports_xformers: bool = True
    supports_sdxl_time_ids: bool = False
    supports_flow_matching: bool = False
    preferred_precision: str = "bfloat16"
    denoiser_attr: str = "unet"
    has_text_encoder_2: bool = False
    supported_target_modules: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["architecture"] = self.architecture.value
        return data


class UnsupportedTrainingConfigError(ValueError):
    """Raised when a requested train/quantization combo is invalid."""


def detect_architecture_from_id(
    model_id: str,
    explicit: ArchitectureFamily | str = "auto",
) -> ArchitectureFamily:
    if explicit not in {"auto", ArchitectureFamily.UNKNOWN, "unknown"} and explicit != "auto":
        if isinstance(explicit, ArchitectureFamily):
            return explicit
        return ArchitectureFamily(explicit)

    lowered = model_id.lower()
    if "flux" in lowered:
        return ArchitectureFamily.FLUX
    if "xl" in lowered or "sdxl" in lowered:
        return ArchitectureFamily.SDXL
    if "stable-diffusion-2" in lowered or "sd-2" in lowered:
        return ArchitectureFamily.SD21
    if "stable-diffusion" in lowered or "sd-v1" in lowered or "sd15" in lowered:
        return ArchitectureFamily.SD15
    return ArchitectureFamily.UNKNOWN


def capabilities_for_architecture(
    architecture: ArchitectureFamily,
    *,
    bitsandbytes_available: bool = False,
) -> ModelCapabilities:
    """Return honest capability flags for an architecture family.

    Notes on quantization:
    - True LLM-style QLoRA is not universally valid for diffusion denoisers.
    - For UNet SD/SDXL we support *experimental* 4/8-bit base linear quantization
      via bitsandbytes when PEFT LoRA adapters remain in higher precision.
    - VAE is never quantized for training correctness.
    - Text-encoder quantization during LoRA training is disabled by default and
      only allowed when explicitly requested and bitsandbytes is present.
    - FLUX transformers may support 4-bit base weights + LoRA, but licenses and
      checkpoint variants differ; we still require bitsandbytes and clear logging.
    """
    caps = ModelCapabilities(
        architecture=architecture,
        supported_target_modules=list(
            DEFAULT_DENOISER_TARGETS.get(architecture, DEFAULT_DENOISER_TARGETS[ArchitectureFamily.UNKNOWN])
        ),
    )

    if architecture in {ArchitectureFamily.SD15, ArchitectureFamily.SD21}:
        caps.denoiser_attr = "unet"
        caps.supports_text_encoder_2_lora = False
        caps.has_text_encoder_2 = False
        caps.supports_4bit = bitsandbytes_available
        caps.supports_8bit = bitsandbytes_available
        caps.preferred_precision = "float16"
        caps.notes.append(
            "SD1.x/2.x: 4-bit QLoRA-style training quantizes UNet Linear layers only; "
            "VAE stays full precision; text-encoder quantization is opt-in and limited."
        )
    elif architecture == ArchitectureFamily.SDXL:
        caps.denoiser_attr = "unet"
        caps.supports_text_encoder_2_lora = True
        caps.has_text_encoder_2 = True
        caps.supports_sdxl_time_ids = True
        caps.supports_4bit = bitsandbytes_available
        caps.supports_8bit = bitsandbytes_available
        caps.preferred_precision = "bfloat16"
        caps.notes.append(
            "SDXL: QLoRA-style mode quantizes UNet base weights when bitsandbytes is "
            "available. This is PEFT-on-quantized-UNet, not identical to LLM QLoRA."
        )
    elif architecture == ArchitectureFamily.FLUX:
        caps.denoiser_attr = "transformer"
        caps.supports_text_encoder_2_lora = True
        caps.has_text_encoder_2 = True
        caps.supports_flow_matching = True
        caps.supports_xformers = False  # frequently incompatible / unnecessary
        caps.supports_4bit = bitsandbytes_available
        caps.supports_8bit = bitsandbytes_available
        caps.preferred_precision = "bfloat16"
        caps.notes.append(
            "FLUX: transformer-based; 4-bit base + LoRA may work with bitsandbytes, "
            "but verify model license and checkpoint compatibility before training."
        )
    else:
        caps.supports_4bit = False
        caps.supports_8bit = bitsandbytes_available
        caps.notes.append(
            "Unknown architecture: LoRA may work if Diffusers UNet/Transformer "
            "attention modules match defaults; 4-bit QLoRA is disabled until "
            "architecture is explicitly set."
        )
    return caps


def discover_linear_module_suffixes(module, limit: int = 64) -> list[str]:
    """Heuristic discovery of Linear module name suffixes suitable for LoRA."""
    import torch.nn as nn

    names: set[str] = set()
    for full_name, child in module.named_modules():
        if isinstance(child, nn.Linear):
            suffix = full_name.split(".")[-1]
            names.add(suffix)
            # Also keep common compound suffixes like to_out.0
            parts = full_name.split(".")
            if len(parts) >= 2 and parts[-1].isdigit():
                names.add(".".join(parts[-2:]))
    ordered = sorted(names)
    return ordered[:limit]


def estimate_lora_trainable_params(
    base_params: int,
    rank: int,
    target_module_count: int = 4,
) -> int:
    """Rough estimate for inspect-model without loading weights.

    Assumes average hidden size ~1024 and `target_module_count` Linear projections
    per attention block, with ~16 blocks. This is indicative only.
    """
    hidden = 1024
    blocks = 16
    # Each LoRA on a Linear(in,out) adds rank*(in+out) params; approximate in=out=hidden.
    per_linear = rank * (hidden + hidden)
    return per_linear * target_module_count * blocks


def validate_config_against_capabilities(
    config: TrainConfig,
    caps: ModelCapabilities,
    *,
    bitsandbytes_available: bool,
) -> None:
    errors: list[str] = []

    if config.lora.enabled and not caps.supports_lora:
        errors.append(f"Architecture {caps.architecture.value} does not support LoRA.")

    te = config.lora.resolved_text_encoder()
    te2 = config.lora.resolved_text_encoder_2()
    if te.enabled and not caps.supports_text_encoder_lora:
        errors.append("Text-encoder LoRA is not supported for this architecture.")
    if te2.enabled and not caps.supports_text_encoder_2_lora:
        errors.append("Text-encoder-2 LoRA is not supported for this architecture.")

    quant: QuantizationConfig = config.quantization
    if quant.enabled:
        if not bitsandbytes_available:
            msg = (
                "Quantization requested but bitsandbytes is not installed/available. "
                "Install bitsandbytes or set quantization.enabled=false."
            )
            if quant.allow_quantization_fallback:
                logger.warning("%s Fallback explicitly allowed; continuing without quantization.", msg)
                config.quantization.enabled = False
            else:
                errors.append(msg + " Set allow_quantization_fallback=true to continue without quantization.")
        else:
            if quant.bits == 4 and not caps.supports_4bit:
                msg = (
                    f"4-bit quantization is not supported for architecture "
                    f"{caps.architecture.value}."
                )
                if quant.allow_quantization_fallback:
                    logger.warning("%s Falling back because allow_quantization_fallback=true.", msg)
                    config.quantization.enabled = False
                else:
                    errors.append(msg)
            if quant.bits == 8 and not caps.supports_8bit:
                msg = (
                    f"8-bit quantization is not supported for architecture "
                    f"{caps.architecture.value}."
                )
                if quant.allow_quantization_fallback:
                    logger.warning("%s Falling back because allow_quantization_fallback=true.", msg)
                    config.quantization.enabled = False
                else:
                    errors.append(msg)
            if quant.quantize_text_encoder and not caps.supports_text_encoder_lora:
                errors.append("quantize_text_encoder requested but text encoder LoRA unsupported.")

    if config.training.enable_xformers and not caps.supports_xformers:
        logger.warning(
            "xFormers requested but marked unsupported for %s; it will be skipped.",
            caps.architecture.value,
        )

    if errors:
        raise UnsupportedTrainingConfigError(
            "Unsupported training configuration:\n- " + "\n- ".join(errors)
        )


def default_target_modules(
    architecture: ArchitectureFamily,
    component: str = "denoiser",
) -> list[str]:
    if component == "denoiser":
        return list(
            DEFAULT_DENOISER_TARGETS.get(
                architecture, DEFAULT_DENOISER_TARGETS[ArchitectureFamily.UNKNOWN]
            )
        )
    return list(DEFAULT_TEXT_ENCODER_TARGETS)
