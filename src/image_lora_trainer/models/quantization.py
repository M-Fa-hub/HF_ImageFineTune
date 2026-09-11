"""Quantization helpers for diffusion PEFT (honest QLoRA-style support)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from image_lora_trainer.config import QuantizationConfig
from image_lora_trainer.logging_utils import get_logger
from image_lora_trainer.utils.device import dtype_from_name

logger = get_logger(__name__)


class QuantizationError(RuntimeError):
    """Raised for invalid or unavailable quantization setups."""


@dataclass
class QuantizationPlan:
    enabled: bool
    bits: int | None
    modules_to_quantize: list[str]
    modules_kept_full_precision: list[str]
    compute_dtype: str
    details: dict[str, Any]

    def log(self) -> None:
        logger.info("Quantization enabled: %s", self.enabled)
        if not self.enabled:
            return
        logger.info("Quantization bits: %s", self.bits)
        logger.info("Modules quantized: %s", ", ".join(self.modules_to_quantize) or "(none)")
        logger.info(
            "Modules kept full precision: %s",
            ", ".join(self.modules_kept_full_precision) or "(none)",
        )
        logger.info("Compute dtype: %s", self.compute_dtype)


def build_bnb_config(quant: QuantizationConfig):
    """Create a bitsandbytes BitsAndBytesConfig or raise clearly."""
    try:
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        raise QuantizationError(
            "transformers.BitsAndBytesConfig unavailable; upgrade transformers."
        ) from exc

    try:
        import bitsandbytes as bnb  # noqa: F401
    except ImportError as exc:
        raise QuantizationError(
            "bitsandbytes is required for quantization. "
            "Install with: pip install 'image-lora-trainer[qlora]'"
        ) from exc

    compute_dtype = dtype_from_name(quant.compute_dtype)

    if quant.bits == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant.quant_type,
            bnb_4bit_use_double_quant=quant.double_quant,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    if quant.bits == 8:
        return BitsAndBytesConfig(load_in_8bit=True)
    raise QuantizationError(f"Unsupported quantization bits: {quant.bits}")


def plan_quantization(
    quant: QuantizationConfig,
    *,
    denoiser_name: str,
    has_text_encoder_2: bool,
) -> QuantizationPlan:
    if not quant.enabled:
        return QuantizationPlan(
            enabled=False,
            bits=None,
            modules_to_quantize=[],
            modules_kept_full_precision=["vae", "text_encoder", "text_encoder_2", denoiser_name],
            compute_dtype=quant.compute_dtype,
            details={},
        )

    quantize: list[str] = []
    keep: list[str] = ["vae"]  # Never quantize VAE for latent encode/decode correctness.

    if quant.quantize_denoiser:
        quantize.append(denoiser_name)
    else:
        keep.append(denoiser_name)

    if quant.quantize_text_encoder:
        quantize.append("text_encoder")
        if has_text_encoder_2:
            quantize.append("text_encoder_2")
        logger.warning(
            "Text-encoder quantization is experimental for diffusion LoRA training. "
            "LoRA adapters remain in floating point; monitor loss stability."
        )
    else:
        keep.append("text_encoder")
        if has_text_encoder_2:
            keep.append("text_encoder_2")

    return QuantizationPlan(
        enabled=True,
        bits=quant.bits,
        modules_to_quantize=quantize,
        modules_kept_full_precision=keep,
        compute_dtype=quant.compute_dtype,
        details={
            "quant_type": quant.quant_type,
            "double_quant": quant.double_quant,
            "note": (
                "This is QLoRA-style PEFT: quantized frozen base Linear weights + "
                "trainable LoRA adapters in higher precision. It is not identical to "
                "LLM QLoRA recipes."
            ),
        },
    )


def cast_trainable_lora_to_dtype(model, dtype) -> None:
    """Ensure LoRA adapter parameters remain in a trainable floating dtype."""
    for name, param in model.named_parameters():
        if param.requires_grad and param.dtype in {dtype}:
            continue
        if "lora_" in name and param.requires_grad:
            param.data = param.data.to(dtype)
