"""Model loading factory for Diffusers text-to-image architectures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from image_lora_trainer.config import ArchitectureFamily, TrainConfig
from image_lora_trainer.logging_utils import get_logger
from image_lora_trainer.models.adapters import AdapterReport, apply_lora_to_components
from image_lora_trainer.models.capabilities import (
    ModelCapabilities,
    capabilities_for_architecture,
    detect_architecture_from_id,
    validate_config_against_capabilities,
)
from image_lora_trainer.models.quantization import (
    QuantizationPlan,
    build_bnb_config,
    cast_trainable_lora_to_dtype,
    plan_quantization,
)
from image_lora_trainer.utils.device import collect_device_info, dtype_from_name
from image_lora_trainer.utils.memory import count_parameters

logger = get_logger(__name__)


@dataclass
class ModelComponents:
    architecture: ArchitectureFamily
    capabilities: ModelCapabilities
    tokenizer: Any
    tokenizer_2: Any | None
    text_encoder: Any
    text_encoder_2: Any | None
    vae: Any
    denoiser: Any
    scheduler: Any
    noise_scheduler_copy: Any | None = None
    quantization_plan: QuantizationPlan | None = None
    adapter_reports: list[AdapterReport] = field(default_factory=list)
    pipeline_cls_name: str = ""
    pretrained_model_name_or_path: str = ""
    revision: str | None = None

    @property
    def is_sdxl(self) -> bool:
        return self.architecture == ArchitectureFamily.SDXL

    @property
    def is_flux(self) -> bool:
        return self.architecture == ArchitectureFamily.FLUX

    def trainable_parameter_counts(self) -> tuple[int, int]:
        modules = [self.denoiser, self.text_encoder]
        if self.text_encoder_2 is not None:
            modules.append(self.text_encoder_2)
        trainable = 0
        total = 0
        for module in modules:
            t, n = count_parameters(module)
            trainable += t
            total += n
        # Include frozen VAE in totals for reporting honesty.
        _, vae_total = count_parameters(self.vae)
        total += vae_total
        return trainable, total


def _weight_dtype(config: TrainConfig) -> torch.dtype:
    mp = config.training.mixed_precision.value
    if mp == "bf16":
        return torch.bfloat16
    if mp == "fp16":
        return torch.float16
    return torch.float32


def _load_sd_family(config: TrainConfig, caps: ModelCapabilities, weight_dtype: torch.dtype):
    from diffusers import (
        AutoencoderKL,
        DDPMScheduler,
        UNet2DConditionModel,
    )
    from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

    model_id = config.model.pretrained_model_name_or_path
    revision = config.model.revision
    variant = config.model.variant
    quant_plan = plan_quantization(
        config.quantization,
        denoiser_name="unet",
        has_text_encoder_2=caps.has_text_encoder_2,
    )
    quant_plan.log()

    bnb_config = None
    if quant_plan.enabled and "unet" in quant_plan.modules_to_quantize:
        bnb_config = build_bnb_config(config.quantization)

    common = {"revision": revision, "variant": variant}

    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer", **_drop_none(common))
    tokenizer_2 = None
    text_encoder_2 = None

    if caps.has_text_encoder_2:
        tokenizer_2 = CLIPTokenizer.from_pretrained(
            model_id, subfolder="tokenizer_2", **_drop_none(common)
        )

    te_kwargs = _drop_none({**common, "torch_dtype": weight_dtype})
    if quant_plan.enabled and "text_encoder" in quant_plan.modules_to_quantize:
        te_kwargs["quantization_config"] = build_bnb_config(config.quantization)

    text_encoder = CLIPTextModel.from_pretrained(
        model_id, subfolder="text_encoder", **te_kwargs
    )

    if caps.has_text_encoder_2:
        te2_kwargs = _drop_none({**common, "torch_dtype": weight_dtype})
        if quant_plan.enabled and "text_encoder_2" in quant_plan.modules_to_quantize:
            te2_kwargs["quantization_config"] = build_bnb_config(config.quantization)
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            model_id, subfolder="text_encoder_2", **te2_kwargs
        )

    vae = AutoencoderKL.from_pretrained(
        model_id,
        subfolder="vae",
        **_drop_none({**common, "torch_dtype": weight_dtype}),
    )
    vae.requires_grad_(False)

    unet_kwargs = _drop_none({**common, "torch_dtype": weight_dtype})
    if bnb_config is not None:
        unet_kwargs["quantization_config"] = bnb_config
        # When loading quantized, dtype is handled by bnb.
        unet_kwargs.pop("torch_dtype", None)

    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", **unet_kwargs)
    scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    return ModelComponents(
        architecture=caps.architecture,
        capabilities=caps,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        vae=vae,
        denoiser=unet,
        scheduler=scheduler,
        quantization_plan=quant_plan,
        pipeline_cls_name="StableDiffusionXLPipeline"
        if caps.has_text_encoder_2
        else "StableDiffusionPipeline",
        pretrained_model_name_or_path=model_id,
        revision=revision,
    )


def _load_flux(config: TrainConfig, caps: ModelCapabilities, weight_dtype: torch.dtype):
    from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxTransformer2DModel
    from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel
    from transformers.models.t5.tokenization_t5_fast import (  # type: ignore[attr-defined]
        T5TokenizerFast,
    )

    model_id = config.model.pretrained_model_name_or_path
    revision = config.model.revision
    variant = config.model.variant
    quant_plan = plan_quantization(
        config.quantization,
        denoiser_name="transformer",
        has_text_encoder_2=True,
    )
    quant_plan.log()

    common = _drop_none({"revision": revision, "variant": variant})

    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer", **common)
    tokenizer_2 = T5TokenizerFast.from_pretrained(model_id, subfolder="tokenizer_2", **common)
    text_encoder = CLIPTextModel.from_pretrained(
        model_id,
        subfolder="text_encoder",
        **_drop_none({**common, "torch_dtype": weight_dtype}),
    )
    text_encoder_2 = T5EncoderModel.from_pretrained(
        model_id,
        subfolder="text_encoder_2",
        **_drop_none({**common, "torch_dtype": weight_dtype}),
    )
    vae = AutoencoderKL.from_pretrained(
        model_id,
        subfolder="vae",
        **_drop_none({**common, "torch_dtype": weight_dtype}),
    )
    vae.requires_grad_(False)

    transformer_kwargs = _drop_none({**common, "torch_dtype": weight_dtype})
    if quant_plan.enabled and "transformer" in quant_plan.modules_to_quantize:
        transformer_kwargs["quantization_config"] = build_bnb_config(config.quantization)
        transformer_kwargs.pop("torch_dtype", None)

    transformer = FluxTransformer2DModel.from_pretrained(
        model_id, subfolder="transformer", **transformer_kwargs
    )
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        model_id, subfolder="scheduler"
    )

    return ModelComponents(
        architecture=ArchitectureFamily.FLUX,
        capabilities=caps,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        vae=vae,
        denoiser=transformer,
        scheduler=scheduler,
        quantization_plan=quant_plan,
        pipeline_cls_name="FluxPipeline",
        pretrained_model_name_or_path=model_id,
        revision=revision,
    )


def _drop_none(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}


def load_model(config: TrainConfig) -> ModelComponents:
    """Load model components, apply LoRA, and prepare for training."""
    device_info = collect_device_info()
    architecture = detect_architecture_from_id(
        config.model.pretrained_model_name_or_path,
        config.model.architecture,
    )
    caps = capabilities_for_architecture(
        architecture,
        bitsandbytes_available=device_info.bitsandbytes_available,
    )
    validate_config_against_capabilities(
        config,
        caps,
        bitsandbytes_available=device_info.bitsandbytes_available,
    )

    weight_dtype = _weight_dtype(config)
    logger.info(
        "Loading model %s as %s (dtype=%s)",
        config.model.pretrained_model_name_or_path,
        architecture.value,
        weight_dtype,
    )

    if architecture == ArchitectureFamily.FLUX:
        components = _load_flux(config, caps, weight_dtype)
    elif architecture in {
        ArchitectureFamily.SD15,
        ArchitectureFamily.SD21,
        ArchitectureFamily.SDXL,
        ArchitectureFamily.UNKNOWN,
    }:
        if architecture == ArchitectureFamily.UNKNOWN:
            logger.warning(
                "Unknown architecture; attempting SD/SDXL-style UNet load. "
                "Set model.architecture explicitly for FLUX or non-standard models."
            )
            # Prefer SDXL heuristics if tokenizer_2 exists is unknown; default SDXL flags off.
            caps = capabilities_for_architecture(
                ArchitectureFamily.SD15,
                bitsandbytes_available=device_info.bitsandbytes_available,
            )
            caps.architecture = ArchitectureFamily.UNKNOWN
        components = _load_sd_family(config, caps, weight_dtype)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    if config.training.gradient_checkpointing and caps.supports_gradient_checkpointing:
        if hasattr(components.denoiser, "enable_gradient_checkpointing"):
            components.denoiser.enable_gradient_checkpointing()
        if config.lora.resolved_text_encoder().enabled and hasattr(
            components.text_encoder, "gradient_checkpointing_enable"
        ):
            components.text_encoder.gradient_checkpointing_enable()
        if (
            components.text_encoder_2 is not None
            and config.lora.resolved_text_encoder_2().enabled
            and hasattr(components.text_encoder_2, "gradient_checkpointing_enable")
        ):
            components.text_encoder_2.gradient_checkpointing_enable()

    if config.training.enable_xformers and caps.supports_xformers:
        try:
            components.denoiser.enable_xformers_memory_efficient_attention()
            logger.info("Enabled xFormers memory-efficient attention.")
        except Exception as exc:
            logger.warning("Failed to enable xFormers: %s", exc)

    if config.training.enable_sliced_attention and hasattr(
        components.denoiser, "set_attention_slice"
    ):
        components.denoiser.set_attention_slice("auto")

    if config.training.enable_vae_slicing and hasattr(components.vae, "enable_slicing"):
        components.vae.enable_slicing()
    if config.training.enable_vae_tiling and hasattr(components.vae, "enable_tiling"):
        components.vae.enable_tiling()

    if config.lora.enabled:
        (
            components.denoiser,
            components.text_encoder,
            components.text_encoder_2,
            components.adapter_reports,
        ) = apply_lora_to_components(
            denoiser=components.denoiser,
            text_encoder=components.text_encoder,
            text_encoder_2=components.text_encoder_2,
            lora_cfg=config.lora,
            caps=caps,
        )
        cast_dtype = dtype_from_name(config.quantization.compute_dtype) if config.quantization.enabled else weight_dtype
        cast_trainable_lora_to_dtype(components.denoiser, cast_dtype)
        cast_trainable_lora_to_dtype(components.text_encoder, cast_dtype)
        if components.text_encoder_2 is not None:
            cast_trainable_lora_to_dtype(components.text_encoder_2, cast_dtype)

    # Freeze VAE always.
    components.vae.requires_grad_(False)
    components.vae.eval()

    trainable, total = components.trainable_parameter_counts()
    logger.info("Trainable parameters: %s / %s", f"{trainable:,}", f"{total:,}")
    for report in components.adapter_reports:
        logger.info(report.summary)

    loaded: ModelComponents = components
    return loaded


def inspect_model_card(model_id: str, architecture: str = "auto") -> dict[str, Any]:
    """Lightweight inspection without downloading full weights when possible."""
    device_info = collect_device_info()
    arch = detect_architecture_from_id(model_id, architecture)
    caps = capabilities_for_architecture(
        arch, bitsandbytes_available=device_info.bitsandbytes_available
    )
    from image_lora_trainer.models.capabilities import estimate_lora_trainable_params

    return {
        "model_id": model_id,
        "architecture": arch.value,
        "capabilities": caps.to_dict(),
        "device": device_info.to_dict(),
        "estimated_lora_trainable_params_rank16": estimate_lora_trainable_params(0, 16),
        "estimated_lora_trainable_params_rank32": estimate_lora_trainable_params(0, 32),
        "default_target_modules": caps.supported_target_modules,
        "notes": caps.notes,
    }
