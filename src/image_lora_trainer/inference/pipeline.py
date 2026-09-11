"""Inference helpers for LoRA-adapted Diffusers pipelines."""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from image_lora_trainer.config import InferenceConfig
from image_lora_trainer.logging_utils import get_logger
from image_lora_trainer.models.capabilities import detect_architecture_from_id
from image_lora_trainer.models.factory import ModelComponents
from image_lora_trainer.utils.device import dtype_from_name, resolve_device

logger = get_logger(__name__)


def _load_pipeline(base_model: str, dtype: torch.dtype, device: str):
    from diffusers import (
        DiffusionPipeline,
        FluxPipeline,
        StableDiffusionPipeline,
        StableDiffusionXLPipeline,
    )

    arch = detect_architecture_from_id(base_model)
    common = {"torch_dtype": dtype}
    if arch.value == "flux":
        pipe = FluxPipeline.from_pretrained(base_model, **common)
    elif arch.value == "sdxl":
        pipe = StableDiffusionXLPipeline.from_pretrained(base_model, **common)
    elif arch.value in {"sd15", "sd21"}:
        pipe = StableDiffusionPipeline.from_pretrained(base_model, **common)
    else:
        pipe = DiffusionPipeline.from_pretrained(base_model, **common)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=False)
    return pipe, arch


def _attach_lora(pipe, lora_path: str | None, scale: float) -> None:
    if not lora_path:
        return
    path = Path(lora_path)
    # Support final/ or final/denoiser layouts.
    candidates = [path]
    if (path / "denoiser").is_dir():
        candidates = [path / "denoiser", path]
    if (path / "adapter_config.json").is_file():
        candidates = [path]

    loaded = False
    for candidate in candidates:
        try:
            pipe.load_lora_weights(str(candidate))
            if hasattr(pipe, "fuse_lora"):
                # Keep unfused by default; just set scale via cross_attention_kwargs where available.
                pass
            if hasattr(pipe, "set_adapters"):
                with contextlib.suppress(Exception):
                    pipe.set_adapters(pipe.get_active_adapters(), adapter_weights=[scale])
            loaded = True
            logger.info("Loaded LoRA weights from %s (scale=%s)", candidate, scale)
            break
        except Exception as exc:
            logger.debug("LoRA load attempt failed for %s: %s", candidate, exc)
    if not loaded:
        # Fallback: PEFT load onto unet/transformer directly.
        denoiser = getattr(pipe, "unet", None) or getattr(pipe, "transformer", None)
        peft_dir = path / "denoiser" if (path / "denoiser").is_dir() else path
        if denoiser is not None and peft_dir.is_dir():
            from peft import PeftModel

            wrapped = PeftModel.from_pretrained(denoiser, str(peft_dir))
            if hasattr(pipe, "unet"):
                pipe.unet = wrapped
            else:
                pipe.transformer = wrapped
            logger.info("Loaded LoRA via PeftModel from %s", peft_dir)
            loaded = True
    if not loaded:
        raise FileNotFoundError(f"Could not load LoRA weights from {lora_path}")


def run_inference(config: InferenceConfig) -> list[Path]:
    device = resolve_device(config.device)
    dtype = dtype_from_name(config.dtype)
    if device == "cpu" and config.dtype in {"bfloat16", "float16"}:
        dtype = torch.float32
        logger.warning("CPU inference falling back to float32.")

    pipe, _arch = _load_pipeline(config.base_model, dtype, device)
    _attach_lora(pipe, config.lora_path, config.lora_scale)

    generator = None
    if config.seed is not None:
        generator = torch.Generator(device=device).manual_seed(config.seed)

    output_path = Path(config.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    result = pipe(
        prompt=config.prompt,
        negative_prompt=config.negative_prompt,
        num_inference_steps=config.num_inference_steps,
        guidance_scale=config.guidance_scale,
        width=config.width,
        height=config.height,
        num_images_per_prompt=config.num_images,
        generator=generator,
    )
    images: list[Image.Image] = result.images
    if config.num_images == 1:
        images[0].save(output_path)
        saved.append(output_path)
    else:
        stem = output_path.stem
        suffix = output_path.suffix or ".png"
        for idx, image in enumerate(images):
            path = output_path.with_name(f"{stem}_{idx:02d}{suffix}")
            image.save(path)
            saved.append(path)
    logger.info("Wrote %d image(s)", len(saved))
    return saved


def generate_validation_images(
    *,
    components: ModelComponents,
    prompts: list[str],
    output_dir: str | Path,
    seed: int,
    num_images_per_prompt: int,
    num_inference_steps: int,
    guidance_scale: float,
    negative_prompt: str | None,
    height: int,
    width: int,
    dtype: torch.dtype,
    accelerator: Any | None = None,
) -> list[Image.Image]:
    """Build a temporary pipeline from live training components for validation."""
    from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = accelerator.device if accelerator is not None else resolve_device()
    unet = accelerator.unwrap_model(components.denoiser) if accelerator else components.denoiser
    text_encoder = (
        accelerator.unwrap_model(components.text_encoder)
        if accelerator
        else components.text_encoder
    )
    text_encoder_2 = None
    if components.text_encoder_2 is not None:
        text_encoder_2 = (
            accelerator.unwrap_model(components.text_encoder_2)
            if accelerator
            else components.text_encoder_2
        )

    was_training = unet.training
    unet.eval()

    images: list[Image.Image] = []
    try:
        pipe: Any
        if components.is_sdxl or components.capabilities.has_text_encoder_2:
            pipe = StableDiffusionXLPipeline(
                vae=components.vae,
                text_encoder=text_encoder,
                text_encoder_2=text_encoder_2,
                tokenizer=components.tokenizer,
                tokenizer_2=components.tokenizer_2,
                unet=unet,
                scheduler=components.scheduler,
            )
        elif components.is_flux:
            logger.warning(
                "FLUX validation generation is best-effort during training; "
                "prefer scripts/inference.py for full FLUX sampling."
            )
            return []
        else:
            pipe = StableDiffusionPipeline(
                vae=components.vae,
                text_encoder=text_encoder,
                tokenizer=components.tokenizer,
                unet=unet,
                scheduler=components.scheduler,
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=False,
            )
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=True)
        generator = torch.Generator(device=device).manual_seed(seed)

        for prompt_idx, prompt in enumerate(prompts):
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                num_images_per_prompt=num_images_per_prompt,
                generator=generator,
            )
            for img_idx, image in enumerate(result.images):
                path = out / f"{prompt_idx:02d}_{img_idx:02d}.png"
                image.save(path)
                images.append(image)
        logger.info("Wrote validation images to %s", out)
    except Exception as exc:
        logger.warning("Validation generation failed: %s", exc)
    finally:
        if was_training:
            unet.train()
    return images


def merge_lora_into_pipeline(
    base_model: str,
    lora_path: str,
    output_dir: str,
    dtype: str = "float16",
) -> Path:
    """Merge LoRA weights into a full pipeline when supported.

    Warnings:
    - Increases disk usage substantially vs adapter-only saves.
    - Quantized base models generally cannot be merged cleanly.
    - Precision may change after fuse/merge.
    """
    logger.warning(
        "Merging LoRA into the base model increases disk usage and may change precision. "
        "Quantized bases are typically unsupported for merge."
    )
    device = "cpu"
    torch_dtype = dtype_from_name(dtype)
    pipe, _ = _load_pipeline(base_model, torch_dtype, device)
    _attach_lora(pipe, lora_path, scale=1.0)
    if hasattr(pipe, "fuse_lora"):
        pipe.fuse_lora()
    if hasattr(pipe, "unload_lora_weights"):
        with contextlib.suppress(Exception):
            pipe.unload_lora_weights()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pipe.save_pretrained(str(out), safe_serialization=True)
    logger.info("Merged pipeline saved to %s", out)
    return out
