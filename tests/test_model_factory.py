"""Model capability and target-module discovery tests (no large downloads)."""

from __future__ import annotations

import torch.nn as nn

from image_lora_trainer.config import QuantizationConfig
from image_lora_trainer.models.capabilities import (
    ArchitectureFamily,
    capabilities_for_architecture,
    default_target_modules,
    discover_linear_module_suffixes,
    estimate_lora_trainable_params,
)
from image_lora_trainer.models.factory import inspect_model_card
from image_lora_trainer.models.quantization import plan_quantization


def test_sdxl_capabilities_with_bnb_flag():
    caps = capabilities_for_architecture(
        ArchitectureFamily.SDXL, bitsandbytes_available=True
    )
    assert caps.supports_lora
    assert caps.supports_4bit
    assert caps.has_text_encoder_2
    assert "to_q" in caps.supported_target_modules


def test_unknown_architecture_disables_4bit():
    caps = capabilities_for_architecture(
        ArchitectureFamily.UNKNOWN, bitsandbytes_available=True
    )
    assert caps.supports_4bit is False


def test_flux_uses_transformer_attr():
    caps = capabilities_for_architecture(
        ArchitectureFamily.FLUX, bitsandbytes_available=False
    )
    assert caps.denoiser_attr == "transformer"
    assert caps.supports_flow_matching
    assert caps.supports_4bit is False


def test_discover_linear_module_suffixes():
    class TinyAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = nn.ModuleDict(
                {
                    "to_q": nn.Linear(8, 8),
                    "to_k": nn.Linear(8, 8),
                    "to_v": nn.Linear(8, 8),
                }
            )
            self.out = nn.Linear(8, 8)

        def forward(self, x):
            return x

    suffixes = discover_linear_module_suffixes(TinyAttn())
    assert "to_q" in suffixes
    assert "to_k" in suffixes


def test_default_target_modules():
    assert default_target_modules(ArchitectureFamily.SD15)[0] == "to_q"
    assert "q_proj" in default_target_modules(ArchitectureFamily.SDXL, "text_encoder")


def test_estimate_lora_params_positive():
    assert estimate_lora_trainable_params(0, 16) > 0


def test_quantization_plan_keeps_vae_full_precision():
    plan = plan_quantization(
        QuantizationConfig(enabled=True, bits=4, quant_type="nf4"),
        denoiser_name="unet",
        has_text_encoder_2=True,
    )
    assert plan.enabled
    assert "unet" in plan.modules_to_quantize
    assert "vae" in plan.modules_kept_full_precision
    assert "text_encoder" in plan.modules_kept_full_precision


def test_inspect_model_card_no_download():
    card = inspect_model_card("stabilityai/stable-diffusion-xl-base-1.0")
    assert card["architecture"] == "sdxl"
    assert "capabilities" in card
    assert "default_target_modules" in card
