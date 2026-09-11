"""LoRA adapter application for denoisers and text encoders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from image_lora_trainer.config import LoRAConfig, LoRAModuleConfig
from image_lora_trainer.logging_utils import get_logger
from image_lora_trainer.models.capabilities import (
    ModelCapabilities,
    default_target_modules,
    discover_linear_module_suffixes,
)
from image_lora_trainer.utils.memory import count_parameters, format_param_counts

logger = get_logger(__name__)


@dataclass
class AdapterReport:
    component: str
    enabled: bool
    target_modules: list[str]
    trainable_parameters: int
    total_parameters: int

    @property
    def summary(self) -> str:
        if not self.enabled:
            return f"{self.component}: LoRA disabled"
        return (
            f"{self.component}: targets={self.target_modules}, "
            f"trainable={format_param_counts(self.trainable_parameters, self.total_parameters)}"
        )


def _peft_config(module_cfg: LoRAModuleConfig, targets: list[str]):
    from peft import LoraConfig

    return LoraConfig(
        r=module_cfg.rank,
        lora_alpha=module_cfg.alpha,
        lora_dropout=module_cfg.dropout,
        bias=module_cfg.bias,
        target_modules=targets,
        init_lora_weights="gaussian",
    )


def _resolve_targets(
    module,
    module_cfg: LoRAModuleConfig,
    caps: ModelCapabilities,
    component: str,
) -> list[str]:
    if module_cfg.target_modules:
        return list(module_cfg.target_modules)
    defaults = default_target_modules(
        caps.architecture,
        component="denoiser" if component == "denoiser" else "text_encoder",
    )
    discovered = set(discover_linear_module_suffixes(module))
    matched = [t for t in defaults if t in discovered or any(t in d for d in discovered)]
    if matched:
        return matched
    # Fall back to intersection with attention-ish names.
    attention_like = [
        n
        for n in discovered
        if any(k in n for k in ("q", "k", "v", "out", "proj", "to_"))
    ]
    if not attention_like:
        raise ValueError(
            f"Could not resolve LoRA target modules for {component}. "
            f"Discovered Linear suffixes: {sorted(discovered)[:30]}"
        )
    logger.warning(
        "Default LoRA targets %s not found on %s; using discovered attention-like modules: %s",
        defaults,
        component,
        attention_like[:12],
    )
    return attention_like[:12]


def inject_lora(
    module,
    module_cfg: LoRAModuleConfig,
    caps: ModelCapabilities,
    component: str,
):
    from peft import get_peft_model

    if not module_cfg.enabled:
        for param in module.parameters():
            param.requires_grad = False
        trainable, total = count_parameters(module)
        return module, AdapterReport(component, False, [], trainable, total)

    targets = _resolve_targets(module, module_cfg, caps, component)
    logger.info("Applying LoRA on %s with targets=%s rank=%d", component, targets, module_cfg.rank)
    peft_cfg = _peft_config(module_cfg, targets)
    peft_model = get_peft_model(module, peft_cfg)
    trainable, total = count_parameters(peft_model)
    logger.info("%s LoRA params: %s", component, format_param_counts(trainable, total))
    return peft_model, AdapterReport(component, True, targets, trainable, total)


def apply_lora_to_components(
    *,
    denoiser,
    text_encoder,
    text_encoder_2,
    lora_cfg: LoRAConfig,
    caps: ModelCapabilities,
) -> tuple[Any, Any, Any, list[AdapterReport]]:
    reports: list[AdapterReport] = []

    denoiser, report = inject_lora(
        denoiser, lora_cfg.resolved_denoiser(), caps, "denoiser"
    )
    reports.append(report)

    text_encoder, report = inject_lora(
        text_encoder, lora_cfg.resolved_text_encoder(), caps, "text_encoder"
    )
    reports.append(report)

    if text_encoder_2 is not None:
        text_encoder_2, report = inject_lora(
            text_encoder_2, lora_cfg.resolved_text_encoder_2(), caps, "text_encoder_2"
        )
        reports.append(report)

    return denoiser, text_encoder, text_encoder_2, reports


def save_lora_weights(denoiser, text_encoder, text_encoder_2, output_dir: str) -> None:
    """Save PEFT adapters for all trained components."""
    from pathlib import Path

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    def _save(model, subdir: str) -> None:
        if model is None:
            return
        if not hasattr(model, "save_pretrained"):
            return
        # Only save if there are trainable / peft params.
        trainable, _ = count_parameters(model)
        if trainable == 0 and not hasattr(model, "peft_config"):
            return
        target = out / subdir
        target.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(target), safe_serialization=True)
        logger.info("Saved LoRA weights to %s", target)

    _save(denoiser, "denoiser")
    _save(text_encoder, "text_encoder")
    _save(text_encoder_2, "text_encoder_2")
