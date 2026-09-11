"""Configuration validation tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from image_lora_trainer.config import (
    ArchitectureFamily,
    QuantizationConfig,
    TrainConfig,
    load_train_config,
    parse_cli_overrides,
)
from image_lora_trainer.models.capabilities import (
    UnsupportedTrainingConfigError,
    capabilities_for_architecture,
    detect_architecture_from_id,
    validate_config_against_capabilities,
)

FIXTURES = Path(__file__).resolve().parents[1] / "configs"


def _minimal_dict(**overrides):
    base = {
        "model": {
            "pretrained_model_name_or_path": "stabilityai/stable-diffusion-xl-base-1.0",
            "architecture": "sdxl",
        },
        "dataset": {"dataset_path": "./examples/dataset"},
        "lora": {"enabled": True, "rank": 16},
        "training": {"max_train_steps": 10, "output_dir": "./outputs/test"},
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            base[key].update(value)
        else:
            base[key] = value
    return base


def test_load_example_configs():
    for name in ("lora_sdxl.yaml", "qlora_sdxl.yaml", "example.yaml"):
        cfg = load_train_config(FIXTURES / name)
        assert cfg.model.pretrained_model_name_or_path
        assert cfg.lora.enabled


def test_invalid_lora_rank():
    with pytest.raises(ValidationError):
        TrainConfig.model_validate(_minimal_dict(lora={"rank": 0}))


def test_quantization_requires_lora():
    with pytest.raises(ValidationError):
        TrainConfig.model_validate(
            _minimal_dict(
                lora={"enabled": False},
                quantization={"enabled": True, "bits": 4, "quant_type": "nf4"},
            )
        )


def test_invalid_4bit_quant_type():
    with pytest.raises(ValidationError):
        QuantizationConfig(enabled=True, bits=4, quant_type="int8")


def test_cache_text_embeddings_with_te_lora_invalid():
    with pytest.raises(ValidationError):
        TrainConfig.model_validate(
            _minimal_dict(
                lora={"train_text_encoder": True},
                dataset={
                    "dataset_path": "./examples/dataset",
                    "cache_text_embeddings": True,
                },
            )
        )


def test_cli_overrides_parsing():
    overrides = parse_cli_overrides(
        ["training.max_train_steps=123", "quantization.enabled=true", "lora.rank=32"]
    )
    assert overrides["training"]["max_train_steps"] == 123
    assert overrides["quantization"]["enabled"] is True
    assert overrides["lora"]["rank"] == 32


def test_architecture_detection():
    assert detect_architecture_from_id("stabilityai/stable-diffusion-xl-base-1.0") == ArchitectureFamily.SDXL
    assert detect_architecture_from_id("black-forest-labs/FLUX.1-dev") == ArchitectureFamily.FLUX
    assert detect_architecture_from_id("runwayml/stable-diffusion-v1-5") == ArchitectureFamily.SD15


def test_unsupported_4bit_without_fallback():
    cfg = TrainConfig.model_validate(
        _minimal_dict(
            model={
                "pretrained_model_name_or_path": "some/unknown-model",
                "architecture": "unknown",
            },
            quantization={
                "enabled": True,
                "bits": 4,
                "quant_type": "nf4",
                "allow_quantization_fallback": False,
            },
        )
    )
    caps = capabilities_for_architecture(
        ArchitectureFamily.UNKNOWN, bitsandbytes_available=True
    )
    with pytest.raises(UnsupportedTrainingConfigError):
        validate_config_against_capabilities(cfg, caps, bitsandbytes_available=True)


def test_quantization_fallback_explicit():
    cfg = TrainConfig.model_validate(
        _minimal_dict(
            model={
                "pretrained_model_name_or_path": "some/unknown-model",
                "architecture": "unknown",
            },
            quantization={
                "enabled": True,
                "bits": 4,
                "quant_type": "nf4",
                "allow_quantization_fallback": True,
            },
        )
    )
    caps = capabilities_for_architecture(
        ArchitectureFamily.UNKNOWN, bitsandbytes_available=True
    )
    validate_config_against_capabilities(cfg, caps, bitsandbytes_available=True)
    assert cfg.quantization.enabled is False
