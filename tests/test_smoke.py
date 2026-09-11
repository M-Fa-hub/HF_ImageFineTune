"""Lightweight smoke tests that avoid downloading production models."""

from __future__ import annotations

from pathlib import Path

from image_lora_trainer.config import load_train_config
from image_lora_trainer.data.validation import validate_local_dataset
from image_lora_trainer.logging_utils import format_startup_summary
from image_lora_trainer.models.capabilities import (
    ArchitectureFamily,
    capabilities_for_architecture,
)
from image_lora_trainer.utils.device import collect_device_info
from image_lora_trainer.utils.seed import set_seed


def test_example_dataset_validates():
    root = Path(__file__).resolve().parents[1] / "examples" / "dataset"
    report = validate_local_dataset(root)
    assert report.ok
    assert report.num_images >= 3


def test_device_info_collects():
    info = collect_device_info()
    assert info.torch_version
    assert isinstance(info.cuda_available, bool)


def test_startup_summary_format():
    text = format_startup_summary({"Model": "x", "Architecture": "sdxl"})
    assert "TRAINING STARTUP SUMMARY" in text
    assert "Model" in text


def test_seed_runs():
    set_seed(123)


def test_config_method_name():
    cfg = load_train_config(
        Path(__file__).resolve().parents[1] / "configs" / "qlora_sdxl.yaml"
    )
    assert cfg.method_name() == "qlora"
    caps = capabilities_for_architecture(ArchitectureFamily.SDXL, bitsandbytes_available=True)
    assert caps.supports_lora
