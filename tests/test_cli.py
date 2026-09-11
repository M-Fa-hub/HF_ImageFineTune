"""CLI parsing tests."""

from __future__ import annotations

import pytest

from image_lora_trainer.cli import build_parser


def test_train_parser_requires_config():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["train"])


def test_train_parser_with_overrides():
    parser = build_parser()
    args = parser.parse_args(
        [
            "train",
            "--config",
            "configs/lora_sdxl.yaml",
            "--set",
            "training.max_train_steps=10",
            "--set",
            "lora.rank=8",
        ]
    )
    assert args.command == "train"
    assert args.config.endswith("lora_sdxl.yaml")
    assert args.overrides == ["training.max_train_steps=10", "lora.rank=8"]


def test_inspect_parser():
    parser = build_parser()
    args = parser.parse_args(
        ["inspect-model", "--model", "stabilityai/stable-diffusion-xl-base-1.0"]
    )
    assert args.command == "inspect-model"
    assert "sdxl" in args.model or args.model.endswith("1.0")


def test_infer_parser():
    parser = build_parser()
    args = parser.parse_args(
        [
            "infer",
            "--model",
            "runwayml/stable-diffusion-v1-5",
            "--lora",
            "outputs/final",
            "--prompt",
            "a cat",
            "--output",
            "out.png",
        ]
    )
    assert args.prompt == "a cat"
    assert args.lora == "outputs/final"


def test_validate_dataset_parser():
    parser = build_parser()
    args = parser.parse_args(["validate-dataset", "--dataset", "./examples/dataset"])
    assert args.dataset == "./examples/dataset"
