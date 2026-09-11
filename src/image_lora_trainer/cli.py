"""Command-line interface for image-lora-trainer."""

from __future__ import annotations

import argparse
import json
import sys

from image_lora_trainer.config import InferenceConfig, load_train_config
from image_lora_trainer.data.validation import validate_local_dataset
from image_lora_trainer.logging_utils import get_logger, setup_logging
from image_lora_trainer.models.factory import inspect_model_card


def _add_override_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Dotted config override, e.g. training.max_train_steps=100",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="image-lora-trainer",
        description="LoRA / QLoRA fine-tuning for Hugging Face Diffusers image models",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Fine-tune a text-to-image model with LoRA/QLoRA")
    train.add_argument("--config", required=True, help="Path to YAML config")
    _add_override_arg(train)

    validate = sub.add_parser("validate-dataset", help="Validate a local image dataset")
    validate.add_argument("--dataset", required=True, help="Dataset directory")
    validate.add_argument("--default-caption", default=None)
    validate.add_argument("--caption-column", default="text")
    validate.add_argument("--file-name-column", default="file_name")

    infer = sub.add_parser("infer", help="Run inference with an optional LoRA adapter")
    infer.add_argument("--model", required=True, help="Base model id or path")
    infer.add_argument("--lora", default=None, help="Path to LoRA adapter directory")
    infer.add_argument("--prompt", required=True)
    infer.add_argument("--negative-prompt", default=None)
    infer.add_argument("--output", default="generated.png")
    infer.add_argument("--seed", type=int, default=42)
    infer.add_argument("--width", type=int, default=1024)
    infer.add_argument("--height", type=int, default=1024)
    infer.add_argument("--steps", type=int, default=30)
    infer.add_argument("--guidance-scale", type=float, default=7.5)
    infer.add_argument("--lora-scale", type=float, default=1.0)
    infer.add_argument("--num-images", type=int, default=1)
    infer.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])

    inspect_p = sub.add_parser("inspect-model", help="Inspect model architecture capabilities")
    inspect_p.add_argument("--model", required=True)
    inspect_p.add_argument(
        "--architecture",
        default="auto",
        choices=["auto", "sd15", "sd21", "sdxl", "flux", "unknown"],
    )

    merge = sub.add_parser("merge-lora", help="Merge LoRA weights into a full pipeline")
    merge.add_argument("--model", required=True)
    merge.add_argument("--lora", required=True)
    merge.add_argument("--output", required=True)
    merge.add_argument("--dtype", default="float16")

    return parser


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    logger = get_logger("image_lora_trainer.cli")
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        config = load_train_config(args.config, overrides=args.overrides)
        from image_lora_trainer.training.trainer import run_training

        summary = run_training(config)
        logger.info("Training complete: %s", summary)
        return 0

    if args.command == "validate-dataset":
        report = validate_local_dataset(
            args.dataset,
            caption_column=args.caption_column,
            file_name_column=args.file_name_column,
            default_caption=args.default_caption,
        )
        print(
            json.dumps(
                {
                    "root": report.root,
                    "num_images": report.num_images,
                    "num_captions": report.num_captions,
                    "ok": report.ok,
                    "issues": [{"path": i.path, "reason": i.reason} for i in report.issues],
                },
                indent=2,
            )
        )
        return 0 if report.ok else 1

    if args.command == "infer":
        from image_lora_trainer.inference.pipeline import run_inference

        cfg = InferenceConfig(
            base_model=args.model,
            lora_path=args.lora,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            output=args.output,
            seed=args.seed,
            width=args.width,
            height=args.height,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            lora_scale=args.lora_scale,
            num_images=args.num_images,
            dtype=args.dtype,
        )
        paths = run_inference(cfg)
        print("\n".join(str(p) for p in paths))
        return 0

    if args.command == "inspect-model":
        card = inspect_model_card(args.model, architecture=args.architecture)
        print(json.dumps(card, indent=2))
        return 0

    if args.command == "merge-lora":
        from image_lora_trainer.inference.pipeline import merge_lora_into_pipeline

        out = merge_lora_into_pipeline(args.model, args.lora, args.output, dtype=args.dtype)
        print(str(out))
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
