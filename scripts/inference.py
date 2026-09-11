#!/usr/bin/env python
"""Inference entrypoint script."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from image_lora_trainer.config import InferenceConfig
from image_lora_trainer.inference.pipeline import run_inference
from image_lora_trainer.logging_utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LoRA image inference")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--lora-path", default=None)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--output", default="generated.png")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument("--num-images", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16")
    return parser.parse_args()


def main() -> int:
    setup_logging()
    args = parse_args()
    cfg = InferenceConfig(
        base_model=args.base_model,
        lora_path=args.lora_path,
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
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
