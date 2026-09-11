"""Checkpoint save / resume utilities."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

import torch

from image_lora_trainer.logging_utils import get_logger
from image_lora_trainer.models.adapters import save_lora_weights
from image_lora_trainer.utils.seed import capture_rng_states, restore_rng_states

logger = get_logger(__name__)

CHECKPOINT_DIR_RE = re.compile(r"^checkpoint-(\d+)$")


def checkpoint_step(path: Path) -> int | None:
    match = CHECKPOINT_DIR_RE.match(path.name)
    return int(match.group(1)) if match else None


def list_checkpoints(output_dir: str | Path) -> list[Path]:
    root = Path(output_dir) / "checkpoints"
    if not root.is_dir():
        return []
    checkpoints = []
    for path in root.iterdir():
        if (
            path.is_dir()
            and checkpoint_step(path) is not None
            and ((path / "trainer_state.json").is_file() or any(path.iterdir()))
        ):
            checkpoints.append(path)
    return sorted(checkpoints, key=lambda p: checkpoint_step(p) or -1)


def find_latest_checkpoint(output_dir: str | Path) -> Path | None:
    checkpoints = list_checkpoints(output_dir)
    return checkpoints[-1] if checkpoints else None


def resolve_resume_path(output_dir: str | Path, resume: str | None) -> Path | None:
    if resume is None:
        return None
    if resume == "latest":
        latest = find_latest_checkpoint(output_dir)
        if latest is None:
            logger.warning("resume_from_checkpoint=latest but no checkpoints found.")
        return latest
    path = Path(resume)
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def save_checkpoint(
    *,
    output_dir: str | Path,
    global_step: int,
    denoiser,
    text_encoder,
    text_encoder_2,
    optimizer,
    lr_scheduler,
    config_dict: dict[str, Any],
    accelerator=None,
    epoch: int = 0,
    checkpoints_total_limit: int | None = 3,
) -> Path:
    ckpt_root = Path(output_dir) / "checkpoints"
    ckpt_dir = ckpt_root / f"checkpoint-{global_step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Adapter weights
    save_lora_weights(denoiser, text_encoder, text_encoder_2, str(ckpt_dir / "lora"))

    state = {
        "global_step": global_step,
        "epoch": epoch,
        "config": config_dict,
    }
    with (ckpt_dir / "trainer_state.json").open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2)

    # Optimizer / scheduler / RNG — best effort; Accelerate unwrap may vary.
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "rng": capture_rng_states(),
        },
        ckpt_dir / "training_state.pt",
    )

    if accelerator is not None:
        try:
            accelerator.save_state(str(ckpt_dir / "accelerator"))
        except Exception as exc:
            logger.warning("accelerator.save_state failed: %s", exc)

    _enforce_checkpoint_limit(ckpt_root, checkpoints_total_limit)
    logger.info("Saved checkpoint to %s", ckpt_dir)
    return ckpt_dir


def load_checkpoint(
    checkpoint_dir: str | Path,
    *,
    denoiser,
    text_encoder,
    text_encoder_2,
    optimizer=None,
    lr_scheduler=None,
    accelerator=None,
) -> dict[str, Any]:
    ckpt = Path(checkpoint_dir)
    state_path = ckpt / "trainer_state.json"
    if not state_path.is_file():
        raise FileNotFoundError(f"Missing trainer_state.json in {ckpt}")

    with state_path.open("r", encoding="utf-8") as handle:
        trainer_state = json.load(handle)

    lora_dir = ckpt / "lora"
    _load_peft_if_exists(denoiser, lora_dir / "denoiser")
    _load_peft_if_exists(text_encoder, lora_dir / "text_encoder")
    if text_encoder_2 is not None:
        _load_peft_if_exists(text_encoder_2, lora_dir / "text_encoder_2")

    training_state_path = ckpt / "training_state.pt"
    if training_state_path.is_file() and optimizer is not None:
        blob = torch.load(training_state_path, map_location="cpu")
        optimizer.load_state_dict(blob["optimizer"])
        if lr_scheduler is not None and "lr_scheduler" in blob:
            lr_scheduler.load_state_dict(blob["lr_scheduler"])
        if "rng" in blob:
            restore_rng_states(blob["rng"])

    accel_dir = ckpt / "accelerator"
    if accelerator is not None and accel_dir.is_dir():
        try:
            accelerator.load_state(str(accel_dir))
        except Exception as exc:
            logger.warning("accelerator.load_state failed: %s", exc)

    logger.info("Resumed from %s (step=%s)", ckpt, trainer_state.get("global_step"))
    result: dict[str, Any] = trainer_state
    return result


def _load_peft_if_exists(model, path: Path) -> None:
    if model is None or not path.is_dir():
        return
    if not hasattr(model, "load_adapter") and not hasattr(model, "load_state_dict"):
        return
    try:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        weight_file = path / "adapter_model.safetensors"
        if weight_file.is_file():
            state = load_file(str(weight_file))
            set_peft_model_state_dict(model, state)
            logger.info("Loaded adapter weights from %s", weight_file)
            return
    except Exception as exc:
        logger.warning("Failed loading adapter from %s: %s", path, exc)


def _enforce_checkpoint_limit(ckpt_root: Path, limit: int | None) -> None:
    if limit is None:
        return
    checkpoints = list_checkpoints(ckpt_root.parent)
    # list_checkpoints expects output_dir, so reimplement locally:
    checkpoints = []
    for path in ckpt_root.iterdir():
        if path.is_dir() and checkpoint_step(path) is not None:
            checkpoints.append(path)
    checkpoints = sorted(checkpoints, key=lambda p: checkpoint_step(p) or -1)
    while len(checkpoints) > limit:
        old = checkpoints.pop(0)
        shutil.rmtree(old, ignore_errors=True)
        logger.info("Removed old checkpoint %s", old)


def save_final_adapter(
    output_dir: str | Path,
    denoiser,
    text_encoder,
    text_encoder_2,
) -> Path:
    final_dir = Path(output_dir) / "final"
    save_lora_weights(denoiser, text_encoder, text_encoder_2, str(final_dir))
    return final_dir
