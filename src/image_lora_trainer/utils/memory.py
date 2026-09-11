"""Memory diagnostics and OOM guidance."""

from __future__ import annotations

from dataclasses import dataclass

from image_lora_trainer.logging_utils import get_logger

logger = get_logger(__name__)

OOM_SUGGESTIONS = [
    "Reduce train_batch_size (often to 1).",
    "Lower preprocess/training resolution.",
    "Enable training.gradient_checkpointing.",
    "Enable quantization (4-bit / 8-bit) if architecture supports it.",
    "Use optimizer.name=adamw_8bit when bitsandbytes is available.",
    "Enable dataset.cache_latents (and disable incompatible augmentations).",
    "Disable text-encoder LoRA training.",
    "Increase gradient_accumulation_steps instead of batch size.",
    "Enable VAE slicing/tiling and attention slicing if available.",
    "Close other GPU processes competing for VRAM.",
]


@dataclass
class MemorySnapshot:
    allocated_gb: float
    reserved_gb: float
    max_allocated_gb: float


def gpu_memory_snapshot(device: int | str = 0) -> MemorySnapshot | None:
    import torch

    if not torch.cuda.is_available():
        return None
    if isinstance(device, str) and device.startswith("cuda"):
        index = int(device.split(":")[-1]) if ":" in device else 0
    elif isinstance(device, str):
        return None
    else:
        index = int(device)

    allocated = torch.cuda.memory_allocated(index) / (1024**3)
    reserved = torch.cuda.memory_reserved(index) / (1024**3)
    peak = torch.cuda.max_memory_allocated(index) / (1024**3)
    return MemorySnapshot(
        allocated_gb=round(allocated, 3),
        reserved_gb=round(reserved, 3),
        max_allocated_gb=round(peak, 3),
    )


def reset_peak_memory_stats() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def format_oom_message(exc: BaseException) -> str:
    lines = [
        "CUDA out-of-memory detected. Training parameters were NOT modified.",
        f"Original error: {exc}",
        "Suggested mitigations:",
    ]
    lines.extend(f"  - {item}" for item in OOM_SUGGESTIONS)
    return "\n".join(lines)


def count_parameters(module) -> tuple[int, int]:
    """Return (trainable, total) parameter counts."""
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    total = sum(p.numel() for p in module.parameters())
    return trainable, total


def format_param_counts(trainable: int, total: int) -> str:
    pct = (100.0 * trainable / total) if total else 0.0
    return f"{trainable:,} / {total:,} ({pct:.4f}%)"
