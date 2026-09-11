"""Logging helpers for training and CLI."""

from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

_SECRET_PATTERNS = (
    re.compile(r"(hf_[A-Za-z0-9]{20,})"),
    re.compile(r"(api[_-]?key['\"]?\s*[:=]\s*['\"]?)([^\s'\"]+)", re.I),
    re.compile(r"(token['\"]?\s*[:=]\s*['\"]?)([^\s'\"]+)", re.I),
    re.compile(r"(Bearer\s+)([A-Za-z0-9\-._~+/]+=*)", re.I),
)


class SecretRedactingFilter(logging.Filter):
    """Prevent accidental leakage of tokens in log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            return True
        redacted = _redact(message)
        if redacted != message:
            record.msg = redacted
            record.args = ()
        return True


def _redact(text: str) -> str:
    out = text
    for pattern in _SECRET_PATTERNS:
        if pattern.groups == 1:
            out = pattern.sub("[REDACTED]", out)
        else:
            out = pattern.sub(r"\1[REDACTED]", out)
    for env_key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "WANDB_API_KEY"):
        secret = os.environ.get(env_key)
        if secret and secret in out:
            out = out.replace(secret, "[REDACTED]")
    return out


def setup_logging(
    level: int = logging.INFO,
    log_file: Path | None = None,
    name: str = "image_lora_trainer",
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    stream.addFilter(SecretRedactingFilter())
    logger.addHandler(stream)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.addFilter(SecretRedactingFilter())
        logger.addHandler(file_handler)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    return logger


def get_logger(name: str = "image_lora_trainer") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logging(name=name)
    return logger


def format_startup_summary(fields: dict[str, Any]) -> str:
    width = max(len(k) for k in fields) if fields else 10
    lines = ["=" * 60, "TRAINING STARTUP SUMMARY", "=" * 60]
    for key, value in fields.items():
        lines.append(f"{key:<{width}} : {value}")
    lines.append("=" * 60)
    return "\n".join(lines)


class ExperimentTracker:
    """Thin wrapper around TensorBoard and optional W&B."""

    def __init__(
        self,
        backend: str,
        output_dir: Path,
        project: str = "image-lora-trainer",
        entity: str | None = None,
        config: dict[str, Any] | None = None,
        run_name: str | None = None,
    ) -> None:
        self.backend = backend
        self.output_dir = Path(output_dir)
        self._tb_writer = None
        self._wandb = None

        if backend in {"tensorboard", "both"}:
            from torch.utils.tensorboard import SummaryWriter

            tb_dir = self.output_dir / "logs" / "tensorboard"
            tb_dir.mkdir(parents=True, exist_ok=True)
            self._tb_writer = SummaryWriter(log_dir=str(tb_dir))

        if backend in {"wandb", "both"}:
            try:
                import wandb
            except ImportError as exc:
                raise ImportError(
                    "W&B logging requested but wandb is not installed. "
                    "Install with: pip install 'image-lora-trainer[wandb]'"
                ) from exc
            self._wandb = wandb
            wandb.init(
                project=project,
                entity=entity,
                name=run_name,
                config=config or {},
                dir=str(self.output_dir / "logs"),
            )

    def log_scalars(self, metrics: dict[str, float], step: int) -> None:
        if self._tb_writer is not None:
            for key, value in metrics.items():
                self._tb_writer.add_scalar(key, value, step)
        if self._wandb is not None:
            self._wandb.log({**metrics, "step": step}, step=step)

    def log_images(self, tag: str, images: list[Any], step: int) -> None:
        if self._tb_writer is not None:
            import numpy as np
            import torch

            tensors = []
            for image in images:
                array = np.asarray(image).astype("float32") / 255.0
                tensor = torch.from_numpy(array).permute(2, 0, 1)
                tensors.append(tensor)
            if tensors:
                grid = torch.stack(tensors, dim=0)
                self._tb_writer.add_images(tag, grid, step)
        if self._wandb is not None:
            self._wandb.log(
                {tag: [self._wandb.Image(img) for img in images], "step": step},
                step=step,
            )

    def close(self) -> None:
        if self._tb_writer is not None:
            self._tb_writer.flush()
            self._tb_writer.close()
        if self._wandb is not None:
            self._wandb.finish()
