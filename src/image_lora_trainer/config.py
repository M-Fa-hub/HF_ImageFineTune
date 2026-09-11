"""Typed configuration schema for image LoRA / QLoRA training."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class ArchitectureFamily(StrEnum):
    """Known Diffusers text-to-image architecture families."""

    SD15 = "sd15"
    SD21 = "sd21"
    SDXL = "sdxl"
    FLUX = "flux"
    UNKNOWN = "unknown"


class MixedPrecision(StrEnum):
    NO = "no"
    FP16 = "fp16"
    BF16 = "bf16"


class CropMode(StrEnum):
    CENTER = "center"
    RANDOM = "random"
    NONE = "none"


class PredictionType(StrEnum):
    EPSILON = "epsilon"
    V_PREDICTION = "v_prediction"
    SAMPLE = "sample"
    FLOW_MATCHING = "flow_matching"
    AUTO = "auto"


class OptimizerName(StrEnum):
    ADAMW = "adamw"
    ADAMW_8BIT = "adamw_8bit"


class SchedulerName(StrEnum):
    CONSTANT = "constant"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"


class LoggingBackend(StrEnum):
    TENSORBOARD = "tensorboard"
    WANDB = "wandb"
    BOTH = "both"
    NONE = "none"


class ModelConfig(BaseModel):
    pretrained_model_name_or_path: str = Field(
        ...,
        description="Hugging Face model id or local path to a Diffusers pipeline.",
    )
    revision: str | None = None
    variant: str | None = None
    architecture: ArchitectureFamily | Literal["auto"] = "auto"
    tokenizer_max_length: int | None = None
    prediction_type: PredictionType = PredictionType.AUTO


class DatasetConfig(BaseModel):
    dataset_path: str | None = Field(
        default=None,
        description="Local folder containing images + metadata.jsonl (or captions).",
    )
    dataset_name: str | None = Field(
        default=None,
        description="Hugging Face dataset id, e.g. owner/dataset.",
    )
    dataset_config_name: str | None = None
    image_column: str = "image"
    caption_column: str = "text"
    file_name_column: str = "file_name"
    default_caption: str | None = None
    instance_prompt: str | None = None
    trigger_word: str | None = None
    caption_prefix: str = ""
    caption_suffix: str = ""
    caption_dropout: float = Field(default=0.0, ge=0.0, le=1.0)
    shuffle_captions: bool = False
    cache_latents: bool = False
    cache_text_embeddings: bool = False
    num_workers: int = Field(default=2, ge=0)
    pin_memory: bool = True

    @model_validator(mode="after")
    def require_source(self) -> DatasetConfig:
        if not self.dataset_path and not self.dataset_name:
            raise ValueError("Provide either dataset_path or dataset_name.")
        if self.dataset_path and self.dataset_name:
            raise ValueError("Provide only one of dataset_path or dataset_name.")
        return self


class PreprocessConfig(BaseModel):
    resolution: int = Field(default=1024, ge=64, le=2048)
    center_crop: bool = True
    random_crop: bool = False
    crop_mode: CropMode = CropMode.CENTER
    horizontal_flip: bool = True
    flip_probability: float = Field(default=0.5, ge=0.0, le=1.0)
    interpolation: Literal["bilinear", "bicubic", "lanczos"] = "lanczos"
    normalize: bool = True
    keep_aspect_ratio: bool = True

    @model_validator(mode="after")
    def sync_crop_flags(self) -> PreprocessConfig:
        if self.random_crop and self.center_crop:
            # Explicit random crop wins over center crop.
            self.center_crop = False
            self.crop_mode = CropMode.RANDOM
        elif self.center_crop:
            self.crop_mode = CropMode.CENTER
        elif self.random_crop:
            self.crop_mode = CropMode.RANDOM
        else:
            self.crop_mode = CropMode.NONE
        return self


class LoRAModuleConfig(BaseModel):
    enabled: bool = True
    rank: int = Field(default=16, ge=1, le=512)
    alpha: int = Field(default=16, ge=1)
    dropout: float = Field(default=0.05, ge=0.0, le=1.0)
    bias: Literal["none", "all", "lora_only"] = "none"
    target_modules: list[str] | None = None
    learning_rate: float | None = None


class LoRAConfig(BaseModel):
    enabled: bool = True
    rank: int = Field(default=16, ge=1, le=512)
    alpha: int = Field(default=16, ge=1)
    dropout: float = Field(default=0.05, ge=0.0, le=1.0)
    bias: Literal["none", "all", "lora_only"] = "none"
    target_modules: list[str] | None = None
    train_text_encoder: bool = False
    train_text_encoder_2: bool = False
    denoiser: LoRAModuleConfig | None = None
    text_encoder: LoRAModuleConfig | None = None
    text_encoder_2: LoRAModuleConfig | None = None

    def resolved_denoiser(self) -> LoRAModuleConfig:
        if self.denoiser is not None:
            return self.denoiser
        return LoRAModuleConfig(
            enabled=self.enabled,
            rank=self.rank,
            alpha=self.alpha,
            dropout=self.dropout,
            bias=self.bias,
            target_modules=self.target_modules,
        )

    def resolved_text_encoder(self) -> LoRAModuleConfig:
        if self.text_encoder is not None:
            return self.text_encoder
        return LoRAModuleConfig(
            enabled=self.train_text_encoder,
            rank=max(4, self.rank // 2),
            alpha=max(4, self.alpha // 2),
            dropout=self.dropout,
            bias=self.bias,
            target_modules=None,
        )

    def resolved_text_encoder_2(self) -> LoRAModuleConfig:
        if self.text_encoder_2 is not None:
            return self.text_encoder_2
        return LoRAModuleConfig(
            enabled=self.train_text_encoder_2,
            rank=max(4, self.rank // 2),
            alpha=max(4, self.alpha // 2),
            dropout=self.dropout,
            bias=self.bias,
            target_modules=None,
        )


class QuantizationConfig(BaseModel):
    enabled: bool = False
    bits: Literal[4, 8] = 4
    quant_type: Literal["nf4", "fp4", "int8"] = "nf4"
    double_quant: bool = True
    compute_dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    quantize_denoiser: bool = True
    quantize_text_encoder: bool = False
    allow_quantization_fallback: bool = False

    @model_validator(mode="after")
    def validate_bits_type(self) -> QuantizationConfig:
        if self.bits == 4 and self.quant_type not in {"nf4", "fp4"}:
            raise ValueError("4-bit quantization requires quant_type 'nf4' or 'fp4'.")
        if self.bits == 8 and self.quant_type in {"nf4", "fp4"}:
            raise ValueError("8-bit quantization requires quant_type 'int8'.")
        return self


class OptimizerConfig(BaseModel):
    name: OptimizerName = OptimizerName.ADAMW
    learning_rate: float = Field(default=1e-4, gt=0)
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = Field(default=0.01, ge=0)
    epsilon: float = Field(default=1e-8, gt=0)
    max_grad_norm: float = Field(default=1.0, gt=0)


class SchedulerConfig(BaseModel):
    name: SchedulerName = SchedulerName.CONSTANT_WITH_WARMUP
    warmup_steps: int = Field(default=100, ge=0)
    num_cycles: int = Field(default=1, ge=1)
    power: float = Field(default=1.0, gt=0)


class TrainingConfig(BaseModel):
    output_dir: str = "./outputs/run"
    run_name: str | None = None
    seed: int = 42
    resolution: int = Field(default=1024, ge=64, le=2048)
    train_batch_size: int = Field(default=1, ge=1)
    gradient_accumulation_steps: int = Field(default=4, ge=1)
    num_train_epochs: int | None = None
    max_train_steps: int | None = 2000
    mixed_precision: MixedPrecision = MixedPrecision.BF16
    gradient_checkpointing: bool = True
    enable_tf32: bool = True
    enable_xformers: bool = False
    enable_sliced_attention: bool = False
    enable_vae_slicing: bool = True
    enable_vae_tiling: bool = False
    cpu_offload: bool = False
    resume_from_checkpoint: str | None = None
    checkpointing_steps: int = Field(default=250, ge=1)
    checkpoints_total_limit: int | None = Field(default=3, ge=1)
    dataloader_drop_last: bool = False
    allow_tf32: bool = True

    @model_validator(mode="after")
    def require_steps_or_epochs(self) -> TrainingConfig:
        if self.max_train_steps is None and self.num_train_epochs is None:
            raise ValueError("Set max_train_steps or num_train_epochs.")
        return self


class ValidationConfig(BaseModel):
    enabled: bool = True
    every_n_steps: int = Field(default=250, ge=1)
    prompts: list[str] = Field(default_factory=list)
    negative_prompt: str | None = None
    num_images_per_prompt: int = Field(default=1, ge=1, le=8)
    seed: int = 42
    num_inference_steps: int = Field(default=30, ge=1)
    guidance_scale: float = 7.5
    height: int | None = None
    width: int | None = None


class LoggingConfig(BaseModel):
    backend: LoggingBackend = LoggingBackend.TENSORBOARD
    log_every_n_steps: int = Field(default=10, ge=1)
    project: str = "image-lora-trainer"
    entity: str | None = None


class HubConfig(BaseModel):
    repo_id: str | None = None
    private: bool = True
    push_to_hub: bool = False
    token_env: str = "HF_TOKEN"


class MemoryConfig(BaseModel):
    allow_quantization_fallback: bool = False


class TrainConfig(BaseModel):
    """Top-level training configuration."""

    model: ModelConfig
    dataset: DatasetConfig
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    quantization: QuantizationConfig = Field(default_factory=QuantizationConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    lr_scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    hub: HubConfig = Field(default_factory=HubConfig)

    @model_validator(mode="after")
    def sync_resolution(self) -> TrainConfig:
        # Keep preprocess and training resolution aligned unless preprocess differs
        # intentionally via preprocess.resolution override already set.
        if self.preprocess.resolution != self.training.resolution:
            # Prefer explicit preprocess resolution for the image pipeline.
            self.training.resolution = self.preprocess.resolution
        if self.quantization.enabled and not self.lora.enabled:
            raise ValueError(
                "Quantization without LoRA is not supported for PEFT fine-tuning. "
                "Enable lora.enabled=true (QLoRA-style training)."
            )
        if self.dataset.cache_text_embeddings and (
            self.lora.train_text_encoder or self.lora.train_text_encoder_2
        ):
            raise ValueError(
                "cache_text_embeddings cannot be used while training text encoders."
            )
        if self.dataset.cache_latents and self.preprocess.random_crop:
            raise ValueError(
                "cache_latents is incompatible with random_crop (cached latents "
                "would not reflect per-step crop augmentation)."
            )
        if self.quantization.allow_quantization_fallback:
            # Mirror into a single place for capability checks.
            pass
        return self

    def method_name(self) -> str:
        if self.quantization.enabled:
            return "qlora" if self.quantization.bits == 4 else "lora_8bit"
        return "lora"

    def to_yaml_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping, got {type(data).__name__}")
    return data


def deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def parse_cli_overrides(pairs: list[str] | None) -> dict[str, Any]:
    """Parse dotted CLI overrides like model.revision=main into nested dicts."""
    if not pairs:
        return {}
    root: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Invalid override '{pair}'. Expected key=value.")
        key, raw = pair.split("=", 1)
        parts = key.strip().split(".")
        cursor: dict[str, Any] = root
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
            if not isinstance(cursor, dict):
                raise ValueError(f"Cannot override non-mapping path in '{pair}'.")
        cursor[parts[-1]] = _coerce_scalar(raw.strip())
    return root


def _coerce_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    if lowered in {"null", "none"}:
        return None
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def load_train_config(
    config_path: str | Path,
    overrides: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> TrainConfig:
    data = load_yaml_config(config_path)
    data = deep_update(data, parse_cli_overrides(overrides))
    if extra:
        data = deep_update(data, extra)
    return TrainConfig.model_validate(data)


def save_train_config(config: TrainConfig, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_yaml_dict(), handle, sort_keys=False, allow_unicode=True)


class InferenceConfig(BaseModel):
    base_model: str
    lora_path: str | None = None
    lora_scale: float = Field(default=1.0, ge=0.0, le=2.0)
    prompt: str
    negative_prompt: str | None = None
    output: str = "generated.png"
    seed: int | None = 42
    width: int = Field(default=1024, ge=64)
    height: int = Field(default=1024, ge=64)
    num_inference_steps: int = Field(default=30, ge=1)
    guidance_scale: float = 7.5
    num_images: int = Field(default=1, ge=1, le=16)
    dtype: Literal["float16", "bfloat16", "float32"] = "bfloat16"
    device: str | None = None

    @field_validator("prompt")
    @classmethod
    def non_empty_prompt(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt must be non-empty")
        return value
