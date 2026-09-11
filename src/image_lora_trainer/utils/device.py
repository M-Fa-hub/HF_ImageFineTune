"""Device and CUDA capability helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class DeviceInfo:
    cuda_available: bool
    device_count: int
    gpu_names: list[str]
    total_vram_gb: list[float]
    cuda_version: str | None
    torch_version: str
    bf16_supported: bool
    tf32_available: bool
    bitsandbytes_available: bool
    xformers_available: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def primary_vram_gb(self) -> float | None:
        return self.total_vram_gb[0] if self.total_vram_gb else None


def _package_available(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False


def collect_device_info() -> DeviceInfo:
    import torch

    cuda_available = torch.cuda.is_available()
    gpu_names: list[str] = []
    total_vram_gb: list[float] = []
    bf16_supported = False
    if cuda_available:
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            gpu_names.append(props.name)
            total_vram_gb.append(round(props.total_memory / (1024**3), 2))
        # Ampere+ generally supports bf16 well; also probe torch API when present.
        try:
            bf16_supported = bool(torch.cuda.is_bf16_supported())
        except Exception:
            major, _ = torch.cuda.get_device_capability(0)
            bf16_supported = major >= 8

    return DeviceInfo(
        cuda_available=cuda_available,
        device_count=torch.cuda.device_count() if cuda_available else 0,
        gpu_names=gpu_names,
        total_vram_gb=total_vram_gb,
        cuda_version=getattr(torch.version, "cuda", None),
        torch_version=torch.__version__,
        bf16_supported=bf16_supported,
        tf32_available=cuda_available,
        bitsandbytes_available=_package_available("bitsandbytes"),
        xformers_available=_package_available("xformers"),
    )


def resolve_device(preferred: str | None = None) -> str:
    import torch

    if preferred:
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def enable_tf32(enabled: bool = True) -> None:
    import torch

    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled


def dtype_from_name(name: str):
    import torch

    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype '{name}'")
    return mapping[name]
