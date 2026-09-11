"""Image preprocessing and caption transforms."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF

from image_lora_trainer.config import CropMode, PreprocessConfig

_INTERPOLATION = {
    "bilinear": transforms.InterpolationMode.BILINEAR,
    "bicubic": transforms.InterpolationMode.BICUBIC,
    "lanczos": transforms.InterpolationMode.LANCZOS,
}


@dataclass
class CaptionSpec:
    default_caption: str | None = None
    instance_prompt: str | None = None
    trigger_word: str | None = None
    caption_prefix: str = ""
    caption_suffix: str = ""
    caption_dropout: float = 0.0
    shuffle_captions: bool = False


def build_caption(raw: str | None, spec: CaptionSpec, rng: random.Random | None = None) -> str:
    rng_obj: random.Random = rng if rng is not None else random.Random()
    if rng_obj.random() < spec.caption_dropout:
        base = ""
    elif raw and raw.strip():
        base = raw.strip()
    elif spec.instance_prompt:
        base = spec.instance_prompt
    elif spec.default_caption:
        base = spec.default_caption
    else:
        base = ""

    if spec.shuffle_captions and base:
        tokens = [t.strip() for t in base.replace(",", ", ").split(",") if t.strip()]
        if len(tokens) > 1:
            rng_obj.shuffle(tokens)
            base = ", ".join(tokens)

    parts: list[str] = []
    if spec.caption_prefix:
        parts.append(spec.caption_prefix.strip())
    if spec.trigger_word and spec.trigger_word not in base:
        parts.append(spec.trigger_word)
    if base:
        parts.append(base)
    if spec.caption_suffix:
        parts.append(spec.caption_suffix.strip())
    return " ".join(p for p in parts if p).strip()


class DiffusionImageProcessor:
    """Deterministic / stochastic preprocessing without naive stretch-distortion."""

    def __init__(self, config: PreprocessConfig) -> None:
        self.config = config
        self.interpolation = _INTERPOLATION[config.interpolation]

    def __call__(self, image: Image.Image) -> dict[str, torch.Tensor | np.ndarray]:
        image = image.convert("RGB")
        target = self.config.resolution

        if self.config.keep_aspect_ratio:
            image = self._resize_short_side(image, target)
            image, crop_coords = self._crop(image, target)
        else:
            image = TF.resize(image, [target, target], interpolation=self.interpolation)
            crop_coords = (0, 0, target, target)

        if self.config.horizontal_flip and random.random() < self.config.flip_probability:
            image = TF.hflip(image)
            left, top, right, bottom = crop_coords
            # Mirror horizontal crop origin for SDXL-style conditioning metadata.
            right - left
            # original width after resize is left+width+(...) — approximate via image size.
            crop_coords = (image.width - right, top, image.width - left, bottom)

        tensor = TF.to_tensor(image)
        if self.config.normalize:
            tensor = TF.normalize(tensor, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

        original_size = np.array([image.height, image.width], dtype=np.int64)
        crop_top_left = np.array([crop_coords[1], crop_coords[0]], dtype=np.int64)
        target_size = np.array([target, target], dtype=np.int64)

        return {
            "pixel_values": tensor,
            "original_sizes": original_size,
            "crop_top_lefts": crop_top_left,
            "target_sizes": target_size,
        }

    def _resize_short_side(self, image: Image.Image, target: int) -> Image.Image:
        width, height = image.size
        short = min(width, height)
        if short == target:
            return image
        scale = target / float(short)
        new_w = max(target, round(width * scale))
        new_h = max(target, round(height * scale))
        resized = TF.resize(image, [new_h, new_w], interpolation=self.interpolation)
        if isinstance(resized, Image.Image):
            return resized
        return Image.fromarray(resized)  # type: ignore[arg-type]

    def _crop(
        self, image: Image.Image, target: int
    ) -> tuple[Image.Image, tuple[int, int, int, int]]:
        width, height = image.size
        if width == target and height == target:
            return image, (0, 0, target, target)

        if self.config.crop_mode == CropMode.RANDOM:
            top = random.randint(0, max(0, height - target))
            left = random.randint(0, max(0, width - target))
        elif self.config.crop_mode == CropMode.CENTER:
            top = max(0, (height - target) // 2)
            left = max(0, (width - target) // 2)
        else:
            # No crop: letterbox-style pad to square without distorting aspect.
            return self._pad_to_square(image, target)

        right = left + target
        bottom = top + target
        return TF.crop(image, top, left, target, target), (left, top, right, bottom)

    def _pad_to_square(
        self, image: Image.Image, target: int
    ) -> tuple[Image.Image, tuple[int, int, int, int]]:
        width, height = image.size
        canvas = Image.new("RGB", (target, target), (0, 0, 0))
        left = (target - width) // 2
        top = (target - height) // 2
        canvas.paste(image, (left, top))
        return canvas, (left, top, left + width, top + height)
