"""Dataset loaders for local folders and Hugging Face datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

from image_lora_trainer.config import DatasetConfig, PreprocessConfig
from image_lora_trainer.data.preprocessing import (
    CaptionSpec,
    DiffusionImageProcessor,
    build_caption,
)
from image_lora_trainer.data.validation import (
    discover_images,
    load_metadata_jsonl,
    validate_local_dataset,
)
from image_lora_trainer.logging_utils import get_logger

logger = get_logger(__name__)


class ImageCaptionDataset(Dataset):
    """Local folder dataset with metadata.jsonl or per-image .txt captions."""

    def __init__(
        self,
        dataset_cfg: DatasetConfig,
        preprocess_cfg: PreprocessConfig,
        *,
        validate: bool = True,
    ) -> None:
        if not dataset_cfg.dataset_path:
            raise ValueError("dataset_path is required for ImageCaptionDataset")
        self.root = Path(dataset_cfg.dataset_path)
        self.dataset_cfg = dataset_cfg
        self.processor = DiffusionImageProcessor(preprocess_cfg)
        self.caption_spec = CaptionSpec(
            default_caption=dataset_cfg.default_caption,
            instance_prompt=dataset_cfg.instance_prompt,
            trigger_word=dataset_cfg.trigger_word,
            caption_prefix=dataset_cfg.caption_prefix,
            caption_suffix=dataset_cfg.caption_suffix,
            caption_dropout=dataset_cfg.caption_dropout,
            shuffle_captions=dataset_cfg.shuffle_captions,
        )

        if validate:
            report = validate_local_dataset(
                self.root,
                caption_column=dataset_cfg.caption_column,
                file_name_column=dataset_cfg.file_name_column,
                default_caption=dataset_cfg.default_caption
                or dataset_cfg.instance_prompt,
            )
            report.raise_if_invalid()

        self.samples = self._build_index()
        if not self.samples:
            raise ValueError(f"No training samples found in {self.root}")
        logger.info("Loaded %d local samples from %s", len(self.samples), self.root)

    def _build_index(self) -> list[dict[str, str]]:
        captions: dict[str, str] = {}
        metadata_path = self.root / "metadata.jsonl"
        if metadata_path.is_file():
            for row in load_metadata_jsonl(metadata_path):
                name = str(row[self.dataset_cfg.file_name_column])
                text = row.get(self.dataset_cfg.caption_column)
                captions[name] = "" if text is None else str(text)

        samples: list[dict[str, str]] = []
        for image_path in discover_images(self.root):
            rel = image_path.relative_to(self.root).as_posix()
            caption = captions.get(image_path.name, captions.get(rel))
            if caption is None:
                txt = image_path.with_suffix(".txt")
                if txt.is_file():
                    caption = txt.read_text(encoding="utf-8")
                else:
                    caption = self.dataset_cfg.default_caption or self.dataset_cfg.instance_prompt
            if caption is None:
                continue
            samples.append({"path": str(image_path), "caption": caption})
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.samples[index]
        with Image.open(item["path"]) as img:
            processed = self.processor(img)
        caption = build_caption(item["caption"], self.caption_spec)
        return {
            "pixel_values": processed["pixel_values"],
            "caption": caption,
            "original_sizes": processed["original_sizes"],
            "crop_top_lefts": processed["crop_top_lefts"],
            "target_sizes": processed["target_sizes"],
        }


class HFImageCaptionDataset(Dataset):
    """Wrapper around datasets.Dataset for Hub-hosted image/caption data."""

    def __init__(
        self,
        dataset_cfg: DatasetConfig,
        preprocess_cfg: PreprocessConfig,
        split: str = "train",
    ) -> None:
        if not dataset_cfg.dataset_name:
            raise ValueError("dataset_name is required for HFImageCaptionDataset")
        from datasets import load_dataset

        self.dataset_cfg = dataset_cfg
        self.processor = DiffusionImageProcessor(preprocess_cfg)
        self.caption_spec = CaptionSpec(
            default_caption=dataset_cfg.default_caption,
            instance_prompt=dataset_cfg.instance_prompt,
            trigger_word=dataset_cfg.trigger_word,
            caption_prefix=dataset_cfg.caption_prefix,
            caption_suffix=dataset_cfg.caption_suffix,
            caption_dropout=dataset_cfg.caption_dropout,
            shuffle_captions=dataset_cfg.shuffle_captions,
        )
        self.ds = load_dataset(
            dataset_cfg.dataset_name,
            name=dataset_cfg.dataset_config_name,
            split=split,
        )
        if len(self.ds) == 0:
            raise ValueError(f"Hugging Face dataset is empty: {dataset_cfg.dataset_name}")
        logger.info(
            "Loaded %d Hub samples from %s", len(self.ds), dataset_cfg.dataset_name
        )

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.ds[index]
        image = row[self.dataset_cfg.image_column]
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        processed = self.processor(image)
        raw_caption = row.get(self.dataset_cfg.caption_column)
        caption = build_caption(
            None if raw_caption is None else str(raw_caption),
            self.caption_spec,
        )
        return {
            "pixel_values": processed["pixel_values"],
            "caption": caption,
            "original_sizes": processed["original_sizes"],
            "crop_top_lefts": processed["crop_top_lefts"],
            "target_sizes": processed["target_sizes"],
        }


def build_dataset(
    dataset_cfg: DatasetConfig,
    preprocess_cfg: PreprocessConfig,
) -> Dataset:
    if dataset_cfg.dataset_path:
        return ImageCaptionDataset(dataset_cfg, preprocess_cfg)
    return HFImageCaptionDataset(dataset_cfg, preprocess_cfg)


def collate_batch(examples: list[dict[str, Any]]) -> dict[str, Any]:
    pixel_values = torch.stack([ex["pixel_values"] for ex in examples])
    captions = [ex["caption"] for ex in examples]
    batch: dict[str, Any] = {
        "pixel_values": pixel_values,
        "captions": captions,
        "original_sizes": torch.tensor(
            np_stack(examples, "original_sizes"), dtype=torch.long
        ),
        "crop_top_lefts": torch.tensor(
            np_stack(examples, "crop_top_lefts"), dtype=torch.long
        ),
        "target_sizes": torch.tensor(np_stack(examples, "target_sizes"), dtype=torch.long),
    }
    return batch


def np_stack(examples: list[dict[str, Any]], key: str):
    import numpy as np

    return np.stack([ex[key] for ex in examples], axis=0)


def write_example_metadata(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
