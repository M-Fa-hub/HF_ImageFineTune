"""Dataset loading and validation tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from image_lora_trainer.config import DatasetConfig, PreprocessConfig
from image_lora_trainer.data.dataset import ImageCaptionDataset, build_dataset
from image_lora_trainer.data.preprocessing import CaptionSpec, build_caption
from image_lora_trainer.data.validation import validate_local_dataset


@pytest.fixture()
def tiny_dataset(tmp_path: Path) -> Path:
    img = Image.new("RGB", (256, 320), color=(120, 80, 60))
    img.save(tmp_path / "0001.jpg")
    Image.new("RGB", (256, 320), color=(30, 30, 30)).save(tmp_path / "0002.png")
    with (tmp_path / "metadata.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"file_name": "0001.jpg", "text": "a cat"}) + "\n")
        handle.write(json.dumps({"file_name": "0002.png", "text": "a dog"}) + "\n")
    return tmp_path


def test_validate_ok(tiny_dataset: Path):
    report = validate_local_dataset(tiny_dataset)
    assert report.ok
    assert report.num_images == 2


def test_missing_caption(tmp_path: Path):
    Image.new("RGB", (64, 64), color=1).save(tmp_path / "only.jpg")
    with (tmp_path / "metadata.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"file_name": "only.jpg", "text": ""}) + "\n")
    report = validate_local_dataset(tmp_path)
    assert not report.ok
    assert any("missing caption" in i.reason for i in report.issues)


def test_corrupted_image(tmp_path: Path):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not-an-image")
    with (tmp_path / "metadata.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"file_name": "bad.jpg", "text": "x"}) + "\n")
    report = validate_local_dataset(tmp_path)
    assert not report.ok
    assert any("corrupted" in i.reason for i in report.issues)


def test_invalid_extension_ignored(tmp_path: Path):
    (tmp_path / "notes.txt").write_text("hello", encoding="utf-8")
    Image.new("RGB", (32, 32), color=2).save(tmp_path / "ok.webp")
    with (tmp_path / "metadata.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"file_name": "ok.webp", "text": "webp ok"}) + "\n")
    report = validate_local_dataset(tmp_path)
    assert report.ok
    assert report.num_images == 1


def test_empty_dataset(tmp_path: Path):
    report = validate_local_dataset(tmp_path)
    assert not report.ok


def test_dataset_loading(tiny_dataset: Path):
    ds_cfg = DatasetConfig(dataset_path=str(tiny_dataset), trigger_word="sks")
    pre_cfg = PreprocessConfig(resolution=128, center_crop=True, horizontal_flip=False)
    dataset = build_dataset(ds_cfg, pre_cfg)
    assert len(dataset) == 2
    item = dataset[0]
    assert item["pixel_values"].shape[0] == 3
    assert item["pixel_values"].shape[1] == 128
    assert "sks" in item["caption"]


def test_caption_builder():
    spec = CaptionSpec(
        trigger_word="sks_person",
        caption_prefix="photo of",
        caption_suffix="high quality",
        caption_dropout=0.0,
    )
    caption = build_caption("standing outside", spec)
    assert caption.startswith("photo of")
    assert "sks_person" in caption
    assert caption.endswith("high quality")


def test_default_caption_folder(tmp_path: Path):
    Image.new("RGB", (64, 64), color=3).save(tmp_path / "a.jpg")
    Image.new("RGB", (64, 64), color=4).save(tmp_path / "b.jpg")
    ds = ImageCaptionDataset(
        DatasetConfig(dataset_path=str(tmp_path), default_caption="sks_style artwork"),
        PreprocessConfig(resolution=64, horizontal_flip=False),
    )
    assert len(ds) == 2
    assert "sks_style" in ds[0]["caption"]
