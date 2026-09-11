"""Dataset validation for local image+caption corpora."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, UnidentifiedImageError

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class DatasetIssue:
    path: str
    reason: str


@dataclass
class DatasetValidationReport:
    root: str
    num_images: int = 0
    num_captions: int = 0
    issues: list[DatasetIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.num_images > 0 and not self.issues

    def raise_if_invalid(self) -> None:
        if self.ok:
            return
        details = "\n".join(f"  - {i.path}: {i.reason}" for i in self.issues[:50])
        more = "" if len(self.issues) <= 50 else f"\n  ... and {len(self.issues) - 50} more"
        raise ValueError(
            f"Dataset validation failed for {self.root}\n"
            f"images={self.num_images}, captions={self.num_captions}, issues={len(self.issues)}\n"
            f"{details}{more}"
        )


def discover_images(root: Path) -> list[Path]:
    images: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            images.append(path)
    return images


def load_metadata_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{line_no}")
            rows.append(row)
    return rows


def validate_local_dataset(
    root: str | Path,
    *,
    caption_column: str = "text",
    file_name_column: str = "file_name",
    default_caption: str | None = None,
    check_image_open: bool = True,
) -> DatasetValidationReport:
    root_path = Path(root)
    report = DatasetValidationReport(root=str(root_path))
    if not root_path.is_dir():
        report.issues.append(DatasetIssue(str(root_path), "directory does not exist"))
        return report

    images = discover_images(root_path)
    report.num_images = len(images)
    if not images:
        report.issues.append(DatasetIssue(str(root_path), "no JPEG/PNG/WEBP images found"))
        return report

    metadata_path = root_path / "metadata.jsonl"
    captions: dict[str, str] = {}
    if metadata_path.is_file():
        for row in load_metadata_jsonl(metadata_path):
            file_name = row.get(file_name_column)
            text = row.get(caption_column)
            if not file_name:
                report.issues.append(
                    DatasetIssue(str(metadata_path), f"missing '{file_name_column}'")
                )
                continue
            if text is None or str(text).strip() == "":
                if default_caption is None:
                    report.issues.append(
                        DatasetIssue(str(file_name), "missing caption and no default_caption")
                    )
                    continue
                text = default_caption
            captions[str(file_name)] = str(text)
        report.num_captions = len(captions)
    elif default_caption:
        report.num_captions = len(images)
    else:
        # Fall back to adjacent .txt caption files.
        for image in images:
            txt = image.with_suffix(".txt")
            if txt.is_file():
                captions[image.name] = txt.read_text(encoding="utf-8").strip()
            else:
                report.issues.append(
                    DatasetIssue(
                        image.name,
                        "no metadata.jsonl, no .txt caption, and no default_caption",
                    )
                )
        report.num_captions = len(captions)

    image_names = {img.name for img in images}
    for name in captions:
        if name not in image_names and not (root_path / name).is_file():
            report.issues.append(DatasetIssue(name, "caption references missing image file"))

    if metadata_path.is_file():
        for image in images:
            rel = image.relative_to(root_path).as_posix()
            if image.name not in captions and rel not in captions and default_caption is None:
                report.issues.append(DatasetIssue(image.name, "image missing caption entry"))

    if check_image_open:
        for image in images:
            try:
                with Image.open(image) as img:
                    img.verify()
            except (UnidentifiedImageError, OSError) as exc:
                report.issues.append(DatasetIssue(str(image), f"corrupted/unreadable: {exc}"))

    return report
