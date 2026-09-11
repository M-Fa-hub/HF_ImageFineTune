#!/usr/bin/env python
"""Validate a local image+caption dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from image_lora_trainer.data.validation import validate_local_dataset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--default-caption", default=None)
    args = parser.parse_args()
    report = validate_local_dataset(args.dataset, default_caption=args.default_caption)
    print(
        json.dumps(
            {
                "ok": report.ok,
                "num_images": report.num_images,
                "num_captions": report.num_captions,
                "issues": [{"path": i.path, "reason": i.reason} for i in report.issues],
            },
            indent=2,
        )
    )
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
