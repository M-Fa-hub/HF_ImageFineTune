#!/usr/bin/env python
"""Train entrypoint script."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running without installation by adding src/ to path.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from image_lora_trainer.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["train", *sys.argv[1:]]))
