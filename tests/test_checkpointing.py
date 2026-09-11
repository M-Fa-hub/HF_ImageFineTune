"""Checkpoint naming and discovery tests."""

from __future__ import annotations

import json
from pathlib import Path

from image_lora_trainer.training.checkpointing import (
    checkpoint_step,
    find_latest_checkpoint,
    list_checkpoints,
    resolve_resume_path,
)


def test_checkpoint_step_parsing():
    assert checkpoint_step(Path("checkpoint-250")) == 250
    assert checkpoint_step(Path("checkpoint-1000")) == 1000
    assert checkpoint_step(Path("final")) is None


def test_find_latest_checkpoint(tmp_path: Path):
    ckpt_root = tmp_path / "checkpoints"
    for step in (100, 500, 250):
        d = ckpt_root / f"checkpoint-{step}"
        d.mkdir(parents=True)
        (d / "trainer_state.json").write_text(
            json.dumps({"global_step": step}), encoding="utf-8"
        )
    latest = find_latest_checkpoint(tmp_path)
    assert latest is not None
    assert latest.name == "checkpoint-500"
    assert [p.name for p in list_checkpoints(tmp_path)] == [
        "checkpoint-100",
        "checkpoint-250",
        "checkpoint-500",
    ]


def test_resolve_resume_latest(tmp_path: Path):
    assert resolve_resume_path(tmp_path, None) is None
    assert resolve_resume_path(tmp_path, "latest") is None
    d = tmp_path / "checkpoints" / "checkpoint-42"
    d.mkdir(parents=True)
    (d / "trainer_state.json").write_text("{}", encoding="utf-8")
    resolved = resolve_resume_path(tmp_path, "latest")
    assert resolved is not None
    assert resolved.name == "checkpoint-42"


def test_resolve_explicit_checkpoint(tmp_path: Path):
    d = tmp_path / "checkpoints" / "checkpoint-7"
    d.mkdir(parents=True)
    (d / "trainer_state.json").write_text("{}", encoding="utf-8")
    assert resolve_resume_path(tmp_path, str(d)) == d
