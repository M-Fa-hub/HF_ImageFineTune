"""Reproducibility helpers."""

from __future__ import annotations

import contextlib
import os
import random
from typing import Any

import numpy as np


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            with contextlib.suppress(Exception):
                torch.use_deterministic_algorithms(True, warn_only=True)
        else:
            torch.backends.cudnn.benchmark = True
    except ImportError:
        pass


def capture_rng_states() -> dict[str, Any]:
    states: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    try:
        import torch

        states["torch"] = torch.get_rng_state()
        if torch.cuda.is_available():
            states["torch_cuda"] = torch.cuda.get_rng_state_all()
    except ImportError:
        pass
    return states


def restore_rng_states(states: dict[str, Any]) -> None:
    if "python" in states:
        random.setstate(states["python"])
    if "numpy" in states:
        np.random.set_state(states["numpy"])
    try:
        import torch

        if "torch" in states:
            torch.set_rng_state(states["torch"])
        if "torch_cuda" in states and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(states["torch_cuda"])
    except ImportError:
        pass
