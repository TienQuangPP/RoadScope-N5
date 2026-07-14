"""Portable PyTorch checkpoint loading across Linux and Windows."""

from __future__ import annotations

import os
from pathlib import Path
import pathlib
from typing import Any

import torch


def load_checkpoint(path: str | Path, map_location: Any) -> dict[str, Any]:
    """Load checkpoints that may contain pathlib paths from another OS."""
    original_posix = pathlib.PosixPath
    original_windows = pathlib.WindowsPath
    if os.name == "nt":
        pathlib.PosixPath = pathlib.WindowsPath
    else:
        pathlib.WindowsPath = pathlib.PosixPath
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    finally:
        pathlib.PosixPath = original_posix
        pathlib.WindowsPath = original_windows
