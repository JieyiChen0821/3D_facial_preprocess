from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from face_preprocess.errors import ArtifactError


def require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - runtime dependent
        raise RuntimeError(
            "PyTorch is required for model inference; install the selected CPU/CUDA lock"
        ) from exc
    return torch


def load_checkpoint(path: str | Path, device: str) -> Mapping[str, Any]:
    """Load an already SHA-verified checkpoint using PyTorch's safe loader."""
    source = Path(path)
    if not source.is_file():
        raise ArtifactError(f"checkpoint does not exist: {source}")
    torch = require_torch()
    try:
        payload = torch.load(source, map_location=device, weights_only=True)
    except TypeError as exc:  # pragma: no cover - only unsupported PyTorch
        raise RuntimeError("PyTorch >=2.5 with weights_only support is required") from exc
    if not isinstance(payload, Mapping):
        raise ArtifactError(f"checkpoint payload must be a mapping: {source}")
    return payload


def require_checkpoint_keys(payload: Mapping[str, Any], keys: set[str], label: str) -> None:
    missing = sorted(keys - set(payload))
    if missing:
        raise ArtifactError(f"{label} checkpoint is missing keys: {', '.join(missing)}")

