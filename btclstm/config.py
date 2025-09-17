"""Configuration loading utilities for the BTC LSTM project."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class Config:
    """Simple wrapper around the YAML configuration structure."""

    data: Dict[str, Any]
    training: Dict[str, Any]
    weights: Dict[str, Any]

    @classmethod
    def from_file(cls, path: Path | str) -> "Config":
        """Load configuration from the provided YAML file."""
        with Path(path).open("r", encoding="utf8") as handle:
            payload = yaml.safe_load(handle)
        return cls(
            data=payload.get("data", {}),
            training=payload.get("training", {}),
            weights=payload.get("weights", {}),
        )


__all__ = ["Config"]
