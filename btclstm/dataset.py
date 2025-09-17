"""Dataset utilities for preparing sequences for the LSTM model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class SequenceConfig:
    sequence_length: int = 48
    prediction_horizon: int = 1


class SequenceDataset(Dataset):
    """Create sliding-window sequences from tabular data."""

    def __init__(self, features: np.ndarray, targets: np.ndarray, config: SequenceConfig):
        self.features = features
        self.targets = targets
        self.config = config

    def __len__(self) -> int:
        length = (
            len(self.features)
            - self.config.sequence_length
            - self.config.prediction_horizon
            + 1
        )
        return max(length, 0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx
        end = idx + self.config.sequence_length
        target_idx = end + self.config.prediction_horizon - 1
        x = self.features[start:end]
        y = self.targets[target_idx]
        return torch.from_numpy(x).float(), torch.tensor(y, dtype=torch.float32)


__all__ = ["SequenceDataset", "SequenceConfig"]
