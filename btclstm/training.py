"""Training utilities for the BTC LSTM predictor."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from .dataset import SequenceConfig, SequenceDataset
from .metrics.evaluation import directional_accuracy, mape, rmse
from .model import LSTMPredictor, ModelConfig


@dataclass
class TrainingConfig:
    sequence: SequenceConfig
    model: ModelConfig
    epochs: int = 10
    batch_size: int = 32
    lr: float = 1e-3
    train_split: float = 0.8


class Trainer:
    """High level helper that orchestrates model training."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LSTMPredictor(config.model).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)

    def _prepare_loaders(self, features: np.ndarray, targets: np.ndarray) -> Tuple[DataLoader, DataLoader]:
        dataset = SequenceDataset(features, targets, self.config.sequence)
        if len(dataset) < 2:
            raise ValueError(
                "Nedostatek dat pro vytvoření sekvencí. Zvyšte limit vstupních dat nebo snižte délku sekvence."
            )
        train_len = max(int(len(dataset) * self.config.train_split), 1)
        val_len = len(dataset) - train_len
        if val_len == 0:
            val_len = 1
            train_len -= 1
        train_dataset, val_dataset = random_split(dataset, [train_len, val_len])
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
        return train_loader, val_loader

    def fit(self, features: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        train_loader, val_loader = self._prepare_loaders(features, targets)
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(self.config.epochs):
            self.model.train()
            train_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                self.optimizer.zero_grad()
                preds = self.model(batch_x)
                loss = self.criterion(preds, batch_y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * batch_x.size(0)
            train_loss /= max(len(train_loader.dataset), 1)

            self.model.eval()
            val_loss = 0.0
            preds_list = []
            targets_list = []
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    preds = self.model(batch_x)
                    loss = self.criterion(preds, batch_y)
                    val_loss += loss.item() * batch_x.size(0)
                    preds_list.append(preds.cpu().numpy())
                    targets_list.append(batch_y.cpu().numpy())
            val_loss /= max(len(val_loader.dataset), 1)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

        if preds_list and targets_list:
            predictions = np.concatenate(preds_list)
            true_values = np.concatenate(targets_list)
            history.update(
                {
                    "rmse": rmse(true_values, predictions),
                    "mape": mape(true_values, predictions),
                    "directional_accuracy": directional_accuracy(true_values, predictions),
                }
            )
        return history


__all__ = ["Trainer", "TrainingConfig"]
