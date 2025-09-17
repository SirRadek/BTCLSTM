"""Evaluation metrics for the BTC LSTM pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))))


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_dir = np.sign(np.diff(y_true, prepend=y_true[0]))
    pred_dir = np.sign(np.diff(y_pred, prepend=y_pred[0]))
    return float((true_dir == pred_dir).mean())


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
    excess = returns - risk_free
    if excess.std() == 0:
        return 0.0
    return float(np.sqrt(252) * excess.mean() / excess.std())


__all__ = ["rmse", "mape", "directional_accuracy", "sharpe_ratio"]
