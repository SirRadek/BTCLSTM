"""Entry point that demonstrates the BTC LSTM pipeline."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from btclstm.config import Config
from btclstm.data.binance import fetch_ohlcv
from btclstm.data.blockchain import fetch_metrics
from btclstm.features.technical import add_indicators
from btclstm.training import Trainer, TrainingConfig
from btclstm.dataset import SequenceConfig
from btclstm.model import ModelConfig


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)


def _load_data(config: Config, offline: bool = False) -> pd.DataFrame:
    data_cfg = config.data
    if offline:
        LOGGER.warning("Running in offline mode, generating synthetic data.")
        periods = data_cfg.get("limit", 500)
        idx = pd.date_range(end=pd.Timestamp.utcnow(), periods=periods, freq="H")
        frame = pd.DataFrame(
            {
                "open": np.random.rand(periods) * 20000 + 20000,
                "high": np.random.rand(periods) * 20000 + 20000,
                "low": np.random.rand(periods) * 20000 + 20000,
                "close": np.random.rand(periods) * 20000 + 20000,
                "volume": np.random.rand(periods) * 10,
                "quote_volume": np.random.rand(periods) * 10,
                "trades": np.random.randint(100, 1000, size=periods),
            },
            index=idx,
        )
        return frame

    try:
        frame = fetch_ohlcv(
            symbol=data_cfg.get("symbol", "BTCUSDT"),
            interval=data_cfg.get("interval", "1h"),
            limit=data_cfg.get("limit", 500),
        )
    except Exception as exc:  # pragma: no cover - network failure path
        LOGGER.error("Failed to download OHLCV data: %s", exc)
        if data_cfg.get("allow_offline", True):
            return _load_data(config, offline=True)
        raise

    try:
        metrics = fetch_metrics(timespan=data_cfg.get("timespan", "30days"))
        frame = frame.join(metrics, how="left").ffill()
    except Exception as exc:  # pragma: no cover - network failure path
        LOGGER.warning("Failed to download on-chain metrics: %s", exc)
    return frame


def _prepare_features(frame: pd.DataFrame, config: Config) -> pd.DataFrame:
    indicators = add_indicators(frame)
    indicators.sort_index(inplace=True)
    indicators = indicators.dropna()

    input_weights: Dict[str, float] = config.weights.get("inputs", {})
    for column, weight in input_weights.items():
        if column in indicators:
            indicators[column] = indicators[column] * float(weight)
    return indicators


def _prepare_targets(frame: pd.DataFrame, prediction_horizon: int) -> pd.Series:
    returns = frame["close"].pct_change(periods=prediction_horizon).shift(-prediction_horizon)
    return returns.dropna()


def main() -> None:
    parser = argparse.ArgumentParser(description="BTC price prediction using an LSTM model")
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    parser.add_argument("--offline", action="store_true", help="Use synthetic data instead of downloading")
    args = parser.parse_args()

    config = Config.from_file(args.config)
    LOGGER.info("Loading data...")
    data_frame = _load_data(config, offline=args.offline)
    LOGGER.info("Generating indicators...")
    features = _prepare_features(data_frame, config)

    seq_len = int(config.training.get("sequence_length", 48))
    pred_horizon = int(config.training.get("prediction_horizon", 1))
    features = features.iloc[:-pred_horizon]
    targets = _prepare_targets(data_frame, pred_horizon).loc[features.index]

    feature_array = features.to_numpy(dtype=np.float32)
    target_array = targets.to_numpy(dtype=np.float32)

    model_config = ModelConfig(
        input_size=feature_array.shape[1],
        hidden_size=int(config.training.get("hidden_size", 64)),
        num_layers=int(config.training.get("num_layers", 2)),
        dropout=float(config.training.get("dropout", 0.2)),
    )
    training_config = TrainingConfig(
        sequence=SequenceConfig(sequence_length=seq_len, prediction_horizon=pred_horizon),
        model=model_config,
        epochs=int(config.training.get("epochs", 5)),
        batch_size=int(config.training.get("batch_size", 32)),
        lr=float(config.training.get("lr", 1e-3)),
        train_split=float(config.training.get("train_split", 0.8)),
    )

    LOGGER.info("Training model on %s samples with %s features", len(target_array), feature_array.shape[1])
    trainer = Trainer(training_config)
    history = trainer.fit(feature_array, target_array)
    for key, value in history.items():
        LOGGER.info("%s: %s", key, value)


if __name__ == "__main__":
    main()
