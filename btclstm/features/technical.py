"""Collection of basic technical indicator utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_gain = up.rolling(window).mean()
    avg_loss = down.rolling(window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "macd_signal": signal_line, "macd_hist": histogram})


def bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    mid = sma(series, window)
    std = series.rolling(window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return pd.DataFrame({"bb_mid": mid, "bb_upper": upper, "bb_lower": lower})


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    high_low = high - low
    high_close = (high - close.shift()).abs()
    low_close = (low - close.shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(window).mean()


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def cmf(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    mf_multiplier = ((close - low) - (high - close)) / (high - low)
    mf_volume = mf_multiplier.fillna(0) * volume
    return mf_volume.rolling(window).sum() / volume.rolling(window).sum()


def add_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a dataframe augmented with a collection of indicators."""
    frame = frame.copy()
    frame["sma_20"] = sma(frame["close"], 20)
    frame["ema_12"] = ema(frame["close"], 12)
    frame["ema_26"] = ema(frame["close"], 26)
    frame["rsi_14"] = rsi(frame["close"], 14)
    frame["atr_14"] = atr(frame["high"], frame["low"], frame["close"], 14)
    frame["obv"] = obv(frame["close"], frame["volume"])
    frame["cmf_20"] = cmf(frame["high"], frame["low"], frame["close"], frame["volume"], 20)
    macd_df = macd(frame["close"])
    for column in macd_df:
        frame[column] = macd_df[column]
    bb_df = bollinger_bands(frame["close"])
    for column in bb_df:
        frame[column] = bb_df[column]
    return frame


__all__ = [
    "sma",
    "ema",
    "rsi",
    "macd",
    "bollinger_bands",
    "atr",
    "obv",
    "cmf",
    "add_indicators",
]
