"""Utility functions for retrieving OHLCV data from the Binance REST API."""
from __future__ import annotations

from datetime import datetime
from typing import Iterable, List, Optional

import pandas as pd
import requests

BASE_URL = "https://api.binance.com/api/v3/klines"


def fetch_ohlcv(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    limit: int = 500,
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    """Download OHLCV candles from Binance."""
    params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
    sess = session or requests.Session()
    response = sess.get(BASE_URL, params=params, timeout=10)
    response.raise_for_status()
    raw_candles: Iterable[List] = response.json()

    frame = pd.DataFrame(
        raw_candles,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "trades",
            "taker_base_volume",
            "taker_quote_volume",
            "ignore",
        ],
    )
    frame = frame.astype(
        {
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "volume": "float64",
            "quote_volume": "float64",
            "taker_base_volume": "float64",
            "taker_quote_volume": "float64",
            "trades": "int64",
        }
    )
    frame["open_time"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
    frame["close_time"] = pd.to_datetime(frame["close_time"], unit="ms", utc=True)
    frame.set_index("close_time", inplace=True)
    frame.sort_index(inplace=True)
    return frame[["open", "high", "low", "close", "volume", "quote_volume", "trades"]]


__all__ = ["fetch_ohlcv"]
