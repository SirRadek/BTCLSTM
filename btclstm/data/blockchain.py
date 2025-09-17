"""Helpers for retrieving basic Blockchain.com on-chain metrics."""
from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional

import pandas as pd
import requests

BASE_URL = "https://api.blockchain.info/charts/{metric}?format=json&timespan={timespan}&sampled=false"


METRICS = {
    "transactions": "n-transactions",
    "addresses": "new_addresses",
}


def fetch_metric(
    metric: str,
    timespan: str = "30days",
    session: Optional[requests.Session] = None,
) -> pd.Series:
    """Fetch a single on-chain metric from Blockchain.com."""
    slug = METRICS.get(metric, metric)
    url = BASE_URL.format(metric=slug, timespan=timespan)
    sess = session or requests.Session()
    response = sess.get(url, timeout=10)
    response.raise_for_status()
    payload: Dict = response.json()
    values = payload.get("values", [])
    frame = pd.DataFrame(values)
    frame["x"] = pd.to_datetime(frame["x"], unit="s", utc=True)
    frame.set_index("x", inplace=True)
    frame.rename(columns={"y": metric}, inplace=True)
    return frame[metric].astype("float64")


def fetch_metrics(timespan: str = "30days") -> pd.DataFrame:
    """Convenience helper returning all supported metrics as a dataframe."""
    session = requests.Session()
    series = {name: fetch_metric(name, timespan=timespan, session=session) for name in METRICS}
    return pd.DataFrame(series)


__all__ = ["fetch_metric", "fetch_metrics", "METRICS"]
