"""
Read ``storage/live_data.csv`` safely while ``main.py`` may be appending (Windows file locks).
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd


def read_live_telemetry(path: Path, *, min_bytes: int = 8, attempts: int = 12) -> pd.DataFrame:
    """
    Load the live CSV with retries. Returns an empty frame if the file is missing,
    too small, or unreadable after all attempts.
    """
    try:
        if not path.is_file() or path.stat().st_size < min_bytes:
            return pd.DataFrame()
    except OSError:
        return pd.DataFrame()

    delay = 0.04
    for i in range(attempts):
        try:
            df = pd.read_csv(path, encoding="utf-8")
            break
        except (PermissionError, OSError, pd.errors.EmptyDataError, UnicodeDecodeError):
            time.sleep(delay * (i + 1))
    else:
        return pd.DataFrame()

    if df.empty:
        return df

    for c in ("actual_demand", "predicted_demand", "price", "station_load"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df
