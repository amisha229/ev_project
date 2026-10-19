"""
Load ``data/dataset.csv``, clean rows, and build model features + optional station metadata.

Canonical model columns:
    hour, day, temperature, traffic, station_load, price, demand, capacity

Extra columns (when present in source) for multi-station UI:
    station_id, lat, lon, station_type
"""

from __future__ import annotations

import numbers
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Candidate source columns in preference order for each canonical field.
FIELD_CANDIDATES = {
    "hour": ["hour", "hr"],
    "day": ["day", "day_id", "day_index"],
    "temperature": ["temperature", "temp", "weather_temp"],
    "traffic": ["traffic", "traffic_index", "traffic_level", "service_rate"],
    "station_load": ["station_load", "utilization", "load", "occupancy"],
    "price": ["price", "price_per_kwh", "base_price", "tariff"],
    "demand": ["demand", "arrivals", "sessions", "transactions"],
    "capacity": ["capacity", "slots", "chargers"],
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_dataset_path() -> Path:
    return project_root() / "data" / "dataset.csv"


def load_raw_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def _pick_first_available(df: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def _apply_aliases(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for canonical, candidates in FIELD_CANDIDATES.items():
        if canonical in df.columns:
            continue
        source = _pick_first_available(df, candidates)
        if source is not None:
            df[canonical] = df[source]
    return df


def _derive_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "hour" in df.columns and "day" in df.columns:
        return df

    n = len(df)
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        if "hour" not in df.columns:
            df["hour"] = ts.dt.hour
        if "day" not in df.columns:
            df["day"] = ts.dt.dayofyear

    if "hour" not in df.columns:
        df["hour"] = np.arange(n, dtype=float) % 24
    if "day" not in df.columns:
        df["day"] = (np.arange(n, dtype=float) // 24) + 1
    return df


def _ensure_core_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features and safe fallbacks rather than crashing on sparse datasets."""
    df = df.copy()
    df = _derive_time_columns(df)

    if "demand" not in df.columns:
        df["demand"] = 0.0
    if "price" not in df.columns:
        base = pd.to_numeric(df.get("base_price", 7.0), errors="coerce")
        if isinstance(base, pd.Series):
            df["price"] = base.fillna(7.0)
        else:
            df["price"] = 7.0

    if "station_load" in df.columns:
        sl = pd.to_numeric(df["station_load"], errors="coerce").fillna(0.0)
        if sl.max(skipna=True) is not None and sl.max() <= 1.5:
            df["station_load"] = sl.clip(0, 1) * 100.0
        else:
            df["station_load"] = sl.clip(lower=0)
    else:
        if "demand" in df.columns and "capacity" in df.columns:
            cap = pd.to_numeric(df["capacity"], errors="coerce").replace(0, np.nan)
            dem = pd.to_numeric(df["demand"], errors="coerce").fillna(0.0)
            df["station_load"] = ((dem / cap).fillna(0.0) * 100.0).clip(0, 100)
        else:
            df["station_load"] = 0.0

    if "traffic" not in df.columns:
        if "energy_kwh" in df.columns:
            e = np.log1p(pd.to_numeric(df["energy_kwh"], errors="coerce").fillna(0.0).clip(lower=0))
        elif "service_rate" in df.columns:
            e = pd.to_numeric(df["service_rate"], errors="coerce").fillna(0.0)
        elif "demand" in df.columns:
            e = pd.to_numeric(df["demand"], errors="coerce").fillna(0.0)
        else:
            e = pd.Series(0.0, index=df.index)
        emin, emax = float(e.min()), float(e.max())
        df["traffic"] = 0.0 if emax <= emin else (e - emin) / (emax - emin) * 100.0

    if "temperature" not in df.columns:
        h = pd.to_numeric(df["hour"], errors="coerce").fillna(0.0).to_numpy()
        d = pd.to_numeric(df["day"], errors="coerce").fillna(1.0).to_numpy()
        base = 26.0 + 5.0 * np.sin(2 * np.pi * h / 24.0)
        seasonal = 2.0 * np.sin(2 * np.pi * d / 60.0)
        noise = np.random.default_rng(42).normal(0, 0.35, size=len(df))
        df["temperature"] = base + seasonal + noise

    if "capacity" in df.columns:
        df["capacity"] = pd.to_numeric(df["capacity"], errors="coerce").fillna(8.0).clip(lower=1.0)
    elif "slots" in df.columns:
        df["capacity"] = pd.to_numeric(df["slots"], errors="coerce").fillna(8.0).clip(lower=1.0)
    else:
        df["capacity"] = 8.0

    return df


def load_and_prepare(
    path: str | Path | None = None,
    sample_frac: float | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Load CSV and return a cleaned table ready for streaming.

    Parameters
    ----------
    path
        Defaults to ``data/dataset.csv`` under the project root.
    sample_frac
        Optional subsample fraction for quick demos.
    """
    path = Path(path) if path is not None else default_dataset_path()
    raw = load_raw_csv(path)
    df = _apply_aliases(raw)
    df = _ensure_core_features(df)

    core = [
        "hour",
        "day",
        "temperature",
        "traffic",
        "station_load",
        "price",
        "demand",
        "capacity",
    ]
    meta = [c for c in ("station_id", "lat", "lon", "station_type") if c in df.columns]
    out = df[core + meta].copy()

    for c in core:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna(subset=["hour", "day", "demand", "price", "station_load"])
    out["demand"] = out["demand"].clip(lower=0)
    out["price"] = out["price"].clip(lower=0.01)
    out["traffic"] = out["traffic"].fillna(0.0).clip(lower=0)
    out["temperature"] = out["temperature"].fillna(out["temperature"].median() if not out["temperature"].dropna().empty else 26.0)
    out["capacity"] = out["capacity"].fillna(8.0).clip(lower=1.0)
    out["station_load"] = out["station_load"].fillna(0.0).clip(lower=0)

    if sample_frac is not None:
        out = out.sample(frac=sample_frac, random_state=random_state)

    return out.reset_index(drop=True)


def row_to_features(row: dict[str, Any]) -> dict[str, float]:
    """Feature dict for River ``predict_one`` / ``learn_one`` (excludes target & capacity)."""
    keys = ["hour", "day", "temperature", "traffic", "station_load", "price"]
    return {k: float(row[k]) for k in keys}


def row_station_id(row: dict[str, Any]) -> str:
    """Stable string id for multi-station displays."""
    sid = row.get("station_id")
    if sid is None:
        return "unknown"
    try:
        if isinstance(sid, float) and np.isnan(sid):
            return "unknown"
    except TypeError:
        pass
    if isinstance(sid, numbers.Integral) or isinstance(sid, np.integer):
        return str(int(sid))
    if isinstance(sid, float) and sid.is_integer():
        return str(int(sid))
    return str(sid)
