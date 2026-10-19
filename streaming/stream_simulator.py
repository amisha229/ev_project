"""
Simulate a real-time EV charging stream.

Yields one dictionary per timestep. When the prepared DataFrame contains
``station_id``, each row is already one station-time record (multi-station stream).
"""

from __future__ import annotations

from typing import Any, Iterator

import pandas as pd


def _to_python(obj: Any) -> Any:
    """Make values JSON/CSV friendly (numpy scalars -> float)."""
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return obj


def stream_rows(df: pd.DataFrame) -> Iterator[dict[str, Any]]:
    """Generator over rows as plain dicts (preserves all columns)."""
    cols = list(df.columns)
    for _, row in df.iterrows():
        yield {c: _to_python(row[c]) for c in cols}
