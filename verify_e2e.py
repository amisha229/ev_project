#!/usr/bin/env python3
"""
Non-UI smoke test: run the engine briefly, read telemetry, build a Plotly figure.

Usage (from ``ev_project``)::

    python verify_e2e.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent


def main() -> None:
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "main.py"),
            "--fresh",
            "--max-rows",
            "30",
            "--print-every",
            "100",
        ],
        cwd=ROOT,
        check=True,
    )

    sys.path.insert(0, str(ROOT / "app"))
    import live_io  # noqa: E402

    df = live_io.read_live_telemetry(ROOT / "storage" / "live_data.csv")
    if len(df) < 25:
        raise SystemExit(f"Expected >=25 telemetry rows, got {len(df)}")

    _ = go.Figure(
        go.Scatter(x=list(range(len(df))), y=df["actual_demand"], name="actual", mode="lines")
    )

    print("verify_e2e: OK (engine + live_io + plotly)", len(df), "rows")


if __name__ == "__main__":
    main()
