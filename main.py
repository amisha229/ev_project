"""
Real-time EV charging engine: stream -> ARF -> MPC -> MAE -> persist.

Run from the ``ev_project`` directory::

    python main.py
    python main.py --max-rows 500 --sleep-ms 50 --fresh

Dashboards read ``storage/live_data.csv`` while this process runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# -----------------------------------------------------------------------------
# Path setup: ``ev_project`` is the working directory for imports
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from river.drift import ADWIN

from evaluation.metrics import mae_value, new_mae, update_mae
from models.arf_model import build_model, learn_step, predict_safe
from preprocessing.preprocess import default_dataset_path, load_and_prepare, row_to_features
from pricing.mpc_model import OnlineDemandCurve, optimal_price_grid
from streaming.stream_simulator import stream_rows

STORAGE_COLUMNS = [
    "time",
    "station_id",
    "station_type",
    "actual_demand",
    "predicted_demand",
    "price",
    "station_load",
]
LIVE_CSV = ROOT / "storage" / "live_data.csv"
RUNTIME_CONFIG = ROOT / "storage" / "runtime_config.json"
RUNTIME_STATE = ROOT / "storage" / "runtime_state.json"


def _default_runtime() -> dict:
    return {
        "lambda_mpc": 0.12,
        "capacity_override": None,
        "high_demand_threshold": 2.5,
        "low_demand_threshold": 0.5,
    }


def load_runtime_config() -> dict:
    cfg = _default_runtime()
    if not RUNTIME_CONFIG.is_file():
        return cfg
    try:
        data = json.loads(RUNTIME_CONFIG.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            cfg.update({k: data[k] for k in cfg if k in data})
            # allow extra keys from file
            for k, v in data.items():
                if k not in cfg:
                    cfg[k] = v
    except (json.JSONDecodeError, OSError):
        pass
    return cfg


def reset_live_csv() -> None:
    LIVE_CSV.parent.mkdir(parents=True, exist_ok=True)
    with LIVE_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=STORAGE_COLUMNS)
        w.writeheader()


def append_live_row(
    *,
    station_id: str,
    station_type: str,
    actual_demand: float,
    predicted_demand: float | None,
    price: float,
    station_load: float,
) -> None:
    LIVE_CSV.parent.mkdir(parents=True, exist_ok=True)
    pred_val = "" if predicted_demand is None else round(float(predicted_demand), 4)
    row = {
        "time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "station_id": station_id,
        "station_type": station_type,
        "actual_demand": round(float(actual_demand), 4),
        "predicted_demand": pred_val,
        "price": round(float(price), 4),
        "station_load": round(float(station_load), 4),
    }
    write_header = not LIVE_CSV.exists() or LIVE_CSV.stat().st_size == 0
    with LIVE_CSV.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=STORAGE_COLUMNS)
        if write_header:
            w.writeheader()
        w.writerow(row)
        f.flush()


def ensure_live_csv_schema() -> None:
    """
    Keep the live telemetry file aligned with the dashboard contract even if an old
    file is present from a previous run.
    """
    if not LIVE_CSV.exists():
        reset_live_csv()
        return
    try:
        with LIVE_CSV.open("r", newline="", encoding="utf-8") as f:
            header = f.readline().strip().split(",")
    except OSError:
        reset_live_csv()
        return
    if header != STORAGE_COLUMNS:
        reset_live_csv()


def write_runtime_state(
    *,
    step: int,
    cycle: int,
    mae: float,
    drift_events: int,
    drift_detected: bool,
    capacity: float,
    predicted_demand: float,
    actual_demand: float,
    price: float,
) -> None:
    payload = {
        "step": int(step),
        "cycle": int(cycle),
        "mae": round(float(mae), 6),
        "drift_events": int(drift_events),
        "drift_detected": bool(drift_detected),
        "last_drift_time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC") if drift_detected else None,
        "capacity": round(float(capacity), 4),
        "predicted_demand": round(float(predicted_demand), 4),
        "actual_demand": round(float(actual_demand), 4),
        "price": round(float(price), 4),
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }
    RUNTIME_STATE.parent.mkdir(parents=True, exist_ok=True)
    RUNTIME_STATE.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_engine(
    *,
    dataset_path: Path | None,
    max_rows: int | None,
    sleep_ms: float,
    fresh: bool,
    print_every: int,
    loop: bool,
) -> None:
    if fresh:
        reset_live_csv()
    else:
        ensure_live_csv_schema()

    path = dataset_path or default_dataset_path()
    df = load_and_prepare(path)
    if max_rows is not None:
        df = df.head(int(max_rows)).reset_index(drop=True)

    model = build_model(seed=42, n_models=8, grace_period=40, with_scaler=True)
    mae = new_mae()
    curve = OnlineDemandCurve(maxlen=500, min_samples=35)
    adwin = ADWIN(delta=0.002, grace_period=12)

    print("EV engine started - streaming rows (Ctrl+C to stop)")
    print("live_data:", LIVE_CSV.resolve())
    print("-" * 88)

    step = 0
    cycle = 0
    drift_events = 0

    while True:
        cycle += 1
        if loop and cycle > 1:
            print(f"Restarting dataset from row 1 (cycle {cycle})")

        for row in stream_rows(df):
            step += 1
            cfg = load_runtime_config()
            lambda_mpc = float(cfg.get("lambda_mpc", 0.12))
            cap_override = cfg.get("capacity_override")
            capacity = float(cap_override) if cap_override is not None else float(row["capacity"])
            station_id = "" if row.get("station_id") is None else str(row.get("station_id"))
            station_type = "" if row.get("station_type") is None else str(row.get("station_type"))

            x = row_to_features(row)
            y_actual = float(row["demand"])
            price_obs = float(row["price"])
            load_pct = float(row["station_load"])

            pred = predict_safe(model, x)
            learn_step(model, x, y_actual)

            curve.update(price_obs, y_actual)
            predicted_demand = max(float(pred) if pred is not None else y_actual, 0.0)
            p_min = max(0.5, price_obs * 0.25)
            p_max = max(12.0, price_obs * 2.0)
            p_star, _, _ = optimal_price_grid(
                predicted_demand,
                curve.beta,
                capacity=capacity,
                lambda_penalty=lambda_mpc,
                p_min=p_min,
                p_max=p_max,
                n_steps=350,
            )

            if pred is not None:
                update_mae(mae, y_actual, pred)
                adwin.update(abs(y_actual - pred))

            append_live_row(
                station_id=station_id,
                station_type=station_type,
                actual_demand=y_actual,
                predicted_demand=pred,
                price=p_star,
                station_load=load_pct,
            )

            mae_v = mae_value(mae)
            drift = bool(adwin.drift_detected) if pred is not None else False
            if drift:
                drift_events += 1
            write_runtime_state(
                step=step,
                cycle=cycle,
                mae=mae_v,
                drift_events=drift_events,
                drift_detected=drift,
                capacity=capacity,
                predicted_demand=predicted_demand,
                actual_demand=y_actual,
                price=p_star,
            )

            if print_every <= 1 or step % print_every == 0:
                ps = f"{pred:7.3f}" if pred is not None else "    n/a"
                drift_note = " DRIFT" if drift else ""
                print(
                    f"{step:5d} | cycle {cycle:3d} | actual {y_actual:5.2f} | pred {ps} | "
                    f"mpc_price {p_star:6.2f} | load% {load_pct:5.1f} | "
                    f"MAE {mae_v:6.4f} | drift {'Y' if drift else 'n'}{drift_note}"
                )

            if sleep_ms > 0:
                time.sleep(sleep_ms / 1000.0)

        if not loop:
            break


def main() -> None:
    p = argparse.ArgumentParser(description="EV charging real-time engine")
    p.add_argument("--dataset", type=Path, default=None, help="CSV path (default data/dataset.csv)")
    p.add_argument("--max-rows", type=int, default=None, help="Stop after N rows (default: full file)")
    p.add_argument("--sleep-ms", type=float, default=0.0, help="Pause between rows (simulated latency)")
    p.add_argument("--fresh", action="store_true", help="Clear live_data.csv before starting")
    p.add_argument("--print-every", type=int, default=25, help="Log every N rows")
    loop_group = p.add_mutually_exclusive_group()
    loop_group.add_argument("--loop", dest="loop", action="store_true", help="Replay the dataset continuously until stopped")
    loop_group.add_argument("--no-loop", dest="loop", action="store_false", help="Run the dataset once and stop")
    p.set_defaults(loop=True)
    args = p.parse_args()

    try:
        run_engine(
            dataset_path=args.dataset,
            max_rows=args.max_rows,
            sleep_ms=args.sleep_ms,
            fresh=args.fresh,
            print_every=args.print_every,
            loop=args.loop,
        )
    except KeyboardInterrupt:
        print("\nStopped by user.")


if __name__ == "__main__":
    main()
