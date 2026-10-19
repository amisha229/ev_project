# Intelligent EV Charging — Demand Prediction & Dynamic Pricing

End-to-end demo: **streaming simulation**, **River ARF** online learning, **MPC-style** pricing, **ADWIN** drift monitoring, dual **Streamlit** dashboards (driver + operator), and **Plotly** analytics.

## Layout

```
ev_project/
├── data/dataset.csv
├── preprocessing/preprocess.py
├── streaming/stream_simulator.py
├── models/arf_model.py
├── pricing/mpc_model.py
├── evaluation/metrics.py
├── storage/live_data.csv          # appended by main.py
├── storage/runtime_config.json    # tuned from operator UI
├── app/user_dashboard.py
├── app/operator_dashboard.py
├── main.py
└── requirements.txt
```

## 1) Install

```bash
cd ev_project
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## Quick verification (no browser)

```bash
python verify_e2e.py
```

## 2) Run the real-time engine

From `ev_project` (this process **streams** the CSV and **appends** rows to `storage/live_data.csv`):

```bash
python main.py --fresh --max-rows 800 --sleep-ms 40 --print-every 50
```

- `--fresh` — reset `live_data.csv` (header only).
- `--sleep-ms` — delay between rows (feels more “real-time”).
- `--max-rows` — optional cap for quicker demos.
- Stop with `Ctrl+C` when finished.

**Note:** River’s adaptive RF is `river.forest.ARFRegressor` (Adaptive Random Forest for regression in current River releases).

## 3) Run the driver UI

Open a **second** terminal (same venv, same `ev_project` folder):

```bash
streamlit run app/user_dashboard.py
```

Auto-refreshes about every **5 seconds**. Shows KPI cards, alerts, and nearby hubs from `data/dataset.csv`.

## 4) Run the operator control center

Third terminal:

```bash
streamlit run app/operator_dashboard.py
```

Auto-refreshes about every **4 seconds**. Includes **Plotly** charts, telemetry table, and sliders that write `storage/runtime_config.json` (picked up by `main.py` on subsequent rows).

## Dataset

Place your CSV at `data/dataset.csv`. The preprocessor accepts the **canonical** columns

`hour, day, temperature, traffic, station_load, price, demand`

or **Bengaluru-style** aliases (`arrivals`, `price_per_kwh`, `utilization`, etc.). Optional: `station_id`, `lat`, `lon`, `station_type`, `slots` (capacity for MPC).

## Live storage schema

`storage/live_data.csv` columns:

`time, actual_demand, predicted_demand, price, station_load`

where `price` is the **MPC-optimal** price for that step.
