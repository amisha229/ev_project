"""
EV Smart Charging Assistant - driver-facing Streamlit UI.

Reads ``storage/live_data.csv`` for live KPIs and ``data/dataset.csv`` for station cards.

Run from ``ev_project``::

    streamlit run app/user_dashboard.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

_APP_DIR = Path(__file__).resolve().parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

import live_io  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LIVE_CSV = ROOT / "storage" / "live_data.csv"
DATASET_CSV = ROOT / "data" / "dataset.csv"
RUNTIME_CONFIG = ROOT / "storage" / "runtime_config.json"


def _load_config() -> dict:
    defaults = {"high_demand_threshold": 2.5, "low_demand_threshold": 0.5}
    if not RUNTIME_CONFIG.is_file():
        return defaults
    try:
        data = json.loads(RUNTIME_CONFIG.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            defaults.update({k: data[k] for k in defaults if k in data})
    except (json.JSONDecodeError, OSError):
        pass
    return defaults


def _read_live() -> pd.DataFrame:
    return live_io.read_live_telemetry(LIVE_CSV)


def _station_key(value: object) -> str:
    if value is None:
        return ""
    if pd.isna(value):
        return ""
    return str(value)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p = math.pi / 180.0
    a = (
        math.sin((lat2 - lat1) * p / 2) ** 2
        + math.cos(lat1 * p) * math.cos(lat2 * p) * math.sin((lon2 - lon1) * p / 2) ** 2
    )
    return 2 * r * math.asin(math.sqrt(a))


def _station_board() -> pd.DataFrame:
    if not DATASET_CSV.is_file():
        return pd.DataFrame()

    raw = pd.read_csv(DATASET_CSV, encoding="utf-8")
    if "station_id" not in raw.columns or "lat" not in raw.columns or "lon" not in raw.columns:
        return pd.DataFrame()

    util = raw["utilization"] if "utilization" in raw.columns else raw.get("station_load", 0)
    raw = raw.assign(_util=pd.to_numeric(util, errors="coerce").fillna(0))

    def _first_nonnull(s: pd.Series) -> str:
        s = s.dropna()
        return str(s.iloc[0]) if len(s) else "Station"

    grouped = raw.groupby("station_id", as_index=False).agg(
        lat=("lat", "mean"),
        lon=("lon", "mean"),
        load=("_util", "mean"),
        name=("station_type", _first_nonnull),
    )
    ref_lat, ref_lon = 12.9716, 77.5946
    grouped["distance_km"] = grouped.apply(
        lambda row: _haversine_km(ref_lat, ref_lon, row["lat"], row["lon"]), axis=1
    )
    return grouped.sort_values("distance_km")


def _recommended_station_id(board: pd.DataFrame) -> str:
    if board.empty:
        return ""
    ranked = board.sort_values(["load", "distance_km"], ascending=[True, True])
    return _station_key(ranked.iloc[0]["station_id"])


def _station_map(board: pd.DataFrame, selected_station_id: str) -> go.Figure:
    center_lat = float(board["lat"].mean()) if not board.empty else 12.9716
    center_lon = float(board["lon"].mean()) if not board.empty else 77.5946

    fig = go.Figure()
    if not board.empty:
        load_pct = board["load"].copy()
        if load_pct.max() <= 1.5:
            load_pct = load_pct * 100.0
        board = board.assign(_load_pct=load_pct.clip(lower=0).fillna(0))

        selected = board[board["station_id"].astype(str) == selected_station_id]
        others = board[board["station_id"].astype(str) != selected_station_id]

        def _hover_text(df: pd.DataFrame) -> list[str]:
            return [
                f"Station {row['station_id']}<br>{row['name']}<br>Distance: {float(row['distance_km']):.1f} km<br>Load: {float(row['_load_pct']):.0f}%"
                for _, row in df.iterrows()
            ]

        if not others.empty:
            fig.add_trace(
                go.Scattermapbox(
                    lat=others["lat"],
                    lon=others["lon"],
                    mode="markers",
                    marker=dict(
                        size=(12 + (others["_load_pct"].clip(0, 100) / 12)).tolist(),
                        color=others["_load_pct"],
                        colorscale="Viridis",
                        cmin=0,
                        cmax=100,
                        opacity=0.72,
                    ),
                    text=_hover_text(others),
                    hoverinfo="text",
                    name="Stations",
                )
            )

        if not selected.empty:
            fig.add_trace(
                go.Scattermapbox(
                    lat=selected["lat"],
                    lon=selected["lon"],
                    mode="markers+text",
                    marker=dict(size=22, color="#ff7a00", opacity=0.95),
                    text=["Selected"],
                    textposition="top center",
                    hovertext=_hover_text(selected),
                    hoverinfo="text",
                    name="Selected station",
                )
            )

    fig.update_layout(
        mapbox=dict(style="open-street-map", center=dict(lat=center_lat, lon=center_lon), zoom=11.0),
        margin=dict(l=0, r=0, t=0, b=0),
        height=420,
        legend=dict(orientation="h", y=0.02, x=0.02),
    )
    return fig


def _station_snapshot(live_df: pd.DataFrame, station_id: str) -> pd.Series | None:
    if live_df.empty or not station_id or "station_id" not in live_df.columns:
        return None
    feed = live_df.copy()
    feed["station_id"] = feed["station_id"].astype(str)
    feed = feed[feed["station_id"] == station_id]
    if feed.empty:
        return None
    return feed.iloc[-1]


def _format_load(value: float) -> float:
    return min(100.0, value * 100.0) if value <= 1.5 else value


def main() -> None:
    st.set_page_config(
        page_title="EV Smart Charging",
        page_icon="EV",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st_autorefresh(interval=50000, key="user_refresh")

    st.markdown(
        """
        <style>
        :root {
            --bg: #f4f7fb;
            --panel-soft: #eef4ff;
            --text: #122033;
            --muted: #5f6f85;
            --line: #d9e2ef;
            --blue: #1f6feb;
            --teal: #0f9d8a;
            --orange: #ff7a00;
            --purple: #8b5cf6;
            --shadow: 0 18px 42px rgba(18, 32, 51, 0.08);
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(31, 111, 235, 0.08), transparent 28%),
                radial-gradient(circle at top right, rgba(15, 157, 138, 0.10), transparent 24%),
                linear-gradient(180deg, #f8fbff 0%, var(--bg) 52%, #eef3f9 100%);
            color: var(--text);
        }
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1.25rem;
        }
        .hero {
            background:
                linear-gradient(135deg, rgba(31, 111, 235, 0.12) 0%, rgba(15, 157, 138, 0.10) 58%, rgba(139, 92, 246, 0.08) 100%);
            padding: 1.3rem 1.4rem;
            border-radius: 24px;
            color: var(--text);
            margin-bottom: 0.95rem;
            box-shadow: 0 20px 40px rgba(31, 111, 235, 0.15);
            position: relative;
            overflow: hidden;
            border: 2px solid rgba(31, 111, 235, 0.25);
        }
        .hero::after {
            content: "";
            position: absolute;
            inset: auto -80px -120px auto;
            width: 280px;
            height: 280px;
            border-radius: 50%;
            background: rgba(31, 111, 235, 0.08);
            filter: blur(8px);
        }
        .hero h1 {
            margin: 0;
            font-size: 2.8rem;
            font-weight: 900;
            letter-spacing: -0.03em;
            background: linear-gradient(135deg, #1f6feb 0%, #0f9d8a 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            position: relative;
            z-index: 1;
            line-height: 1.1;
        }
        .hero p {
            margin: 0.4rem 0 0 0;
            opacity: 0.94;
            font-size: 0.95rem;
            max-width: 40rem;
            position: relative;
            z-index: 1;
        }
        .section-title {
            margin: 0.4rem 0 0.8rem 0;
            color: var(--text);
            font-size: 1.05rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            padding-left: 0.75rem;
            border-left: 4px solid var(--blue);
        }
        .kpi {
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.98) 0%, rgba(248, 251, 255, 0.98) 100%);
            border-radius: 20px;
            padding: 1.2rem 1.2rem 1.15rem;
            border: 1px solid var(--line);
            box-shadow: 0 18px 42px rgba(18, 32, 51, 0.10);
            min-height: 132px;
            border-top: 4px solid var(--blue);
            transition: all 0.2s ease;
        }
        .kpi:hover {
            transform: translateY(-2px);
            box-shadow: 0 22px 50px rgba(31, 111, 235, 0.18);
        }
        .kpi .eyebrow {
            display: inline-block;
            padding: 0.28rem 0.6rem;
            border-radius: 999px;
            background: rgba(31, 111, 235, 0.15);
            color: var(--blue);
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }
        .kpi .label {
            margin-top: 0.85rem;
            font-size: 0.82rem;
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.07em;
        }
        .kpi .value {
            font-size: 1.8rem;
            font-weight: 800;
            color: var(--text);
            margin-top: 0.25rem;
            letter-spacing: -0.03em;
        }
        .kpi .hint {
            margin-top: 0.35rem;
            color: var(--muted);
            font-size: 0.86rem;
        }
        .card {
            border-radius: 20px;
            padding: 1.05rem 1.1rem;
            border: 1px solid var(--line);
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.98) 0%, rgba(241, 247, 255, 0.98) 100%);
            margin-bottom: 0.8rem;
            box-shadow: 0 18px 42px rgba(31, 111, 235, 0.12);
            border-left: 4px solid var(--teal);
        }
        .card h4 {
            margin: 0 0 0.45rem 0;
            color: var(--text);
            font-size: 1rem;
            letter-spacing: -0.01em;
        }
        .card .pill {
            display: inline-block;
            margin-bottom: 0.65rem;
            padding: 0.24rem 0.55rem;
            border-radius: 999px;
            background: rgba(15, 157, 138, 0.12);
            color: var(--teal);
            font-size: 0.74rem;
            font-weight: 700;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }
        .muted {
            color: var(--muted);
            font-size: 0.9rem;
            margin: 0;
        }
        .station-meta {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.6rem;
            margin-top: 0.75rem;
        }
        .station-meta .mini {
            background: linear-gradient(180deg, rgba(247, 250, 255, 0.98) 0%, rgba(240, 245, 252, 0.98) 100%);
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 0.75rem 0.8rem;
        }
        .station-meta .mini .k {
            color: var(--muted);
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 0.25rem;
        }
        .station-meta .mini .v {
            color: var(--text);
            font-size: 1rem;
            font-weight: 800;
            letter-spacing: -0.02em;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero">
          <h1>EV Smart Charging Assistant</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cfg = _load_config()
    df = _read_live()
    if df.empty:
        st.warning("No live data yet. Start the engine: `python main.py --fresh --loop --sleep-ms 30`")
        st.stop()

    board = _station_board()
    recommended_station_id = _recommended_station_id(board)
    if "selected_station_id" not in st.session_state:
        st.session_state.selected_station_id = recommended_station_id
    if st.session_state.selected_station_id and not board.empty:
        available_ids = {str(v) for v in board["station_id"].tolist()}
        if st.session_state.selected_station_id not in available_ids:
            st.session_state.selected_station_id = recommended_station_id
    elif not st.session_state.selected_station_id:
        st.session_state.selected_station_id = recommended_station_id

    selected_station_id = _station_key(st.session_state.selected_station_id)
    selected_board = board[board["station_id"].astype(str) == selected_station_id] if not board.empty else pd.DataFrame()
    selected_snapshot = _station_snapshot(df, selected_station_id)

    if selected_snapshot is None:
        fallback_row = df.iloc[-1]
        price = float(fallback_row["price"]) if pd.notna(fallback_row["price"]) else 0.0
        pred = float(fallback_row["predicted_demand"]) if pd.notna(fallback_row.get("predicted_demand")) else None
        load = float(fallback_row["station_load"]) if pd.notna(fallback_row["station_load"]) else 0.0
        last_time = str(fallback_row.get("time", ""))
    else:
        price = float(selected_snapshot["price"]) if pd.notna(selected_snapshot["price"]) else 0.0
        pred = float(selected_snapshot["predicted_demand"]) if pd.notna(selected_snapshot.get("predicted_demand")) else None
        load = float(selected_snapshot["station_load"]) if pd.notna(selected_snapshot["station_load"]) else 0.0
        last_time = str(selected_snapshot.get("time", ""))

    station_name = "Station"
    station_distance = 0.0
    station_load = _format_load(load)
    if not selected_board.empty:
        chosen_row = selected_board.iloc[0]
        station_name = str(chosen_row["name"])
        station_distance = float(chosen_row["distance_km"])
        station_load = _format_load(float(chosen_row["load"]))
    elif not board.empty:
        chosen_row = board[board["station_id"].astype(str) == selected_station_id].iloc[0]
        station_name = str(chosen_row["name"])
        station_distance = float(chosen_row["distance_km"])
        station_load = _format_load(float(chosen_row["load"]))
    else:
        chosen_row = None

    left, right = st.columns([1.2, 1.0])
    with left:
        st.markdown('<div class="section-title">Nearby station map</div>', unsafe_allow_html=True)
        if board.empty:
            st.info("Add `station_id`, `lat`, and `lon` to `dataset.csv` to enable the map.")
        else:
            st.plotly_chart(_station_map(board, selected_station_id), width="stretch")
    with right:
        st.markdown('<div class="section-title">Choose your station</div>', unsafe_allow_html=True)
        if board.empty:
            st.info("Add station location data to enable selection.")
        else:
            options = [str(v) for v in board["station_id"].tolist()]
            label_map = {
                str(row["station_id"]): f"Hub {row['station_id']} - {row['name']} ({float(row['distance_km']):.1f} km)"
                for _, row in board.iterrows()
            }
            default_index = options.index(selected_station_id) if selected_station_id in options else 0
            chosen = st.selectbox(
                "Available stations",
                options=options,
                index=default_index,
                format_func=lambda sid: label_map.get(sid, sid),
                label_visibility="collapsed",
            )
            if chosen != st.session_state.selected_station_id:
                st.session_state.selected_station_id = chosen
                st.rerun()

            chosen_row = selected_board.iloc[0] if not selected_board.empty else board[board["station_id"].astype(str) == chosen].iloc[0]
            load_v = float(chosen_row["load"]) if pd.notna(chosen_row["load"]) else 0.0
            load_bar = _format_load(load_v)
            badge = "Recommended" if chosen == recommended_station_id else "Selected"
            st.markdown(
                f"""
                <div class="card" style="margin-top:0.9rem;margin-bottom:0;">
                  <div class="pill">{badge}</div>
                  <h4>Hub {chosen_row['station_id']} - {chosen_row['name']}</h4>
                  <p class="muted">{float(chosen_row['distance_km']):.1f} km away | Average load {load_bar:.0f}%</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown('<div class="section-title" style="margin-top:1rem;">Station summary</div>', unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class="station-meta" style="margin-top:0.25rem;">
                    <div class="mini"><div class="k">Station</div><div class="v">{chosen_row['station_id']}</div></div>
                    <div class="mini"><div class="k">Type</div><div class="v">{chosen_row['name']}</div></div>
                    <div class="mini"><div class="k">Distance</div><div class="v">{float(chosen_row['distance_km']):.1f} km</div></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f'<div class="kpi"><div class="eyebrow">Live</div><div class="label">Current price</div><div class="value">₹{price:.2f} / kWh</div><div class="hint">Adaptive MPC recommendation</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        pred_txt = f"{pred:.2f}" if pred is not None else "--"
        st.markdown(
            f'<div class="kpi"><div class="eyebrow">Model</div><div class="label">Predicted demand (vehicles/hour)</div><div class="value">{pred_txt}</div><div class="hint">Next-step expected charging activity</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="kpi"><div class="eyebrow">Telemetry</div><div class="label">Station load</div><div class="value">{station_load:.1f}%</div><div class="hint">Latest observed utilization</div></div>',
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f'<div class="kpi"><div class="eyebrow">Location</div><div class="label">Distance to station</div><div class="value">{station_distance:.1f} km</div><div class="hint">{station_name}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">Charging recommendation</div>', unsafe_allow_html=True)
    hi = float(cfg["high_demand_threshold"])
    lo = float(cfg["low_demand_threshold"])
    pred_for_rule = pred if pred is not None else float(selected_snapshot["actual_demand"] if selected_snapshot is not None and pd.notna(selected_snapshot.get("actual_demand")) else 0.0)

    if pred_for_rule >= hi:
        st.error(
            "⚠️ High demand detected — dynamic pricing is elevated to manage congestion. "
            "Consider off-peak charging or selecting a nearby station."
        )
    elif pred_for_rule <= lo:
        st.success(
            "✅ Low demand — optimal time for cost-efficient charging."
        )
    else:
        st.info("Demand is **moderate**. Standard pricing and availability expected.")
    selected_station_label = selected_station_id or "all stations"
    st.caption(f"Selected station: `{selected_station_label}` | Last update (UTC row timestamp): `{last_time}`")


if __name__ == "__main__":
    main()
