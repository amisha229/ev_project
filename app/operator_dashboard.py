"""
EV Charging Control Center - operator analytics (Streamlit + Plotly).

Reads ``storage/live_data.csv`` and can tune ``storage/runtime_config.json``.

Run from ``ev_project``::

    streamlit run app/operator_dashboard.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
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
RUNTIME_CONFIG = ROOT / "storage" / "runtime_config.json"
RUNTIME_STATE = ROOT / "storage" / "runtime_state.json"


def _default_cfg() -> dict:
    return {
        "lambda_mpc": 0.12,
        "capacity_override": None,
        "high_demand_threshold": 2.5,
        "low_demand_threshold": 0.5,
    }


def _load_cfg() -> dict:
    cfg = _default_cfg()
    if RUNTIME_CONFIG.is_file():
        try:
            data = json.loads(RUNTIME_CONFIG.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                cfg.update(data)
        except (json.JSONDecodeError, OSError):
            pass
    return cfg


def _save_cfg(cfg: dict) -> None:
    RUNTIME_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    RUNTIME_CONFIG.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def _read_live() -> pd.DataFrame:
    return live_io.read_live_telemetry(LIVE_CSV)


def _load_runtime_state() -> dict:
    if not RUNTIME_STATE.is_file():
        return {"drift_events": 0, "drift_detected": False}
    try:
        data = json.loads(RUNTIME_STATE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {"drift_events": 0, "drift_detected": False}
    except (json.JSONDecodeError, OSError):
        return {"drift_events": 0, "drift_detected": False}


def _impact_metrics(df: pd.DataFrame, capacity: float) -> tuple[float, float]:
    """
    Approximate demo metrics:
    - baseline overload uses predicted demand as the no-control pressure signal
    - observed overload uses actual demand after pricing decisions
    """
    if df.empty or capacity <= 0:
        return 0.0, 100.0

    baseline = int((df["predicted_demand"].fillna(df["actual_demand"]) > capacity).sum())
    observed = int((df["actual_demand"] > capacity).sum())
    if baseline <= 0:
        congestion_reduction = 100.0 if observed == 0 else 0.0
    else:
        congestion_reduction = max(0.0, (baseline - observed) / baseline * 100.0)

    load_std = float(df["station_load"].fillna(0.0).std(ddof=0))
    stability = max(0.0, min(100.0, 100.0 - load_std))
    return congestion_reduction, stability


def _station_health(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "station_id" not in df.columns:
        return pd.DataFrame()

    frame = df.copy()
    frame["station_id"] = frame["station_id"].astype(str)
    summary = frame.groupby("station_id", as_index=False).agg(
        avg_demand=("actual_demand", "mean"),
        avg_predicted=("predicted_demand", "mean"),
        avg_price=("price", "mean"),
        avg_load=("station_load", "mean"),
        rows=("station_id", "size"),
    )
    for column in ("avg_demand", "avg_predicted", "avg_price", "avg_load"):
        summary[column] = pd.to_numeric(summary[column], errors="coerce").fillna(0.0)
    return summary.sort_values(["avg_load", "avg_price"], ascending=[False, False])


def main() -> None:
    st.set_page_config(page_title="EV Control Center", page_icon="Ops", layout="wide")
    st_autorefresh(interval=49000, key="operator_refresh")

    st.markdown(
        """
        <style>
        :root {
            --bg: #f4f7fb;
            --text: #142235;
            --muted: #607187;
            --line: #d8e2ef;
            --panel: rgba(255, 255, 255, 0.96);
            --accent: #1f6feb;
            --accent-dark: #174ea6;
            --teal: #0f9d8a;
            --orange: #ff7a00;
            --purple: #8b5cf6;
            --shadow: 0 18px 42px rgba(20, 34, 53, 0.08);
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(18, 59, 100, 0.08), transparent 24%),
                radial-gradient(circle at top right, rgba(15, 157, 138, 0.08), transparent 22%),
                linear-gradient(180deg, #f9fbfe 0%, var(--bg) 56%, #edf2f8 100%);
            color: var(--text);
        }
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1.35rem;
        }
        .stApp,
        .stApp p,
        .stApp label,
        .stApp span,
        .stApp div,
        .stMarkdown,
        .stCaption {
            color: var(--text);
        }
        .shell {
            background: linear-gradient(135deg, rgba(31, 111, 235, 0.12) 0%, rgba(15, 157, 138, 0.10) 58%, rgba(139, 92, 246, 0.08) 100%);
            border-radius: 24px;
            padding: 1.3rem 1.4rem;
            margin-bottom: 0.95rem;
            border: 2px solid rgba(31, 111, 235, 0.25);
            box-shadow: 0 20px 40px rgba(31, 111, 235, 0.15);
            position: relative;
            overflow: hidden;
        }
        .shell::after {
            content: "";
            position: absolute;
            inset: auto -70px -90px auto;
            width: 220px;
            height: 220px;
            border-radius: 50%;
            background: rgba(31, 111, 235, 0.08);
            filter: blur(8px);
        }
        .big-title {
            font-size: 2.8rem;
            font-weight: 900;
            background: linear-gradient(135deg, #1f6feb 0%, #0f9d8a 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin: 0;
            letter-spacing: -0.03em;
            line-height: 1.1;
            position: relative;
            z-index: 1;
        }
        .sub {
            color: var(--muted);
            margin: 0.35rem 0 0 0;
            max-width: 42rem;
            position: relative;
            z-index: 1;
        }
        .section-title {
            margin: 0.15rem 0 0.75rem 0;
            color: var(--text);
            font-size: 1.04rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            padding-left: 0.7rem;
            border-left: 4px solid var(--accent);
            background: linear-gradient(90deg, rgba(31, 111, 235, 0.06) 0%, transparent 80%);
            padding-top: 0.5rem;
            padding-bottom: 0.5rem;
            padding-right: 0.7rem;
            border-radius: 6px;
        }
        div[data-testid="stMetric"] {
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.98) 0%, rgba(248, 251, 255, 0.98) 100%);
            padding: 0.95rem;
            border-radius: 18px;
            border: 1px solid var(--line);
            box-shadow: 0 14px 28px rgba(31, 111, 235, 0.12);
            border-top: 4px solid var(--accent);
            transition: all 0.2s ease;
        }
        div[data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 22px 42px rgba(31, 111, 235, 0.18);
        }
        div[data-testid="stMetricLabel"] {
            color: var(--muted);
            font-weight: 600;
        }
        div[data-testid="stMetricValue"] {
            color: var(--text);
            font-weight: 800;
        }
        div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
            color: var(--text);
        }
        div[data-testid="stForm"] {
            background: rgba(255, 255, 255, 0.86);
            border: 1px solid var(--line);
            border-radius: 22px;
            padding: 1rem 1rem 0.4rem 1rem;
            box-shadow: var(--shadow);
        }
        div[data-testid="stForm"] label,
        div[data-testid="stForm"] p,
        div[data-testid="stForm"] span,
        div[data-testid="stForm"] div {
            color: var(--text);
        }
        div[data-testid="stSlider"] [data-baseweb="slider"] {
            padding-top: 0.25rem;
            padding-bottom: 0.25rem;
        }
        div[data-testid="stSlider"] [role="slider"] {
            background: var(--accent);
            border-color: var(--accent);
        }
        div[data-testid="stSlider"] [data-testid="stTickBar"] {
            background: #cfd9e7;
        }
        div[data-testid="stNumberInput"] input,
        div[data-testid="stTextInput"] input,
        div[data-baseweb="select"] > div {
            background: var(--panel);
            color: var(--text);
            border-color: var(--line);
        }
        div[data-testid="stNumberInput"] button {
            color: var(--text);
        }
        div[data-testid="stFormSubmitButton"] button,
        .stButton > button {
            background: linear-gradient(180deg, var(--accent) 0%, var(--accent-dark) 100%);
            color: #ffffff;
            border: none;
            border-radius: 12px;
            font-weight: 700;
            box-shadow: 0 10px 20px rgba(31, 111, 235, 0.22);
        }
        div[data-testid="stFormSubmitButton"] button:hover,
        .stButton > button:hover {
            background: linear-gradient(180deg, #2a79f0 0%, #1a56bd 100%);
            color: #ffffff;
        }
        div[data-testid="stFormSubmitButton"] button p,
        .stButton > button p {
            color: #ffffff;
        }
        div[data-testid="stDataFrame"] {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 18px;
            box-shadow: var(--shadow);
        }
        .status-row {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.7rem;
            margin: 0.85rem 0 0.35rem 0;
        }
        .status-chip {
            background: linear-gradient(135deg, rgba(31, 111, 235, 0.08) 0%, rgba(15, 157, 138, 0.06) 100%);
            border: 1.5px solid rgba(31, 111, 235, 0.25);
            border-radius: 16px;
            padding: 0.75rem 0.85rem;
            box-shadow: 0 12px 24px rgba(31, 111, 235, 0.10);
            transition: all 0.2s ease;
        }
        .status-chip:hover {
            transform: translateY(-1px);
            box-shadow: 0 16px 32px rgba(31, 111, 235, 0.15);
        }
        .status-chip .k {
            color: var(--muted);
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.07em;
        }
        .status-chip .v {
            color: var(--text);
            font-size: 1rem;
            font-weight: 800;
            margin-top: 0.2rem;
        }
        iframe[title^="st.iframe"] {
            border-radius: 18px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="shell">
          <p class="big-title">EV Charging Control Center</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df = _read_live()
    if df.empty:
        st.warning("No telemetry yet. Run `python main.py --fresh --loop --sleep-ms 25` in a terminal.")
        st.stop()

    runtime_state = _load_runtime_state()
    cfg = _load_cfg()
    tail = df.tail(500).reset_index(drop=True)
    mae_series = (tail["actual_demand"] - tail["predicted_demand"]).abs()
    mae = float(mae_series.dropna().mean()) if mae_series.notna().any() else 0.0
    capacity = float(cfg.get("capacity_override") or runtime_state.get("capacity") or 8.0)
    congestion_reduction, stability_score = _impact_metrics(tail, capacity)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Avg demand (vehicles/hour)", f"{np.nanmean(tail['actual_demand'].to_numpy(dtype=float)):.3f}")
    m2.metric("Avg MPC price (₹/kWh)", f"Rs {np.nanmean(tail['price'].to_numpy(dtype=float)):.2f}")
    m3.metric("MAE (actual vs pred)", f"{mae:.4f}")
    m4.metric("Congestion reduction (%)", f"{congestion_reduction:.1f}")
    m5.metric("Load stability index", f"{stability_score:.1f}")
    st.caption("Lower MAE indicates better prediction accuracy")

    d1, d2, d3 = st.columns(3)
    d1.metric("Drift events", int(runtime_state.get("drift_events", 0)))
    d2.metric("Active capacity", f"{capacity:.1f}")
    d3.metric("Recent rows", len(tail))

    st.markdown(
        f"""
        <div class="status-row">
            <div class="status-chip"><div class="k">System status</div><div class="v">Live telemetry active</div></div>
            <div class="status-chip"><div class="k">Control mode</div><div class="v">MPC pricing enabled</div></div>
            <div class="status-chip"><div class="k">Attention</div><div class="v">{('Drift detected' if runtime_state.get('drift_detected') else 'Stable regime')}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if runtime_state.get("drift_detected"):
        st.warning("Concept Drift Detected: prediction behavior changed recently. Consider reviewing thresholds or data regime.")

    st.markdown("### System Flow")
    st.markdown(
        "`Streaming Data` -> `ARF Prediction` -> `MPC Pricing` -> `Dashboard` -> `Feedback Loop`"
    )

    health = _station_health(tail)
    if not health.empty:
        st.markdown("### Station health")
        st.caption("Stations are ranked by load so busy sites are visible first.")
        h1, h2 = st.columns([1.15, 0.85])
        with h1:
            fig3 = go.Figure()
            fig3.add_trace(
                go.Bar(
                    x=health["station_id"],
                    y=health["avg_load"],
                    marker_color="#1f6feb",
                    name="Average load",
                )
            )
            fig3.update_layout(
                title=dict(text="Average station load", font=dict(size=20, color="#0f1f33"), x=0.03),
                template="plotly_white",
                height=360,
                margin=dict(l=40, r=20, t=50, b=40),
                paper_bgcolor="rgba(255,255,255,0.96)",
                plot_bgcolor="#f7faff",
                font=dict(color="#142235"),
            )
            fig3.update_xaxes(title="Station ID", tickangle=-35, gridcolor="#dfe7f2")
            fig3.update_yaxes(title="Load", gridcolor="#dfe7f2")
            st.plotly_chart(fig3, width="stretch")
        with h2:
            st.dataframe(
                health[["station_id", "avg_load", "avg_price", "avg_demand", "avg_predicted", "rows"]],
                width="stretch",
                height=360,
            )

    st.markdown("---")
    c1, c2 = st.columns(2)
    idx = list(range(len(tail)))

    with c1:
        fig1 = go.Figure()
        fig1.add_trace(
            go.Scatter(
                x=idx,
                y=tail["actual_demand"],
                mode="lines",
                name="Actual demand",
                line=dict(color="#1f6feb", width=2.5, shape="linear"),
            )
        )
        fig1.add_trace(
            go.Scatter(
                x=idx,
                y=tail["predicted_demand"],
                mode="lines",
                name="Predicted demand",
                line=dict(color="#c97a12", width=2.5, dash="dot", shape="linear"),
            )
        )
        fig1.update_layout(
            title=dict(text="Demand vs Predicted", font=dict(size=20, color="#0f1f33"), x=0.03),
            template="plotly_white",
            hovermode="x unified",
            height=400,
            margin=dict(l=40, r=20, t=50, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            paper_bgcolor="rgba(255,255,255,0.96)",
            plot_bgcolor="#f7faff",
            font=dict(color="#142235"),
        )
        fig1.update_xaxes(title="Stream step (recent window)", gridcolor="#dfe7f2", zeroline=False)
        fig1.update_yaxes(title="Demand", gridcolor="#dfe7f2", zeroline=False)
        st.plotly_chart(fig1, width="stretch")

    with c2:
        fig2 = go.Figure()
        fig2.add_trace(
            go.Scatter(
                x=idx,
                y=tail["price"],
                mode="lines",
                name="MPC price",
                fill="tozeroy",
                fillcolor="rgba(15, 157, 138, 0.14)",
                line=dict(color="#0f9d8a", width=2.5, shape="linear"),
            )
        )
        fig2.update_layout(
            title=dict(text="Dynamic Price Trend", font=dict(size=20, color="#0f1f33"), x=0.03),
            template="plotly_white",
            height=400,
            margin=dict(l=40, r=20, t=50, b=40),
            paper_bgcolor="rgba(255,255,255,0.96)",
            plot_bgcolor="#f7faff",
            font=dict(color="#142235"),
        )
        fig2.update_xaxes(title="Stream step (recent window)", gridcolor="#dfe7f2", zeroline=False)
        fig2.update_yaxes(title="Price (Rs)", gridcolor="#dfe7f2", zeroline=False)
        st.plotly_chart(fig2, width="stretch")

    st.markdown('<div class="section-title">Live telemetry</div>', unsafe_allow_html=True)
    st.dataframe(df.tail(40), width="stretch", height=360)

    st.markdown("---")
    st.markdown('<div class="section-title">Simulation controls</div>', unsafe_allow_html=True)

    with st.form("ctrl"):
        lc, rc = st.columns(2)
        lam = lc.slider("MPC lambda (congestion penalty)", 0.01, 0.6, float(cfg.get("lambda_mpc", 0.12)), 0.01)
        cap_in = rc.number_input(
            "Capacity override (slots, 0 = use dataset per row)",
            min_value=0,
            max_value=64,
            value=int(cfg["capacity_override"]) if cfg.get("capacity_override") else 0,
        )
        h1, h2 = st.columns(2)
        hi = h1.slider("User UI: high demand alert >=", 0.5, 8.0, float(cfg.get("high_demand_threshold", 2.5)))
        lo = h2.slider("User UI: low demand alert <=", 0.0, 5.0, float(cfg.get("low_demand_threshold", 0.5)))
        submitted = st.form_submit_button("Apply settings")
        if submitted:
            new_cfg = _default_cfg()
            new_cfg.update(cfg)
            new_cfg["lambda_mpc"] = lam
            new_cfg["capacity_override"] = None if cap_in <= 0 else float(cap_in)
            new_cfg["high_demand_threshold"] = max(float(hi), float(lo))
            new_cfg["low_demand_threshold"] = min(float(hi), float(lo))
            _save_cfg(new_cfg)
            if float(lo) > float(hi):
                st.warning("Thresholds were normalized so low-demand stays below high-demand.")
            st.success("Saved. Running `main.py` picks this up on the next row.")


if __name__ == "__main__":
    main()
