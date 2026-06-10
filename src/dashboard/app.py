"""
Streamlit monitoring dashboard for real-time fault prediction.

Features:
  - Live signal plots (rolling 5-second window)
  - Fault probability gauges per class (green / yellow / red)
  - Alert log with timestamp, fault type, confidence, contributing features
  - Model switcher (RF / XGBoost / NN)
  - Historical fault trend chart
  - Batch CSV analysis mode

Launch:
    streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import get_config
from src.inference.predictor import RealTimePredictor, PredictionResult

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Power Grid Fault Monitor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

CFG = get_config("config/config.yaml")
MODELS_DIR = PROJECT_ROOT / CFG["paths"]["models_out"]
SYNTHETIC_DIR = PROJECT_ROOT / CFG["paths"]["synthetic_data"]
DASH_CFG = CFG["dashboard"]
INF_CFG = CFG["inference"]

FAULT_NAMES = ["Normal", "SLG", "LL", "3PH", "Overload", "VoltageSag", "Harmonic"]
CHANNEL_NAMES = ["Va", "Vb", "Vc", "Ia", "Ib", "Ic", "Freq", "PF", "THD_V", "In"]

YELLOW_THRESH = INF_CFG["fault_prob_yellow_threshold"]
RED_THRESH = INF_CFG["fault_prob_red_threshold"]
ROLLING_WINDOW_S = DASH_CFG["rolling_window_seconds"]
FS = CFG["data_generation"]["sample_rate"]
ROLLING_N = ROLLING_WINDOW_S * FS   # samples in rolling display window

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------

def _init_state():
    defaults = {
        "signal_buffer": pd.DataFrame(columns=CHANNEL_NAMES),
        "alert_log": [],
        "fault_counts": {name: 0 for name in FAULT_NAMES},
        "predictor": None,
        "loaded_model": None,
        "sim_step": 0,
        "running": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ---------------------------------------------------------------------------
# Helper: load predictor
# ---------------------------------------------------------------------------

def _load_predictor(model_name: str) -> Optional[RealTimePredictor]:
    try:
        predictor = RealTimePredictor(CFG, model_name=model_name)
        predictor.load_model(model_name, MODELS_DIR)
        return predictor
    except Exception as e:
        st.error(f"Failed to load model '{model_name}': {e}")
        return None

# ---------------------------------------------------------------------------
# Helper: simulate sensor stream from synthetic data
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _load_synthetic_stream() -> np.ndarray:
    """Load synthetic test data as a numpy array for simulation."""
    test_csv = SYNTHETIC_DIR / "test.csv"
    if not test_csv.exists():
        # Generate minimal demo data if synthetic not yet created
        rng = np.random.default_rng(42)
        n = 10000
        t = np.arange(n) / FS
        v_nom = CFG["data_generation"]["voltage_nominal"]
        data = np.column_stack([
            v_nom * np.sin(2 * np.pi * 50 * t + a)
            for a in [0, -2.094, 2.094]
        ] + [
            50 * np.sin(2 * np.pi * 50 * t - 0.524 + a)
            for a in [0, -2.094, 2.094]
        ] + [
            np.full(n, 50.01),
            np.full(n, 0.95),
            np.full(n, 2.5),
            np.full(n, 0.5),
        ])
        return data.astype(np.float32)
    df = pd.read_csv(test_csv)[CHANNEL_NAMES]
    return df.values.astype(np.float32)


def _get_next_sample(stream: np.ndarray) -> np.ndarray:
    idx = st.session_state["sim_step"] % len(stream)
    st.session_state["sim_step"] += 1
    return stream[idx]


# ---------------------------------------------------------------------------
# Helper: gauge colour
# ---------------------------------------------------------------------------

def _prob_color(prob: float) -> str:
    if prob >= RED_THRESH:
        return "#e74c3c"
    elif prob >= YELLOW_THRESH:
        return "#f39c12"
    return "#27ae60"


def _fault_gauge_html(fault_name: str, prob: float) -> str:
    color = _prob_color(prob)
    pct = int(prob * 100)
    return f"""
    <div style="margin:6px 0; padding:8px; border-radius:6px; background:{color}22; border-left:4px solid {color}">
        <b style="color:{color}">{fault_name}</b>
        <span style="float:right; color:{color}; font-weight:bold">{pct}%</span>
        <div style="background:#ddd; border-radius:4px; height:8px; margin-top:4px">
            <div style="background:{color}; width:{pct}%; height:8px; border-radius:4px"></div>
        </div>
    </div>
    """

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚡ Fault Monitor")
    st.markdown("---")

    model_choice = st.selectbox(
        "Active Model",
        options=["random_forest", "xgboost", "neural_network"],
        index=0,
    )

    if model_choice != st.session_state["loaded_model"]:
        with st.spinner(f"Loading {model_choice}..."):
            pred = _load_predictor(model_choice)
            if pred is not None:
                st.session_state["predictor"] = pred
                st.session_state["loaded_model"] = model_choice
                st.success(f"Model loaded: {model_choice}")

    st.markdown("---")
    st.subheader("Simulation Control")
    col_run, col_stop = st.columns(2)
    with col_run:
        if st.button("▶ Start"):
            st.session_state["running"] = True
    with col_stop:
        if st.button("⏹ Stop"):
            st.session_state["running"] = False

    refresh_rate = st.slider("Refresh (ms)", 100, 2000, 500, step=100)

    st.markdown("---")
    st.subheader("Alert Thresholds")
    st.markdown(f"🟡 Yellow: >{int(YELLOW_THRESH*100)}%")
    st.markdown(f"🔴 Red: >{int(RED_THRESH*100)}%")

    st.markdown("---")
    st.markdown("**Mode**")
    mode = st.radio("", ["Live Simulation", "Batch CSV Analysis"])

# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------

st.title("⚡ Power Grid Fault Prediction Dashboard")
st.caption(f"Model: `{st.session_state.get('loaded_model', 'none')}` | "
           f"Sample rate: {FS} Hz | Rolling window: {ROLLING_WINDOW_S}s")

# ---------------------------------------------------------------------------
# Batch CSV Analysis Mode
# ---------------------------------------------------------------------------

if mode == "Batch CSV Analysis":
    st.subheader("Batch CSV Analysis")
    uploaded = st.file_uploader("Upload sensor CSV", type=["csv"])
    if uploaded is not None:
        batch_df = pd.read_csv(uploaded)
        missing = [c for c in CHANNEL_NAMES if c not in batch_df.columns]
        if missing:
            st.error(f"CSV missing columns: {missing}")
        else:
            predictor = st.session_state.get("predictor")
            if predictor is None:
                st.warning("Load a model first in the sidebar.")
            else:
                X = batch_df[CHANNEL_NAMES].values
                results = predictor.push_batch(X)
                results_clean = [r for r in results if r is not None]

                if results_clean:
                    result_df = pd.DataFrame([{
                        "Fault Class": r.fault_class,
                        "Fault Label": r.fault_label,
                        "Confidence": f"{r.confidence:.3f}",
                        "Latency (ms)": f"{r.latency_ms:.1f}",
                    } for r in results_clean])

                    st.write(f"**{len(results_clean)} predictions generated**")
                    st.dataframe(result_df, use_container_width=True)

                    # Show distribution
                    label_counts = result_df["Fault Label"].value_counts()
                    st.bar_chart(label_counts)
                else:
                    st.info("No predictions triggered (buffer not full). Upload a larger file.")
    st.stop()

# ---------------------------------------------------------------------------
# Live Simulation Mode
# ---------------------------------------------------------------------------

# Layout: signal plots | fault gauges
col_signal, col_gauges = st.columns([2, 1])

with col_signal:
    st.subheader("Live Sensor Signals")
    placeholder_voltage = st.empty()
    placeholder_current = st.empty()
    placeholder_freq = st.empty()

with col_gauges:
    st.subheader("Fault Probabilities")
    placeholder_gauges = st.empty()

# Alert log and trend chart
st.markdown("---")
col_alerts, col_trend = st.columns([3, 2])
with col_alerts:
    st.subheader("Alert Log")
    placeholder_alerts = st.empty()
with col_trend:
    st.subheader("Fault Trend (last 60 s)")
    placeholder_trend = st.empty()

# ---------------------------------------------------------------------------
# Simulation loop
# ---------------------------------------------------------------------------

if not st.session_state["running"]:
    st.info("Press **▶ Start** in the sidebar to begin simulation.")
else:
    stream = _load_synthetic_stream()
    predictor: Optional[RealTimePredictor] = st.session_state.get("predictor")

    if predictor is None:
        st.error("No model loaded. Select a model in the sidebar.")
        st.stop()

    # Run a batch of samples per refresh cycle
    samples_per_refresh = max(1, int(refresh_rate / 1000 * FS))
    current_probs = {name: 0.0 for name in FAULT_NAMES}

    for _ in range(samples_per_refresh):
        sample = _get_next_sample(stream)

        # Append to display buffer
        row = pd.DataFrame([{ch: float(sample[i]) for i, ch in enumerate(CHANNEL_NAMES)}])
        st.session_state["signal_buffer"] = pd.concat(
            [st.session_state["signal_buffer"], row], ignore_index=True
        ).tail(ROLLING_N)

        # Push to predictor
        result: Optional[PredictionResult] = predictor.push(sample)
        if result is not None:
            current_probs = result.per_class_probabilities
            if result.is_fault():
                alert = {
                    "Timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3],
                    "Fault": result.fault_label,
                    "Confidence": f"{result.confidence:.3f}",
                    "Latency (ms)": f"{result.latency_ms:.1f}",
                    "Top Features": ", ".join(
                        f"{f}={v:.3f}" for f, v in result.top_contributing_features[:3]
                    ),
                }
                st.session_state["alert_log"].insert(0, alert)
                st.session_state["alert_log"] = st.session_state["alert_log"][:100]
                st.session_state["fault_counts"][result.fault_label] += 1

    # --- Render signal plots ---
    buf = st.session_state["signal_buffer"]
    n_pts = len(buf)

    if n_pts > 0:
        import plotly.graph_objects as go

        with placeholder_voltage.container():
            fig_v = go.Figure()
            for ch in ["Va", "Vb", "Vc"]:
                fig_v.add_trace(go.Scatter(y=buf[ch].values, name=ch, mode="lines"))
            fig_v.update_layout(
                title="Phase Voltages (V)", height=200, margin=dict(l=0, r=0, t=30, b=0),
                xaxis_title="Sample", yaxis_title="V", showlegend=True,
                legend=dict(orientation="h"),
            )
            st.plotly_chart(fig_v, use_container_width=True)

        with placeholder_current.container():
            fig_i = go.Figure()
            for ch in ["Ia", "Ib", "Ic"]:
                fig_i.add_trace(go.Scatter(y=buf[ch].values, name=ch, mode="lines"))
            fig_i.update_layout(
                title="Phase Currents (A)", height=200, margin=dict(l=0, r=0, t=30, b=0),
                xaxis_title="Sample", yaxis_title="A", showlegend=True,
                legend=dict(orientation="h"),
            )
            st.plotly_chart(fig_i, use_container_width=True)

        with placeholder_freq.container():
            fig_f = go.Figure()
            fig_f.add_trace(go.Scatter(y=buf["Freq"].values, name="Frequency", mode="lines",
                                       line=dict(color="#27ae60")))
            fig_f.update_layout(
                title="Frequency (Hz)", height=150, margin=dict(l=0, r=0, t=30, b=0),
                xaxis_title="Sample",
            )
            st.plotly_chart(fig_f, use_container_width=True)

    # --- Render gauges ---
    with placeholder_gauges.container():
        gauges_html = "".join(
            _fault_gauge_html(name, current_probs.get(name, 0.0))
            for name in FAULT_NAMES
        )
        st.markdown(gauges_html, unsafe_allow_html=True)

    # --- Alert log ---
    with placeholder_alerts.container():
        if st.session_state["alert_log"]:
            alert_df = pd.DataFrame(st.session_state["alert_log"])
            st.dataframe(alert_df, use_container_width=True, height=250)
        else:
            st.info("No faults detected yet.")

    # --- Trend chart ---
    with placeholder_trend.container():
        counts = st.session_state["fault_counts"]
        trend_df = pd.DataFrame.from_dict(
            {"Fault Type": list(counts.keys()), "Count": list(counts.values())}
        ).query("Count > 0")
        if not trend_df.empty:
            st.bar_chart(trend_df.set_index("Fault Type"))
        else:
            st.info("No fault events recorded.")

    time.sleep(refresh_rate / 1000)
    st.rerun()
