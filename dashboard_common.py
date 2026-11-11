from pathlib import Path
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "vehicle_data.csv"
ML_DIR = BASE_DIR / "output" / "ml_prediction"
REQUIRED_COLUMNS = ["id", "state", "rto", "maker", "count"]

@st.cache_data(show_spinner="Loading vehicle data...")
def load_data(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"state": "string", "rto": "string", "maker": "string"})
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")
    frame = frame[REQUIRED_COLUMNS].copy()
    frame["id"] = pd.to_numeric(frame["id"], errors="coerce")
    frame["count"] = pd.to_numeric(frame["count"], errors="coerce")
    for column in ["state", "rto", "maker"]:
        frame[column] = frame[column].str.strip()
    frame = frame.dropna(subset=REQUIRED_COLUMNS)
    frame = frame[frame["count"] >= 0]
    frame["id"] = frame["id"].astype("int64")
    frame["count"] = frame["count"].astype("int64")
    return frame.drop_duplicates()

def load_vehicle_data():
    if not DATA_FILE.exists():
        st.error(f"Could not find {DATA_FILE}. Put vehicle_data.csv next to app.py and restart the dashboard.")
        st.stop()
    try:
        return load_data(str(DATA_FILE))
    except Exception as exc:
        st.error(f"Unable to load the dataset: {exc}")
        st.stop()

@st.cache_data(show_spinner="Loading ML results...")
def load_ml_results():
    metrics_file = ML_DIR / "ml_metrics.csv"
    predictions_file = ML_DIR / "ml_predictions.csv"
    metrics = pd.read_csv(metrics_file) if metrics_file.exists() else None
    predictions = pd.read_csv(predictions_file) if predictions_file.exists() else None
    return metrics, predictions

def apply_saved_filters(data):
    filtered = data
    state = st.session_state.get("dashboard_state", "All states")
    rto = st.session_state.get("dashboard_rto", "All RTOs")
    maker = st.session_state.get("dashboard_maker", "All makers")
    if state != "All states":
        filtered = filtered[filtered["state"] == state]
    if rto != "All RTOs":
        filtered = filtered[filtered["rto"] == rto]
    if maker != "All makers":
        filtered = filtered[filtered["maker"] == maker]
    return filtered

def save_filter(widget_key, saved_key):
    st.session_state[saved_key] = st.session_state[widget_key]

def dashboard_style():
    st.markdown("""
    <style>
    .stApp { background:#f3f6f5; }
    .block-container { max-width:1500px; padding-top:1rem; padding-bottom:2rem; }
    header[data-testid="stHeader"] { background:#087f73; }
    [data-testid="stSidebar"] { background:#e8f3f0; }
    [data-testid="stSidebar"] * { color:#163b38; }
    .dash-top { background:#087f73; color:white; padding:11px 16px; margin:0 0 12px;
        border-radius:3px; font-size:1rem; font-weight:800; letter-spacing:.2px; }
    .dash-filter-label { color:#087f73; font-size:.7rem; font-weight:800; margin-bottom:2px; }
    .dash-kpi { min-height:68px; padding:11px 13px; color:white; border-radius:4px;
        box-shadow:0 3px 8px rgba(15,23,42,.13); margin:2px 0 8px; }
    .dash-kpi-label { font-size:.7rem; font-weight:650; opacity:.95; }
    .dash-kpi-value { font-size:1.2rem; line-height:1.2; font-weight:850; margin-top:5px; }
    .dash-green { background:#9cbd35; } .dash-brown { background:#a87b43; }
    .dash-blue { background:#54a8d8; } .dash-orange { background:#e88c36; }
    .dash-teal { background:#268f9e; }
    .dash-section { background:#078f82; color:#fff; text-align:center; padding:6px 8px;
        font-size:.76rem; font-weight:800; margin:9px 0; border-radius:3px; }
    div[data-testid="stVerticalBlockBorderWrapper"] { background:#fff; border-color:#d6e3e0; border-radius:5px; }
    </style>
    """, unsafe_allow_html=True)

def dashboard_kpis(filtered):
    entries = [
        ("REGISTRATION RECORDS", f"{len(filtered):,}", "dash-green"),
        ("TOTAL REGISTRATIONS", f"{int(filtered['count'].sum()):,}", "dash-brown"),
        ("STATES / UTs", f"{filtered['state'].nunique():,}", "dash-blue"),
        ("RTOs", f"{filtered['rto'].nunique():,}", "dash-orange"),
        ("VEHICLE MAKERS", f"{filtered['maker'].nunique():,}", "dash-teal"),
    ]
    cols = st.columns(5)
    for col, (label, value, color) in zip(cols, entries):
        with col:
            st.markdown(f'<div class="dash-kpi {color}"><div class="dash-kpi-label">{label}</div><div class="dash-kpi-value">{value}</div></div>', unsafe_allow_html=True)
