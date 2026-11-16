"""Vehicle Registration Analytics - Streamlit application matching the Video Demo design."""

from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "vehicle_data.csv"
ML_DIR = BASE_DIR / "output" / "ml_prediction"

REQUIRED_COLUMNS = ["id", "state", "rto", "maker", "count"]

st.set_page_config(
    page_title="Vehicle Registration Analytics",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ------------------------------ Styling Matching Video Demo ------------------------------
st.markdown(textwrap.dedent("""
<style>
/* Main Canvas */
.stApp {
    background: #82cbe5;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

.block-container {
    max-width: 1200px;
    padding-top: 0.4rem;
    padding-bottom: 2rem;
}

/* Streamlit Top Nav Bar Matching Video Demo */
div[class*="st-key-top_navigation"] {
    background: #f7fcff !important;
    border-bottom: 1px solid #dceaf0 !important;
    padding: 8px 10px !important;
    margin: -0.4rem -1rem 1.2rem -1rem !important;
    border-radius: 0 0 10px 10px !important;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.06) !important;
}

div[class*="st-key-top_navigation"] [data-testid="stHorizontalBlock"] {
    gap: 8px !important;
}

div[class*="st-key-top_navigation"] [data-testid="stButton"] > button {
    background: #ffffff !important;
    color: #334155 !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 6px !important;
    min-height: 36px !important;
    padding: 4px 10px !important;
    font-size: 11px !important;
    font-weight: 700 !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
    transition: all 0.2s ease !important;
}

div[class*="st-key-top_navigation"] [data-testid="stButton"] > button:hover {
    background: #eff6ff !important;
    color: #2563eb !important;
    border-color: #93c5fd !important;
    transform: translateY(-1px);
}

/* Active Nav Button */
div[class*="st-key-top_navigation"] [data-testid="stButton"] > button[kind="primary"] {
    background: #ffffff !important;
    color: #2563eb !important;
    border: 2px solid #2563eb !important;
    box-shadow: 0 2px 6px rgba(37, 99, 235, 0.2) !important;
}

/* Landing Home Card (Scene 1 in Video Demo) */
.landing-area {
    min-height: 70vh;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 12px;
}

.landing-card {
    width: 370px;
    max-width: calc(100vw - 36px);
    box-sizing: border-box;
    background: #f8fafc;
    border-radius: 14px;
    padding: 28px 28px 24px;
    box-shadow: 0 20px 40px rgba(15, 23, 42, 0.22);
    border: 1px solid rgba(255, 255, 255, 0.8);
    text-align: center;
}

.landing-icon {
    font-size: 32px;
    margin-bottom: 8px;
}

.landing-title {
    color: #293241;
    font-size: 28px;
    line-height: 1.05;
    font-weight: 900;
    letter-spacing: -0.5px;
    margin: 0;
}

.landing-title span.accent {
    color: #2563eb;
    display: block;
}

.landing-subtitle {
    color: #64748b;
    font-size: 12px;
    line-height: 1.5;
    margin: 14px 0 18px;
}

.tech-badges {
    display: flex;
    justify-content: center;
    gap: 6px;
    margin-bottom: 18px;
}

.tech-badge {
    font-size: 10px;
    font-weight: 700;
    padding: 3px 8px;
    border-radius: 9999px;
}
.badge-pyspark { background: #e0f2fe; color: #0369a1; }
.badge-hadoop { background: #fef3c7; color: #b45309; }
.badge-streamlit { background: #dcfce7; color: #15803d; }

.landing-buttons {
    display: flex;
    gap: 10px;
}

.landing-buttons a {
    flex: 1;
    text-align: center;
    color: white !important;
    text-decoration: none !important;
    font-size: 12px;
    font-weight: 700;
    padding: 10px 8px;
    border-radius: 8px;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.12);
    transition: opacity 0.2s;
}

.landing-buttons .learn { background: #2563eb; }
.landing-buttons .start { background: #16a34a; }
.landing-buttons a:hover { opacity: 0.9; }

/* Content Cards & Banners */
.content-card {
    background: #f8fafc;
    border-radius: 14px;
    padding: 24px;
    box-shadow: 0 10px 25px rgba(15, 23, 42, 0.14);
    margin: 18px auto;
    border: 1px solid rgba(255, 255, 255, 0.7);
}

.page-hero {
    background: linear-gradient(115deg, #2563eb, #7c3aed, #c0268c);
    color: white;
    padding: 22px 26px;
    border-radius: 14px;
    margin: 8px 0 20px;
    box-shadow: 0 10px 24px rgba(37, 99, 235, 0.25);
}
.page-hero h1 { color: white; margin: 0; font-size: 1.75rem; font-weight: 800; }
.page-hero p { color: #f1f5f9; margin: 6px 0 0; font-size: 0.95rem; }

/* Video Demo Metric Cards */
[data-testid="stMetric"] {
    background: white !important;
    border: 1px solid #dbeafe !important;
    border-radius: 12px !important;
    padding: 14px 16px !important;
    box-shadow: 0 4px 12px rgba(30, 64, 175, 0.07) !important;
    text-align: center;
}
[data-testid="stMetricLabel"] { color: #64748b !important; font-size: 11px !important; font-weight: 700 !important; text-transform: uppercase; }
[data-testid="stMetricValue"] { font-weight: 900 !important; font-size: 1.5rem !important; }

/* Colored Metrics matching Video Demo */
.metric-blue [data-testid="stMetricValue"] { color: #1d4ed8 !important; }
.metric-indigo [data-testid="stMetricValue"] { color: #4338ca !important; }
.metric-green [data-testid="stMetricValue"] { color: #047857 !important; }
.metric-amber [data-testid="stMetricValue"] { color: #b45309 !important; }

/* Feature Cards (Key Features Page) */
.feature-box {
    background: white;
    border-radius: 10px;
    padding: 16px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
    height: 100%;
}
.feature-box h4 {
    margin: 8px 0 4px 0;
    color: #1e293b;
    font-size: 13px;
    font-weight: 800;
}
.feature-box p {
    color: #64748b;
    font-size: 11px;
    line-height: 1.45;
    margin: 0;
}

.pipeline-banner {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 10px;
    padding: 14px 18px;
    color: #1e3a8a;
    font-size: 12px;
    line-height: 1.5;
    margin-top: 14px;
}

/* Prediction Output Card in ML page */
.prediction-badge-box {
    background: #ffffff;
    border-radius: 14px;
    padding: 22px;
    text-align: center;
    margin: 14px 0;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.06);
}

/* Remove default Streamlit header */
header[data-testid="stHeader"], div[data-testid="stToolbar"] {
    display: none !important;
}
</style>
"""), unsafe_allow_html=True)

# ------------------------------ Navigation ------------------------------
PAGES = [
    "Home",
    "Key Features",
    "Our Mission",
    "Analytics Filters",
    "Regional Analysis",
    "ML Prediction",
]

if "current_page" not in st.session_state:
    st.session_state["current_page"] = "Home"

requested_page = st.query_params.get("page")
if requested_page in PAGES:
    st.session_state["current_page"] = requested_page

def go_to_page(page_name):
    st.session_state["current_page"] = page_name

# Navigation Container
with st.container(key="top_navigation"):
    nav_cols = st.columns(6, gap="small")
    for col, page_name in zip(nav_cols, PAGES):
        with col:
            is_current = st.session_state["current_page"] == page_name
            st.button(
                page_name,
                key=f"nav_{page_name}",
                type="primary" if is_current else "secondary",
                use_container_width=True,
                on_click=go_to_page,
                args=(page_name,),
            )

page = st.session_state["current_page"]


# ------------------------------ Data helpers ------------------------------
@st.cache_data(show_spinner="Loading vehicle registration dataset...")
def load_vehicle_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"vehicle_data.csv was not found beside app.py. Expected: {DATA_FILE}"
        )

    frame = pd.read_csv(
        DATA_FILE,
        dtype={"state": "string", "rto": "string", "maker": "string"},
    )
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


def get_filtered_data(frame):
    chosen_states = st.session_state.get("filter_states", [])
    chosen_rtos = st.session_state.get("filter_rtos", [])
    chosen_makers = st.session_state.get("filter_makers", [])

    result = frame
    if chosen_states:
        result = result[result["state"].isin(chosen_states)]
    if chosen_rtos:
        result = result[result["rto"].isin(chosen_rtos)]
    if chosen_makers:
        result = result[result["maker"].isin(chosen_makers)]
    return result


@st.cache_data(show_spinner="Loading machine-learning results...")
def load_ml_results():
    metrics_path = ML_DIR / "ml_metrics.csv"
    predictions_path = ML_DIR / "ml_predictions.csv"
    metrics = pd.read_csv(metrics_path) if metrics_path.exists() else None
    predictions = pd.read_csv(predictions_path) if predictions_path.exists() else None
    return metrics, predictions


@st.cache_resource(show_spinner="Training Logistic Regression model for live predictions...")
def train_live_model(data_signature, training_frame):
    feature_columns = ["state", "rto", "maker"]
    training = training_frame[feature_columns + ["count"]].copy()

    training["registration_category"] = pd.cut(
        training["count"],
        bins=[-1, 25, 176, float("inf")],
        labels=["Low", "Medium", "High"],
    )
    training = training.dropna(subset=feature_columns + ["registration_category"])

    X = training[feature_columns].astype(str)
    y = training["registration_category"].astype(str)

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), feature_columns)
        ],
        remainder="drop",
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(
                max_iter=1000,
                solver="saga",
                class_weight="balanced",
                random_state=42,
            )),
        ]
    )
    model.fit(X, y)
    return model, len(training), training["registration_category"].value_counts().to_dict()


def page_header(title, subtitle):
    st.markdown(
        f'<div class="page-hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


# ------------------------------ Scene 1: Home (Matching Video Demo) ------------------------------
if page == "Home":
    st.markdown(textwrap.dedent("""
    <div class="landing-area">
      <div class="landing-card">
        <div class="landing-icon">🚗</div>
        <h1 class="landing-title">
          <span>VEHICLE</span>
          <span>REGISTRATION</span>
          <span class="accent">ANALYTICS</span>
        </h1>
        <p class="landing-subtitle">
          Big Data Analytics for Vehicle Registration Trends and
          Regional Growth Analysis in India.
        </p>
        <div class="tech-badges">
          <span class="tech-badge badge-pyspark">PySpark</span>
          <span class="tech-badge badge-hadoop">Hadoop HDFS</span>
          <span class="tech-badge badge-streamlit">Streamlit</span>
        </div>
        <div class="landing-buttons">
          <a class="learn" href="?page=Key%20Features">Learn More</a>
          <a class="start" href="?page=Analytics%20Filters">Get Started</a>
        </div>
      </div>
    </div>
    """), unsafe_allow_html=True)


# ------------------------------ Scene 2: Key Features ------------------------------
elif page == "Key Features":
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown("## Key Features")
    st.caption("Distributed data architecture and analytical capabilities.")
    
    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown("""
        <div class="feature-box">
          <div style="font-size: 20px;">📊</div>
          <h4>Regional Analytics</h4>
          <p>Explore registration totals across 36 Indian states and 1,417 Regional Transport Offices (RTOs).</p>
        </div>
        """, unsafe_allow_html=True)

    with f2:
        st.markdown("""
        <div class="feature-box">
          <div style="font-size: 20px;">🏭</div>
          <h4>Manufacturer Trends</h4>
          <p>Benchmark 6,232 vehicle manufacturers and analyze market share distribution with interactive visuals.</p>
        </div>
        """, unsafe_allow_html=True)

    with f3:
        st.markdown("""
        <div class="feature-box">
          <div style="font-size: 20px;">🤖</div>
          <h4>ML Classification</h4>
          <p>Predict registration volume tiers (Low, Medium, High) using trained Logistic Regression models.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="pipeline-banner">
      <strong>⚡ Big Data Processing Pipeline:</strong> Raw dataset is stored on <strong>Hadoop HDFS</strong>. Distributed cleaning, validation, and aggregations are computed with <strong>PySpark</strong> and <strong>Spark SQL</strong>, and rendered via <strong>Streamlit</strong>.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ------------------------------ Scene 3: Our Mission ------------------------------
elif page == "Our Mission":
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown("## Our Mission")
    st.write(
        "Our mission is to provide an efficient Big Data analytics framework for "
        "understanding vehicle registration patterns across Indian states, RTO "
        "regions, and vehicle manufacturers."
    )
    st.write(
        "The project integrates **Hadoop HDFS**, **PySpark**, **Spark SQL**, **Logistic Regression**, "
        "and **Streamlit** to support scalable data preparation, regional analytics, classification, "
        "and clear interactive presentation of results."
    )
    st.markdown("### Project Objectives")
    st.markdown(
        "- **Store & Process:** Efficiently process large-scale vehicle registration records.\n"
        "- **Clean & Validate:** Automated preprocessing pipeline before analytics.\n"
        "- **Regional Summaries:** Generate state-wise, RTO-wise, and manufacturer-wise insights.\n"
        "- **Predictive Modeling:** Classify registration volumes into Low, Medium, and High tiers.\n"
        "- **Interactive Delivery:** Enable self-service analytics for automotive analysts and policymakers."
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ------------------------------ Scene 4: Analytics Filters & Data Selection ------------------------------
elif page == "Analytics Filters":
    page_header(
        "📊 Analytics Filters and Data Selection",
        "Review the dataset and select states, RTOs, and manufacturers for focused analysis.",
    )

    try:
        data = load_vehicle_data()
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    states = sorted(data["state"].dropna().unique().tolist())
    rtos = sorted(data["rto"].dropna().unique().tolist())
    makers = sorted(data["maker"].dropna().unique().tolist())

    # 4 Colored Metric Cards matching Video Demo
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown('<div class="metric-blue">', unsafe_allow_html=True)
        st.metric("Total Records", f"{len(data):,}")
        st.markdown('</div>', unsafe_allow_html=True)
    with k2:
        st.markdown('<div class="metric-indigo">', unsafe_allow_html=True)
        st.metric("Total Registrations", f"{int(data['count'].sum()):,}")
        st.markdown('</div>', unsafe_allow_html=True)
    with k3:
        st.markdown('<div class="metric-green">', unsafe_allow_html=True)
        st.metric("States / UTs", f"{data['state'].nunique():,}")
        st.markdown('</div>', unsafe_allow_html=True)
    with k4:
        st.markdown('<div class="metric-amber">', unsafe_allow_html=True)
        st.metric("RTOs", f"{data['rto'].nunique():,}")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown("### Choose Analysis Filters")

    current_states = st.session_state.get("filter_states", [])
    current_states = [v for v in current_states if v in states]
    st.multiselect(
        "Select State(s)",
        options=states,
        default=current_states,
        key="filter_states",
        help="Leave empty to include all states and Union Territories.",
    )

    state_scope = (
        data[data["state"].isin(st.session_state["filter_states"])]
        if st.session_state["filter_states"] else data
    )
    rto_options = sorted(state_scope["rto"].dropna().unique().tolist())
    current_rtos = [
        v for v in st.session_state.get("filter_rtos", []) if v in rto_options
    ]
    st.multiselect(
        "Select RTO(s)",
        options=rto_options,
        default=current_rtos,
        key="filter_rtos",
        help="Leave empty to include all RTOs in the selected state scope.",
    )

    rto_scope = (
        state_scope[state_scope["rto"].isin(st.session_state["filter_rtos"])]
        if st.session_state["filter_rtos"] else state_scope
    )
    maker_options = sorted(rto_scope["maker"].dropna().unique().tolist())
    current_makers = [
        v for v in st.session_state.get("filter_makers", []) if v in maker_options
    ]
    st.multiselect(
        "Select Vehicle Maker(s)",
        options=maker_options,
        default=current_makers,
        key="filter_makers",
        help="Leave empty to include all makers in the selected scope.",
    )

    filtered = get_filtered_data(data)
    
    # Filter count banner matching Video Demo
    st.markdown(f"""
    <div style="background:#eff6ff; border:1px solid #bfdbfe; border-radius:10px; padding:12px 16px; margin:16px 0; display:flex; justify-content:space-between; align-items:center;">
        <div>
            <span style="color:#1e3a8a; font-weight:700; font-size:13px;">Filtered Registration Count:</span>
            <span style="color:#1d4ed8; font-weight:900; font-size:18px; margin-left:8px;">{int(filtered['count'].sum()):,}</span>
        </div>
        <span style="color:#64748b; font-size:12px;">{len(filtered):,} records match selection</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Filtered Records Preview")
    st.dataframe(filtered.head(100), use_container_width=True, hide_index=True)
    
    st.download_button(
        "📥 Download Filtered Dataset (CSV)",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_vehicle_registrations.csv",
        mime="text/csv",
    )
    st.caption("ℹ️ Selected filters are automatically retained when you navigate to Regional Analysis.")
    st.markdown("</div>", unsafe_allow_html=True)


# ------------------------------ Scene 5: Regional & Manufacturer Analysis ------------------------------
elif page == "Regional Analysis":
    page_header(
        "📈 Regional and Manufacturer Analysis",
        "Choose a region and manufacturer to generate focused registration insights.",
    )

    try:
        data = load_vehicle_data()
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown("### Select Regional Details")
    st.caption("Select a state first. RTO and manufacturer dropdowns will automatically update.")

    state_options = ["All States / UTs"] + sorted(
        data["state"].dropna().astype(str).unique().tolist()
    )
    selected_region_state = st.selectbox(
        "Select State / Union Territory",
        options=state_options,
        key="regional_state_choice",
    )

    if selected_region_state == "All States / UTs":
        state_scope = data
    else:
        state_scope = data[data["state"].astype(str) == selected_region_state]

    rto_options = ["All RTOs"] + sorted(
        state_scope["rto"].dropna().astype(str).unique().tolist()
    )
    selected_region_rto = st.selectbox(
        "Select RTO",
        options=rto_options,
        key="regional_rto_choice",
    )

    if selected_region_rto == "All RTOs":
        rto_scope = state_scope
    else:
        rto_scope = state_scope[state_scope["rto"].astype(str) == selected_region_rto]

    maker_options = ["All Manufacturers"] + sorted(
        rto_scope["maker"].dropna().astype(str).unique().tolist()
    )
    selected_region_maker = st.selectbox(
        "Select Vehicle Manufacturer",
        options=maker_options,
        key="regional_maker_choice",
    )

    if st.button("🔎 Analyze Region", type="primary", use_container_width=True, key="run_regional_analysis"):
        st.session_state["regional_analysis_selection"] = {
            "state": selected_region_state,
            "rto": selected_region_rto,
            "maker": selected_region_maker,
        }

    selection = st.session_state.get("regional_analysis_selection")

    if selection is None:
        st.info("Select the regional details above and click **Analyze Region** to display statistics and charts.")
    else:
        if selection["state"] == "All States / UTs":
            regional_data = data.copy()
        else:
            regional_data = data[data["state"].astype(str) == selection["state"]].copy()

        if selection["rto"] != "All RTOs":
            regional_data = regional_data[regional_data["rto"].astype(str) == selection["rto"]]

        if selection["maker"] != "All Manufacturers":
            regional_data = regional_data[regional_data["maker"].astype(str) == selection["maker"]]

        st.markdown("---")
        st.markdown(f"### Analysis Results")
        st.markdown(f"""
        <div style="background:#f1f5f9; padding:8px 14px; border-radius:8px; font-size:12px; color:#475569; margin-bottom:14px;">
            <strong>Scope:</strong> {selection['state']} &nbsp;|&nbsp; {selection['rto']} &nbsp;|&nbsp; {selection['maker']}
        </div>
        """, unsafe_allow_html=True)

        if regional_data.empty:
            st.warning("No records found for this filter selection.")
        else:
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Records", f"{len(regional_data):,}")
            with m2:
                st.metric("Registrations", f"{int(regional_data['count'].sum()):,}")
            with m3:
                st.metric("RTOs", f"{regional_data['rto'].nunique():,}")
            with m4:
                st.metric("Manufacturers", f"{regional_data['maker'].nunique():,}")

            c_left, c_right = st.columns(2)
            with c_left:
                st.markdown("### Registration by State")
                state_totals = (
                    regional_data.groupby("state")["count"]
                    .sum()
                    .nlargest(10)
                    .sort_values()
                )
                if not state_totals.empty:
                    st.bar_chart(state_totals)

            with c_right:
                st.markdown("### Registration by RTO")
                rto_totals = (
                    regional_data.groupby("rto")["count"]
                    .sum()
                    .nlargest(10)
                    .sort_values()
                )
                if not rto_totals.empty:
                    st.bar_chart(rto_totals)

            st.markdown("### Top 10 Vehicle Manufacturers")
            maker_totals = (
                regional_data.groupby("maker")["count"]
                .sum()
                .nlargest(10)
                .sort_values()
            )
            if not maker_totals.empty:
                st.bar_chart(maker_totals)

            st.markdown("### Regional Registration Details")
            regional_summary = (
                regional_data.groupby(["state", "rto"], as_index=False)["count"]
                .sum()
                .sort_values("count", ascending=False)
                .head(100)
            )
            st.dataframe(regional_summary, use_container_width=True, hide_index=True)

            st.markdown("### Manufacturer Registration Details")
            manufacturer_summary = (
                regional_data.groupby("maker", as_index=False)["count"]
                .sum()
                .sort_values("count", ascending=False)
                .head(100)
            )
            st.dataframe(manufacturer_summary, use_container_width=True, hide_index=True)

            st.download_button(
                "📥 Download Regional Analysis CSV",
                data=regional_data.to_csv(index=False).encode("utf-8"),
                file_name="regional_analysis_results.csv",
                mime="text/csv",
                key="download_regional_analysis",
            )
    st.markdown("</div>", unsafe_allow_html=True)


# ------------------------------ Scene 6: ML Prediction Results ------------------------------
elif page == "ML Prediction":
    page_header(
        "🤖 Machine Learning Prediction Results",
        "Enter vehicle registration attributes to generate a live Logistic Regression prediction.",
    )

    try:
        data = load_vehicle_data()
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    data_signature = (len(data), int(data["count"].sum()), int(data["id"].sum()))
    try:
        live_model, trained_rows, class_counts = train_live_model(data_signature, data)
    except Exception as exc:
        st.error(f"Could not train Logistic Regression model: {exc}")
        st.stop()

    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown("### Model Information")
    info1, info2, info3 = st.columns(3)
    info1.metric("Algorithm", "Logistic Regression")
    info2.metric("Training Records", f"{trained_rows:,}")
    info3.metric("Input Features", "3 (State, RTO, Maker)")

    st.markdown("### Registration Category Rules")
    r1, r2, r3 = st.columns(3)
    r1.metric("Low Volume", "0 – 25")
    r2.metric("Medium Volume", "26 – 176")
    r3.metric("High Volume", "Above 176")

    st.markdown("---")
    st.markdown("### Make a Live Prediction")
    st.caption("Select attributes below and click Predict to calculate the registration tier in real time.")

    states = sorted(data["state"].dropna().astype(str).unique().tolist())
    selected_state = st.selectbox("Select State", options=states, key="live_predict_state")

    state_data = data[data["state"].astype(str) == selected_state]
    rto_options = sorted(state_data["rto"].dropna().astype(str).unique().tolist())
    selected_rto = st.selectbox("Select RTO", options=rto_options, key="live_predict_rto")

    rto_data = state_data[state_data["rto"].astype(str) == selected_rto]
    maker_options = sorted(rto_data["maker"].dropna().astype(str).unique().tolist())
    selected_maker = st.selectbox("Select Vehicle Maker", options=maker_options, key="live_predict_maker")

    if st.button("🔮 Predict Registration Category", type="primary", use_container_width=True):
        input_record = pd.DataFrame([{
            "state": selected_state,
            "rto": selected_rto,
            "maker": selected_maker,
        }])

        predicted_category = live_model.predict(input_record)[0]
        probabilities = live_model.predict_proba(input_record)[0]
        class_names = live_model.named_steps["classifier"].classes_
        confidence = float(max(probabilities)) * 100

        st.session_state["live_prediction_result"] = {
            "state": selected_state,
            "rto": selected_rto,
            "maker": selected_maker,
            "category": predicted_category,
            "confidence": confidence,
            "probabilities": dict(zip(class_names, probabilities)),
        }

    result = st.session_state.get("live_prediction_result")
    if result:
        st.markdown("---")
        category_colors = {
            "Low": "#2563eb",
            "Medium": "#d97706",
            "High": "#16a34a",
        }
        res_color = category_colors.get(result["category"], "#334155")
        
        # High-impact prediction badge matching Video Demo
        st.markdown(f"""
        <div class="prediction-badge-box" style="border: 2px solid {res_color};">
            <div style="color:#64748b; font-size:12px; font-weight:700; text-transform:uppercase;">
                Predicted Registration Category
            </div>
            <div style="color:{res_color}; font-size:36px; font-weight:900; margin:4px 0;">
                {result['category'].upper()}
            </div>
            <div style="color:#334155; font-size:13px; font-weight:600;">
                Input: {result['state']} &nbsp;|&nbsp; {result['rto']} &nbsp;|&nbsp; {result['maker']}
            </div>
            <div style="color:#059669; font-size:14px; font-weight:800; margin-top:8px;">
                Model Confidence: {result['confidence']:.2f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Class Probability Distribution")
        prob_df = pd.DataFrame({
            "Category": list(result["probabilities"].keys()),
            "Probability": [float(v) for v in result["probabilities"].values()],
        })
        st.bar_chart(prob_df.set_index("Category"))

    st.markdown("---")
    st.markdown("### Training Class Distribution")
    dist_df = pd.DataFrame([
        {"Category": k, "Records": int(v)} for k, v in class_counts.items()
    ]).sort_values("Category")
    st.bar_chart(dist_df.set_index("Category"))

    # Evaluated Performance Metrics matching Video Demo
    try:
        saved_metrics, _ = load_ml_results()
    except Exception:
        saved_metrics = None

    if saved_metrics is not None:
        st.markdown("### Previously Evaluated Model Performance")
        st.caption("Hold-out test evaluation scores:")
        metric_map = dict(zip(saved_metrics["Metric"], saved_metrics["Score"]))
        ep1, ep2, ep3, ep4 = st.columns(4)
        for col, m_name, color in zip(
            [ep1, ep2, ep3, ep4],
            ["Accuracy", "Precision", "Recall", "F1 Score"],
            ["metric-blue", "metric-indigo", "metric-green", "metric-amber"],
        ):
            val = metric_map.get(m_name)
            if val is not None:
                with col:
                    st.markdown(f'<div class="{color}">', unsafe_allow_html=True)
                    st.metric(m_name, f"{float(val) * 100:.2f}%")
                    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
