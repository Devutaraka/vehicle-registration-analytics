"""Vehicle Registration Analytics - six-page Streamlit application."""

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

# ------------------------------ Styling ------------------------------
st.markdown(textwrap.dedent("""
<style>
.stApp {
    background: #82cbe5;
}
.block-container {
    max-width: 1200px;
    padding-top: .4rem;
    padding-bottom: 2rem;
}
.top-navigation {
    background: #f7fcff;
    border-bottom: 1px solid #dceaf0;
    padding: 9px 8px;
    margin: -0.4rem -1rem 1.2rem -1rem;
    text-align: center;
    white-space: normal;
}
.top-navigation a {
    display: inline-block;
    color: #334155 !important;
    text-decoration: none !important;
    font-size: 11px;
    font-weight: 650;
    margin: 0 12px;
}
.top-navigation a:hover {
    color: #2563eb !important;
}
.landing-area {
    min-height: 68vh;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 12px;
}
.landing-card {
    width: 360px;
    max-width: calc(100vw - 36px);
    box-sizing: border-box;
    background: #f8fafc;
    border-radius: 12px;
    padding: 24px 26px 22px;
    box-shadow: 0 14px 28px rgba(15,23,42,.20);
}
.landing-icon {
    font-size: 22px;
    margin-bottom: 10px;
}
.landing-title {
    color: #293241;
    font-size: 30px;
    line-height: 1.02;
    font-weight: 900;
    letter-spacing: .2px;
    margin: 0;
}
.landing-title span {
    display: block;
    white-space: nowrap;
}
.landing-subtitle {
    color: #64748b;
    font-size: 12px;
    line-height: 1.45;
    margin: 13px 0 18px;
}
.landing-buttons {
    display: flex;
    gap: 9px;
}
.landing-buttons a {
    flex: 1;
    text-align: center;
    color: white !important;
    text-decoration: none !important;
    font-size: 11px;
    font-weight: 700;
    padding: 9px 8px;
    border-radius: 6px;
}
.landing-buttons .learn { background: #2563eb; }
.landing-buttons .start { background: #16a34a; }
.landing-buttons a:hover { opacity: .9; }
.content-card {
    background: #f8fafc;
    border-radius: 13px;
    padding: 24px;
    box-shadow: 0 10px 24px rgba(15,23,42,.13);
    margin: 18px auto;
}
.page-hero {
    background: linear-gradient(115deg,#2563eb,#7c3aed,#c0268c);
    color: white;
    padding: 22px 26px;
    border-radius: 15px;
    margin: 8px 0 20px;
}
.page-hero h1 { color: white; margin: 0; font-size: 1.8rem; }
.page-hero p { color: white; margin: 6px 0 0; }
.section-heading {
    color: #172554;
    font-size: 1.35rem;
    font-weight: 800;
    margin: 18px 0 10px;
}
[data-testid="stMetric"] {
    background: white;
    border: 1px solid #dbeafe;
    border-radius: 14px;
    padding: 14px 16px;
    box-shadow: 0 7px 18px rgba(30,64,175,.08);
}
[data-testid="stMetricLabel"] { color: #475569; }
[data-testid="stMetricValue"] { color: #1d4ed8; font-weight: 800; }
.stButton > button, .stDownloadButton > button {
    border: 0;
    border-radius: 7px;
    font-weight: 700;
}
div[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}
 
div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button {
    min-height: 2.45rem;
    font-size: .78rem;
    white-space: normal;
    padding: .35rem .25rem;
}

/* Force the six navigation controls to remain visible in all Streamlit themes. */
div[class*="st-key-top_navigation"] {
    background: transparent !important;
    border: 0 !important;
    padding: 0 !important;
    margin: -0.4rem 0 0.6rem 0 !important;
    box-shadow: none !important;
}
div[class*="st-key-top_navigation"] [data-testid="stHorizontalBlock"] {
    gap: 6px !important;
}
div[class*="st-key-top_navigation"] [data-testid="stButton"] > button {
    background: #ffffff !important;
    color: #26364a !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 5px !important;
    min-height: 34px !important;
    padding: 4px 5px !important;
    opacity: 1 !important;
    box-shadow: none !important;
}
div[class*="st-key-top_navigation"] [data-testid="stButton"] > button p,
div[class*="st-key-top_navigation"] [data-testid="stButton"] > button span,
div[class*="st-key-top_navigation"] [data-testid="stButton"] > button div {
    color: #26364a !important;
    opacity: 1 !important;
    font-size: 11px !important;
    font-weight: 700 !important;
}
div[class*="st-key-top_navigation"] [data-testid="stButton"] > button:hover {
    background: #eaf4ff !important;
    border-color: #60a5fa !important;
}


/* Remove Streamlit's built-in white header strip and toolbar */
header[data-testid="stHeader"],
div[data-testid="stToolbar"],
div[data-testid="stDecoration"],
#MainMenu,
.stAppDeployButton {
    display: none !important;
    height: 0 !important;
    visibility: hidden !important;
}
.stApp > header {
    display: none !important;
}
section[data-testid="stMain"] {
    padding-top: 0 !important;
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

# Read page links from the landing card when the user clicks Learn More/Get Started.
requested_page = st.query_params.get("page")
if requested_page in PAGES:
    st.session_state["current_page"] = requested_page

def go_to_page(page_name):
    st.session_state["current_page"] = page_name

# Reliable, visible Streamlit navigation: six separate page buttons.
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
    """Train the same three-class Logistic Regression approach for interactive predictions."""
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


# ------------------------------ Fig. 7: Home ------------------------------
if page == "Home":
    st.markdown(textwrap.dedent("""
    <div class="landing-area">
      <div class="landing-card">
        <div class="landing-icon">🚗</div>
        <h1 class="landing-title">
          <span>VEHICLE</span>
          <span>REGISTRATION</span>
          <span>ANALYTICS</span>
        </h1>
        <p class="landing-subtitle">
          Big Data Analytics for Vehicle Registration Trends and
          Regional Growth Analysis in India.
        </p>
        <div class="landing-buttons">
          <a class="learn" href="?page=Key%20Features">Learn More</a>
          <a class="start" href="?page=Analytics%20Filters">Get Started</a>
        </div>
      </div>
    </div>
    """), unsafe_allow_html=True)



# ------------------------------ Fig. 8: Features ------------------------------
elif page == "Key Features":
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown("## Key Features")
    f1, f2, f3 = st.columns(3)

    with f1:
        st.markdown("### 📊 Regional Analytics")
        st.write(
            "Explore registration totals by state and Regional Transport Office (RTO). "
            "Use filters to focus on selected regions."
        )
    with f2:
        st.markdown("### 🏭 Manufacturer Analysis")
        st.write(
            "Identify leading vehicle manufacturers and compare their registration "
            "volumes using interactive charts and tables."
        )
    with f3:
        st.markdown("### 🤖 ML Classification")
        st.write(
            "Use Logistic Regression to classify records into Low, Medium, and High "
            "registration categories and inspect model evaluation results."
        )

    st.markdown("---")
    st.markdown("### Big Data Processing")
    st.write(
        "Hadoop HDFS stores the raw dataset. PySpark performs cleaning, validation, "
        "and aggregation. Spark SQL supports analytical queries, and Streamlit "
        "presents the results through an interactive dashboard."
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ------------------------------ Fig. 9: Our Mission ------------------------------
elif page == "Our Mission":
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown("## Our Mission")
    st.write(
        "Our mission is to provide an efficient Big Data analytics framework for "
        "understanding vehicle registration patterns across Indian states, RTO "
        "regions, and vehicle manufacturers."
    )
    st.write(
        "The project integrates Hadoop HDFS, PySpark, Spark SQL, Logistic Regression, "
        "and Streamlit to support data preparation, regional analysis, classification, "
        "and clear presentation of results."
    )
    st.markdown("### Project Objectives")
    st.markdown(
        "- Store and process the vehicle registration dataset efficiently.\n"
        "- Clean and validate registration records before analysis.\n"
        "- Generate state-wise, RTO-wise, and manufacturer-wise summaries.\n"
        "- Classify registration records into Low, Medium, and High categories.\n"
        "- Present analysis and model results through an interactive dashboard."
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ------------------------------ Fig. 10: Analytics Filters and Data Selection ------------------------------
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

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Records", f"{len(data):,}")
    k2.metric("Total Registrations", f"{int(data['count'].sum()):,}")
    k3.metric("States / UTs", f"{data['state'].nunique():,}")
    k4.metric("RTOs", f"{data['rto'].nunique():,}")

    st.markdown('<div class="section-heading">Choose Analysis Filters</div>',
                unsafe_allow_html=True)

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
    st.metric("Filtered Registration Count", f"{int(filtered['count'].sum()):,}")
    st.caption(f"{len(filtered):,} records match the current filter selection.")

    st.markdown("### Filtered Records Preview")
    st.dataframe(filtered.head(100), use_container_width=True, hide_index=True)
    st.download_button(
        "Download Filtered Dataset",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_vehicle_registrations.csv",
        mime="text/csv",
    )
    st.info("Your selected filters are retained when you open the Regional Analysis page.")


# ------------------------------ Fig. 11: Regional and Manufacturer Analysis ------------------------------
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

    st.markdown("### Select Regional Details")
    st.caption(
        "Choose a state first. The RTO and manufacturer lists update according "
        "to your selection."
    )

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
        state_scope = data[
            data["state"].astype(str) == selected_region_state
        ]

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
        rto_scope = state_scope[
            state_scope["rto"].astype(str) == selected_region_rto
        ]

    maker_options = ["All Manufacturers"] + sorted(
        rto_scope["maker"].dropna().astype(str).unique().tolist()
    )
    selected_region_maker = st.selectbox(
        "Select Vehicle Manufacturer",
        options=maker_options,
        key="regional_maker_choice",
    )

    if st.button(
        "🔎 Analyze Region",
        type="primary",
        use_container_width=True,
        key="run_regional_analysis",
    ):
        st.session_state["regional_analysis_selection"] = {
            "state": selected_region_state,
            "rto": selected_region_rto,
            "maker": selected_region_maker,
        }

    selection = st.session_state.get("regional_analysis_selection")

    if selection is None:
        st.info(
            "Select the required details and click **Analyze Region** "
            "to display regional statistics and charts."
        )
    else:
        # Rebuild the result from the saved selection so it remains visible
        # after Streamlit reruns.
        if selection["state"] == "All States / UTs":
            regional_data = data.copy()
        else:
            regional_data = data[
                data["state"].astype(str) == selection["state"]
            ].copy()

        if selection["rto"] != "All RTOs":
            regional_data = regional_data[
                regional_data["rto"].astype(str) == selection["rto"]
            ]

        if selection["maker"] != "All Manufacturers":
            regional_data = regional_data[
                regional_data["maker"].astype(str) == selection["maker"]
            ]

        st.markdown("---")
        st.markdown("### Analysis Results")
        st.caption(
            f"Scope: {selection['state']}  |  {selection['rto']}  |  "
            f"{selection['maker']}"
        )

        if regional_data.empty:
            st.warning("No records were found for this selection.")
        else:
            total_records = len(regional_data)
            total_registrations = int(regional_data["count"].sum())
            number_of_rtos = regional_data["rto"].nunique()
            number_of_makers = regional_data["maker"].nunique()

            metric1, metric2, metric3, metric4 = st.columns(4)
            metric1.metric("Records", f"{total_records:,}")
            metric2.metric("Registrations", f"{total_registrations:,}")
            metric3.metric("RTOs", f"{number_of_rtos:,}")
            metric4.metric("Manufacturers", f"{number_of_makers:,}")

            chart_left, chart_right = st.columns(2)

            with chart_left:
                st.markdown("### Registration by State")
                state_totals = (
                    regional_data.groupby("state")["count"]
                    .sum()
                    .nlargest(10)
                    .sort_values()
                )
                if not state_totals.empty:
                    st.bar_chart(state_totals)

            with chart_right:
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
            st.dataframe(
                regional_summary,
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("### Manufacturer Registration Details")
            manufacturer_summary = (
                regional_data.groupby("maker", as_index=False)["count"]
                .sum()
                .sort_values("count", ascending=False)
                .head(100)
            )
            st.dataframe(
                manufacturer_summary,
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Download Regional Analysis CSV",
                data=regional_data.to_csv(index=False).encode("utf-8"),
                file_name="regional_analysis_results.csv",
                mime="text/csv",
                key="download_regional_analysis",
            )

# ------------------------------ Fig. 12: ML Prediction Results ------------------------------
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

    # Train the live model from the supplied dataset, using registration count
    # only to create the target class; prediction inputs remain state, RTO, maker.
    data_signature = (
        len(data),
        int(data["count"].sum()),
        int(data["id"].sum()),
    )
    try:
        live_model, trained_rows, class_counts = train_live_model(
            data_signature, data
        )
    except Exception as exc:
        st.error(f"Could not train the Logistic Regression model: {exc}")
        st.stop()

    st.markdown("### Model Information")
    info1, info2, info3 = st.columns(3)
    info1.metric("Algorithm", "Logistic Regression")
    info2.metric("Training Records", f"{trained_rows:,}")
    info3.metric("Input Features", "3")
    st.caption(
        "The model uses State, RTO, and Vehicle Maker as input features. "
        "Registration count is used only to assign training labels."
    )

    st.markdown("### Registration Category Rules")
    rule1, rule2, rule3 = st.columns(3)
    rule1.metric("Low", "0–25")
    rule2.metric("Medium", "26–176")
    rule3.metric("High", "Above 176")

    st.markdown("### Make a Live Prediction")
    st.write(
        "Choose the vehicle registration details below and select **Predict**. "
        "The trained model will estimate the registration category."
    )

    states = sorted(data["state"].dropna().astype(str).unique().tolist())
    selected_state = st.selectbox(
        "Select State",
        options=states,
        key="live_predict_state",
    )

    state_data = data[data["state"].astype(str) == selected_state]
    rto_options = sorted(state_data["rto"].dropna().astype(str).unique().tolist())
    selected_rto = st.selectbox(
        "Select RTO",
        options=rto_options,
        key="live_predict_rto",
    )

    rto_data = state_data[state_data["rto"].astype(str) == selected_rto]
    maker_options = sorted(rto_data["maker"].dropna().astype(str).unique().tolist())
    selected_maker = st.selectbox(
        "Select Vehicle Maker",
        options=maker_options,
        key="live_predict_maker",
    )

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
        st.markdown("### Prediction Output")
        st.write(
            f"**State:** {result['state']}  \n"
            f"**RTO:** {result['rto']}  \n"
            f"**Vehicle Maker:** {result['maker']}"
        )

        category_colors = {
            "Low": "#2563eb",
            "Medium": "#d97706",
            "High": "#16a34a",
        }
        result_color = category_colors.get(result["category"], "#334155")
        st.markdown(
            f"""
            <div style="
                background:#f8fafc;
                border-radius:14px;
                padding:22px;
                text-align:center;
                border:2px solid {result_color};
                margin:12px 0;
            ">
                <div style="color:#475569;font-size:14px;font-weight:600;">
                    Predicted Registration Category
                </div>
                <div style="color:{result_color};font-size:34px;font-weight:900;">
                    {result['category']}
                </div>
                <div style="color:#475569;font-size:13px;">
                    Model confidence: {result['confidence']:.2f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### Class Probability")
        probability_frame = pd.DataFrame({
            "Category": list(result["probabilities"].keys()),
            "Probability": [float(v) for v in result["probabilities"].values()],
        })
        st.bar_chart(probability_frame.set_index("Category"))

        st.caption(
            "Confidence is the highest predicted class probability. "
            "This is a classification estimate based on the dataset patterns, "
            "not a forecast of future registrations."
        )

    st.markdown("---")
    st.markdown("### Training Class Distribution")
    distribution = pd.DataFrame(
        [{"Category": key, "Records": int(value)}
         for key, value in class_counts.items()]
    ).sort_values("Category")
    st.bar_chart(distribution.set_index("Category"))

    # Keep the previously evaluated hold-out metrics available as a separate
    # evaluation reference, if the generated CSVs exist.
    try:
        saved_metrics, saved_predictions = load_ml_results()
    except Exception:
        saved_metrics, saved_predictions = None, None

    if saved_metrics is not None:
        st.markdown("### Previously Evaluated Model Performance")
        st.caption(
            "These are the saved hold-out evaluation metrics from ml_prediction.py; "
            "they are separate from the live model fitted above."
        )
        metric_map = dict(zip(saved_metrics["Metric"], saved_metrics["Score"]))
        metric_cols = st.columns(4)
        for col, metric_name in zip(
            metric_cols, ["Accuracy", "Precision", "Recall", "F1 Score"]
        ):
            value = metric_map.get(metric_name)
            if value is not None:
                col.metric(metric_name, f"{float(value) * 100:.2f}%")

