import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import textwrap
from dashboard_common import (
    apply_saved_filters,
    dashboard_kpis,
    dashboard_style,
    load_ml_results,
    load_vehicle_data,
    save_filter,
)

def render_home():
    st.markdown("""
    <style>
    .stApp { background:#82cbe5; }
    .home-card { max-width:690px; margin:8vh auto 0; background:#f6fbfc; border-radius:16px;
        padding:38px 44px; box-shadow:0 14px 30px rgba(15,23,42,.18); }
    .home-icon { font-size:2rem; margin-bottom:8px; }
    .home-title { color:#263246; font-size:2.4rem; line-height:1.02; font-weight:900; margin:0; }
    .home-copy { color:#64748b; font-size:.95rem; line-height:1.55; margin:14px 0 22px; }
    .home-actions { display:flex; gap:10px; }
    .home-actions a { flex:1; text-align:center; text-decoration:none; color:#fff !important;
        padding:10px 12px; border-radius:6px; font-weight:700; background:#2563eb; }
    .home-actions a:last-child { background:#16a34a; }
    </style>
    <div class="home-card">
      <div class="home-icon">🚘</div>
      <h1 class="home-title">VEHICLE REGISTRATION ANALYTICS</h1>
      <p class="home-copy">Explore registration patterns by state, RTO, and vehicle manufacturer. Use the top menu to open filters, regional analysis, project information, and machine learning results.</p>
      <p style="color:#087f73;font-weight:700;margin:0;">Choose a page from the menu above to get started.</p>
    </div>
    """, unsafe_allow_html=True)

def render_features():
    st.markdown('<div class="dash-top">KEY FEATURES / PROJECT FEATURES</div>', unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.markdown("### Interactive registration filters")
            st.write("Filter the vehicle registration dataset by State / UT, RTO, and vehicle maker. Selections carry into the regional analysis page.")
            st.markdown("### Regional comparisons")
            st.write("Compare registration counts across states and RTOs, then explore their contribution to the selected data.")
    with right:
        with st.container(border=True):
            st.markdown("### Manufacturer analysis")
            st.write("Review leading vehicle makers and their share of registrations.")
            st.markdown("### Machine learning results")
            st.write("Review model metrics, predicted categories, the confusion matrix, and test set predictions.")

def render_mission():
    st.markdown('<div class="dash-top">INFORMATION / OUR MISSION PAGE</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="max-width:900px;margin:18px auto;background:#fff;border:1px solid #d6e3e0;border-radius:8px;padding:26px 30px;">
    <h2 style="color:#163b38;margin-top:0;">Our Mission</h2>
    <p style="color:#52616b;line-height:1.8;">To provide an accessible Big Data framework for understanding vehicle registration patterns across Indian states, RTO regions, and vehicle manufacturers.</p>
    <p style="color:#52616b;line-height:1.8;">The project brings together data processing, regional comparisons, manufacturer analysis, and machine learning classification in one dashboard.</p>
    </div>
    """, unsafe_allow_html=True)

def render_filters():
    st.markdown('<div class="dash-top">ANALYTICS FILTERS AND DATA SELECTION</div>', unsafe_allow_html=True)
    data = load_vehicle_data()
    states = sorted(data["state"].dropna().unique().tolist())
    rtos = sorted(data["rto"].dropna().unique().tolist())
    makers = sorted(data["maker"].dropna().unique().tolist())
    filter_options = [
        ("state_filter_widget", "dashboard_state", "STATE / UT", ["All states"] + states, "All states"),
        ("rto_filter_widget", "dashboard_rto", "RTO", ["All RTOs"] + rtos, "All RTOs"),
        ("maker_filter_widget", "dashboard_maker", "VEHICLE MAKER", ["All makers"] + makers, "All makers"),
    ]
    cols = st.columns(3)
    for col, (widget_key, saved_key, label, options, default) in zip(cols, filter_options):
        if st.session_state.get(saved_key) not in options:
            st.session_state[saved_key] = default
        if widget_key not in st.session_state:
            st.session_state[widget_key] = st.session_state[saved_key]
        with col:
            st.markdown(f'<div class="dash-filter-label">{label}</div>', unsafe_allow_html=True)
            st.selectbox(label, options, key=widget_key, label_visibility="collapsed",
                         on_change=save_filter, args=(widget_key, saved_key))
    filtered = apply_saved_filters(data)
    dashboard_kpis(filtered)
    st.markdown('<div class="dash-section">SELECTED REGISTRATION DATA</div>', unsafe_allow_html=True)
    st.caption(f"Showing {len(filtered):,} records. These filters are also used on Regional and Manufacturer Analysis.")
    st.dataframe(filtered, use_container_width=True, hide_index=True)
    st.download_button("Download Selected CSV", data=filtered.to_csv(index=False).encode("utf-8"),
                       file_name="selected_vehicle_registrations.csv", mime="text/csv")

def render_regional():
    st.markdown('<div class="dash-top">REGIONAL AND MANUFACTURER ANALYSIS</div>', unsafe_allow_html=True)
    data = load_vehicle_data()
    filtered = apply_saved_filters(data)
    selection = [st.session_state.get("dashboard_state", "All states"),
                 st.session_state.get("dashboard_rto", "All RTOs"),
                 st.session_state.get("dashboard_maker", "All makers")]
    st.caption("Current filter selection: " + " · ".join(selection))
    dashboard_kpis(filtered)
    st.markdown('<div class="dash-section">REGIONAL REGISTRATION OVERVIEW</div>', unsafe_allow_html=True)
    left, right = st.columns(2)
    for column, group_col, title in [(left, "state", "Registrations by State"),
                                     (right, "rto", "Registrations by RTO")]:
        with column:
            with st.container(border=True):
                st.markdown(f"**{title}**")
                values = filtered.groupby(group_col)["count"].sum().nlargest(10).sort_values()
                if not values.empty:
                    fig, ax = plt.subplots(figsize=(7, 2.8))
                    fig.patch.set_facecolor("white")
                    ax.set_facecolor("white")
                    ax.bar(range(len(values)), values.values, color="#a82732", width=.68)
                    ax.set_xticks(range(len(values)))
                    ax.set_xticklabels(values.index, rotation=35, ha="right", fontsize=8)
                    ax.grid(axis="y", color="#d9e2e0", linewidth=.7)
                    ax.set_axisbelow(True)
                    for side in ("top", "right", "left"):
                        ax.spines[side].set_visible(False)
                    ax.tick_params(axis="y", labelsize=8, length=0)
                    ax.tick_params(axis="x", length=0)
                    fig.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                else:
                    st.info("No records match the selected filters.")
    st.markdown('<div class="dash-section">MANUFACTURER AND REGIONAL MIX</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    pie_specs = [(cols[0], "maker", "VEHICLE MAKER MIX"),
                 (cols[1], "state", "STATE REGISTRATION SHARE"),
                 (cols[2], "rto", "RTO REGISTRATION SHARE")]
    palette = ["#b52e38", "#e29a35", "#168f83", "#558cc5", "#a47ab1"]
    for col, group_col, title in pie_specs:
        with col:
            st.markdown(f'<div class="dash-section">{title}</div>', unsafe_allow_html=True)
            with st.container(border=True):
                values = filtered.groupby(group_col)["count"].sum().nlargest(5)
                if not values.empty:
                    fig, ax = plt.subplots(figsize=(4, 2.7))
                    fig.patch.set_facecolor("white")
                    ax.pie(values.values, labels=values.index, autopct="%1.0f%%", startangle=90,
                           colors=palette, textprops={"fontsize": 7})
                    ax.axis("equal")
                    fig.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                else:
                    st.info("No records match the selected filters.")

def weighted_prediction_metrics(predictions):
    classes = ["Low", "Medium", "High"]
    matrix = pd.crosstab(predictions["actual_category"], predictions["predicted_category"])
    matrix = matrix.reindex(index=classes, columns=classes, fill_value=0)
    total = int(matrix.values.sum())
    if total == 0:
        return {"Accuracy": 0.0, "Precision": 0.0, "Recall": 0.0, "F1 Score": 0.0}
    weighted_precision = weighted_recall = weighted_f1 = 0.0
    for category in classes:
        support = int(matrix.loc[category].sum())
        predicted_count = int(matrix[category].sum())
        true_positive = int(matrix.loc[category, category])
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        weight = support / total
        weighted_precision += precision * weight
        weighted_recall += recall * weight
        weighted_f1 += f1 * weight
    accuracy = sum(int(matrix.loc[label, label]) for label in classes) / total
    return {"Accuracy": accuracy, "Precision": weighted_precision,
            "Recall": weighted_recall, "F1 Score": weighted_f1}

def render_ml():
    dashboard_style()
    st.markdown(textwrap.dedent("""
    <div class="hero">
        <h1>🤖 Machine Learning Prediction Results</h1>
        <p>Logistic Regression classification of vehicle registration records</p>
    </div>
    """), unsafe_allow_html=True)

    st.caption(
        "Classification of vehicle registration records into Low, Medium, "
        "and High registration categories."
    )

    metrics, predictions = load_ml_results()

    if metrics is None or predictions is None:

        st.warning(
            "ML prediction files were not found. "
            "Run ml_prediction.py first."
        )

    else:

        prediction_states = sorted(predictions["state"].dropna().unique().tolist())
        filter_col, state_summary_col = st.columns([1, 2])
        with filter_col:
            st.markdown('<div class="dash-filter-label">STATE / UT</div>', unsafe_allow_html=True)
            selected_state = st.selectbox(
                "Filter ML results by state",
                ["All states"] + prediction_states,
                key="ml_prediction_state",
                label_visibility="collapsed",
            )
        if selected_state != "All states":
            predictions = predictions[predictions["state"] == selected_state].copy()
        if predictions.empty:
            st.info("No test-set predictions are available for this state.")
            st.stop()
        with state_summary_col:
            st.caption(f"Showing {len(predictions):,} test-set predictions for {selected_state}. Metrics, confusion matrix, and results below use this selection.")
        st.markdown(textwrap.dedent("""
        <div class="model-banner">
            <span class="status-pill">MODEL READY</span>
            <h3 style="margin:10px 0 4px;color:#312e81;">
                Logistic Regression Classification
            </h3>
            <p style="margin:0;color:#475569;">
                Classifying vehicle registration records into Low, Medium,
                and High categories using State, RTO, and Maker features.
            </p>
        </div>
        """), unsafe_allow_html=True)

        st.markdown("### Model Information")

        info1, info2, info3, info4 = st.columns(4)

        with info1:
            st.markdown("**Model**")
            st.write("Logistic Regression")

        with info2:
            st.markdown("**Features**")
            st.write("State + RTO + Maker")

        with info3:
            st.markdown("**Classes**")
            st.write("Low / Medium / High")

        with info4:
            st.markdown("**Train/Test**")
            st.write("80% / 20%")

        # ----------------------------------------------------
        # PERFORMANCE METRICS
        # ----------------------------------------------------

        st.markdown('<div class="section-title">📈 Model Performance</div>',
                    unsafe_allow_html=True)

        metric_values = weighted_prediction_metrics(predictions)

        accuracy = metric_values["Accuracy"]
        precision = metric_values["Precision"]
        recall = metric_values["Recall"]
        f1 = metric_values["F1 Score"]

        m1, m2, m3, m4 = st.columns(4)

        m1.metric("Accuracy", f"{accuracy * 100:.2f}%")
        m2.metric("Precision", f"{precision * 100:.2f}%")
        m3.metric("Recall", f"{recall * 100:.2f}%")
        m4.metric("F1 Score", f"{f1 * 100:.2f}%")

        # ----------------------------------------------------
        # ACTUAL VS PREDICTED
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">🎯 Actual vs Predicted Categories</div>',
            unsafe_allow_html=True
        )

        comparison = (
            predictions.groupby(
                ["actual_category", "predicted_category"]
            )
            .size()
            .reset_index(name="records")
        )

        comparison_pivot = comparison.pivot(
            index="actual_category",
            columns="predicted_category",
            values="records"
        ).fillna(0)

        for column in ["Low", "Medium", "High"]:
            if column not in comparison_pivot.columns:
                comparison_pivot[column] = 0

        comparison_pivot = comparison_pivot[
            ["Low", "Medium", "High"]
        ]

        st.dataframe(
            comparison_pivot.astype(int),
            use_container_width=True
        )

        # ----------------------------------------------------
        # CONFUSION MATRIX
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">🧩 Confusion Matrix</div>',
            unsafe_allow_html=True
        )

        cm = pd.crosstab(
            predictions["actual_category"],
            predictions["predicted_category"],
            rownames=["Actual"],
            colnames=["Predicted"],
            dropna=False
        )

        class_order = ["Low", "Medium", "High"]
        cm = cm.reindex(
            index=class_order,
            columns=class_order,
            fill_value=0
        )

        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#f8fafc")

        image = ax.imshow(cm.values, cmap="viridis")

        ax.set_xticks(range(len(class_order)))
        ax.set_yticks(range(len(class_order)))
        ax.set_xticklabels(class_order)
        ax.set_yticklabels(class_order)

        ax.set_xlabel("Predicted Category")
        ax.set_ylabel("Actual Category")
        ax.set_title(
            "ML Confusion Matrix",
            fontsize=15,
            fontweight="bold",
            pad=14
        )

        for i in range(len(class_order)):
            for j in range(len(class_order)):
                ax.text(
                    j,
                    i,
                    f"{cm.iloc[i, j]:,}",
                    ha="center",
                    va="center"
                )

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # ----------------------------------------------------
        # PREDICTION RESULTS
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">📋 ML Prediction Results</div>',
            unsafe_allow_html=True
        )

        st.caption(
            f"Showing {len(predictions):,} test-set predictions."
        )

        st.markdown(textwrap.dedent("""
        <div style="display:flex;gap:10px;flex-wrap:wrap;margin:8px 0 14px;">
            <span style="background:#dcfce7;color:#166534;padding:7px 12px;border-radius:999px;font-weight:700;">LOW</span>
            <span style="background:#fef3c7;color:#92400e;padding:7px 12px;border-radius:999px;font-weight:700;">MEDIUM</span>
            <span style="background:#fee2e2;color:#991b1b;padding:7px 12px;border-radius:999px;font-weight:700;">HIGH</span>
        </div>
        """), unsafe_allow_html=True)

        st.dataframe(
            predictions,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download ML Predictions",
            data=predictions.to_csv(index=False).encode("utf-8"),
            file_name="ml_predictions.csv",
            mime="text/csv"
        )
