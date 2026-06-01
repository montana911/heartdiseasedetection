"""
Streamlit Web Application
=========================

The web interface for the trained Heart Disease Prediction pipeline.
The actual machine learning logic lives in the ``src`` package — this
file is purely UI. It loads a pre-trained pipeline via
``HeartDiseasePipeline.load()`` and uses its public methods for live
prediction and metric display.

Run from the project root:

    streamlit run app.py

The app expects a trained pipeline at ``models/pipeline.pkl``. If one
does not exist, the user is prompted to run ``python train.py`` first.
"""

from __future__ import annotations
from pathlib import Path

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

from src             import HeartDiseasePipeline, ResultsVisualizer
from src.config      import (
    AGE_BRACKETS, RISK_BANDS, get_risk_band, MODEL_DIR,
)


# ─── Page setup ──────────────────────────────────────────────
st.set_page_config(
    page_title="Cardiovascular Risk Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── Pipeline loader ─────────────────────────────────────────
@st.cache_resource(show_spinner="Loading trained pipeline …")
def load_pipeline() -> HeartDiseasePipeline:
    """
    Load the trained pipeline from disk. Cached so subsequent
    reruns of the Streamlit app are instant.
    """
    path = MODEL_DIR / "pipeline.pkl"
    if not path.exists():
        st.error(
            f"No trained pipeline found at `{path}`. "
            f"Train one first:\n\n```bash\npython train.py\n```"
        )
        st.stop()
    return HeartDiseasePipeline.load(path)


pipeline = load_pipeline()


# ─── Sidebar form ────────────────────────────────────────────
def render_input_form() -> dict:
    """Render the patient input form and return the input dictionary."""
    with st.sidebar:
        st.header("🩺 Patient Information")
        st.caption("Adjust the inputs and click Predict.")

        st.subheader("Demographics")
        age = st.selectbox("Age Category", AGE_BRACKETS, index=8)
        sex = st.radio("Sex", ["Male", "Female"], horizontal=True)
        race = st.selectbox("Race / Ethnicity", [
            "White", "Black", "Hispanic", "Asian",
            "American Indian/Alaskan Native", "Other"
        ])

        st.subheader("Physical Measurements")
        bmi        = st.slider("BMI", 12.0, 60.0, 27.0, step=0.1)
        sleep_time = st.slider("Sleep (hours/night)", 1, 14, 7)

        st.subheader("Health History")
        c1, c2 = st.columns(2)
        with c1:
            smoking        = st.radio("Smoked 100+ cigarettes?", ["No","Yes"], horizontal=True)
            alcohol        = st.radio("Heavy drinker?",          ["No","Yes"], horizontal=True)
            stroke         = st.radio("Ever had a stroke?",      ["No","Yes"], horizontal=True)
            asthma         = st.radio("Has asthma?",             ["No","Yes"], horizontal=True)
        with c2:
            diff_walking      = st.radio("Difficulty walking?",   ["No","Yes"], horizontal=True)
            kidney_disease    = st.radio("Kidney disease?",       ["No","Yes"], horizontal=True)
            skin_cancer       = st.radio("Skin cancer history?",  ["No","Yes"], horizontal=True)
            physical_activity = st.radio("Active in past 30 days?", ["Yes","No"], horizontal=True)

        st.subheader("Recent Health Status")
        physical_health = st.slider("Days of poor physical health (0-30)", 0, 30, 0)
        mental_health   = st.slider("Days of poor mental health (0-30)",   0, 30, 0)
        gen_health      = st.selectbox(
            "Self-rated health", ["Excellent","Very good","Good","Fair","Poor"], index=2)
        diabetic        = st.selectbox(
            "Diabetic?",
            ["No","No, borderline diabetes","Yes","Yes (during pregnancy)"])

        st.divider()
        predict = st.button("🔮 Predict Risk", type="primary",
                            use_container_width=True)

    return {
        "predict_clicked": predict,
        "inputs": {
            "age": age, "sex": sex, "race": race,
            "bmi": bmi, "sleep_time": sleep_time,
            "smoking": smoking, "alcohol": alcohol,
            "stroke": stroke, "asthma": asthma,
            "diff_walking": diff_walking,
            "kidney_disease": kidney_disease, "skin_cancer": skin_cancer,
            "physical_activity": physical_activity,
            "physical_health": physical_health,
            "mental_health":   mental_health,
            "gen_health": gen_health, "diabetic": diabetic,
        },
    }


# ─── Result rendering ─────────────────────────────────────────
def render_prediction_card(probability: float) -> None:
    """Display the prediction in a coloured risk card."""
    band = get_risk_band(probability)

    bg_map = {"LOW RISK": "#e5f7e8", "MODERATE RISK": "#fff5e5", "HIGH RISK": "#ffe5e5"}
    bg     = bg_map.get(band.name, "#f0f0f0")

    st.markdown(
        f'<div style="background:{bg}; border-left:5px solid {band.color};'
        f' padding:1.2rem; border-radius:8px; margin:1rem 0;">'
        f'<h2 style="margin:0;color:{band.color}">{band.name}</h2>'
        f'<h1 style="margin:0;font-size:3rem">{probability:.1%}</h1>'
        f'<p style="margin:0.5rem 0 0 0">Predicted probability of '
        f'cardiovascular disease</p></div>',
        unsafe_allow_html=True,
    )
    st.info(band.advice)
    st.caption(
        "**Disclaimer:** This is a bachelor thesis project, not a "
        "medical device. Clinical decisions require a qualified "
        "healthcare professional."
    )


def render_risk_gauge(probability: float) -> None:
    """Plot the risk probability as a horizontal banded gauge."""
    fig, ax = plt.subplots(figsize=(6, 3))
    for band in RISK_BANDS:
        ax.barh([0],
                [band.upper_bound - band.lower_bound],
                left=[band.lower_bound * 100],
                color=band.color, height=0.45, alpha=0.7)

    ax.axvline(probability * 100, color="black", linewidth=3)
    ax.text(probability * 100, 0.35, f"{probability:.1%}",
            ha="center", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 0.6)
    ax.set_yticks([])
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Predicted Risk Probability", fontweight="bold")
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ─── Main layout ──────────────────────────────────────────────
st.title("❤️ Cardiovascular Risk Predictor")
st.caption("Hybrid Ensemble Machine Learning  ·  Bachelor Thesis Demo")

form_state = render_input_form()

tab_predict, tab_performance, tab_about = st.tabs([
    "🔮 Live Prediction", "📊 Model Performance", "📖 About",
])

# Tab 1 — Live prediction
with tab_predict:
    if form_state["predict_clicked"]:
        probability, _ = pipeline.predict_single(form_state["inputs"])

        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Risk Assessment")
            render_prediction_card(probability)

        with col2:
            st.subheader("Risk Visualisation")
            render_risk_gauge(probability)

            st.subheader("Top Predictive Factors")
            importance = pipeline.get_feature_importance().head(8)
            st.dataframe(importance.rename("Importance").to_frame(),
                         use_container_width=True)
    else:
        st.markdown(
            "👈 **Fill in the sidebar and click Predict Risk to see the "
            "model's assessment.**"
        )

# Tab 2 — Model performance
with tab_performance:
    st.subheader("Test-Set Metrics")
    st.dataframe(pipeline.evaluator.comparison_table(),
                 use_container_width=True)

    st.markdown(
        "> **Note on F1**: F1 around 0.35 is mathematically expected at "
        "9% class prevalence. **AUC-ROC and Recall** are the meaningful "
        "metrics for this screening task."
    )

    viz = ResultsVisualizer(pipeline.evaluator)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("ROC Curves")
        st.pyplot(viz.plot_roc_curves())
    with col2:
        st.subheader("Confusion Matrix — Voting Ensemble")
        st.pyplot(viz.plot_confusion_matrix("VotingEnsemble"))

    st.subheader("Feature Importance")
    st.pyplot(viz.plot_feature_importance(
        feature_names=pipeline.preprocessor.get_feature_names(),
        importances=pipeline.ensemble.base_models[
            "RandomForest"].feature_importances_,
    ))

# Tab 3 — About
with tab_about:
    st.markdown(
        """
        ### About This Project

        Bachelor thesis on cardiovascular risk prediction using a hybrid
        ensemble. Trained on 538,510 records from the CDC's Behavioral
        Risk Factor Surveillance System (2020 + 2022 cycles).

        **Models combined in the Voting Ensemble**

        - Logistic Regression — global linear boundary, calibrated probabilities
        - Random Forest — 200 trees, feature importance, non-linear interactions
        - HistGradient Boosting — sequential error correction, fast at scale

        **Code structure**

        The pipeline is implemented as a Python package (`src/`) with
        single-responsibility modules: `data_loader`, `preprocessor`,
        `imbalance_handler`, `models`, `evaluator`, `visualizer`, and
        `pipeline`. This `app.py` is a thin Streamlit UI on top.

        See the project [GitHub repository](https://github.com/montana911/heartdiseasedetection)
        for full source and documentation.
        """
    )
