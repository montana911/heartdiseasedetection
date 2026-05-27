"""
=============================================================
  Cardiovascular Risk Prediction — Streamlit Web App
  Thesis Defence Demo  |  Bachelor CS  |  2025-2026
=============================================================
 browser will open at  http://localhost:8501
=============================================================
"""

import warnings
warnings.filterwarnings("ignore")

import os
import pickle
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import (RandomForestClassifier,
                                     HistGradientBoostingClassifier,
                                     VotingClassifier)
from sklearn.metrics         import (roc_curve, confusion_matrix,
                                     accuracy_score, precision_score,
                                     recall_score, f1_score, roc_auc_score)
from imblearn.over_sampling  import SMOTE


# =============================================================
# PAGE CONFIGURATION
# =============================================================
st.set_page_config(
    page_title="Cardiovascular Risk Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a cleaner look
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1f4e79;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #555;
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
    }
    .stat-box {
        background-color: #f0f4f8;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f4e79;
        margin: 0.5rem 0;
    }
    .risk-high {
        background: linear-gradient(135deg, #ffe5e5 0%, #ffcccc 100%);
        border-left: 5px solid #c0392b;
        padding: 1.2rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .risk-medium {
        background: linear-gradient(135deg, #fff5e5 0%, #ffe8cc 100%);
        border-left: 5px solid #e67e22;
        padding: 1.2rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .risk-low {
        background: linear-gradient(135deg, #e5f7e8 0%, #ccf2d4 100%);
        border-left: 5px solid #27ae60;
        padding: 1.2rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================
# CONSTANTS
# =============================================================
SEED = 42
MODEL_PATH = Path("model_cache.pkl")

AGE_ORDER = [
    "18-24", "25-29", "30-34", "35-39", "40-44", "45-49",
    "50-54", "55-59", "60-64", "65-69", "70-74", "75-79",
    "80 or older",
]

RACE_OPTIONS = [
    "White", "Black", "Hispanic", "Asian",
    "American Indian/Alaskan Native", "Other",
]

DIABETIC_OPTIONS = [
    "No", "No, borderline diabetes",
    "Yes", "Yes (during pregnancy)",
]

GENHEALTH_OPTIONS = ["Excellent", "Very good", "Good", "Fair", "Poor"]


# =============================================================
# MODEL TRAINING — runs once, then cached
# =============================================================
@st.cache_resource
def train_and_cache_model():
    """
    Train the full pipeline and cache the model + scaler + feature names.
    Returns a dict with everything needed for live prediction.
    """
    # Try to load from disk first
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)

    # Otherwise train from scratch
    if not (Path("heart_2020_cleaned_final.csv").exists() and
            Path("heart_2022_cleaned_final.csv").exists()):
        st.error("CSV files not found. Place heart_2020_cleaned_final.csv "
                 "and heart_2022_cleaned_final.csv in the same folder as app.py.")
        st.stop()

    # Load and combine
    df1 = pd.read_csv("heart_2020_cleaned_final.csv")
    df2 = pd.read_csv("heart_2022_cleaned_final.csv")
    df = pd.concat([df1, df2], ignore_index=True)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # Encode
    yes_no_cols = [
        "HeartDisease", "Smoking", "AlcoholDrinking", "Stroke",
        "DiffWalking", "PhysicalActivity", "Asthma",
        "KidneyDisease", "SkinCancer",
    ]
    for c in yes_no_cols:
        df[c] = df[c].map({"Yes": 1, "No": 0})

    df["AgeCategory"] = pd.Categorical(
        df["AgeCategory"], categories=AGE_ORDER, ordered=True
    ).codes

    df_enc = pd.get_dummies(
        df, columns=["Sex", "Race", "Diabetic", "GenHealth"],
        drop_first=True
    )

    X = df_enc.drop("HeartDisease", axis=1)
    y = df_enc["HeartDisease"]
    feature_names = list(X.columns)

    # Split + scale + SMOTE
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED
    )
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)

    X_tr_bal, y_tr_bal = SMOTE(random_state=SEED).fit_resample(X_tr_sc, y_tr)

    # Train all models
    lr  = LogisticRegression(max_iter=1000, class_weight="balanced",
                             random_state=SEED)
    rf  = RandomForestClassifier(n_estimators=200, max_depth=15,
                                 class_weight="balanced",
                                 n_jobs=-1, random_state=SEED)
    hgb = HistGradientBoostingClassifier(max_iter=100, max_depth=5,
                                         random_state=SEED)

    voting = VotingClassifier(
        estimators=[("lr", lr), ("rf", rf), ("hgb", hgb)],
        voting="soft", n_jobs=-1
    )

    # Train Voting (which trains all base models internally)
    voting.fit(X_tr_bal, y_tr_bal)

    # Compute metrics for the dashboard tab
    metrics = {}
    for name, model_obj in [
        ("Logistic Regression", voting.named_estimators_["lr"]),
        ("Random Forest",       voting.named_estimators_["rf"]),
        ("HistGradBoost",       voting.named_estimators_["hgb"]),
        ("Voting Ensemble",     voting),
    ]:
        y_pred = model_obj.predict(X_te_sc)
        y_prob = model_obj.predict_proba(X_te_sc)[:, 1]
        metrics[name] = {
            "accuracy":  accuracy_score(y_te, y_pred),
            "precision": precision_score(y_te, y_pred),
            "recall":    recall_score(y_te, y_pred),
            "f1":        f1_score(y_te, y_pred),
            "auc":       roc_auc_score(y_te, y_prob),
            "y_pred":    y_pred,
            "y_prob":    y_prob,
        }

    # Feature importance from Random Forest
    rf_trained = voting.named_estimators_["rf"]
    importances = pd.Series(
        rf_trained.feature_importances_, index=feature_names
    ).sort_values(ascending=False)

    cache = {
        "voting":         voting,
        "scaler":         scaler,
        "feature_names":  feature_names,
        "metrics":        metrics,
        "y_te":           y_te.values,
        "importances":    importances,
        "n_records":      len(df),
        "positive_rate":  y.mean(),
    }

    # Save to disk for next run
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(cache, f)

    return cache


# =============================================================
# FEATURE VECTOR BUILDER
# =============================================================
def build_feature_vector(inputs, feature_names):
    """Convert user inputs dict to a row matching the training schema."""
    row = {f: 0 for f in feature_names}

    # Numeric and ordinal
    row["BMI"]              = inputs["bmi"]
    row["PhysicalHealth"]   = inputs["physical_health"]
    row["MentalHealth"]     = inputs["mental_health"]
    row["SleepTime"]        = inputs["sleep_time"]
    row["AgeCategory"]      = AGE_ORDER.index(inputs["age"])

    # Binary yes/no
    row["Smoking"]          = 1 if inputs["smoking"] == "Yes" else 0
    row["AlcoholDrinking"]  = 1 if inputs["alcohol"] == "Yes" else 0
    row["Stroke"]           = 1 if inputs["stroke"] == "Yes" else 0
    row["DiffWalking"]      = 1 if inputs["diff_walking"] == "Yes" else 0
    row["PhysicalActivity"] = 1 if inputs["physical_activity"] == "Yes" else 0
    row["Asthma"]           = 1 if inputs["asthma"] == "Yes" else 0
    row["KidneyDisease"]    = 1 if inputs["kidney_disease"] == "Yes" else 0
    row["SkinCancer"]       = 1 if inputs["skin_cancer"] == "Yes" else 0

    # One-hot Sex (drop_first removed "Female")
    if inputs["sex"] == "Male":
        row["Sex_Male"] = 1

    # One-hot Race (drop_first removed "American Indian/Alaskan Native")
    race_key = f"Race_{inputs['race']}"
    if race_key in row:
        row[race_key] = 1

    # One-hot Diabetic (drop_first removed "No")
    diabetic_key = f"Diabetic_{inputs['diabetic']}"
    if diabetic_key in row:
        row[diabetic_key] = 1

    # One-hot GenHealth (drop_first removed "Excellent")
    genhealth_key = f"GenHealth_{inputs['gen_health']}"
    if genhealth_key in row:
        row[genhealth_key] = 1

    return pd.DataFrame([row])[feature_names]


# =============================================================
# MAIN APP
# =============================================================
st.markdown('<div class="main-title">❤️ Cardiovascular Risk Predictor</div>',
            unsafe_allow_html=True)
st.markdown('<div class="subtitle">Hybrid Ensemble Machine Learning | '
            'Bachelor Thesis Demo</div>', unsafe_allow_html=True)

# Load/train model
with st.spinner("Loading model... (first run takes ~2-3 minutes)"):
    cache = train_and_cache_model()

# ─── Tabs ────────────────────────────────────────────────────
tab_predict, tab_performance, tab_about = st.tabs([
    "🔮 Live Prediction",
    "📊 Model Performance",
    "📖 About the Project",
])

# =============================================================
# TAB 1 — LIVE PREDICTION
# =============================================================
with tab_predict:
    st.markdown("### Enter Patient Information")
    st.markdown("Fill in the form on the left, then click **Predict Risk** "
                "to see the model's assessment.")

    with st.sidebar:
        st.header("🩺 Patient Information")
        st.caption("Adjust the inputs and click Predict.")

        st.subheader("Demographics")
        age = st.selectbox("Age Category", AGE_ORDER, index=8)
        sex = st.radio("Sex", ["Male", "Female"], horizontal=True)
        race = st.selectbox("Race / Ethnicity", RACE_OPTIONS)

        st.subheader("Physical Measurements")
        bmi = st.slider("BMI (Body Mass Index)",
                        min_value=12.0, max_value=60.0,
                        value=27.0, step=0.1)
        sleep_time = st.slider("Average Sleep (hours per night)",
                               min_value=1, max_value=14, value=7)

        st.subheader("Health History")
        c1, c2 = st.columns(2)
        with c1:
            smoking = st.radio("Ever smoked 100+ cigarettes?",
                               ["No", "Yes"], horizontal=True)
            alcohol = st.radio("Heavy drinker?",
                               ["No", "Yes"], horizontal=True)
            stroke = st.radio("Ever had a stroke?",
                              ["No", "Yes"], horizontal=True)
            asthma = st.radio("Has asthma?",
                              ["No", "Yes"], horizontal=True)
        with c2:
            diff_walking = st.radio("Difficulty walking?",
                                    ["No", "Yes"], horizontal=True)
            kidney_disease = st.radio("Kidney disease?",
                                      ["No", "Yes"], horizontal=True)
            skin_cancer = st.radio("Skin cancer history?",
                                   ["No", "Yes"], horizontal=True)
            physical_activity = st.radio("Active in past 30 days?",
                                         ["Yes", "No"], horizontal=True)

        st.subheader("Recent Health Status")
        physical_health = st.slider("Days of poor physical health (last 30)",
                                    min_value=0, max_value=30, value=0)
        mental_health = st.slider("Days of poor mental health (last 30)",
                                  min_value=0, max_value=30, value=0)
        gen_health = st.selectbox("Self-rated general health",
                                  GENHEALTH_OPTIONS, index=2)
        diabetic = st.selectbox("Diabetic?", DIABETIC_OPTIONS)

        predict_btn = st.button("🔮 Predict Risk",
                                type="primary",
                                use_container_width=True)

    if predict_btn:
        inputs = {
            "age": age, "sex": sex, "race": race,
            "bmi": bmi, "sleep_time": sleep_time,
            "smoking": smoking, "alcohol": alcohol, "stroke": stroke,
            "asthma": asthma, "diff_walking": diff_walking,
            "kidney_disease": kidney_disease, "skin_cancer": skin_cancer,
            "physical_activity": physical_activity,
            "physical_health": physical_health,
            "mental_health": mental_health,
            "gen_health": gen_health, "diabetic": diabetic,
        }

        X_user = build_feature_vector(inputs, cache["feature_names"])
        X_user_sc = cache["scaler"].transform(X_user)
        probability = cache["voting"].predict_proba(X_user_sc)[0, 1]

        # ─── Risk display ─────────────────────────────────
        col_a, col_b = st.columns([1, 1])

        with col_a:
            st.markdown("### Risk Assessment")

            if probability >= 0.5:
                risk_label = "HIGH RISK"
                risk_class = "risk-high"
                recommendation = (
                    "The model flags this profile as elevated risk. "
                    "**Recommendation:** Clinical follow-up advised. This is a "
                    "screening result, not a diagnosis."
                )
            elif probability >= 0.25:
                risk_label = "MODERATE RISK"
                risk_class = "risk-medium"
                recommendation = (
                    "The model places this profile in the moderate risk zone. "
                    "**Recommendation:** Lifestyle review and routine "
                    "monitoring suggested."
                )
            else:
                risk_label = "LOW RISK"
                risk_class = "risk-low"
                recommendation = (
                    "The model indicates low predicted risk. "
                    "**Recommendation:** Continue standard preventive care. "
                    "Risk profile can change — periodic reassessment is "
                    "still wise."
                )

            st.markdown(
                f'<div class="{risk_class}">'
                f'<h2 style="margin:0">{risk_label}</h2>'
                f'<h1 style="margin:0;font-size:3rem">{probability:.1%}</h1>'
                f'<p style="margin:0.5rem 0 0 0">Predicted probability of '
                f'cardiovascular disease</p>'
                f'</div>',
                unsafe_allow_html=True
            )

            st.info(recommendation)

            st.caption(
                "**Important disclaimer:** This tool is a thesis project "
                "demonstration. It is **not** a medical device. Any clinical "
                "decision must be made by a qualified healthcare professional."
            )

        with col_b:
            st.markdown("### Risk Visualisation")

            # Risk gauge using matplotlib
            fig, ax = plt.subplots(figsize=(6, 3.5))
            colors = ["#27ae60", "#f39c12", "#c0392b"]
            zones = [(0, 25), (25, 50), (50, 100)]
            for (lo, hi), c in zip(zones, colors):
                ax.barh([0], [hi - lo], left=[lo], color=c, height=0.4, alpha=0.7)

            ax.axvline(probability * 100, color="black", linewidth=3)
            ax.text(probability * 100, 0.35,
                    f"{probability:.1%}",
                    ha="center", fontsize=13, fontweight="bold")

            ax.set_xlim(0, 100)
            ax.set_ylim(-0.5, 0.6)
            ax.set_yticks([])
            ax.set_xticks([0, 25, 50, 75, 100])
            ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
            ax.set_xlabel("Predicted Risk Probability", fontweight="bold")
            ax.text(12.5, -0.3, "LOW", ha="center", fontweight="bold",
                    color="#27ae60")
            ax.text(37.5, -0.3, "MODERATE", ha="center", fontweight="bold",
                    color="#e67e22")
            ax.text(75, -0.3, "HIGH", ha="center", fontweight="bold",
                    color="#c0392b")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.markdown("### Top Risk Factors in This Prediction")
            # Show how the user's inputs align with the top 8 features
            top_features = cache["importances"].head(8)
            feature_table = []
            for fname, importance in top_features.items():
                user_val = X_user[fname].values[0]
                feature_table.append({
                    "Feature": fname,
                    "Your Value": f"{user_val:.2f}" if isinstance(user_val, float)
                                   else int(user_val),
                    "Importance": f"{importance:.3f}",
                })
            st.dataframe(pd.DataFrame(feature_table),
                         hide_index=True, use_container_width=True)
    else:
        st.markdown("---")
        st.markdown(
            "👈 **Adjust the inputs in the sidebar and click "
            "Predict Risk to see the assessment.**"
        )
        st.markdown(
            "The model was trained on 538,510 BRFSS survey records and "
            "achieves an AUC-ROC of 0.834 on held-out test data."
        )


# =============================================================
# TAB 2 — MODEL PERFORMANCE
# =============================================================
with tab_performance:
    st.markdown("### Model Performance on Held-Out Test Set")
    st.caption(
        f"Trained on {cache['n_records']:,} records "
        f"({cache['positive_rate']:.1%} positive rate). "
        f"Evaluated on a stratified 20% test set."
    )

    # Metrics summary table
    metrics_df = pd.DataFrame({
        name: {k: v for k, v in m.items() if k in
               ["accuracy", "precision", "recall", "f1", "auc"]}
        for name, m in cache["metrics"].items()
    }).T
    metrics_df.columns = ["Accuracy", "Precision", "Recall",
                          "F1-Score", "AUC-ROC"]
    metrics_df = metrics_df.round(4)

    st.dataframe(metrics_df, use_container_width=True)

    st.markdown("""
    > **Note on F1-Score**: F1 in the 0.34 to 0.39 range is the
    > mathematically expected consequence of the 9% class imbalance.
    > **AUC-ROC and Recall are the meaningful comparison metrics**
    > for this screening task.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### ROC Curves")
        fig, ax = plt.subplots(figsize=(7, 6))
        colors = {"Logistic Regression": "#3498db", "Random Forest": "#e67e22",
                  "HistGradBoost": "#2ecc71", "Voting Ensemble": "#e74c3c"}

        for name, m in cache["metrics"].items():
            fpr, tpr, _ = roc_curve(cache["y_te"], m["y_prob"])
            ax.plot(fpr, tpr, lw=2, color=colors[name],
                    label=f"{name} (AUC={m['auc']:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
        ax.set_xlabel("False Positive Rate", fontweight="bold")
        ax.set_ylabel("True Positive Rate", fontweight="bold")
        ax.set_title("Receiver Operating Characteristic")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.markdown("#### Confusion Matrix — Voting Ensemble")
        cm = confusion_matrix(cache["y_te"],
                              cache["metrics"]["Voting Ensemble"]["y_pred"])
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["No Disease", "Heart Disease"])
        ax.set_yticklabels(["No Disease", "Heart Disease"])
        ax.set_xlabel("Predicted", fontweight="bold")
        ax.set_ylabel("Actual", fontweight="bold")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i,j]:,}",
                        ha="center", va="center",
                        color="white" if cm[i,j] > cm.max()/2 else "black",
                        fontsize=15, fontweight="bold")
        ax.set_title("Voting Ensemble — Confusion Matrix")
        plt.colorbar(im, ax=ax)
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("### Top 14 Predictive Features (Random Forest)")
    fig, ax = plt.subplots(figsize=(11, 5.5))
    top = cache["importances"].head(14).sort_values()
    ax.barh(top.index, top.values, color="#e74c3c",
            edgecolor="white", alpha=0.85)
    ax.set_xlabel("Mean Gini Importance", fontweight="bold")
    ax.set_title("Most Important Features for Cardiovascular Risk Prediction")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# =============================================================
# TAB 3 — ABOUT
# =============================================================
with tab_about:
    st.markdown("""
    ## About This Project

    This web application is the live demonstration component of a bachelor
    graduation thesis in Computer Science. The underlying machine learning
    pipeline is the same one described in the thesis document — no
    simplifications.

    ### The Goal
    Cardiovascular disease accounts for roughly 17.9 million deaths globally
    every year. Many of those deaths are preventable if elevated risk is
    identified early enough, but standard clinical tools (ECG, blood panels,
    imaging) are too slow and expensive for population-scale screening.
    This project investigates whether routine health survey data — the kind
    public health agencies already collect — contains enough predictive
    signal to act as a low-cost first-pass screening layer.

    ### The Data
    The model is trained on merged data from the 2020 and 2022 cycles of
    the U.S. CDC Behavioral Risk Factor Surveillance System (BRFSS):

    - **538,510 records** after deduplication and outlier removal
    - **18 features** per record covering demographics, lifestyle,
      health history, and self-reported wellness
    - **~9% positive class rate** (matches the U.S. adult prevalence
      of diagnosed cardiovascular disease)

    ### The Model
    A Soft Voting Ensemble combining three classifiers with complementary
    inductive biases:

    1. **Logistic Regression** — global linear boundary, calibrated probabilities
    2. **Random Forest (200 trees, depth 15)** — non-linear interactions,
       feature importance rankings
    3. **HistGradient Boosting (100 iterations)** — sequential error correction,
       efficient at scale

    The ensemble averages predicted probabilities from all three. Stacking
    was tested during development but rejected — the base models correlate
    at r > 0.94 on this data, leaving the meta-learner with no useful signal
    to combine.

    ### The Results
    On a held-out test set of 107,702 records:

    - **AUC-ROC = 0.834** (Voting Ensemble)
    - **Recall = 0.625** at default threshold
    - **Precision = 0.278**
    - **Cross-validation std deviation < 0.005** across five folds

    Note: F1-score reads low (~0.35) because of the 9% class imbalance.
    This is mathematically expected, not a model weakness. AUC-ROC and
    Recall are the metrics that matter for a screening application.

    ### Methodology Safeguards
    Several methodology choices keep the evaluation honest:

    - SMOTE oversampling applied **only** to the training partition,
      never before the train/test split
    - StandardScaler fitted **only** on the training partition
    - Cross-validation includes SMOTE and scaling **inside** each fold
      (no preprocessing leakage between folds)
    - Accuracy is reported but never used as a primary metric
      (misleading on imbalanced data)

    ### Important Disclaimer
    **This is a thesis project. It is not a medical device.**
    The model produces a statistical risk estimate based on patterns in
    historical survey data. It cannot replace a clinical examination,
    diagnostic tests, or medical advice. Any health decision should be
    made in consultation with a qualified healthcare professional.

    ---

    *Cardiovascular Risk Prediction Using a Hybrid Ensemble Learning Approach*
    *Bachelor Thesis, Computer Science, 2025-2026*
    """)
