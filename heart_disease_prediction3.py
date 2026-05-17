# =============================================================
#  Heart Disease Prediction — Hybrid Machine Learning
#  Bachelor Thesis | Computer Science | 2024-2025
# =============================================================


import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing   import StandardScaler
from sklearn.linear_model    import LogisticRegression
from sklearn.svm             import SVC
from sklearn.ensemble        import (RandomForestClassifier,
                                     StackingClassifier,
                                     VotingClassifier)
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.metrics         import (accuracy_score, precision_score,
                                     recall_score, f1_score,
                                     roc_auc_score, confusion_matrix,
                                     roc_curve, classification_report)
from imblearn.over_sampling  import SMOTE

SEED = 42
np.random.seed(SEED)
sns.set_theme(style="whitegrid")


# =============================================================
# STEP 1 — LOAD THE PRE-CLEANED DATASETS
# =============================================================

print("\n--- STEP 1: Loading data ---")

df_2020 = pd.read_csv("heart_2020_cleaned_final.csv")
df_2022 = pd.read_csv("heart_2022_cleaned_final.csv")

# Combine both datasets into one and shuffle the rows
df = pd.concat([df_2020, df_2022], ignore_index=True)
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

print(f"2020 dataset : {len(df_2020):,} rows")
print(f"2022 dataset : {len(df_2022):,} rows")
print(f"Combined     : {len(df):,} rows")
print(f"Positive rate: {(df['HeartDisease'] == 'Yes').mean():.1%}  "
      f"(people with heart disease)")


# =============================================================
# STEP 2 — ENCODE THE FEATURES
# =============================================================

print("\n--- STEP 2: Encoding features ---")

dp = df.copy()

# A. Convert Yes/No to 1/0
yes_no_cols = [
    "HeartDisease", "Smoking", "AlcoholDrinking", "Stroke",
    "DiffWalking", "PhysicalActivity", "Asthma",
    "KidneyDisease", "SkinCancer"
]
for col in yes_no_cols:
    dp[col] = dp[col].map({"Yes": 1, "No": 0})

# B. Encode age brackets as ordered integers
age_order = [
    "18-24", "25-29", "30-34", "35-39", "40-44", "45-49",
    "50-54", "55-59", "60-64", "65-69", "70-74", "75-79",
    "80 or older"
]
dp["AgeCategory"] = pd.Categorical(
    dp["AgeCategory"], categories=age_order, ordered=True
).codes

# C. One-hot encode remaining text columns
dp = pd.get_dummies(
    dp, columns=["Sex", "Race", "Diabetic", "GenHealth"],
    drop_first=True
)

print(f"Total features after encoding: {dp.shape[1] - 1}")

# Separate features (X) from the target column (y)
X = dp.drop("HeartDisease", axis=1)
y = dp["HeartDisease"]


# =============================================================
# STEP 3 — SPLIT AND BALANCE THE DATA
# =============================================================


print("\n--- STEP 3: Splitting and balancing ---")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=SEED
)

# Scale features: zero mean, unit variance
# Fit on training data only — do NOT fit on test data [remember old boy check with twelve]
scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# Apply SMOTE to balance the training set only
smote = SMOTE(random_state=SEED)
X_train_bal, y_train_bal = smote.fit_resample(X_train_sc, y_train)

print(f"Training (before SMOTE): "
      f"{(y_train == 1).sum():,} positive  /  {(y_train == 0).sum():,} negative")
print(f"Training (after SMOTE) : "
      f"{(y_train_bal == 1).sum():,} positive  /  {(y_train_bal == 0).sum():,} negative")
print(f"Test set (unchanged)   : "
      f"{(y_test == 1).sum():,} positive  /  {(y_test == 0).sum():,} negative")


# =============================================================
# STEP 4 — DEFINE THE MODELS
# =============================================================


print("\n--- STEP 4: Defining models ---")

base_models = [
    ("Logistic Regression",
     LogisticRegression(max_iter=1000, class_weight="balanced",
                        random_state=SEED)),

    ("SVM",
     SVC(kernel="rbf", probability=True, class_weight="balanced",
         random_state=SEED)),

    ("Random Forest",
     RandomForestClassifier(n_estimators=200, max_depth=15,
                            class_weight="balanced",
                            n_jobs=-1, random_state=SEED)),

    ("KNN",
     KNeighborsClassifier(n_neighbors=7, n_jobs=-1)),
]

stacking = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(max_iter=1000,
                                       random_state=SEED),
    cv=StratifiedKFold(n_splits=5, shuffle=True,
                       random_state=SEED),
    passthrough=True,  # meta-model also sees the raw features
    n_jobs=-1,
)

voting = VotingClassifier(
    estimators=base_models,
    voting="soft",     # average the probabilities, not hard votes
    n_jobs=-1,
)

all_models = base_models + [
    ("Stacking (Hybrid)", stacking),
    ("Voting (Hybrid)",   voting),
]

print(f"Models ready: {[name for name, _ in all_models]}")


# =============================================================
# STEP 5 — TRAIN AND EVALUATE EVERY MODEL
# =============================================================

print("\n--- STEP 5: Training and evaluating ---")

results = {}

for name, model in all_models:
    print(f"  Training {name} ...", end=" ", flush=True)

    model.fit(X_train_bal, y_train_bal)
    y_pred = model.predict(X_test_sc)
    y_prob = model.predict_proba(X_test_sc)[:, 1]

    results[name] = {
        "Accuracy" : round(accuracy_score(y_test, y_pred),  4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall"   : round(recall_score(y_test, y_pred),    4),
        "F1"       : round(f1_score(y_test, y_pred),        4),
        "AUC-ROC"  : round(roc_auc_score(y_test, y_prob),   4),
        "_pred"    : y_pred,
        "_prob"    : y_prob,
    }

    print(f"Recall={results[name]['Recall']}  "
          f"AUC={results[name]['AUC-ROC']}  "
          f"F1={results[name]['F1']}")

# Build summary table (drop internal arrays)
metric_cols = ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]
summary = pd.DataFrame(
    {name: {m: v for m, v in r.items() if m in metric_cols}
     for name, r in results.items()}
).T

# Rank models by the average of the three most important metrics
summary["Avg Score"] = summary[["Recall", "F1", "AUC-ROC"]] \
                       .mean(axis=1).round(4)
summary = summary.sort_values("Avg Score", ascending=False)

print("\n" + "=" * 62)
print("RESULTS  —  ranked by Recall + F1 + AUC-ROC average")
print("=" * 62)
print(summary.drop(columns="Avg Score").to_string())
print("\nNote: Low F1 (~0.35-0.56) is expected with 9% class imbalance.")

best_model = summary.index[0]
print(f"\nBest model: {best_model}")
print(f"\nFull report for {best_model}:")
print(classification_report(
    y_test, results[best_model]["_pred"],
    target_names=["No Heart Disease", "Heart Disease"]
))


# =============================================================
# STEP 6 — PLOTS
# =============================================================

print("\n--- STEP 6: Generating plots ---")

COLORS = ["#3498db", "#2ecc71", "#e67e22",
          "#9b59b6", "#e74c3c", "#1abc9c"]

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle(
    f"Heart Disease Prediction — Results Dashboard\n"
    f"({len(df):,} records  ·  {X.shape[1]} features  ·  "
    f"{y.mean():.1%} positive rate)",
    fontsize=15, fontweight="bold"
)

# Plot 1 — metric bar chart
summary.drop(columns="Avg Score").astype(float).plot(
    kind="bar", ax=axes[0, 0], edgecolor="white",
    width=0.8, colormap="RdYlGn"
)
axes[0, 0].set_title("All Models — All Metrics", fontweight="bold")
axes[0, 0].set_ylim(0, 1.12)
axes[0, 0].set_ylabel("Score")
axes[0, 0].tick_params(axis="x", rotation=25)
axes[0, 0].axhline(0.8, color="gray", linestyle="--",
                    lw=0.8, alpha=0.6)
axes[0, 0].legend(loc="upper right", fontsize=8)
axes[0, 0].annotate(
    "Low F1 expected\n(9% class imbalance)",
    xy=(0.01, 0.38), xycoords="axes fraction",
    fontsize=8, color="#c0392b", style="italic"
)

# Plot 2 — ROC curves
for (name, _), color in zip(all_models, COLORS):
    fpr, tpr, _ = roc_curve(y_test, results[name]["_prob"])
    axes[0, 1].plot(
        fpr, tpr, lw=2, color=color,
        label=f"{name}  ({results[name]['AUC-ROC']})"
    )
axes[0, 1].plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
axes[0, 1].set_title("ROC Curves", fontweight="bold")
axes[0, 1].set_xlabel("False Positive Rate")
axes[0, 1].set_ylabel("True Positive Rate")
axes[0, 1].legend(fontsize=7.5, loc="lower right")

# Plot 3 — Feature importance (Random Forest)
rf_model = dict(base_models)["Random Forest"]
importance_series = pd.Series(
    rf_model.feature_importances_, index=X.columns
)
importance_series.nlargest(14).sort_values().plot(
    kind="barh", ax=axes[0, 2],
    color="#e74c3c", edgecolor="white", alpha=0.85
)
axes[0, 2].set_title("Top 14 Features (Random Forest)",
                      fontweight="bold")
axes[0, 2].set_xlabel("Importance Score")

# Plot 4 — Confusion matrix (Stacking)
sns.heatmap(
    confusion_matrix(y_test, results["Stacking (Hybrid)"]["_pred"]),
    annot=True, fmt=",d", cmap="Blues", ax=axes[1, 0],
    xticklabels=["No Disease", "Heart Disease"],
    yticklabels=["No Disease", "Heart Disease"],
    linewidths=1, linecolor="white"
)
axes[1, 0].set_title("Confusion Matrix — Stacking",
                      fontweight="bold")
axes[1, 0].set_xlabel("Predicted")
axes[1, 0].set_ylabel("Actual")

# Plot 5 — Confusion matrix (Voting)
sns.heatmap(
    confusion_matrix(y_test, results["Voting (Hybrid)"]["_pred"]),
    annot=True, fmt=",d", cmap="Blues", ax=axes[1, 1],
    xticklabels=["No Disease", "Heart Disease"],
    yticklabels=["No Disease", "Heart Disease"],
    linewidths=1, linecolor="white"
)
axes[1, 1].set_title("Confusion Matrix — Voting",
                      fontweight="bold")
axes[1, 1].set_xlabel("Predicted")
axes[1, 1].set_ylabel("Actual")

# Plot 6 — Overall ranking bar
axes[1, 2].barh(
    summary.index,
    summary["Avg Score"].astype(float),
    color=COLORS[:len(summary)], edgecolor="white"
)
axes[1, 2].set_title(
    "Overall Ranking\n(avg of Recall + F1 + AUC-ROC)",
    fontweight="bold"
)
axes[1, 2].set_xlim(0, 1)
for i, (idx, row) in enumerate(summary.iterrows()):
    axes[1, 2].text(
        float(row["Avg Score"]) + 0.005, i,
        str(row["Avg Score"]), va="center", fontsize=9
    )

plt.tight_layout()
plt.savefig("results_dashboard.png", dpi=150, bbox_inches="tight")
plt.show()
print("  Saved → results_dashboard.png")


# =============================================================
# STEP 7 — CROSS-VALIDATION
# =============================================================


print("\n--- STEP 7: Cross-validation (Stacking model) ---")

cv_results = {"F1": [], "AUC-ROC": [], "Recall": []}

for fold, (tr_idx, val_idx) in enumerate(
    StratifiedKFold(n_splits=5, shuffle=True,
                    random_state=SEED).split(X, y), 1
):
    # Scale inside the fold
    fold_scaler = StandardScaler()
    X_fold_tr   = fold_scaler.fit_transform(X.iloc[tr_idx])
    X_fold_val  = fold_scaler.transform(X.iloc[val_idx])
    y_fold_tr   = y.iloc[tr_idx].values
    y_fold_val  = y.iloc[val_idx].values

    # Balance inside the fold — not before
    X_bal, y_bal = SMOTE(random_state=SEED).fit_resample(
        X_fold_tr, y_fold_tr
    )

    # Train a fresh stacking model for this fold
    fold_stack = StackingClassifier(
        estimators=base_models,
        final_estimator=LogisticRegression(max_iter=1000,
                                           random_state=SEED),
        cv=3, passthrough=True, n_jobs=-1
    )
    fold_stack.fit(X_bal, y_bal)

    y_fold_pred = fold_stack.predict(X_fold_val)
    y_fold_prob = fold_stack.predict_proba(X_fold_val)[:, 1]

    fold_f1  = round(f1_score(y_fold_val, y_fold_pred), 4)
    fold_auc = round(roc_auc_score(y_fold_val, y_fold_prob), 4)
    fold_rec = round(recall_score(y_fold_val, y_fold_pred), 4)

    cv_results["F1"].append(fold_f1)
    cv_results["AUC-ROC"].append(fold_auc)
    cv_results["Recall"].append(fold_rec)

    print(f"  Fold {fold}:  F1={fold_f1}  "
          f"AUC={fold_auc}  Recall={fold_rec}")

print(f"\n  {'Metric':<12} {'Mean':>8} {'Std Dev':>10}")
print(f"  {'-' * 32}")
for metric, values in cv_results.items():
    print(f"  {metric:<12} "
          f"{np.mean(values):>8.4f}  "
          f"{np.std(values):>9.4f}")


# =============================================================
# FINAL SUMMARY
# =============================================================

print(f"""
{'=' * 62}
  DONE
{'=' * 62}
  Records used      : {len(df):,}
    from 2020 file  : {len(df_2020):,}
    from 2022 file  : {len(df_2022):,}
  Features          : {X.shape[1]}
  Positive rate     : {y.mean():.2%}

  Best model        : {best_model}
  Recall            : {summary.loc[best_model, 'Recall']}
  AUC-ROC           : {summary.loc[best_model, 'AUC-ROC']}
  F1-Score          : {summary.loc[best_model, 'F1']}  ← low due to 9% imbalance

  5-fold CV (Stacking):
    AUC-ROC         : {np.mean(cv_results['AUC-ROC']):.4f} ± {np.std(cv_results['AUC-ROC']):.4f}
    Recall          : {np.mean(cv_results['Recall']):.4f} ± {np.std(cv_results['Recall']):.4f}

  Key rules followed:
    SMOTE after split     → no synthetic data in test set
    Scaler fit on train   → no test-set leakage
    SMOTE inside CV folds → honest cross-validation score
    Recall over Accuracy  → medical screening priority
{'=' * 62}
""")
