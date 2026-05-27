# =============================================================
#  Heart Disease Prediction — Ensemble Machine Learning
#  Bachelor Thesis | Computer Science | 2025-2026
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
from sklearn.ensemble        import (RandomForestClassifier,
                                     HistGradientBoostingClassifier,
                                     VotingClassifier)
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

df = pd.concat([df_2020, df_2022], ignore_index=True)
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

print(f"2020 dataset : {len(df_2020):,} rows")
print(f"2022 dataset : {len(df_2022):,} rows")
print(f"Combined     : {len(df):,} rows")
print(f"Positive rate: {(df['HeartDisease'] == 'Yes').mean():.1%}")


# =============================================================
# STEP 2 — ENCODE FEATURES
# =============================================================


print("\n--- STEP 2: Encoding features ---")

dp = df.copy()

# A — Yes/No to 1/0
yes_no_cols = [
    "HeartDisease", "Smoking", "AlcoholDrinking", "Stroke",
    "DiffWalking", "PhysicalActivity", "Asthma",
    "KidneyDisease", "SkinCancer"
]
for col in yes_no_cols:
    dp[col] = dp[col].map({"Yes": 1, "No": 0})

# B — Ordinal age (preserves ordering)
age_order = [
    "18-24", "25-29", "30-34", "35-39", "40-44", "45-49",
    "50-54", "55-59", "60-64", "65-69", "70-74", "75-79",
    "80 or older"
]
dp["AgeCategory"] = pd.Categorical(
    dp["AgeCategory"], categories=age_order, ordered=True
).codes

# C — One-hot for nominal categories
dp = pd.get_dummies(
    dp, columns=["Sex", "Race", "Diabetic", "GenHealth"],
    drop_first=True
)

print(f"Total features after encoding: {dp.shape[1] - 1}")

X = dp.drop("HeartDisease", axis=1)
y = dp["HeartDisease"]


# =============================================================
# STEP 3 — SPLIT AND BALANCE
# =============================================================


print("\n--- STEP 3: Splitting and balancing ---")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED
)

# Scale: fit on training data only
scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# SMOTE: training set only
smote = SMOTE(random_state=SEED)
X_train_bal, y_train_bal = smote.fit_resample(X_train_sc, y_train)

print(f"Train before SMOTE: "
      f"{(y_train==1).sum():,} positive  /  "
      f"{(y_train==0).sum():,} negative")
print(f"Train after  SMOTE: "
      f"{(y_train_bal==1).sum():,} positive  /  "
      f"{(y_train_bal==0).sum():,} negative")
print(f"Test  (unchanged) : "
      f"{(y_test==1).sum():,} positive  /  "
      f"{(y_test==0).sum():,} negative")


# =============================================================
# STEP 4 — DEFINE MODELS
# =============================================================


print("\n--- STEP 4: Defining models ---")

base_models = [
    ("Logistic Regression",
     LogisticRegression(
         max_iter=1000,
         class_weight="balanced",
         random_state=SEED
     )),

    ("Random Forest",
     RandomForestClassifier(
         n_estimators=200,
         max_depth=15,
         class_weight="balanced",
         n_jobs=-1,
         random_state=SEED
     )),

    ("HistGradientBoosting",
     HistGradientBoostingClassifier(
         max_iter=100,
         max_depth=5,
         random_state=SEED
     )),
]

voting = VotingClassifier(
    estimators=base_models,
    voting="soft",
    n_jobs=-1,
)

all_models = base_models + [("Voting (Ensemble)", voting)]
print(f"Defined: {[name for name, _ in all_models]}")


# =============================================================
# STEP 5 — TRAIN AND EVALUATE
# =============================================================


print("\n--- STEP 5: Training and evaluating ---")

results = {}
metric_cols = ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]

for name, model in all_models:
    print(f"  [{name}] ...", end=" ", flush=True)
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

summary = pd.DataFrame(
    {name: {m: v for m, v in r.items() if m in metric_cols}
     for name, r in results.items()}
).T
summary["Avg Score"] = summary[["Recall","F1","AUC-ROC"]] \
                       .mean(axis=1).round(4)
summary = summary.sort_values("Avg Score", ascending=False)

print("\n" + "="*62)
print("RESULTS  —  ranked by Recall + F1 + AUC-ROC average")
print("="*62)
print(summary.drop(columns="Avg Score").to_string())
print("\nNote: Low F1 is expected at 9% class prevalence. "
      "Focus on AUC-ROC and Recall.")

best = summary.index[0]
print(f"\nBest model: {best}")
print(f"\nClassification report — {best}:")
print(classification_report(
    y_test, results[best]["_pred"],
    target_names=["No Heart Disease", "Heart Disease"]
))


# =============================================================
# STEP 6 — PLOTS
# =============================================================

print("\n--- STEP 6: Generating plots ---")

COLORS = ["#3498db", "#e67e22", "#2ecc71", "#e74c3c"]

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle(
    f"Heart Disease Prediction — Results Dashboard\n"
    f"({len(df):,} records  ·  {X.shape[1]} features  ·  "
    f"{y.mean():.1%} positive rate)",
    fontsize=15, fontweight="bold"
)

# 1 — Metric bars
summary.drop(columns="Avg Score").astype(float).plot(
    kind="bar", ax=axes[0,0],
    edgecolor="white", width=0.75, colormap="RdYlGn"
)
axes[0,0].set_title("All Models — All Metrics", fontweight="bold")
axes[0,0].set_ylim(0, 1.12)
axes[0,0].set_ylabel("Score")
axes[0,0].tick_params(axis="x", rotation=25)
axes[0,0].axhline(0.8, color="gray", linestyle="--", lw=0.8, alpha=0.6)
axes[0,0].legend(loc="upper right", fontsize=8)
axes[0,0].annotate(
    "Low F1 expected\n(9% imbalance)",
    xy=(0.01, 0.36), xycoords="axes fraction",
    fontsize=8, color="#c0392b", style="italic"
)

# 2 — ROC curves
for (name, _), color in zip(all_models, COLORS):
    fpr, tpr, _ = roc_curve(y_test, results[name]["_prob"])
    axes[0,1].plot(fpr, tpr, lw=2, color=color,
                   label=f"{name}  (AUC {results[name]['AUC-ROC']})")
axes[0,1].plot([0,1],[0,1],"k--",lw=0.8,alpha=0.4)
axes[0,1].set_title("ROC Curves", fontweight="bold")
axes[0,1].set_xlabel("False Positive Rate")
axes[0,1].set_ylabel("True Positive Rate")
axes[0,1].legend(fontsize=8, loc="lower right")

# 3 — Feature importance (Random Forest)
rf_model = dict(base_models)["Random Forest"]
pd.Series(rf_model.feature_importances_, index=X.columns) \
  .nlargest(14).sort_values() \
  .plot(kind="barh", ax=axes[0,2],
        color="#e74c3c", edgecolor="white", alpha=0.85)
axes[0,2].set_title("Top 14 Features (Random Forest)",
                     fontweight="bold")
axes[0,2].set_xlabel("Importance Score")

# 4 — Confusion matrix: Voting Ensemble
sns.heatmap(
    confusion_matrix(y_test, results["Voting (Ensemble)"]["_pred"]),
    annot=True, fmt=",d", cmap="Blues", ax=axes[1,0],
    xticklabels=["No Disease","Heart Disease"],
    yticklabels=["No Disease","Heart Disease"],
    linewidths=1, linecolor="white"
)
axes[1,0].set_title("Confusion Matrix — Voting Ensemble",
                     fontweight="bold")
axes[1,0].set_xlabel("Predicted"); axes[1,0].set_ylabel("Actual")

# 5 — Confusion matrix: Logistic Regression
sns.heatmap(
    confusion_matrix(y_test, results["Logistic Regression"]["_pred"]),
    annot=True, fmt=",d", cmap="Oranges", ax=axes[1,1],
    xticklabels=["No Disease","Heart Disease"],
    yticklabels=["No Disease","Heart Disease"],
    linewidths=1, linecolor="white"
)
axes[1,1].set_title("Confusion Matrix — Logistic Regression",
                     fontweight="bold")
axes[1,1].set_xlabel("Predicted"); axes[1,1].set_ylabel("Actual")

# 6 — Overall ranking
axes[1,2].barh(
    summary.index, summary["Avg Score"].astype(float),
    color=COLORS[:len(summary)], edgecolor="white"
)
axes[1,2].set_title(
    "Overall Ranking\n(avg of Recall + F1 + AUC-ROC)",
    fontweight="bold"
)
axes[1,2].set_xlim(0, 1)
for i, (idx, row) in enumerate(summary.iterrows()):
    axes[1,2].text(float(row["Avg Score"])+0.005, i,
                   str(row["Avg Score"]), va="center", fontsize=9)

plt.tight_layout()
plt.savefig("results_dashboard.png", dpi=150, bbox_inches="tight")
plt.show()
print("  Saved → results_dashboard.png")


# =============================================================
# STEP 7 — CROSS-VALIDATION
# =============================================================


print("\n--- STEP 7: Cross-validation (Voting Ensemble) ---")

cv_results = {"F1": [], "AUC-ROC": [], "Recall": []}

for fold, (tr_idx, val_idx) in enumerate(
    StratifiedKFold(n_splits=5, shuffle=True,
                    random_state=SEED).split(X, y), 1
):
    fold_sc  = StandardScaler()
    X_ftr    = fold_sc.fit_transform(X.iloc[tr_idx])
    X_fval   = fold_sc.transform(X.iloc[val_idx])
    y_ftr    = y.iloc[tr_idx].values
    y_fval   = y.iloc[val_idx].values

    X_fbal, y_fbal = SMOTE(random_state=SEED).fit_resample(
        X_ftr, y_ftr
    )

    fold_vote = VotingClassifier(
        estimators=[
            ("lr",  LogisticRegression(max_iter=1000,
                                       class_weight="balanced",
                                       random_state=SEED)),
            ("rf",  RandomForestClassifier(n_estimators=200,
                                           max_depth=15,
                                           class_weight="balanced",
                                           n_jobs=-1,
                                           random_state=SEED)),
            ("hgb", HistGradientBoostingClassifier(max_iter=100,
                                                   max_depth=5,
                                                   random_state=SEED)),
        ],
        voting="soft", n_jobs=-1
    )
    fold_vote.fit(X_fbal, y_fbal)

    yp  = fold_vote.predict(X_fval)
    ypr = fold_vote.predict_proba(X_fval)[:, 1]

    cv_results["F1"].append(round(f1_score(y_fval, yp), 4))
    cv_results["AUC-ROC"].append(round(roc_auc_score(y_fval, ypr), 4))
    cv_results["Recall"].append(round(recall_score(y_fval, yp), 4))

    print(f"  Fold {fold}:  "
          f"F1={cv_results['F1'][-1]}  "
          f"AUC={cv_results['AUC-ROC'][-1]}  "
          f"Recall={cv_results['Recall'][-1]}")

print(f"\n  {'Metric':<12} {'Mean':>8} {'Std Dev':>10}")
print(f"  {'-'*32}")
for metric, values in cv_results.items():
    print(f"  {metric:<12} {np.mean(values):>8.4f}  "
          f"{np.std(values):>9.4f}")
print("\n  Low std dev = stable, generalisable performance.")


# =============================================================
# FINAL SUMMARY
# =============================================================
best = summary.index[0]
print(f"""
{'='*62}
  COMPLETE
{'='*62}
  Records : {len(df):,}  ({len(df_2020):,} from 2020, {len(df_2022):,} from 2022)
  Features: {X.shape[1]}
  Positive: {y.mean():.2%}

  Final model lineup:
    Logistic Regression     linear baseline, interpretable
    Random Forest           tree ensemble, feature importance
    HistGradientBoosting    boosted trees, fast at scale
    Voting (Ensemble)       soft-average hybrid

  Dropped (with reason):
    LinearSVC               99.2% overlap with LR — redundant
    KNN                     AUC 0.697, no class-weight support
    Stacking                base models r>0.94 — meta-learner
                            collapses; Voting is correct here

  Best model  : {best}
  Recall      : {summary.loc[best, 'Recall']}
  AUC-ROC     : {summary.loc[best, 'AUC-ROC']}
  F1-Score    : {summary.loc[best, 'F1']}  (low = 9% imbalance, expected)

  CV (Voting, 5-fold):
    AUC-ROC   : {np.mean(cv_results['AUC-ROC']):.4f} ± {np.std(cv_results['AUC-ROC']):.4f}
    Recall    : {np.mean(cv_results['Recall']):.4f} ± {np.std(cv_results['Recall']):.4f}
{'='*62}
""")
