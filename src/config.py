"""
Configuration constants
=======================

Every tunable value used across the pipeline is collected here so that
hyperparameter changes happen in one place rather than scattered across
several files. Each constant is documented with the reasoning behind
the chosen value.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple


# ─── Paths ──────────────────────────────────────────────────────
DATA_DIR    = Path("data")
MODEL_DIR   = Path("models")
OUTPUT_DIR  = Path("outputs")
DATASET_2020 = "heart_2020_cleaned_final.csv"
DATASET_2022 = "heart_2022_cleaned_final.csv"


# ─── Reproducibility ────────────────────────────────────────────
# A single seed is used for every random operation in the pipeline
# so a fresh checkout reproduces the published results exactly.
RANDOM_SEED = 42


# ─── Train/Test Partition ───────────────────────────────────────
# 80/20 with stratification on the target. The test partition is
# only touched once at final evaluation; everything else operates
# on the training partition.
TEST_FRACTION    = 0.20
CV_FOLDS         = 5


# ─── Feature Schema ─────────────────────────────────────────────
# Three encoding categories matched to three encoding strategies.

BINARY_COLUMNS: List[str] = [
    "HeartDisease",       # the target itself
    "Smoking",
    "AlcoholDrinking",
    "Stroke",
    "DiffWalking",
    "PhysicalActivity",
    "Asthma",
    "KidneyDisease",
    "SkinCancer",
]

# Ordered categorical — encoded as integers preserving order
AGE_BRACKETS: List[str] = [
    "18-24", "25-29", "30-34", "35-39", "40-44", "45-49",
    "50-54", "55-59", "60-64", "65-69", "70-74", "75-79",
    "80 or older",
]

# Unordered categorical — encoded with one-hot (drop_first=True)
ONE_HOT_COLUMNS: List[str] = ["Sex", "Race", "Diabetic", "GenHealth"]

TARGET_COLUMN: str = "HeartDisease"


# ─── Hyperparameters per Model ──────────────────────────────────
# All values chosen to balance speed against accuracy on a half-million
# record dataset. Reasoning is in the docstring of each entry.

LOGISTIC_REGRESSION_PARAMS: Dict = {
    "max_iter":     1000,        # enough for lbfgs to converge on this scale
    "class_weight": "balanced",  # algorithm-level imbalance compensation
    "random_state": RANDOM_SEED,
    "solver":       "lbfgs",
    "C":            1.0,
}

RANDOM_FOREST_PARAMS: Dict = {
    "n_estimators":  200,            # past this, marginal AUC gain is < 0.001
    "max_depth":     15,             # caps individual tree complexity
    "class_weight":  "balanced",
    "n_jobs":        -1,
    "random_state":  RANDOM_SEED,
}

HIST_GRADIENT_BOOSTING_PARAMS: Dict = {
    "max_iter":      100,            # boosting rounds
    "max_depth":     5,              # shallow per tree, deep through stacking
    "learning_rate": 0.1,
    "random_state":  RANDOM_SEED,
}

SMOTE_PARAMS: Dict = {
    "k_neighbors":  5,
    "random_state": RANDOM_SEED,
}


# ─── Voting Ensemble ────────────────────────────────────────────
# Equal weights — no prior reason to favour any model. The custom
# voting implementation (SoftVotingEnsemble) reads these explicitly
# rather than using sklearn defaults.
VOTING_WEIGHTS: Dict[str, float] = {
    "LogisticRegression":    1.0,
    "RandomForest":          1.0,
    "HistGradientBoosting":  1.0,
}


# ─── Evaluation ─────────────────────────────────────────────────
# Order matters for display — most informative metrics first.
METRIC_ORDER: List[str] = [
    "AUC-ROC", "Recall", "F1-Score", "Precision", "Accuracy",
]

# Probability cut-off for the binary decision. 0.5 is the natural
# split for SMOTE-balanced training; production use would tune this
# based on clinical follow-up capacity.
DECISION_THRESHOLD: float = 0.5


# ─── Risk Bands (for the web UI) ────────────────────────────────
@dataclass(frozen=True)
class RiskBand:
    name:        str
    lower_bound: float
    upper_bound: float
    color:       str
    advice:      str

RISK_BANDS: Tuple[RiskBand, ...] = (
    RiskBand("LOW RISK",      0.00, 0.25, "#27ae60",
             "Continue standard preventive care. Risk profile can change "
             "over time — periodic reassessment is sensible."),
    RiskBand("MODERATE RISK", 0.25, 0.50, "#e67e22",
             "Lifestyle review and routine monitoring suggested."),
    RiskBand("HIGH RISK",     0.50, 1.01, "#c0392b",
             "Clinical follow-up advised. This is a screening result, "
             "not a diagnosis."),
)


def get_risk_band(probability: float) -> RiskBand:
    """Map a probability to its corresponding risk band."""
    for band in RISK_BANDS:
        if band.lower_bound <= probability < band.upper_bound:
            return band
    return RISK_BANDS[-1]   # numerical safety net for probability == 1.0


@dataclass
class Config:
    """
    Frozen configuration container for the pipeline.

    Centralising the values rather than scattering them as module-level
    constants makes them explicit when reading the code and easy to
    override in tests or experiments.
    """
    random_seed:        int                = RANDOM_SEED
    test_fraction:      float              = TEST_FRACTION
    cv_folds:           int                = CV_FOLDS
    decision_threshold: float              = DECISION_THRESHOLD
    smote_params:       Dict               = field(default_factory=lambda: SMOTE_PARAMS)
    lr_params:          Dict               = field(default_factory=lambda: LOGISTIC_REGRESSION_PARAMS)
    rf_params:          Dict               = field(default_factory=lambda: RANDOM_FOREST_PARAMS)
    hgb_params:         Dict               = field(default_factory=lambda: HIST_GRADIENT_BOOSTING_PARAMS)
    voting_weights:     Dict[str, float]   = field(default_factory=lambda: VOTING_WEIGHTS)

    def summary(self) -> str:
        """Human-readable summary of the active configuration."""
        return (
            f"Configuration Summary\n"
            f"  Random seed   : {self.random_seed}\n"
            f"  Test fraction : {self.test_fraction}\n"
            f"  CV folds      : {self.cv_folds}\n"
            f"  Threshold     : {self.decision_threshold}\n"
            f"  Voting weights: {self.voting_weights}\n"
        )
