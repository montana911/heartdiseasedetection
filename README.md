# Heart disease prediction using hybrid machine learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.30+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

thesis implementation of a soft-voting ensemble for cardiovascular
disease screening, trained on 538,510 records from the U.S. CDC Behavioral
Risk Factor Surveillance System (BRFSS) 2020 and 2022 cycles.

---

## Project Overview

This project investigates whether routine health survey data — the kind
collected at population scale by public health agencies — carries enough
predictive signal to identify high cardiovascular risk without any clinical
testing. The model achieves AUC-ROC = 0.835 on a held-out test set of
107,702 records, demonstrating that lifestyle survey data alone is sufficient
for first-pass screening.

The project includes a training pipeline, an interactive web application
for live risk prediction, a command-line inference tool, and full
reproducibility documentation.

---

## Repository Structure

```
heartdiseasedetection/
├── src/                            Source package (single-responsibility modules)
│   ├── __init__.py                 Package metadata and public API
│   ├── config.py                   Centralised constants and hyperparameters
│   ├── data_loader.py              DatasetLoader — load and merge BRFSS cycles
│   ├── preprocessor.py             FeaturePreprocessor — encode, split, scale
│   ├── imbalance_handler.py        ImbalanceHandler — SMOTE with leakage protection
│   ├── models.py                   ModelFactory + custom SoftVotingEnsemble
│   ├── evaluator.py                ModelEvaluator — manual metric implementations
│   ├── visualizer.py               ResultsVisualizer — all thesis plots
│   └── pipeline.py                 HeartDiseasePipeline — end-to-end orchestrator
├── data/                           BRFSS cleaned CSV files (not in repo — see Setup)
│   ├── heart_2020_cleaned_final.csv
│   └── heart_2022_cleaned_final.csv
├── models/                         Trained pipeline pickle files (generated)
│   └── pipeline.pkl                (created by train.py)
├── outputs/                        Generated plots and reports
├── train.py                        Entry point — trains and saves the pipeline
├── predict.py                      Entry point — CLI inference for a single patient
├── app.py                          Entry point — Streamlit web application
├── requirements.txt                Python dependency list
└── README.md                       This file
```

---

## System Architecture

The project follows a **single-responsibility module pattern**. Each module
in `src/` has one clear job; the `HeartDiseasePipeline` orchestrator
composes them into the end-to-end workflow.

```
                ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
Entry points:   │  train.py   │  │   app.py    │  │ predict.py  │
                └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
                       └────────────────┼────────────────┘
                                        ▼
                       ┌──────────────────────────────────┐
Orchestrator:          │     HeartDiseasePipeline         │
                       │  (composes the modules below)    │
                       └─────────────────┬────────────────┘
                                         │
       ┌───────────┬───────────┬─────────┴────┬───────────┬────────────┐
       ▼           ▼           ▼              ▼           ▼            ▼
   Dataset      Feature      Imbalance     SoftVoting    Model     Results
   Loader       Preprocessor Handler       Ensemble      Evaluator Visualizer
       │           │           │              │           │            │
       ▼           ▼           ▼              ▼           ▼            ▼
   ── pandas ── numpy ── scikit-learn ── imbalanced-learn ── matplotlib ──
```

### Why This Structure?

1. **Single responsibility per module** — `DatasetLoader` only loads,
   `ImbalanceHandler` only balances. This makes each piece independently
   testable and replaceable.

2. **Leakage prevention is explicit in the API** — the `ImbalanceHandler`
   method is named `balance_training_set()` to make clear that calling it
   on anything other than training data is a bug, not an option.

3. **The ensemble is a custom class, not just `VotingClassifier(voting="soft")`** —
   the `SoftVotingEnsemble` in `src/models.py` implements the weighted
   averaging algorithm explicitly:

   ```
   P_ensemble(y=1 | x) = Σₖ (wₖ · Pₖ(y=1 | x)) / Σₖ wₖ
   ```

   It also exposes per-base-model probabilities through
   `predict_proba_per_model()`, which sklearn's built-in voting classifier
   does not — used for the diversity-correlation analysis that justified
   dropping stacking.

4. **Metrics are computed manually** — `ModelEvaluator` implements
   Accuracy / Precision / Recall / F1 from the confusion matrix directly,
   so the formulas are visible in code rather than hidden behind library
   calls.

5. **Entry points are thin** — `train.py`, `predict.py`, and `app.py` all
   work through `HeartDiseasePipeline`. None of them duplicate the ML
   logic. This answers the common question of "which file is the real
   work?": it's the `src/` package, and every entry point uses it.

---

## Models

The hybrid ensemble combines three base classifiers chosen for
complementary inductive biases.

| Model | Role | Hyperparameters |
|---|---|---|
| **Logistic Regression** | Linear baseline with well-calibrated probabilities | `max_iter=1000`, `class_weight=balanced` |
| **Random Forest** | Captures non-linear interactions; provides feature importance | `n_estimators=200`, `max_depth=15`, `class_weight=balanced` |
| **HistGradientBoosting** | Sequential error correction; fast on large datasets | `max_iter=100`, `max_depth=5` |
| **Voting Ensemble** | Weighted average of the three base probabilities (custom implementation) | `voting=soft`, equal weights |

### Why a Voting Ensemble and Not Stacking?

Stacking was tested during development and rejected. The three base
models' probability outputs correlate at r > 0.94 on this dataset,
meaning they agree on nearly every prediction. A meta-learner trained
on inputs that are 94% correlated has no useful signal to combine and
collapses to noise. Soft voting is the correct ensemble choice when
base model outputs are this strongly correlated; the
`SoftVotingEnsemble.diversity_correlation()` method computes this
correlation directly from the fitted models.

LinearSVC and KNN were also tested and dropped: LinearSVC agreed with
Logistic Regression on 99.2% of predictions (zero ensemble diversity),
and KNN scored 13 AUC-ROC points below the weakest retained model.

---

## Results

Tested on a held-out 20% partition (107,702 records, never seen during
training):

| Model | Accuracy | Precision | Recall | F1 | **AUC-ROC** |
|---|---|---|---|---|---|
| Logistic Regression | 0.742 | 0.228 | 0.777 | 0.353 | **0.835** |
| Random Forest | 0.812 | 0.267 | 0.618 | 0.373 | 0.828 |
| HistGradientBoosting | 0.880 | 0.344 | 0.355 | 0.349 | 0.825 |
| **Voting Ensemble** | **0.819** | **0.278** | **0.625** | **0.385** | **0.834** |

Five-fold cross-validation confirms stability (std dev < 0.005 on every
metric).

**Note on F1**: F1 around 0.35 is mathematically expected at 9% class
prevalence and is not a sign of poor model performance. AUC-ROC and
Recall are the meaningful metrics for this screening task.

---

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/montana911/heartdiseasedetection.git
cd heartdiseasedetection
```

### 2. Install Python Dependencies

Python 3.10 or later is required.

```bash
pip install -r requirements.txt
```

This installs scikit-learn, imbalanced-learn, pandas, numpy, matplotlib,
seaborn, and Streamlit (~3 minutes).

### 3. Download the Datasets

Both CSV files come from the
[Kaggle BRFSS dataset](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease)
by Kamil Pytlak. Place the cleaned versions into the `data/` folder:

```
data/
├── heart_2020_cleaned_final.csv
└── heart_2022_cleaned_final.csv
```

The cleaned files are produced by the cleaning step (run separately,
outside the modelling code). If you need the cleaning script, see the
project notebooks folder.

---

## Usage

### Training

Train the full pipeline and save it to disk:

```bash
python train.py
```

The script will:

1. Load and merge the two BRFSS CSV files
2. Encode features (binary, ordinal, one-hot)
3. Stratified 80/20 split — training partition is **the only thing** subsequent steps see
4. Fit a `StandardScaler` on the training partition only
5. Apply SMOTE to the scaled training partition only
6. Train the three base classifiers and the Voting Ensemble
7. Evaluate every model on the held-out test partition
8. Save the entire trained pipeline to `models/pipeline.pkl`

First run takes 2–3 minutes on a typical laptop.

### Web Application (interactive risk prediction)

```bash
streamlit run app.py
```

Your browser opens at `http://localhost:8501`. The app has three tabs:

- **🔮 Live Prediction** — enter a patient profile, get an instant risk score
- **📊 Model Performance** — see metrics, ROC curves, confusion matrix, feature importance
- **📖 About** — project context

The app loads the model from `models/pipeline.pkl`. If no trained model
exists, it will prompt you to run `python train.py` first.

### Command-Line Inference

Predict for a single patient from a JSON file:

```bash
python predict.py path/to/patient.json
```

Or from stdin:

```bash
cat patient.json | python predict.py
```

Or use the built-in example profile:

```bash
python predict.py
```

A patient JSON looks like:

```json
{
  "age":               "60-64",
  "sex":               "Male",
  "race":              "White",
  "bmi":               28.5,
  "sleep_time":        6,
  "smoking":           "Yes",
  "stroke":            "No",
  "diff_walking":      "Yes",
  "physical_activity": "No",
  "gen_health":        "Fair",
  "diabetic":          "Yes",
  ...
}
```

---

## Design Decisions

A few of the more interesting decisions, documented for the thesis defence.

### Why Cleaning Lives Outside the Modelling Code

Cleaning the raw BRFSS CSVs (deduplication, BMI outlier removal,
schema harmonisation between 2020 and 2022) is a one-time operation.
Including it inside the modelling pipeline means it runs every time
the model is trained, slowing development. Separating it allows the
modelling code to assume valid input and stay focused on its actual
job. The cleaned files are produced once and committed alongside the code.

### Why Cycles Are Shuffled at Load Time

If the 2020 records always preceded the 2022 records in the merged
DataFrame, stratified cross-validation could end up with one cycle in
training and the other in validation by accident. Shuffling at load
time guarantees the two cycles interleave so each CV fold contains a
roughly proportional mix.

### Why SMOTE Is Inside `ImbalanceHandler` and Not Inline

Decoupling SMOTE from the rest of the pipeline allows experiments with
alternative imbalance strategies (class weights only, undersampling,
ADASYN) by swapping the handler — no other module changes. The wrapper
also makes the leakage-safe API explicit through the method name:
`balance_training_set()` is impossible to misuse by accident.

### Why the Voting Algorithm Is Implemented Manually

scikit-learn's `VotingClassifier(voting="soft")` would have done the
job in one line. Implementing it manually in `SoftVotingEnsemble` makes
the algorithm visible in the source code, allows weights to be tuned
independently of sklearn's API, exposes per-base-model probabilities
for diagnostic use (needed for the diversity correlation analysis),
and removes a layer of indirection between the design and the code.

### Why Cross-Validation Re-Applies Scaling and SMOTE Inside Each Fold

Fitting the scaler once on the full training set before cross-validating
would leak information from each validation fold into the scaler's
parameters. The same applies to SMOTE. Both must be re-fit inside each
fold using only that fold's training portion. The `HeartDiseasePipeline`
class enforces this by instantiating fresh `FeaturePreprocessor` and
`ImbalanceHandler` instances per fold rather than reusing them.

---


## Data Source

Pytlak, K. (2021). *Personal Key Indicators of Heart Disease* [Data set].
Kaggle.
[https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease)
