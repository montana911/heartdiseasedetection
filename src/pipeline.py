"""
End-to-end pipeline
===================

The HeartDiseasePipeline composes the smaller modules into a single
high-level workflow. It is the only class most callers need to know
about — train.py and app.py both work through it rather than touching
the individual building blocks.

The composition pattern matters here: each step in the workflow is
an instance of a single-responsibility class, and the pipeline simply
calls them in the correct order. This makes it possible to swap any
single step (e.g. use a different imbalance handler) without rewriting
the rest of the workflow.

Workflow stages
---------------
  1. load           DatasetLoader merges the two CSVs
  2. preprocess     FeaturePreprocessor encodes, splits, scales
  3. balance        ImbalanceHandler applies SMOTE to training only
  4. fit            SoftVotingEnsemble trains the three base models
  5. evaluate       ModelEvaluator scores everything on the test set
  6. persist        Pickle the entire trained pipeline for the web app
"""

from __future__ import annotations
from pathlib import Path
from typing  import Dict, Tuple, Optional
import pickle
import logging

import numpy  as np
import pandas as pd

from .config             import Config, MODEL_DIR
from .data_loader        import DatasetLoader
from .preprocessor       import FeaturePreprocessor
from .imbalance_handler  import ImbalanceHandler
from .models             import ModelFactory, SoftVotingEnsemble
from .evaluator          import ModelEvaluator


logger = logging.getLogger(__name__)


class HeartDiseasePipeline:
    """
    End-to-end orchestrator for the heart disease prediction system.

    Internally composes DatasetLoader, FeaturePreprocessor,
    ImbalanceHandler, SoftVotingEnsemble, and ModelEvaluator. Each
    component is exposed as an attribute after fit_full_pipeline()
    so the web app can interrogate them directly (e.g. fetching
    feature importance from the Random Forest base model).
    """

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or Config()

        # Owned components
        self.loader        = DatasetLoader(random_seed=self.config.random_seed)
        self.preprocessor  = FeaturePreprocessor(
            test_fraction=self.config.test_fraction,
            random_seed=self.config.random_seed,
        )
        self.imbalance     = ImbalanceHandler(
            k_neighbors=self.config.smote_params["k_neighbors"],
            random_seed=self.config.random_seed,
        )
        self.ensemble:  Optional[SoftVotingEnsemble] = None
        self.evaluator: ModelEvaluator = ModelEvaluator()

        # Training data preserved for diagnostics
        self.X_test:  Optional[np.ndarray] = None
        self.y_test:  Optional[np.ndarray] = None
        self.is_trained = False

    # ─── Full training run ─────────────────────────────────────

    def fit_full_pipeline(self) -> Dict[str, float]:
        """
        Execute every step from raw CSVs through to a fully trained
        and evaluated ensemble.

        Returns the comparison table of test-set metrics as a dict
        of dicts (one entry per model).
        """
        logger.info("=" * 60)
        logger.info("Stage 1/5 — Loading data")
        self.loader.load_and_merge()

        logger.info("Stage 2/5 — Preprocessing")
        X_train, X_test, y_train, y_test = \
            self.preprocessor.fit_transform(self.loader.df_merged)
        self.X_test, self.y_test = X_test, y_test.values

        logger.info("Stage 3/5 — Handling class imbalance")
        X_bal, y_bal = self.imbalance.balance_training_set(X_train, y_train.values)

        logger.info("Stage 4/5 — Training ensemble")
        base_models    = ModelFactory.make_base_classifiers()
        self.ensemble  = SoftVotingEnsemble(
            base_models=base_models,
            weights=self.config.voting_weights,
            threshold=self.config.decision_threshold,
        )
        self.ensemble.fit(X_bal, y_bal)

        logger.info("Stage 5/5 — Evaluating models")
        self._evaluate_all_models()

        self.is_trained = True
        logger.info("Pipeline training complete.")
        return self.evaluator.comparison_table().to_dict(orient="index")

    # ─── Inference ─────────────────────────────────────────────

    def predict_single(self, user_inputs: dict) -> Tuple[float, str]:
        """
        Predict for a single user via the web app.

        Returns
        -------
        probability : float in [0, 1]
        risk_label  : human-readable risk band name
        """
        from .config import get_risk_band
        if not self.is_trained:
            raise RuntimeError("Pipeline must be trained before predict_single.")

        X = self.preprocessor.transform_single(user_inputs)
        probability = float(self.ensemble.predict_proba(X)[0])
        band = get_risk_band(probability)
        return probability, band.name

    # ─── Persistence ───────────────────────────────────────────

    def save(self, path: Path | str = MODEL_DIR / "pipeline.pkl") -> Path:
        """Serialise the entire trained pipeline to a single pickle file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Pipeline saved to %s", path)
        return path

    @classmethod
    def load(cls, path: Path | str = MODEL_DIR / "pipeline.pkl"
             ) -> "HeartDiseasePipeline":
        """Reload a previously trained pipeline from disk."""
        with open(path, "rb") as f:
            return pickle.load(f)

    # ─── Diagnostics ───────────────────────────────────────────

    def get_feature_importance(self) -> pd.Series:
        """Random-Forest feature importance, indexed by feature name."""
        rf = self.ensemble.base_models["RandomForest"]
        return pd.Series(
            rf.feature_importances_,
            index=self.preprocessor.get_feature_names(),
        ).sort_values(ascending=False)

    def get_diversity_correlations(self) -> Dict[Tuple[str, str], float]:
        """Pairwise correlation between base-model probabilities (test set)."""
        return self.ensemble.diversity_correlation(self.X_test)

    def summary(self) -> str:
        """Multi-section text summary of the trained pipeline."""
        parts = ["=" * 60, "Heart Disease Pipeline Summary", "=" * 60, ""]
        parts.append(self.loader.summary())
        parts.append(self.imbalance.summary())
        parts.append("Test-set Metrics")
        parts.append(self.evaluator.comparison_table().to_string())
        return "\n".join(parts)

    # ─── Internals ─────────────────────────────────────────────

    def _evaluate_all_models(self) -> None:
        """Evaluate the three base models plus the ensemble on the test set."""
        # Individual base models
        for name, model in self.ensemble.base_models.items():
            y_pred  = model.predict(self.X_test)
            y_proba = model.predict_proba(self.X_test)[:, 1]
            self.evaluator.evaluate(name, self.y_test, y_pred, y_proba)

        # Ensemble itself
        y_pred  = self.ensemble.predict(self.X_test)
        y_proba = self.ensemble.predict_proba(self.X_test)
        self.evaluator.evaluate("VotingEnsemble", self.y_test, y_pred, y_proba)
