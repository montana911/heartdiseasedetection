"""
Model definitions and the custom soft-voting ensemble
=====================================================

This module contains two responsibilities split into two classes:

  ModelFactory
      Constructs the three base classifiers with the hyperparameters
      specified in config.py. Centralising construction means a
      hyperparameter change only needs to happen in one place.

  SoftVotingEnsemble
      Custom implementation of soft voting. Rather than using sklearn's
      built-in VotingClassifier, this class explicitly implements the
      averaging step so the algorithm is visible in the code:

          P_ensemble(y=1) = Σ wₖ · Pₖ(y=1)  /  Σ wₖ

      The class also offers per-base-model probability access — useful
      for diagnostics and for the diversity-correlation analysis that
      justified dropping stacking.
"""

from __future__ import annotations
from typing import Dict, List, Tuple
import logging

import numpy as np

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import (
    RandomForestClassifier, HistGradientBoostingClassifier,
)

from .config import (
    LOGISTIC_REGRESSION_PARAMS,
    RANDOM_FOREST_PARAMS,
    HIST_GRADIENT_BOOSTING_PARAMS,
    VOTING_WEIGHTS,
)


logger = logging.getLogger(__name__)


class ModelFactory:
    """
    Constructs the three base classifiers from centralised config.

    Keeping construction in one place means the rest of the code never
    sees raw hyperparameter dictionaries — it just asks for "the base
    classifiers" and receives correctly-configured instances.
    """

    @staticmethod
    def make_logistic_regression() -> LogisticRegression:
        """Linear baseline with balanced class weighting."""
        return LogisticRegression(**LOGISTIC_REGRESSION_PARAMS)

    @staticmethod
    def make_random_forest() -> RandomForestClassifier:
        """Tree ensemble with feature importance and balanced classes."""
        return RandomForestClassifier(**RANDOM_FOREST_PARAMS)

    @staticmethod
    def make_hist_gradient_boosting() -> HistGradientBoostingClassifier:
        """Histogram-based boosting, native imbalance handling."""
        return HistGradientBoostingClassifier(**HIST_GRADIENT_BOOSTING_PARAMS)

    @classmethod
    def make_base_classifiers(cls) -> Dict[str, object]:
        """
        Return an ordered dictionary of the three base classifiers,
        keyed by name. The order matters for reproducibility and for
        the visualisations that compare them side-by-side.
        """
        return {
            "LogisticRegression":   cls.make_logistic_regression(),
            "RandomForest":         cls.make_random_forest(),
            "HistGradientBoosting": cls.make_hist_gradient_boosting(),
        }


class SoftVotingEnsemble:
    """
    Manual implementation of weighted soft voting.

    Unlike sklearn's VotingClassifier, this class:

    1. Stores each fitted base model individually so we can interrogate
       them separately (used for the inter-model correlation analysis
       that justified dropping stacking).
    2. Computes the weighted probability average explicitly rather than
       delegating to sklearn internals, making the algorithm readable
       in the code.
    3. Exposes per-base-model probabilities through predict_proba_per_model(),
       which sklearn's VotingClassifier does not expose directly.

    The decision rule is the standard one:

        ŷ = 1 if P_ensemble(y=1) ≥ threshold else 0

    with the threshold defaulting to 0.5 but tunable for deployment.

    Usage
    -----
    >>> ensemble = SoftVotingEnsemble(ModelFactory.make_base_classifiers())
    >>> ensemble.fit(X_train, y_train)
    >>> probs = ensemble.predict_proba(X_test)        # combined probability
    >>> per_model = ensemble.predict_proba_per_model(X_test)  # diagnostic
    """

    def __init__(self,
                 base_models: Dict[str, object],
                 weights: Dict[str, float] = None,
                 threshold: float = 0.5) -> None:
        if not base_models:
            raise ValueError("At least one base model is required.")

        self.base_models = base_models
        self.weights     = self._normalise_weights(weights or VOTING_WEIGHTS,
                                                    base_models.keys())
        self.threshold   = threshold
        self.is_fitted   = False

    # ─── Training ──────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SoftVotingEnsemble":
        """
        Train every base model on the same training data. Returns self
        so the call chains in the standard scikit-learn way.
        """
        logger.info("Fitting %d base models …", len(self.base_models))
        for name, model in self.base_models.items():
            logger.info("  Training %s", name)
            model.fit(X, y)
        self.is_fitted = True
        return self

    # ─── Prediction ────────────────────────────────────────────

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Combined positive-class probability for each row of X.

        Implements the weighted average explicitly:

            P_ensemble(y=1 | x) = Σₖ (wₖ · Pₖ(y=1 | x)) / Σₖ wₖ

        Returns
        -------
        array of shape (n_samples,) containing values in [0, 1].
        """
        self._require_fitted()
        per_model = self.predict_proba_per_model(X)

        weighted_sum  = np.zeros(X.shape[0])
        weight_total  = 0.0
        for name, probs in per_model.items():
            w = self.weights[name]
            weighted_sum += w * probs
            weight_total += w
        return weighted_sum / weight_total

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Hard class prediction by thresholding the combined probability.
        """
        return (self.predict_proba(X) >= self.threshold).astype(int)

    def predict_proba_per_model(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Per-base-model positive-class probabilities. Used by the
        diversity analysis to compute pairwise Pearson correlations
        between the base models' outputs.
        """
        self._require_fitted()
        return {
            name: model.predict_proba(X)[:, 1]
            for name, model in self.base_models.items()
        }

    # ─── Diagnostics ───────────────────────────────────────────

    def diversity_correlation(self, X: np.ndarray) -> Dict[Tuple[str, str], float]:
        """
        Pairwise Pearson correlations between base-model probabilities.

        High pairwise correlation (above ~0.9) means the base models
        are making similar predictions, in which case stacking adds
        no value over voting — the empirical justification for the
        ensemble design choice documented in Chapter IV of the thesis.
        """
        per_model = self.predict_proba_per_model(X)
        names = list(per_model.keys())
        result: Dict[Tuple[str, str], float] = {}
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                result[(a, b)] = float(np.corrcoef(per_model[a],
                                                    per_model[b])[0, 1])
        return result

    # ─── Internals ─────────────────────────────────────────────

    @staticmethod
    def _normalise_weights(raw: Dict[str, float],
                             model_keys) -> Dict[str, float]:
        """
        Fill in any missing weights with 1.0 (equal voting) and drop
        weights for models that are not in the ensemble.
        """
        return {name: float(raw.get(name, 1.0)) for name in model_keys}

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(
                "SoftVotingEnsemble must be fitted before prediction."
            )
