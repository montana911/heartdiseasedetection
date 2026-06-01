"""
Class imbalance handling
========================

Wraps SMOTE oversampling behind a class that documents the leakage-safe
usage pattern and makes the contract explicit: balancing only happens on
training data, and only the post-split scaled training matrix is ever
passed in.

The wrapper is deliberately thin — the synthetic minority generation
algorithm itself comes from imbalanced-learn — but the wrapper exists
because the methodology around when and how to apply it is the part
that matters for this thesis.
"""

from __future__ import annotations
from typing import Tuple
import logging

import numpy as np
from imblearn.over_sampling import SMOTE

from .config import SMOTE_PARAMS, RANDOM_SEED


logger = logging.getLogger(__name__)


class ImbalanceHandler:
    """
    SMOTE-based oversampling of the minority class.

    SMOTE (Synthetic Minority Over-sampling Technique) generates new
    minority samples by interpolating between each existing minority
    point and one of its k nearest neighbours of the same class:

        x_new = x_i + λ · (x_neighbour - x_i)

    with λ sampled uniformly from [0, 1]. This produces meaningful
    variation rather than duplicate copies, which helps the model learn
    smoother boundaries in the minority region.

    The class only exposes balance_training_set() because the design
    enforces single-purpose usage. Calling it with anything other than
    training data is the leakage error this thesis explicitly avoids.

    Usage
    -----
    >>> handler = ImbalanceHandler()
    >>> X_balanced, y_balanced = handler.balance_training_set(X_train, y_train)
    """

    def __init__(self, k_neighbors: int = 5,
                 random_seed:  int = RANDOM_SEED) -> None:
        self.k_neighbors = k_neighbors
        self.random_seed = random_seed
        self._smote = SMOTE(
            k_neighbors=k_neighbors,
            random_state=random_seed,
        )
        # Recorded after the first call so summary() can describe the effect
        self._original_size:  int | None = None
        self._balanced_size:  int | None = None
        self._minority_before: int | None = None
        self._minority_after:  int | None = None

    def balance_training_set(self, X_train: np.ndarray,
                              y_train: np.ndarray
                              ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply SMOTE to the training partition and return the balanced
        feature matrix and target vector.

        The method name includes "training_set" deliberately to make
        callers aware that passing anything else (a full dataset, or a
        validation fold) is a bug — that's exactly the leakage pattern
        the methodology section warns about.

        Parameters
        ----------
        X_train : array of shape (n_train, n_features)
                  The scaled training features.
        y_train : array of shape (n_train,)
                  The training labels.

        Returns
        -------
        X_balanced, y_balanced : balanced versions of the inputs,
                                 sized roughly 2 × majority count.
        """
        self._original_size   = len(y_train)
        self._minority_before = int(np.sum(y_train == 1))

        logger.info(
            "Applying SMOTE — minority class will grow from %d to match the "
            "majority of %d records …",
            self._minority_before,
            self._original_size - self._minority_before,
        )

        X_bal, y_bal = self._smote.fit_resample(X_train, y_train)

        self._balanced_size  = len(y_bal)
        self._minority_after = int(np.sum(y_bal == 1))

        logger.info("SMOTE complete: %d → %d records",
                    self._original_size, self._balanced_size)
        return X_bal, y_bal

    def summary(self) -> str:
        """Report what the most recent balance_training_set() call did."""
        if self._original_size is None:
            return "ImbalanceHandler: not yet applied to any data."

        pct_before = self._minority_before / self._original_size * 100
        pct_after  = self._minority_after  / self._balanced_size  * 100
        return (
            f"SMOTE Summary\n"
            f"  Before : {self._original_size:,} records "
            f"({pct_before:.1f}% positive)\n"
            f"  After  : {self._balanced_size:,} records "
            f"({pct_after:.1f}% positive)\n"
            f"  Added  : {self._minority_after - self._minority_before:,} "
            f"synthetic positive cases\n"
        )
