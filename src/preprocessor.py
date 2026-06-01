"""
Feature preprocessing
=====================

Handles the three encoding strategies and the partition-then-fit pattern
that keeps the test set independent. The class is stateful: it learns
encoding parameters during fit() on the training partition and reuses
them in transform() for the test partition or for live inference.

This separation matters because applying a fresh StandardScaler to the
test partition would let test statistics leak into the model's view of
the data. By preserving the fitted scaler, we guarantee that the test
records are transformed using only what was learned from training.
"""

from __future__ import annotations
from typing import List, Tuple, Optional
import logging

import numpy  as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler

from .config import (
    BINARY_COLUMNS, AGE_BRACKETS, ONE_HOT_COLUMNS,
    TARGET_COLUMN, TEST_FRACTION, RANDOM_SEED,
)


logger = logging.getLogger(__name__)


class FeaturePreprocessor:
    """
    End-to-end feature preprocessor with leakage-safe partitioning.

    Three encoding strategies are applied based on the type of each
    feature:

      Binary (Yes/No)        →  mapped to {0, 1}
      Ordered categorical    →  integer-encoded preserving order
      Unordered categorical  →  one-hot encoded with drop_first

    The class follows the standard scikit-learn fit/transform contract
    so it can be slotted into a pipeline alongside other estimators.

    Usage
    -----
    >>> pp = FeaturePreprocessor()
    >>> X_train, X_test, y_train, y_test = pp.fit_transform(df)
    >>> X_user_encoded = pp.transform_single(user_inputs)  # for inference
    """

    def __init__(self, test_fraction: float = TEST_FRACTION,
                 random_seed: int = RANDOM_SEED) -> None:
        self.test_fraction = test_fraction
        self.random_seed   = random_seed

        # Fitted state (populated during fit_transform)
        self.scaler:        Optional[StandardScaler] = None
        self.feature_names: Optional[List[str]]      = None
        self.original_columns: Optional[List[str]]   = None
        self.is_fitted:     bool                     = False

    # ─── Public API ────────────────────────────────────────────

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray,
                                                       pd.Series, pd.Series]:
        """
        Apply all preprocessing steps in the correct order and return
        four scaled arrays ready for model training.

        Order matters:
          1. Encode categoricals (no statistics involved → safe before split)
          2. Stratified 80/20 split
          3. Fit StandardScaler on training partition only
          4. Apply the fitted scaler to both partitions

        Returns
        -------
        X_train_scaled : numpy array, shape (n_train, n_features)
        X_test_scaled  : numpy array, shape (n_test, n_features)
        y_train        : pandas Series
        y_test         : pandas Series
        """
        encoded = self._encode_features(df)

        # Separate target before scaling
        y = encoded[TARGET_COLUMN]
        X = encoded.drop(columns=[TARGET_COLUMN])
        self.feature_names = list(X.columns)

        # Stratified split keeps the 9% positive rate in both partitions
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_fraction,
            stratify=y,
            random_state=self.random_seed,
        )
        logger.info("Train: %d records, Test: %d records",
                    len(X_train), len(X_test))

        # Fit the scaler on training only — this is the leakage barrier
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled  = self.scaler.transform(X_test)

        self.is_fitted = True
        return X_train_scaled, X_test_scaled, y_train, y_test

    def transform_single(self, raw_inputs: dict) -> np.ndarray:
        """
        Encode a single user input dictionary into the scaled feature
        vector expected by the trained models. Used by the web app
        for live inference.

        Raises RuntimeError if called before fit_transform().
        """
        if not self.is_fitted:
            raise RuntimeError(
                "FeaturePreprocessor must be fitted before transform_single()."
            )

        # Build a one-row DataFrame with the same columns as training
        row = self._build_inference_row(raw_inputs)
        return self.scaler.transform(row.values)

    def get_feature_names(self) -> List[str]:
        """Return the list of feature names after encoding."""
        if not self.is_fitted:
            raise RuntimeError("Call fit_transform() first.")
        return list(self.feature_names)

    # ─── Encoding implementation ───────────────────────────────

    def _encode_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all three encoding strategies. This method is decomposed
        into named helpers below so the thesis text can refer to each
        stage individually.
        """
        out = df.copy()
        out = self._encode_binary_columns(out)
        out = self._encode_ordinal_age(out)
        out = self._encode_one_hot(out)
        out = self._coerce_numeric_and_clean(out)
        return out

    def _encode_binary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Yes/No → 1/0 for the nine binary columns (target included)."""
        for col in BINARY_COLUMNS:
            if col in df.columns:
                df[col] = df[col].map({"Yes": 1, "No": 0})
        return df

    def _encode_ordinal_age(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Age brackets carry order — 65-69 sits closer to 70-74 than to
        18-24. Integer encoding preserves that. One-hot would discard
        it and waste a strong predictor.
        """
        if "AgeCategory" not in df.columns:
            return df
        df["AgeCategory"] = pd.Categorical(
            df["AgeCategory"],
            categories=AGE_BRACKETS,
            ordered=True,
        ).codes
        return df

    def _encode_one_hot(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Unordered categoricals → one-hot with drop_first=True.
        Dropping one column per variable avoids perfect multicollinearity
        between the resulting dummy columns.
        """
        return pd.get_dummies(
            df,
            columns=[c for c in ONE_HOT_COLUMNS if c in df.columns],
            drop_first=True,
        )

    def _coerce_numeric_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Make sure every column is numeric, then drop any rows where
        the encoding produced NaN (would happen if a value outside the
        expected categories slipped through).
        """
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        before = len(df)
        df = df.dropna().reset_index(drop=True)
        if (dropped := before - len(df)) > 0:
            logger.warning("Dropped %d rows containing post-encoding NaN", dropped)
        return df

    # ─── Inference-time encoding ───────────────────────────────

    def _build_inference_row(self, inputs: dict) -> pd.DataFrame:
        """
        Convert a dict of human-readable inputs from the web form into
        a single-row DataFrame with the same column order as the
        training matrix.
        """
        # Initialise every column to 0 — most one-hot columns stay at 0
        row = {feat: 0 for feat in self.feature_names}

        # Numeric and ordinal — direct copy
        if "BMI" in row:              row["BMI"]            = inputs.get("bmi", 0.0)
        if "PhysicalHealth" in row:   row["PhysicalHealth"] = inputs.get("physical_health", 0)
        if "MentalHealth" in row:     row["MentalHealth"]   = inputs.get("mental_health", 0)
        if "SleepTime" in row:        row["SleepTime"]      = inputs.get("sleep_time", 7)
        if "AgeCategory" in row:
            age_str = inputs.get("age", "30-34")
            row["AgeCategory"] = AGE_BRACKETS.index(age_str) if age_str in AGE_BRACKETS else 0

        # Binary fields — explicit yes-to-1 mapping
        binary_inputs = {
            "Smoking":          inputs.get("smoking",           "No"),
            "AlcoholDrinking":  inputs.get("alcohol",           "No"),
            "Stroke":           inputs.get("stroke",            "No"),
            "DiffWalking":      inputs.get("diff_walking",      "No"),
            "PhysicalActivity": inputs.get("physical_activity", "Yes"),
            "Asthma":           inputs.get("asthma",            "No"),
            "KidneyDisease":    inputs.get("kidney_disease",    "No"),
            "SkinCancer":       inputs.get("skin_cancer",       "No"),
        }
        for col, val in binary_inputs.items():
            if col in row:
                row[col] = 1 if val == "Yes" else 0

        # One-hot fields — set the matching column to 1 (others stay 0)
        # because drop_first=True, the reference categories are absent
        # by design (Female / American Indian-Alaskan Native / No / Excellent)
        for prefix, value in (("Sex",       inputs.get("sex",      "Female")),
                              ("Race",      inputs.get("race",     "White")),
                              ("Diabetic",  inputs.get("diabetic", "No")),
                              ("GenHealth", inputs.get("gen_health", "Good"))):
            key = f"{prefix}_{value}"
            if key in row:
                row[key] = 1

        return pd.DataFrame([row])[self.feature_names]
