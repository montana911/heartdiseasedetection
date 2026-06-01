"""
Model evaluation
================

The ModelEvaluator computes the full metric set used throughout the
thesis. Most metrics could be obtained by a single call to sklearn,
but where the underlying formula is short and illustrative, the
calculation is performed manually so the algorithm is visible in
the code rather than hidden behind a library call.

Manual implementations live here for:

  Accuracy   = (TP + TN) / total
  Precision  = TP / (TP + FP)
  Recall     = TP / (TP + FN)
  F1-Score   = 2·P·R / (P + R)

Sklearn is still used for roc_auc_score because computing AUC
correctly requires the full ROC curve, which is not as readable
when expanded inline.
"""

from __future__ import annotations
from typing import Dict, Tuple, List
from dataclasses import dataclass
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

from .config import METRIC_ORDER


logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Container for one model's evaluation on a single test set."""
    model_name:        str
    accuracy:          float
    precision:         float
    recall:            float
    f1_score:          float
    auc_roc:           float
    true_positive:     int
    true_negative:     int
    false_positive:    int
    false_negative:    int
    y_true:            np.ndarray
    y_pred:            np.ndarray
    y_proba:           np.ndarray

    def as_dict(self) -> Dict[str, float]:
        """Metric-only view for table display."""
        return {
            "Accuracy":  self.accuracy,
            "Precision": self.precision,
            "Recall":    self.recall,
            "F1-Score":  self.f1_score,
            "AUC-ROC":   self.auc_roc,
        }

    def confusion_counts(self) -> Tuple[int, int, int, int]:
        """Return (TN, FP, FN, TP) — the standard ordering."""
        return (self.true_negative, self.false_positive,
                self.false_negative, self.true_positive)


class ModelEvaluator:
    """
    Evaluates a fitted model on a held-out test set.

    The evaluator computes every metric used in the thesis tables and
    figures from a single set of (y_true, y_pred, y_proba). Designed
    so the same instance can score multiple models for direct
    comparison.
    """

    def __init__(self) -> None:
        self._results: Dict[str, EvaluationResult] = {}

    # ─── Public API ────────────────────────────────────────────

    def evaluate(self, model_name: str,
                 y_true:  np.ndarray,
                 y_pred:  np.ndarray,
                 y_proba: np.ndarray) -> EvaluationResult:
        """
        Compute every metric for one model and store the result.

        Parameters
        ----------
        model_name : key used in the comparison table
        y_true     : the actual labels from the test set
        y_pred     : hard 0/1 predictions
        y_proba    : positive-class probabilities (for AUC-ROC)
        """
        tn, fp, fn, tp = self._unpack_confusion_matrix(y_true, y_pred)

        result = EvaluationResult(
            model_name=model_name,
            accuracy        = self._accuracy(tp, tn, fp, fn),
            precision       = self._precision(tp, fp),
            recall          = self._recall(tp, fn),
            f1_score        = self._f1_score(tp, fp, fn),
            auc_roc         = roc_auc_score(y_true, y_proba),
            true_positive   = int(tp),
            true_negative   = int(tn),
            false_positive  = int(fp),
            false_negative  = int(fn),
            y_true  = y_true,
            y_pred  = y_pred,
            y_proba = y_proba,
        )

        self._results[model_name] = result
        return result

    def comparison_table(self) -> pd.DataFrame:
        """
        Side-by-side metric table for all evaluated models.

        Columns ordered by METRIC_ORDER (config.py) so the most
        informative metrics (AUC-ROC, Recall) come first.
        """
        if not self._results:
            raise RuntimeError("No models evaluated yet.")
        rows = {name: r.as_dict() for name, r in self._results.items()}
        df = pd.DataFrame(rows).T
        df = df[METRIC_ORDER]
        return df.round(4)

    def clinical_summary(self, model_name: str,
                          population_size: int = 100_000,
                          base_rate: float = 0.091) -> str:
        """
        Translate a confusion matrix into a deployment-scale description.

        Used in Chapter VII to make the numbers concrete: instead of
        reporting "Recall = 0.78", we say "of every 100,000 people
        screened, the model catches around 7,040 of the 9,060 actual
        cases".
        """
        if model_name not in self._results:
            raise KeyError(f"No evaluation found for {model_name}.")

        r = self._results[model_name]
        actual_positives    = int(population_size * base_rate)
        actual_negatives    = population_size - actual_positives
        caught              = int(r.recall * actual_positives)
        missed              = actual_positives - caught
        false_alarms        = int(actual_negatives * (1 - r.true_negative /
                                  (r.true_negative + r.false_positive)))

        return (
            f"Clinical translation ({model_name}, {population_size:,} screened)\n"
            f"  Real positive cases     : {actual_positives:,}\n"
            f"  Correctly flagged       : ~{caught:,}\n"
            f"  Missed (false negatives): ~{missed:,}\n"
            f"  False alarms            : ~{false_alarms:,}\n"
        )

    # ─── Metric implementations (manual) ───────────────────────
    # These are the formulas readers of the thesis are expected to
    # know. Expanding them in code rather than calling a library makes
    # the algorithm visible.

    @staticmethod
    def _accuracy(tp: int, tn: int, fp: int, fn: int) -> float:
        total = tp + tn + fp + fn
        return float((tp + tn) / total) if total else 0.0

    @staticmethod
    def _precision(tp: int, fp: int) -> float:
        denom = tp + fp
        return float(tp / denom) if denom else 0.0

    @staticmethod
    def _recall(tp: int, fn: int) -> float:
        denom = tp + fn
        return float(tp / denom) if denom else 0.0

    @classmethod
    def _f1_score(cls, tp: int, fp: int, fn: int) -> float:
        precision = cls._precision(tp, fp)
        recall    = cls._recall(tp, fn)
        denom     = precision + recall
        return float(2 * precision * recall / denom) if denom else 0.0

    @staticmethod
    def _unpack_confusion_matrix(y_true: np.ndarray,
                                  y_pred: np.ndarray
                                  ) -> Tuple[int, int, int, int]:
        """Return (TN, FP, FN, TP) in the standard sklearn ordering."""
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        return int(tn), int(fp), int(fn), int(tp)

    # ─── Access ────────────────────────────────────────────────

    def get_result(self, model_name: str) -> EvaluationResult:
        if model_name not in self._results:
            raise KeyError(f"No evaluation found for {model_name}.")
        return self._results[model_name]

    @property
    def model_names(self) -> List[str]:
        return list(self._results.keys())
