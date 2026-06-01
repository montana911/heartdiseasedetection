"""
Visualisation
=============

Collects every plotting routine used in the thesis behind a single
ResultsVisualizer class. The class operates on the EvaluationResult
objects produced by ModelEvaluator, so plots and tables share the
same numbers — there is no risk of the figures showing one set of
results and the tables showing another.

Output goes to two places:

  - PNG files on disk (for embedding in the thesis document)
  - matplotlib Figure objects (for inline display in the web app)
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve

from .evaluator import EvaluationResult
from .config    import OUTPUT_DIR


# Colour palette consistent across every plot in the thesis
PALETTE = {
    "LogisticRegression":   "#3498db",
    "RandomForest":         "#e67e22",
    "HistGradientBoosting": "#2ecc71",
    "VotingEnsemble":       "#e74c3c",
}


class ResultsVisualizer:
    """
    All thesis plots in one place.

    Each plot is a separate method so it can be regenerated individually
    when only some results change.

    Usage
    -----
    >>> viz = ResultsVisualizer(evaluator)
    >>> viz.plot_metric_comparison(save_to="metrics.png")
    >>> viz.plot_roc_curves(save_to="roc.png")
    """

    def __init__(self, evaluator, output_dir: Path = OUTPUT_DIR) -> None:
        self.evaluator  = evaluator
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid", font_scale=1.0)

    # ─── Plot 1: Metric comparison bars ────────────────────────

    def plot_metric_comparison(self, save_to: str | None = None) -> plt.Figure:
        """All metrics for all models in one grouped bar chart."""
        table = self.evaluator.comparison_table()
        fig, ax = plt.subplots(figsize=(11, 5.5))
        table.plot(kind="bar", ax=ax, edgecolor="white",
                   width=0.78, colormap="RdYlGn")
        ax.set_title("All Models — All Metrics", fontweight="bold")
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("Score")
        ax.tick_params(axis="x", rotation=15)
        ax.axhline(0.8, color="gray", linestyle="--", lw=0.8, alpha=0.6)
        ax.annotate("Low F1 expected\n(9% imbalance)",
                    xy=(0.01, 0.36), xycoords="axes fraction",
                    fontsize=9, color="#c0392b", style="italic")
        ax.legend(loc="upper right", fontsize=9)
        plt.tight_layout()
        return self._maybe_save(fig, save_to)

    # ─── Plot 2: ROC curves ────────────────────────────────────

    def plot_roc_curves(self, save_to: str | None = None) -> plt.Figure:
        """Overlay all models' ROC curves on a single set of axes."""
        fig, ax = plt.subplots(figsize=(7.5, 6))
        for name in self.evaluator.model_names:
            r = self.evaluator.get_result(name)
            fpr, tpr, _ = roc_curve(r.y_true, r.y_proba)
            ax.plot(fpr, tpr, lw=2,
                    color=PALETTE.get(name, "#555555"),
                    label=f"{name}  (AUC {r.auc_roc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
        ax.set_title("ROC Curves", fontweight="bold")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right", fontsize=10)
        plt.tight_layout()
        return self._maybe_save(fig, save_to)

    # ─── Plot 3: Feature importance ────────────────────────────

    def plot_feature_importance(self, feature_names: List[str],
                                  importances:    np.ndarray,
                                  top_n: int = 14,
                                  save_to: str | None = None) -> plt.Figure:
        """Top-N features ranked by their importance score."""
        series = pd.Series(importances, index=feature_names)\
                   .nlargest(top_n)\
                   .sort_values()
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(series.index, series.values,
                color="#e74c3c", edgecolor="white", alpha=0.85)
        ax.set_title(f"Top {top_n} Features (Random Forest)",
                     fontweight="bold")
        ax.set_xlabel("Importance Score")
        plt.tight_layout()
        return self._maybe_save(fig, save_to)

    # ─── Plot 4 & 5: Confusion matrices ────────────────────────

    def plot_confusion_matrix(self, model_name: str,
                               cmap: str = "Blues",
                               save_to: str | None = None) -> plt.Figure:
        """Confusion matrix heat-map for a single model."""
        r  = self.evaluator.get_result(model_name)
        cm = np.array([[r.true_negative, r.false_positive],
                       [r.false_negative, r.true_positive]])
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(cm, annot=True, fmt=",d", cmap=cmap, ax=ax,
                    xticklabels=["No Disease", "Heart Disease"],
                    yticklabels=["No Disease", "Heart Disease"],
                    linewidths=1, linecolor="white", annot_kws={"size": 14})
        ax.set_title(f"Confusion Matrix — {model_name}", fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        plt.tight_layout()
        return self._maybe_save(fig, save_to)

    # ─── Plot 6: Overall ranking ───────────────────────────────

    def plot_overall_ranking(self, save_to: str | None = None) -> plt.Figure:
        """Composite score (Recall + F1 + AUC) / 3 for each model."""
        table = self.evaluator.comparison_table()
        composite = table[["Recall", "F1-Score", "AUC-ROC"]].mean(axis=1)\
                                                              .sort_values()
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = [PALETTE.get(n, "#888") for n in composite.index]
        ax.barh(composite.index, composite.values, color=colors,
                edgecolor="white")
        ax.set_title("Overall Ranking\n(mean of Recall + F1 + AUC-ROC)",
                     fontweight="bold")
        ax.set_xlim(0, 1)
        for i, (idx, val) in enumerate(composite.items()):
            ax.text(val + 0.005, i, f"{val:.3f}", va="center", fontsize=10)
        plt.tight_layout()
        return self._maybe_save(fig, save_to)

    # ─── Internal helpers ──────────────────────────────────────

    def _maybe_save(self, fig: plt.Figure,
                    save_to: str | None) -> plt.Figure:
        """Optionally write the figure to disk and return it either way."""
        if save_to:
            path = self.output_dir / save_to
            fig.savefig(path, dpi=150, bbox_inches="tight")
        return fig
