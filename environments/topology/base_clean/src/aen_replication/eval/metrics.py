"""Metrics for binary ambiguity classification."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def safe_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Return AUROC when both classes are present, else NaN."""

    labels = np.unique(y_true)
    if labels.size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def binary_classification_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.0,
) -> dict[str, Any]:
    """Compute balanced binary metrics from decision scores."""

    scores = np.asarray(scores, dtype=float)
    if not np.all(np.isfinite(scores)):
        scores = np.nan_to_num(scores, nan=threshold, posinf=threshold, neginf=threshold)
    predictions = (scores >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    macro_precision = float(precision_score(y_true, predictions, average="macro", zero_division=0))
    macro_recall = float(recall_score(y_true, predictions, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true, predictions, average="macro", zero_division=0))
    return {
        "auroc": safe_auroc(y_true, scores),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": macro_precision,
        "recall": macro_recall,
        "f1": macro_f1,
        "macro_f1": macro_f1,
        "confusion_matrix": matrix.tolist(),
    }
