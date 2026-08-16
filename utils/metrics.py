"""
=============================================================
utils/metrics.py
=============================================================
Centralized computation of all evaluation metrics:
  Accuracy, Precision, Recall, F1, ROC-AUC, Average Precision
Also provides MetricTracker for accumulating per-batch metrics
during training and evaluation loops.
=============================================================
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve,
    precision_recall_curve, average_precision_score,
)
from typing import Tuple, Dict


def compute_metrics(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute all classification metrics.

    Args:
        y_true:       ground truth labels (0 or 1)
        y_pred_probs: predicted probabilities (0.0 to 1.0)
        threshold:    decision boundary

    Returns:
        dict of metric_name -> float value
    """
    y_pred = (y_pred_probs >= threshold).astype(int)

    metrics = {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"]       = float(roc_auc_score(y_true, y_pred_probs))
        metrics["avg_precision"] = float(average_precision_score(y_true, y_pred_probs))
    else:
        metrics["roc_auc"]       = 0.0
        metrics["avg_precision"] = 0.0

    return metrics


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Return 2×2 confusion matrix [[TN,FP],[FN,TP]]."""
    y_pred = (y_pred_probs >= threshold).astype(int)
    return confusion_matrix(y_true, y_pred)


def get_roc_curve_data(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (fpr, tpr, thresholds) for ROC curve plotting."""
    return roc_curve(y_true, y_pred_probs)


def get_precision_recall_curve_data(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (precision, recall, thresholds) for PR curve plotting."""
    return precision_recall_curve(y_true, y_pred_probs)


def get_classification_report(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    threshold: float = 0.5,
    class_names: list = None,
) -> str:
    """Return sklearn classification report string."""
    if class_names is None:
        class_names = ["Non-Violence", "Violence"]
    y_pred = (y_pred_probs >= threshold).astype(int)
    return classification_report(y_true, y_pred, target_names=class_names)


class MetricTracker:
    """
    Accumulates predictions over a full epoch then computes metrics.

    Usage:
        tracker = MetricTracker()
        for batch in loader:
            probs = model(batch)
            tracker.update(probs, labels, loss.item())
        metrics = tracker.compute()
        tracker.reset()
    """

    def __init__(self):
        self.all_labels  = []
        self.all_probs   = []
        self.total_loss  = 0.0
        self.num_batches = 0

    def update(self, probs: torch.Tensor, labels: torch.Tensor, loss: float = 0.0):
        self.all_probs.extend(probs.detach().cpu().numpy().flatten().tolist())
        self.all_labels.extend(labels.detach().cpu().numpy().flatten().astype(int).tolist())
        self.total_loss  += loss
        self.num_batches += 1

    def compute(self, threshold: float = 0.5) -> Dict[str, float]:
        y_true = np.array(self.all_labels)
        y_pred = np.array(self.all_probs)
        metrics = compute_metrics(y_true, y_pred, threshold)
        metrics["loss"] = self.total_loss / max(self.num_batches, 1)
        return metrics

    def reset(self):
        self.all_labels.clear()
        self.all_probs.clear()
        self.total_loss  = 0.0
        self.num_batches = 0

    def get_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.array(self.all_labels), np.array(self.all_probs)
