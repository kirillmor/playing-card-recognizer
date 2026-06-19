from __future__ import annotations

from typing import Any

import pandas as pd
import torch
from torch import Tensor


def compute_confusion_matrix(
    targets: Tensor,
    predictions: Tensor,
    num_classes: int,
) -> Tensor:
    """Compute multiclass confusion matrix.

    Rows are true classes, columns are predicted classes.
    """
    targets = targets.to(dtype=torch.int64).cpu()
    predictions = predictions.to(dtype=torch.int64).cpu()

    indices = targets * num_classes + predictions
    counts = torch.bincount(indices, minlength=num_classes * num_classes)

    return counts.reshape(num_classes, num_classes)


def build_classification_report(
    confusion_matrix: Tensor,
    class_names: list[str],
) -> pd.DataFrame:
    """Build per-class precision, recall, F1, and support table."""
    matrix = confusion_matrix.to(dtype=torch.float64)

    true_positive = torch.diag(matrix)
    support = matrix.sum(dim=1)
    predicted_positive = matrix.sum(dim=0)

    precision = _safe_divide(true_positive, predicted_positive)
    recall = _safe_divide(true_positive, support)
    f1_score = _safe_divide(2 * precision * recall, precision + recall)

    return pd.DataFrame(
        {
            "class_name": class_names,
            "support": support.to(dtype=torch.int64).tolist(),
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "f1_score": f1_score.tolist(),
        }
    )


def summarize_metrics(
    confusion_matrix: Tensor,
    classification_report: pd.DataFrame,
    top_k_accuracy: float,
) -> dict[str, Any]:
    """Compute summary metrics from confusion matrix and per-class report."""
    matrix = confusion_matrix.to(dtype=torch.float64)

    correct = torch.diag(matrix).sum().item()
    total = matrix.sum().item()
    accuracy = correct / total if total > 0 else 0.0

    support = classification_report["support"]
    f1_score = classification_report["f1_score"]

    weighted_f1 = (
        float((f1_score * support).sum() / support.sum()) if int(support.sum()) > 0 else 0.0
    )

    return {
        "num_samples": int(total),
        "accuracy": accuracy,
        "macro_precision": float(classification_report["precision"].mean()),
        "macro_recall": float(classification_report["recall"].mean()),
        "macro_f1": float(classification_report["f1_score"].mean()),
        "weighted_f1": weighted_f1,
        "top_k_accuracy": float(top_k_accuracy),
    }


def compute_top_k_accuracy(
    logits: Tensor,
    targets: Tensor,
    top_k: int,
) -> float:
    """Compute top-k accuracy from logits."""
    top_k = min(top_k, logits.shape[1])

    top_predictions = logits.topk(k=top_k, dim=1).indices
    correct = top_predictions.eq(targets.unsqueeze(1)).any(dim=1)

    return float(correct.to(dtype=torch.float32).mean().item())


def _safe_divide(numerator: Tensor, denominator: Tensor) -> Tensor:
    """Divide tensors and return zero where denominator is zero."""
    return torch.where(
        denominator > 0,
        numerator / denominator,
        torch.zeros_like(numerator),
    )
