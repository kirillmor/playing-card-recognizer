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


def compute_summary_metrics_from_outputs(
    logits: Tensor,
    targets: Tensor,
    num_classes: int,
    class_names: list[str],
    top_k: int,
) -> dict[str, Any]:
    """Compute summary metrics directly from logits and targets."""
    predictions = logits.argmax(dim=1)

    confusion_matrix = compute_confusion_matrix(
        targets=targets,
        predictions=predictions,
        num_classes=num_classes,
    )
    classification_report = build_classification_report(
        confusion_matrix=confusion_matrix,
        class_names=class_names,
    )
    top_k_accuracy = compute_top_k_accuracy(
        logits=logits,
        targets=targets,
        top_k=top_k,
    )

    return summarize_metrics(
        confusion_matrix=confusion_matrix,
        classification_report=classification_report,
        top_k_accuracy=top_k_accuracy,
    )


def bootstrap_summary_metric_confidence_intervals(
    logits: Tensor,
    targets: Tensor,
    num_classes: int,
    class_names: list[str],
    top_k: int,
    num_bootstrap_samples: int,
    confidence_level: float,
    seed: int,
) -> pd.DataFrame:
    """Estimate confidence intervals for summary metrics using bootstrap sampling."""
    if num_bootstrap_samples <= 0:
        raise ValueError("num_bootstrap_samples must be positive.")

    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1.")

    logits = logits.cpu()
    targets = targets.cpu()

    num_examples = targets.numel()
    if num_examples == 0:
        raise ValueError("Cannot bootstrap metrics for an empty target tensor.")

    generator = torch.Generator()
    generator.manual_seed(seed)

    bootstrap_values: dict[str, list[float]] = {
        "accuracy": [],
        "macro_precision": [],
        "macro_recall": [],
        "macro_f1": [],
        "weighted_f1": [],
        "top_k_accuracy": [],
    }

    for _ in range(num_bootstrap_samples):
        sample_indices = torch.randint(
            low=0,
            high=num_examples,
            size=(num_examples,),
            generator=generator,
        )

        sampled_logits = logits[sample_indices]
        sampled_targets = targets[sample_indices]

        metrics = compute_summary_metrics_from_outputs(
            logits=sampled_logits,
            targets=sampled_targets,
            num_classes=num_classes,
            class_names=class_names,
            top_k=top_k,
        )

        for metric_name in bootstrap_values:
            bootstrap_values[metric_name].append(float(metrics[metric_name]))

    alpha = 1.0 - confidence_level
    lower_quantile = alpha / 2.0
    upper_quantile = 1.0 - alpha / 2.0

    rows: list[dict[str, float | str]] = []

    for metric_name, values in bootstrap_values.items():
        values_tensor = torch.tensor(values, dtype=torch.float64)

        rows.append(
            {
                "metric": metric_name,
                "mean": float(values_tensor.mean().item()),
                "std": float(values_tensor.std(unbiased=True).item()),
                "ci_lower": float(torch.quantile(values_tensor, lower_quantile).item()),
                "ci_upper": float(torch.quantile(values_tensor, upper_quantile).item()),
                "confidence_level": confidence_level,
                "num_bootstrap_samples": num_bootstrap_samples,
            }
        )

    return pd.DataFrame(rows)


def _safe_divide(numerator: Tensor, denominator: Tensor) -> Tensor:
    """Divide tensors and return zero where denominator is zero."""
    return torch.where(
        denominator > 0,
        numerator / denominator,
        torch.zeros_like(numerator),
    )
