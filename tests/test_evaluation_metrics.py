from __future__ import annotations

import pytest
from torch import tensor

from card_recognizer.evaluation.metrics import (
    bootstrap_summary_metric_confidence_intervals,
    build_classification_report,
    compute_confusion_matrix,
    compute_top_k_accuracy,
    summarize_metrics,
)


def test_compute_confusion_matrix() -> None:
    targets = tensor([0, 1, 1, 2])
    predictions = tensor([0, 1, 0, 2])

    confusion_matrix = compute_confusion_matrix(
        targets=targets,
        predictions=predictions,
        num_classes=3,
    )

    assert confusion_matrix.tolist() == [
        [1, 0, 0],
        [1, 1, 0],
        [0, 0, 1],
    ]


def test_classification_report_and_summary_metrics() -> None:
    confusion_matrix = tensor(
        [
            [1, 0, 0],
            [1, 1, 0],
            [0, 0, 1],
        ]
    )

    report = build_classification_report(
        confusion_matrix=confusion_matrix,
        class_names=["a", "b", "c"],
    )
    summary = summarize_metrics(
        confusion_matrix=confusion_matrix,
        classification_report=report,
        top_k_accuracy=1.0,
    )

    assert list(report["class_name"]) == ["a", "b", "c"]
    assert summary["num_samples"] == 4
    assert summary["accuracy"] == pytest.approx(0.75)
    assert summary["top_k_accuracy"] == pytest.approx(1.0)


def test_compute_top_k_accuracy() -> None:
    logits = tensor(
        [
            [3.0, 1.0, 0.0],
            [0.0, 2.0, 3.0],
        ]
    )
    targets = tensor([0, 1])

    top_1_accuracy = compute_top_k_accuracy(logits=logits, targets=targets, top_k=1)
    top_2_accuracy = compute_top_k_accuracy(logits=logits, targets=targets, top_k=2)

    assert top_1_accuracy == pytest.approx(0.5)
    assert top_2_accuracy == pytest.approx(1.0)


def test_bootstrap_summary_metric_confidence_intervals() -> None:
    logits = tensor(
        [
            [4.0, 1.0, 0.0],
            [0.0, 4.0, 1.0],
            [0.0, 1.0, 4.0],
            [3.0, 2.0, 1.0],
        ]
    )
    targets = tensor([0, 1, 2, 1])

    report = bootstrap_summary_metric_confidence_intervals(
        logits=logits,
        targets=targets,
        num_classes=3,
        class_names=["a", "b", "c"],
        top_k=2,
        num_bootstrap_samples=20,
        confidence_level=0.95,
        seed=42,
    )

    assert set(report["metric"]) == {
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "top_k_accuracy",
    }
    assert set(report.columns) == {
        "metric",
        "mean",
        "std",
        "ci_lower",
        "ci_upper",
        "confidence_level",
        "num_bootstrap_samples",
    }
    assert (report["ci_lower"] <= report["ci_upper"]).all()
