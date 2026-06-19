from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from card_recognizer.selection.model_selection import (
    build_model_comparison_table,
    select_best_model,
)


def _write_evaluation_report(
    reports_root: Path,
    model_name: str,
    accuracy: float,
    macro_f1: float,
) -> None:
    model_dir = reports_root / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "num_samples": 10,
        "accuracy": accuracy,
        "macro_precision": macro_f1,
        "macro_recall": macro_f1,
        "macro_f1": macro_f1,
        "weighted_f1": macro_f1,
        "top_k_accuracy": 1.0,
    }
    (model_dir / "summary_metrics.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )

    bootstrap_report = pd.DataFrame(
        [
            {
                "metric": "macro_f1",
                "mean": macro_f1,
                "std": 0.01,
                "ci_lower": macro_f1 - 0.02,
                "ci_upper": macro_f1 + 0.02,
                "confidence_level": 0.95,
                "num_bootstrap_samples": 20,
            }
        ]
    )
    bootstrap_report.to_csv(model_dir / "bootstrap_confidence_intervals.csv", index=False)


def test_build_model_comparison_table_and_select_best_model(tmp_path: Path) -> None:
    reports_root = tmp_path / "evaluation"

    _write_evaluation_report(
        reports_root=reports_root,
        model_name="baseline_cnn",
        accuracy=0.7,
        macro_f1=0.65,
    )
    _write_evaluation_report(
        reports_root=reports_root,
        model_name="efficientnet_b0",
        accuracy=0.9,
        macro_f1=0.88,
    )

    comparison_table = build_model_comparison_table(
        reports_root=reports_root,
        model_names=["baseline_cnn", "efficientnet_b0"],
        summary_filename="summary_metrics.json",
        bootstrap_filename="bootstrap_confidence_intervals.csv",
    )
    best_model = select_best_model(
        comparison_table=comparison_table,
        metric="macro_f1",
        higher_is_better=True,
    )

    assert list(comparison_table["model_name"]) == ["baseline_cnn", "efficientnet_b0"]
    assert best_model["model_name"] == "efficientnet_b0"
    assert best_model["selection_metric_value"] == pytest.approx(0.88)
    assert "macro_f1_ci_lower" in comparison_table.columns
    assert "macro_f1_ci_upper" in comparison_table.columns


def test_select_best_model_raises_for_missing_metric() -> None:
    comparison_table = pd.DataFrame(
        [
            {
                "model_name": "baseline_cnn",
                "accuracy": 0.7,
            }
        ]
    )

    with pytest.raises(ValueError, match="Metric 'macro_f1' is not available"):
        select_best_model(
            comparison_table=comparison_table,
            metric="macro_f1",
            higher_is_better=True,
        )
