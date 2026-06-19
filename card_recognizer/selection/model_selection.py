from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_model_evaluation(
    reports_root: Path,
    model_name: str,
    summary_filename: str,
    bootstrap_filename: str,
) -> dict[str, Any]:
    """Load evaluation summary and bootstrap intervals for one model."""
    model_report_dir = reports_root / model_name
    summary_path = model_report_dir / summary_filename
    bootstrap_path = model_report_dir / bootstrap_filename

    if not summary_path.is_file():
        raise FileNotFoundError(f"Evaluation summary does not exist: {summary_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    row: dict[str, Any] = {
        "model_name": model_name,
        "report_dir": str(model_report_dir),
    }

    for metric_name, metric_value in summary.items():
        row[metric_name] = metric_value

    if bootstrap_path.is_file():
        bootstrap_report = pd.read_csv(bootstrap_path)

        for metric_row in bootstrap_report.itertuples(index=False):
            metric_name = str(metric_row.metric)
            row[f"{metric_name}_bootstrap_mean"] = float(metric_row.mean)
            row[f"{metric_name}_bootstrap_std"] = float(metric_row.std)
            row[f"{metric_name}_ci_lower"] = float(metric_row.ci_lower)
            row[f"{metric_name}_ci_upper"] = float(metric_row.ci_upper)

    return row


def build_model_comparison_table(
    reports_root: Path,
    model_names: list[str],
    summary_filename: str,
    bootstrap_filename: str,
) -> pd.DataFrame:
    """Build comparison table for several evaluated models."""
    rows = [
        load_model_evaluation(
            reports_root=reports_root,
            model_name=model_name,
            summary_filename=summary_filename,
            bootstrap_filename=bootstrap_filename,
        )
        for model_name in model_names
    ]

    return pd.DataFrame(rows)


def select_best_model(
    comparison_table: pd.DataFrame,
    metric: str,
    higher_is_better: bool,
) -> dict[str, Any]:
    """Select the best model according to the configured metric."""
    if metric not in comparison_table.columns:
        available_columns = ", ".join(comparison_table.columns)
        raise ValueError(
            f"Metric '{metric}' is not available in comparison table. "
            f"Available columns: {available_columns}"
        )

    if comparison_table.empty:
        raise ValueError("Cannot select best model from an empty comparison table.")

    metric_values = pd.to_numeric(comparison_table[metric], errors="raise")
    best_index = metric_values.idxmax() if higher_is_better else metric_values.idxmin()
    best_row = comparison_table.loc[best_index].to_dict()

    return {
        "model_name": str(best_row["model_name"]),
        "selection_metric": metric,
        "selection_metric_value": float(best_row[metric]),
        "higher_is_better": higher_is_better,
        "report_dir": str(best_row["report_dir"]),
    }


def save_comparison_outputs(
    comparison_table: pd.DataFrame,
    best_model: dict[str, Any],
    output_dir: Path,
    comparison_filename: str,
    comparison_markdown_filename: str,
    best_model_filename: str,
) -> None:
    """Save comparison CSV, Markdown summary, and best model metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison_path = output_dir / comparison_filename
    markdown_path = output_dir / comparison_markdown_filename
    best_model_path = output_dir / best_model_filename

    comparison_table.to_csv(comparison_path, index=False)

    markdown_path.write_text(
        build_comparison_markdown(
            comparison_table=comparison_table,
            best_model=best_model,
        ),
        encoding="utf-8",
    )

    best_model_path.write_text(
        json.dumps(best_model, indent=2) + "\n",
        encoding="utf-8",
    )


def build_comparison_markdown(
    comparison_table: pd.DataFrame,
    best_model: dict[str, Any],
) -> str:
    """Build a lightweight Markdown comparison report without extra dependencies."""
    preferred_columns = [
        "model_name",
        "num_samples",
        "accuracy",
        "accuracy_ci_lower",
        "accuracy_ci_upper",
        "macro_f1",
        "macro_f1_ci_lower",
        "macro_f1_ci_upper",
        "weighted_f1",
        "top_k_accuracy",
    ]
    columns = [column for column in preferred_columns if column in comparison_table.columns]

    lines = [
        "# Model comparison",
        "",
        "## Selected model",
        "",
        f"- Model: `{best_model['model_name']}`",
        f"- Metric: `{best_model['selection_metric']}`",
        f"- Metric value: `{best_model['selection_metric_value']:.6f}`",
        "",
        "## Comparison table",
        "",
    ]

    lines.extend(_dataframe_to_markdown(comparison_table[columns]))
    lines.append("")

    return "\n".join(lines)


def _dataframe_to_markdown(dataframe: pd.DataFrame) -> list[str]:
    """Convert DataFrame to a simple Markdown table."""
    columns = list(dataframe.columns)

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]

    for _, row in dataframe.iterrows():
        values = [_format_markdown_value(row[column]) for column in columns]
        lines.append("| " + " | ".join(values) + " |")

    return lines


def _format_markdown_value(value: Any) -> str:
    """Format scalar values for Markdown tables."""
    if isinstance(value, float):
        return f"{value:.6f}"

    return str(value)
