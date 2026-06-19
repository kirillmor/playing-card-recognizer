from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import Tensor


def save_confusion_matrix_plot(
    confusion_matrix: Tensor,
    output_path: Path,
    title: str,
    normalize: bool,
) -> None:
    """Save confusion matrix plot."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    matrix = confusion_matrix.to(dtype=torch.float32)

    if normalize:
        row_sums = matrix.sum(dim=1, keepdim=True)
        matrix = torch.where(row_sums > 0, matrix / row_sums, torch.zeros_like(matrix))

    plt.figure(figsize=(10, 8))
    plt.imshow(matrix.numpy(), interpolation="nearest")
    plt.title(title)
    plt.xlabel("Predicted class index")
    plt.ylabel("True class index")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_worst_classes_plot(
    classification_report: pd.DataFrame,
    output_path: Path,
    top_n: int = 15,
) -> None:
    """Save a horizontal bar chart with lowest per-class F1 scores."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    worst_classes = classification_report.nsmallest(top_n, "f1_score")

    plt.figure(figsize=(10, 7))
    plt.barh(worst_classes["class_name"], worst_classes["f1_score"])
    plt.title(f"Lowest {len(worst_classes)} classes by F1 score")
    plt.xlabel("F1 score")
    plt.ylabel("Class")
    plt.xlim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
