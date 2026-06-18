from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import lightning as L
import matplotlib.pyplot as plt
import torch
from lightning.pytorch.callbacks import Callback


class MetricsPlotCallback(Callback):
    """Collect epoch metrics and save training plots."""

    def __init__(self, plots_dir: Path, model_name: str) -> None:
        super().__init__()
        self.plots_dir = plots_dir
        self.model_name = model_name
        self.history: dict[str, list[float]] = defaultdict(list)
        self.epochs: list[int] = []

    def on_validation_epoch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        """Collect metrics after each validation epoch."""
        if trainer.sanity_checking:
            return

        epoch = int(trainer.current_epoch)
        self.epochs.append(epoch)

        for metric_name, metric_value in trainer.callback_metrics.items():
            if not isinstance(metric_value, torch.Tensor):
                continue

            if metric_value.ndim != 0:
                continue

            self.history[metric_name].append(float(metric_value.detach().cpu().item()))

    def on_fit_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        """Save metrics history and plots at the end of fit."""
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        self._save_history()
        self._plot_metric_pair(
            train_metric="train_loss",
            val_metric="val_loss",
            title="Loss",
            ylabel="Loss",
            filename="loss.png",
        )
        self._plot_metric_pair(
            train_metric="train_accuracy",
            val_metric="val_accuracy",
            title="Accuracy",
            ylabel="Accuracy",
            filename="accuracy.png",
        )
        self._plot_metric_pair(
            train_metric="train_macro_f1",
            val_metric="val_macro_f1",
            title="Macro F1",
            ylabel="Macro F1",
            filename="macro_f1.png",
        )
        self._plot_metric_pair(
            train_metric="train_top3_accuracy",
            val_metric="val_top3_accuracy",
            title="Top-3 Accuracy",
            ylabel="Top-3 Accuracy",
            filename="top3_accuracy.png",
        )

    def _save_history(self) -> None:
        """Save collected metrics as JSON."""
        history_path = self.plots_dir / "metrics_history.json"
        serializable_history = {
            "epochs": self.epochs,
            "metrics": dict(self.history),
        }
        history_path.write_text(
            json.dumps(serializable_history, indent=2) + "\n",
            encoding="utf-8",
        )

    def _plot_metric_pair(
        self,
        train_metric: str,
        val_metric: str,
        title: str,
        ylabel: str,
        filename: str,
    ) -> None:
        """Plot train/validation metric pair if at least one exists."""
        has_train = train_metric in self.history
        has_val = val_metric in self.history

        if not has_train and not has_val:
            return

        plt.figure(figsize=(8, 5))

        if has_train:
            train_values = self.history[train_metric]
            plt.plot(
                self.epochs[: len(train_values)],
                train_values,
                label=train_metric,
            )

        if has_val:
            val_values = self.history[val_metric]
            plt.plot(
                self.epochs[: len(val_values)],
                val_values,
                label=val_metric,
            )

        plt.title(f"{self.model_name}: {title}")
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.plots_dir / filename, dpi=150)
        plt.close()
