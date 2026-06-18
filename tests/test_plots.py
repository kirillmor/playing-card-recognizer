from __future__ import annotations

from pathlib import Path

from card_recognizer.training.plots import MetricsPlotCallback


def test_metrics_plot_callback_saves_history_and_plots(tmp_path: Path) -> None:
    callback = MetricsPlotCallback(
        plots_dir=tmp_path,
        model_name="baseline_cnn",
    )
    callback.epochs.extend([0, 1])
    callback.history["train_loss"].extend([4.0, 3.0])
    callback.history["val_loss"].extend([4.2, 3.5])
    callback.history["train_accuracy"].extend([0.1, 0.2])
    callback.history["val_accuracy"].extend([0.08, 0.18])
    callback.history["train_macro_f1"].extend([0.05, 0.12])
    callback.history["val_macro_f1"].extend([0.04, 0.10])

    callback._save_history()
    callback._plot_metric_pair(
        train_metric="train_loss",
        val_metric="val_loss",
        title="Loss",
        ylabel="Loss",
        filename="loss.png",
    )

    assert (tmp_path / "metrics_history.json").is_file()
    assert (tmp_path / "loss.png").is_file()
