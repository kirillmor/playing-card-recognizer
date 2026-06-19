from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import hydra
import mlflow
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from card_recognizer.data.datamodule import CardsDataModule
from card_recognizer.evaluation.metrics import (
    bootstrap_summary_metric_confidence_intervals,
    build_classification_report,
    compute_confusion_matrix,
    compute_top_k_accuracy,
    summarize_metrics,
)
from card_recognizer.evaluation.plots import (
    save_confusion_matrix_plot,
    save_worst_classes_plot,
)
from card_recognizer.models.factory import create_model
from card_recognizer.models.lightning_module import CardClassifierModule

console = Console()


def resolve_device(device_name: str) -> torch.device:
    """Resolve evaluation device."""
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torch.device(device_name)


def resolve_checkpoint_path(config: DictConfig, project_root: Path) -> Path:
    """Resolve checkpoint path from config or choose the latest available checkpoint."""
    checkpoint_path = config.evaluation.get("checkpoint_path")

    if checkpoint_path:
        candidate_path = Path(str(checkpoint_path))
        if not candidate_path.is_absolute():
            candidate_path = project_root / candidate_path

        if not candidate_path.is_file():
            raise FileNotFoundError(f"Checkpoint file does not exist: {candidate_path}")

        return candidate_path

    checkpoint_dir = project_root / config.paths.artifacts_dir / "checkpoints" / config.model.name

    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")

    checkpoints = sorted(
        [path for path in checkpoint_dir.glob("*.ckpt") if path.name != "last.ckpt"],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if checkpoints:
        return checkpoints[0]

    last_checkpoint = checkpoint_dir / "last.ckpt"
    if last_checkpoint.is_file():
        return last_checkpoint

    raise FileNotFoundError(f"No checkpoint files found in: {checkpoint_dir}")


def build_eval_dataloader(
    datamodule: CardsDataModule,
    split: str,
) -> tuple[DataLoader, ImageFolder]:
    """Build dataloader and return the corresponding dataset."""
    if split == "valid":
        datamodule.setup("validate")

        if datamodule.valid_dataset is None:
            raise RuntimeError("Validation dataset was not initialized.")

        return datamodule.val_dataloader(), datamodule.valid_dataset

    if split == "test":
        datamodule.setup("test")

        if datamodule.test_dataset is None:
            raise RuntimeError("Test dataset was not initialized.")

        return datamodule.test_dataloader(), datamodule.test_dataset

    raise ValueError(f"Unsupported evaluation split: {split}")


def collect_predictions(
    module: CardClassifierModule,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Collect logits and targets for a dataloader."""
    logits_batches: list[Tensor] = []
    target_batches: list[Tensor] = []

    module.eval()
    module.to(device)

    with torch.inference_mode():
        for images, targets in dataloader:
            images = images.to(device)
            logits = module(images)

            logits_batches.append(logits.cpu())
            target_batches.append(targets.cpu())

    return torch.cat(logits_batches), torch.cat(target_batches)


def build_predictions_table(
    dataset: ImageFolder,
    logits: Tensor,
    targets: Tensor,
    top_k: int,
) -> pd.DataFrame:
    """Build a predictions table with top-k labels and probabilities."""
    probabilities = logits.softmax(dim=1)
    top_k = min(top_k, probabilities.shape[1])
    top_probabilities, top_indices = probabilities.topk(k=top_k, dim=1)

    predicted_indices = logits.argmax(dim=1)
    class_names = dataset.classes

    rows: list[dict[str, Any]] = []

    for sample_index, (image_path, _) in enumerate(dataset.samples):
        true_index = int(targets[sample_index].item())
        predicted_index = int(predicted_indices[sample_index].item())

        row: dict[str, Any] = {
            "image_path": str(image_path),
            "true_class": class_names[true_index],
            "predicted_class": class_names[predicted_index],
            "confidence": float(probabilities[sample_index, predicted_index].item()),
            "correct": predicted_index == true_index,
        }

        for rank in range(top_k):
            class_index = int(top_indices[sample_index, rank].item())
            row[f"top_{rank + 1}_class"] = class_names[class_index]
            row[f"top_{rank + 1}_confidence"] = float(top_probabilities[sample_index, rank].item())

        rows.append(row)

    return pd.DataFrame(rows)


def log_evaluation_to_mlflow(
    config: DictConfig,
    summary_metrics: dict[str, Any],
    bootstrap_report: pd.DataFrame | None,
    report_dir: Path,
    plots_dir: Path,
    checkpoint_path: Path,
) -> None:
    """Log evaluation metrics and artifacts to MLflow."""
    if not bool(config.evaluation.log_to_mlflow):
        return

    mlflow.set_tracking_uri(str(config.logging.tracking_uri))
    mlflow.set_experiment(str(config.logging.experiment_name))

    with mlflow.start_run(run_name=str(config.evaluation.mlflow_run_name)):
        mlflow.log_param("model.name", str(config.model.name))
        mlflow.log_param("evaluation.split", str(config.evaluation.split))
        mlflow.log_param("checkpoint_path", str(checkpoint_path))

        for metric_name, metric_value in summary_metrics.items():
            if isinstance(metric_value, int | float):
                mlflow.log_metric(f"eval_{metric_name}", float(metric_value))

        if bootstrap_report is not None:
            for row in bootstrap_report.itertuples(index=False):
                metric_name = str(row.metric)
                mlflow.log_metric(f"eval_{metric_name}_ci_lower", float(row.ci_lower))
                mlflow.log_metric(f"eval_{metric_name}_ci_upper", float(row.ci_upper))
                mlflow.log_metric(f"eval_{metric_name}_bootstrap_mean", float(row.mean))
                mlflow.log_metric(f"eval_{metric_name}_bootstrap_std", float(row.std))

            mlflow.log_param(
                "evaluation.bootstrap.num_samples",
                int(config.evaluation.bootstrap.num_samples),
            )
            mlflow.log_param(
                "evaluation.bootstrap.confidence_level",
                float(config.evaluation.bootstrap.confidence_level),
            )

        mlflow.log_artifacts(str(report_dir), artifact_path="evaluation")

        if plots_dir.exists():
            mlflow.log_artifacts(str(plots_dir), artifact_path="plots")


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(config: DictConfig) -> None:
    """Run model evaluation and save reports."""
    project_root = Path(hydra.utils.get_original_cwd())

    console.rule("[bold green]Evaluation configuration")
    console.print(OmegaConf.to_yaml(config, resolve=True))

    checkpoint_path = resolve_checkpoint_path(config=config, project_root=project_root)
    device = resolve_device(str(config.evaluation.device))

    console.print(f"[bold]Checkpoint:[/bold] {checkpoint_path}")
    console.print(f"[bold]Device:[/bold] {device}")

    datamodule = CardsDataModule(
        data_config=config.data,
        project_root=project_root,
    )
    dataloader, dataset = build_eval_dataloader(
        datamodule=datamodule,
        split=str(config.evaluation.split),
    )

    model = create_model(config.model)
    module = CardClassifierModule.load_from_checkpoint(
        checkpoint_path,
        model=model,
        optimizer_config=config.optimizer,
        trainer_config=config.trainer,
        num_classes=int(config.data.num_classes),
    )

    logits, targets = collect_predictions(
        module=module,
        dataloader=dataloader,
        device=device,
    )
    predictions = logits.argmax(dim=1)

    confusion_matrix = compute_confusion_matrix(
        targets=targets,
        predictions=predictions,
        num_classes=int(config.data.num_classes),
    )
    classification_report = build_classification_report(
        confusion_matrix=confusion_matrix,
        class_names=dataset.classes,
    )
    top_k_accuracy = compute_top_k_accuracy(
        logits=logits,
        targets=targets,
        top_k=int(config.evaluation.top_k),
    )
    summary_metrics = summarize_metrics(
        confusion_matrix=confusion_matrix,
        classification_report=classification_report,
        top_k_accuracy=top_k_accuracy,
    )

    bootstrap_report = None
    if bool(config.evaluation.bootstrap.enabled):
        bootstrap_report = bootstrap_summary_metric_confidence_intervals(
            logits=logits,
            targets=targets,
            num_classes=int(config.data.num_classes),
            class_names=dataset.classes,
            top_k=int(config.evaluation.top_k),
            num_bootstrap_samples=int(config.evaluation.bootstrap.num_samples),
            confidence_level=float(config.evaluation.bootstrap.confidence_level),
            seed=int(config.evaluation.bootstrap.seed),
        )

    predictions_table = build_predictions_table(
        dataset=dataset,
        logits=logits,
        targets=targets,
        top_k=int(config.evaluation.top_k),
    )

    report_dir = project_root / str(config.evaluation.reports_dir)
    plots_dir = project_root / str(config.evaluation.plots_dir)

    report_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    summary_path = report_dir / "summary_metrics.json"
    report_path = report_dir / "classification_report.csv"
    predictions_path = report_dir / "predictions.csv"
    confusion_matrix_path = report_dir / "confusion_matrix.csv"
    bootstrap_path = report_dir / "bootstrap_confidence_intervals.csv"

    summary_path.write_text(
        json.dumps(summary_metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    classification_report.to_csv(report_path, index=False)
    predictions_table.to_csv(predictions_path, index=False)
    pd.DataFrame(confusion_matrix.numpy()).to_csv(confusion_matrix_path, index=False)

    if bootstrap_report is not None:
        bootstrap_report.to_csv(bootstrap_path, index=False)

    save_confusion_matrix_plot(
        confusion_matrix=confusion_matrix,
        output_path=plots_dir / "confusion_matrix.png",
        title=f"{config.model.name}: confusion matrix",
        normalize=False,
    )
    save_confusion_matrix_plot(
        confusion_matrix=confusion_matrix,
        output_path=plots_dir / "confusion_matrix_normalized.png",
        title=f"{config.model.name}: normalized confusion matrix",
        normalize=True,
    )
    save_worst_classes_plot(
        classification_report=classification_report,
        output_path=plots_dir / "worst_classes_by_f1.png",
    )

    log_evaluation_to_mlflow(
        config=config,
        summary_metrics=summary_metrics,
        bootstrap_report=bootstrap_report,
        report_dir=report_dir,
        plots_dir=plots_dir,
        checkpoint_path=checkpoint_path,
    )

    console.rule("[bold green]Evaluation summary")
    console.print(json.dumps(summary_metrics, indent=2))
    console.print(f"[bold]Reports:[/bold] {report_dir}")
    console.print(f"[bold]Plots:[/bold] {plots_dir}")


if __name__ == "__main__":
    main()
