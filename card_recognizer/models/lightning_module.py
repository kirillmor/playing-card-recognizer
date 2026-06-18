from __future__ import annotations

from typing import Any

import lightning as L
import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)


class CardClassifierModule(L.LightningModule):
    """LightningModule for multiclass playing card classification."""

    def __init__(
        self,
        model: nn.Module,
        optimizer_config: DictConfig,
        trainer_config: DictConfig,
        num_classes: int,
    ) -> None:
        super().__init__()

        self.model = model
        self.optimizer_config = optimizer_config
        self.trainer_config = trainer_config
        self.num_classes = num_classes
        self.loss_function = nn.CrossEntropyLoss()

        self.save_hyperparameters(
            {
                "optimizer": OmegaConf.to_container(optimizer_config, resolve=True),
                "trainer": OmegaConf.to_container(trainer_config, resolve=True),
                "num_classes": num_classes,
            }
        )

        metrics = MetricCollection(
            {
                "accuracy": MulticlassAccuracy(num_classes=num_classes, average="micro"),
                "macro_f1": MulticlassF1Score(num_classes=num_classes, average="macro"),
                "macro_precision": MulticlassPrecision(num_classes=num_classes, average="macro"),
                "macro_recall": MulticlassRecall(num_classes=num_classes, average="macro"),
                "top3_accuracy": MulticlassAccuracy(
                    num_classes=num_classes,
                    average="micro",
                    top_k=3,
                ),
            }
        )

        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    def forward(self, images: Tensor) -> Tensor:
        """Return class logits."""
        return self.model(images)

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Run one training step."""
        loss, logits, targets = self._shared_step(batch)
        batch_size = targets.size(0)

        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log_dict(
            self.train_metrics(logits, targets),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
        )

        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        """Run one validation step."""
        loss, logits, targets = self._shared_step(batch)
        batch_size = targets.size(0)

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log_dict(
            self.val_metrics(logits, targets),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )

    def test_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        """Run one test step."""
        loss, logits, targets = self._shared_step(batch)
        batch_size = targets.size(0)

        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log_dict(
            self.test_metrics(logits, targets),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )

    def configure_optimizers(self) -> Any:
        """Configure optimizer and optional scheduler."""
        optimizer_name = str(self.optimizer_config.name).lower()
        learning_rate = self._get_learning_rate()
        weight_decay = float(self.optimizer_config.get("weight_decay", 0.0))

        if optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )
        elif optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        scheduler_config = self.optimizer_config.get("scheduler")
        if scheduler_config is None or str(scheduler_config.name).lower() == "none":
            return optimizer

        if str(scheduler_config.name).lower() == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=int(self.trainer_config.max_epochs),
                eta_min=float(scheduler_config.min_lr),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                },
            }

        raise ValueError(f"Unsupported scheduler: {scheduler_config.name}")

    def _shared_step(self, batch: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        """Run forward pass and compute loss."""
        images, targets = batch
        logits = self(images)
        loss = self.loss_function(logits, targets)

        return loss, logits, targets

    def _get_learning_rate(self) -> float:
        """Get learning rate from optimizer config.

        Baseline configs use `lr`.
        Fine-tuning configs may use `head_lr` and `backbone_lr`.
        """
        if "lr" in self.optimizer_config:
            return float(self.optimizer_config.lr)

        if "head_lr" in self.optimizer_config:
            return float(self.optimizer_config.head_lr)

        raise ValueError("Optimizer config must contain either `lr` or `head_lr`.")
