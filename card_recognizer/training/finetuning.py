from __future__ import annotations

from typing import Protocol

import lightning as L
from lightning.pytorch.callbacks import Callback


class UnfreezableBackbone(Protocol):
    """Protocol for models that support backbone unfreezing."""

    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone parameters."""


class BackboneUnfreezingCallback(Callback):
    """Unfreeze model backbone at the beginning of a configured epoch."""

    def __init__(self, unfreeze_at_epoch: int) -> None:
        super().__init__()

        if unfreeze_at_epoch < 0:
            raise ValueError("unfreeze_at_epoch must be non-negative.")

        self.unfreeze_at_epoch = unfreeze_at_epoch
        self.has_unfrozen = False

    def on_train_epoch_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        """Unfreeze backbone when the configured epoch starts."""
        if self.has_unfrozen:
            return

        if trainer.current_epoch < self.unfreeze_at_epoch:
            return

        model = getattr(pl_module, "model", None)

        if not hasattr(model, "unfreeze_backbone"):
            return

        model.unfreeze_backbone()
        self.has_unfrozen = True
