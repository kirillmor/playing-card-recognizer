from __future__ import annotations

from pathlib import Path
from typing import Literal

import lightning as L
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from card_recognizer.data.transforms import build_eval_transforms, build_train_transforms

Stage = Literal["fit", "validate", "test", "predict"] | None


class CardsDataModule(L.LightningDataModule):
    """PyTorch Lightning DataModule for playing card image classification."""

    def __init__(self, data_config: DictConfig, project_root: Path) -> None:
        super().__init__()
        self.data_config = data_config
        self.project_root = project_root

        self.train_dataset: ImageFolder | None = None
        self.valid_dataset: ImageFolder | None = None
        self.test_dataset: ImageFolder | None = None

    def setup(self, stage: Stage = None) -> None:
        """Create datasets for the requested stage."""
        if stage in ("fit", None):
            self.train_dataset = ImageFolder(
                root=self._resolve_path(str(self.data_config.train_dir)),
                transform=build_train_transforms(self.data_config),
            )
            self.valid_dataset = ImageFolder(
                root=self._resolve_path(str(self.data_config.valid_dir)),
                transform=build_eval_transforms(self.data_config),
            )

        if stage in ("validate", None):
            self.valid_dataset = ImageFolder(
                root=self._resolve_path(str(self.data_config.valid_dir)),
                transform=build_eval_transforms(self.data_config),
            )

        if stage in ("test", None):
            self.test_dataset = ImageFolder(
                root=self._resolve_path(str(self.data_config.test_dir)),
                transform=build_eval_transforms(self.data_config),
            )

    def train_dataloader(self) -> DataLoader:
        """Return train dataloader."""
        if self.train_dataset is None:
            raise RuntimeError("Train dataset is not initialized. Call setup('fit') first.")

        return DataLoader(
            self.train_dataset,
            batch_size=int(self.data_config.batch_size),
            shuffle=True,
            num_workers=int(self.data_config.num_workers),
            pin_memory=bool(self.data_config.pin_memory),
        )

    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader."""
        if self.valid_dataset is None:
            raise RuntimeError("Validation dataset is not initialized. Call setup('fit') first.")

        return DataLoader(
            self.valid_dataset,
            batch_size=int(self.data_config.batch_size),
            shuffle=False,
            num_workers=int(self.data_config.num_workers),
            pin_memory=bool(self.data_config.pin_memory),
        )

    def test_dataloader(self) -> DataLoader:
        """Return test dataloader."""
        if self.test_dataset is None:
            raise RuntimeError("Test dataset is not initialized. Call setup('test') first.")

        return DataLoader(
            self.test_dataset,
            batch_size=int(self.data_config.batch_size),
            shuffle=False,
            num_workers=int(self.data_config.num_workers),
            pin_memory=bool(self.data_config.pin_memory),
        )

    @property
    def class_to_idx(self) -> dict[str, int]:
        """Return class-to-index mapping from the train dataset."""
        if self.train_dataset is None:
            raise RuntimeError("Train dataset is not initialized. Call setup('fit') first.")

        return dict(self.train_dataset.class_to_idx)

    def _resolve_path(self, path: str) -> Path:
        """Resolve a path relative to the original project root unless it is absolute."""
        candidate_path = Path(path)

        if candidate_path.is_absolute():
            return candidate_path

        return self.project_root / candidate_path
