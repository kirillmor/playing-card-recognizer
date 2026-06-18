from __future__ import annotations

from pathlib import Path

import hydra
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from omegaconf import DictConfig, OmegaConf
from rich.console import Console

from card_recognizer.data.datamodule import CardsDataModule
from card_recognizer.models.factory import create_model
from card_recognizer.models.lightning_module import CardClassifierModule

console = Console()


def build_callbacks(config: DictConfig, project_root: Path) -> list:
    """Build Lightning callbacks from config."""
    callbacks = []

    checkpoint_dir = project_root / config.paths.artifacts_dir / "checkpoints" / config.model.name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    callbacks.append(
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="{epoch:02d}-{val_macro_f1:.4f}",
            monitor=str(config.trainer.checkpoint.monitor),
            mode=str(config.trainer.checkpoint.mode),
            save_top_k=int(config.trainer.checkpoint.save_top_k),
            save_last=True,
        )
    )

    if bool(config.trainer.early_stopping.enabled):
        callbacks.append(
            EarlyStopping(
                monitor=str(config.trainer.early_stopping.monitor),
                mode=str(config.trainer.early_stopping.mode),
                patience=int(config.trainer.early_stopping.patience),
            )
        )

    return callbacks


def build_trainer(config: DictConfig, project_root: Path) -> L.Trainer:
    """Build Lightning Trainer from Hydra config."""
    return L.Trainer(
        max_epochs=int(config.trainer.max_epochs),
        accelerator=str(config.trainer.accelerator),
        devices=int(config.trainer.devices),
        precision=str(config.trainer.precision),
        log_every_n_steps=int(config.trainer.log_every_n_steps),
        check_val_every_n_epoch=int(config.trainer.check_val_every_n_epoch),
        limit_train_batches=config.trainer.limit_train_batches,
        limit_val_batches=config.trainer.limit_val_batches,
        limit_test_batches=config.trainer.limit_test_batches,
        deterministic=bool(config.trainer.deterministic),
        enable_progress_bar=bool(config.trainer.enable_progress_bar),
        callbacks=build_callbacks(config=config, project_root=project_root),
        default_root_dir=project_root / "outputs" / "lightning",
        logger=False,
    )


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(config: DictConfig) -> None:
    """Run model training."""
    project_root = Path(hydra.utils.get_original_cwd())

    console.rule("[bold green]Training configuration")
    console.print(OmegaConf.to_yaml(config, resolve=True))

    L.seed_everything(int(config.seed), workers=True)

    datamodule = CardsDataModule(
        data_config=config.data,
        project_root=project_root,
    )
    model = create_model(config.model)

    lightning_module = CardClassifierModule(
        model=model,
        optimizer_config=config.optimizer,
        trainer_config=config.trainer,
        num_classes=int(config.data.num_classes),
    )

    trainer = build_trainer(config=config, project_root=project_root)

    console.rule("[bold green]Starting training")
    trainer.fit(model=lightning_module, datamodule=datamodule)

    console.rule("[bold green]Running test evaluation")
    trainer.test(model=lightning_module, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()
