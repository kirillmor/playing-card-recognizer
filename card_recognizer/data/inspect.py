from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig
from rich.console import Console
from rich.table import Table

from card_recognizer.data.datamodule import CardsDataModule

console = Console()


def make_dataset_table(datamodule: CardsDataModule) -> Table:
    """Create a table with dataset sizes."""
    table = Table(title="DataModule summary")
    table.add_column("Split", justify="left")
    table.add_column("Samples", justify="right")
    table.add_column("Classes", justify="right")

    if datamodule.train_dataset is not None:
        table.add_row(
            "train",
            str(len(datamodule.train_dataset)),
            str(len(datamodule.train_dataset.classes)),
        )

    if datamodule.valid_dataset is not None:
        table.add_row(
            "valid",
            str(len(datamodule.valid_dataset)),
            str(len(datamodule.valid_dataset.classes)),
        )

    if datamodule.test_dataset is not None:
        table.add_row(
            "test",
            str(len(datamodule.test_dataset)),
            str(len(datamodule.test_dataset.classes)),
        )

    return table


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(config: DictConfig) -> None:
    """Inspect the configured DataModule and print one batch shape."""
    project_root = Path(hydra.utils.get_original_cwd())

    datamodule = CardsDataModule(
        data_config=config.data,
        project_root=project_root,
    )
    datamodule.setup("fit")
    datamodule.setup("test")

    console.rule("[bold green]Inspecting DataModule")
    console.print(make_dataset_table(datamodule))

    images, labels = next(iter(datamodule.train_dataloader()))

    console.print(f"[bold]Image batch shape:[/bold] {tuple(images.shape)}")
    console.print(f"[bold]Label batch shape:[/bold] {tuple(labels.shape)}")
    console.print(f"[bold]Image dtype:[/bold] {images.dtype}")
    console.print(f"[bold]Label dtype:[/bold] {labels.dtype}")
    console.print(f"[bold]Number of classes:[/bold] {len(datamodule.class_to_idx)}")


if __name__ == "__main__":
    main()
