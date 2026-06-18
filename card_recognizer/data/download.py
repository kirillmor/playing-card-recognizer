from __future__ import annotations

import subprocess
from pathlib import Path

import hydra
from omegaconf import DictConfig
from rich.console import Console

console = Console()


def run_command(command: list[str]) -> None:
    """Run a shell command and fail loudly if it exits with non-zero status."""
    console.print("[bold]Running command:[/bold]", " ".join(command))
    subprocess.run(command, check=True)


def download_from_kaggle(dataset_slug: str, output_dir: Path) -> None:
    """Download and unzip a Kaggle dataset into the given output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    run_command(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            dataset_slug,
            "-p",
            str(output_dir),
            "--unzip",
        ]
    )


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(config: DictConfig) -> None:
    """Download raw dataset.

    Kaggle credentials must be configured before running this command.
    The recommended options are either environment variables:

    KAGGLE_USERNAME=...
    KAGGLE_KEY=...

    or a local ~/.kaggle/kaggle.json file.
    """
    project_root = Path(hydra.utils.get_original_cwd())
    raw_dir = project_root / config.data.raw_dir

    console.rule("[bold green]Downloading dataset")
    console.print(f"[bold]Dataset:[/bold] {config.data.kaggle_dataset}")
    console.print(f"[bold]Output directory:[/bold] {raw_dir}")

    download_from_kaggle(
        dataset_slug=str(config.data.kaggle_dataset),
        output_dir=raw_dir,
    )

    console.print("[bold green]Dataset download finished.[/bold green]")


if __name__ == "__main__":
    main()
