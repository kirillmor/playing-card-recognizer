from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from rich.console import Console

console = Console()


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(config: DictConfig) -> None:
    """Run training pipeline.

    At this stage, the function only loads and prints the composed Hydra config.
    Real data loading, model creation, and training will be added in the next steps.
    """
    original_working_dir = hydra.utils.get_original_cwd()

    console.rule("[bold green]Playing Card Recognizer training config")
    console.print(f"[bold]Original working directory:[/bold] {original_working_dir}")
    console.print(f"[bold]Current working directory:[/bold] {Path.cwd()}")
    console.print()
    console.print(OmegaConf.to_yaml(config, resolve=True))


if __name__ == "__main__":
    main()
