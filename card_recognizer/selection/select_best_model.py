from __future__ import annotations

import json
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from rich.console import Console

from card_recognizer.selection.model_selection import (
    build_model_comparison_table,
    save_comparison_outputs,
    select_best_model,
)

console = Console()


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(config: DictConfig) -> None:
    """Compare evaluated models and select the best one."""
    project_root = Path(hydra.utils.get_original_cwd())

    console.rule("[bold green]Model selection configuration")
    console.print(OmegaConf.to_yaml(config.selection, resolve=True))

    reports_root = project_root / str(config.selection.reports_root)
    output_dir = project_root / str(config.selection.output_dir)

    model_names = [str(model_name) for model_name in config.selection.models]

    comparison_table = build_model_comparison_table(
        reports_root=reports_root,
        model_names=model_names,
        summary_filename=str(config.selection.summary_filename),
        bootstrap_filename=str(config.selection.bootstrap_filename),
    )
    best_model = select_best_model(
        comparison_table=comparison_table,
        metric=str(config.selection.metric),
        higher_is_better=bool(config.selection.higher_is_better),
    )

    save_comparison_outputs(
        comparison_table=comparison_table,
        best_model=best_model,
        output_dir=output_dir,
        comparison_filename=str(config.selection.comparison_filename),
        comparison_markdown_filename=str(config.selection.comparison_markdown_filename),
        best_model_filename=str(config.selection.best_model_filename),
    )

    console.rule("[bold green]Best model")
    console.print(json.dumps(best_model, indent=2))
    console.print(f"[bold]Comparison outputs:[/bold] {output_dir}")


if __name__ == "__main__":
    main()
