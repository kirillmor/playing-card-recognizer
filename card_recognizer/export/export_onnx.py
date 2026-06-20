from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from rich.console import Console

from card_recognizer.inference.checkpoint import resolve_checkpoint_path
from card_recognizer.models.factory import create_model

console = Console()


def _strip_lightning_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert LightningModule state_dict keys to raw model keys."""

    model_state_dict: dict[str, torch.Tensor] = {}

    for key, value in state_dict.items():
        if key.startswith("model."):
            model_state_dict[key.removeprefix("model.")] = value

    if model_state_dict:
        return model_state_dict

    return state_dict


def load_model_from_checkpoint(config: DictConfig, checkpoint_path: Path) -> torch.nn.Module:
    """Build the configured model and load weights from a Lightning checkpoint."""

    model = create_model(config.model)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model_state_dict = _strip_lightning_module_prefix(state_dict)

    missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)

    if missing_keys:
        console.print(f"[yellow]Missing keys while loading checkpoint: {missing_keys}[/yellow]")

    if unexpected_keys:
        console.print(
            f"[yellow]Unexpected keys while loading checkpoint: {unexpected_keys}[/yellow]"
        )

    return model


def export_to_onnx(config: DictConfig) -> dict[str, Any]:
    """Export model to ONNX and return export metadata."""

    checkpoint_path = resolve_checkpoint_path(
        checkpoint_path=config.export.checkpoint_path,
        checkpoint_dir=config.export.checkpoint_dir,
        monitor=config.export.checkpoint_monitor,
        mode=config.export.checkpoint_mode,
    )

    output_dir = Path(config.export.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / str(config.export.output_filename)

    image_size = int(config.data.image_size)
    batch_size = int(config.export.batch_size)

    model = load_model_from_checkpoint(config, checkpoint_path)
    model.eval()

    dummy_input = torch.randn(batch_size, 3, image_size, image_size)

    dynamic_axes = None
    if bool(config.export.dynamic_batch):
        dynamic_axes = {
            str(config.export.input_name): {0: "batch_size"},
            str(config.export.output_name): {0: "batch_size"},
        }

    console.print(f"[bold]Checkpoint:[/bold] {checkpoint_path}")
    console.print(f"[bold]ONNX output:[/bold] {output_path}")

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=int(config.export.opset_version),
            do_constant_folding=True,
            input_names=[str(config.export.input_name)],
            output_names=[str(config.export.output_name)],
            dynamic_axes=dynamic_axes,
            dynamo=bool(config.export.dynamo),
        )

    metadata = {
        "model_name": str(config.model.name),
        "checkpoint_path": str(checkpoint_path),
        "onnx_path": str(output_path.resolve()),
        "opset_version": int(config.export.opset_version),
        "input_name": str(config.export.input_name),
        "output_name": str(config.export.output_name),
        "image_size": image_size,
        "dynamic_batch": bool(config.export.dynamic_batch),
    }

    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    console.print(f"[green]Exported ONNX model:[/green] {output_path}")
    console.print(f"[green]Saved metadata:[/green] {metadata_path}")

    return metadata


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(config: DictConfig) -> None:
    console.rule("[bold]ONNX export")
    console.print(OmegaConf.to_yaml(config))
    export_to_onnx(config)


if __name__ == "__main__":
    main()
