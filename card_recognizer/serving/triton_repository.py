"""Build a Triton Inference Server model repository."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from rich.console import Console

console = Console()


def build_config_pbtxt(
    *,
    model_name: str,
    platform: str,
    input_name: str,
    output_name: str,
    image_size: int,
    num_classes: int,
    max_batch_size: int,
    instance_kind: str,
) -> str:
    """Build Triton config.pbtxt for the card classifier."""

    return f"""name: "{model_name}"
platform: "{platform}"
max_batch_size: {max_batch_size}

input [
  {{
    name: "{input_name}"
    data_type: TYPE_FP32
    dims: [3, {image_size}, {image_size}]
  }}
]

output [
  {{
    name: "{output_name}"
    data_type: TYPE_FP32
    dims: [{num_classes}]
  }}
]

instance_group [
  {{
    kind: {instance_kind}
  }}
]
"""


def _resolve_path(path: str | Path, project_root: Path) -> Path:
    candidate_path = Path(path)
    if candidate_path.is_absolute():
        return candidate_path
    return project_root / candidate_path


def _copy_if_exists(source_path: Path, target_path: Path) -> bool:
    if not source_path.exists():
        return False
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)
    return True


def _build_single_model(
    *,
    model_config: DictConfig,
    serving_config: DictConfig,
    project_root: Path,
) -> dict[str, Any]:
    model_name = str(model_config.name)
    version_dir = (
        _resolve_path(serving_config.model_repository_dir, project_root)
        / model_name
        / str(model_config.version)
    )
    version_dir.mkdir(parents=True, exist_ok=True)

    model_root = version_dir.parent
    config_path = model_root / "config.pbtxt"
    config_path.write_text(
        build_config_pbtxt(
            model_name=model_name,
            platform=str(model_config.platform),
            input_name=str(serving_config.input_name),
            output_name=str(serving_config.output_name),
            image_size=int(serving_config.image_size),
            num_classes=int(serving_config.num_classes),
            max_batch_size=int(serving_config.max_batch_size),
            instance_kind=str(model_config.instance_kind),
        ),
        encoding="utf-8",
    )

    source_path = _resolve_path(model_config.source_path, project_root)
    target_path = version_dir / str(model_config.filename)
    copied_model = _copy_if_exists(source_path, target_path)

    return {
        "model_name": model_name,
        "platform": str(model_config.platform),
        "source_path": str(source_path),
        "target_path": str(target_path),
        "config_path": str(config_path),
        "copied_model": copied_model,
    }


def build_triton_repository(config: DictConfig) -> dict[str, Any]:
    """Create/update a Triton model repository from exported artifacts."""

    project_root = Path(get_original_cwd())
    serving_config = config.serving
    model_repository_dir = _resolve_path(serving_config.model_repository_dir, project_root)
    model_repository_dir.mkdir(parents=True, exist_ok=True)

    built_models: list[dict[str, Any]] = []
    if bool(serving_config.onnx_model.enabled):
        built_models.append(
            _build_single_model(
                model_config=serving_config.onnx_model,
                serving_config=serving_config,
                project_root=project_root,
            )
        )

    if bool(serving_config.tensorrt_model.enabled):
        built_models.append(
            _build_single_model(
                model_config=serving_config.tensorrt_model,
                serving_config=serving_config,
                project_root=project_root,
            )
        )

    labels_source_path = _resolve_path(serving_config.labels_path, project_root)
    labels_target_path = model_repository_dir / "class_to_idx.json"
    copied_labels = _copy_if_exists(labels_source_path, labels_target_path)

    report = {
        "model_repository_dir": str(model_repository_dir),
        "models": built_models,
        "labels_source_path": str(labels_source_path),
        "labels_target_path": str(labels_target_path),
        "copied_labels": copied_labels,
    }
    report_path = model_repository_dir / "repository_manifest.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    console.print(json.dumps(report, indent=2))
    console.print(f"[green]Triton model repository prepared:[/green] {model_repository_dir}")
    return report


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(config: DictConfig) -> None:
    console.rule("[bold]Triton model repository")
    console.print(OmegaConf.to_yaml(config))
    build_triton_repository(config)


if __name__ == "__main__":
    main()
