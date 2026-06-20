"""Simple Triton HTTP client for playing-card classification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from PIL import Image
from rich.console import Console

console = Console()


def _resolve_path(path: str | Path, project_root: Path) -> Path:
    candidate_path = Path(path)
    if candidate_path.is_absolute():
        return candidate_path
    return project_root / candidate_path


def load_idx_to_class(class_mapping_path: Path) -> dict[int, str]:
    """Load class index mapping saved by the data validation stage."""

    if not class_mapping_path.exists():
        return {}

    class_to_idx = json.loads(class_mapping_path.read_text(encoding="utf-8"))
    return {int(index): str(class_name) for class_name, index in class_to_idx.items()}


def preprocess_image(
    image_path: Path,
    *,
    image_size: int,
    mean: list[float],
    std: list[float],
) -> np.ndarray:
    """Preprocess an image into NCHW float32 array."""

    image = Image.open(image_path).convert("RGB")
    image = image.resize((image_size, image_size))
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    image_array = (image_array - np.asarray(mean, dtype=np.float32)) / np.asarray(
        std,
        dtype=np.float32,
    )
    image_array = np.transpose(image_array, (2, 0, 1))
    return np.expand_dims(image_array, axis=0).astype(np.float32)


def softmax(logits: np.ndarray) -> np.ndarray:
    """Compute stable softmax."""

    shifted_logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(shifted_logits)
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)


def build_prediction_response(
    probabilities: np.ndarray,
    *,
    idx_to_class: dict[int, str],
    top_k: int,
) -> dict[str, Any]:
    """Convert probabilities to structured response."""

    probabilities_1d = probabilities[0]
    top_indices = np.argsort(probabilities_1d)[::-1][:top_k]
    top_predictions = [
        {
            "class_index": int(class_index),
            "class_name": idx_to_class.get(int(class_index), str(int(class_index))),
            "confidence": float(probabilities_1d[class_index]),
        }
        for class_index in top_indices
    ]

    return {
        "predicted_class": top_predictions[0]["class_name"],
        "predicted_class_index": top_predictions[0]["class_index"],
        "confidence": top_predictions[0]["confidence"],
        "top_k": top_predictions,
    }


def run_triton_prediction(config: DictConfig) -> dict[str, Any]:
    """Run one image through Triton HTTP inference."""

    try:
        import tritonclient.http as httpclient
    except ImportError as error:
        raise ImportError("Install Triton HTTP client with: uv add 'tritonclient[http]'") from error

    project_root = Path(get_original_cwd())
    image_path_value = config.serving.client.image_path
    if image_path_value is None:
        raise ValueError("Set serving.client.image_path=/path/to/image.jpg")

    image_path = _resolve_path(image_path_value, project_root)
    class_mapping_path = _resolve_path(config.serving.labels_path, project_root)
    image_array = preprocess_image(
        image_path,
        image_size=int(config.serving.image_size),
        mean=[float(value) for value in config.data.normalization.mean],
        std=[float(value) for value in config.data.normalization.std],
    )

    client = httpclient.InferenceServerClient(url=str(config.serving.client.url))
    infer_input = httpclient.InferInput(
        str(config.serving.input_name),
        image_array.shape,
        "FP32",
    )
    infer_input.set_data_from_numpy(image_array)
    infer_output = httpclient.InferRequestedOutput(str(config.serving.output_name))

    result = client.infer(
        model_name=str(config.serving.client.model_name),
        inputs=[infer_input],
        outputs=[infer_output],
    )

    logits = result.as_numpy(str(config.serving.output_name))
    probabilities = softmax(logits)
    response = build_prediction_response(
        probabilities,
        idx_to_class=load_idx_to_class(class_mapping_path),
        top_k=int(config.serving.client.top_k),
    )
    response["model_name"] = str(config.serving.client.model_name)
    response["image_path"] = str(image_path)

    console.print(json.dumps(response, indent=2))
    return response


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(config: DictConfig) -> None:
    console.rule("[bold]Triton inference client")
    console.print(f"[bold]Triton URL:[/bold] {config.serving.client.url}")
    console.print(f"[bold]Triton model:[/bold] {config.serving.client.model_name}")
    console.print(f"[bold]Image:[/bold] {config.serving.client.image_path}")
    run_triton_prediction(config)


if __name__ == "__main__":
    main()
