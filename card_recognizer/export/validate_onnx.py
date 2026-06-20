from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import onnxruntime as ort
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from rich.console import Console

from card_recognizer.data.datamodule import CardsDataModule
from card_recognizer.export.export_onnx import load_model_from_checkpoint
from card_recognizer.inference.checkpoint import resolve_checkpoint_path

console = Console()


def _get_dataloader(data_module: CardsDataModule, split: str):
    if split == "train":
        return data_module.train_dataloader()
    if split in {"valid", "val", "validation"}:
        return data_module.val_dataloader()
    if split == "test":
        return data_module.test_dataloader()

    raise ValueError(f"Unsupported validation split: {split}")


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def validate_onnx(config: DictConfig) -> dict[str, Any]:
    """Compare PyTorch and ONNX Runtime predictions."""

    checkpoint_path = resolve_checkpoint_path(
        checkpoint_path=config.export.checkpoint_path,
        checkpoint_dir=config.export.checkpoint_dir,
        monitor=config.export.checkpoint_monitor,
        mode=config.export.checkpoint_mode,
    )

    onnx_path = Path(config.export.output_dir) / str(config.export.output_filename)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX file does not exist: {onnx_path}. Run export_onnx first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model_from_checkpoint(config, checkpoint_path)
    model.to(device)
    model.eval()

    providers = list(config.export.validation.providers)
    session = ort.InferenceSession(str(onnx_path), providers=providers)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    project_root = Path(get_original_cwd())
    data_module = CardsDataModule(config.data, project_root=project_root)
    data_module.setup("test")

    split = str(config.export.validation.split)
    dataloader = _get_dataloader(data_module, split)

    max_batches = config.export.validation.max_batches
    max_batches = None if max_batches is None else int(max_batches)

    total_samples = 0
    top1_agreements = 0
    top3_agreements = 0
    max_abs_diff = 0.0
    sum_abs_diff = 0.0
    num_logits = 0

    pytorch_correct = 0
    onnx_correct = 0

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images = images.to(device)
            targets = targets.to(device)

            pytorch_logits = model(images)
            onnx_logits_np = session.run(
                [output_name],
                {input_name: _to_numpy(images).astype(np.float32)},
            )[0]
            onnx_logits = torch.from_numpy(onnx_logits_np).to(device)

            abs_diff = torch.abs(pytorch_logits - onnx_logits)
            max_abs_diff = max(max_abs_diff, float(abs_diff.max().item()))
            sum_abs_diff += float(abs_diff.sum().item())
            num_logits += int(abs_diff.numel())

            pytorch_top1 = torch.argmax(pytorch_logits, dim=1)
            onnx_top1 = torch.argmax(onnx_logits, dim=1)

            pytorch_top3 = torch.topk(pytorch_logits, k=3, dim=1).indices
            onnx_top3 = torch.topk(onnx_logits, k=3, dim=1).indices

            top1_agreements += int((pytorch_top1 == onnx_top1).sum().item())
            top3_agreements += int((pytorch_top3 == onnx_top3).all(dim=1).sum().item())

            pytorch_correct += int((pytorch_top1 == targets).sum().item())
            onnx_correct += int((onnx_top1 == targets).sum().item())
            total_samples += int(targets.numel())

    mean_abs_diff = sum_abs_diff / max(num_logits, 1)

    report = {
        "model_name": str(config.model.name),
        "checkpoint_path": str(checkpoint_path),
        "onnx_path": str(onnx_path.resolve()),
        "split": split,
        "providers": providers,
        "num_samples": total_samples,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "top1_agreement": top1_agreements / max(total_samples, 1),
        "top3_agreement": top3_agreements / max(total_samples, 1),
        "pytorch_accuracy": pytorch_correct / max(total_samples, 1),
        "onnx_accuracy": onnx_correct / max(total_samples, 1),
        "atol": float(config.export.atol),
        "rtol": float(config.export.rtol),
    }

    report_dir = Path(config.export.validation.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    report_path = report_dir / str(config.export.validation.report_filename)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    console.print(json.dumps(report, indent=2))
    console.print(f"[green]Saved ONNX validation report:[/green] {report_path}")

    return report


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(config: DictConfig) -> None:
    console.rule("[bold]ONNX validation")
    console.print(OmegaConf.to_yaml(config))
    validate_onnx(config)


if __name__ == "__main__":
    main()
