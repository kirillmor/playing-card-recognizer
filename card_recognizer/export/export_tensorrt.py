from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf
from rich.console import Console

console = Console()


def _shape_argument(input_name: str, batch_size: int, channels: int, image_size: int) -> str:
    return f"{input_name}:{batch_size}x{channels}x{image_size}x{image_size}"


def build_trtexec_command(config: DictConfig) -> list[str]:
    """Build a trtexec command from Hydra config."""

    onnx_path = Path(config.export.onnx_path)
    engine_path = Path(config.export.output_dir) / str(config.export.engine_filename)
    input_name = str(config.export.input_name)
    input_channels = int(config.export.input_channels)
    image_size = int(config.export.image_size)

    min_shape = _shape_argument(
        input_name,
        int(config.export.min_batch_size),
        input_channels,
        image_size,
    )
    opt_shape = _shape_argument(
        input_name,
        int(config.export.opt_batch_size),
        input_channels,
        image_size,
    )
    max_shape = _shape_argument(
        input_name,
        int(config.export.max_batch_size),
        input_channels,
        image_size,
    )

    command = [
        str(config.export.trtexec_path),
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--minShapes={min_shape}",
        f"--optShapes={opt_shape}",
        f"--maxShapes={max_shape}",
    ]

    precision = str(config.export.precision).lower()
    if precision == "fp16":
        command.append("--fp16")
    elif precision != "fp32":
        raise ValueError(f"Unsupported TensorRT precision: {precision}")

    if bool(config.export.verbose):
        command.append("--verbose")

    return command


def export_tensorrt_engine(config: DictConfig) -> dict[str, Any]:
    """Run trtexec and save export metadata."""

    trtexec_path = str(config.export.trtexec_path)
    if shutil.which(trtexec_path) is None:
        raise FileNotFoundError(
            f"Could not find '{trtexec_path}'. Install TensorRT or run inside an "
            "NVIDIA TensorRT/Triton container that provides trtexec."
        )

    onnx_path = Path(config.export.onnx_path)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX file does not exist: {onnx_path}. Run ONNX export first.")

    output_dir = Path(config.export.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    engine_path = output_dir / str(config.export.engine_filename)
    log_path = output_dir / str(config.export.log_filename)
    report_path = output_dir / str(config.export.report_filename)

    command = build_trtexec_command(config)
    console.print("[bold]Running TensorRT export command:[/bold]")
    console.print(" ".join(command))

    completed_process = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    log_path.write_text(completed_process.stdout, encoding="utf-8")
    report = {
        "model_name": str(config.model.name),
        "onnx_path": str(onnx_path.resolve()),
        "engine_path": str(engine_path.resolve()),
        "precision": str(config.export.precision),
        "min_batch_size": int(config.export.min_batch_size),
        "opt_batch_size": int(config.export.opt_batch_size),
        "max_batch_size": int(config.export.max_batch_size),
        "input_name": str(config.export.input_name),
        "output_name": str(config.export.output_name),
        "image_size": int(config.export.image_size),
        "command": command,
        "returncode": int(completed_process.returncode),
        "engine_exists": engine_path.exists(),
        "engine_size_bytes": engine_path.stat().st_size if engine_path.exists() else 0,
        "log_path": str(log_path.resolve()),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if completed_process.returncode != 0 or not engine_path.exists():
        console.print(f"[red]TensorRT export failed. See log:[/red] {log_path}")
        raise RuntimeError(
            f"TensorRT export failed with return code {completed_process.returncode}"
        )

    console.print(f"[green]TensorRT engine saved:[/green] {engine_path}")
    console.print(f"[green]Export report saved:[/green] {report_path}")
    return report


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(config: DictConfig) -> None:
    console.rule("[bold]TensorRT export")
    console.print(OmegaConf.to_yaml(config))
    export_tensorrt_engine(config)


if __name__ == "__main__":
    main()
