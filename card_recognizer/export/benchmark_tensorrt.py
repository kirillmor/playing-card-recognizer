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


def build_benchmark_command(config: DictConfig) -> list[str]:
    """Build a trtexec benchmark command for an existing TensorRT engine."""

    engine_path = Path(config.export.output_dir) / str(config.export.engine_filename)
    input_shape = (
        f"{config.export.input_name}:{int(config.export.opt_batch_size)}x"
        f"{int(config.export.input_channels)}x{int(config.export.image_size)}x"
        f"{int(config.export.image_size)}"
    )
    return [
        str(config.export.trtexec_path),
        f"--loadEngine={engine_path}",
        f"--shapes={input_shape}",
        f"--warmUp={int(config.export.benchmark.warmup_ms)}",
        f"--duration={int(config.export.benchmark.duration_s)}",
    ]


def benchmark_tensorrt_engine(config: DictConfig) -> dict[str, Any]:
    """Run trtexec benchmark and save log/report."""

    trtexec_path = str(config.export.trtexec_path)
    if shutil.which(trtexec_path) is None:
        raise FileNotFoundError(f"Could not find '{trtexec_path}'.")

    output_dir = Path(config.export.output_dir)
    engine_path = output_dir / str(config.export.engine_filename)
    if not engine_path.exists():
        raise FileNotFoundError(f"TensorRT engine does not exist: {engine_path}")

    log_path = output_dir / str(config.export.benchmark.log_filename)
    report_path = output_dir / str(config.export.benchmark.report_filename)
    command = build_benchmark_command(config)

    console.print("[bold]Running TensorRT benchmark command:[/bold]")
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
        "engine_path": str(engine_path.resolve()),
        "command": command,
        "returncode": int(completed_process.returncode),
        "log_path": str(log_path.resolve()),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if completed_process.returncode != 0:
        raise RuntimeError(
            f"TensorRT benchmark failed with return code {completed_process.returncode}"
        )

    console.print(f"[green]TensorRT benchmark report saved:[/green] {report_path}")
    return report


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(config: DictConfig) -> None:
    console.rule("[bold]TensorRT benchmark")
    console.print(OmegaConf.to_yaml(config))
    benchmark_tensorrt_engine(config)


if __name__ == "__main__":
    main()
