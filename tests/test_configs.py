from __future__ import annotations

from pathlib import Path


def test_base_config_exists() -> None:
    assert Path("configs/config.yaml").is_file()


def test_config_groups_exist() -> None:
    expected_config_files = [
        Path("configs/data/cards.yaml"),
        Path("configs/model/baseline_cnn.yaml"),
        Path("configs/model/efficientnet_b0.yaml"),
        Path("configs/optimizer/adam.yaml"),
        Path("configs/optimizer/adamw.yaml"),
        Path("configs/trainer/cpu.yaml"),
        Path("configs/trainer/gpu.yaml"),
        Path("configs/logging/mlflow.yaml"),
        Path("configs/inference/local.yaml"),
    ]

    for config_file in expected_config_files:
        assert config_file.is_file(), f"Missing config file: {config_file}"
