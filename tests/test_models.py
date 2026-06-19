from __future__ import annotations

from omegaconf import OmegaConf
from torch import randn

from card_recognizer.models.factory import create_model


def test_baseline_cnn_forward_shape() -> None:
    model_config = OmegaConf.create(
        {
            "name": "baseline_cnn",
            "num_classes": 3,
            "architecture": {
                "input_channels": 3,
                "conv_channels": [8, 16],
                "kernel_size": 3,
                "dropout": 0.1,
            },
        }
    )

    model = create_model(model_config)
    images = randn(2, 3, 64, 64)

    logits = model(images)

    assert tuple(logits.shape) == (2, 3)


def test_factory_creates_efficientnet_without_pretrained_weights() -> None:
    model_config = OmegaConf.create(
        {
            "name": "efficientnet_b0",
            "num_classes": 3,
            "pretrained": False,
            "weights": "DEFAULT",
            "dropout": 0.2,
            "training_strategy": {
                "freeze_backbone_epochs": 0,
                "fine_tune_epochs": 1,
            },
        }
    )

    model = create_model(model_config)
    images = randn(2, 3, 224, 224)

    logits = model(images)

    assert tuple(logits.shape) == (2, 3)
