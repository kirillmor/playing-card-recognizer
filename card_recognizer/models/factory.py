from __future__ import annotations

from omegaconf import DictConfig
from torch import nn

from card_recognizer.models.baseline_cnn import BaselineCNN
from card_recognizer.models.efficientnet import EfficientNetB0Classifier


def create_model(model_config: DictConfig) -> nn.Module:
    """Create a model from Hydra model config."""
    model_name = str(model_config.name)

    if model_name == "baseline_cnn":
        architecture = model_config.architecture

        return BaselineCNN(
            input_channels=int(architecture.input_channels),
            conv_channels=[int(channel) for channel in architecture.conv_channels],
            kernel_size=int(architecture.kernel_size),
            dropout=float(architecture.dropout),
            num_classes=int(model_config.num_classes),
        )

    if model_name == "efficientnet_b0":
        model = EfficientNetB0Classifier(
            num_classes=int(model_config.num_classes),
            pretrained=bool(model_config.pretrained),
            dropout=float(model_config.dropout),
        )

        freeze_backbone_epochs = int(model_config.training_strategy.freeze_backbone_epochs)
        if freeze_backbone_epochs > 0:
            model.freeze_backbone()

        return model

    raise ValueError(f"Unknown model name: {model_name}")
