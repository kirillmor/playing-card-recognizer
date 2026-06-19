from __future__ import annotations

from omegaconf import OmegaConf
from torch import randn

from card_recognizer.models.efficientnet import EfficientNetB0Classifier
from card_recognizer.models.factory import create_model


def test_efficientnet_forward_shape_without_pretrained_weights() -> None:
    model = EfficientNetB0Classifier(
        num_classes=5,
        pretrained=False,
        dropout=0.2,
    )
    images = randn(2, 3, 224, 224)

    logits = model(images)

    assert tuple(logits.shape) == (2, 5)


def test_efficientnet_freeze_and_unfreeze_backbone() -> None:
    model = EfficientNetB0Classifier(
        num_classes=5,
        pretrained=False,
        dropout=0.2,
    )

    model.freeze_backbone()

    classifier_requires_grad = []
    backbone_requires_grad = []

    for name, parameter in model.model.named_parameters():
        if name.startswith("classifier."):
            classifier_requires_grad.append(parameter.requires_grad)
        else:
            backbone_requires_grad.append(parameter.requires_grad)

    assert all(classifier_requires_grad)
    assert not any(backbone_requires_grad)

    model.unfreeze_backbone()

    assert all(parameter.requires_grad for parameter in model.parameters())


def test_factory_creates_frozen_efficientnet() -> None:
    model_config = OmegaConf.create(
        {
            "name": "efficientnet_b0",
            "num_classes": 5,
            "pretrained": False,
            "weights": "DEFAULT",
            "dropout": 0.2,
            "training_strategy": {
                "freeze_backbone_epochs": 5,
                "fine_tune_epochs": 20,
            },
        }
    )

    model = create_model(model_config)

    trainable_parameter_names = [
        name for name, parameter in model.model.named_parameters() if parameter.requires_grad
    ]

    assert trainable_parameter_names
    assert all(name.startswith("classifier.") for name in trainable_parameter_names)
