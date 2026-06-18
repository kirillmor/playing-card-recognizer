from __future__ import annotations

from omegaconf import OmegaConf
from torch import randint, randn

from card_recognizer.models.factory import create_model
from card_recognizer.models.lightning_module import CardClassifierModule


def test_lightning_module_forward_shape() -> None:
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
    optimizer_config = OmegaConf.create(
        {
            "name": "adam",
            "lr": 0.001,
            "weight_decay": 0.0,
            "scheduler": {"name": "none"},
        }
    )
    trainer_config = OmegaConf.create({"max_epochs": 1})

    model = create_model(model_config)
    module = CardClassifierModule(
        model=model,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
        num_classes=3,
    )

    images = randn(2, 3, 64, 64)
    labels = randint(low=0, high=3, size=(2,))

    logits = module(images)
    loss, shared_logits, targets = module._shared_step((images, labels))

    assert tuple(logits.shape) == (2, 3)
    assert tuple(shared_logits.shape) == (2, 3)
    assert tuple(targets.shape) == (2,)
    assert loss.ndim == 0
