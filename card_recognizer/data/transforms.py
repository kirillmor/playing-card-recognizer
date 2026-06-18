from __future__ import annotations

from collections.abc import Sequence

from omegaconf import DictConfig
from torchvision import transforms


def _to_float_list(values: Sequence[float]) -> list[float]:
    """Convert a sequence of numeric values to a plain list of floats."""
    return [float(value) for value in values]


def build_train_transforms(data_config: DictConfig) -> transforms.Compose:
    """Build preprocessing and augmentation transforms for the training split."""
    image_size = int(data_config.image_size)

    transform_steps = [
        transforms.Resize((image_size, image_size)),
    ]

    random_rotation_degrees = float(data_config.augmentation.random_rotation_degrees)
    if random_rotation_degrees > 0:
        transform_steps.append(transforms.RandomRotation(degrees=random_rotation_degrees))

    random_horizontal_flip_p = float(data_config.augmentation.random_horizontal_flip_p)
    if random_horizontal_flip_p > 0:
        transform_steps.append(transforms.RandomHorizontalFlip(p=random_horizontal_flip_p))

    color_jitter_config = data_config.augmentation.color_jitter
    transform_steps.append(
        transforms.ColorJitter(
            brightness=float(color_jitter_config.brightness),
            contrast=float(color_jitter_config.contrast),
            saturation=float(color_jitter_config.saturation),
            hue=float(color_jitter_config.hue),
        )
    )

    transform_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=_to_float_list(data_config.normalization.mean),
                std=_to_float_list(data_config.normalization.std),
            ),
        ]
    )

    return transforms.Compose(transform_steps)


def build_eval_transforms(data_config: DictConfig) -> transforms.Compose:
    """Build deterministic preprocessing transforms for validation, test, and inference."""
    image_size = int(data_config.image_size)

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=_to_float_list(data_config.normalization.mean),
                std=_to_float_list(data_config.normalization.std),
            ),
        ]
    )
