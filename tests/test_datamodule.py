from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf
from PIL import Image

from card_recognizer.data.datamodule import CardsDataModule
from card_recognizer.data.transforms import build_eval_transforms, build_train_transforms


def create_tiny_imagefolder_dataset(root: Path) -> None:
    """Create a tiny ImageFolder-compatible dataset."""
    split_names = ["train", "valid", "test"]
    class_names = ["ace of clubs", "king of spades"]

    for split_name in split_names:
        for class_name in class_names:
            class_dir = root / split_name / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

            for image_index in range(2):
                image = Image.new(
                    mode="RGB",
                    size=(32, 32),
                    color=(image_index * 40, 10, 120),
                )
                image.save(class_dir / f"{image_index}.jpg")


def make_data_config(dataset_root: Path):
    """Create a minimal data config for tests."""
    return OmegaConf.create(
        {
            "num_classes": 2,
            "train_dir": str(dataset_root / "train"),
            "valid_dir": str(dataset_root / "valid"),
            "test_dir": str(dataset_root / "test"),
            "image_size": 64,
            "batch_size": 2,
            "num_workers": 0,
            "pin_memory": False,
            "normalization": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            "augmentation": {
                "random_horizontal_flip_p": 0.0,
                "random_rotation_degrees": 0.0,
                "color_jitter": {
                    "brightness": 0.0,
                    "contrast": 0.0,
                    "saturation": 0.0,
                    "hue": 0.0,
                },
            },
        }
    )


def test_train_and_eval_transforms_are_created(tmp_path: Path) -> None:
    data_config = make_data_config(tmp_path)

    train_transforms = build_train_transforms(data_config)
    eval_transforms = build_eval_transforms(data_config)

    assert train_transforms is not None
    assert eval_transforms is not None


def test_datamodule_loads_batches(tmp_path: Path) -> None:
    create_tiny_imagefolder_dataset(tmp_path)
    data_config = make_data_config(tmp_path)

    datamodule = CardsDataModule(
        data_config=data_config,
        project_root=Path("."),
    )
    datamodule.setup("fit")
    datamodule.setup("test")

    images, labels = next(iter(datamodule.train_dataloader()))

    assert tuple(images.shape) == (2, 3, 64, 64)
    assert tuple(labels.shape) == (2,)

    assert datamodule.train_dataset is not None
    assert datamodule.valid_dataset is not None
    assert datamodule.test_dataset is not None

    assert len(datamodule.train_dataset) == 4
    assert len(datamodule.valid_dataset) == 4
    assert len(datamodule.test_dataset) == 4
    assert datamodule.class_to_idx == {
        "ace of clubs": 0,
        "king of spades": 1,
    }
