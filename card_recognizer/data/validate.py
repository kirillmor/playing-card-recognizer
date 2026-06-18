from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import hydra
from omegaconf import DictConfig
from PIL import Image
from rich.console import Console
from rich.table import Table

console = Console()


@dataclass(frozen=True)
class SplitSummary:
    """Summary for one dataset split."""

    name: str
    num_classes: int
    num_images: int


def is_image_file(path: Path, allowed_extensions: set[str]) -> bool:
    """Return True if path looks like an image file by extension."""
    return path.is_file() and path.suffix.lower() in allowed_extensions


def resolve_dataset_root(raw_dir: Path) -> Path:
    """Resolve actual dataset root.

    Usually Kaggle extracts train/valid/test directly into raw_dir.
    If the archive contains an extra nested directory, this function detects it.
    """
    expected_splits = {"train", "valid", "test"}

    if expected_splits.issubset({path.name for path in raw_dir.iterdir() if path.is_dir()}):
        return raw_dir

    nested_candidates = [
        child
        for child in raw_dir.iterdir()
        if child.is_dir()
        and expected_splits.issubset({path.name for path in child.iterdir() if path.is_dir()})
    ]

    if len(nested_candidates) == 1:
        return nested_candidates[0]

    raise FileNotFoundError(
        f"Could not find dataset root with train/valid/test directories inside {raw_dir}"
    )


def list_class_names(split_dir: Path) -> list[str]:
    """List class directory names for a dataset split."""
    return sorted(path.name for path in split_dir.iterdir() if path.is_dir())


def count_images(split_dir: Path, allowed_extensions: set[str]) -> int:
    """Count image files inside a split directory."""
    return sum(
        1
        for path in split_dir.rglob("*")
        if is_image_file(path=path, allowed_extensions=allowed_extensions)
    )


def validate_images_are_readable(split_dir: Path, allowed_extensions: set[str]) -> None:
    """Validate that all images in a split can be opened with Pillow."""
    image_paths = [
        path
        for path in split_dir.rglob("*")
        if is_image_file(path=path, allowed_extensions=allowed_extensions)
    ]

    for image_path in image_paths:
        try:
            with Image.open(image_path) as image:
                image.verify()
        except Exception as error:
            raise ValueError(f"Failed to read image: {image_path}") from error


def build_class_mapping(class_names: list[str]) -> dict[str, int]:
    """Build deterministic class-to-index mapping."""
    return {class_name: index for index, class_name in enumerate(sorted(class_names))}


def save_json(data: dict[str, int], output_path: Path) -> None:
    """Save dictionary as UTF-8 JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def make_summary_table(split_summaries: list[SplitSummary]) -> Table:
    """Create a rich table with dataset split statistics."""
    table = Table(title="Dataset summary")
    table.add_column("Split", justify="left")
    table.add_column("Classes", justify="right")
    table.add_column("Images", justify="right")

    for summary in split_summaries:
        table.add_row(
            summary.name,
            str(summary.num_classes),
            str(summary.num_images),
        )

    return table


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(config: DictConfig) -> None:
    """Validate dataset structure and save class mapping."""
    project_root = Path(hydra.utils.get_original_cwd())
    raw_dir = project_root / config.data.raw_dir
    class_mapping_path = project_root / config.data.class_mapping_path
    allowed_extensions = {extension.lower() for extension in config.data.allowed_extensions}

    console.rule("[bold green]Validating dataset")
    console.print(f"[bold]Raw directory:[/bold] {raw_dir}")

    dataset_root = resolve_dataset_root(raw_dir)
    console.print(f"[bold]Resolved dataset root:[/bold] {dataset_root}")

    split_names = ["train", "valid", "test"]
    split_dirs = {split_name: dataset_root / split_name for split_name in split_names}

    for split_name, split_dir in split_dirs.items():
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing split directory: {split_name}: {split_dir}")

    train_class_names = list_class_names(split_dirs["train"])
    expected_num_classes = int(config.data.num_classes)

    if len(train_class_names) != expected_num_classes:
        raise ValueError(
            f"Expected {expected_num_classes} classes in train split, got {len(train_class_names)}"
        )

    split_summaries: list[SplitSummary] = []

    for split_name, split_dir in split_dirs.items():
        class_names = list_class_names(split_dir)

        if sorted(class_names) != train_class_names:
            raise ValueError(f"Class names in {split_name} do not match train split classes.")

        validate_images_are_readable(
            split_dir=split_dir,
            allowed_extensions=allowed_extensions,
        )

        split_summaries.append(
            SplitSummary(
                name=split_name,
                num_classes=len(class_names),
                num_images=count_images(
                    split_dir=split_dir,
                    allowed_extensions=allowed_extensions,
                ),
            )
        )

    class_to_idx = build_class_mapping(train_class_names)
    save_json(class_to_idx, class_mapping_path)

    console.print(make_summary_table(split_summaries))
    console.print(f"[bold green]Saved class mapping:[/bold green] {class_mapping_path}")


if __name__ == "__main__":
    main()
