from pathlib import Path

import pytest

from card_recognizer.inference.checkpoint import (
    parse_metric_from_checkpoint_name,
    resolve_checkpoint_path,
)


def test_parse_metric_from_checkpoint_name() -> None:
    path = Path("epoch=11-val_macro_f1=0.8768.ckpt")

    assert parse_metric_from_checkpoint_name(path, "val_macro_f1") == pytest.approx(0.8768)


def test_resolve_checkpoint_path_prefers_best_metric(tmp_path: Path) -> None:
    low = tmp_path / "epoch=1-val_macro_f1=0.5000.ckpt"
    high = tmp_path / "epoch=2-val_macro_f1=0.8000.ckpt"
    last = tmp_path / "last.ckpt"

    low.write_text("low")
    high.write_text("high")
    last.write_text("last")

    selected = resolve_checkpoint_path(
        checkpoint_path=None,
        checkpoint_dir=tmp_path,
        monitor="val_macro_f1",
        mode="max",
    )

    assert selected == high.resolve()


def test_resolve_checkpoint_path_uses_explicit_path(tmp_path: Path) -> None:
    explicit = tmp_path / "custom.ckpt"
    explicit.write_text("checkpoint")

    selected = resolve_checkpoint_path(
        checkpoint_path=explicit,
        checkpoint_dir=tmp_path / "other",
    )

    assert selected == explicit.resolve()


def test_resolve_checkpoint_path_raises_for_missing_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint_path(
            checkpoint_path=None,
            checkpoint_dir=tmp_path / "missing",
        )
