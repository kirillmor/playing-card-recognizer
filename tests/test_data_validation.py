from __future__ import annotations

from pathlib import Path

from card_recognizer.data.validate import build_class_mapping, is_image_file


def test_is_image_file_accepts_allowed_extension(tmp_path: Path) -> None:
    image_path = tmp_path / "card.JPG"
    image_path.write_bytes(b"fake")

    assert is_image_file(image_path, {".jpg", ".jpeg", ".png"})


def test_is_image_file_rejects_unknown_extension(tmp_path: Path) -> None:
    text_path = tmp_path / "README.txt"
    text_path.write_text("not an image", encoding="utf-8")

    assert not is_image_file(text_path, {".jpg", ".jpeg", ".png"})


def test_build_class_mapping_is_sorted() -> None:
    class_mapping = build_class_mapping(["king of spades", "ace of clubs"])

    assert class_mapping == {
        "ace of clubs": 0,
        "king of spades": 1,
    }
