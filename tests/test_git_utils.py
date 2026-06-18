from __future__ import annotations

from pathlib import Path

from card_recognizer.utils.git import get_git_commit_hash, get_git_dirty_state


def test_git_helpers_return_safe_values() -> None:
    project_root = Path(".")

    commit_hash = get_git_commit_hash(project_root)
    dirty_state = get_git_dirty_state(project_root)

    assert isinstance(commit_hash, str)
    assert isinstance(dirty_state, bool)
