from __future__ import annotations

import subprocess
from pathlib import Path


def _run_git_command(project_root: Path, command: list[str]) -> str | None:
    """Run a git command and return stdout if it succeeds."""
    try:
        completed_process = subprocess.run(
            command,
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

    return completed_process.stdout.strip()


def get_git_commit_hash(project_root: Path) -> str:
    """Return current git commit hash or 'unknown' if unavailable."""
    commit_hash = _run_git_command(project_root, ["git", "rev-parse", "HEAD"])
    return commit_hash if commit_hash else "unknown"


def get_git_dirty_state(project_root: Path) -> bool:
    """Return True if the git working tree has uncommitted changes."""
    status = _run_git_command(project_root, ["git", "status", "--short"])
    return bool(status)
