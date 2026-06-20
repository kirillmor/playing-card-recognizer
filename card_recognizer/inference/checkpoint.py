from __future__ import annotations

import re
from pathlib import Path


def parse_metric_from_checkpoint_name(
    checkpoint_path: str | Path,
    metric_name: str,
) -> float | None:
    """Parse a metric value from a Lightning checkpoint filename.

    Example:
        epoch=11-val_macro_f1=0.8768.ckpt -> 0.8768
    """

    name = Path(checkpoint_path).name
    pattern = rf"{re.escape(metric_name)}=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    match = re.search(pattern, name)

    if match is None:
        return None

    return float(match.group(1))


def list_checkpoints(checkpoint_dir: str | Path) -> list[Path]:
    """Return checkpoint files from a directory."""

    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        return []

    return sorted(path for path in checkpoint_dir.glob("*.ckpt") if path.is_file())


def resolve_checkpoint_path(
    checkpoint_path: str | Path | None,
    checkpoint_dir: str | Path,
    monitor: str = "val_macro_f1",
    mode: str = "max",
) -> Path:
    """Resolve a checkpoint path for export/evaluation.

    Priority:
        1. Explicit checkpoint_path if provided.
        2. Checkpoint with the best parsed monitored metric.
        3. Most recently modified non-last checkpoint.
        4. Most recently modified checkpoint.
    """

    if checkpoint_path is not None and str(checkpoint_path).strip():
        resolved_path = Path(checkpoint_path)

        if not resolved_path.exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {resolved_path}")

        return resolved_path.resolve()

    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = list_checkpoints(checkpoint_dir)

    if not checkpoints:
        raise FileNotFoundError(f"No .ckpt files found in: {checkpoint_dir}")

    scored_checkpoints: list[tuple[float, Path]] = []
    for candidate in checkpoints:
        metric_value = parse_metric_from_checkpoint_name(candidate, monitor)
        if metric_value is not None:
            scored_checkpoints.append((metric_value, candidate))

    if scored_checkpoints:
        reverse = mode == "max"
        return sorted(scored_checkpoints, key=lambda item: item[0], reverse=reverse)[0][1].resolve()

    non_last_checkpoints = [
        checkpoint for checkpoint in checkpoints if checkpoint.name != "last.ckpt"
    ]
    fallback_candidates = non_last_checkpoints or checkpoints

    return max(fallback_candidates, key=lambda path: path.stat().st_mtime).resolve()
