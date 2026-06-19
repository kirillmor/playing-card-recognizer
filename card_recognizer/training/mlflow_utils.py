from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig, OmegaConf

from card_recognizer.utils.git import get_git_commit_hash, get_git_dirty_state


def flatten_dict(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested dictionary for MLflow params."""
    flattened: dict[str, Any] = {}

    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)

        if isinstance(value, dict):
            flattened.update(flatten_dict(value, prefix=full_key))
        elif isinstance(value, list):
            flattened[full_key] = json.dumps(value)
        else:
            flattened[full_key] = value

    return flattened


def build_mlflow_logger(config: DictConfig) -> MLFlowLogger | bool:
    """Build MLflow logger if logging is enabled."""
    if not bool(config.logging.enabled):
        return False

    logger = MLFlowLogger(
        experiment_name=str(config.logging.experiment_name),
        run_name=str(config.logging.run_name),
        tracking_uri=str(config.logging.tracking_uri),
        log_model=bool(config.logging.log_model),
    )

    # Force lazy MLflow run creation immediately.
    _ = logger.experiment
    _ = logger.run_id

    # Log a tiny technical metric so the run appears in MLflow immediately.
    logger.log_metrics({"run_started": 1.0}, step=0)

    return logger


def log_hyperparameters_and_git_metadata(
    logger: MLFlowLogger | bool,
    config: DictConfig,
    project_root: Path,
) -> None:
    """Log Hydra config and git metadata to MLflow."""
    if not isinstance(logger, MLFlowLogger):
        return

    if bool(config.logging.log_hyperparams):
        config_dict = OmegaConf.to_container(config, resolve=True)
        if isinstance(config_dict, dict):
            logger.log_hyperparams(flatten_dict(config_dict))

    if bool(config.logging.log_git_commit):
        git_commit = get_git_commit_hash(project_root)
        git_dirty = get_git_dirty_state(project_root)

        logger.log_hyperparams(
            {
                "git.commit": git_commit,
                "git.dirty": git_dirty,
            }
        )

        run_id = logger.run_id
        if run_id is not None:
            logger.experiment.set_tag(run_id, "git_commit", git_commit)
            logger.experiment.set_tag(run_id, "git_dirty", str(git_dirty))


def log_artifacts_directory(
    logger: MLFlowLogger | bool,
    local_dir: Path,
    artifact_path: str,
) -> None:
    """Log a local directory to MLflow artifacts if it exists."""
    if not isinstance(logger, MLFlowLogger):
        return

    if not local_dir.exists():
        return

    run_id = logger.run_id
    if run_id is None:
        return

    logger.experiment.log_artifacts(
        run_id=run_id,
        local_dir=str(local_dir),
        artifact_path=artifact_path,
    )
