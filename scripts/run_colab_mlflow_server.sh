#!/usr/bin/env bash
set -euo pipefail

# Colab often uses port 8080 for Jupyter-related services.
# Use 5000 for the MLflow tracking server inside the Colab runtime.
MLFLOW_HOST="${MLFLOW_HOST:-127.0.0.1}"
MLFLOW_PORT="${MLFLOW_PORT:-5000}"
MLFLOW_BACKEND_STORE_URI="${MLFLOW_BACKEND_STORE_URI:-sqlite:///mlflow.db}"
MLFLOW_DEFAULT_ARTIFACT_ROOT="${MLFLOW_DEFAULT_ARTIFACT_ROOT:-./mlruns}"

mlflow server \
  --host "${MLFLOW_HOST}" \
  --port "${MLFLOW_PORT}" \
  --backend-store-uri "${MLFLOW_BACKEND_STORE_URI}" \
  --default-artifact-root "${MLFLOW_DEFAULT_ARTIFACT_ROOT}"
