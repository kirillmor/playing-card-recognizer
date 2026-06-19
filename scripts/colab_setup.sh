#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/content/playing-card-recognizer}"
PYTHON_VERSION="${PYTHON_VERSION:-3.13.14}"

echo "[Colab setup] Project root: ${PROJECT_ROOT}"
echo "[Colab setup] Python version: ${PYTHON_VERSION}"

cd "${PROJECT_ROOT}"

if ! command -v uv >/dev/null 2>&1; then
  echo "[Colab setup] Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
else
  echo "[Colab setup] uv is already installed."
fi

export PATH="${HOME}/.local/bin:${PATH}"

echo "[Colab setup] Installing pinned Python..."
uv python install "${PYTHON_VERSION}"

echo "[Colab setup] Synchronizing dependencies..."
uv sync --dev

echo "[Colab setup] Checking environment..."
uv run python --version

uv run python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
print("device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("device name:", torch.cuda.get_device_name(0))
else:
    raise SystemExit("CUDA is not available. Enable GPU runtime in Colab.")
PY

echo "[Colab setup] Done."
