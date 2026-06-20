#!/usr/bin/env bash
set -euo pipefail

TRITON_IMAGE="${TRITON_IMAGE:-nvcr.io/nvidia/tritonserver:24.12-py3}"
MODEL_REPOSITORY="${MODEL_REPOSITORY:-$(pwd)/deployment/triton_model_repository}"
TRITON_ENABLE_GPU="${TRITON_ENABLE_GPU:-true}"
TRITON_LOAD_MODEL="${TRITON_LOAD_MODEL:-}"

DOCKER_ARGS=(
  --rm
  --shm-size=1g
  -p 8000:8000
  -p 8001:8001
  -p 8002:8002
  -v "${MODEL_REPOSITORY}:/models"
)

if [ "${TRITON_ENABLE_GPU}" = "true" ]; then
  DOCKER_ARGS+=(--gpus=all)
fi

TRITON_ARGS=(
  tritonserver
  --model-repository=/models
)

if [ -n "${TRITON_LOAD_MODEL}" ]; then
  TRITON_ARGS+=(
    --model-control-mode=explicit
    --load-model="${TRITON_LOAD_MODEL}"
  )
fi

docker run "${DOCKER_ARGS[@]}" "${TRITON_IMAGE}" "${TRITON_ARGS[@]}"

# TRITON_ENABLE_GPU=false TRITON_LOAD_MODEL=card_recognizer_onnx bash scripts/run_triton_server.sh
# TRITON_ENABLE_GPU=true TRITON_LOAD_MODEL=card_recognizer_tensorrt bash scripts/run_triton_server.sh
