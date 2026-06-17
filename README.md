# Playing Card Recognizer

A reproducible MLOps project for playing card image classification.

## Project overview

The goal of this project is to build an image classification system that recognizes
a single playing card from an input image. The task is formulated as multiclass
image classification with 53 classes.

The project is focused not only on model quality, but also on reproducibility,
experiment tracking, data versioning, model packaging, and inference serving.

## Python version

This project is pinned to Python 3.13.14.

## Planned stack

- Python
- PyTorch
- PyTorch Lightning
- TorchMetrics
- Hydra
- DVC
- MLflow
- ONNX
- TensorRT
- Triton Inference Server

## Setup

```bash
uv python install 3.13.14
uv python pin 3.13.14
uv sync --dev
uv run pre-commit install
uv run pre-commit run --all-files
uv run pytest
