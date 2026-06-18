# Playing Card Recognizer

A reproducible MLOps project for playing card image classification.

## Project overview

The goal of this project is to build an image classification system that recognizes a single playing card from an input image. The task is formulated as multiclass image classification with 53 classes.

The project is focused not only on model quality, but also on reproducibility, experiment tracking, data versioning, model packaging, and inference serving.

## Python version

This project is pinned to Python 3.13.14.

```bash
uv python install 3.13.14
uv python pin 3.13.14
uv run python --version
```

Expected output:

```text
Python 3.13.14
```

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

## Repository structure

Current high-level structure:

```text
playing-card-recognizer/
├── configs/
│   ├── config.yaml
│   ├── data/cards.yaml
│   ├── inference/local.yaml
│   ├── logging/mlflow.yaml
│   ├── model/baseline_cnn.yaml
│   ├── model/efficientnet_b0.yaml
│   ├── optimizer/adam.yaml
│   ├── optimizer/adamw.yaml
│   ├── trainer/cpu.yaml
│   └── trainer/gpu.yaml
├── card_recognizer/
│   ├── __init__.py
│   ├── commands.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download.py
│   │   └── validate.py
│   ├── training/train.py
│   ├── models/
│   ├── inference/
│   ├── export/
│   └── utils/
├── tests/
│   ├── test_configs.py
│   ├── test_data_validation.py
│   └── test_package.py
├── data/raw/cards.dvc
├── artifacts/class_to_idx.json.dvc
├── plots/
├── reports/
├── scripts/
├── .dvc/
├── .dvcignore
├── .gitignore
├── .python-version
├── .pre-commit-config.yaml
├── pyproject.toml
├── uv.lock
└── README.md
```

## Setup

Install the pinned Python version and synchronize the project environment:

```bash
uv python install 3.13.14
uv python pin 3.13.14
uv sync --dev
```

Install and run pre-commit checks:

```bash
uv run pre-commit install
uv run pre-commit run --all-files
```

Run tests:

```bash
uv run pytest
```

Check the project CLI:

```bash
uv run card-recognizer
```

## Configuration

The project uses Hydra for hierarchical configuration management.

Main config:

```text
configs/config.yaml
```

Config groups:

```text
configs/data/
configs/model/
configs/optimizer/
configs/trainer/
configs/logging/
configs/inference/
```

The default configuration currently uses:

- dataset config: `configs/data/cards.yaml`
- model config: `configs/model/efficientnet_b0.yaml`
- optimizer config: `configs/optimizer/adamw.yaml`
- trainer config: `configs/trainer/cpu.yaml`
- logging config: `configs/logging/mlflow.yaml`
- inference config: `configs/inference/local.yaml`

## Train config check

At the current stage, the training entrypoint only loads and prints the composed Hydra configuration. Real data loading, model creation, training, validation, and logging will be added in later stages.

Run the default config:

```bash
uv run python -m card_recognizer.training.train
```

Run with overrides:

```bash
uv run python -m card_recognizer.training.train model=baseline_cnn optimizer=adam data.batch_size=16
```

This is useful because experiments can be configured from the command line without editing Python code.

## Dataset

The project uses the Kaggle dataset:

```text
gpiosenka/cards-image-datasetclassification
```

The expected raw dataset layout is:

```text
data/raw/cards/
├── train/
│   ├── ace of clubs/
│   ├── ace of diamonds/
│   └── ...
├── valid/
│   ├── ace of clubs/
│   ├── ace of diamonds/
│   └── ...
└── test/
    ├── ace of clubs/
    ├── ace of diamonds/
    └── ...
```

Each split contains one subdirectory per class.

## Kaggle credentials

For local development, the dataset may be downloaded from Kaggle. Depending on the local environment and Kaggle access settings, credentials may be required.

Option 1: environment variables:

```bash
export KAGGLE_USERNAME="<your-kaggle-username>"
export KAGGLE_KEY="<your-kaggle-api-key>"
```

Option 2: local Kaggle token:

```bash
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

Do not commit `kaggle.json`.

## Data download and validation

Download raw data:

```bash
uv run python -m card_recognizer.data.download
```

Validate dataset structure and save the class mapping:

```bash
uv run python -m card_recognizer.data.validate
```

The validation command checks that:

- `train`, `valid`, and `test` directories exist;
- each split contains the same set of class directories;
- the train split contains the expected number of classes;
- image files can be opened with Pillow;
- `artifacts/class_to_idx.json` is generated.

Expected dataset summary:

```text
train: 53 classes, 7624 images
valid: 53 classes, 265 images
test: 53 classes, 265 images
```

## DVC

The project uses DVC for data and generated artifact tracking.

Initialize DVC:

```bash
uv run dvc init
```

For local development, create local DVC remotes outside the repository:

```bash
mkdir -p ../dvc-storage/playing-card-recognizer/data
mkdir -p ../dvc-storage/playing-card-recognizer/models
uv run dvc remote add -d data-remote ../dvc-storage/playing-card-recognizer/data
uv run dvc remote add models-remote ../dvc-storage/playing-card-recognizer/models
```

Track raw data and generated class mapping:

```bash
uv run dvc add data/raw/cards
uv run dvc add artifacts/class_to_idx.json
```

Push DVC-tracked files to the configured remote:

```bash
uv run dvc push -r data-remote
```

Pull DVC-tracked files after cloning the repository:

```bash
uv run dvc pull
```

The following files are committed to git:

```text
data/raw/cards.dvc
artifacts/class_to_idx.json.dvc
.dvc/config
.dvcignore
```

The following files are not committed to git:

```text
data/raw/cards/
artifacts/class_to_idx.json
```

## Development checks

Before committing changes, run:

```bash
uv run pytest
uv run pre-commit run --all-files
```

If pre-commit modifies files, stage the changes and run it again:

```bash
git add .
uv run pre-commit run --all-files
```

## Git hygiene

Generated/local artifacts must not be committed:

```text
outputs/
data/raw/cards/
artifacts/class_to_idx.json
mlruns/
mlflow.db
*.ckpt
*.pth
*.pt
*.onnx
*.plan
*.engine
*.trt
kaggle.json
```

Hydra creates `outputs/` directories for local run artifacts. They are useful for debugging individual runs, but they should stay out of git.

DVC metadata files such as `.dvc` files should be committed because they allow other users to reproduce the data state without committing the actual data files.

## Current status

Implemented:

- Python package structure
- uv-based dependency management
- pinned Python version
- Ruff formatting and linting
- pre-commit hooks
- basic package import test
- Hydra configuration skeleton
- training config inspection entrypoint
- Kaggle dataset download utility
- dataset validation utility
- deterministic `class_to_idx.json` generation
- DVC initialization and tracking metadata

Not implemented yet:

- PyTorch Lightning DataModule
- image preprocessing and augmentation transforms
- baseline CNN
- EfficientNet-B0 fine-tuning
- MLflow experiment logging
- training plots
- ONNX/TensorRT export
- Triton inference serving

## Next steps

- Implement image transforms
- Implement `LightningDataModule`
- Add tests for batch loading and tensor shapes
- Implement baseline CNN
- Implement the first real PyTorch Lightning training loop
