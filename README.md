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

## Main stack

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
│   ├── data/
│   │   └── cards.yaml
│   ├── inference/
│   │   └── local.yaml
│   ├── logging/
│   │   └── mlflow.yaml
│   ├── model/
│   │   ├── baseline_cnn.yaml
│   │   └── efficientnet_b0.yaml
│   ├── optimizer/
│   │   ├── adam.yaml
│   │   └── adamw.yaml
│   └── trainer/
│       ├── cpu.yaml
│       └── gpu.yaml
├── card_recognizer/
│   ├── __init__.py
│   ├── commands.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── datamodule.py
│   │   ├── download.py
│   │   ├── inspect.py
│   │   ├── transforms.py
│   │   └── validate.py
│   ├── export/
│   ├── inference/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline_cnn.py
│   │   ├── factory.py
│   │   └── lightning_module.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── mlflow_utils.py
│   │   ├── plots.py
│   │   └── train.py
│   └── utils/
│       ├── __init__.py
│       └── git.py
├── tests/
│   ├── test_configs.py
│   ├── test_datamodule.py
│   ├── test_data_validation.py
│   ├── test_git_utils.py
│   ├── test_lightning_module.py
│   ├── test_mlflow_utils.py
│   ├── test_models.py
│   ├── test_package.py
│   └── test_plots.py
├── scripts/
│   └── run_mlflow_server.sh
├── data/
│   └── raw/
│       └── cards.dvc
├── artifacts/
│   └── class_to_idx.json.dvc
├── plots/
├── reports/
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

## Dependency notes

The project uses MLflow together with Python 3.13. The dependency constraints should keep MLflow compatible with pandas and protobuf:

```text
pandas>=2.2,<3
mlflow==3.14.0
protobuf>=5.29,<6
```

Verify MLflow:

```bash
uv run python -c "import pandas; import mlflow; import google.protobuf; print('pandas', pandas.__version__); print('mlflow', mlflow.__version__); print('protobuf', google.protobuf.__version__)"
uv run mlflow --version
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
- model config: `configs/model/baseline_cnn.yaml`
- optimizer config: `configs/optimizer/adam.yaml`
- trainer config: `configs/trainer/cpu.yaml`
- logging config: `configs/logging/mlflow.yaml`
- inference config: `configs/inference/local.yaml`

The EfficientNet-B0 config is already present, but the current training pipeline starts with the baseline CNN because it is simpler and easier to debug end-to-end.

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

## DataModule inspection

The project provides a diagnostic command that builds the PyTorch Lightning DataModule and prints dataset sizes and one training batch shape.

Run with the default config:

```bash
uv run python -m card_recognizer.data.inspect
```

For CPU-friendly debugging:

```bash
uv run python -m card_recognizer.data.inspect data.batch_size=8 data.num_workers=0
```

Expected dataset summary:

```text
train: 7624 samples, 53 classes
valid: 265 samples, 53 classes
test: 265 samples, 53 classes
```

Expected batch format:

```text
images: [batch_size, 3, 224, 224], dtype=torch.float32
labels: [batch_size], dtype=torch.int64
```

## Baseline model

The current baseline is a small convolutional neural network implemented in:

```text
card_recognizer/models/baseline_cnn.py
```

Architecture:

```text
Input: [3, 224, 224]

ConvBlock 1: 3   -> 32
ConvBlock 2: 32  -> 64
ConvBlock 3: 64  -> 128
ConvBlock 4: 128 -> 256

Each ConvBlock:
Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d

Head:
AdaptiveAvgPool2d -> Flatten -> Dropout -> Linear(num_classes)
```

The baseline is intentionally simple. Its purpose is to verify the full training pipeline before adding transfer learning with EfficientNet-B0.

## Training

The training entrypoint is:

```bash
uv run python -m card_recognizer.training.train
```

The training pipeline currently includes:

- Hydra config loading;
- deterministic seeding;
- `CardsDataModule`;
- baseline CNN model creation;
- PyTorch Lightning `LightningModule`;
- `CrossEntropyLoss`;
- Adam/AdamW optimizer support;
- optional cosine scheduler support;
- TorchMetrics classification metrics;
- model checkpointing by `val_macro_f1`;
- early stopping by `val_macro_f1`;
- MLflow logger integration;
- hyperparameter logging;
- git commit hash logging;
- git dirty-state logging;
- local plot saving;
- plot artifact logging to MLflow;
- final test evaluation using the best checkpoint.

Logged metrics:

```text
train_loss
train_accuracy
train_macro_f1
train_macro_precision
train_macro_recall
train_top3_accuracy

val_loss
val_accuracy
val_macro_f1
val_macro_precision
val_macro_recall
val_top3_accuracy

test_loss
test_accuracy
test_macro_f1
test_macro_precision
test_macro_recall
test_top3_accuracy
```

## MLflow Tracking Server

Start the local MLflow Tracking Server in a separate terminal:

```bash
uv run bash scripts/run_mlflow_server.sh
```

The script starts MLflow on:

```text
http://127.0.0.1:8080
```

The tracking backend is stored locally in:

```text
mlflow.db
```

Artifacts are stored locally in:

```text
mlruns/
```

Both `mlflow.db` and `mlruns/` are local generated artifacts and must not be committed to git.

## Smoke training with MLflow

Run a short CPU-friendly smoke training job:

```bash
uv run python -m card_recognizer.training.train \
  data.batch_size=8 \
  data.num_workers=0 \
  trainer.max_epochs=1 \
  trainer.limit_train_batches=5 \
  trainer.limit_val_batches=2 \
  trainer.limit_test_batches=2
```

Expected behavior:

- train dataloader is created;
- model forward pass works;
- loss is computed;
- validation runs;
- metrics are computed;
- MLflow run is created;
- hyperparameters are logged;
- git metadata is logged;
- checkpoint is saved;
- local plots are saved;
- plots are logged to MLflow artifacts;
- test evaluation runs using the best checkpoint.

Open the MLflow UI:

```text
http://127.0.0.1:8080
```

Expected MLflow experiment:

```text
card-recognition
```

Expected MLflow run content:

```text
params
metrics
artifacts
tags
```

Useful metadata to check:

```text
git_commit
git_dirty
```

## Training without MLflow

For local debugging, MLflow can be disabled:

```bash
uv run python -m card_recognizer.training.train \
  logging.enabled=false \
  data.batch_size=8 \
  data.num_workers=0 \
  trainer.max_epochs=1 \
  trainer.limit_train_batches=5 \
  trainer.limit_val_batches=2 \
  trainer.limit_test_batches=2
```

Local plots are still saved if `logging.save_plots=true`.

## Training plots

Training plots are saved under:

```text
plots/<model_name>/
```

For the baseline CNN:

```text
plots/baseline_cnn/
├── accuracy.png
├── loss.png
├── macro_f1.png
├── metrics_history.json
└── top3_accuracy.png
```

At least three plots are generated during training. These plots are local generated artifacts and should not be committed to git.

## Checkpoints

Checkpoints are saved under:

```text
artifacts/checkpoints/<model_name>/
```

For the baseline CNN:

```text
artifacts/checkpoints/baseline_cnn/
```

Check generated checkpoints:

```bash
find artifacts/checkpoints -maxdepth 3 -type f | sort
```

Checkpoint files are local/generated artifacts and must not be committed to git.

## Full baseline training

Run a longer baseline training job on CPU:

```bash
uv run python -m card_recognizer.training.train \
  model=baseline_cnn \
  optimizer=adam \
  trainer=cpu \
  data.batch_size=32 \
  data.num_workers=0 \
  trainer.max_epochs=3
```

If a GPU is available, use:

```bash
uv run python -m card_recognizer.training.train \
  model=baseline_cnn \
  optimizer=adam \
  trainer=gpu \
  data.batch_size=64
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
artifacts/checkpoints/
plots/*/
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
- Kaggle dataset download utility
- dataset validation utility
- deterministic `class_to_idx.json` generation
- DVC initialization and tracking metadata
- image preprocessing and augmentation transforms
- PyTorch Lightning DataModule
- DataModule inspection command
- DataModule tests on a tiny synthetic ImageFolder dataset
- baseline CNN
- model factory
- PyTorch Lightning multiclass classification module
- metrics with TorchMetrics
- baseline smoke training pipeline
- checkpointing and early stopping callbacks
- MLflow Tracking Server script
- Lightning MLflow logger integration
- hyperparameter logging
- git commit hash logging
- git dirty-state logging
- local training plots
- plot artifact logging to MLflow
- metrics history JSON

Not implemented yet:

- EfficientNet-B0 fine-tuning
- ONNX/TensorRT export
- local inference API
- Triton inference serving

## Next steps

- Implement EfficientNet-B0 transfer learning
- Compare baseline CNN and EfficientNet-B0
- Add richer evaluation reports
- Export the best model to ONNX
- Prepare TensorRT and Triton serving
