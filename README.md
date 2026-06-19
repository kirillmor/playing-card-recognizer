# Playing Card Recognizer

A reproducible MLOps project for playing card image classification.

## Project overview

The goal of this project is to build an image classification system that recognizes a single playing card from an input image. The task is formulated as multiclass image classification with 53 classes.

The project is focused not only on model quality, but also on reproducibility, experiment tracking, data versioning, model packaging, and inference serving.

## Current verified stage

At the current stage, the project has a working GPU-enabled baseline training pipeline with MLflow experiment tracking.

Verified locally:

```text
GPU: NVIDIA GeForce RTX 3050 Ti Laptop GPU
VRAM: 4096 MiB
PyTorch: CUDA-enabled build
Training backend: PyTorch Lightning
Experiment tracking: MLflow Tracking Server
```

The following GPU run has been tested successfully:

```bash
uv run python -m card_recognizer.training.train \
  model=baseline_cnn \
  optimizer=adam \
  trainer=gpu \
  data.batch_size=64 \
  data.num_workers=4 \
  trainer.max_epochs=3
```

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
- Ruff
- pre-commit
- uv

Planned production stack:

- ONNX
- TensorRT
- Triton Inference Server

## Setup

Install the pinned Python version and synchronize the project environment:

```bash
uv python install 3.13.14
uv python pin 3.13.14
uv sync --dev
```

Install pre-commit hooks:

```bash
uv run pre-commit install
```

Run tests:

```bash
uv run pytest
```

Run all quality checks:

```bash
uv run pre-commit run --all-files
```

## CUDA / GPU verification

Check that the NVIDIA driver is visible:

```bash
nvidia-smi
```

Check that PyTorch sees CUDA:

```bash
uv run python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
print("device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("device name:", torch.cuda.get_device_name(0))
PY
```

Expected behavior:

```text
cuda available: True
device count: 1
device name: NVIDIA GeForce RTX 3050 Ti Laptop GPU
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

The GPU trainer config uses:

```yaml
accelerator: gpu
devices: 1
precision: 16-mixed
```

Mixed precision is useful on laptop GPUs because it reduces VRAM usage and can speed up training.

## Dataset

The project uses the Kaggle dataset:

```text
gpiosenka/cards-image-datasetclassification
```

The expected raw dataset layout is:

```text
data/raw/cards/
├── train/
├── valid/
└── test/
```

Each split contains one subdirectory per class.

Expected dataset summary:

```text
train: 53 classes, 7624 images
valid: 53 classes, 265 images
test: 53 classes, 265 images
```

## Kaggle credentials

For local development, the dataset may be downloaded from Kaggle.

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

For debugging:

```bash
uv run python -m card_recognizer.data.inspect data.batch_size=8 data.num_workers=0
```

For a faster local run with worker processes:

```bash
uv run python -m card_recognizer.data.inspect data.batch_size=32 data.num_workers=4
```

Expected batch format:

```text
images: [batch_size, 3, 224, 224], dtype=torch.float32
labels: [batch_size], dtype=torch.int64
```

## Models

### Baseline CNN

The baseline model is implemented in:

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

The baseline is intentionally simple. Its purpose is to verify the full MLOps pipeline before comparing it with a stronger transfer learning model.

### EfficientNet-B0

The project also includes EfficientNet-B0 configuration and transfer learning support.

Expected strategy:

```text
1. Replace the classifier head with a 53-class head.
2. Freeze the pretrained backbone for the first training stage.
3. Train the classification head.
4. Unfreeze the backbone.
5. Fine-tune with separate learning rates for backbone and head.
```

The EfficientNet-B0 model uses the `adamw` optimizer config with separate learning rates:

```yaml
backbone_lr: 0.0001
head_lr: 0.001
weight_decay: 0.0001
```

## Training

The training entrypoint is:

```bash
uv run python -m card_recognizer.training.train
```

The training pipeline includes:

- Hydra config loading;
- deterministic seeding;
- `CardsDataModule`;
- model factory;
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

The server is available at:

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

To reset local MLflow state during development:

```bash
rm -rf mlruns mlflow.db
```

## Baseline CNN GPU training

Run the verified baseline GPU training command:

```bash
uv run python -m card_recognizer.training.train \
  model=baseline_cnn \
  optimizer=adam \
  trainer=gpu \
  data.batch_size=64 \
  data.num_workers=4 \
  trainer.max_epochs=3
```

This command should:

- use the CUDA GPU;
- create an MLflow run;
- log hyperparameters;
- log train/validation/test metrics;
- log git metadata;
- save checkpoints;
- save plots under `plots/baseline_cnn/`;
- log plot artifacts to MLflow.

## EfficientNet-B0 GPU training

For EfficientNet-B0 on a 4 GB laptop GPU, start with a smaller batch size:

```bash
uv run python -m card_recognizer.training.train \
  model=efficientnet_b0 \
  optimizer=adamw \
  trainer=gpu \
  data.batch_size=8 \
  data.num_workers=4 \
  trainer.max_epochs=3
```

If there is no CUDA out-of-memory error, try:

```bash
uv run python -m card_recognizer.training.train \
  model=efficientnet_b0 \
  optimizer=adamw \
  trainer=gpu \
  data.batch_size=16 \
  data.num_workers=4 \
  trainer.max_epochs=10
```

If CUDA runs out of memory, decrease the batch size:

```text
data.batch_size=8
data.batch_size=4
```

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

Very short smoke runs may produce sparse or visually empty plots because there are too few epochs. For meaningful plots, run at least 3 epochs.

## Checkpoints

Checkpoints are saved under:

```text
artifacts/checkpoints/<model_name>/
```

For the baseline CNN:

```text
artifacts/checkpoints/baseline_cnn/
```

Checkpoint files are local/generated artifacts and must not be committed to git.

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
__pycache__/
```

DVC metadata files such as `.dvc` files should be committed because they allow other users to reproduce the data state without committing the actual data files.

## Current status

Implemented:

- Python package structure
- uv-based dependency management
- pinned Python version
- Ruff formatting and linting
- pre-commit hooks
- Hydra configuration skeleton
- Kaggle dataset download utility
- dataset validation utility
- deterministic `class_to_idx.json` generation
- DVC initialization and tracking metadata
- image preprocessing and augmentation transforms
- PyTorch Lightning DataModule
- DataModule inspection command
- baseline CNN
- EfficientNet-B0 transfer learning support
- model factory
- PyTorch Lightning multiclass classification module
- metrics with TorchMetrics
- checkpointing and early stopping callbacks
- MLflow Tracking Server script
- Lightning MLflow logger integration
- hyperparameter logging
- git commit hash logging
- git dirty-state logging
- local training plots
- plot artifact logging to MLflow
- CUDA/GPU training verification for baseline CNN

Not implemented yet:

- final EfficientNet-B0 experiment comparison
- richer evaluation reports
- ONNX export
- TensorRT export
- local inference API
- Triton inference serving

## Next steps

- Run a longer EfficientNet-B0 experiment on GPU.
- Compare baseline CNN and EfficientNet-B0 in MLflow.
- Add evaluation reports and confusion matrix.
- Export the best model to ONNX.
- Prepare TensorRT and Triton serving.
