# Playing Card Recognizer

A reproducible MLOps project for playing card image classification.

## Project overview

The goal is to build a system that recognizes a single playing card from an input image. The task is formulated as multiclass image classification with 53 classes.

The project focuses on reproducibility, experiment tracking, data versioning, evaluation reporting, model comparison, and future inference serving.

## Current stage

Implemented and verified:

- PyTorch Lightning training pipeline;
- baseline CNN model;
- EfficientNet-B0 transfer learning support;
- CUDA/GPU training support;
- MLflow experiment tracking;
- local training plots;
- DVC-tracked dataset metadata;
- evaluation reports;
- bootstrap confidence intervals for evaluation metrics;
- confusion matrix plots;
- per-class classification report;
- predictions table;
- model comparison reports;
- best-model selection by configurable metric.

Verified local GPU setup:

```text
GPU: NVIDIA GeForce RTX 3050 Ti Laptop GPU
VRAM: 4096 MiB
PyTorch: CUDA-enabled build
Training backend: PyTorch Lightning
Experiment tracking: MLflow Tracking Server
```

A baseline GPU run was tested successfully:

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

The project is pinned to Python 3.13.14.

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
- torchvision
- Hydra
- DVC
- MLflow
- pandas
- matplotlib
- Ruff
- pre-commit
- uv

Planned production stack:

- ONNX
- TensorRT
- Triton Inference Server

## Repository structure

```text
playing-card-recognizer/
├── configs/
│   ├── config.yaml
│   ├── data/cards.yaml
│   ├── evaluation/default.yaml
│   ├── inference/local.yaml
│   ├── logging/mlflow.yaml
│   ├── model/baseline_cnn.yaml
│   ├── model/efficientnet_b0.yaml
│   ├── optimizer/adam.yaml
│   ├── optimizer/adamw.yaml
│   ├── selection/default.yaml
│   └── trainer/
│       ├── cpu.yaml
│       └── gpu.yaml
├── card_recognizer/
│   ├── data/
│   ├── evaluation/
│   ├── export/
│   ├── inference/
│   ├── models/
│   ├── selection/
│   ├── training/
│   └── utils/
├── tests/
├── scripts/
│   └── run_mlflow_server.sh
├── data/raw/cards.dvc
├── artifacts/class_to_idx.json.dvc
├── plots/.gitkeep
├── reports/.gitkeep
├── pyproject.toml
├── uv.lock
└── README.md
```

## Setup

Install Python and synchronize dependencies:

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

Run all checks:

```bash
uv run pre-commit run --all-files
```

## CUDA / GPU verification

Check NVIDIA driver visibility:

```bash
nvidia-smi
```

Check PyTorch CUDA support:

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

Expected behavior on the verified machine:

```text
cuda available: True
device count: 1
device name: NVIDIA GeForce RTX 3050 Ti Laptop GPU
```

## Configuration

Hydra is used for hierarchical configuration.

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
configs/evaluation/
configs/selection/
```

The default configuration uses:

- dataset config: `configs/data/cards.yaml`;
- model config: `configs/model/baseline_cnn.yaml`;
- optimizer config: `configs/optimizer/adam.yaml`;
- trainer config: `configs/trainer/cpu.yaml`;
- logging config: `configs/logging/mlflow.yaml`;
- inference config: `configs/inference/local.yaml`;
- evaluation config: `configs/evaluation/default.yaml`;
- selection config: `configs/selection/default.yaml`.

GPU trainer config:

```yaml
accelerator: gpu
devices: 1
precision: 16-mixed
```

## Dataset

The project uses the Kaggle dataset:

```text
gpiosenka/cards-image-datasetclassification
```

Expected raw dataset layout:

```text
data/raw/cards/
├── train/
├── valid/
└── test/
```

Expected dataset summary:

```text
train: 53 classes, 7624 images
valid: 53 classes, 265 images
test: 53 classes, 265 images
```

## Kaggle credentials

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

Download data:

```bash
uv run python -m card_recognizer.data.download
```

Validate data and generate class mapping:

```bash
uv run python -m card_recognizer.data.validate
```

The validation command checks split structure, class consistency, image readability, and creates:

```text
artifacts/class_to_idx.json
```

## DVC

DVC is used for data and generated artifact tracking.

Local DVC remotes:

```bash
mkdir -p ../dvc-storage/playing-card-recognizer/data
mkdir -p ../dvc-storage/playing-card-recognizer/models
uv run dvc remote add -d data-remote ../dvc-storage/playing-card-recognizer/data
uv run dvc remote add models-remote ../dvc-storage/playing-card-recognizer/models
```

Track raw data and class mapping:

```bash
uv run dvc add data/raw/cards
uv run dvc add artifacts/class_to_idx.json
```

Push DVC-tracked files:

```bash
uv run dvc push -r data-remote
```

Pull after cloning:

```bash
uv run dvc pull
```

Committed DVC metadata:

```text
data/raw/cards.dvc
artifacts/class_to_idx.json.dvc
.dvc/config
.dvcignore
```

Actual raw data and generated class mapping are not committed to git.

## DataModule inspection

Run:

```bash
uv run python -m card_recognizer.data.inspect
```

Debug run:

```bash
uv run python -m card_recognizer.data.inspect data.batch_size=8 data.num_workers=0
```

Faster local run:

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

Implemented in:

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

### EfficientNet-B0

The project includes EfficientNet-B0 transfer learning support.

Strategy:

```text
1. Replace classifier head with a 53-class head.
2. Freeze pretrained backbone for the first stage.
3. Train classifier head.
4. Unfreeze backbone.
5. Fine-tune with separate learning rates.
```

AdamW config:

```yaml
backbone_lr: 0.0001
head_lr: 0.001
weight_decay: 0.0001
```

## Training

Training entrypoint:

```bash
uv run python -m card_recognizer.training.train
```

The training pipeline includes:

- Hydra config loading;
- deterministic seeding;
- PyTorch Lightning DataModule;
- model factory;
- LightningModule;
- CrossEntropyLoss;
- Adam/AdamW optimizer support;
- optional cosine scheduler support;
- TorchMetrics classification metrics;
- checkpointing by `val_macro_f1`;
- early stopping by `val_macro_f1`;
- MLflow logger integration;
- hyperparameter logging;
- git metadata logging;
- local plot saving;
- plot artifact logging to MLflow;
- final test evaluation using the best checkpoint.

Logged training metrics:

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

Start MLflow server in a separate terminal:

```bash
uv run bash scripts/run_mlflow_server.sh
```

Server URL:

```text
http://127.0.0.1:8080
```

Local MLflow files:

```text
mlflow.db
mlruns/
```

These are generated local files and are not committed.

Reset local MLflow state:

```bash
rm -rf mlruns mlflow.db
```

## Baseline CNN GPU training

```bash
uv run python -m card_recognizer.training.train \
  model=baseline_cnn \
  optimizer=adam \
  trainer=gpu \
  data.batch_size=64 \
  data.num_workers=4 \
  trainer.max_epochs=3
```

This should:

- use CUDA GPU;
- create an MLflow run;
- log hyperparameters;
- log metrics;
- log git metadata;
- save checkpoints;
- save plots under `plots/baseline_cnn/`;
- log plot artifacts to MLflow.

## EfficientNet-B0 GPU training

For a 4 GB laptop GPU, start with a smaller batch size:

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

## Evaluation

Evaluation entrypoint:

```bash
uv run python -m card_recognizer.evaluation.evaluate
```

The evaluation pipeline includes:

- checkpoint resolution;
- validation or test split evaluation;
- logits and prediction collection;
- top-k prediction extraction;
- confusion matrix computation;
- per-class precision, recall, F1, and support;
- macro/weighted summary metrics;
- bootstrap confidence intervals;
- predictions CSV;
- confusion matrix plots;
- worst-classes-by-F1 plot;
- optional MLflow logging.

Evaluation outputs:

```text
reports/evaluation/<model_name>/
├── bootstrap_confidence_intervals.csv
├── classification_report.csv
├── confusion_matrix.csv
├── predictions.csv
└── summary_metrics.json

plots/<model_name>/
├── confusion_matrix.png
├── confusion_matrix_normalized.png
└── worst_classes_by_f1.png
```

Evaluation reports and plots are generated artifacts and are not committed.

### Baseline CNN evaluation

```bash
uv run python -m card_recognizer.evaluation.evaluate \
  model=baseline_cnn \
  optimizer=adam \
  trainer=gpu \
  data.batch_size=64 \
  data.num_workers=4 \
  evaluation.split=test
```

Smoke evaluation with fewer bootstrap samples:

```bash
uv run python -m card_recognizer.evaluation.evaluate \
  model=baseline_cnn \
  optimizer=adam \
  trainer=gpu \
  data.batch_size=64 \
  data.num_workers=4 \
  evaluation.split=test \
  evaluation.bootstrap.num_samples=50
```

### EfficientNet-B0 evaluation

```bash
uv run python -m card_recognizer.evaluation.evaluate \
  model=efficientnet_b0 \
  model.pretrained=false \
  optimizer=adamw \
  trainer=gpu \
  data.batch_size=8 \
  data.num_workers=4 \
  evaluation.split=test
```

`model.pretrained=false` avoids downloading pretrained weights during evaluation. The trained weights are loaded from the checkpoint.

## Model comparison and best-model selection

Selection entrypoint:

```bash
uv run python -m card_recognizer.selection.select_best_model
```

The selection pipeline reads:

```text
reports/evaluation/<model_name>/summary_metrics.json
reports/evaluation/<model_name>/bootstrap_confidence_intervals.csv
```

It generates:

```text
reports/model_comparison/
├── best_model.json
├── comparison.csv
└── comparison.md
```

Selection config:

```text
configs/selection/default.yaml
```

Default metric:

```yaml
metric: macro_f1
higher_is_better: true
```

Default models:

```yaml
models:
  - baseline_cnn
  - efficientnet_b0
```

Run comparison after evaluating models:

```bash
uv run python -m card_recognizer.selection.select_best_model
```

If only baseline has been evaluated:

```bash
uv run python -m card_recognizer.selection.select_best_model \
  selection.models='[baseline_cnn]'
```

Useful local checks:

```bash
cat reports/model_comparison/best_model.json
cat reports/model_comparison/comparison.md
```

Model comparison outputs are generated reports and are not committed.

## Google Colab workflow plan

Local development is used for:

- implementation;
- unit tests;
- pre-commit checks;
- smoke training;
- smoke evaluation.

Google Colab is planned for:

- longer baseline CNN runs;
- longer EfficientNet-B0 runs;
- final model comparison;
- final checkpoint generation before ONNX/TensorRT export.

The repository remains the source of truth. Colab is used only as a compute backend.

Planned additions:

```text
configs/trainer/colab.yaml
configs/logging/colab.yaml
scripts/colab_setup.sh
docs/colab_training.md or notebooks/colab_training.ipynb
```

Expected Colab process:

```text
1. Enable GPU runtime.
2. Clone the GitHub repository.
3. Install uv.
4. Run uv sync --dev.
5. Pull data through DVC or download through Kaggle.
6. Start or configure MLflow tracking.
7. Run long training.
8. Run evaluation.
9. Run model comparison.
10. Save the best checkpoint and reports.
```

## Generated artifacts and git hygiene

Generated/local artifacts must not be committed:

```text
outputs/
data/raw/cards/
artifacts/class_to_idx.json
artifacts/checkpoints/
plots/baseline_cnn/
plots/efficientnet_b0/
reports/evaluation/
reports/model_comparison/
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

Recommended keep-files for empty generated-output roots:

```text
plots/.gitkeep
reports/.gitkeep
```

DVC metadata files such as `.dvc` files should be committed.

## Development checks before commit

```bash
uv run pytest
uv run pre-commit run --all-files
```

If pre-commit modifies files:

```bash
git add .
uv run pre-commit run --all-files
```

Check that generated reports are not staged:

```bash
git status --short --untracked-files=all
git check-ignore -v reports/model_comparison/best_model.json
```

## Current status

Implemented:

- Python package structure;
- uv-based dependency management;
- pinned Python version;
- Ruff formatting and linting;
- pre-commit hooks;
- Hydra configuration;
- Kaggle dataset download utility;
- dataset validation utility;
- deterministic `class_to_idx.json` generation;
- DVC initialization and tracking metadata;
- image preprocessing and augmentation transforms;
- PyTorch Lightning DataModule;
- DataModule inspection command;
- baseline CNN;
- EfficientNet-B0 transfer learning support;
- model factory;
- Lightning multiclass classification module;
- metrics with TorchMetrics;
- checkpointing and early stopping callbacks;
- MLflow Tracking Server script;
- MLflow logger integration;
- hyperparameter logging;
- git metadata logging;
- local training plots;
- plot artifact logging to MLflow;
- CUDA/GPU training verification;
- standalone evaluation pipeline;
- summary evaluation metrics;
- bootstrap confidence intervals;
- per-class classification report;
- predictions table;
- confusion matrix CSV;
- confusion matrix plots;
- worst-classes-by-F1 plot;
- evaluation artifacts logged to MLflow;
- model comparison;
- best-model selection by configurable metric.

Not implemented yet:

- Colab workflow documentation/scripts;
- final EfficientNet-B0 long experiment;
- ONNX export;
- TensorRT export;
- local inference API;
- Triton inference serving.

## Next steps

- Add Google Colab workflow for longer GPU training.
- Run longer EfficientNet-B0 experiments on Colab.
- Compare baseline CNN and EfficientNet-B0 using `macro_f1`.
- Select the best checkpoint.
- Export the selected model to ONNX.
- Prepare TensorRT and Triton serving.
