# Playing Card Recognizer

Image classification project for recognizing playing cards from images.

The project solves a 53-class classification task: each image contains one playing card, and the model predicts the card class.

The repository is organized as a production-style Python package rather than a notebook. Training, evaluation, data validation, MLflow logging, Hydra configuration, DVC data tracking, and model comparison are implemented as reproducible command-line workflows.

---

## Project status

Implemented:

* PyTorch / Lightning training pipeline
* Baseline CNN model
* EfficientNet-B0 fine-tuning pipeline
* Two-phase EfficientNet training:

  * frozen backbone phase
  * full fine-tuning phase
* Hydra configuration system
* MLflow experiment tracking
* DVC tracking for data artifacts
* Dataset validation utilities
* Evaluation reports:

  * summary metrics
  * classification report
  * confusion matrix
  * normalized confusion matrix
  * predictions table
  * bootstrap confidence intervals
* Model comparison and best model selection
* Local and Google Colab training workflows
* Pre-commit checks with Ruff

---

## Task

The goal is to classify images of playing cards into 53 classes.

Dataset:

* Kaggle dataset: `gpiosenka/cards-image-datasetclassification`
* Image size: `224x224x3`
* Splits:

  * train: 7,624 images
  * valid: 265 images
  * test: 265 images
* Number of classes: 53

The validation and test splits are small: approximately 5 images per class. For this reason, macro-averaged metrics and bootstrap confidence intervals are important for model evaluation.

---

## Repository structure

```text
playing-card-recognizer/
├── card_recognizer/
│   ├── data/
│   │   ├── datamodule.py
│   │   ├── download.py
│   │   ├── inspect.py
│   │   ├── transforms.py
│   │   └── validate.py
│   ├── evaluation/
│   │   ├── evaluate.py
│   │   ├── metrics.py
│   │   └── plots.py
│   ├── models/
│   │   ├── baseline_cnn.py
│   │   ├── efficientnet.py
│   │   ├── factory.py
│   │   └── lightning_module.py
│   ├── selection/
│   │   ├── model_selection.py
│   │   └── select_best_model.py
│   ├── training/
│   │   ├── finetuning.py
│   │   ├── mlflow_utils.py
│   │   ├── plots.py
│   │   └── train.py
│   └── utils/
│       └── git.py
├── configs/
│   ├── data/
│   ├── evaluation/
│   ├── inference/
│   ├── logging/
│   ├── model/
│   ├── optimizer/
│   ├── selection/
│   └── trainer/
├── docs/
│   └── colab_training.md
├── scripts/
│   ├── colab_setup.sh
│   ├── run_colab_mlflow_server.sh
│   └── run_mlflow_server.sh
├── tests/
├── pyproject.toml
├── uv.lock
└── README.md
```

Generated runtime outputs are intentionally not committed:

```text
data/raw/cards/
artifacts/checkpoints/
plots/
reports/evaluation/
reports/model_comparison/
mlruns/
mlflow.db
```

---

## Environment

The project uses `uv` for dependency management.

Local development was configured with Python 3.13. In Colab, use the Python selector `3.13` because the exact local patch version may not be available as a managed `uv` Python build.

Install dependencies locally:

```bash
uv sync --dev
```

Run tests:

```bash
uv run pytest
```

Run pre-commit checks:

```bash
uv run pre-commit run --all-files
```

---

## Configuration

The project uses Hydra configs.

Main config:

```text
configs/config.yaml
```

Important config groups:

```text
configs/data/cards.yaml
configs/model/baseline_cnn.yaml
configs/model/efficientnet_b0.yaml
configs/optimizer/adam.yaml
configs/optimizer/adamw.yaml
configs/trainer/cpu.yaml
configs/trainer/gpu.yaml
configs/trainer/colab.yaml
configs/logging/mlflow.yaml
configs/logging/colab.yaml
configs/evaluation/default.yaml
configs/selection/default.yaml
```

Default config composition:

```yaml
defaults:
  - data: cards
  - model: baseline_cnn
  - optimizer: adam
  - trainer: cpu
  - logging: mlflow
  - inference: local
  - evaluation: default
  - selection: default
  - _self_
```

---

## Data

### Download dataset

```bash
uv run python -m card_recognizer.data.download
```

If Kaggle credentials are required, place `kaggle.json` in the standard Kaggle CLI location:

```text
~/.kaggle/kaggle.json
```

Do not commit `kaggle.json`.

### Validate dataset

```bash
uv run python -m card_recognizer.data.validate
```

Validation checks:

* expected train/valid/test split structure
* number of classes
* image readability
* class mapping consistency
* deterministic `class_to_idx` artifact

The class mapping is saved to:

```text
artifacts/class_to_idx.json
```

---

## DVC

The project uses DVC for data/artifact tracking.

Example local setup:

```bash
uv run dvc init
mkdir -p ../dvc-storage/playing-card-recognizer/data
mkdir -p ../dvc-storage/playing-card-recognizer/models

uv run dvc remote add -d data-remote ../dvc-storage/playing-card-recognizer/data
uv run dvc remote add models-remote ../dvc-storage/playing-card-recognizer/models
```

Track dataset and class mapping:

```bash
uv run dvc add data/raw/cards
uv run dvc add artifacts/class_to_idx.json
uv run dvc push -r data-remote
```

Pull tracked data:

```bash
uv run dvc pull
```

---

## Inspect DataModule

```bash
uv run python -m card_recognizer.data.inspect \
  data.batch_size=8 \
  data.num_workers=0
```

Expected image tensor shape:

```text
[batch_size, 3, 224, 224]
```

---

## Models

### Baseline CNN

The baseline model is a small convolutional neural network trained from scratch.

Config:

```text
configs/model/baseline_cnn.yaml
```

It is used as a lower-bound reference model.

### EfficientNet-B0

EfficientNet-B0 is used as the main transfer learning model.

Config:

```text
configs/model/efficientnet_b0.yaml
```

Training strategy:

```yaml
training_strategy:
  freeze_backbone_epochs: 5
  fine_tune_epochs: 20
```

The EfficientNet training run has two phases:

```text
epochs 0-4:
  backbone is frozen
  only classifier head is trained

epochs 5+:
  backbone is unfrozen
  the full model is fine-tuned
```

The transition is implemented by `BackboneUnfreezingCallback`.

---

## MLflow

Start a local MLflow server:

```bash
bash scripts/run_mlflow_server.sh
```

Default local server:

```text
http://127.0.0.1:8080
```

MLflow logs:

* hyperparameters
* Hydra config values
* git commit hash
* git dirty state
* train metrics
* validation metrics
* test metrics
* plots
* checkpoints
* evaluation reports

---

## Local training

### Baseline CNN smoke training

```bash
uv run python -m card_recognizer.training.train \
  model=baseline_cnn \
  optimizer=adam \
  trainer=cpu \
  logging.enabled=false \
  data.batch_size=8 \
  data.num_workers=0 \
  trainer.max_epochs=1 \
  trainer.limit_train_batches=5 \
  trainer.limit_val_batches=2 \
  trainer.limit_test_batches=2
```

### Baseline CNN GPU training

```bash
uv run python -m card_recognizer.training.train \
  model=baseline_cnn \
  optimizer=adam \
  trainer=gpu \
  data.batch_size=64 \
  data.num_workers=4 \
  trainer.max_epochs=8 \
  trainer.early_stopping.patience=3
```

### EfficientNet-B0 GPU training

```bash
uv run python -m card_recognizer.training.train \
  model=efficientnet_b0 \
  optimizer=adamw \
  trainer=gpu \
  data.batch_size=32 \
  data.num_workers=4 \
  trainer.max_epochs=25
```

A longer quality-oriented EfficientNet run:

```bash
uv run python -m card_recognizer.training.train \
  model=efficientnet_b0 \
  optimizer=adamw \
  trainer=gpu \
  data.batch_size=32 \
  data.num_workers=4 \
  trainer.max_epochs=40 \
  trainer.early_stopping.patience=10 \
  optimizer.head_lr=0.0005 \
  optimizer.backbone_lr=0.00005 \
  optimizer.weight_decay=0.0001
```

---

## Google Colab workflow

Colab is used as a GPU compute backend while keeping the project as a normal Python repository.

Detailed Colab instructions are in:

```text
docs/colab_training.md
```

### Colab setup

In Colab:

```bash
%cd /content
!git clone <YOUR_GITHUB_REPO_URL> playing-card-recognizer
%cd /content/playing-card-recognizer
```

Run setup:

```bash
!bash scripts/colab_setup.sh
```

Set matplotlib backend explicitly:

```bash
%env MPLBACKEND=Agg
```

### Start MLflow in Colab

In Colab, port `8080` may be occupied by Jupyter services. The Colab MLflow workflow uses port `5000`.

```bash
!pkill -f "mlflow server" || true
!pkill -f "uvicorn.*mlflow" || true
!pkill -f "huey.*mlflow" || true
!pkill -f "mlflow.server.jobs" || true

!rm -f mlflow_server.log mlflow_server.pid

!bash -lc 'nohup uv run --python 3.13 bash scripts/run_colab_mlflow_server.sh \
  > mlflow_server.log 2>&1 & echo $! > mlflow_server.pid'

!sleep 10
!tail -80 mlflow_server.log
```

Check server:

```bash
!ss -ltnp | grep ':5000' || true
```

Check MLflow API:

```bash
!curl -s http://127.0.0.1:5000/api/2.0/mlflow/experiments/search \
  -H "Content-Type: application/json" \
  -d '{"max_results": 10}'
```

### Colab smoke training

```bash
!uv run --python 3.13 python -m card_recognizer.training.train \
  model=baseline_cnn \
  optimizer=adam \
  trainer=colab \
  logging=colab \
  data.batch_size=16 \
  data.num_workers=2 \
  trainer.max_epochs=1 \
  trainer.limit_train_batches=2 \
  trainer.limit_val_batches=1 \
  trainer.limit_test_batches=1
```

### Colab baseline training

```bash
!uv run --python 3.13 python -m card_recognizer.training.train \
  model=baseline_cnn \
  optimizer=adam \
  trainer=colab \
  logging=colab \
  data.batch_size=64 \
  data.num_workers=4 \
  trainer.max_epochs=8 \
  trainer.early_stopping.patience=3
```

### Colab EfficientNet-B0 training

```bash
!uv run --python 3.13 python -m card_recognizer.training.train \
  model=efficientnet_b0 \
  optimizer=adamw \
  trainer=colab \
  logging=colab \
  data.batch_size=32 \
  data.num_workers=4 \
  trainer.max_epochs=25
```

If CUDA runs out of memory, reduce batch size:

```text
data.batch_size=16
data.batch_size=8
```

---

## Evaluation

Standalone evaluation is the source of final report metrics.

### Evaluate baseline CNN

```bash
uv run python -m card_recognizer.evaluation.evaluate \
  model=baseline_cnn \
  optimizer=adam \
  trainer=gpu \
  data.batch_size=64 \
  data.num_workers=4 \
  evaluation.split=test \
  evaluation.bootstrap.num_samples=1000
```

### Evaluate EfficientNet-B0

```bash
uv run python -m card_recognizer.evaluation.evaluate \
  model=efficientnet_b0 \
  model.pretrained=false \
  optimizer=adamw \
  trainer=gpu \
  data.batch_size=32 \
  data.num_workers=4 \
  evaluation.split=test \
  evaluation.bootstrap.num_samples=1000
```

In Colab, use:

```bash
!uv run --python 3.13 python -m card_recognizer.evaluation.evaluate \
  model=efficientnet_b0 \
  model.pretrained=false \
  optimizer=adamw \
  trainer=colab \
  logging=colab \
  data.batch_size=32 \
  data.num_workers=4 \
  evaluation.split=test \
  evaluation.bootstrap.num_samples=1000
```

Evaluation outputs:

```text
reports/evaluation/<model_name>/
├── bootstrap_confidence_intervals.csv
├── classification_report.csv
├── confusion_matrix.csv
├── predictions.csv
└── summary_metrics.json
```

Evaluation plots:

```text
plots/<model_name>/
├── confusion_matrix.png
├── confusion_matrix_normalized.png
└── worst_classes_by_f1.png
```

---

## Metrics

The project uses three metric scopes.

### Train metrics

Prefix:

```text
train_*
```

Examples:

```text
train_loss
train_accuracy
train_macro_f1
train_macro_precision
train_macro_recall
train_top3_accuracy
```

Train metrics are used to monitor whether the model is learning. They are not used for final model selection.

### Validation metrics

Prefix:

```text
val_*
```

Examples:

```text
val_loss
val_accuracy
val_macro_f1
val_macro_precision
val_macro_recall
val_top3_accuracy
```

Validation metrics are used for:

* early stopping
* checkpoint selection

Current checkpoint monitor:

```yaml
monitor: val_macro_f1
mode: max
```

This means that the saved best checkpoint is selected by maximum validation macro F1, not by minimum validation loss.

A checkpoint name like:

```text
epoch=11-val_macro_f1=0.8768.ckpt
```

means that this checkpoint was selected using validation macro F1 at epoch 11.

### Test / evaluation metrics

Prefix in training logs:

```text
test_*
```

Standalone evaluation output:

```text
reports/evaluation/<model_name>/summary_metrics.json
```

Final reported metrics should come from standalone evaluation reports, not from intermediate training logs.

The standalone evaluation collects predictions for the full split and computes metrics once on the whole dataset.

Important final metrics:

```text
accuracy
macro_precision
macro_recall
macro_f1
weighted_f1
top_k_accuracy
```

Bootstrap confidence intervals are saved separately in:

```text
bootstrap_confidence_intervals.csv
```

Bootstrap is not the source of the point estimates in `summary_metrics.json`. It is used to estimate metric uncertainty.

---

## Latest reference result

Latest EfficientNet-B0 standalone evaluation on the test split:

```json
{
  "num_samples": 265,
  "accuracy": 0.9207547169811321,
  "macro_precision": 0.9358715184186883,
  "macro_recall": 0.920754716981132,
  "macro_f1": 0.9211270909384117,
  "weighted_f1": 0.9211270909384117,
  "top_k_accuracy": 0.9886792302131653
}
```

These metrics were produced by:

```bash
python -m card_recognizer.evaluation.evaluate \
  model=efficientnet_b0 \
  model.pretrained=false \
  optimizer=adamw \
  trainer=colab \
  logging=colab \
  data.batch_size=32 \
  data.num_workers=4 \
  evaluation.split=test \
  evaluation.bootstrap.num_samples=1000
```

Note: validation metrics and test metrics are expected to differ because they are computed on different splits.

---

## Model comparison

Run:

```bash
uv run python -m card_recognizer.selection.select_best_model
```

Colab:

```bash
!uv run --python 3.13 python -m card_recognizer.selection.select_best_model
```

Model comparison config:

```text
configs/selection/default.yaml
```

Default comparison metric:

```yaml
metric: macro_f1
higher_is_better: true
```

The selector reads:

```text
reports/evaluation/<model_name>/summary_metrics.json
```

and compares models by test-set `macro_f1`.

Generated outputs:

```text
reports/model_comparison/
├── best_model.json
├── comparison.csv
└── comparison.md
```

In short:

```text
checkpoints are selected by validation macro F1
final model comparison is performed by test macro F1
```

---

## Saving Colab outputs

Colab runtime storage is temporary. Save generated outputs before disconnecting.

```bash
!tar -czf colab_training_outputs.tar.gz artifacts/checkpoints plots reports mlruns mlflow.db
!ls -lh colab_training_outputs.tar.gz
```

Mount Google Drive:

```python
from google.colab import drive

drive.mount("/content/drive")
```

Copy archive:

```bash
!cp colab_training_outputs.tar.gz /content/drive/MyDrive/
```

---

## Opening Colab MLflow results locally

After downloading `colab_training_outputs.tar.gz`, unpack it in the project root:

```bash
tar -xzf colab_training_outputs.tar.gz
```

Run local MLflow server over the unpacked Colab results:

```bash
uv run mlflow server \
  --host 127.0.0.1 \
  --port 8080 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns
```

Open:

```text
http://127.0.0.1:8080
```

---

## Git workflow

Before committing:

```bash
uv run pre-commit run --all-files
uv run pytest
git status --short
```

Do commit:

```text
source code
configs
tests
docs
DVC metadata
pyproject.toml
uv.lock
README.md
```

Do not commit:

```text
data/raw/cards/
artifacts/checkpoints/
plots/
reports/evaluation/
reports/model_comparison/
mlruns/
mlflow.db
*.ckpt
*.pth
*.pt
*.onnx
*.engine
*.trt
kaggle.json
```

Suggested commit for Colab workflow fixes:

```bash
git add README.md
git add configs/logging/colab.yaml
git add scripts/colab_setup.sh
git add scripts/run_colab_mlflow_server.sh
git add docs/colab_training.md
git commit -m "Document and fix Colab training workflow"
```

---

## Notes and known caveats

* Colab and local `localhost` are different machines. A local MLflow server on the laptop is not visible from the Colab runtime.
* In Colab, MLflow uses `127.0.0.1:5000`, not `127.0.0.1:8080`.
* In local development, MLflow uses `127.0.0.1:8080`.
* Colab should run commands with `uv run --python 3.13`.
* `MPLBACKEND=Agg` should be used for CLI plot generation in Colab.
* Final metrics should be taken from standalone evaluation reports under `reports/evaluation/`.
* The validation and test splits are small, so macro metrics and bootstrap confidence intervals should be interpreted together.
