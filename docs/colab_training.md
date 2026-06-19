# Colab training workflow

This document describes how to run long GPU training and evaluation jobs in Google Colab while keeping the repository as a normal production-style Python project.

Colab is used only as a compute backend. The source code, configs, tests, and project structure remain in the GitHub repository.

## Why Colab is used

Local machine:

- source code development;
- unit tests;
- pre-commit checks;
- quick smoke training;
- quick smoke evaluation;
- debugging.

Google Colab:

- longer GPU training;
- longer EfficientNet-B0 experiments;
- evaluation with bootstrap confidence intervals;
- model comparison;
- best checkpoint generation before ONNX/TensorRT export.

The project should not become notebook-based. The notebook or Colab cells should only clone the repository and run the same package commands that are used locally.

## 1. Enable GPU runtime

In Colab:

```text
Runtime -> Change runtime type -> Hardware accelerator -> GPU
```

Then verify the GPU:

```bash
!nvidia-smi
```

If no GPU is shown, reconnect to a GPU runtime or try again later.

## 2. Clone the repository

Replace the URL with the project repository URL:

```bash
%cd /content
!git clone <YOUR_GITHUB_REPO_URL> playing-card-recognizer
%cd /content/playing-card-recognizer
```

If the repository is private, configure GitHub authentication first.

Recommended workflow:

1. develop locally;
2. commit changes;
3. push to GitHub;
4. open Colab;
5. clone or pull the latest repo;
6. run training/evaluation from the repository code.

## 3. Run setup

```bash
!bash scripts/colab_setup.sh
```

This script:

- installs `uv` if it is not already installed;
- installs the pinned Python version;
- synchronizes dependencies from `uv.lock`;
- checks PyTorch CUDA availability;
- fails early if GPU is not available.

## 4. Configure Kaggle credentials

Option A: use environment variables:

```python
import os

os.environ["KAGGLE_USERNAME"] = "<your-kaggle-username>"
os.environ["KAGGLE_KEY"] = "<your-kaggle-api-key>"
```

Option B: upload `kaggle.json` manually:

```python
from google.colab import files

uploaded = files.upload()
```

Then move it:

```bash
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/kaggle.json
!chmod 600 ~/.kaggle/kaggle.json
```

Do not commit `kaggle.json`.

## 5. Download and validate data

Option A: download from Kaggle:

```bash
!uv run python -m card_recognizer.data.download
!uv run python -m card_recognizer.data.validate
```

Option B: pull from DVC remote, if the remote is configured and accessible from Colab:

```bash
!uv run dvc pull
```

For the current project stage, Kaggle download is usually the simplest Colab path.

## 6. Start MLflow server

Start MLflow in the background:

```bash
!nohup uv run bash scripts/run_mlflow_server.sh > mlflow_server.log 2>&1 &
```

Check that it started:

```bash
!sleep 5
!tail -50 mlflow_server.log
```

Training code will use:

```text
http://127.0.0.1:8080
```

This works because both the MLflow server and the training process run inside the same Colab runtime.

## 7. Run baseline training

```bash
!uv run python -m card_recognizer.training.train \
  model=baseline_cnn \
  optimizer=adam \
  trainer=colab \
  logging=colab \
  data.batch_size=64 \
  data.num_workers=4 \
  trainer.max_epochs=5
```

## 8. Run EfficientNet-B0 training

Start with:

```bash
!uv run python -m card_recognizer.training.train \
  model=efficientnet_b0 \
  optimizer=adamw \
  trainer=colab \
  logging=colab \
  data.batch_size=32 \
  data.num_workers=4 \
  trainer.max_epochs=25
```

If CUDA runs out of memory, decrease batch size:

```text
data.batch_size=16
data.batch_size=8
```

If the assigned Colab GPU has more memory, `data.batch_size=32` may work well. If the GPU is weaker or memory is limited, use `16` or `8`.

## 9. Run evaluation

Baseline:

```bash
!uv run python -m card_recognizer.evaluation.evaluate \
  model=baseline_cnn \
  optimizer=adam \
  trainer=colab \
  logging=colab \
  data.batch_size=64 \
  data.num_workers=4 \
  evaluation.split=test \
  evaluation.bootstrap.num_samples=1000
```

EfficientNet-B0:

```bash
!uv run python -m card_recognizer.evaluation.evaluate \
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

If CUDA runs out of memory during evaluation, reduce `data.batch_size`.

## 10. Run model comparison

```bash
!uv run python -m card_recognizer.selection.select_best_model
```

Generated files:

```text
reports/model_comparison/
├── best_model.json
├── comparison.csv
└── comparison.md
```

## 11. Save results

Important generated outputs:

```text
artifacts/checkpoints/
plots/
reports/
mlruns/
mlflow.db
```

These files are not committed to git.

Recommended options:

1. download them manually from Colab;
2. copy them to Google Drive;
3. push model artifacts to a DVC model remote.

Mount Google Drive:

```python
from google.colab import drive

drive.mount("/content/drive")
```

Create an archive:

```bash
!tar -czf colab_training_outputs.tar.gz artifacts/checkpoints plots reports mlruns mlflow.db
```

Copy to Drive:

```bash
!cp colab_training_outputs.tar.gz /content/drive/MyDrive/
```

## 12. Git rule

Only commit source code, configs, docs, tests, and DVC metadata.

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

## 13. Typical full Colab sequence

```bash
%cd /content
!git clone <YOUR_GITHUB_REPO_URL> playing-card-recognizer
%cd /content/playing-card-recognizer

!bash scripts/colab_setup.sh

# Configure Kaggle credentials before this step.
!uv run python -m card_recognizer.data.download
!uv run python -m card_recognizer.data.validate

!nohup uv run bash scripts/run_mlflow_server.sh > mlflow_server.log 2>&1 &
!sleep 5
!tail -50 mlflow_server.log

!uv run python -m card_recognizer.training.train \
  model=efficientnet_b0 \
  optimizer=adamw \
  trainer=colab \
  logging=colab \
  data.batch_size=32 \
  data.num_workers=4 \
  trainer.max_epochs=25

!uv run python -m card_recognizer.evaluation.evaluate \
  model=efficientnet_b0 \
  model.pretrained=false \
  optimizer=adamw \
  trainer=colab \
  logging=colab \
  data.batch_size=32 \
  data.num_workers=4 \
  evaluation.split=test \
  evaluation.bootstrap.num_samples=1000

!uv run python -m card_recognizer.selection.select_best_model
```

## 14. What to commit after Colab experiments

Usually, do not commit generated Colab outputs.

Commit only source-code changes, configs, docs, tests, and DVC metadata. If the best model checkpoint should become part of the reproducible project state, track it through a DVC model remote rather than committing the binary checkpoint to git.
