# Colab training workflow

This document describes how to run longer GPU training and evaluation jobs in Google Colab while keeping the repository as a normal production-style Python project.

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

The project should not become notebook-based. Colab cells should only clone the repository and run the same package commands that are used locally.

## 1. Enable GPU runtime

In Colab:

```text
Runtime -> Change runtime type -> Hardware accelerator -> GPU
```

Verify the assigned GPU:

```bash
!nvidia-smi
```

## 2. Clone the repository

Replace the URL with the project repository URL:

```bash
%cd /content
!git clone <YOUR_GITHUB_REPO_URL> playing-card-recognizer
%cd /content/playing-card-recognizer
```

Recommended workflow:

1. develop locally;
2. commit changes;
3. push to GitHub;
4. open Colab;
5. clone or pull the latest repository;
6. run training and evaluation from the repository code.

## 3. Run setup

```bash
!bash scripts/colab_setup.sh
```

The setup script:

- installs `uv` if needed;
- installs an available Python 3.13 build;
- synchronizes dependencies from `uv.lock`;
- sets `MPLBACKEND=Agg`;
- checks PyTorch CUDA availability;
- fails early if GPU is not available.

The local project may pin a specific patch version such as Python 3.13.14. In Colab, `uv` may not provide that exact managed Python build. Therefore, the Colab setup uses the Python selector `3.13` and runs commands with `uv run --python 3.13`.

## 4. Configure matplotlib backend

Colab/Jupyter may set an inline matplotlib backend that is invalid inside the project uv environment. For CLI training and evaluation commands, use the non-interactive Agg backend:

```bash
%env MPLBACKEND=Agg
```

The setup script also exports `MPLBACKEND=Agg`, but setting it in the notebook makes the following cells explicit and robust.

## 5. Download and validate data

If Kaggle access works without manually uploading `kaggle.json`, skip the credentials step and run the download command directly.

```bash
!uv run --python 3.13 python -m card_recognizer.data.download
!uv run --python 3.13 python -m card_recognizer.data.validate
```

If Kaggle credentials are required, upload `kaggle.json` manually:

```python
from google.colab import files

uploaded = files.upload()
```

Then move it to the default Kaggle CLI location:

```bash
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/kaggle.json
!chmod 600 ~/.kaggle/kaggle.json
```

Do not commit `kaggle.json`.

## 6. Inspect DataModule

```bash
!uv run --python 3.13 python -m card_recognizer.data.inspect \
  data.batch_size=8 \
  data.num_workers=2
```

## 7. Start MLflow server

In Colab, port 8080 may be occupied by Jupyter Server. Use port 5000 for MLflow.

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

Check that MLflow listens on port 5000:

```bash
!ss -ltnp | grep ':5000' || true
```

Check the MLflow API:

```bash
!curl -s http://127.0.0.1:5000/api/2.0/mlflow/experiments/search \
  -H "Content-Type: application/json" \
  -d '{"max_results": 10}'
```

If the response is JSON, the MLflow server is working.

## 8. Smoke training

Run a short training job before long experiments:

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

The `logging=colab` config uses:

```text
http://127.0.0.1:5000
```

## 9. Baseline CNN training

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

Baseline CNN is a lower-bound reference model. It is not expected to outperform the pretrained EfficientNet-B0 model.

## 10. EfficientNet-B0 training

Start with:

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

The EfficientNet-B0 training strategy is configured as one run with two phases:

```text
epochs 0-4: backbone frozen, classifier head training
epochs 5+: backbone unfrozen, full fine-tuning
```

If CUDA runs out of memory, decrease batch size:

```text
data.batch_size=16
data.batch_size=8
```

For a longer quality-oriented run, use:

```bash
!uv run --python 3.13 python -m card_recognizer.training.train \
  model=efficientnet_b0 \
  optimizer=adamw \
  trainer=colab \
  logging=colab \
  data.batch_size=32 \
  data.num_workers=4 \
  trainer.max_epochs=40 \
  trainer.early_stopping.patience=10 \
  optimizer.head_lr=0.0005 \
  optimizer.backbone_lr=0.00005 \
  optimizer.weight_decay=0.0001
```

## 11. Evaluation

Baseline:

```bash
!uv run --python 3.13 python -m card_recognizer.evaluation.evaluate \
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

`model.pretrained=false` avoids downloading pretrained weights during evaluation. The trained weights are loaded from the checkpoint.

## 12. Model comparison

```bash
!uv run --python 3.13 python -m card_recognizer.selection.select_best_model
```

Generated files:

```text
reports/model_comparison/
├── best_model.json
├── comparison.csv
└── comparison.md
```

Default comparison metric:

```text
macro_f1
```

## 13. Save Colab outputs

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

Copy archive to Drive:

```bash
!cp colab_training_outputs.tar.gz /content/drive/MyDrive/
```

## 14. Open Colab MLflow results locally

After downloading `colab_training_outputs.tar.gz`, unpack it in the project root:

```bash
tar -xzf colab_training_outputs.tar.gz
```

Then run a local MLflow server over the unpacked Colab results:

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

## 15. Git rule

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
