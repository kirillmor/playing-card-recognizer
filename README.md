# Playing Card Recognizer

`playing-card-recognizer` — это учебный MLOps-проект для распознавания игральных карт по изображению. Проект решает задачу многоклассовой классификации изображений: на вход подаётся одно изображение карты, а модель возвращает один из 53 классов и оценку уверенности.

Проект оформлен не как notebook, а как воспроизводимый Python-пакет с конфигурациями, управлением данными, обучением, логированием экспериментов, оценкой качества, экспортом модели и inference-serving контуром.

## Содержание

* [1. Краткое описание проекта](#1-краткое-описание-проекта)
* [2. Текущий статус](#2-текущий-статус)
* [3. Постановка задачи](#3-постановка-задачи)
* [4. Данные](#4-данные)
* [5. Метрики и валидация](#5-метрики-и-валидация)
* [6. Структура репозитория](#6-структура-репозитория)
* [7. Setup](#7-setup)
* [8. Конфигурации Hydra](#8-конфигурации-hydra)
* [9. DVC: данные и модельные артефакты](#9-dvc-данные-и-модельные-артефакты)
* [10. Data workflow](#10-data-workflow)
* [11. MLflow](#11-mlflow)
* [12. Train](#12-train)
* [13. Evaluation](#13-evaluation)
* [14. Model selection](#14-model-selection)
* [15. Production preparation](#15-production-preparation)
* [16. Infer](#16-infer)
* [17. Google Colab workflow](#17-google-colab-workflow)
* [18. Краткая карта основных команд](#18-краткая-карта-основных-команд)


## 1. Краткое описание проекта

Цель проекта — построить воспроизводимый MLOps-пайплайн для классификации изображений игральных карт.

Пайплайн включает:

* загрузку датасета из Kaggle;
* валидацию структуры данных;
* подготовку `class_to_idx` mapping;
* DVC-версионирование данных и модельных артефактов;
* обучение baseline CNN;
* fine-tuning EfficientNet-B0;
* логирование экспериментов в MLflow;
* standalone evaluation с отчётами и графиками;
* выбор лучшей модели;
* экспорт модели в ONNX;
* валидацию ONNX через ONNX Runtime;
* TensorRT export CLI через `trtexec`;
* подготовку Triton Inference Server model repository;
* inference через Triton HTTP client.

Главная модель проекта — `EfficientNet-B0`, дообученная под 53 класса игральных карт.

## 2. Текущий статус

Реализовано:

* Python-пакет `card_recognizer`;
* управление зависимостями через `uv`;
* `pyproject.toml` и `uv.lock`;
* pre-commit hooks;
* Ruff linting/formatting;
* PyTorch Lightning training pipeline;
* baseline CNN;
* EfficientNet-B0 transfer learning;
* двухфазное обучение EfficientNet-B0:

  * сначала обучение классификационной головы при замороженном backbone;
  * затем fine-tuning всей модели;
* Hydra config system;
* DVC data remote и DVC model remote;
* Kaggle dataset download utility;
* dataset validation utility;
* deterministic `class_to_idx.json` artifact;
* MLflow experiment tracking;
* train/validation/test metrics;
* evaluation reports;
* confusion matrix plots;
* bootstrap confidence intervals;
* model comparison;
* ONNX export;
* ONNX Runtime validation;
* TensorRT export command через Python CLI;
* TensorRT benchmark command через Python CLI;
* Triton model repository;
* Triton `config.pbtxt` для ONNX и TensorRT моделей;
* Triton HTTP inference client;
* локальный и Google Colab workflow.

## 3. Постановка задачи

### 3.1. Тип задачи

Проект решает задачу компьютерного зрения: классификация изображения одной игральной карты.

Формально:

```text
input:  одно RGB-изображение игральной карты
output: класс карты из фиксированного набора 53 классов
```

Это supervised multiclass image classification.

### 3.2. Входные данные

На вход системе подаётся одно изображение в одном из стандартных форматов:

```text
.jpg
.jpeg
.png
```

Ожидается, что на изображении находится одна игральная карта.

В inference-пайплайне изображение проходит следующую обработку:

1. чтение файла;
2. преобразование в RGB;
3. resize до `224x224`;
4. нормализация ImageNet mean/std;
5. преобразование в формат `NCHW`;
6. добавление batch dimension;
7. отправка в модель или Triton Inference Server.

### 3.3. Выходные данные

На выходе система возвращает структурированный результат:

```json
{
  "predicted_class": "ace of clubs",
  "predicted_class_index": 0,
  "confidence": 0.9866,
  "top_k": [
    {
      "class_index": 0,
      "class_name": "ace of clubs",
      "confidence": 0.9866
    }
  ],
  "model_name": "card_recognizer_onnx",
  "image_path": "data/raw/cards/test/ace of clubs/1.jpg"
}
```

Основной результат — `predicted_class`. Дополнительно возвращаются индекс класса, confidence и top-k predictions.

### 3.4. Практическая мотивация

Такой сервис может использоваться как отдельное приложение для распознавания карточных изображений или как компонент более сложной системы анализа визуальных данных. В рамках курса главный акцент сделан не только на качестве модели, но и на воспроизводимости всего жизненного цикла ML-модели: от данных до production-style inference.

## 4. Данные

### 4.1. Источник данных

Используется Kaggle dataset:

```text
gpiosenka/cards-image-datasetclassification
```

Страница датасета:

```text
https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification
```

### 4.2. Характеристики датасета

| Свойство           |          Значение |
| ------------------ | ----------------: |
| Тип данных         |       изображения |
| Формат изображений |               JPG |
| Размер изображения |       `224x224x3` |
| Количество классов |                53 |
| Train              | 7 624 изображения |
| Validation         |   265 изображений |
| Test               |   265 изображений |
| Общий размер       |      около 150 MB |

### 4.3. Структура данных

После скачивания данные имеют структуру:

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

Каждая подпапка внутри `train`, `valid`, `test` соответствует одному классу.

### 4.4. Особенности данных

В датасете уже есть исходное разбиение на `train`, `valid`, `test`. В проекте оно сохраняется без пересборки, чтобы эксперименты были воспроизводимыми и не зависели от дополнительной логики split-generation.

Важно учитывать ограничение: validation и test splits маленькие — всего 265 изображений на 53 класса, то есть примерно по 5 изображений на класс. Поэтому итоговые метрики могут иметь повышенную дисперсию. Для этого в проекте используются macro-метрики и bootstrap confidence intervals.

## 5. Метрики и валидация

### 5.1. Основные метрики

В проекте используются:

* `accuracy`;
* `macro_precision`;
* `macro_recall`;
* `macro_f1`;
* `weighted_f1`;
* `top_k_accuracy`, обычно top-3 accuracy.

### 5.2. Почему важен Macro F1

Задача содержит 53 класса. Даже если дисбаланс не экстремальный, важно контролировать качество не только в среднем по объектам, но и по классам. `macro_f1` усредняет F1-score по классам и поэтому лучше показывает, насколько устойчиво модель работает на разных картах.

### 5.3. Train / validation / test metrics

В проекте есть три уровня метрик.

Train metrics:

```text
train_loss
train_accuracy
train_macro_f1
train_macro_precision
train_macro_recall
train_top3_accuracy
```

Validation metrics:

```text
val_loss
val_accuracy
val_macro_f1
val_macro_precision
val_macro_recall
val_top3_accuracy
```

Test/evaluation metrics:

```text
accuracy
macro_precision
macro_recall
macro_f1
weighted_f1
top_k_accuracy
```

### 5.4. Checkpoint selection

Лучший checkpoint выбирается по validation macro F1:

```yaml
monitor: val_macro_f1
mode: max
```

Это значит, что checkpoint выбирается не по минимальному `val_loss`, а по максимальному `val_macro_f1`.

Пример имени checkpoint:

```text
epoch=11-val_macro_f1=0.8768.ckpt
```

### 5.5. Финальная оценка модели

Финальные метрики нужно брать из standalone evaluation report, а не из промежуточных training logs.

Файл с итоговыми метриками:

```text
reports/evaluation/<model_name>/summary_metrics.json
```

Последний референсный результат EfficientNet-B0 на test split:

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

## 6. Структура репозитория

Актуальная структура проекта:

```text
playing-card-recognizer/
├── .dvc/
│   └── config
├── .dvcignore
├── .gitignore
├── .pre-commit-config.yaml
├── .python-version
├── README.md
├── pyproject.toml
├── uv.lock
├── artifacts/
│   ├── class_to_idx.json.dvc
│   ├── checkpoints/
│   │   ├── .gitignore
│   │   └── efficientnet_b0.dvc
│   └── exports/
│       └── onnx/
│           └── efficientnet_b0/
│               └── model.onnx.dvc
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
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluate.py
│   │   ├── metrics.py
│   │   └── plots.py
│   ├── export/
│   │   ├── __init__.py
│   │   ├── benchmark_tensorrt.py
│   │   ├── export_onnx.py
│   │   ├── export_tensorrt.py
│   │   └── validate_onnx.py
│   ├── inference/
│   │   ├── __init__.py
│   │   └── checkpoint.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline_cnn.py
│   │   ├── efficientnet.py
│   │   ├── factory.py
│   │   └── lightning_module.py
│   ├── selection/
│   │   ├── __init__.py
│   │   ├── model_selection.py
│   │   └── select_best_model.py
│   ├── serving/
│   │   ├── __init__.py
│   │   ├── triton_client.py
│   │   └── triton_repository.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── finetuning.py
│   │   ├── mlflow_utils.py
│   │   ├── plots.py
│   │   └── train.py
│   └── utils/
│       ├── __init__.py
│       └── git.py
├── configs/
│   ├── config.yaml
│   ├── data/
│   │   └── cards.yaml
│   ├── evaluation/
│   │   └── default.yaml
│   ├── export/
│   │   ├── onnx.yaml
│   │   └── tensorrt.yaml
│   ├── inference/
│   │   └── local.yaml
│   ├── logging/
│   │   ├── colab.yaml
│   │   └── mlflow.yaml
│   ├── model/
│   │   ├── baseline_cnn.yaml
│   │   └── efficientnet_b0.yaml
│   ├── optimizer/
│   │   ├── adam.yaml
│   │   └── adamw.yaml
│   ├── selection/
│   │   └── default.yaml
│   ├── serving/
│   │   └── triton.yaml
│   └── trainer/
│       ├── colab.yaml
│       ├── cpu.yaml
│       └── gpu.yaml
├── data/
│   └── raw/
│       └── cards.dvc
├── deployment/
│   └── triton_model_repository/
│       ├── card_recognizer_onnx/
│       │   ├── 1/
│       │   │   └── .gitkeep
│       │   └── config.pbtxt
│       └── card_recognizer_tensorrt/
│           ├── 1/
│           │   └── .gitkeep
│           └── config.pbtxt
├── plots/
│   └── .gitkeep
├── reports/
│   └── .gitkeep
├── scripts/
│   ├── colab_setup.sh
│   ├── run_colab_mlflow_server.sh
│   └── run_triton_server.sh
└── tests/
    ├── __init__.py
    └── ...
```

### 6.1. Что не хранится в Git

Не коммитятся runtime artifacts:

```text
data/raw/cards/
artifacts/checkpoints/**/*.ckpt
artifacts/exports/**/*.onnx
artifacts/exports/**/*.plan
artifacts/exports/**/*.engine
artifacts/exports/**/*.trt
plots/
reports/
mlruns/
mlflow.db
kaggle.json
```

Большие данные и модели хранятся через DVC.

## 7. Setup

### 7.1. Требования

Локальная разработка рассчитана на:

```text
Python >=3.13,<3.14
uv
Git
DVC
Docker, если нужно запускать Triton Inference Server
NVIDIA GPU + TensorRT/trtexec, если нужно собирать TensorRT engine
```

Для CPU smoke-тестов GPU не нужен.

### 7.2. Клонирование репозитория

```bash
git clone https://github.com/kirillmor/playing-card-recognizer.git
cd playing-card-recognizer
```

### 7.3. Установка uv

Если `uv` ещё не установлен:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

После установки проверьте:

```bash
uv --version
```

### 7.4. Установка зависимостей

```bash
uv sync --dev
```

Проверка окружения:

```bash
uv run python --version
uv run python -c "import torch; print(torch.__version__)"
```

### 7.5. Установка pre-commit

```bash
uv run pre-commit install
```

Проверка всех хуков:

```bash
uv run pre-commit run --all-files
```

### 7.6. Запуск тестов

```bash
uv run pytest
```

### 7.7. Проверка entrypoint пакета

В проекте есть console entrypoint:

```bash
uv run card-recognizer
```

Основные production/training workflows запускаются через Python modules, описанные ниже.

## 8. Конфигурации Hydra

Проект использует Hydra. Главная точка входа:

```text
configs/config.yaml
```

Основная композиция:

```yaml
defaults:
  - data: cards
  - model: baseline_cnn
  - optimizer: adam
  - trainer: cpu
  - logging: mlflow
  - inference: local
  - evaluation: default
  - export: onnx
  - serving: triton
  - selection: default
  - _self_

project_name: playing-card-recognizer
seed: 42
paths:
  data_dir: data
  artifacts_dir: artifacts
  plots_dir: plots
  reports_dir: reports
```

Главные группы конфигов:

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
configs/export/onnx.yaml
configs/export/tensorrt.yaml
configs/serving/triton.yaml
configs/selection/default.yaml
```

Пример override:

```bash
uv run python -m card_recognizer.training.train \
  model=efficientnet_b0 \
  optimizer=adamw \
  trainer=gpu \
  data.batch_size=32 \
  trainer.max_epochs=25
```

## 9. DVC: данные и модельные артефакты

### 9.1. Зачем нужен DVC

DVC используется для хранения больших файлов, которые нельзя коммитить в Git:

* датасет;
* `class_to_idx.json`;
* checkpoints;
* ONNX-модель;
* TensorRT engine, если он собран.

Git хранит только `.dvc` pointer-файлы, а реальные данные лежат в DVC remote storage.

### 9.2. DVC remotes

В проекте используются два разных remote storage:

```text
data-remote   — для данных и class mapping
models-remote — для checkpoints, ONNX и TensorRT artifacts
```

Пример `.dvc/config`:

```ini
[core]
    remote = data-remote
['remote "data-remote"']
    url = ../../dvc-storage/playing-card-recognizer/data
['remote "models-remote"']
    url = ../../dvc-storage/playing-card-recognizer/models
```

В учебном проекте используется локальное DVC-хранилище. В реальном production-проекте его можно заменить на S3, MinIO, Google Drive или другое объектное хранилище.

### 9.3. Первичная настройка локального DVC storage

Если remotes ещё не настроены:

```bash
mkdir -p ../dvc-storage/playing-card-recognizer/data
mkdir -p ../dvc-storage/playing-card-recognizer/models

uv run dvc remote add -d data-remote ../dvc-storage/playing-card-recognizer/data
uv run dvc remote add models-remote ../dvc-storage/playing-card-recognizer/models
```

Проверка:

```bash
uv run dvc remote list
cat .dvc/config
```

### 9.4. Получение данных через DVC

Если DVC remote доступен:

```bash
uv run dvc pull
```

Можно проверить состояние:

```bash
uv run dvc status
uv run dvc status -r data-remote
uv run dvc status -r models-remote
```

### 9.5. Fallback: скачать данные из Kaggle

Если локальный DVC remote недоступен на новой машине, данные можно скачать из открытого источника:

```bash
uv run python -m card_recognizer.data.download
```

Если Kaggle требует credentials, положите `kaggle.json` в стандартное место:

```text
~/.kaggle/kaggle.json
```

Не коммитьте `kaggle.json`.

### 9.6. Добавление данных в DVC

```bash
uv run dvc add data/raw/cards
uv run dvc add artifacts/class_to_idx.json

uv run dvc push -r data-remote \
  data/raw/cards.dvc \
  artifacts/class_to_idx.json.dvc
```

В Git добавляются только DVC metadata:

```bash
git add data/raw/cards.dvc artifacts/class_to_idx.json.dvc .gitignore
```

### 9.7. Добавление модельных артефактов в DVC

Checkpoint EfficientNet-B0:

```bash
uv run dvc add artifacts/checkpoints/efficientnet_b0
uv run dvc push -r models-remote artifacts/checkpoints/efficientnet_b0.dvc
```

ONNX-модель:

```bash
uv run dvc add artifacts/exports/onnx/efficientnet_b0/model.onnx
uv run dvc push -r models-remote artifacts/exports/onnx/efficientnet_b0/model.onnx.dvc
```

TensorRT engine, если он собран:

```bash
uv run dvc add artifacts/exports/tensorrt/efficientnet_b0/model.plan
uv run dvc push -r models-remote artifacts/exports/tensorrt/efficientnet_b0/model.plan.dvc
```

## 10. Data workflow

### 10.1. Скачивание датасета

```bash
uv run python -m card_recognizer.data.download
```

Результат:

```text
data/raw/cards/
```

### 10.2. Валидация датасета

```bash
uv run python -m card_recognizer.data.validate
```

Проверяется:

* наличие `train`, `valid`, `test`;
* количество классов;
* читаемость изображений;
* соответствие class mapping;
* сохранение `class_to_idx.json`.

Артефакт class mapping:

```text
artifacts/class_to_idx.json
```

### 10.3. Проверка DataModule

```bash
uv run python -m card_recognizer.data.inspect \
  data.batch_size=8 \
  data.num_workers=0
```

Ожидаемая форма батча:

```text
[batch_size, 3, 224, 224]
```

## 11. MLflow

### 11.1. Запуск локального MLflow server

```bash
uv run mlflow server \
  --host 127.0.0.1 \
  --port 8080 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns
```

Локальный адрес:

```text
http://127.0.0.1:8080
```

### 11.2. Что логируется

В MLflow логируются:

* гиперпараметры;
* Hydra config values;
* git commit hash;
* git dirty state;
* train metrics;
* validation metrics;
* test metrics;
* plots;
* checkpoints;
* evaluation reports.

### 11.3. Где лежат MLflow artifacts

Runtime-директории:

```text
mlruns/
mlflow.db
```

Они не коммитятся в Git.

## 12. Train

### 12.1. Baseline CNN

Baseline CNN — простая сверточная сеть, обучаемая с нуля. Она используется как нижняя граница качества.

Config:

```text
configs/model/baseline_cnn.yaml
```

Smoke training на CPU:

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

Полное baseline GPU training:

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

### 12.2. EfficientNet-B0

EfficientNet-B0 — основная модель проекта.

Config:

```text
configs/model/efficientnet_b0.yaml
```

Стратегия обучения:

```yaml
training_strategy:
  freeze_backbone_epochs: 5
  fine_tune_epochs: 20
```

Смысл фаз:

```text
epochs 0-4: backbone frozen, train classifier head only
epochs 5+: backbone unfrozen, fine-tune full model
```

Обучение EfficientNet-B0 на GPU:

```bash
uv run python -m card_recognizer.training.train \
  model=efficientnet_b0 \
  optimizer=adamw \
  trainer=gpu \
  data.batch_size=32 \
  data.num_workers=4 \
  trainer.max_epochs=25
```

Более длинный quality-oriented запуск:

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

### 12.3. Где сохраняются checkpoints

```text
artifacts/checkpoints/<model_name>/
```

Пример:

```text
artifacts/checkpoints/efficientnet_b0/epoch=11-val_macro_f1=0.8768.ckpt
```

Checkpoints не коммитятся в Git. Они отслеживаются через DVC.

## 13. Evaluation

Standalone evaluation — источник финальных метрик для отчёта.

### 13.1. Evaluate baseline CNN

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

### 13.2. Evaluate EfficientNet-B0

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

### 13.3. Evaluation outputs

Standalone evaluation сохраняет табличные отчёты в директорию:

```text
reports/evaluation/<model_name>/
```

Создаются следующие файлы:

```text
reports/evaluation/<model_name>/
├── summary_metrics.json
├── classification_report.csv
├── predictions.csv
├── confusion_matrix.csv
└── bootstrap_confidence_intervals.csv
```

Описание файлов:

| Файл                                 | Что содержит                                                                                             |
| ------------------------------------ | -------------------------------------------------------------------------------------------------------- |
| `summary_metrics.json`               | итоговые point-estimates: accuracy, macro precision, macro recall, macro F1, weighted F1, top-k accuracy |
| `classification_report.csv`          | per-class precision, recall, F1-score и support                                                          |
| `predictions.csv`                    | предсказание для каждого изображения: true class, predicted class, confidence и top-k probabilities      |
| `confusion_matrix.csv`               | confusion matrix в табличном виде                                                                        |
| `bootstrap_confidence_intervals.csv` | bootstrap mean/std и confidence intervals для основных метрик; создаётся, если включён bootstrap         |

Графики и история метрик сохраняются отдельно:

```text
plots/<model_name>/
├── accuracy.png
├── loss.png
├── macro_f1.png
├── top3_accuracy.png
├── confusion_matrix.png
├── confusion_matrix_normalized.png
├── worst_classes_by_f1.png
└── metrics_history.json
```

Описание файлов:

| Файл                              | Что содержит                                                            |
| --------------------------------- | ----------------------------------------------------------------------- |
| `accuracy.png`                    | динамика accuracy по эпохам для train/validation                        |
| `loss.png`                        | динамика loss по эпохам для train/validation                            |
| `macro_f1.png`                    | динамика macro F1 по эпохам для train/validation                        |
| `top3_accuracy.png`               | динамика top-3 accuracy по эпохам для train/validation                  |
| `confusion_matrix.png`            | обычная confusion matrix на выбранном evaluation split                  |
| `confusion_matrix_normalized.png` | normalized confusion matrix на выбранном evaluation split               |
| `worst_classes_by_f1.png`         | классы с худшим F1-score по результатам standalone evaluation           |
| `metrics_history.json`            | история train/validation метрик, сохранённая в машинно-читаемом формате |

Дополнительно другие стадии проекта создают собственные отчёты:

```text
reports/model_comparison/
├── best_model.json
├── comparison.csv
└── comparison.md

reports/export/<model_name>/
└── onnx_validation.json
```

Важно: `reports/` и `plots/` являются runtime outputs. Они нужны для анализа результатов, но generated files обычно не коммитятся в Git. В репозитории остаются только пустые директории с `.gitkeep`.


### 13.4. Bootstrap confidence intervals

Bootstrap используется для оценки неопределённости метрик на маленьком test split.

Point estimates лежат в:

```text
summary_metrics.json
```

Confidence intervals лежат в:

```text
bootstrap_confidence_intervals.csv
```

## 14. Model selection

Model selection сравнивает модели по standalone evaluation reports.

Запуск:

```bash
uv run python -m card_recognizer.selection.select_best_model
```

Config:

```text
configs/selection/default.yaml
```

Default metric:

```yaml
metric: macro_f1
higher_is_better: true
```

Входы:

```text
reports/evaluation/<model_name>/summary_metrics.json
```

Выходы:

```text
reports/model_comparison/
├── best_model.json
├── comparison.csv
└── comparison.md
```

Кратко:

```text
checkpoint selection: validation macro F1
final model comparison: test macro F1
```

## 15. Production preparation

Этот раздел описывает подготовку обученной модели к inference/deployment.

Production-поставка модели состоит из:

```text
artifacts/checkpoints/efficientnet_b0/
artifacts/exports/onnx/efficientnet_b0/model.onnx
artifacts/exports/tensorrt/efficientnet_b0/model.plan
artifacts/class_to_idx.json
deployment/triton_model_repository/
card_recognizer/serving/triton_client.py
configs/serving/triton.yaml
```

`model.plan` появляется только после успешного TensorRT export в окружении с NVIDIA TensorRT и `trtexec`.

### 15.1. ONNX export

ONNX export оформлен как Python CLI command.

```bash
uv run python -m card_recognizer.export.export_onnx \
  model=efficientnet_b0 \
  model.pretrained=false \
  optimizer=adamw \
  trainer=gpu
```

Если нужно явно указать checkpoint:

```bash
uv run python -m card_recognizer.export.export_onnx \
  model=efficientnet_b0 \
  model.pretrained=false \
  optimizer=adamw \
  trainer=gpu \
  'export.checkpoint_path="artifacts/checkpoints/efficientnet_b0/epoch=11-val_macro_f1=0.8768.ckpt"'
```

Выход:

```text
artifacts/exports/onnx/efficientnet_b0/model.onnx
artifacts/exports/onnx/efficientnet_b0/metadata.json
```

ONNX config:

```text
configs/export/onnx.yaml
```

Ключевые параметры:

```yaml
opset_version: 17
input_name: images
output_name: logits
batch_size: 1
dynamic_batch: true
```

### 15.2. ONNX validation

ONNX validation сравнивает PyTorch checkpoint и ONNX Runtime predictions.

```bash
uv run python -m card_recognizer.export.validate_onnx \
  model=efficientnet_b0 \
  model.pretrained=false \
  optimizer=adamw \
  trainer=gpu \
  data.batch_size=32 \
  data.num_workers=4
```

Ожидаемый хороший результат:

```text
top1_agreement = 1.0
pytorch_accuracy == onnx_accuracy
```

Report:

```text
reports/export/efficientnet_b0/onnx_validation.json
```

Референсный validation report:

```json
{
  "model_name": "efficientnet_b0",
  "num_samples": 265,
  "max_abs_diff": 0.03411734104156494,
  "mean_abs_diff": 0.005037371549541817,
  "top1_agreement": 1.0,
  "top3_agreement": 0.9886792452830189,
  "pytorch_accuracy": 0.9207547169811321,
  "onnx_accuracy": 0.9207547169811321
}
```

### 15.3. TensorRT export

TensorRT export оформлен как Python CLI command поверх `trtexec`.

Требования:

```text
NVIDIA GPU
NVIDIA driver
TensorRT
trtexec executable
```

Запуск:

```bash
uv run python -m card_recognizer.export.export_tensorrt \
  export=tensorrt \
  model=efficientnet_b0 \
  model.pretrained=false
```

Config:

```text
configs/export/tensorrt.yaml
```

Ключевые параметры:

```yaml
onnx_path: artifacts/exports/onnx/${model.name}/model.onnx
output_dir: artifacts/exports/tensorrt/${model.name}
engine_filename: model.plan
input_name: images
output_name: logits
min_batch_size: 1
opt_batch_size: 8
max_batch_size: 32
precision: fp16
trtexec_path: trtexec
```

Ожидаемые выходы:

```text
artifacts/exports/tensorrt/efficientnet_b0/model.plan
artifacts/exports/tensorrt/efficientnet_b0/tensorrt_export_report.json
artifacts/exports/tensorrt/efficientnet_b0/trtexec_export.log
```

Важно: TensorRT engine зависит от GPU, TensorRT version и окружения. Поэтому `model.plan` обычно собирается в целевом deployment environment.

### 15.4. TensorRT benchmark

После появления `model.plan` можно запустить benchmark:

```bash
uv run python -m card_recognizer.export.benchmark_tensorrt \
  export=tensorrt \
  model=efficientnet_b0 \
  model.pretrained=false
```

Ожидаемые выходы:

```text
artifacts/exports/tensorrt/efficientnet_b0/tensorrt_benchmark_report.json
artifacts/exports/tensorrt/efficientnet_b0/trtexec_benchmark.log
```

## 16. Infer

В проекте основной production-style inference реализован через Triton Inference Server.

### 16.1. Triton model repository

Triton model repository находится здесь:

```text
deployment/triton_model_repository/
```

Структура:

```text
deployment/triton_model_repository/
├── card_recognizer_onnx/
│   ├── 1/
│   │   └── model.onnx
│   └── config.pbtxt
├── card_recognizer_tensorrt/
│   ├── 1/
│   │   └── model.plan
│   └── config.pbtxt
├── class_to_idx.json
└── repository_manifest.json
```

В Git хранятся только:

```text
config.pbtxt
.gitkeep
```

Сами `model.onnx`, `model.plan`, `class_to_idx.json`, `repository_manifest.json` являются generated/runtime artifacts и не коммитятся.

### 16.2. Сборка Triton repository

```bash
uv run python -m card_recognizer.serving.triton_repository \
  model=efficientnet_b0 \
  model.pretrained=false
```

Команда копирует доступные artifacts в Triton layout:

* ONNX model в `card_recognizer_onnx/1/model.onnx`;
* TensorRT engine в `card_recognizer_tensorrt/1/model.plan`, если он существует;
* `class_to_idx.json` в корень model repository;
* создаёт `repository_manifest.json`.

Если TensorRT engine ещё не собран, ONNX-модель всё равно можно запускать отдельно.

### 16.3. Triton ONNX config

ONNX model config:

```text
deployment/triton_model_repository/card_recognizer_onnx/config.pbtxt
```

Смысл:

```protobuf
name: "card_recognizer_onnx"
platform: "onnxruntime_onnx"
max_batch_size: 32
input [
  {
    name: "images"
    data_type: TYPE_FP32
    dims: [3, 224, 224]
  }
]
output [
  {
    name: "logits"
    data_type: TYPE_FP32
    dims: [53]
  }
]
instance_group [
  {
    kind: KIND_CPU
  }
]
```

### 16.4. Triton TensorRT config

TensorRT model config:

```text
deployment/triton_model_repository/card_recognizer_tensorrt/config.pbtxt
```

Смысл:

```protobuf
name: "card_recognizer_tensorrt"
platform: "tensorrt_plan"
max_batch_size: 32
input [
  {
    name: "images"
    data_type: TYPE_FP32
    dims: [3, 224, 224]
  }
]
output [
  {
    name: "logits"
    data_type: TYPE_FP32
    dims: [53]
  }
]
instance_group [
  {
    kind: KIND_GPU
  }
]
```

### 16.5. Запуск Triton ONNX на CPU

Для локальной проверки без GPU:

```bash
TRITON_ENABLE_GPU=false TRITON_LOAD_MODEL=card_recognizer_onnx \
  bash scripts/run_triton_server.sh
```

Проверка готовности:

```bash
curl -v localhost:8000/v2/health/ready
```

Проверка metadata:

```bash
curl -s localhost:8000/v2/models/card_recognizer_onnx | python3 -m json.tool
```

### 16.6. Запуск Triton TensorRT на GPU

После сборки `model.plan`:

```bash
TRITON_ENABLE_GPU=true TRITON_LOAD_MODEL=card_recognizer_tensorrt \
  bash scripts/run_triton_server.sh
```

### 16.7. Triton HTTP client

Клиент находится здесь:

```text
card_recognizer/serving/triton_client.py
```

Запуск ONNX inference:

```bash
uv run python -m card_recognizer.serving.triton_client \
  model=efficientnet_b0 \
  model.pretrained=false \
  serving.client.model_name=card_recognizer_onnx \
  'serving.client.image_path="data/raw/cards/test/ace of clubs/1.jpg"'
```

Запуск TensorRT inference:

```bash
uv run python -m card_recognizer.serving.triton_client \
  model=efficientnet_b0 \
  model.pretrained=false \
  serving.client.model_name=card_recognizer_tensorrt \
  'serving.client.image_path="data/raw/cards/test/ace of clubs/1.jpg"'
```

### 16.8. Что делает Triton client

Клиент:

1. читает изображение;
2. приводит его к RGB;
3. делает resize до `224x224`;
4. нормализует ImageNet mean/std;
5. преобразует в `NCHW float32`;
6. отправляет HTTP request в Triton;
7. получает logits;
8. применяет stable softmax;
9. формирует top-k predictions;
10. возвращает JSON-like response.

### 16.9. Пример успешного ответа

```json
{
  "predicted_class": "ace of clubs",
  "predicted_class_index": 0,
  "confidence": 0.9866,
  "top_k": [
    {
      "class_index": 0,
      "class_name": "ace of clubs",
      "confidence": 0.9866
    },
    {
      "class_index": 13,
      "class_name": "four of clubs",
      "confidence": 0.0041
    },
    {
      "class_index": 39,
      "class_name": "seven of clubs",
      "confidence": 0.0032
    }
  ],
  "model_name": "card_recognizer_onnx",
  "image_path": "data/raw/cards/test/ace of clubs/1.jpg"
}
```

## 17. Google Colab workflow

Colab используется как GPU runtime, но проект остаётся обычным Python repository.

### 17.1. Clone в Colab

```bash
%cd /content
!git clone https://github.com/kirillmor/playing-card-recognizer.git
%cd /content/playing-card-recognizer
```

### 17.2. Setup в Colab

```bash
!bash scripts/colab_setup.sh
```

Colab-команды лучше запускать через:

```bash
uv run --python 3.13
```

Установить matplotlib backend:

```bash
%env MPLBACKEND=Agg
```

### 17.3. MLflow в Colab

В Colab порт `8080` может быть занят Jupyter-сервисами, поэтому Colab workflow использует порт `5000`.

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

Проверка:

```bash
!ss -ltnp | grep ':5000' || true
```

### 17.4. Colab smoke training

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

### 17.5. Colab EfficientNet-B0 training

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

Если CUDA OOM:

```text
data.batch_size=16
```

или:

```text
data.batch_size=8
```

### 17.6. Сохранение Colab outputs

Colab runtime storage временный. Перед выключением runtime сохраните результаты:

```bash
!tar -czf colab_training_outputs.tar.gz artifacts/checkpoints plots reports mlruns mlflow.db
!ls -lh colab_training_outputs.tar.gz
```

Если нужен Google Drive:

```python
from google.colab import drive

drive.mount("/content/drive")
```

```bash
!cp colab_training_outputs.tar.gz /content/drive/MyDrive/
```

### 17.7. Открытие Colab MLflow локально

После скачивания архива:

```bash
tar -xzf colab_training_outputs.tar.gz
```

Запустить локальный MLflow server:

```bash
uv run mlflow server \
  --host 127.0.0.1 \
  --port 8080 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns
```

Открыть:

```text
http://127.0.0.1:8080
```

## 18. Краткая карта основных команд

```bash
# setup
uv sync --dev
uv run pre-commit install
uv run pytest

# data
uv run dvc pull
uv run python -m card_recognizer.data.download
uv run python -m card_recognizer.data.validate
uv run python -m card_recognizer.data.inspect

```bash
# mlflow
uv run mlflow server \
  --host 127.0.0.1 \
  --port 8080 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns
```


# train
uv run python -m card_recognizer.training.train model=baseline_cnn optimizer=adam trainer=cpu
uv run python -m card_recognizer.training.train model=efficientnet_b0 optimizer=adamw trainer=gpu

# evaluate
uv run python -m card_recognizer.evaluation.evaluate model=efficientnet_b0 model.pretrained=false optimizer=adamw trainer=gpu

# model selection
uv run python -m card_recognizer.selection.select_best_model

# onnx
uv run python -m card_recognizer.export.export_onnx model=efficientnet_b0 model.pretrained=false optimizer=adamw trainer=gpu
uv run python -m card_recognizer.export.validate_onnx model=efficientnet_b0 model.pretrained=false optimizer=adamw trainer=gpu

# tensorrt
uv run python -m card_recognizer.export.export_tensorrt export=tensorrt model=efficientnet_b0 model.pretrained=false
uv run python -m card_recognizer.export.benchmark_tensorrt export=tensorrt model=efficientnet_b0 model.pretrained=false

# triton
uv run python -m card_recognizer.serving.triton_repository model=efficientnet_b0 model.pretrained=false
TRITON_ENABLE_GPU=false TRITON_LOAD_MODEL=card_recognizer_onnx bash scripts/run_triton_server.sh
uv run python -m card_recognizer.serving.triton_client model=efficientnet_b0 model.pretrained=false serving.client.model_name=card_recognizer_onnx 'serving.client.image_path="data/raw/cards/test/ace of clubs/1.jpg"'
```
