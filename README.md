# SeqTrainer BenchLab

SeqTrainer BenchLab is a lightweight web workbench for DNA sequence benchmark runs.
It wraps the practical capabilities available in the current SeqTrainer main branch:
dataset loading, sequence preprocessing, k-mer and GC features, simple baselines, metrics,
predictions, and reproducibility manifests.

This MVP is intentionally deployable as a single FastAPI service with static frontend assets.

## What Works Now

- Upload CSV, TSV, FASTA, FA, or SBOL/XML files
- Preview and validate sequence datasets
- Select sequence and target columns for tabular files
- Generate GC-content and k-mer features
- Generate optional one-hot features
- Run baseline regression models:
  - Linear Regression
  - Random Forest
  - Gradient Boosting
- Show train/test split size, seed, selected columns, preprocessing config, and data cleanup status after each run
- For 0/1 labels, report thresholded classification metrics alongside the current-main regression baselines:
  - Accuracy
  - Precision
  - Recall
  - F1
  - Confusion counts
- Export:
  - metrics
  - predictions
  - run manifest
  - dataset manifest
  - preprocessing config
  - training config

## Future Model Slots

These are not selectable in the web app until the upstream project exposes stable APIs.
The current upstream repository includes DNABERT2 notebooks and experimental GNN code, but
the web app only enables the baseline sklearn models from the main branch scripts.

For binary label datasets, such as `label` values of `0` and `1`, BenchLab still trains the
current-main sklearn regression baselines. It then applies a `0.5` prediction threshold to
report classification-style metrics. This keeps the app faithful to the available SeqTrainer
main-branch model code while making promoter/non-promoter results easier to read.

## SeqTrainer Integration

This web app is built around the current upstream project at
https://github.com/SynBioDex/SeqTrainer.

The upstream main branch is research-oriented, with reusable kernels in `src/seqtrainer`
and model workflows in notebooks/scripts. BenchLab uses a web-safe adapter of those current
capabilities so the application can run on modern Python web hosts:

- SBOL/XML sequence extraction mirrors `dataset_builder.get_sequence_from_sbol`
- numerical SBOL target discovery mirrors `dataset_builder.get_y_label` behavior
- GC, k-mer, padding, and one-hot feature generation mirror `preprocessing.py`
- baseline model choices mirror `hpc/sklearn_tuning.py`

DNABERT2 and GNN support are shown as planned because the current upstream implementation is
not yet packaged as stable API endpoints.

## Quick Start

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

## Deployment

This repo can be hosted as a Python web service on Render, Railway, Fly.io, Azure App Service,
or any Docker-capable host.

Recommended runtime: Python 3.11.

Render:

1. Push this repository to GitHub.
2. In Render, create a new Blueprint or Web Service from the GitHub repository.
3. Use:
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - Runtime: Python 3.11
4. Add optional email environment variables if SMTP delivery is needed.
5. For production, add persistent storage or object storage for run artifacts. The app deletes
   uploaded source datasets after each successful benchmark by default, but exported run
   artifacts still live under `storage/runs`.

Docker:

```bash
docker build -t seqtrainer-benchlab .
docker run -p 8000:8000 seqtrainer-benchlab
```

## Data Format

CSV/TSV files should contain:

- one sequence column
- optional numeric target column for benchmark runs, such as `target`, `label`, `y`, or `expression`
- optional ID column

FASTA/FA files create sequence-only datasets.

SBOL/XML files are parsed for the first `sbol:elements` sequence and available numerical values.

## Reproducibility

Every run writes artifacts to `storage/runs/<run_id>`:

- `metrics.json`
- `predictions.csv`
- `run_manifest.json`
- `dataset_manifest.json`
- `preprocessing_config.json`
- `training_config.json`

Uploaded source datasets are deleted automatically after a successful benchmark run by default.
The run keeps the dataset manifest, metrics, predictions, and configs, but not the uploaded
raw dataset file. To keep uploaded datasets during development, set:

```text
DELETE_DATASETS_AFTER_RUN=false
```

## Email Results

The Results screen can email a run summary and export link.

For hosted SMTP delivery, configure these environment variables:

```text
SMTP_HOST
SMTP_PORT
SMTP_USERNAME
SMTP_PASSWORD
SMTP_FROM
SMTP_TLS=true
```

If SMTP is not configured, the app opens a prefilled email draft using the user's mail app.

## Notes

This branch deliberately avoids notebooks and hardcoded research paths. The first goal is a
stable web service contract that can later call richer SeqTrainer model backends.
