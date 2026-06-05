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
- Export:
  - metrics
  - predictions
  - run manifest
  - dataset manifest
  - preprocessing config
  - training config

## Future Model Slots

These are shown as planned integrations, not enabled in the MVP:

- DNABERT2
- CNN sequence model
- iPro-MP
- GNN over SBOL graphs

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

Docker:

```bash
docker build -t seqtrainer-benchlab .
docker run -p 8000:8000 seqtrainer-benchlab
```

## Data Format

CSV/TSV files should contain:

- one sequence column
- optional target column for benchmark runs
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

## Notes

This branch deliberately avoids notebooks and hardcoded research paths. The first goal is a
stable web service contract that can later call richer SeqTrainer model backends.
