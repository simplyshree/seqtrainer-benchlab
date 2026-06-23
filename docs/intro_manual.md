# SeqTrainer BenchLab Quick Start

SeqTrainer BenchLab is a local-first web tool for planning and running small DNA sequence classification benchmarks. It helps scientists upload labeled sequence data, choose preprocessing and benchmark settings, run quick local baseline models, or export a reproducible JSON/Docker bundle for later Colab, HPC, or local replay.

## What You Need

- Python 3.11
- Git
- Optional: Docker Desktop, if you want to use the exported Docker replay files

## Start The App Locally

```powershell
git clone https://github.com/simplyshree/seqtrainer-benchlab.git
cd seqtrainer-benchlab
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open the website:

```text
http://127.0.0.1:8000
```

## Prepare Your Dataset

Use a CSV/TSV file with one DNA sequence column and one label column.

Recommended beginner format:

```csv
sequence,label
ATGCGTACGTAG,1
TTTACCGGATCA,0
GCGCGTATATGC,1
```

The label column can use `0` and `1` for binary classification, such as non-promoter and promoter.

## Basic Website Workflow

1. Choose **Beginner** or **Advanced** mode.
2. Go to **Dataset**.
3. Upload your sequence + label dataset.
4. Review the dataset analysis, including row count, columns, target labels, and class balance.
5. Optional: import a saved `run_config.json` or `benchmark_plan.json` from the JSON card at the top of the Dataset page.
6. Go to **Preprocess** if you want to preview generated features.
7. Go to **Benchmark**.
8. Choose one of two paths:
   - **Run Small Local Benchmark**: trains the easy local baseline models and shows metrics.
   - **Export Run Config Without Training**: skips model training and creates a reproducible JSON/Docker bundle.
9. Go to **Results**.
10. Download the export bundle.

## Importing A Saved JSON

The JSON import is shown near the top of the Dataset page.

Important order:

1. Upload the matching dataset first.
2. Click **Import JSON**.
3. Select `run_config.json` or `benchmark_plan.json`.
4. BenchLab fills in preprocessing, split, threshold, balancing, and model settings.
5. Review the settings and run or export again.

The JSON does not contain the raw dataset. It stores the settings and dataset checksum, so the original dataset must be available again for full replay.

## What The Export Bundle Contains

Each completed run or plan-only export can include:

```text
run_config.json
requirements.lock.txt
Dockerfile.repro
environment.yml
metrics.json
predictions.csv
run_manifest.json
dataset_manifest.json
preprocessing_config.json
training_config.json
benchmark_plan.json
environment.json
run_config.schema.json
```

For **Export Run Config Without Training**, metrics and predictions may be empty because no model was trained.

## How To Check A Run Config Later

```powershell
.\.venv\Scripts\activate
python -m app.replay --config path\to\run_config.json --dry-run
```

This validates the JSON and reports:

- selected models
- preprocessing settings
- split and seed settings
- dependency summary
- dataset availability
- checksum status

## Docker Replay

From an exported run folder:

```powershell
docker build -f Dockerfile.repro -t seqtrainer-benchlab-repro .
docker run --rm seqtrainer-benchlab-repro
```

Docker improves environment reproducibility, but exact GPU driver behavior is not guaranteed.

## Current Model Scope

The app currently supports easy local baseline models from the BenchLab/SeqTrainer-inspired workflow:

- Linear regression baseline
- Random forest
- Gradient boosting

CNN, DNABERT2, and iPro-MP are included as planned comparison targets in exported configs for future Colab/HPC workflows, but they are not fully automatic web runners yet.

## Best Practice For Sharing

Send collaborators:

- the raw dataset, if allowed
- the export ZIP
- `run_config.json`
- this manual

Then they can upload the dataset, import the JSON, and reproduce or modify the benchmark settings without guessing the original choices.
