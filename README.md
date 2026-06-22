# SeqTrainer BenchLab

SeqTrainer BenchLab is a local-first web workbench for DNA sequence benchmark runs.
It wraps the practical capabilities available in the current SeqTrainer main branch:
dataset loading, sequence preprocessing, k-mer and GC features, simple baselines, metrics,
predictions, and reproducibility manifests.

This MVP is intended to be run locally, preferably with Docker. It is not designed as a
shared web service because it has no login system, no per-user isolation, and works with
uploaded biological datasets.

## What Works Now

- Upload CSV, TSV, FASTA, FA, or SBOL/XML files
- Preview and validate sequence datasets
- Select sequence and target columns for tabular files
- Generate GC-content and k-mer features
- Generate optional one-hot features
- Analyze target labels for class counts and imbalance warnings
- Generate a reproducible benchmark-plan JSON and Codex prompt for Colab/HPC workflows
- Run baseline regression models:
  - Linear Regression
  - Random Forest
  - Gradient Boosting
- Show train/test split size, seed, selected columns, preprocessing config, and data cleanup status after each run
- Configure split strategy, validation size, reruns, CV folds, training cycles/epochs, early stopping patience, threshold policy, and class balancing
- For 0/1 labels, report thresholded classification metrics alongside the current-main regression baselines:
  - Accuracy
  - Precision
  - Recall
  - F1
  - MCC
  - Confusion counts
- Export:
  - canonical `run_config.json`
  - metrics
  - predictions
  - run manifest
  - dataset manifest
  - preprocessing config
  - training config
  - benchmark plan JSON
  - dependency lockfile, JSON schema, Dockerfile.repro, and environment.yml

## Benchmark Planning

BenchLab has two modes:

1. Local small-run mode for quick sklearn baselines.
2. Planning/export mode for CNN, DNABERT2, and iPro-MP comparison workflows on Colab or HPC.

The local app is intentionally conservative. If a dataset is larger than 1000 rows, the UI
warns the scientist and the small-run path uses the first 1000 rows by default. Full-size runs
should use the exported JSON plan and Codex prompt to create local, Colab, Docker, or HPC
notebooks.

The exported plan records:

- raw dataset identity and SHA256
- selected sequence and label columns
- fixed split strategy and materialized split-manifest requirement
- threshold strategy, including user/literature input, validation MCC, F1, median, or biological goal
- shared or per-model threshold policy
- class imbalance summary and optional balancing strategy
- reruns/seeds, CV folds, and early stopping patience
- training cycles/epochs for future CNN, DNABERT2, iPro-MP, Docker, Colab, or HPC runners
- CNN, DNABERT2, and iPro-MP as Colab/HPC comparison targets
- required artifacts for future reproducibility

The JSON can be reused later to:

- recreate the same split manifest and seeds
- generate Colab or HPC notebooks from the Codex prompt
- configure Docker/container runs with the same preprocessing and thresholds
- compare future model outputs against the same benchmark contract
- import the plan back into BenchLab or another benchmark tool
- audit exactly why a threshold, balancing strategy, or row cap was chosen

## Future Model Slots

CNN, DNABERT2, and iPro-MP are selectable in the benchmark planning/export section. They are
not executed inside the local FastAPI app yet. The current upstream repository includes
DNABERT2 notebooks and experimental GNN code, but the web app only enables the baseline
sklearn models from the main branch scripts for local execution.

For binary label datasets, such as `label` values of `0` and `1`, BenchLab still trains the
current-main sklearn regression baselines. It then applies a `0.5` prediction threshold to
report classification-style metrics. This keeps the app faithful to the available SeqTrainer
main-branch model code while making promoter/non-promoter results easier to read.

## SeqTrainer Integration

This web app is built around the current upstream project at
https://github.com/SynBioDex/SeqTrainer.

The upstream main branch is research-oriented, with reusable kernels in `src/seqtrainer`
and model workflows in notebooks/scripts. BenchLab uses a local web adapter of those current
capabilities so the application can run consistently on a scientist's machine:

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

## Local Docker Use

Recommended runtime: Docker or Python 3.11.

Build the local image:

```bash
docker build -t seqtrainer-benchlab .
```

Run it locally only:

```bash
docker run --rm -p 127.0.0.1:8000:8000 seqtrainer-benchlab
```

Open:

```text
http://127.0.0.1:8000
```

By default, container storage is temporary. When the container stops, run artifacts inside
the container are removed. If you deliberately want local persistent artifacts, mount a
local storage folder:

```powershell
docker run --rm -p 127.0.0.1:8000:8000 -v ${PWD}\storage:/app/storage seqtrainer-benchlab
```

Do not bind this app to a public interface or deploy it as an open multi-user website. It is
intended for local use because uploaded datasets and predictions can contain sensitive
research data.

## Local Security Model

BenchLab includes local safety guardrails, but it is still not a hosted multi-user product.

- The Docker image runs as a non-root user.
- Uploaded source datasets are deleted after successful benchmark runs by default.
- Runtime data under `storage/` is ignored by git and excluded from Docker builds.
- Dataset and run IDs must be UUIDs, which prevents path traversal through API routes.
- Uploads are limited to 50 MB by default.
- Local quick runs are capped at 1000 rows by default.
- Preprocessing settings are bounded on the backend, including k-mer size and one-hot length.
- Browser write requests are allowed only from local origins unless explicitly configured.
- API responses use basic security headers and no-store caching for `/api/*` responses.

Optional local environment controls:

```text
SEQTRAINER_MAX_UPLOAD_MB=50
SEQTRAINER_MAX_LOCAL_ROWS=1000
SEQTRAINER_MAX_KMER_SIZE=6
SEQTRAINER_MAX_ONE_HOT_LENGTH=1000
SEQTRAINER_ALLOWED_ORIGINS=http://127.0.0.1:8000,http://localhost:8000
```

## Data Format

CSV/TSV files should contain:

- one sequence column
- optional numeric target column for benchmark runs, such as `target`, `label`, `y`, or `expression`
- optional ID column

FASTA/FA files create sequence-only datasets.

SBOL/XML files are parsed for the first `sbol:elements` sequence and available numerical values.

## Reproducibility

Every local run writes artifacts to `storage/runs/<run_id>` when storage is persistent:

- `run_config.json`
- `metrics.json`
- `predictions.csv`
- `run_manifest.json`
- `dataset_manifest.json`
- `preprocessing_config.json`
- `training_config.json`
- `benchmark_plan.json`
- `environment.json`
- `requirements.lock.txt`
- `run_config.schema.json`
- `Dockerfile.repro`
- `environment.yml`

Uploaded source datasets are deleted automatically after a successful benchmark run by default.
The run keeps the dataset manifest, metrics, predictions, and configs, but not the uploaded
raw dataset file. Runtime data is ignored by git.

To keep uploaded datasets during development, set:

```text
DELETE_DATASETS_AFTER_RUN=false
```

To wipe local user data, stop the app and delete:

```text
storage/datasets
storage/runs
```

The current repository has been scrubbed of previous local run artifacts and uploaded
datasets.

## Replaying From JSON

Every benchmark run now writes a canonical `run_config.json`. It combines UI/API inputs,
dataset checksum, model choices, preprocessing, split settings, threshold policy, dependency
snapshot, safe environment variables, git metadata, timestamp, and CPU/GPU/CUDA information.

Validate and dry-run a saved config:

```powershell
python -m app.reproducibility.run_from_config --config path\to\run_config.json --dry-run
```

Export the JSON Schema:

```powershell
python -m app.reproducibility.export_schema --output schemas\run_config.schema.json
```

Example config:

```text
examples/reproducibility/run_config.example.json
```

More detail:

```text
docs/reproducible_runs.md
```

## Email Results

The Results screen can prepare a local email summary and export link.

For optional SMTP delivery on your own machine, configure these environment variables:

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
stable local tool contract that can later call richer SeqTrainer model backends.

This app does not include authentication, accounts, or multi-user isolation. Keep it local
unless those features are added.
