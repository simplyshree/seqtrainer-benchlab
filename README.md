# SeqTrainer BenchLab

SeqTrainer BenchLab is a local-first web tool for planning and running small DNA sequence
benchmark experiments. It helps a scientist upload a labeled sequence dataset, inspect the
data, choose preprocessing and benchmark settings, run lightweight baseline models, and export
a reproducible `run_config.json` for later Colab, Docker, or HPC work.

This repository is not yet a full hosted model-training platform. It is a practical BenchLab
MVP: useful for small local checks, reproducibility planning, and producing clear experiment
configuration files that can be reused in larger workflows.

## Current Deliverables

- Local FastAPI web app with beginner and advanced workflow modes.
- Dataset upload and inspection for CSV, TSV, FASTA/FA, and SBOL/XML inputs.
- Tabular benchmark support when the file has one sequence column and one numeric label/target column.
- Dataset summary with row count, detected columns, class counts, and imbalance warnings.
- Small-run benchmark mode capped at 1000 rows by default for local safety.
- Baseline model runs using the practical current SeqTrainer-style sklearn options:
  - Linear Regression
  - Random Forest
  - Gradient Boosting
- DNA preprocessing options:
  - sequence cleanup
  - GC-content features
  - k-mer features, default k-mer size `6`
  - optional one-hot features
- Benchmark settings for:
  - split strategy
  - test and validation size
  - seed
  - reruns
  - CV folds
  - early stopping patience
  - threshold policy
  - class balancing choice
- Results view with key binary classification metrics when labels are `0/1`:
  - Accuracy
  - Precision
  - Recall
  - F1
  - MCC
  - confusion counts
- Exportable artifacts:
  - `run_config.json`
  - `metrics.json`
  - `predictions.csv`
  - `run_manifest.json`
  - `dataset_manifest.json`
  - `preprocessing_config.json`
  - `training_config.json`
  - `benchmark_plan.json`
  - `requirements.lock.txt`
  - `Dockerfile.repro`
  - `environment.yml`
  - `run_config.schema.json`

## What This Tool Is For

BenchLab is best used as a reproducibility and experiment-planning layer around SeqTrainer-style
DNA sequence benchmarks.

Use it to:

- quickly check whether a labeled sequence dataset is usable
- detect simple class imbalance before training
- create a consistent benchmark plan before running heavier models
- run lightweight local baseline models on small datasets
- export a complete experiment configuration for later replay
- hand off the same settings to a notebook, Colab session, Docker container, or HPC job
- compare future model outputs against a recorded benchmark setup

## What This Tool Does Not Do Yet

- It does not train DNABERT2 or iPro-MP inside the web app.
- It does not provide GPU scheduling or HPC job submission.
- It does not store raw datasets in `run_config.json`.
- It does not provide authentication, accounts, or multi-user isolation.
- It should not be deployed as a public open web service without additional security work.
- It does not guarantee exact reproduction of GPU drivers or external model checkpoints.

CNN, DNABERT2, and iPro-MP are treated as planned comparison targets in the benchmark planning
flow. The current local execution path focuses on lightweight baseline models that can run on a
normal laptop.

## How The JSON Fits In

Every run or exported plan can produce a canonical `run_config.json`.

That file records:

- dataset identity and SHA256 checksum
- selected sequence and label columns
- preprocessing settings
- split settings and seed
- threshold policy
- balancing choice
- selected models
- training settings such as reruns, CV folds, cycles, and early stopping
- Python version and package lock information
- Docker base image
- safe environment metadata
- CPU/GPU hardware summary
- replay commands and known limitations

If the same dataset is used again, the checksum helps verify exact replay. If a different
dataset is uploaded, the JSON should be used as a template: BenchLab can reuse the settings,
but metrics and predictions must be recalculated for the new data.

## Recommended Workflow

1. Start BenchLab locally.
2. Upload a dataset with sequence and label columns.
3. Review dataset summary and class balance.
4. Choose beginner mode for simple defaults or advanced mode for split, threshold, CV, and seed settings.
5. Either run a small local benchmark or skip directly to exporting the run configuration.
6. Download `run_config.json` or the full run bundle.
7. Use the JSON later in a local script, notebook, Docker container, Colab, or HPC workflow.

## Data Format

For benchmark runs, CSV/TSV files should contain:

- one DNA sequence column
- one numeric label/target column, for example `label`, `target`, `y`, or `expression`

Example:

```csv
sequence,label
ATGCGTACGTAG,1
TTAACCGGTATA,0
```

FASTA/FA files are accepted for sequence upload and planning, but they do not contain labels by
default. A labeled benchmark run needs labels from a tabular file or a compatible metadata source.

SBOL/XML parsing is experimental and extracts available sequence/numeric fields when present.

## Quick Start With Python

Recommended Python version: `3.11`.

```powershell
cd C:\Users\Sgoff\MYfile\Desktop\PYThh\seqtrainer-benchlab
py -3.11 -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8001 --reload
```

Open:

```text
http://127.0.0.1:8001
```


## VS Code Local Run

For this two-mode BenchLab repo, use port `8001` during local development so it does not conflict with any older local checkout that may still be running on port `8000`.

In VS Code:

1. Open this folder: `C:\Users\Sgoff\MYfile\Desktop\PYThh\seqtrainer-benchlab`
2. Run the task **Run SeqTrainer BenchLab (two-mode UI, port 8001)**.
3. Open `http://127.0.0.1:8001/`.

The first page should show **Choose Workflow Mode**, **Beginner Student Mode**, and **Advanced Research Mode**.

## Quick Start With Docker

Build:

```powershell
docker build -t seqtrainer-benchlab .
```

Run locally:

```powershell
docker run --rm -p 127.0.0.1:8000:8000 seqtrainer-benchlab
```

Open:

```text
http://127.0.0.1:8000
```

To persist run artifacts on your own machine:

```powershell
docker run --rm -p 127.0.0.1:8000:8000 -v ${PWD}\storage:/app/storage seqtrainer-benchlab
```

## Replay And Validation

Validate a saved config without training:

```powershell
python -m app.replay --config path\to\run_config.json --dry-run
```

Run the easy local replay example:

```powershell
python -m app.reproducibility.run_from_config --config examples\reproducibility\easy_run_config.json --output-dir storage\easy_replay
```

Beginner notebook:

```text
notebooks/easy_model_replay_beginner.ipynb
```

Step-by-step script:

```text
notebooks/easy_model_replay_steps.py
```

## HPC Or Colab Use

BenchLab does not run the HPC job for you. It creates the reproducible plan.

Typical HPC/Colab workflow:

1. Export `run_config.json` from BenchLab.
2. Move `run_config.json` and the dataset to the compute environment.
3. Validate the config with dry-run mode.
4. Install dependencies using `requirements.lock.txt`, `environment.yml`, or `Dockerfile.repro`.
5. Run a notebook or script that reads the JSON and applies the same preprocessing, split, seed,
   threshold, and model settings.
6. Save new metrics and predictions as a new result bundle.

On HPC systems that do not allow Docker directly, use Apptainer/Singularity if available.

## Local Security And Data Handling

BenchLab is designed for local use.

- The Docker image runs as a non-root user.
- Uploaded source datasets are deleted after successful benchmark runs by default.
- Runtime data under `storage/` is ignored by git and excluded from Docker builds.
- Uploads are limited to 50 MB by default.
- Local quick runs are capped at 1000 rows by default.
- Dataset and run IDs are UUIDs.
- Backend preprocessing settings are bounded.
- Secrets such as tokens, passwords, SMTP credentials, and API keys are excluded from exported configs.

Optional environment settings:

```text
SEQTRAINER_MAX_UPLOAD_MB=50
SEQTRAINER_MAX_LOCAL_ROWS=1000
SEQTRAINER_MAX_KMER_SIZE=6
SEQTRAINER_MAX_ONE_HOT_LENGTH=1000
DELETE_DATASETS_AFTER_RUN=true
```

To wipe local runtime data:

```powershell
Remove-Item -Recurse -Force storage\datasets, storage\runs
```

Only run this command if you intentionally want to delete local uploaded datasets and run outputs.

## Development Checks

Install dev dependencies:

```powershell
python -m pip install -r requirements-dev.txt
```

Run tests:

```powershell
python -m pytest
```

Check frontend JavaScript syntax:

```powershell
node --check app\static\app.js
```

Export the JSON Schema:

```powershell
python -m app.reproducibility.export_schema --output schemas\run_config.schema.json
```

## Project Structure

```text
app/
  main.py                     FastAPI routes and local benchmark orchestration
  static/                     Web UI
  reproducibility/            run_config schema, builders, replay helpers
docs/
  intro_manual.md             Short collaborator-facing usage guide
  reproducible_runs.md        Details about run_config and replay artifacts
examples/
  reproducibility/            Example reproducibility configs
notebooks/
  easy_model_replay_beginner.ipynb
  easy_model_replay_steps.py
schemas/
  run_config.schema.json
tests/
  test_reproducibility.py
```

## Relationship To SeqTrainer

BenchLab is inspired by and aligned with the current upstream SeqTrainer project:

```text
https://github.com/SynBioDex/SeqTrainer
```

It uses SeqTrainer-style dataset preprocessing and simple baseline benchmarking ideas, adapted
into a local web workflow. The goal is to make the experiment setup easier for students and
scientists, then export a reproducible plan that can be used in larger SeqTrainer, notebook,
Colab, Docker, or HPC workflows.

## Roadmap

- Add stable CNN runner integration.
- Add DNABERT2 runner integration through notebook/HPC execution first.
- Add iPro-MP-compatible dataset and inference handoff.
- Add materialized split manifest creation for raw datasets.
- Add richer model-comparison reports across repeated seeds.
- Add optional authenticated deployment mode if the project becomes a hosted service.

