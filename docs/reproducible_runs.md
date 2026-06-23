# Reproducible Runs

SeqTrainer BenchLab writes a canonical `run_config.json` for every benchmark run. This file
combines the values that used to live across separate artifacts into one validated config:

- dataset identity, checksum, columns, and row count
- split strategy, seed, fold/rerun settings, and observed split rows
- preprocessing settings
- selected local and Colab/HPC model targets
- threshold, balancing, and training settings
- Python version, pinned packages, safe environment variables, git metadata, and hardware info

The intended flow is:

```text
UI/API input
-> BenchmarkRequest
-> ReproducibleRunConfig
-> run_config.json
-> replay/model runner consumes the same config
-> metrics, predictions, and reproducibility artifacts
```

## Export From A Normal Run

Run BenchLab normally and start a benchmark. The run directory contains:

- `run_config.json`
- `requirements.lock.txt`
- `run_config.schema.json`
- `Dockerfile.repro`
- `environment.yml`
- `metrics.json`
- `predictions.csv`
- `run_manifest.json`
- `dataset_manifest.json`
- `preprocessing_config.json`
- `training_config.json`
- `benchmark_plan.json`
- `environment.json`

The export ZIP from the Results screen includes these same files.

## Replay From JSON

Dry-run validation:

```powershell
python -m app.reproducibility.run_from_config --config path\to\run_config.json --dry-run
```

Compatibility command:

```powershell
python -m app.replay --config path\to\run_config.json --dry-run
```

Dry-run API:

```text
POST /api/replay-config
```

The API accepts either a raw `run_config.json` object or:

```json
{
  "dry_run": true,
  "config": {}
}
```

Dry-run mode validates the JSON, verifies the dataset checksum if the dataset file exists,
sets random seeds, and writes a replay artifact directory without running heavy training.

Run the easy local models in one command:

```powershell
python -m app.reproducibility.run_from_config --config examples\reproducibility\easy_run_config.json --output-dir storage\easy_replay
```

This currently supports:

- `linear_regression`
- `random_forest`
- `gradient_boosting`

You can run a subset:

```powershell
python -m app.reproducibility.run_from_config --config examples\reproducibility\easy_run_config.json --models random_forest gradient_boosting
```

For step-by-step notebook-style usage, open:

```text
notebooks/easy_model_replay_steps.py
```

For a beginner-friendly Jupyter notebook, open:

```text
notebooks/easy_model_replay_beginner.ipynb
```

Full CNN, DNABERT2, and iPro-MP replay is intentionally a later adapter step. The command is
structured so those runners can be connected without duplicating training logic.

## Export JSON Schema

```powershell
python -m app.reproducibility.export_schema --output schemas\run_config.schema.json
```

## Regenerate Reproducibility Artifacts

The replay command writes:

- `run_config.json`
- `requirements.lock.txt`
- `environment.yml`
- `Dockerfile.repro`
- `replay_summary.json`

## Limitations

- Exact GPU driver reproduction is not guaranteed.
- Docker is more reproducible than a raw pip/venv setup.
- The dataset must be available locally or downloadable for true replay.
- If `DELETE_DATASETS_AFTER_RUN=true`, the exported config is complete but local replay is partial until the raw dataset is restored or re-uploaded.
- Passwords, API keys, SMTP secrets, and private tokens are intentionally excluded.
- Full CNN/DNABERT2/iPro-MP replay still needs runner adapters connected to this config.
