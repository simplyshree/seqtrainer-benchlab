# %% [markdown]
# # Easy Model Replay From `run_config.json`
#
# This notebook-style script runs the lightweight BenchLab models:
#
# - Linear Regression
# - Random Forest
# - Gradient Boosting
#
# It uses the canonical `run_config.json` format, so the model parameters come from the
# exported reproducibility config rather than from hardcoded notebook values.

# %%
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.reproducibility.config import ReproducibleRunConfig
from app.reproducibility.run_from_config import replay_from_config, verify_dataset_checksum

# %% [markdown]
# ## 1. Choose a config
#
# Use the included example first. Later, replace this path with a `run_config.json`
# exported from BenchLab.

# %%
config_path = Path("examples/reproducibility/easy_run_config.json")
output_dir = Path("storage/easy_model_replay_notebook")

# %% [markdown]
# ## 2. Load and inspect the config

# %%
config = ReproducibleRunConfig.load(config_path)
print("Schema:", config.schema_version)
print("Dataset:", config.dataset.path)
print("Sequence column:", config.dataset.sequence_column)
print("Target column:", config.dataset.target_column)
print("Models:", [model.model_name for model in config.models])
print("Seed:", config.split.random_seed)
print("Preprocessing:", config.preprocessing.model_dump(mode="json"))

# %% [markdown]
# ## 3. Verify dataset checksum
#
# If the dataset exists locally, this should return `True`.
# If the dataset path is not available, it returns `None`.

# %%
print("Checksum verified:", verify_dataset_checksum(config))

# %% [markdown]
# ## 4. Dry-run first
#
# Dry-run validates the JSON and writes reproducibility artifacts without training.

# %%
dry_summary = replay_from_config(config_path, output_dir=output_dir / "dry_run", dry_run=True)
dry_summary

# %% [markdown]
# ## 5. Run the easy models
#
# This trains only the supported sklearn baselines. Heavy targets such as DNABERT2
# or iPro-MP are skipped by this easy runner.

# %%
run_summary = replay_from_config(config_path, output_dir=output_dir / "trained", dry_run=False)
run_summary

# %% [markdown]
# ## 6. Inspect outputs

# %%
trained_dir = Path(run_summary["output_dir"])
print("Metrics:", trained_dir / "metrics.json")
print("Predictions:", trained_dir / "predictions.csv")
print("Replay summary:", trained_dir / "replay_summary.json")
print("Repro config:", trained_dir / "run_config.json")
