from __future__ import annotations

import argparse
import json
import random
import sys
import uuid
from pathlib import Path

import numpy as np

from app.model_runners.sklearn_easy import run_easy_models
from app.seqtrainer_core import file_sha256

from .config import ReproducibleRunConfig
from .env import write_dockerfile_repro, write_environment_yml, write_requirements_lock

REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_dataset_path(dataset_path: str | None, config_path: str | Path | None = None) -> Path | None:
    if not dataset_path:
        return None
    path = Path(dataset_path)
    if path.is_absolute():
        return path

    candidates = [Path.cwd() / path, REPO_ROOT / path]
    if config_path:
        config_dir = Path(config_path).resolve().parent
        candidates.extend([config_dir / path, config_dir.parent / path, config_dir.parent.parent / path])

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (REPO_ROOT / path).resolve()


def verify_dataset_checksum(config: ReproducibleRunConfig, config_path: str | Path | None = None) -> bool | None:
    dataset_path = resolve_dataset_path(config.dataset.path, config_path)
    if not dataset_path:
        return None
    if not dataset_path.exists():
        return None
    return file_sha256(dataset_path) == config.dataset.sha256


def set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def replay_from_config(
    config_path: str | Path,
    output_dir: str | Path | None = None,
    dry_run: bool = False,
    models: list[str] | None = None,
) -> dict:
    config = ReproducibleRunConfig.load(config_path)
    resolved_dataset_path = resolve_dataset_path(config.dataset.path, config_path)
    if resolved_dataset_path is not None:
        config.dataset.path = str(resolved_dataset_path)
    checksum_ok = verify_dataset_checksum(config, config_path)
    if checksum_ok is False:
        raise ValueError("Dataset checksum mismatch.")

    set_random_seeds(config.split.random_seed)
    output = Path(output_dir) if output_dir else Path("storage") / "replayed_runs" / str(uuid.uuid4())
    output.mkdir(parents=True, exist_ok=True)
    config.write_json(output / "run_config.json")
    write_requirements_lock(config.dependencies.pip_packages, output)
    write_environment_yml(config.dependencies.python_version, config.dependencies.pip_packages, output)
    write_dockerfile_repro(config.dependencies.python_version, output)

    summary = {
        "dry_run": dry_run,
        "output_dir": str(output),
        "schema_version": config.schema_version,
        "dataset_checksum_verified": checksum_ok,
        "models": [model.model_name for model in config.models],
        "random_seed": config.split.random_seed,
        "preprocessing": config.preprocessing.model_dump(mode="json"),
    }

    if not dry_run:
        run_result = run_easy_models(config, output, requested_models=models)
        summary.update(
            {
                "runnable_models": run_result["runnable_models"],
                "skipped_models": run_result["skipped_models"],
                "metrics_path": run_result["metrics_path"],
                "predictions_path": run_result["predictions_path"],
            }
        )

    (output / "replay_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate and replay a SeqTrainer BenchLab run_config.json.")
    parser.add_argument("--config", required=True, help="Path to run_config.json")
    parser.add_argument("--output-dir", help="Directory for replay artifacts")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and write replay artifacts without training")
    parser.add_argument(
        "--models",
        nargs="*",
        help="Optional easy-model subset: linear_regression random_forest gradient_boosting",
    )
    args = parser.parse_args(argv)

    try:
        summary = replay_from_config(args.config, args.output_dir, dry_run=args.dry_run, models=args.models)
    except Exception as exc:
        print(f"Replay failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
