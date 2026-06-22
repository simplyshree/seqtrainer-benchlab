from __future__ import annotations

import argparse
import json
import random
import sys
import uuid
from pathlib import Path

import numpy as np

from app.seqtrainer_core import file_sha256

from .config import ReproducibleRunConfig
from .env import write_dockerfile_repro, write_environment_yml, write_requirements_lock


def verify_dataset_checksum(config: ReproducibleRunConfig) -> bool | None:
    if not config.dataset.path:
        return None
    dataset_path = Path(config.dataset.path)
    if not dataset_path.exists():
        return None
    return file_sha256(dataset_path) == config.dataset.sha256


def set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def replay_from_config(config_path: str | Path, output_dir: str | Path | None = None, dry_run: bool = False) -> dict:
    config = ReproducibleRunConfig.load(config_path)
    checksum_ok = verify_dataset_checksum(config)
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
    (output / "replay_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    if not dry_run:
        raise NotImplementedError(
            "Full model replay is intentionally not wired in this MVP. Use --dry-run to validate the config "
            "or connect this command to the SeqTrainer CNN/DNABERT2/iPro-MP runner adapters."
        )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate and replay a SeqTrainer BenchLab run_config.json.")
    parser.add_argument("--config", required=True, help="Path to run_config.json")
    parser.add_argument("--output-dir", help="Directory for replay artifacts")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and write replay artifacts without training")
    args = parser.parse_args(argv)

    try:
        summary = replay_from_config(args.config, args.output_dir, dry_run=args.dry_run)
    except Exception as exc:
        print(f"Replay failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

