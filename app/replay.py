from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.reproducibility.config import ReproducibleRunConfig
from app.reproducibility.run_from_config import replay_from_config, verify_dataset_checksum


def load_run_config(path: str | Path) -> ReproducibleRunConfig:
    return ReproducibleRunConfig.load(path)


def validate_run_config(config: dict[str, Any] | ReproducibleRunConfig) -> ReproducibleRunConfig:
    if isinstance(config, ReproducibleRunConfig):
        return config
    return ReproducibleRunConfig.model_validate(config)


def replay_config_dict(config: dict[str, Any], dry_run: bool = True) -> dict[str, Any]:
    validated = validate_run_config(config)
    with tempfile.TemporaryDirectory() as tmp:
        config_path = Path(tmp) / "run_config.json"
        validated.write_json(config_path)
        return replay_from_config(config_path, output_dir=Path(tmp) / "replay", dry_run=dry_run)


def replay_from_loaded_config(config: ReproducibleRunConfig, dry_run: bool = True) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        config_path = Path(tmp) / "run_config.json"
        config.write_json(config_path)
        return replay_from_config(config_path, output_dir=Path(tmp) / "replay", dry_run=dry_run)


def dry_run_summary(config: ReproducibleRunConfig) -> dict[str, Any]:
    checksum = verify_dataset_checksum(config)
    dataset_available = bool(config.dataset.path and Path(config.dataset.path).exists())
    return {
        "valid": True,
        "dry_run": True,
        "dataset_available": dataset_available,
        "dataset_checksum_verified": checksum,
        "missing_dataset_warning": None if dataset_available else "Raw dataset is not available locally; full replay requires re-uploading or restoring it.",
        "models": [model.model_name for model in config.models],
        "preprocessing": config.preprocessing.model_dump(mode="json"),
        "split": config.split.model_dump(mode="json"),
        "dependencies": config.dependencies.model_dump(mode="json"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate and replay a BenchLab run_config.json.")
    parser.add_argument("--config", required=True, help="Path to run_config.json")
    parser.add_argument("--dry-run", action="store_true", help="Validate and summarize without training")
    parser.add_argument("--output-dir", help="Output directory for replay artifacts")
    args = parser.parse_args(argv)

    try:
        config = load_run_config(args.config)
        if args.dry_run:
            summary = dry_run_summary(config)
        else:
            summary = replay_from_config(args.config, output_dir=args.output_dir, dry_run=False)
    except (ValidationError, Exception) as exc:
        print(f"Replay failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

