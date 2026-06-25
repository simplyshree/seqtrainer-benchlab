from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from .env import sanitize_env_vars, sanitize_pip_packages


SCHEMA_VERSION = "1.0.0"


class DatasetSpec(BaseModel):
    path: str | None = None
    uri: str | None = None
    sha256: str
    size_bytes: int | None = None
    sequence_column: str
    target_column: str
    dataset_id: str | None = None
    original_name: str | None = None
    row_count: int | None = None
    source_format: str | None = None
    columns: list[str] = Field(default_factory=list)
    source_dataset_removed_after_run: bool = True
    run_mode: str | None = None
    plan_only: bool = False
    note: str | None = None


class SplitSpec(BaseModel):
    split_strategy: str
    test_size: float
    validation_size: float
    train_csv: str | None = None
    validation_csv: str | None = None
    test_csv: str | None = None
    random_seed: int
    train_rows: int | None = None
    validation_rows: int | None = None
    test_rows: int | None = None
    cv_folds: int | None = None
    reruns: int | None = None


class PreprocessingSpec(BaseModel):
    use_gc: bool = True
    use_kmers: bool = True
    normalize_kmers: bool = True
    use_one_hot: bool = False
    kmer_size: int = Field(default=6, ge=1, le=6)
    sequence_length: int | None = Field(default=150, ge=1)
    additional_options: dict[str, Any] = Field(default_factory=dict)


class ModelSpec(BaseModel):
    model_name: str
    model_variant: str | None = None
    checkpoint: str | None = None
    tokenizer: str | None = None
    runner: str | None = None


class TrainingSpec(BaseModel):
    batch_size: int | None = None
    epochs: int | None = None
    cycles: int | None = None
    learning_rate: float | None = None
    weight_decay: float | None = None
    dropout: float | None = None
    early_stopping_patience: int | None = None
    class_weighting: str | None = None
    balance_strategy: str = "none"
    threshold_strategy: str
    threshold_value: float | None = None
    threshold_scope: str
    primary_metrics: list[str] = Field(default_factory=lambda: ["MCC", "F1", "precision", "recall", "accuracy"])
    biological_goal: str | None = None
    local_row_limit: int | None = None
    row_cap_applied: bool = False
    reruns: int | None = None
    cv_folds: int | None = None


class ThresholdSpec(BaseModel):
    threshold_strategy: str
    threshold_value: float | None = None
    threshold_scope: str
    biological_goal: str | None = None
    resolved_classification_threshold: float | None = None


class BalanceSpec(BaseModel):
    balance_strategy: str = "none"
    class_balance_applied: bool = False
    row_cap_applied: bool = False
    local_row_limit: int | None = None
    rows_used: int | None = None


class ModelSelectionSpec(BaseModel):
    local_models: list[str] = Field(default_factory=list)
    comparison_models: list[str] = Field(default_factory=list)
    runnable_local_models: list[str] = Field(default_factory=list)
    planned_hpc_models: list[str] = Field(default_factory=list)
    model_requirements: dict[str, str] = Field(default_factory=dict)


class DependencySpec(BaseModel):
    python_version: str
    python_implementation: str | None = None
    docker_base_image: str = "python:3.11-slim"
    pip_packages: list[str] = Field(default_factory=list)
    system_packages: list[str] = Field(default_factory=list)
    conda_dependencies: list[str] = Field(default_factory=list)
    cuda_version: str | None = None
    cudnn_version: str | None = None
    requirements_file: str = "requirements.txt"
    requirements_lock_file: str = "requirements.lock.txt"
    package_manager: str = "pip"
    notes: str | None = None

    @field_validator("pip_packages", mode="before")
    @classmethod
    def sanitize_packages(cls, value: Any) -> list[str]:
        return sanitize_pip_packages(value or [])


class HardwareSpec(BaseModel):
    device: str = "cpu"
    platform: str | None = None
    machine: str | None = None
    processor: str | None = None
    cpu_count: int | None = None
    gpu_available: bool = False
    cuda_available: bool = False
    cuda_version: str | None = None
    gpu_name: str | None = None
    gpu_count: int = 0
    gpu_names: list[str] = Field(default_factory=list)


class MetadataSpec(BaseModel):
    created_at: str
    commit: str | None = None
    git_commit: str | None = None
    repo_url: str | None = None
    branch: str | None = None
    tool_version: str | None = None
    run_id: str | None = None
    completed_at: str | None = None
    elapsed_seconds: float | None = None
    repo: str | None = None
    upstream_seqtrainer_source: dict[str, Any] = Field(default_factory=dict)
    artifact_paths: dict[str, str] = Field(default_factory=dict)


class EnvironmentSpec(BaseModel):
    safe_env_vars: dict[str, str] = Field(default_factory=dict)
    excluded_secret_patterns: list[str] = Field(default_factory=lambda: ["SECRET", "TOKEN", "PASSWORD", "KEY"])

    @field_validator("safe_env_vars", mode="before")
    @classmethod
    def sanitize_environment(cls, value: Any) -> dict[str, str]:
        return sanitize_env_vars(value or {})


class ReplaySpec(BaseModel):
    dry_run_command: str
    local_replay_command: str
    docker_build_command: str = "docker build -f Dockerfile.repro -t seqtrainer-benchlab-repro ."
    limitations: list[str] = Field(default_factory=list)


class ReproducibleRunConfig(BaseModel):
    schema_version: str = SCHEMA_VERSION
    dataset: DatasetSpec
    split: SplitSpec
    preprocessing: PreprocessingSpec
    models: list[ModelSpec]
    model_selection: ModelSelectionSpec | None = None
    threshold: ThresholdSpec | None = None
    balance: BalanceSpec | None = None
    training: TrainingSpec
    dependencies: DependencySpec
    hardware: HardwareSpec
    environment: EnvironmentSpec | None = None
    env_vars: dict[str, str] = Field(default_factory=dict)
    metadata: MetadataSpec
    replay: ReplaySpec | None = None

    @field_validator("env_vars", mode="before")
    @classmethod
    def sanitize_legacy_environment(cls, value: Any) -> dict[str, str]:
        return sanitize_env_vars(value or {})

    @classmethod
    def load(cls, path: str | Path) -> "ReproducibleRunConfig":
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


def json_schema() -> dict[str, Any]:
    return ReproducibleRunConfig.model_json_schema()
