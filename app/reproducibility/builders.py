from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import (
    DatasetSpec,
    BalanceSpec,
    DependencySpec,
    EnvironmentSpec,
    HardwareSpec,
    MetadataSpec,
    ModelSpec,
    ModelSelectionSpec,
    PreprocessingSpec,
    ReproducibleRunConfig,
    ReplaySpec,
    SplitSpec,
    ThresholdSpec,
    TrainingSpec,
)
from .env import (
    DOCKER_BASE_IMAGE,
    get_git_branch,
    get_git_commit,
    get_hardware_info,
    get_pip_freeze,
    get_relevant_env_vars,
    get_repo_url,
)


def build_run_config(
    *,
    request: Any,
    dataset_manifest: dict[str, Any],
    preprocessing_config: dict[str, Any],
    training_config: dict[str, Any],
    environment: dict[str, Any],
    run_manifest: dict[str, Any],
    repo_root: Path,
    dataset_path: str | None = None,
) -> ReproducibleRunConfig:
    packages = environment.get("pip_packages") or get_pip_freeze()
    hardware_info = environment.get("hardware") or get_hardware_info()
    local_models = list(training_config.get("models", []))
    comparison_models = list(training_config.get("comparison_models_for_colab_hpc", []))
    runnable_local = [model for model in local_models if model in {"linear_regression", "random_forest", "gradient_boosting"}]
    planned_hpc = [model for model in comparison_models if model in {"cnn", "dnabert2", "ipro_mp"}]

    models = [
        ModelSpec(model_name=model_name, runner="benchlab_sklearn_baseline")
        for model_name in training_config.get("models", [])
    ]
    models.extend(
        ModelSpec(model_name=model_name, runner="external_colab_hpc_target")
        for model_name in training_config.get("comparison_models_for_colab_hpc", [])
        if model_name not in {model.model_name for model in models}
    )

    preprocessing_known = {
        "use_gc",
        "use_kmers",
        "normalize_kmers",
        "use_one_hot",
        "kmer_size",
        "sequence_length",
    }
    preprocessing = PreprocessingSpec(
        use_gc=bool(preprocessing_config.get("use_gc", True)),
        use_kmers=bool(preprocessing_config.get("use_kmers", True)),
        normalize_kmers=bool(preprocessing_config.get("normalize_kmers", True)),
        use_one_hot=bool(preprocessing_config.get("use_one_hot", False)),
        kmer_size=int(preprocessing_config.get("kmer_size", 6)),
        sequence_length=int(preprocessing_config.get("sequence_length", 150)),
        additional_options={key: value for key, value in preprocessing_config.items() if key not in preprocessing_known},
    )

    return ReproducibleRunConfig(
        dataset=DatasetSpec(
            path=dataset_path,
            sha256=dataset_manifest["sha256"],
            size_bytes=dataset_manifest.get("upload_bytes"),
            sequence_column=training_config.get("sequence_col") or getattr(request, "sequence_col", "sequence"),
            target_column=training_config.get("target_col") or getattr(request, "target_col", "target"),
            dataset_id=dataset_manifest.get("dataset_id"),
            original_name=dataset_manifest.get("original_name"),
            row_count=dataset_manifest.get("row_count"),
            source_format=dataset_manifest.get("source_format"),
            columns=dataset_manifest.get("columns", []),
            source_dataset_removed_after_run=bool(run_manifest.get("source_dataset_removed_after_run", dataset_path is None)),
            note="Raw dataset must be retained locally or re-uploaded to fully replay model training.",
        ),
        split=SplitSpec(
            split_strategy=training_config.get("split_strategy", getattr(request, "split_strategy", "fixed_train_validation_test")),
            test_size=float(training_config.get("test_size", getattr(request, "test_size", 0.2))),
            validation_size=float(training_config.get("validation_size", getattr(request, "validation_size", 0.1))),
            random_seed=int(training_config.get("random_seed", getattr(request, "random_seed", 42))),
            train_rows=training_config.get("train_rows"),
            test_rows=training_config.get("test_rows"),
            cv_folds=training_config.get("cv_folds"),
            reruns=training_config.get("reruns"),
        ),
        preprocessing=preprocessing,
        models=models or [ModelSpec(model_name="linear_regression", runner="benchlab_sklearn_baseline")],
        model_selection=ModelSelectionSpec(
            local_models=local_models,
            comparison_models=comparison_models,
            runnable_local_models=runnable_local,
            planned_hpc_models=planned_hpc,
            model_requirements={
                "linear_regression": "sklearn",
                "random_forest": "sklearn",
                "gradient_boosting": "sklearn",
                "cnn": "planned external / SeqTrainer comparison target",
                "dnabert2": "planned external / transformer target",
                "ipro_mp": "planned external / external inference target",
            },
        ),
        threshold=ThresholdSpec(
            threshold_strategy=training_config.get("threshold_strategy", "user_or_literature"),
            threshold_value=training_config.get("classification_threshold"),
            threshold_scope=training_config.get("threshold_scope", "shared"),
            biological_goal=training_config.get("biological_goal"),
            resolved_classification_threshold=training_config.get("classification_threshold"),
        ),
        balance=BalanceSpec(
            balance_strategy=training_config.get("class_balance_strategy", "none"),
            class_balance_applied=bool(training_config.get("class_balance_applied", False)),
            row_cap_applied=bool(training_config.get("row_cap_applied", False)),
            local_row_limit=training_config.get("local_row_limit"),
            rows_used=training_config.get("rows_used"),
        ),
        training=TrainingSpec(
            cycles=training_config.get("training_cycles"),
            early_stopping_patience=training_config.get("early_stopping_patience"),
            balance_strategy=training_config.get("class_balance_strategy", "none"),
            threshold_strategy=training_config.get("threshold_strategy", "user_or_literature"),
            threshold_value=training_config.get("classification_threshold"),
            threshold_scope=training_config.get("threshold_scope", "shared"),
            biological_goal=training_config.get("biological_goal"),
            local_row_limit=training_config.get("local_row_limit"),
            row_cap_applied=bool(training_config.get("row_cap_applied", False)),
        ),
        dependencies=DependencySpec(
            python_version=environment.get("python_version", "unknown"),
            python_implementation=environment.get("python_implementation"),
            docker_base_image=DOCKER_BASE_IMAGE,
            pip_packages=packages,
            cuda_version=hardware_info.get("cuda_version"),
            notes="requirements.lock.txt is generated for this run; Docker replay is more stable than raw virtualenv replay.",
        ),
        hardware=HardwareSpec(
            device=hardware_info.get("device", "cpu"),
            platform=hardware_info.get("platform"),
            machine=hardware_info.get("machine"),
            processor=hardware_info.get("processor"),
            cpu_count=hardware_info.get("cpu_count"),
            gpu_available=bool(hardware_info.get("gpu_available", False)),
            cuda_available=bool(hardware_info.get("cuda_available", False)),
            cuda_version=hardware_info.get("cuda_version"),
            gpu_name=hardware_info.get("gpu_name"),
        ),
        environment=EnvironmentSpec(safe_env_vars=environment.get("env_vars") or get_relevant_env_vars()),
        env_vars=environment.get("env_vars") or get_relevant_env_vars(),
        metadata=MetadataSpec(
            created_at=run_manifest.get("created_at"),
            completed_at=run_manifest.get("completed_at"),
            elapsed_seconds=run_manifest.get("elapsed_seconds"),
            git_commit=environment.get("git_commit") or get_git_commit(repo_root),
            branch=environment.get("branch") or get_git_branch(repo_root),
            repo_url=environment.get("repo_url") or get_repo_url(repo_root),
            repo=environment.get("repo_url") or get_repo_url(repo_root),
            upstream_seqtrainer_source=environment.get("upstream_seqtrainer_source", {}),
            tool_version=run_manifest.get("seqtrainer_benchlab_version"),
            run_id=run_manifest.get("run_id"),
            artifact_paths=run_manifest.get("artifact_paths", {}),
        ),
        replay=ReplaySpec(
            dry_run_command="python -m app.replay --config run_config.json --dry-run",
            local_replay_command="python -m app.replay --config run_config.json",
            limitations=[
                "Exact GPU driver reproduction is not guaranteed.",
                "Raw dataset must be retained or re-uploaded for full replay.",
                "CNN, DNABERT2, and iPro-MP are planned comparison targets unless stable runners are connected.",
            ],
        ),
    )
