from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from app.reproducibility.config import ReproducibleRunConfig
from app.seqtrainer_core import build_features, read_dataset


EASY_MODEL_MAP = {
    "linear_regression": LinearRegression,
    "random_forest": RandomForestRegressor,
    "gradient_boosting": GradientBoostingRegressor,
}


def is_binary_numeric_target(target: pd.Series) -> bool:
    values = set(pd.to_numeric(target, errors="coerce").dropna().astype(float).unique().tolist())
    return bool(values) and values.issubset({0.0, 1.0})


def selected_easy_models(config: ReproducibleRunConfig, requested_models: list[str] | None = None) -> tuple[list[str], list[str]]:
    requested = requested_models or [model.model_name for model in config.models]
    runnable = [name for name in requested if name in EASY_MODEL_MAP]
    skipped = [name for name in requested if name not in EASY_MODEL_MAP]
    return runnable, skipped


def run_easy_models(
    config: ReproducibleRunConfig,
    output_dir: str | Path,
    requested_models: list[str] | None = None,
) -> dict[str, Any]:
    if not config.dataset.path:
        raise ValueError("config.dataset.path is required to replay easy models.")
    dataset_path = Path(config.dataset.path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    runnable_models, skipped_models = selected_easy_models(config, requested_models)
    if not runnable_models:
        raise ValueError("No runnable easy models selected. Use linear_regression, random_forest, or gradient_boosting.")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    df = read_dataset(dataset_path)
    if config.dataset.sequence_column not in df.columns:
        raise ValueError(f"Sequence column not found: {config.dataset.sequence_column}")
    if config.dataset.target_column not in df.columns:
        raise ValueError(f"Target column not found: {config.dataset.target_column}")

    preprocessing = config.preprocessing.model_dump(mode="json")
    features = build_features(df, config.dataset.sequence_column, preprocessing)
    target = pd.to_numeric(df[config.dataset.target_column], errors="coerce")
    valid_mask = target.notna()
    features = features.loc[valid_mask].reset_index(drop=True)
    target = target.loc[valid_mask].reset_index(drop=True)
    if len(features) < 5:
        raise ValueError("Need at least 5 rows with numeric targets for replay.")

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=config.split.test_size,
        random_state=config.split.random_seed,
    )
    binary_target = is_binary_numeric_target(target)
    threshold = config.training.threshold_value if config.training.threshold_value is not None else 0.5

    metrics: dict[str, Any] = {}
    prediction_frames: list[pd.DataFrame] = []

    for model_name in runnable_models:
        model_cls = EASY_MODEL_MAP[model_name]
        probe = model_cls()
        model = model_cls(random_state=config.split.random_seed) if "random_state" in probe.get_params() else model_cls()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        model_metrics: dict[str, Any] = {
            "task_type": "regression",
            "mae": float(mean_absolute_error(y_test, predictions)),
            "mse": float(mse),
            "rmse": float(mse**0.5),
            "r2": float(r2_score(y_test, predictions)),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
        }
        prediction_payload: dict[str, Any] = {
            "model": model_name,
            "actual": y_test.to_numpy(),
            "prediction": predictions,
        }
        if binary_target:
            model_threshold = float(pd.Series(predictions).median()) if config.training.threshold_strategy == "median" else float(threshold)
            actual_classes = y_test.astype(int).to_numpy()
            predicted_classes = (predictions >= model_threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(actual_classes, predicted_classes, labels=[0, 1]).ravel()
            model_metrics.update(
                {
                    "task_type": "binary_label_regression_baseline",
                    "classification_threshold": model_threshold,
                    "threshold_strategy": config.training.threshold_strategy,
                    "accuracy": float(accuracy_score(actual_classes, predicted_classes)),
                    "precision": float(precision_score(actual_classes, predicted_classes, zero_division=0)),
                    "recall": float(recall_score(actual_classes, predicted_classes, zero_division=0)),
                    "f1": float(f1_score(actual_classes, predicted_classes, zero_division=0)),
                    "mcc": float(matthews_corrcoef(actual_classes, predicted_classes)),
                    "true_negative": int(tn),
                    "false_positive": int(fp),
                    "false_negative": int(fn),
                    "true_positive": int(tp),
                }
            )
            prediction_payload["predicted_label"] = predicted_classes
        metrics[model_name] = model_metrics
        joblib.dump(model, output / f"{model_name}.joblib")
        prediction_frames.append(pd.DataFrame(prediction_payload))

    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    metrics_path = output / "metrics.json"
    predictions_path = output / "predictions.csv"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    predictions_df.to_csv(predictions_path, index=False)

    return {
        "runnable_models": runnable_models,
        "skipped_models": skipped_models,
        "metrics": metrics,
        "metrics_path": str(metrics_path),
        "predictions_path": str(predictions_path),
    }
