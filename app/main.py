from __future__ import annotations

import json
import os
import re
import smtplib
import shutil
import sys
import time
import uuid
import zipfile
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

import joblib
import numpy as np
import pandas as pd
import sklearn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
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

from .seqtrainer_core import build_features, file_sha256, read_dataset


APP_ROOT = Path(__file__).resolve().parent
REPO_ROOT = APP_ROOT.parent
STORAGE_ROOT = Path(os.getenv("SEQTRAINER_STORAGE", REPO_ROOT / "storage")).expanduser().resolve()
DATASETS_ROOT = STORAGE_ROOT / "datasets"
RUNS_ROOT = STORAGE_ROOT / "runs"
DEFAULT_SMALL_DATASET_LIMIT = 1000
DELETE_DATASETS_AFTER_RUN = os.getenv("DELETE_DATASETS_AFTER_RUN", "true").lower() not in {"0", "false", "no"}
MAX_UPLOAD_BYTES = int(os.getenv("SEQTRAINER_MAX_UPLOAD_MB", "50")) * 1024 * 1024
MAX_LOCAL_RUN_ROWS = int(os.getenv("SEQTRAINER_MAX_LOCAL_ROWS", str(DEFAULT_SMALL_DATASET_LIMIT)))
MAX_KMER_SIZE = int(os.getenv("SEQTRAINER_MAX_KMER_SIZE", "6"))
MAX_ONE_HOT_LENGTH = int(os.getenv("SEQTRAINER_MAX_ONE_HOT_LENGTH", "1000"))
ALLOWED_DATASET_EXTENSIONS = {".csv", ".tsv", ".txt", ".fa", ".fasta", ".xml", ".rdf", ".sbol"}
ID_PATTERN = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
LOCAL_ORIGIN_HOSTS = {"localhost", "127.0.0.1", "::1"}

MODEL_MAP = {
    "linear_regression": LinearRegression,
    "random_forest": RandomForestRegressor,
    "gradient_boosting": GradientBoostingRegressor,
}

TARGET_COLUMN_CANDIDATES = ["target", "label", "y", "class", "promoter", "expression", "expn_med"]

SEQTRAINER_SOURCE = {
    "repository": "SynBioDex/SeqTrainer",
    "url": "https://github.com/SynBioDex/SeqTrainer",
    "verified_main_commit": "5e9701d",
    "verified_on": "2026-06-05",
}

CAPABILITIES = {
    "source": SEQTRAINER_SOURCE,
    "supported_formats": ["csv", "tsv", "txt", "fasta", "fa", "xml", "rdf", "sbol"],
    "seqtrainer_functions": [
        "dataset_builder.get_sequence_from_sbol",
        "dataset_builder.find_possible_y_uris",
        "dataset_builder.get_y_label",
        "dataset_builder.build_dataset",
        "preprocessing.one_hot_encode",
        "preprocessing.pad_sequence",
        "preprocessing.process_seqs",
        "preprocessing.calc_gc",
        "preprocessing.generate_kmer_counts",
        "hpc.sklearn_tuning.MODEL_MAP",
    ],
    "live_models": [
        {"id": "linear_regression", "name": "Linear Regression", "source": "sklearn_tuning lr"},
        {"id": "random_forest", "name": "Random Forest", "source": "sklearn_tuning rfr"},
        {"id": "gradient_boosting", "name": "Gradient Boosting", "source": "sklearn_tuning gbr"},
    ],
    "upstream_research_assets": [
        {"name": "DNABERT2 notebooks", "status": "present in upstream; not exposed as a stable web endpoint"},
        {"name": "SBOL GNN experiment", "status": "present in upstream; not exposed as a stable web endpoint"},
    ],
}


class BenchmarkRequest(BaseModel):
    dataset_id: str
    sequence_col: str = "sequence"
    target_col: str = "target"
    models: list[str] = Field(default_factory=lambda: ["linear_regression"])
    comparison_models: list[str] = Field(default_factory=lambda: ["cnn", "dnabert2", "ipro_mp"])
    preprocessing: dict[str, Any] = Field(default_factory=lambda: {"use_gc": True, "use_kmers": True, "kmer_size": 6})
    test_size: float = Field(default=0.2, ge=0.05, le=0.5)
    validation_size: float = Field(default=0.1, ge=0.0, le=0.4)
    random_seed: int = Field(default=42, ge=0, le=2_147_483_647)
    split_strategy: str = "fixed_train_validation_test"
    threshold_strategy: str = "user_or_literature"
    threshold_value: float | None = None
    threshold_scope: str = "shared"
    biological_goal: str = "limit_false_positives"
    balance_strategy: str = "none"
    max_rows: int = Field(default=DEFAULT_SMALL_DATASET_LIMIT, ge=5, le=MAX_LOCAL_RUN_ROWS)
    reruns: int = Field(default=3, ge=1, le=100)
    cv_folds: int = Field(default=5, ge=2, le=10)
    training_cycles: int = Field(default=20, ge=1, le=1000)
    early_stopping_patience: int = Field(default=5, ge=0, le=50)


class BenchmarkPlanRequest(BenchmarkRequest):
    run_local_baseline: bool = False


class EmailRunRequest(BaseModel):
    email: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def runtime_environment() -> dict[str, Any]:
    return {
        "python_version": sys.version.split()[0],
        "python_implementation": sys.implementation.name,
        "packages": {
            "fastapi": getattr(sys.modules.get("fastapi"), "__version__", "unknown"),
            "joblib": getattr(joblib, "__version__", "unknown"),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }


def ensure_storage() -> None:
    DATASETS_ROOT.mkdir(parents=True, exist_ok=True)
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)


def dataset_dir(dataset_id: str) -> Path:
    return DATASETS_ROOT / validate_storage_id(dataset_id, "Dataset")


def run_dir(run_id: str) -> Path:
    return RUNS_ROOT / validate_storage_id(run_id, "Run")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def validate_storage_id(value: str, resource_name: str) -> str:
    if not ID_PATTERN.fullmatch(value):
        raise HTTPException(status_code=404, detail=f"{resource_name} not found")
    return value


def safe_dataset_filename(filename: str | None) -> str:
    path = Path(filename or "dataset").name
    suffix = Path(path).suffix.lower()
    if suffix not in ALLOWED_DATASET_EXTENSIONS:
        supported = ", ".join(sorted(ALLOWED_DATASET_EXTENSIONS))
        raise HTTPException(status_code=400, detail=f"Unsupported dataset format: {suffix or 'missing extension'}. Supported: {supported}")

    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(path).stem).strip("._-")[:80]
    return f"{stem or 'dataset'}{suffix}"


def save_upload_file(file: UploadFile, destination: Path) -> int:
    bytes_written = 0
    with destination.open("wb") as handle:
        while chunk := file.file.read(1024 * 1024):
            bytes_written += len(chunk)
            if bytes_written > MAX_UPLOAD_BYTES:
                max_mb = MAX_UPLOAD_BYTES // (1024 * 1024)
                raise HTTPException(status_code=413, detail=f"Dataset upload is too large. Maximum allowed size is {max_mb} MB.")
            handle.write(chunk)
    return bytes_written


def parse_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def bounded_int(value: Any, default: int, minimum: int, maximum: int, name: str) -> int:
    try:
        parsed = int(value if value is not None else default)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"{name} must be an integer.") from exc
    if parsed < minimum or parsed > maximum:
        raise HTTPException(status_code=400, detail=f"{name} must be between {minimum} and {maximum}.")
    return parsed


def sanitize_preprocessing(config: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise HTTPException(status_code=400, detail="Preprocessing config must be a JSON object.")

    sanitized = {
        "use_gc": parse_bool(config.get("use_gc"), True),
        "use_kmers": parse_bool(config.get("use_kmers"), True),
        "normalize_kmers": parse_bool(config.get("normalize_kmers"), True),
        "use_one_hot": parse_bool(config.get("use_one_hot"), False),
        "kmer_size": bounded_int(config.get("kmer_size"), 6, 1, MAX_KMER_SIZE, "kmer_size"),
        "sequence_length": bounded_int(config.get("sequence_length"), 150, 20, MAX_ONE_HOT_LENGTH, "sequence_length"),
    }
    if not any([sanitized["use_gc"], sanitized["use_kmers"], sanitized["use_one_hot"]]):
        raise HTTPException(status_code=400, detail="Select at least one preprocessing feature family.")
    return sanitized


def is_allowed_local_origin(origin: str) -> bool:
    allowed_origins = {item.strip() for item in os.getenv("SEQTRAINER_ALLOWED_ORIGINS", "").split(",") if item.strip()}
    if origin in allowed_origins:
        return True
    parsed = urlparse(origin)
    return parsed.scheme in {"http", "https"} and parsed.hostname in LOCAL_ORIGIN_HOSTS


def build_run_email_body(run_id: str, base_url: str) -> str:
    folder = run_dir(run_id)
    manifest = read_json(folder / "run_manifest.json")
    dataset_manifest = read_json(folder / "dataset_manifest.json")
    training_config = read_json(folder / "training_config.json")
    metrics = read_json(folder / "metrics.json")
    environment = read_json(folder / "environment.json") if (folder / "environment.json").exists() else {}
    export_url = f"{base_url.rstrip('/')}/api/runs/{run_id}/export"

    lines = [
        "SeqTrainer BenchLab results",
        "",
        f"Run ID: {run_id}",
        f"Created: {manifest.get('created_at', 'unknown')}",
        f"Completed: {manifest.get('completed_at', 'unknown')}",
        f"Elapsed seconds: {manifest.get('elapsed_seconds', 'unknown')}",
        f"Dataset: {dataset_manifest.get('original_name', dataset_manifest.get('dataset_id', 'unknown'))}",
        f"Dataset SHA256: {dataset_manifest.get('sha256', 'unknown')}",
        f"Sequence column: {training_config.get('sequence_col', 'unknown')}",
        f"Target column: {training_config.get('target_col', 'unknown')}",
        f"Models: {', '.join(training_config.get('models', []))}",
        f"Python: {environment.get('python_version', 'unknown')}",
        "",
        "Metrics:",
    ]
    for model_name, values in metrics.items():
        lines.append(f"- {model_name}")
        for metric_name in [
            "task_type",
            "mae",
            "mse",
            "rmse",
            "r2",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "mcc",
            "train_rows",
            "test_rows",
        ]:
            if metric_name in values:
                lines.append(f"  {metric_name}: {values[metric_name]}")
    lines.extend(["", f"Export bundle: {export_url}", "", "Generated by SeqTrainer BenchLab."])
    return "\n".join(lines)


def send_smtp_email(to_email: str, subject: str, body: str) -> bool:
    host = os.getenv("SMTP_HOST")
    if not host:
        return False

    port = int(os.getenv("SMTP_PORT", "587"))
    username = os.getenv("SMTP_USERNAME")
    password = os.getenv("SMTP_PASSWORD")
    from_email = os.getenv("SMTP_FROM", username or "seqtrainer-benchlab@example.com")
    use_tls = os.getenv("SMTP_TLS", "true").lower() not in {"0", "false", "no"}

    message = EmailMessage()
    message["From"] = from_email
    message["To"] = to_email
    message["Subject"] = subject
    message.set_content(body)

    with smtplib.SMTP(host, port, timeout=20) as server:
        if use_tls:
            server.starttls()
        if username and password:
            server.login(username, password)
        server.send_message(message)
    return True


def load_dataset(dataset_id: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    folder = dataset_dir(dataset_id)
    manifest_path = folder / "dataset_manifest.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")
    manifest = read_json(manifest_path)
    df = pd.read_csv(folder / "dataset.csv")
    return df, manifest


def infer_column(columns: list[str], candidates: list[str]) -> str | None:
    normalized = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]
    return None


def is_binary_numeric_target(target: pd.Series) -> bool:
    values = set(pd.to_numeric(target, errors="coerce").dropna().astype(float).unique().tolist())
    return bool(values) and values.issubset({0.0, 1.0})


def summarize_target(df: pd.DataFrame, target_col: str | None) -> dict[str, Any]:
    if not target_col or target_col not in df.columns:
        return {"available": False, "message": "No target column detected yet."}

    target = df[target_col].dropna()
    counts = target.astype(str).value_counts().to_dict()
    total = int(target.shape[0])
    numeric_target = pd.to_numeric(target, errors="coerce")
    binary_target = is_binary_numeric_target(numeric_target)
    minority_fraction = None
    imbalance_detected = False
    if len(counts) >= 2 and total:
        minority_fraction = min(counts.values()) / total
        imbalance_detected = minority_fraction < 0.35

    return {
        "available": True,
        "target_col": target_col,
        "row_count_with_target": total,
        "class_counts": counts,
        "class_count": len(counts),
        "binary_numeric": binary_target,
        "minority_fraction": round(minority_fraction, 4) if minority_fraction is not None else None,
        "imbalance_detected": imbalance_detected,
        "recommendation": "Consider class balancing or class-weighted training." if imbalance_detected else "No strong class imbalance detected.",
    }


def make_benchmark_plan(request: BenchmarkPlanRequest | BenchmarkRequest, dataset_manifest: dict[str, Any]) -> dict[str, Any]:
    threshold_value = request.threshold_value
    if request.threshold_strategy == "median":
        threshold_value = "median_from_validation_or_training_labels"
    elif request.threshold_strategy == "validation_mcc":
        threshold_value = "selected_on_validation_split_by_mcc"

    return {
        "schema": "seqtrainer_benchlab_benchmark_plan/v1",
        "created_at": utc_now(),
        "dataset": {
            "dataset_id": request.dataset_id,
            "original_name": dataset_manifest.get("original_name"),
            "sha256": dataset_manifest.get("sha256"),
            "rows_uploaded": dataset_manifest.get("row_count"),
            "large_dataset_policy": f"Local quick-run mode uses first {request.max_rows} rows unless run on Colab/HPC.",
            "analysis": dataset_manifest.get("target_summary", {}),
        },
        "columns": {
            "sequence_col": request.sequence_col,
            "target_col": request.target_col,
        },
        "split": {
            "strategy": request.split_strategy,
            "test_size": request.test_size,
            "validation_size": request.validation_size,
            "random_seed": request.random_seed,
            "cv_folds": request.cv_folds,
            "reruns": request.reruns,
            "materialize_manifest": True,
            "notebook": "03_materialize_dataset_splits.ipynb",
        },
        "threshold": {
            "strategy": request.threshold_strategy,
            "value": threshold_value,
            "scope": request.threshold_scope,
            "biological_goal": request.biological_goal,
            "false_positive_cost": "high",
        },
        "preprocessing": sanitize_preprocessing(request.preprocessing),
        "class_balance": {
            "strategy": request.balance_strategy,
            "activate_only_if_imbalanced": True,
            "summary_required_before_training": True,
        },
        "models": {
            "local_quick_run": request.models,
            "colab_hpc_comparison": request.comparison_models,
            "notes": "CNN, DNABERT2, and iPro-MP are exported as reproducible Colab/HPC protocol targets.",
        },
        "training": {
            "training_cycles": request.training_cycles,
            "early_stopping_patience": request.early_stopping_patience,
            "package_versions_required": True,
            "docker_recommended": True,
        },
        "artifacts": [
            "raw_data_source",
            "split_manifest",
            "seed",
            "config_json",
            "package_versions",
            "threshold",
            "metrics",
            "predictions",
            "model_checkpoint",
            "docker_image_or_container_spec",
        ],
    }


def make_codex_prompt(plan: dict[str, Any]) -> str:
    return (
        "Create a Colab/HPC-ready SeqTrainer benchmark notebook series from this JSON plan. "
        "Materialize one fixed train/validation/test split manifest first, then run comparable "
        "CNN, DNABERT2, and iPro-MP classification benchmarks using the same preprocessing, "
        "same split manifest, same threshold policy, same seeds, and the listed artifacts. "
        "If class imbalance is detected, apply the configured balancing strategy only when it is scientifically justified. "
        "Export metrics, predictions, manifests, package versions, model checkpoints, and a Docker/container note.\n\n"
        f"{json.dumps(plan, indent=2, sort_keys=True)}"
    )


def maybe_cap_rows(df: pd.DataFrame, max_rows: int) -> tuple[pd.DataFrame, bool]:
    if max_rows > 0 and len(df) > max_rows:
        return df.head(max_rows).copy(), True
    return df, False


def maybe_balance_binary_rows(df: pd.DataFrame, target_col: str, strategy: str, random_seed: int) -> tuple[pd.DataFrame, bool]:
    if strategy not in {"undersample", "auto_undersample"} or target_col not in df.columns:
        return df, False
    numeric_target = pd.to_numeric(df[target_col], errors="coerce")
    if not is_binary_numeric_target(numeric_target):
        return df, False
    working = df.assign(_bench_target=numeric_target)
    groups = [group for _, group in working.dropna(subset=["_bench_target"]).groupby("_bench_target")]
    if len(groups) != 2:
        return df, False
    counts = [len(group) for group in groups]
    minority_fraction = min(counts) / sum(counts)
    if strategy == "auto_undersample" and minority_fraction >= 0.35:
        return df, False
    min_count = min(len(group) for group in groups)
    balanced = pd.concat([group.sample(n=min_count, random_state=random_seed) for group in groups], ignore_index=True)
    return balanced.drop(columns=["_bench_target"]).sample(frac=1, random_state=random_seed).reset_index(drop=True), True


app = FastAPI(title="SeqTrainer BenchLab", version="0.1.0")


@app.middleware("http")
async def security_middleware(request: Request, call_next):
    if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
        origin = request.headers.get("origin")
        if origin and not is_allowed_local_origin(origin):
            return JSONResponse(status_code=403, content={"detail": "Origin not allowed for this local BenchLab instance."})

    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.on_event("startup")
def on_startup() -> None:
    ensure_storage()


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "seqtrainer-benchlab"}


@app.get("/api/capabilities")
def capabilities() -> dict[str, Any]:
    return CAPABILITIES


@app.post("/api/datasets")
async def upload_dataset(file: UploadFile = File(...)) -> dict[str, Any]:
    ensure_storage()
    original_name = safe_dataset_filename(file.filename)
    dataset_id = str(uuid.uuid4())
    folder = dataset_dir(dataset_id)
    folder.mkdir(parents=True, exist_ok=True)

    raw_path = folder / original_name
    try:
        upload_bytes = save_upload_file(file, raw_path)
    except HTTPException:
        shutil.rmtree(folder, ignore_errors=True)
        raise

    try:
        df = read_dataset(raw_path)
    except Exception as exc:
        shutil.rmtree(folder, ignore_errors=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if df.empty:
        shutil.rmtree(folder, ignore_errors=True)
        raise HTTPException(status_code=400, detail="Dataset contains no rows")

    df.to_csv(folder / "dataset.csv", index=False)
    suggested_target_col = infer_column(list(df.columns), TARGET_COLUMN_CANDIDATES)
    manifest = {
        "dataset_id": dataset_id,
        "original_name": original_name,
        "created_at": utc_now(),
        "row_count": int(len(df)),
        "columns": list(df.columns),
        "suggested_sequence_col": infer_column(list(df.columns), ["sequence", "seq", "variant", "dna"]),
        "suggested_target_col": suggested_target_col,
        "target_summary": summarize_target(df, suggested_target_col),
        "large_dataset_warning": int(len(df)) > MAX_LOCAL_RUN_ROWS,
        "local_small_run_limit": MAX_LOCAL_RUN_ROWS,
        "sha256": file_sha256(raw_path),
        "source_format": raw_path.suffix.lower(),
        "upload_bytes": upload_bytes,
    }
    write_json(folder / "dataset_manifest.json", manifest)
    return {"dataset": manifest, "preview": df.head(20).fillna("").to_dict(orient="records")}


@app.get("/api/datasets")
def list_datasets() -> dict[str, Any]:
    ensure_storage()
    datasets = []
    for folder in sorted(DATASETS_ROOT.iterdir()):
        manifest_path = folder / "dataset_manifest.json"
        if manifest_path.exists():
            datasets.append(read_json(manifest_path))
    return {"datasets": datasets}


@app.get("/api/datasets/{dataset_id}")
def get_dataset(dataset_id: str) -> dict[str, Any]:
    df, manifest = load_dataset(dataset_id)
    return {"dataset": manifest, "preview": df.head(50).fillna("").to_dict(orient="records")}


@app.post("/api/preprocess/{dataset_id}")
def preview_preprocessing(dataset_id: str, sequence_col: str = Form("sequence"), config: str = Form("{}")) -> dict[str, Any]:
    df, _ = load_dataset(dataset_id)
    if sequence_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column not found: {sequence_col}")
    try:
        parsed_config = json.loads(config)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Preprocessing config must be valid JSON.") from exc
    parsed_config = sanitize_preprocessing(parsed_config)
    features = build_features(df, sequence_col, parsed_config)
    return {
        "feature_count": int(features.shape[1]),
        "row_count": int(features.shape[0]),
        "columns": list(features.columns[:40]),
        "preview": features.head(5).round(6).to_dict(orient="records"),
    }


@app.post("/api/benchmark-plan")
def generate_benchmark_plan(request: BenchmarkPlanRequest) -> dict[str, Any]:
    _, dataset_manifest = load_dataset(request.dataset_id)
    plan = make_benchmark_plan(request, dataset_manifest)
    return {"plan": plan, "codex_prompt": make_codex_prompt(plan)}


@app.post("/api/benchmarks")
def run_benchmark(request: BenchmarkRequest) -> dict[str, Any]:
    started_monotonic = time.perf_counter()
    started_at = utc_now()
    df, dataset_manifest = load_dataset(request.dataset_id)
    if request.sequence_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column not found: {request.sequence_col}")
    target_col = request.target_col
    if target_col not in df.columns:
        inferred_target = infer_column(list(df.columns), TARGET_COLUMN_CANDIDATES)
        if inferred_target:
            target_col = inferred_target
        else:
            available = ", ".join(str(column) for column in df.columns)
            raise HTTPException(status_code=400, detail=f"Target column not found: {request.target_col}. Available columns: {available}")

    selected_models = request.models or ["linear_regression"]
    unknown_models = [model for model in selected_models if model not in MODEL_MAP]
    if unknown_models:
        raise HTTPException(status_code=400, detail=f"Unsupported model(s): {', '.join(unknown_models)}")

    preprocessing_config = sanitize_preprocessing(request.preprocessing)
    source_rows = int(len(df))
    df, row_cap_applied = maybe_cap_rows(df, request.max_rows)
    df, class_balance_applied = maybe_balance_binary_rows(df, target_col, request.balance_strategy, request.random_seed)

    features = build_features(df, request.sequence_col, preprocessing_config)
    target = pd.to_numeric(df[target_col], errors="coerce")
    valid_mask = target.notna()
    features = features.loc[valid_mask].reset_index(drop=True)
    target = target.loc[valid_mask].reset_index(drop=True)

    if len(features) < 5:
        raise HTTPException(status_code=400, detail="Need at least 5 rows with numeric targets for a benchmark")

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=request.test_size,
        random_state=request.random_seed,
    )
    train_rows = int(len(X_train))
    test_rows = int(len(X_test))
    rows_used = int(len(features))
    binary_target = is_binary_numeric_target(target)

    run_id = str(uuid.uuid4())
    folder = run_dir(run_id)
    folder.mkdir(parents=True, exist_ok=True)

    metrics: dict[str, Any] = {}
    prediction_frames = []

    for model_name in selected_models:
        model_cls = MODEL_MAP[model_name]
        model = model_cls(random_state=request.random_seed) if "random_state" in model_cls().get_params() else model_cls()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        model_metrics = {
            "task_type": "regression",
            "mae": float(mean_absolute_error(y_test, predictions)),
            "mse": float(mse),
            "rmse": float(mse**0.5),
            "r2": float(r2_score(y_test, predictions)),
            "train_rows": train_rows,
            "test_rows": test_rows,
        }
        prediction_payload: dict[str, Any] = {
            "model": model_name,
            "actual": y_test.to_numpy(),
            "prediction": predictions,
        }
        if binary_target:
            threshold = request.threshold_value if request.threshold_value is not None else 0.5
            if request.threshold_strategy == "median":
                threshold = float(pd.Series(predictions).median())
            actual_classes = y_test.astype(int).to_numpy()
            predicted_classes = (predictions >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(actual_classes, predicted_classes, labels=[0, 1]).ravel()
            model_metrics.update(
                {
                    "task_type": "binary_label_regression_baseline",
                    "classification_threshold": threshold,
                    "threshold_strategy": request.threshold_strategy,
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
        joblib.dump(model, folder / f"{model_name}.joblib")
        prediction_frames.append(pd.DataFrame(prediction_payload))

    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    predictions_df.to_csv(folder / "predictions.csv", index=False)

    training_config = {
        "models": selected_models,
        "test_size": request.test_size,
        "validation_size": request.validation_size,
        "random_seed": request.random_seed,
        "split_strategy": request.split_strategy,
        "sequence_col": request.sequence_col,
        "target_col": target_col,
        "rows_uploaded": int(dataset_manifest.get("row_count", source_rows)),
        "local_row_limit": request.max_rows,
        "row_cap_applied": row_cap_applied,
        "class_balance_strategy": request.balance_strategy,
        "class_balance_applied": class_balance_applied,
        "rows_used": rows_used,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "binary_target": binary_target,
        "threshold_strategy": request.threshold_strategy,
        "threshold_scope": request.threshold_scope,
        "biological_goal": request.biological_goal,
        "classification_threshold": threshold if binary_target else None,
        "comparison_models_for_colab_hpc": request.comparison_models,
        "reruns": request.reruns,
        "cv_folds": request.cv_folds,
        "training_cycles": request.training_cycles,
        "early_stopping_patience": request.early_stopping_patience,
    }
    elapsed_seconds = round(time.perf_counter() - started_monotonic, 4)
    environment = runtime_environment()
    request_for_plan = (
        request.model_copy(update={"preprocessing": preprocessing_config})
        if hasattr(request, "model_copy")
        else request.copy(update={"preprocessing": preprocessing_config})
    )
    benchmark_plan = make_benchmark_plan(request_for_plan, dataset_manifest)
    run_manifest = {
        "run_id": run_id,
        "created_at": started_at,
        "completed_at": utc_now(),
        "elapsed_seconds": elapsed_seconds,
        "dataset_id": request.dataset_id,
        "dataset_sha256": dataset_manifest["sha256"],
        "seqtrainer_benchlab_version": "0.1.0",
        "rows_used": rows_used,
        "row_cap_applied": row_cap_applied,
        "class_balance_applied": class_balance_applied,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "target_kind": "binary_numeric_label" if binary_target else "numeric_regression",
        "artifact_paths": {
            "metrics": "metrics.json",
            "predictions": "predictions.csv",
            "dataset_manifest": "dataset_manifest.json",
            "preprocessing_config": "preprocessing_config.json",
            "training_config": "training_config.json",
            "benchmark_plan": "benchmark_plan.json",
            "environment": "environment.json",
        },
    }

    write_json(folder / "metrics.json", metrics)
    write_json(folder / "dataset_manifest.json", dataset_manifest)
    write_json(folder / "preprocessing_config.json", preprocessing_config)
    write_json(folder / "training_config.json", training_config)
    write_json(folder / "benchmark_plan.json", benchmark_plan)
    write_json(folder / "environment.json", environment)
    write_json(folder / "run_manifest.json", run_manifest)

    dataset_removed = False
    if DELETE_DATASETS_AFTER_RUN:
        shutil.rmtree(dataset_dir(request.dataset_id), ignore_errors=True)
        dataset_removed = True
        run_manifest["source_dataset_removed_after_run"] = True
        write_json(folder / "run_manifest.json", run_manifest)

    return {
        "run": run_manifest,
        "metrics": metrics,
        "predictions": predictions_df.head(50).round(6).to_dict(orient="records"),
        "dataset": dataset_manifest,
        "training_config": training_config,
        "preprocessing_config": preprocessing_config,
        "environment": environment,
        "benchmark_plan": benchmark_plan,
        "codex_prompt": make_codex_prompt(benchmark_plan),
        "dataset_removed": dataset_removed,
    }


@app.get("/api/runs")
def list_runs() -> dict[str, Any]:
    ensure_storage()
    runs = []
    for folder in sorted(RUNS_ROOT.iterdir()):
        manifest_path = folder / "run_manifest.json"
        if manifest_path.exists():
            runs.append(read_json(manifest_path))
    return {"runs": runs}


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    folder = run_dir(run_id)
    manifest_path = folder / "run_manifest.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="Run not found")
    predictions = pd.read_csv(folder / "predictions.csv")
    return {
        "run": read_json(manifest_path),
        "metrics": read_json(folder / "metrics.json"),
        "dataset": read_json(folder / "dataset_manifest.json"),
        "training_config": read_json(folder / "training_config.json"),
        "preprocessing_config": read_json(folder / "preprocessing_config.json"),
        "benchmark_plan": read_json(folder / "benchmark_plan.json") if (folder / "benchmark_plan.json").exists() else {},
        "environment": read_json(folder / "environment.json") if (folder / "environment.json").exists() else {},
        "predictions": predictions.head(100).round(6).to_dict(orient="records"),
    }


@app.post("/api/runs/{run_id}/email")
def email_run(run_id: str, request: Request, payload: EmailRunRequest) -> dict[str, Any]:
    folder = run_dir(run_id)
    if not (folder / "run_manifest.json").exists():
        raise HTTPException(status_code=404, detail="Run not found")

    to_email = payload.email.strip()
    if "@" not in to_email or "." not in to_email.split("@")[-1]:
        raise HTTPException(status_code=400, detail="Enter a valid email address")

    subject = f"SeqTrainer BenchLab results {run_id[:8]}"
    body = build_run_email_body(run_id, str(request.base_url))

    try:
        sent = send_smtp_email(to_email, subject, body)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Email service failed: {exc}") from exc

    if sent:
        return {"mode": "smtp", "message": f"Results sent to {to_email}"}

    mailto = f"mailto:{quote(to_email)}?subject={quote(subject)}&body={quote(body)}"
    return {
        "mode": "mailto",
        "message": "SMTP is not configured. Open the generated email draft to send the results.",
        "mailto": mailto,
    }


@app.get("/api/runs/{run_id}/export")
def export_run(run_id: str) -> FileResponse:
    folder = run_dir(run_id)
    if not (folder / "run_manifest.json").exists():
        raise HTTPException(status_code=404, detail="Run not found")
    zip_path = folder / f"{run_id}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for artifact in [
            "metrics.json",
            "predictions.csv",
            "run_manifest.json",
            "dataset_manifest.json",
            "preprocessing_config.json",
            "training_config.json",
            "benchmark_plan.json",
            "environment.json",
        ]:
            artifact_path = folder / artifact
            if artifact_path.exists():
                archive.write(artifact_path, arcname=artifact)
    return FileResponse(zip_path, filename=f"seqtrainer-benchlab-run-{run_id}.zip")


app.mount("/", StaticFiles(directory=APP_ROOT / "static", html=True), name="static")
