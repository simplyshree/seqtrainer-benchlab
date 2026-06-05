from __future__ import annotations

import json
import os
import shutil
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from .seqtrainer_core import build_features, file_sha256, read_dataset


APP_ROOT = Path(__file__).resolve().parent
REPO_ROOT = APP_ROOT.parent
STORAGE_ROOT = Path(os.getenv("SEQTRAINER_STORAGE", REPO_ROOT / "storage"))
DATASETS_ROOT = STORAGE_ROOT / "datasets"
RUNS_ROOT = STORAGE_ROOT / "runs"

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
    preprocessing: dict[str, Any] = Field(default_factory=lambda: {"use_gc": True, "use_kmers": True, "kmer_size": 4})
    test_size: float = 0.2
    random_seed: int = 42


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_storage() -> None:
    DATASETS_ROOT.mkdir(parents=True, exist_ok=True)
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)


def dataset_dir(dataset_id: str) -> Path:
    return DATASETS_ROOT / dataset_id


def run_dir(run_id: str) -> Path:
    return RUNS_ROOT / run_id


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


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


app = FastAPI(title="SeqTrainer BenchLab", version="0.1.0")


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
    dataset_id = str(uuid.uuid4())
    folder = dataset_dir(dataset_id)
    folder.mkdir(parents=True, exist_ok=True)

    original_name = Path(file.filename or "dataset").name
    raw_path = folder / original_name
    with raw_path.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)

    try:
        df = read_dataset(raw_path)
    except Exception as exc:
        shutil.rmtree(folder, ignore_errors=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if df.empty:
        shutil.rmtree(folder, ignore_errors=True)
        raise HTTPException(status_code=400, detail="Dataset contains no rows")

    df.to_csv(folder / "dataset.csv", index=False)
    manifest = {
        "dataset_id": dataset_id,
        "original_name": original_name,
        "created_at": utc_now(),
        "row_count": int(len(df)),
        "columns": list(df.columns),
        "suggested_sequence_col": infer_column(list(df.columns), ["sequence", "seq", "variant", "dna"]),
        "suggested_target_col": infer_column(list(df.columns), TARGET_COLUMN_CANDIDATES),
        "sha256": file_sha256(raw_path),
        "source_format": raw_path.suffix.lower(),
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
    parsed_config = json.loads(config)
    features = build_features(df, sequence_col, parsed_config)
    return {
        "feature_count": int(features.shape[1]),
        "row_count": int(features.shape[0]),
        "columns": list(features.columns[:40]),
        "preview": features.head(5).round(6).to_dict(orient="records"),
    }


@app.post("/api/benchmarks")
def run_benchmark(request: BenchmarkRequest) -> dict[str, Any]:
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

    run_id = str(uuid.uuid4())
    folder = run_dir(run_id)
    folder.mkdir(parents=True, exist_ok=True)

    features = build_features(df, request.sequence_col, request.preprocessing)
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

    metrics: dict[str, Any] = {}
    prediction_frames = []

    for model_name in selected_models:
        model_cls = MODEL_MAP[model_name]
        model = model_cls(random_state=request.random_seed) if "random_state" in model_cls().get_params() else model_cls()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        metrics[model_name] = {
            "mae": float(mean_absolute_error(y_test, predictions)),
            "mse": float(mse),
            "rmse": float(mse**0.5),
            "r2": float(r2_score(y_test, predictions)),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
        }
        joblib.dump(model, folder / f"{model_name}.joblib")
        prediction_frames.append(
            pd.DataFrame(
                {
                    "model": model_name,
                    "actual": y_test.to_numpy(),
                    "prediction": predictions,
                }
            )
        )

    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    predictions_df.to_csv(folder / "predictions.csv", index=False)

    training_config = {
        "models": selected_models,
        "test_size": request.test_size,
        "random_seed": request.random_seed,
        "sequence_col": request.sequence_col,
        "target_col": target_col,
    }
    run_manifest = {
        "run_id": run_id,
        "created_at": utc_now(),
        "dataset_id": request.dataset_id,
        "dataset_sha256": dataset_manifest["sha256"],
        "seqtrainer_benchlab_version": "0.1.0",
        "artifact_paths": {
            "metrics": "metrics.json",
            "predictions": "predictions.csv",
            "dataset_manifest": "dataset_manifest.json",
            "preprocessing_config": "preprocessing_config.json",
            "training_config": "training_config.json",
        },
    }

    write_json(folder / "metrics.json", metrics)
    write_json(folder / "dataset_manifest.json", dataset_manifest)
    write_json(folder / "preprocessing_config.json", request.preprocessing)
    write_json(folder / "training_config.json", training_config)
    write_json(folder / "run_manifest.json", run_manifest)

    return {"run": run_manifest, "metrics": metrics, "predictions": predictions_df.head(50).round(6).to_dict(orient="records")}


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
        "predictions": predictions.head(100).round(6).to_dict(orient="records"),
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
        ]:
            archive.write(folder / artifact, arcname=artifact)
    return FileResponse(zip_path, filename=f"seqtrainer-benchlab-run-{run_id}.zip")


app.mount("/", StaticFiles(directory=APP_ROOT / "static", html=True), name="static")
