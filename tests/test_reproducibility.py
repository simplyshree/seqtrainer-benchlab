from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from app.main import BenchmarkRequest
from app.replay import dry_run_summary, validate_run_config
from app.reproducibility.builders import build_run_config
from app.reproducibility.config import ReproducibleRunConfig
from app.reproducibility.env import get_relevant_env_vars, write_dockerfile_repro, write_environment_yml, write_requirements_lock
from app.reproducibility.run_from_config import replay_from_config, verify_dataset_checksum
from app.seqtrainer_core import file_sha256


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CONFIG = REPO_ROOT / "examples" / "reproducibility" / "run_config.example.json"
EASY_CONFIG = REPO_ROOT / "examples" / "reproducibility" / "easy_run_config.json"


class ReproducibilityTests(unittest.TestCase):
    def test_example_config_validates(self) -> None:
        config = ReproducibleRunConfig.load(EXAMPLE_CONFIG)
        self.assertEqual(config.schema_version, "1.0.0")
        self.assertEqual(config.preprocessing.kmer_size, 6)

    def test_missing_required_fields_fail_validation(self) -> None:
        with self.assertRaises(ValidationError):
            ReproducibleRunConfig.model_validate({"schema_version": "1.0.0"})

    def test_dataset_checksum_verification(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dataset.csv"
            path.write_text("sequence,label\nAAAA,0\nCCCC,1\n", encoding="utf-8")
            config = ReproducibleRunConfig.load(EXAMPLE_CONFIG)
            config.dataset.path = str(path)
            config.dataset.sha256 = file_sha256(path)
            self.assertTrue(verify_dataset_checksum(config))
            config.dataset.sha256 = "0" * 64
            self.assertFalse(verify_dataset_checksum(config))

    def test_requirements_lock_contains_pins(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_requirements_lock(["numpy==2.2.1", "pandas==2.2.3"], Path(tmp))
            text = path.read_text(encoding="utf-8")
            self.assertIn("numpy==2.2.1", text)
            self.assertIn("pandas==2.2.3", text)

    def test_dockerfile_repro_contains_python_and_requirements(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_dockerfile_repro("3.11.9", Path(tmp))
            text = path.read_text(encoding="utf-8")
            self.assertIn("FROM python:3.11-slim", text)
            self.assertIn("pip install --no-cache-dir -r requirements.lock.txt", text)
            self.assertNotIn("SMTP_PASSWORD", text)

    def test_environment_yml_is_generated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_environment_yml("3.11.9", ["numpy==2.2.1"], Path(tmp))
            text = path.read_text(encoding="utf-8")
            self.assertIn("name: seqtrainer-benchlab-repro", text)
            self.assertIn("numpy==2.2.1", text)

    def test_safe_env_vars_exclude_secrets(self) -> None:
        with patch.dict("os.environ", {"SEQTRAINER_MAX_LOCAL_ROWS": "1000", "SMTP_PASSWORD": "secret", "API_TOKEN": "token"}, clear=False):
            env = get_relevant_env_vars()
            self.assertEqual(env.get("SEQTRAINER_MAX_LOCAL_ROWS"), "1000")
            self.assertNotIn("SMTP_PASSWORD", env)
            self.assertNotIn("API_TOKEN", env)

    def test_replay_command_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary = replay_from_config(EXAMPLE_CONFIG, output_dir=tmp, dry_run=True)
            self.assertTrue(summary["dry_run"])
            self.assertTrue((Path(tmp) / "run_config.json").exists())
            self.assertTrue((Path(tmp) / "requirements.lock.txt").exists())

    def test_new_run_config_validation_and_dry_summary(self) -> None:
        config = validate_run_config(ReproducibleRunConfig.load(EASY_CONFIG).to_dict())
        summary = dry_run_summary(config)
        self.assertTrue(summary["valid"])
        self.assertIn("linear_regression", summary["models"])

    def test_replay_command_runs_easy_models(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary = replay_from_config(EASY_CONFIG, output_dir=tmp, dry_run=False, models=["linear_regression"])
            self.assertFalse(summary["dry_run"])
            self.assertEqual(summary["runnable_models"], ["linear_regression"])
            self.assertTrue((Path(tmp) / "metrics.json").exists())
            self.assertTrue((Path(tmp) / "predictions.csv").exists())

    def test_replay_resolves_dataset_when_started_from_notebooks_folder(self) -> None:
        config = ReproducibleRunConfig.load(EASY_CONFIG)
        with patch("pathlib.Path.cwd", return_value=REPO_ROOT / "notebooks"):
            self.assertTrue(verify_dataset_checksum(config, EASY_CONFIG))
            with tempfile.TemporaryDirectory() as tmp:
                summary = replay_from_config(EASY_CONFIG, output_dir=tmp, dry_run=False, models=["linear_regression"])
                self.assertEqual(summary["runnable_models"], ["linear_regression"])

    def test_benchmark_request_converts_to_run_config(self) -> None:
        request = BenchmarkRequest(dataset_id="00000000-0000-0000-0000-000000000000", target_col="label")
        config = build_run_config(
            request=request,
            dataset_manifest={
                "dataset_id": request.dataset_id,
                "original_name": "small.csv",
                "sha256": "1" * 64,
                "upload_bytes": 12,
                "row_count": 10,
                "source_format": ".csv",
            },
            preprocessing_config={"use_gc": True, "use_kmers": True, "normalize_kmers": True, "kmer_size": 6, "sequence_length": 150},
            training_config={
                "models": ["linear_regression"],
                "comparison_models_for_colab_hpc": ["cnn", "dnabert2", "ipro_mp"],
                "test_size": 0.2,
                "validation_size": 0.1,
                "random_seed": 42,
                "split_strategy": "fixed_train_validation_test",
                "sequence_col": "sequence",
                "target_col": "label",
                "local_row_limit": 1000,
                "row_cap_applied": False,
                "class_balance_strategy": "none",
                "threshold_strategy": "user_or_literature",
                "threshold_scope": "shared",
                "biological_goal": "limit_false_positives",
                "training_cycles": 20,
                "early_stopping_patience": 5,
                "train_rows": 8,
                "test_rows": 2,
                "cv_folds": 5,
                "reruns": 3,
            },
            environment={"python_version": "3.11.0", "pip_packages": ["numpy==2.2.1"], "hardware": {"device": "cpu"}},
            run_manifest={"run_id": "run", "created_at": "2026-06-22T00:00:00+00:00", "tool_version": "0.1.0"},
            repo_root=REPO_ROOT,
        )
        self.assertEqual(config.dataset.target_column, "label")
        self.assertEqual(config.models[0].model_name, "linear_regression")
        self.assertEqual(config.split.random_seed, 42)


if __name__ == "__main__":
    unittest.main()
