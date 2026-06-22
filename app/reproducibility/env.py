from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


SAFE_ENV_PREFIXES = ("SEQTRAINER_", "BENCHLAB_", "CUDA_", "PYTHONHASHSEED")
SECRET_MARKERS = ("SECRET", "TOKEN", "PASSWORD", "KEY", "SMTP_PASSWORD")


def get_python_version() -> str:
    return sys.version.split()[0]


def get_pip_freeze() -> list[str]:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return []
    return sorted(line.strip() for line in result.stdout.splitlines() if line.strip())


def _git(args: list[str], repo_root: Path) -> str | None:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value or None


def get_git_commit(repo_root: Path) -> str | None:
    return _git(["rev-parse", "HEAD"], repo_root)


def get_git_branch(repo_root: Path) -> str | None:
    return _git(["branch", "--show-current"], repo_root)


def get_repo_url(repo_root: Path) -> str | None:
    return _git(["config", "--get", "remote.origin.url"], repo_root)


def get_relevant_env_vars() -> dict[str, str]:
    safe: dict[str, str] = {}
    for name, value in os.environ.items():
        upper = name.upper()
        if any(marker in upper for marker in SECRET_MARKERS):
            continue
        if upper.startswith(SAFE_ENV_PREFIXES):
            safe[name] = value
    return dict(sorted(safe.items()))


def get_hardware_info() -> dict[str, Any]:
    cuda_available = False
    cuda_version = os.getenv("CUDA_VERSION")
    gpu_name = None

    try:
        import torch  # type: ignore

        cuda_available = bool(torch.cuda.is_available())
        cuda_version = cuda_version or getattr(torch.version, "cuda", None)
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        pass

    if gpu_name is None:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                check=False,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            result = None
        if result and result.returncode == 0:
            gpu_name = result.stdout.splitlines()[0].strip() if result.stdout.splitlines() else None
            cuda_available = bool(gpu_name)

    return {
        "device": "cuda" if cuda_available else "cpu",
        "gpu_available": cuda_available,
        "cuda_available": cuda_available,
        "cuda_version": cuda_version,
        "gpu_name": gpu_name,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def write_requirements_lock(pip_packages: list[str], output_dir: Path) -> Path:
    path = output_dir / "requirements.lock.txt"
    path.write_text("\n".join(pip_packages) + ("\n" if pip_packages else ""), encoding="utf-8")
    return path


def write_environment_yml(python_version: str, pip_packages: list[str], output_dir: Path) -> Path:
    path = output_dir / "environment.yml"
    lines = [
        "name: seqtrainer-benchlab-repro",
        "channels:",
        "  - conda-forge",
        "dependencies:",
        f"  - python={python_version}",
        "  - pip",
        "  - pip:",
    ]
    lines.extend(f"      - {package}" for package in pip_packages)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_dockerfile_repro(python_version: str, output_dir: Path) -> Path:
    major_minor = ".".join(python_version.split(".")[:2])
    path = output_dir / "Dockerfile.repro"
    path.write_text(
        "\n".join(
            [
                f"FROM python:{major_minor}-slim",
                "WORKDIR /app",
                "COPY requirements.lock.txt .",
                "RUN pip install --no-cache-dir -r requirements.lock.txt",
                "COPY . .",
                'CMD ["python", "-m", "app.reproducibility.run_from_config", "--config", "run_config.json", "--dry-run"]',
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path
