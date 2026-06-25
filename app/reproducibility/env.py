from __future__ import annotations

import os
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit


DOCKER_BASE_IMAGE = "python:3.11-slim"
SAFE_ENV_NAMES = {
    "SEQTRAINER_STORAGE",
    "DELETE_DATASETS_AFTER_RUN",
    "SEQTRAINER_MAX_UPLOAD_MB",
    "SEQTRAINER_MAX_LOCAL_ROWS",
    "SEQTRAINER_MAX_KMER_SIZE",
    "SEQTRAINER_MAX_ONE_HOT_LENGTH",
    "CUDA_VISIBLE_DEVICES",
    "PYTHONHASHSEED",
}
SAFE_ENV_PREFIXES = ("SEQTRAINER_", "BENCHLAB_", "CUDA_", "PYTHONHASHSEED")
SECRET_MARKERS = ("SECRET", "TOKEN", "PASSWORD", "KEY", "SMTP_PASSWORD")


def get_python_version() -> str:
    return sys.version.split()[0]


def get_pip_freeze() -> list[str]:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []
    return sanitize_pip_packages(result.stdout.splitlines())


def _sanitize_requirement_line(line: str) -> str | None:
    if not line or line.startswith("#"):
        return None
    if " @ " not in line:
        package_name = line.split("==", 1)[0].split("[", 1)[0].strip().upper()
        if package_name in SECRET_MARKERS or any(package_name.endswith(f"_{marker}") for marker in SECRET_MARKERS):
            return None
        if "://" in line and "@" in line.split("://", 1)[1].split("/", 1)[0]:
            return None
        return line

    package, location = line.split(" @ ", 1)
    package_name = package.split("[", 1)[0].strip().upper()
    if package_name in SECRET_MARKERS or any(package_name.endswith(f"_{marker}") for marker in SECRET_MARKERS):
        return None
    try:
        parsed = urlsplit(location)
    except ValueError:
        return None
    if not parsed.scheme or not parsed.netloc:
        return line
    hostname = parsed.hostname
    if not hostname:
        return None
    netloc = hostname
    if parsed.port:
        netloc = f"{netloc}:{parsed.port}"
    safe_location = urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))
    return f"{package} @ {safe_location}"


def sanitize_pip_packages(packages: list[str] | tuple[str, ...] | Any) -> list[str]:
    if isinstance(packages, str):
        packages = [packages]
    safe_packages = []
    for raw_package in packages or []:
        line = _sanitize_requirement_line(str(raw_package).strip())
        if line:
            safe_packages.append(line)
    return sorted(set(safe_packages), key=str.lower)


def get_requirement_pins(repo_root: Path) -> list[str]:
    requirements = repo_root / "requirements.txt"
    if requirements.exists():
        pins = []
        for line in requirements.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                pins.append(stripped)
        if pins:
            return pins
    return get_pip_freeze()


def get_installed_requirement_versions(repo_root: Path) -> dict[str, str]:
    versions: dict[str, str] = {}
    for requirement in get_requirement_pins(repo_root):
        package = requirement.split("==")[0].split("[")[0].strip()
        if not package:
            continue
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return dict(sorted(versions.items()))


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


def sanitize_env_vars(values: dict[str, Any]) -> dict[str, str]:
    safe: dict[str, str] = {}
    for name, value in values.items():
        upper = name.upper()
        if any(marker in upper for marker in SECRET_MARKERS):
            continue
        if upper in SAFE_ENV_NAMES or upper.startswith(SAFE_ENV_PREFIXES):
            safe[name] = str(value)
    return dict(sorted(safe.items()))


def get_relevant_env_vars() -> dict[str, str]:
    return sanitize_env_vars(dict(os.environ))


def get_hardware_info() -> dict[str, Any]:
    cuda_available = False
    cuda_version = os.getenv("CUDA_VERSION")
    gpu_name = None
    gpu_names: list[str] = []

    try:
        import torch  # type: ignore

        cuda_available = bool(torch.cuda.is_available())
        cuda_version = cuda_version or getattr(torch.version, "cuda", None)
        if cuda_available:
            gpu_names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
            gpu_name = gpu_names[0] if gpu_names else None
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
            gpu_names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            gpu_name = gpu_names[0] if gpu_names else None
            cuda_available = bool(gpu_name)

    return {
        "device": "cuda" if cuda_available else "cpu",
        "gpu_available": cuda_available,
        "cuda_available": cuda_available,
        "cuda_version": cuda_version,
        "gpu_name": gpu_name,
        "gpu_count": len(gpu_names),
        "gpu_names": gpu_names,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }


def write_requirements_lock(pip_packages: list[str], output_dir: Path, run_id: str | None = None, python_version: str | None = None) -> Path:
    path = output_dir / "requirements.lock.txt"
    header = [
        "# Generated by SeqTrainer BenchLab",
        "# Reproducibility dependency lock for this run.",
        f"# Python: {python_version or get_python_version()}",
    ]
    if run_id:
        header.append(f"# Run ID: {run_id}")
    safe_packages = sanitize_pip_packages(pip_packages)
    path.write_text("\n".join(header + safe_packages) + "\n", encoding="utf-8")
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
    path = output_dir / "Dockerfile.repro"
    path.write_text(
        "\n".join(
            [
                f"FROM {DOCKER_BASE_IMAGE}",
                "WORKDIR /app",
                "ENV DELETE_DATASETS_AFTER_RUN=true",
                "COPY requirements.lock.txt .",
                "RUN pip install --no-cache-dir -r requirements.lock.txt",
                "COPY . .",
                'CMD ["python", "-m", "app.replay", "--config", "run_config.json", "--dry-run"]',
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path
