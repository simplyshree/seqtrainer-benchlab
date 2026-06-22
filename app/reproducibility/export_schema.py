from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import json_schema


def export_schema(output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_schema(), indent=2, sort_keys=True), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export the BenchLab run_config.json schema.")
    parser.add_argument("--output", default="schemas/run_config.schema.json")
    args = parser.parse_args(argv)
    path = export_schema(args.output)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

