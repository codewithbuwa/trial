from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


def git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode:
        return None
    return completed.stdout.strip() or None


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    return str(value)


def write_run_manifest(
    output_dir: str | Path,
    *,
    method: str,
    args: argparse.Namespace,
    extra: dict[str, Any] | None = None,
) -> Path:
    path = Path(output_dir) / "run_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": method,
        "git_commit": git_commit(),
        "config": jsonable(vars(args)),
    }
    if extra:
        payload.update(jsonable(extra))
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


__all__ = ["write_run_manifest"]
