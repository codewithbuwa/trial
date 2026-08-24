"""Dataset loading and schema helpers for TRL experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Literal

RowKind = Literal["sft", "dpo", "kto", "cpo"]


class JsonlSchemaError(ValueError):
    """Raised when a JSONL row does not match the expected training schema."""


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSONL file into dictionaries."""

    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise JsonlSchemaError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise JsonlSchemaError(f"{path}:{line_no}: expected a JSON object")
            rows.append(row)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write dictionaries to JSONL, creating the parent directory if needed."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def require_fields(row: dict[str, Any], fields: Iterable[str], *, row_index: int) -> None:
    missing = [field for field in fields if field not in row]
    if missing:
        joined = ", ".join(missing)
        raise JsonlSchemaError(f"row {row_index}: missing required field(s): {joined}")


def validate_rows(rows: Iterable[dict[str, Any]], kind: RowKind) -> list[dict[str, Any]]:
    """Validate and return rows for a supported training schema."""

    required: dict[RowKind, tuple[str, ...]] = {
        "sft": ("instruction", "chosen"),
        "dpo": ("instruction", "chosen", "rejected"),
        "kto": ("instruction", "completion", "label"),
        "cpo": ("prompt_id", "instruction", "completion", "label", "cluster_id"),
    }
    validated = list(rows)
    for index, row in enumerate(validated):
        require_fields(row, required[kind], row_index=index)
        if kind in {"kto", "cpo"} and not isinstance(row["label"], bool):
            raise JsonlSchemaError(f"row {index}: {kind.upper()} label must be boolean")
    return validated


def load_training_rows(path: str | Path, kind: RowKind) -> list[dict[str, Any]]:
    """Load and validate JSONL rows for a training job."""

    return validate_rows(load_jsonl(path), kind)
