from __future__ import annotations

import argparse
import json
from pathlib import Path

from cpo_trl.utils.run_manifest import write_run_manifest


def test_write_run_manifest_serializes_training_args(tmp_path: Path) -> None:
    args = argparse.Namespace(
        output_dir=tmp_path / "run",
        train_file=Path("data/processed/cpo/train.jsonl"),
        model_name_or_path="Qwen/Qwen2.5-1.5B-Instruct",
        seed=42,
    )

    path = write_run_manifest(args.output_dir, method="cpo", args=args, extra={"experiment": "E0"})

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["method"] == "cpo"
    assert payload["experiment"] == "E0"
    assert payload["config"]["train_file"] == "data/processed/cpo/train.jsonl"
    assert payload["config"]["seed"] == 42
