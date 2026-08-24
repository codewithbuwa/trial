from __future__ import annotations

import json
from pathlib import Path

from scripts.evaluate.generate_from_prompts import parse_prompt_file


def test_parse_prompt_file_reads_jsonl_manifest(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    path.write_text(
        json.dumps(
            {
                "prompt_id": "p1",
                "instruction": "Explain gradients",
                "input": "",
                "cluster_id": "math",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    prompts = parse_prompt_file(path)

    assert prompts == [
        {
            "prompt_id": "p1",
            "instruction": "Explain gradients",
            "input": "",
            "cluster_id": "math",
        }
    ]
