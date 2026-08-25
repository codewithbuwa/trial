from __future__ import annotations

import argparse
from pathlib import Path

from cpo_trl.data.datasets import write_jsonl
from scripts.audit.data_audit import build_audit


def write_split(root: Path, kind: str, split: str, rows: list[dict[str, object]]) -> None:
    write_jsonl(root / kind / f"{split}.jsonl", rows)


def test_data_audit_reports_splits_clusters_pairs_and_sampler(tmp_path: Path) -> None:
    data_root = tmp_path / "processed"
    dpo_rows = [
        {
            "prompt_id": "p0",
            "instruction": "code",
            "input": "",
            "chosen": "good",
            "rejected": "bad",
            "cluster_id": "coding",
        },
        {
            "prompt_id": "p1",
            "instruction": "math",
            "input": "",
            "chosen": "good",
            "rejected": "bad",
            "cluster_id": "math",
        },
    ]
    cpo_rows = [
        {
            "prompt_id": prompt_id,
            "instruction": instruction,
            "input": "",
            "completion": completion,
            "label": label,
            "cluster_id": cluster_id,
        }
        for prompt_id, instruction, cluster_id in (("p0", "code", "coding"), ("p1", "math", "math"))
        for completion, label in (("good", True), ("bad", False))
    ]
    kto_rows = [{**row, "cluster_id": "general"} for row in cpo_rows]
    sft_rows = [
        {"prompt_id": row["prompt_id"], "instruction": row["instruction"], "input": "", "chosen": "good"}
        for row in dpo_rows
    ]
    for split in ("train", "validation", "test"):
        write_split(data_root, "sft", split, sft_rows if split == "train" else [])
        write_split(data_root, "dpo", split, dpo_rows if split == "train" else [])
        write_split(data_root, "kto", split, kto_rows if split == "train" else [])
        write_split(data_root, "cpo", split, cpo_rows if split == "train" else [])

    audit = build_audit(argparse.Namespace(data_root=data_root, batch_size=4, seed=0))

    assert audit["experiment"] == "E1_data_audit"
    assert audit["splits"]["dpo"]["train"]["pairs"] == 2
    assert audit["splits"]["cpo"]["train"]["labels"] == {"desirable": 2, "undesirable": 2}
    assert audit["splits"]["cpo"]["train"]["pairing"]["pair_eligible_prompt_cluster_groups"] == 2
    assert audit["sampler"]["available"] is True
    assert audit["sampler"]["stats"]["dataset_coverage"] == 1.0
    assert audit["prompt_overlap"]["cpo"]["train_vs_validation"] == []
