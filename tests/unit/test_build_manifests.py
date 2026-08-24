from __future__ import annotations

from scripts.data.build_manifests import balanced_prompt_rows, unique_prompt_rows


def test_unique_prompt_rows_keeps_first_prompt_metadata() -> None:
    rows = [
        {"prompt_id": "p1", "instruction": "A", "input": "", "cluster_id": "coding"},
        {"prompt_id": "p1", "instruction": "A duplicate", "input": "", "cluster_id": "math"},
        {"prompt_id": "p2", "instruction": "B", "input": "x", "cluster_id": "writing"},
    ]

    prompts = unique_prompt_rows(rows)

    assert prompts == [
        {"prompt_id": "p1", "instruction": "A", "input": "", "cluster_id": "coding"},
        {"prompt_id": "p2", "instruction": "B", "input": "x", "cluster_id": "writing"},
    ]


def test_balanced_prompt_rows_uses_equal_cluster_counts() -> None:
    prompts = [
        {"prompt_id": "g1", "instruction": "G1", "input": "", "cluster_id": "general"},
        {"prompt_id": "g2", "instruction": "G2", "input": "", "cluster_id": "general"},
        {"prompt_id": "c1", "instruction": "C1", "input": "", "cluster_id": "coding"},
    ]

    balanced = balanced_prompt_rows(prompts, per_cluster=None, seed=0)

    counts: dict[str, int] = {}
    for row in balanced:
        counts[row["cluster_id"]] = counts.get(row["cluster_id"], 0) + 1
    assert counts == {"coding": 1, "general": 1}
