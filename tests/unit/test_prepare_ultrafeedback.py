from __future__ import annotations

import pytest

from scripts.data.prepare_ultrafeedback import (
    assign_embedding_clusters,
    assign_random_clusters,
    assign_single_cluster,
    build_outputs,
    split_by_prompt,
)


def pair_row(prompt_id: str) -> dict[str, str]:
    return {
        "prompt_id": prompt_id,
        "instruction": f"instruction {prompt_id}",
        "input": "",
        "chosen": f"chosen {prompt_id}",
        "rejected": f"rejected {prompt_id}",
        "cluster_id": "general",
    }


def test_split_by_prompt_keeps_prompt_ids_disjoint() -> None:
    rows = [pair_row("p0"), pair_row("p0"), pair_row("p1"), pair_row("p2"), pair_row("p3")]

    train_rows, eval_rows, test_rows = split_by_prompt(rows, eval_ratio=0.25, test_ratio=0.25, seed=0)

    train_prompt_ids = {row["prompt_id"] for row in train_rows}
    eval_prompt_ids = {row["prompt_id"] for row in eval_rows}
    test_prompt_ids = {row["prompt_id"] for row in test_rows}
    assert train_prompt_ids
    assert eval_prompt_ids
    assert test_prompt_ids
    assert train_prompt_ids.isdisjoint(eval_prompt_ids)
    assert train_prompt_ids.isdisjoint(test_prompt_ids)
    assert eval_prompt_ids.isdisjoint(test_prompt_ids)


def test_split_by_prompt_rejects_overlapping_ratios() -> None:
    rows = [pair_row("p0"), pair_row("p1")]

    with pytest.raises(ValueError, match="eval_ratio \\+ test_ratio"):
        split_by_prompt(rows, eval_ratio=0.5, test_ratio=0.5, seed=0)


def test_split_by_prompt_can_disable_eval_and_test() -> None:
    rows = [pair_row("p0"), pair_row("p1")]

    train_rows, eval_rows, test_rows = split_by_prompt(rows, eval_ratio=0.0, test_ratio=0.0, seed=0)

    assert train_rows == rows
    assert eval_rows == []
    assert test_rows == []


def test_build_outputs_flattens_kto_and_cpo_after_split() -> None:
    outputs = build_outputs([pair_row("p0")])

    assert len(outputs["sft"]) == 1
    assert len(outputs["dpo"]) == 1
    assert len(outputs["kto"]) == 2
    assert len(outputs["cpo"]) == 2
    assert {row["label"] for row in outputs["kto"]} == {True, False}
    assert {row["label"] for row in outputs["cpo"]} == {True, False}
    assert {row["cluster_id"] for row in outputs["kto"]} == {"general"}
    assert all(row["prompt_id"] == "p0" for rows in outputs.values() for row in rows)


def test_assign_random_clusters_is_prompt_stable_and_balanced() -> None:
    rows = [pair_row(f"p{index // 2}") for index in range(16)]

    clustered = assign_random_clusters(rows, seed=0, n_clusters=4, prefix="r")
    clustered_again = assign_random_clusters(rows, seed=0, n_clusters=4, prefix="r")

    assert clustered == clustered_again
    by_prompt = {}
    for row in clustered:
        by_prompt.setdefault(row["prompt_id"], set()).add(row["cluster_id"])
    assert all(len(cluster_ids) == 1 for cluster_ids in by_prompt.values())
    counts = {}
    for cluster_ids in by_prompt.values():
        cluster_id = next(iter(cluster_ids))
        counts[cluster_id] = counts.get(cluster_id, 0) + 1
    assert counts == {"r_0": 2, "r_1": 2, "r_2": 2, "r_3": 2}


def test_assign_random_clusters_matched_preserves_cluster_distribution() -> None:
    rows = [pair_row(f"p{index}") for index in range(6)]
    for index, row in enumerate(rows):
        row["cluster_id"] = "general" if index < 3 else ("coding" if index < 5 else "math")

    clustered = assign_random_clusters(rows, seed=0, matched=True)

    original_counts = {}
    clustered_counts = {}
    for row in rows:
        original_counts[row["cluster_id"]] = original_counts.get(row["cluster_id"], 0) + 1
    for row in clustered:
        clustered_counts[row["cluster_id"]] = clustered_counts.get(row["cluster_id"], 0) + 1
    assert clustered_counts == original_counts
    assert {row["prompt_id"] for row in clustered} == {row["prompt_id"] for row in rows}


def test_assign_single_cluster_collapses_all_rows() -> None:
    rows = [pair_row("p0"), pair_row("p1")]

    clustered = assign_single_cluster(rows)

    assert {row["cluster_id"] for row in clustered} == {"global"}
    assert {row["prompt_id"] for row in clustered} == {"p0", "p1"}


def test_assign_embedding_clusters_is_prompt_stable() -> None:
    pytest.importorskip("sklearn")

    class FakeEmbedder:
        def encode(self, sentences, **_kwargs):
            return [
                [0.0, 0.0] if sentence.endswith(("p0", "p1")) else [10.0, 10.0]
                for sentence in sentences
            ]

    rows = [pair_row("p0"), pair_row("p0"), pair_row("p1"), pair_row("p2"), pair_row("p3")]

    clustered = assign_embedding_clusters(
        rows,
        seed=0,
        n_clusters=2,
        prefix="e",
        embedder=FakeEmbedder(),
    )

    by_prompt = {}
    for row in clustered:
        by_prompt.setdefault(row["prompt_id"], set()).add(row["cluster_id"])
    assert all(len(cluster_ids) == 1 for cluster_ids in by_prompt.values())
    assert {row["cluster_id"] for row in clustered} == {"e_0", "e_1"}
