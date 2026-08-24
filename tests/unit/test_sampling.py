from __future__ import annotations

import pytest

from cpo_trl.sampling import CPOPairAwareBatchSampler


def row(prompt_id: str, cluster_id: str, label: bool) -> dict[str, object]:
    return {"prompt_id": prompt_id, "cluster_id": cluster_id, "label": label}


def test_pair_aware_sampler_builds_same_prompt_same_cluster_pairs() -> None:
    rows = [
        row("p0", "coding", True),
        row("p0", "coding", False),
        row("p1", "coding", True),
        row("p1", "coding", False),
        row("p2", "math", True),
    ]

    sampler = CPOPairAwareBatchSampler(rows, batch_size=4, seed=0)
    batches = list(sampler)

    assert sampler.stats.eligible_prompt_groups == 2
    assert sampler.stats.skipped_prompt_groups == 1
    assert sampler.stats.pairs == 2
    assert sampler.stats.cluster_pairs == {"coding": 2}
    assert sampler.stats.unpaired_rows_seen == 1
    assert sampler.stats.unpaired_rows_used == 1
    assert sampler.stats.unpaired_rows_dropped == 0
    assert sampler.stats.dataset_rows_used == 5
    assert sampler.stats.dataset_coverage == 1.0
    assert sampler.stats.cluster_unary_rows == {"math": 1}
    assert sampler.stats.baseline_ready_batches >= 1
    assert len(batches) == 2
    assert sorted(index for batch in batches for index in batch) == [0, 1, 2, 3, 4]
    for batch in batches:
        if len(batch) < 2:
            continue
        first, second = batch[:2]
        assert rows[first]["prompt_id"] == rows[second]["prompt_id"]
        assert rows[first]["cluster_id"] == rows[second]["cluster_id"]
        assert {rows[first]["label"], rows[second]["label"]} == {True, False}


def test_pair_aware_sampler_uses_proportional_cluster_counts() -> None:
    rows = []
    for index in range(6):
        rows.extend([row(f"g{index}", "general", True), row(f"g{index}", "general", False)])
    for index in range(2):
        rows.extend([row(f"m{index}", "math", True), row(f"m{index}", "math", False)])

    sampler = CPOPairAwareBatchSampler(rows, batch_size=2, seed=0)

    assert sampler.stats.cluster_counts["general"] == 6
    assert sampler.stats.cluster_counts["math"] == 2
    assert sampler.stats.cluster_pairs["general"] == 6
    assert sampler.stats.cluster_pairs["math"] == 2
    assert sampler.stats.dataset_coverage == 1.0
    assert sampler.stats.baseline_ready_rate == 0.0


def test_pair_aware_sampler_preserves_unary_only_rows_as_fillers() -> None:
    rows = [
        row("p0", "coding", True),
        row("p0", "coding", False),
        row("p1", "coding", True),
        row("p2", "math", False),
    ]

    sampler = CPOPairAwareBatchSampler(rows, batch_size=4, seed=0)
    batches = list(sampler)

    assert sampler.stats.pairs == 1
    assert sampler.stats.unpaired_rows_seen == 2
    assert sampler.stats.unpaired_rows_used == 2
    assert sampler.stats.unpaired_rows_dropped == 0
    assert sampler.stats.dataset_coverage == 1.0
    assert sorted(index for batch in batches for index in batch) == [0, 1, 2, 3]


def test_pair_aware_sampler_prefers_same_cluster_different_prompt_fillers() -> None:
    rows = [
        row("p0", "coding", True),
        row("p0", "coding", False),
        row("p0", "coding", True),
        row("p1", "coding", True),
        row("p2", "math", False),
    ]

    sampler = CPOPairAwareBatchSampler(rows, batch_size=4, seed=0)
    first_batch = list(sampler)[0]
    first_batch_clusters = [rows[index]["cluster_id"] for index in first_batch]
    first_batch_prompts = [rows[index]["prompt_id"] for index in first_batch]

    assert first_batch_clusters[:3] == ["coding", "coding", "coding"]
    assert "p1" in first_batch_prompts
    assert sampler.stats.baseline_ready_batches >= 1
    assert sampler.stats.baseline_ready_rate > 0.0


def test_pair_aware_sampler_reshuffles_between_epochs() -> None:
    rows = []
    for index in range(12):
        rows.extend([row(f"p{index}", "general", True), row(f"p{index}", "general", False)])

    sampler = CPOPairAwareBatchSampler(rows, batch_size=4, seed=0)
    epoch_zero = list(sampler)
    sampler.set_epoch(1)
    epoch_one = list(sampler)

    assert epoch_zero != epoch_one
    assert sorted(index for batch in epoch_zero for index in batch) == sorted(
        index for batch in epoch_one for index in batch
    )
    assert sampler.stats.dataset_coverage == 1.0


def test_pair_aware_sampler_requires_pair_capacity() -> None:
    with pytest.raises(ValueError, match="batch_size >= 2"):
        CPOPairAwareBatchSampler([row("p0", "coding", True)], batch_size=1)
