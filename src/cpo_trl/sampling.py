"""Sampling utilities for CPO training."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterator

from torch.utils.data import Sampler


@dataclass(frozen=True)
class PairAwareBatchStats:
    eligible_prompt_groups: int
    skipped_prompt_groups: int
    batches: int
    pairs: int
    cluster_counts: dict[str, int]
    cluster_pairs: dict[str, int]
    paired_rows_seen: int
    unpaired_rows_seen: int
    unpaired_rows_used: int
    unpaired_rows_dropped: int
    dataset_rows_used: int
    dataset_coverage: float
    cluster_paired_rows: dict[str, int]
    cluster_unary_rows: dict[str, int]
    baseline_ready_batches: int
    baseline_ready_rate: float


class CPOPairAwareBatchSampler(Sampler[list[int]]):
    """Build batches from same-cluster prompt groups containing both labels."""

    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        batch_size: int,
        seed: int = 42,
        cluster_sampling: str = "proportional",
    ) -> None:
        if batch_size < 2:
            raise ValueError("pair-aware CPO batching requires batch_size >= 2")
        if cluster_sampling != "proportional":
            raise ValueError(f"unsupported cluster_sampling: {cluster_sampling}")
        self.rows = rows
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        self.cluster_sampling = cluster_sampling
        self.pairs_per_batch = max(1, batch_size // 2)
        self._batches, self.stats = self._build_batches()

    def __iter__(self) -> Iterator[list[int]]:
        return iter(self._batches)

    def __len__(self) -> int:
        return len(self._batches)

    def _prompt_id(self, row_index: int) -> str:
        return str(self.rows[row_index].get("prompt_id"))

    def _cluster_id(self, row_index: int) -> str:
        return str(self.rows[row_index].get("cluster_id", "unknown"))

    def set_epoch(self, epoch: int) -> None:
        """Reshuffle deterministic batches for a new epoch."""

        if epoch == self.epoch:
            return
        self.epoch = epoch
        self._batches, self.stats = self._build_batches()

    def _build_batches(self) -> tuple[list[list[int]], PairAwareBatchStats]:
        grouped: dict[tuple[str, str], dict[bool, list[int]]] = defaultdict(lambda: {True: [], False: []})
        for index, row in enumerate(self.rows):
            key = (str(row.get("cluster_id", "unknown")), str(row.get("prompt_id")))
            grouped[key][bool(row["label"])].append(index)

        rng = random.Random(self.seed + self.epoch)
        cluster_pairs: dict[str, list[tuple[int, int]]] = defaultdict(list)
        cluster_unary: dict[str, list[int]] = defaultdict(list)
        paired_row_indices: set[int] = set()
        unpaired_row_indices: set[int] = set()
        skipped_prompt_groups = 0
        for (cluster_id, _prompt_id), label_groups in grouped.items():
            positives = label_groups[True]
            negatives = label_groups[False]
            if not positives or not negatives:
                skipped_prompt_groups += 1
                cluster_unary[cluster_id].extend(positives)
                cluster_unary[cluster_id].extend(negatives)
                unpaired_row_indices.update(positives)
                unpaired_row_indices.update(negatives)
                continue
            rng.shuffle(positives)
            rng.shuffle(negatives)
            paired_row_indices.update(positives)
            paired_row_indices.update(negatives)
            pair_count = max(len(positives), len(negatives))
            for offset in range(pair_count):
                cluster_pairs[cluster_id].append(
                    (
                        positives[offset % len(positives)],
                        negatives[offset % len(negatives)],
                    )
                )

        for pairs in cluster_pairs.values():
            rng.shuffle(pairs)
        for rows in cluster_unary.values():
            rng.shuffle(rows)

        batches: list[list[int]] = []
        cluster_counts: dict[str, int] = {}
        cluster_pair_counts: dict[str, int] = {}
        used_indices: set[int] = set()
        used_unpaired_indices: set[int] = set()

        def pop_unary_from_cluster(cluster_id: str, excluded_prompt_ids: set[str]) -> int | None:
            rows = cluster_unary.get(cluster_id)
            if not rows:
                return None
            for offset in range(len(rows) - 1, -1, -1):
                if self._prompt_id(rows[offset]) not in excluded_prompt_ids:
                    return rows.pop(offset)
            return rows.pop()

        def pop_global_unary(excluded_prompt_ids: set[str]) -> int | None:
            candidate_clusters = [
                cluster_id
                for cluster_id, rows in cluster_unary.items()
                if any(self._prompt_id(row_index) not in excluded_prompt_ids for row_index in rows)
            ]
            if not candidate_clusters:
                candidate_clusters = [cluster_id for cluster_id, rows in cluster_unary.items() if rows]
            if not candidate_clusters:
                return None
            weights = [len(cluster_unary[cluster_id]) for cluster_id in candidate_clusters]
            cluster_id = rng.choices(candidate_clusters, weights=weights, k=1)[0]
            return pop_unary_from_cluster(cluster_id, excluded_prompt_ids)

        active_pair_clusters = [cluster_id for cluster_id, pairs in cluster_pairs.items() if pairs]
        while active_pair_clusters:
            weights = [len(cluster_pairs[cluster_id]) for cluster_id in active_pair_clusters]
            cluster_id = rng.choices(active_pair_clusters, weights=weights, k=1)[0]
            batch_pairs: list[tuple[int, int]] = []
            pair_limit = 1 if cluster_unary.get(cluster_id) else self.pairs_per_batch
            while cluster_pairs[cluster_id] and len(batch_pairs) < pair_limit:
                batch_pairs.append(cluster_pairs[cluster_id].pop())
            if not cluster_pairs[cluster_id]:
                active_pair_clusters = [cluster for cluster in active_pair_clusters if cluster != cluster_id]
            if not batch_pairs:
                continue
            batch = [index for pair in batch_pairs for index in pair]
            batch_prompt_ids = {self._prompt_id(index) for index in batch}
            while len(batch) < self.batch_size and cluster_unary.get(cluster_id):
                unary_index = pop_unary_from_cluster(cluster_id, batch_prompt_ids)
                if unary_index is None:
                    break
                batch.append(unary_index)
                batch_prompt_ids.add(self._prompt_id(unary_index))
                used_unpaired_indices.add(unary_index)
            while len(batch) < self.batch_size:
                unary_index = pop_global_unary(batch_prompt_ids)
                if unary_index is None:
                    break
                batch.append(unary_index)
                batch_prompt_ids.add(self._prompt_id(unary_index))
                used_unpaired_indices.add(unary_index)
            batches.append(batch)
            used_indices.update(batch)
            cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1
            cluster_pair_counts[cluster_id] = cluster_pair_counts.get(cluster_id, 0) + len(batch_pairs)

        active_unary_clusters = [cluster_id for cluster_id, rows in cluster_unary.items() if rows]
        while active_unary_clusters:
            weights = [len(cluster_unary[cluster_id]) for cluster_id in active_unary_clusters]
            cluster_id = rng.choices(active_unary_clusters, weights=weights, k=1)[0]
            batch: list[int] = []
            batch_prompt_ids: set[str] = set()
            while len(batch) < self.batch_size:
                unary_index = pop_global_unary(batch_prompt_ids)
                if unary_index is None:
                    break
                batch.append(unary_index)
                batch_prompt_ids.add(self._prompt_id(unary_index))
                used_unpaired_indices.add(unary_index)
            if batch:
                batches.append(batch)
                used_indices.update(batch)
                cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1
            active_unary_clusters = [cluster for cluster, rows in cluster_unary.items() if rows]

        cluster_paired_rows: dict[str, int] = {}
        cluster_unary_rows: dict[str, int] = {}
        for row_index in paired_row_indices:
            cluster_id = str(self.rows[row_index].get("cluster_id", "unknown"))
            cluster_paired_rows[cluster_id] = cluster_paired_rows.get(cluster_id, 0) + 1
        for row_index in unpaired_row_indices:
            cluster_id = str(self.rows[row_index].get("cluster_id", "unknown"))
            cluster_unary_rows[cluster_id] = cluster_unary_rows.get(cluster_id, 0) + 1
        unpaired_rows_used = len(used_unpaired_indices)
        unpaired_rows_dropped = len(unpaired_row_indices - used_unpaired_indices)
        baseline_ready_batches = 0
        for batch in batches:
            prompts_by_cluster: dict[str, set[str]] = {}
            for row_index in batch:
                prompts_by_cluster.setdefault(self._cluster_id(row_index), set()).add(self._prompt_id(row_index))
            if any(len(prompt_ids) >= 2 for prompt_ids in prompts_by_cluster.values()):
                baseline_ready_batches += 1

        stats = PairAwareBatchStats(
            eligible_prompt_groups=sum(1 for groups in grouped.values() if groups[True] and groups[False]),
            skipped_prompt_groups=skipped_prompt_groups,
            batches=len(batches),
            pairs=sum(cluster_pair_counts.values()),
            cluster_counts=cluster_counts,
            cluster_pairs=cluster_pair_counts,
            paired_rows_seen=len(paired_row_indices),
            unpaired_rows_seen=len(unpaired_row_indices),
            unpaired_rows_used=unpaired_rows_used,
            unpaired_rows_dropped=unpaired_rows_dropped,
            dataset_rows_used=len(used_indices),
            dataset_coverage=len(used_indices) / len(self.rows) if self.rows else 0.0,
            cluster_paired_rows=cluster_paired_rows,
            cluster_unary_rows=cluster_unary_rows,
            baseline_ready_batches=baseline_ready_batches,
            baseline_ready_rate=baseline_ready_batches / len(batches) if batches else 0.0,
        )
        return batches, stats
